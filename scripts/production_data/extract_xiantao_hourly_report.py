#!/usr/bin/env python3
"""Extract a small, publication-safe plant historian snapshot from yulin.

The yulin PostgreSQL database stores hourly operating reports as JSON. This
script extracts selected process variables from ``<schema>.report_hourly`` and
writes:

* a private CSV under ``results/production_validation/raw``;
* an aggregate summary JSON that can be committed; and
* a PDF figure for the manuscript.

Connection parameters are read from environment variables. Do not put database
passwords in this file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import statistics
import struct
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


TARGETS: dict[str, dict[str, str]] = {
    "unit1": {
        "DCS1_LOAD_3S": "load_mw",
        "DCS1_MAIN_PRESS": "main_pressure_mpa",
        "DCS1_MAIN_TEMP": "main_temperature_c",
        "DCS1_REHEAT_PRESS": "reheat_pressure_mpa",
        "DCS1_REHEAT_TEMP": "reheat_temperature_c",
        "DCS1_FUELFLOWRP": "fuel_flow_tph",
        "DCS1_TOTALFLOW": "air_flow_tph",
    },
    "unit2": {
        "DCS2_LOAD_3S": "load_mw",
        "DCS2_MAIN_PRESS": "main_pressure_mpa",
        "DCS2_MAIN_TEMP": "main_temperature_c",
        "DCS2_REHEAT_PRESS": "reheat_pressure_mpa",
        "DCS2_REHEAT_TEMP": "reheat_temperature_c",
        "DCS2_FUELFLOWRP": "fuel_flow_tph",
        "DCS2_TOTALFLOW": "air_flow_tph",
    },
}


LABELS = {
    "load_mw": "Generator active power",
    "main_pressure_mpa": "Main steam pressure",
    "main_temperature_c": "Main steam temperature",
    "reheat_pressure_mpa": "Reheat steam pressure",
    "reheat_temperature_c": "Reheat steam temperature",
    "fuel_flow_tph": "Fuel flow",
    "air_flow_tph": "Total air flow",
}


UNITS = {
    "load_mw": "MW",
    "main_pressure_mpa": "MPa",
    "main_temperature_c": "degC",
    "reheat_pressure_mpa": "MPa",
    "reheat_temperature_c": "degC",
    "fuel_flow_tph": "t/h",
    "air_flow_tph": "t/h",
}


class PgWireClient:
    """Minimal PostgreSQL simple-query client.

    This avoids adding a driver dependency to the paper repository. It supports
    cleartext and md5 password authentication and returns all values as text.
    """

    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.sock: socket.socket | None = None

    def __enter__(self) -> "PgWireClient":
        self.sock = socket.create_connection((self.host, self.port), timeout=20)
        self.sock.settimeout(240)
        params = OrderedDict(
            [
                ("user", self.user),
                ("database", self.database),
                ("client_encoding", "UTF8"),
                ("application_name", "rocbf_xiantao_hourly_extract"),
            ]
        )
        body = struct.pack("!I", 196608)
        for key, value in params.items():
            body += key.encode() + b"\0" + value.encode() + b"\0"
        body += b"\0"
        self.sock.sendall(struct.pack("!I", len(body) + 4) + body)

        while True:
            typ = self._recvall(1)
            payload = self._read_payload()
            if typ == b"R":
                auth = struct.unpack("!I", payload[:4])[0]
                if auth == 0:
                    continue
                if auth == 3:
                    self._send_msg(b"p", self.password.encode() + b"\0")
                    continue
                if auth == 5:
                    salt = payload[4:8]
                    inner = hashlib.md5((self.password + self.user).encode()).hexdigest().encode()
                    outer = "md5" + hashlib.md5(inner + salt).hexdigest()
                    self._send_msg(b"p", outer.encode() + b"\0")
                    continue
                raise RuntimeError(f"Unsupported PostgreSQL auth method: {auth}")
            if typ == b"E":
                raise RuntimeError(self._pg_error(payload))
            if typ == b"Z":
                return self

    def __exit__(self, *_exc: object) -> None:
        if self.sock is not None:
            self.sock.close()

    def query(self, sql: str) -> list[dict[str, str | None]]:
        self._send_msg(b"Q", sql.encode("utf-8") + b"\0")
        cols: list[str] = []
        rows: list[dict[str, str | None]] = []
        while True:
            typ = self._recvall(1)
            payload = self._read_payload()
            if typ == b"T":
                off = 0
                ncols = struct.unpack("!H", payload[off : off + 2])[0]
                off += 2
                cols = []
                for _ in range(ncols):
                    name, off = self._cstring(payload, off)
                    off += 18
                    cols.append(name)
            elif typ == b"D":
                off = 0
                nvals = struct.unpack("!H", payload[off : off + 2])[0]
                off += 2
                vals: list[str | None] = []
                for _ in range(nvals):
                    length = struct.unpack("!i", payload[off : off + 4])[0]
                    off += 4
                    if length == -1:
                        vals.append(None)
                    else:
                        vals.append(payload[off : off + length].decode("utf-8", "replace"))
                        off += length
                rows.append(dict(zip(cols, vals)))
            elif typ == b"E":
                raise RuntimeError(self._pg_error(payload))
            elif typ == b"Z":
                return rows

    def _recvall(self, nbytes: int) -> bytes:
        assert self.sock is not None
        chunks = bytearray()
        while len(chunks) < nbytes:
            chunk = self.sock.recv(nbytes - len(chunks))
            if not chunk:
                raise EOFError("PostgreSQL connection closed")
            chunks.extend(chunk)
        return bytes(chunks)

    def _read_payload(self) -> bytes:
        length = struct.unpack("!I", self._recvall(4))[0]
        return self._recvall(length - 4)

    def _send_msg(self, typ: bytes, payload: bytes) -> None:
        assert self.sock is not None
        self.sock.sendall(typ + struct.pack("!I", len(payload) + 4) + payload)

    @staticmethod
    def _cstring(payload: bytes, off: int) -> tuple[str, int]:
        end = payload.index(b"\0", off)
        return payload[off:end].decode("utf-8", "replace"), end + 1

    def _pg_error(self, payload: bytes) -> dict[str, str]:
        fields: dict[str, str] = {}
        off = 0
        while off < len(payload) and payload[off] != 0:
            code = chr(payload[off])
            off += 1
            text, off = self._cstring(payload, off)
            fields[code] = text
        return fields


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def iter_report_values(obj: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def walk(node: Any, context: dict[str, Any] | None = None) -> None:
        context = context or {}
        if isinstance(node, dict):
            current = {
                "unit": node.get("unit", context.get("unit")),
                "fatherName": node.get("fatherName", context.get("fatherName")),
                "pageName": node.get("pageName", context.get("pageName")),
            }
            datas = node.get("datas")
            if isinstance(datas, list):
                for entry in datas:
                    if isinstance(entry, dict) and entry.get("pointName"):
                        found.append(
                            {
                                "pointName": entry.get("pointName"),
                                "name": entry.get("name"),
                                "value": entry.get("value"),
                                "unit": current.get("unit"),
                                "fatherName": current.get("fatherName"),
                                "pageName": current.get("pageName"),
                            }
                        )
            for value in node.values():
                if isinstance(value, (dict, list)):
                    walk(value, current)
        elif isinstance(node, list):
            for item in node:
                walk(item, context)

    walk(obj)
    return found


def extract_rows(client: PgWireClient, schema: str, start: str, end: str) -> pd.DataFrame:
    targets = TARGETS[schema]
    wanted = set(targets)
    sql = f"""
        select report_date, report_hour, report_tabindex, report_json
        from {schema}.report_hourly
        where report_date >= {sql_quote(start)}
          and report_date <= {sql_quote(end)}
          and report_tabindex in ('2', '3', '11')
        order by report_date, report_hour::int, report_tabindex;
    """
    rows = client.query(sql)
    by_time: dict[datetime, dict[str, Any]] = {}
    metadata: dict[str, dict[str, str | None]] = {}
    for row in rows:
        report_date = row["report_date"]
        report_hour = row["report_hour"]
        report_json = row["report_json"]
        if not report_date or not report_hour or not report_json:
            continue
        timestamp = datetime.strptime(f"{report_date} {int(float(report_hour)):02d}:00", "%Y-%m-%d %H:%M")
        record = by_time.setdefault(timestamp, {"timestamp": timestamp.isoformat(timespec="minutes")})
        try:
            parsed = json.loads(report_json)
        except json.JSONDecodeError:
            continue
        for item in iter_report_values(parsed):
            point = item.get("pointName")
            if point not in wanted:
                continue
            col = targets[point]
            try:
                value = float(item.get("value"))
            except (TypeError, ValueError):
                continue
            if col not in record:
                record[col] = value
                metadata[col] = {
                    "point_name": point,
                    "label": item.get("name"),
                    "unit": item.get("unit") or UNITS[col],
                    "report_context": " / ".join(
                        part for part in [item.get("fatherName"), item.get("pageName")] if part
                    ),
                }
    frame = pd.DataFrame(by_time.values()).sort_values("timestamp")
    for col in TARGETS[schema].values():
        if col not in frame.columns:
            frame[col] = pd.NA
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame.attrs["metadata"] = metadata
    return frame


def summarize(frame: pd.DataFrame, schema: str, start: str, end: str) -> dict[str, Any]:
    variables: dict[str, dict[str, float | int | None | str]] = {}
    for col in TARGETS[schema].values():
        series = pd.to_numeric(frame[col], errors="coerce").dropna()
        if series.empty:
            variables[col] = {"valid_count": 0}
            continue
        variables[col] = {
            "label": LABELS[col],
            "unit": UNITS[col],
            "valid_count": int(series.count()),
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std(ddof=0)), 4),
            "min": round(float(series.min()), 4),
            "median": round(float(series.median()), 4),
            "max": round(float(series.max()), 4),
            "p05": round(float(series.quantile(0.05)), 4),
            "p95": round(float(series.quantile(0.95)), 4),
        }

    load = pd.to_numeric(frame["load_mw"], errors="coerce")
    pressure = pd.to_numeric(frame["main_pressure_mpa"], errors="coerce")
    fuel = pd.to_numeric(frame["fuel_flow_tph"], errors="coerce")
    load_step = load.diff().abs().dropna()
    pressure_step = pressure.diff().abs().dropna()
    correlations = {}
    if load.notna().sum() > 2 and pressure.notna().sum() > 2:
        correlations["load_pressure"] = round(float(load.corr(pressure)), 4)
    if load.notna().sum() > 2 and fuel.notna().sum() > 2:
        correlations["load_fuel"] = round(float(load.corr(fuel)), 4)

    return {
        "source": {
            "database": os.environ.get("PGDATABASE", "yulin"),
            "schema": schema,
            "table": f"{schema}.report_hourly",
            "date_start": start,
            "date_end": end,
            "sampling": "hourly report snapshots",
            "raw_csv_public": False,
        },
        "rows": int(len(frame)),
        "time_start": str(frame["timestamp"].min()) if not frame.empty else None,
        "time_end": str(frame["timestamp"].max()) if not frame.empty else None,
        "variables": variables,
        "dynamic_features": {
            "load_step_abs_p95_mw_per_h": round(float(load_step.quantile(0.95)), 4) if not load_step.empty else None,
            "load_step_abs_max_mw_per_h": round(float(load_step.max()), 4) if not load_step.empty else None,
            "pressure_step_abs_p95_mpa_per_h": round(float(pressure_step.quantile(0.95)), 4)
            if not pressure_step.empty
            else None,
            "pressure_step_abs_max_mpa_per_h": round(float(pressure_step.max()), 4) if not pressure_step.empty else None,
        },
        "correlations": correlations,
        "point_metadata": frame.attrs.get("metadata", {}),
    }


def plot_snapshot(frame: pd.DataFrame, summary: dict[str, Any], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    time = pd.to_datetime(frame["timestamp"])
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.2), sharex=True)
    fig.subplots_adjust(hspace=0.18)

    axes[0].plot(time, frame["load_mw"], color="#225ea8", lw=1.2)
    axes[0].set_ylabel("Power (MW)")
    axes[0].set_title("Plant historian snapshot from Xiantao Unit 2")

    axes[1].plot(time, frame["main_pressure_mpa"], color="#238b45", lw=1.2, label="Main steam")
    axes[1].set_ylabel("Pressure (MPa)")
    axes[1].legend(loc="lower right", frameon=False)

    axes[2].plot(time, frame["main_temperature_c"], color="#cb181d", lw=1.0, label="Main steam")
    axes[2].plot(time, frame["reheat_temperature_c"], color="#f16913", lw=1.0, label="Reheat")
    axes[2].set_ylabel("Temp. (degC)")
    axes[2].legend(loc="lower right", frameon=False, ncols=2)

    axes[3].plot(time, frame["fuel_flow_tph"], color="#6a51a3", lw=1.0, label="Fuel")
    if frame["air_flow_tph"].notna().sum() > 0:
        axr = axes[3].twinx()
        axr.plot(time, frame["air_flow_tph"], color="#969696", lw=0.9, label="Air")
        axr.set_ylabel("Air (t/h)")
        axr.tick_params(axis="y", colors="#636363")
    axes[3].set_ylabel("Fuel (t/h)")
    years = sorted(pd.Series(time).dt.year.dropna().unique())
    axes[3].set_xlabel(f"Date ({years[0]})" if len(years) == 1 else "Date")
    locator = mdates.AutoDateLocator(minticks=4, maxticks=6)
    axes[3].xaxis.set_major_locator(locator)
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d" if len(years) == 1 else "%Y-%m-%d"))

    for ax in axes:
        ax.grid(True, axis="y", color="#d9d9d9", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    subtitle = (
        f"{summary['source']['date_start']} to {summary['source']['date_end']}; "
        f"n={summary['rows']} hourly records; raw historian data are proprietary"
    )
    fig.text(0.12, 0.015, subtitle, fontsize=8, color="#555555")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", choices=sorted(TARGETS), default="unit2")
    parser.add_argument("--start", default="2026-06-01")
    parser.add_argument("--end", default="2026-07-02")
    parser.add_argument("--start-ts", help="Optional inclusive ISO timestamp filter after extraction.")
    parser.add_argument("--end-ts", help="Optional inclusive ISO timestamp filter after extraction.")
    parser.add_argument("--raw-output", default="results/production_validation/raw/xiantao_unit2_hourly_2026-06-01_2026-07-02.csv")
    parser.add_argument("--summary-output", default="results/production_validation/xiantao_unit2_hourly_summary.json")
    parser.add_argument("--figure-output", default="paper/figures/Figure_9_production_historian.pdf")
    args = parser.parse_args()

    host = os.environ.get("PGHOST", "127.0.0.1")
    port = int(os.environ.get("PGPORT", "15432"))
    user = os.environ.get("PGUSER", "postgres")
    password = os.environ["PGPASSWORD"]
    database = os.environ.get("PGDATABASE", "yulin")

    with PgWireClient(host, port, user, password, database) as client:
        frame = extract_rows(client, args.schema, args.start, args.end)

    if args.start_ts:
        frame = frame[pd.to_datetime(frame["timestamp"]) >= pd.to_datetime(args.start_ts)]
    if args.end_ts:
        frame = frame[pd.to_datetime(frame["timestamp"]) <= pd.to_datetime(args.end_ts)]
    frame = frame.reset_index(drop=True)

    raw_output = Path(args.raw_output)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(raw_output, index=False, quoting=csv.QUOTE_MINIMAL)

    summary = summarize(frame, args.schema, args.start, args.end)
    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    figure_output = Path(args.figure_output)
    plot_snapshot(frame, summary, figure_output)
    print(json.dumps({"rows": len(frame), "summary": str(summary_output), "figure": str(figure_output)}, indent=2))


if __name__ == "__main__":
    main()

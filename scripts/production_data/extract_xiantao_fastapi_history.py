#!/usr/bin/env python3
"""Extract a bounded Xiantao historian window from the CRICP FastAPI adapter.

The adapter exposes ``/gethistrange_millisecondtimestamp`` with epoch-second
request bounds and millisecond timestamps in the response. This script keeps
queries deliberately small, converts the response to a private CSV, writes a
commit-safe derived summary, and renders the plant-historian figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


POINTS = {
    "DCS2_LOAD_3S": "load_mw",
    "DCS2_MAIN_PRESS": "main_pressure_mpa",
    "DCS2_MAIN_TEMP": "main_temperature_c",
    "DCS2_REHEAT_PRESS": "reheat_pressure_mpa",
    "DCS2_REHEAT_TEMP": "reheat_temperature_c",
    "DCS2_FUELFLOWRP": "fuel_flow_tph",
    "DCS2_TOTALFLOW": "air_flow_tph",
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


def bounded_request(base_url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/gethistrange_millisecondtimestamp",
        data=body,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8-sig"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"FastAPI history request failed: {exc}") from exc


def load_response(path: Path | None, base_url: str | None, args: argparse.Namespace) -> dict[str, Any]:
    if path:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    if not base_url:
        raise SystemExit("--base-url or --input-json is required")
    estimated = math.ceil((args.end_sec - args.start_sec) / args.interval_sec)
    if estimated > args.max_samples:
        raise SystemExit(
            f"Refusing {estimated} samples in one request; increase interval or "
            f"set --max-samples explicitly after checking database load."
        )
    payload = {
        "nameList": list(POINTS),
        "startSecStamp": args.start_sec,
        "endSecStamp": args.end_sec,
        "secInterval": args.interval_sec,
    }
    return bounded_request(base_url, payload, args.timeout)


def invalid_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        return float(value) <= -999.0
    except (TypeError, ValueError):
        return True


def response_to_frame(response: dict[str, Any], tz_name: str) -> pd.DataFrame:
    if response.get("code") != 200 or not response.get("success"):
        raise ValueError(f"Unsuccessful FastAPI response: {response}")
    data = response.get("data")
    if not isinstance(data, dict):
        raise ValueError("FastAPI response has no data object")
    raw_time = data.get("millisecondtimestamp")
    if not isinstance(raw_time, list) or not raw_time:
        raise ValueError("FastAPI response has no millisecondtimestamp array")

    tz = ZoneInfo(tz_name)
    rows: list[dict[str, Any]] = []
    for idx, stamp in enumerate(raw_time):
        timestamp = datetime.fromtimestamp(float(stamp) / 1000.0, tz=tz)
        row: dict[str, Any] = {
            "timestamp": timestamp.isoformat(timespec="seconds"),
            "millisecondtimestamp": int(float(stamp)),
        }
        for point, col in POINTS.items():
            values = data.get(point)
            value = values[idx] if isinstance(values, list) and idx < len(values) else values
            row[col] = None if invalid_value(value) else float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def variable_summary(frame: pd.DataFrame, col: str) -> dict[str, Any]:
    series = pd.to_numeric(frame[col], errors="coerce").dropna()
    if series.empty:
        return {"label": LABELS[col], "unit": UNITS[col], "valid_count": 0}
    return {
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


def summarize(frame: pd.DataFrame, interval_sec: int, source_note: str) -> dict[str, Any]:
    load = pd.to_numeric(frame["load_mw"], errors="coerce")
    pressure = pd.to_numeric(frame["main_pressure_mpa"], errors="coerce")
    fuel = pd.to_numeric(frame["fuel_flow_tph"], errors="coerce")
    load_step = load.diff().abs().dropna()
    pressure_step = pressure.diff().abs().dropna()
    seconds_per_hour = 3600 / interval_sec
    return {
        "source": {
            "interface": "CRICP FastAPI historian adapter",
            "endpoint": "/gethistrange_millisecondtimestamp",
            "unit": "Xiantao Unit 2",
            "sampling_interval_sec": interval_sec,
            "raw_data_public": False,
            "note": source_note,
        },
        "rows": int(len(frame)),
        "time_start": str(frame["timestamp"].min()) if not frame.empty else None,
        "time_end": str(frame["timestamp"].max()) if not frame.empty else None,
        "variables": {col: variable_summary(frame, col) for col in POINTS.values()},
        "dynamic_features": {
            "load_step_abs_p95_mw_per_sample": round(float(load_step.quantile(0.95)), 4)
            if not load_step.empty
            else None,
            "load_step_abs_max_mw_per_sample": round(float(load_step.max()), 4) if not load_step.empty else None,
            "pressure_step_abs_p95_mpa_per_sample": round(float(pressure_step.quantile(0.95)), 4)
            if not pressure_step.empty
            else None,
            "pressure_step_abs_max_mpa_per_sample": round(float(pressure_step.max()), 4)
            if not pressure_step.empty
            else None,
            "load_step_abs_p95_mw_per_h_equiv": round(float(load_step.quantile(0.95) * seconds_per_hour), 4)
            if not load_step.empty
            else None,
            "load_step_abs_max_mw_per_h_equiv": round(float(load_step.max() * seconds_per_hour), 4)
            if not load_step.empty
            else None,
            "pressure_step_abs_p95_mpa_per_h_equiv": round(
                float(pressure_step.quantile(0.95) * seconds_per_hour), 4
            )
            if not pressure_step.empty
            else None,
            "pressure_step_abs_max_mpa_per_h_equiv": round(float(pressure_step.max() * seconds_per_hour), 4)
            if not pressure_step.empty
            else None,
        },
        "correlations": {
            "load_pressure": round(float(load.corr(pressure)), 4)
            if load.notna().sum() > 2 and pressure.notna().sum() > 2
            else None,
            "load_fuel": round(float(load.corr(fuel)), 4)
            if load.notna().sum() > 2 and fuel.notna().sum() > 2
            else None,
        },
        "point_mapping": POINTS,
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
    time = pd.to_datetime(frame["timestamp"]).dt.tz_localize(None)
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
    axr = axes[3].twinx()
    axr.plot(time, frame["air_flow_tph"], color="#969696", lw=0.9, label="Air")
    axr.set_ylabel("Air (t/h)")
    axr.tick_params(axis="y", colors="#636363")
    axes[3].set_ylabel("Fuel (t/h)")
    years = sorted(pd.Series(time).dt.year.dropna().unique())
    axes[3].set_xlabel(f"Date/time ({years[0]})" if len(years) == 1 else "Date/time")
    locator = mdates.AutoDateLocator(minticks=4, maxticks=6)
    axes[3].xaxis.set_major_locator(locator)
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

    for ax in axes:
        ax.grid(True, axis="y", color="#d9d9d9", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    start_label = time.min().strftime("%Y-%m-%d %H:%M")
    end_label = time.max().strftime("%Y-%m-%d %H:%M")
    subtitle = (
        f"{start_label} to {end_label}; n={summary['rows']} records at "
        f"{summary['source']['sampling_interval_sec']} s; raw historian data are proprietary"
    )
    fig.text(0.12, 0.015, subtitle, fontsize=8, color="#555555")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("CRICP_BASE_URL"))
    parser.add_argument("--input-json", type=Path)
    parser.add_argument("--start-sec", type=int)
    parser.add_argument("--end-sec", type=int)
    parser.add_argument("--interval-sec", type=int, default=300)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--source-note", default="bounded low-rate query for manuscript grounding")
    parser.add_argument("--raw-output", type=Path, default=Path("results/production_validation/raw/xiantao_unit2_fastapi.csv"))
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/production_validation/xiantao_unit2_fastapi_summary.json"),
    )
    parser.add_argument("--figure-output", type=Path, default=Path("paper/figures/Figure_9_production_historian.pdf"))
    args = parser.parse_args()

    if not args.input_json and (args.start_sec is None or args.end_sec is None):
        raise SystemExit("--start-sec and --end-sec are required when fetching from --base-url")
    if args.interval_sec <= 0:
        raise SystemExit("--interval-sec must be positive")

    response = load_response(args.input_json, args.base_url, args)
    frame = response_to_frame(response, args.timezone)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.raw_output, index=False, quoting=csv.QUOTE_MINIMAL)

    summary = summarize(frame, args.interval_sec, args.source_note)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_snapshot(frame, summary, args.figure_output)
    print(json.dumps({"rows": len(frame), "summary": str(args.summary_output), "figure": str(args.figure_output)}, indent=2))


if __name__ == "__main__":
    main()

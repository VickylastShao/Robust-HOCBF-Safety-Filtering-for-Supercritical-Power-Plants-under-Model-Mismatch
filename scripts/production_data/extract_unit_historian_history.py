#!/usr/bin/env python3
"""Extract a bounded the unit historian window from the CRICP FastAPI adapter.

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
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.mc_figure_style import apply_times_new_roman_style


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

WINDOW_COLORS = {
    "MW01": "#0072B2",
    "MW02": "#009E73",
    "MW03": "#D55E00",
}

WINDOW_MARKERS = {
    "MW01": "o",
    "MW02": "s",
    "MW03": "^",
}


def load_validation_context(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    frame = pd.read_csv(path)
    required = [
        "window_id",
        "timestamp",
        "generator_active_power_mw",
        "main_steam_pressure_mpa",
    ]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required historian-context columns: {missing}")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["generator_active_power_mw"] = pd.to_numeric(
        frame["generator_active_power_mw"], errors="coerce"
    )
    frame["main_steam_pressure_mpa"] = pd.to_numeric(
        frame["main_steam_pressure_mpa"], errors="coerce"
    )
    return frame.dropna(
        subset=["window_id", "generator_active_power_mw", "main_steam_pressure_mpa"]
    )


def validation_window_summary(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for pair_id, group in frame.groupby("window_id", sort=True):
        rows.append(
            {
                "window": str(pair_id),
                "rows": int(len(group)),
                "time_start": group["timestamp"].min().isoformat(),
                "time_end": group["timestamp"].max().isoformat(),
                "load_min_mw": round(float(group["generator_active_power_mw"].min()), 4),
                "load_median_mw": round(float(group["generator_active_power_mw"].median()), 4),
                "load_max_mw": round(float(group["generator_active_power_mw"].max()), 4),
                "pressure_min_mpa": round(float(group["main_steam_pressure_mpa"].min()), 4),
                "pressure_median_mpa": round(float(group["main_steam_pressure_mpa"].median()), 4),
                "pressure_max_mpa": round(float(group["main_steam_pressure_mpa"].max()), 4),
            }
        )
    return rows


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
            "unit": "660 MW ultra-supercritical unit",
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


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.09, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#e2e2e2", lw=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_range_panel(
    ax: plt.Axes,
    stats: list[dict[str, Any]],
    metric: str,
    envelope: pd.Series,
    xlabel: str,
) -> None:
    pairs = [row["window"] for row in stats]
    positions = list(range(len(pairs)))
    envelope = pd.to_numeric(envelope, errors="coerce").dropna()
    ax.axvspan(float(envelope.min()), float(envelope.max()), color="#eeeeee", zorder=0, label="24 h min--max")
    ax.axvspan(float(envelope.quantile(0.05)), float(envelope.quantile(0.95)), color="#d9d9d9", zorder=1, label="24 h p05--p95")
    ax.axvline(float(envelope.median()), color="#7f7f7f", lw=0.8, ls=":", zorder=2)
    for y, row in zip(positions, stats, strict=True):
        color = WINDOW_COLORS.get(row["window"], "#4d4d4d")
        marker = WINDOW_MARKERS.get(row["window"], "o")
        lo = row[f"{metric}_min_mw"] if metric == "load" else row[f"{metric}_min_mpa"]
        med = row[f"{metric}_median_mw"] if metric == "load" else row[f"{metric}_median_mpa"]
        hi = row[f"{metric}_max_mw"] if metric == "load" else row[f"{metric}_max_mpa"]
        ax.hlines(y, lo, hi, color=color, lw=2.3, zorder=3)
        ax.plot(med, y, marker=marker, ms=4.7, color=color, mec="white", mew=0.7, zorder=4)
    ax.set_yticks(positions)
    ax.set_yticklabels(pairs)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    style_axis(ax)


def plot_snapshot(
    frame: pd.DataFrame,
    summary: dict[str, Any],
    output: Path,
    validation_log: pd.DataFrame | None = None,
) -> None:
    apply_times_new_roman_style(base_size=9)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    time = pd.to_datetime(frame["timestamp"]).dt.tz_localize(None)
    stats = validation_window_summary(validation_log)

    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 1.0])
    ax_env = fig.add_subplot(grid[:, 0])
    ax_load = fig.add_subplot(grid[0, 1])
    ax_pressure = fig.add_subplot(grid[1, 1], sharey=ax_load if stats else None)

    ax_env.plot(
        frame["load_mw"],
        frame["main_pressure_mpa"],
        color="#bdbdbd",
        lw=0.85,
        alpha=0.85,
        zorder=1,
        label="24 h historian trajectory",
    )
    ax_env.scatter(
        frame["load_mw"],
        frame["main_pressure_mpa"],
        s=15,
        color="#4d4d4d",
        alpha=0.40,
        edgecolors="none",
        zorder=2,
        label="300 s samples",
    )
    for row in stats:
        color = WINDOW_COLORS.get(row["window"], "#4d4d4d")
        marker = WINDOW_MARKERS.get(row["window"], "o")
        width = row["load_max_mw"] - row["load_min_mw"]
        height = row["pressure_max_mpa"] - row["pressure_min_mpa"]
        ax_env.add_patch(
            Rectangle(
                (row["load_min_mw"], row["pressure_min_mpa"]),
                width,
                height,
                facecolor=color,
                edgecolor=color,
                lw=1.1,
                alpha=0.16,
                zorder=3,
            )
        )
        ax_env.plot(
            row["load_median_mw"],
            row["pressure_median_mpa"],
            marker=marker,
            ms=5.6,
            color=color,
            mec="white",
            mew=0.75,
            zorder=4,
        )
        label = ax_env.text(
            row["load_median_mw"],
            row["pressure_median_mpa"],
            f" {row['window']}",
            color=color,
            fontsize=8,
            fontweight="bold",
            va="center",
            ha="left",
            zorder=5,
        )
        label.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])
    ax_env.set_xlabel("Generator active power (MW)")
    ax_env.set_ylabel("Main-steam pressure (MPa)")
    ax_env.set_title("Historian operating envelope and context-window placement")
    ax_env.text(
        0.03,
        0.97,
        "24 h historian envelope\n"
        f"load--pressure r = {summary['correlations']['load_pressure']:.3f}",
        transform=ax_env.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#4d4d4d",
    )
    style_axis(ax_env)
    panel_label(ax_env, "a")

    if stats:
        draw_range_panel(ax_load, stats, "load", frame["load_mw"], "Load range (MW)")
        draw_range_panel(ax_pressure, stats, "pressure", frame["main_pressure_mpa"], "Pressure range (MPa)")
        ax_load.set_title("Load ranges across context windows")
        ax_pressure.set_title("Pressure ranges across context windows")
        ax_pressure.tick_params(labelleft=True)
        ax_load.legend(loc="lower right", frameon=False, fontsize=7)
    else:
        ax_load.axis("off")
        ax_pressure.axis("off")
    panel_label(ax_load, "b")
    panel_label(ax_pressure, "c")

    start_label = time.min().strftime("%Y-%m-%d %H:%M")
    end_label = time.max().strftime("%Y-%m-%d %H:%M")
    subtitle = (
        f"{start_label} to {end_label}; n={summary['rows']} records at "
        f"{summary['source']['sampling_interval_sec']} s; selected windows use 5 s historian records"
    )
    fig.text(0.02, -0.015, subtitle, fontsize=8, color="#555555")
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
    parser.add_argument("--raw-output", type=Path, default=Path("results/production_validation/raw/unit_660mw_fastapi.csv"))
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/production_validation/unit_660mw_fastapi_summary.json"),
    )
    parser.add_argument("--figure-output", type=Path, default=Path("paper/figures/Figure_9_production_historian.pdf"))
    parser.add_argument(
        "--validation-context",
        type=Path,
        default=Path(
            "results/production_validation/low_mid_load_historian_context/"
            "MW01_MW03_historian_context.csv"
        ),
    )
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
    validation_log = load_validation_context(args.validation_context)
    summary["selected_validation_windows"] = validation_window_summary(validation_log)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_snapshot(frame, summary, args.figure_output, validation_log)
    print(json.dumps({"rows": len(frame), "summary": str(args.summary_output), "figure": str(args.figure_output)}, indent=2))


if __name__ == "__main__":
    main()

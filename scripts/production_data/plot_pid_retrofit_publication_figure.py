#!/usr/bin/env python3
"""Draw the manuscript-ready PID-to-retrofit historian comparison figure.

The upstream analysis script selects candidate windows. This script is only a
publication renderer for the selected 5 s pair, with metrics recomputed from
the native samples and saved beside the figure for auditability.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.mc_figure_style import apply_times_new_roman_style


PID_COLOR = "#8c510a"
RETROFIT_COLOR = "#01665e"
NEUTRAL = "#4d4d4d"


def load_window(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["timestamp", "load_mw", "pressure_mpa", "pressure_setpoint_mpa", "fuel_flow", "air_flow"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise SystemExit(f"{path} is missing required columns: {missing}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    frame["relative_min"] = (frame["timestamp"] - frame["timestamp"].iloc[0]).dt.total_seconds() / 60.0
    frame["pressure_error_mpa"] = frame["pressure_mpa"] - frame["pressure_setpoint_mpa"]
    frame["fuel_per_mw"] = frame["fuel_flow"] / frame["load_mw"].replace(0, np.nan)
    frame["air_fuel_ratio"] = frame["air_flow"] / frame["fuel_flow"].replace(0, np.nan)
    frame["period_label"] = label
    return frame


def rolling(series: pd.Series, window_samples: int) -> pd.Series:
    return series.rolling(window=window_samples, center=True, min_periods=max(3, window_samples // 3)).mean()


def ecdf(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(pd.to_numeric(series, errors="coerce").dropna().to_numpy())
    if len(values) == 0:
        return values, values
    probs = np.arange(1, len(values) + 1, dtype=float) / len(values)
    return values, probs


def load_rmse(pre: pd.DataFrame, post: pd.DataFrame) -> float:
    duration = min(pre["relative_min"].max(), post["relative_min"].max())
    grid = np.linspace(0.0, duration, min(len(pre), len(post)))
    pre_load = np.interp(grid, pre["relative_min"], pre["load_mw"])
    post_load = np.interp(grid, post["relative_min"], post["load_mw"])
    return float(np.sqrt(np.mean((post_load - pre_load) ** 2)))


def metrics(pre: pd.DataFrame, post: pd.DataFrame) -> dict[str, Any]:
    pre_std = float(pre["pressure_error_mpa"].std(ddof=0))
    post_std = float(post["pressure_error_mpa"].std(ddof=0))
    return {
        "pre_window_start": pre["timestamp"].iloc[0].isoformat(),
        "pre_window_end": pre["timestamp"].iloc[-1].isoformat(),
        "post_window_start": post["timestamp"].iloc[0].isoformat(),
        "post_window_end": post["timestamp"].iloc[-1].isoformat(),
        "sample_interval_sec": float(pre["timestamp"].diff().dropna().dt.total_seconds().median()),
        "pre_rows": int(len(pre)),
        "post_rows": int(len(post)),
        "mean_load_pre_mw": float(pre["load_mw"].mean()),
        "mean_load_post_mw": float(post["load_mw"].mean()),
        "mean_load_diff_mw": float(post["load_mw"].mean() - pre["load_mw"].mean()),
        "load_profile_rmse_mw": load_rmse(pre, post),
        "pressure_error_std_pre_mpa": pre_std,
        "pressure_error_std_post_mpa": post_std,
        "pressure_error_std_reduction_fraction": float((pre_std - post_std) / pre_std),
        "pressure_error_abs_p95_pre_mpa": float(pre["pressure_error_mpa"].abs().quantile(0.95)),
        "pressure_error_abs_p95_post_mpa": float(post["pressure_error_mpa"].abs().quantile(0.95)),
        "fuel_per_mw_mean_pre": float(pre["fuel_per_mw"].mean()),
        "fuel_per_mw_mean_post": float(post["fuel_per_mw"].mean()),
        "air_fuel_ratio_mean_pre": float(pre["air_fuel_ratio"].mean()),
        "air_fuel_ratio_mean_post": float(post["air_fuel_ratio"].mean()),
    }


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.075, 1.03, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def plot_smoothed(ax: plt.Axes, frame: pd.DataFrame, y: str, color: str, label: str, smooth_samples: int) -> None:
    ax.plot(frame["relative_min"], frame[y], color=color, alpha=0.18, lw=0.45)
    ax.plot(frame["relative_min"], rolling(frame[y], smooth_samples), color=color, lw=1.25, label=label)


def draw(pre: pd.DataFrame, post: pd.DataFrame, output_pdf: Path, output_png: Path | None, output_svg: Path | None) -> None:
    apply_times_new_roman_style(base_size=8.0)
    plt.rcParams.update({"axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7})
    smooth = max(3, int(round(60.0 / 5.0)))
    m = metrics(pre, post)

    fig = plt.figure(figsize=(7.2, 7.1), constrained_layout=True)
    grid = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 1.0, 1.05])
    axes = [
        fig.add_subplot(grid[0, :]),
        fig.add_subplot(grid[1, :]),
        fig.add_subplot(grid[2, :]),
        fig.add_subplot(grid[3, 0]),
        fig.add_subplot(grid[3, 1]),
    ]
    labels = ["a", "b", "c", "d", "e"]

    plot_smoothed(axes[0], pre, "load_mw", PID_COLOR, "Historical PID", smooth)
    plot_smoothed(axes[0], post, "load_mw", RETROFIT_COLOR, "Post-retrofit", smooth)
    axes[0].set_ylabel("Load (MW)")
    axes[0].legend(loc="upper left", frameon=False, ncols=2, handlelength=2.8)
    axes[0].text(
        0.98,
        0.88,
        "5 s samples; heavy lines: 60 s rolling mean\n"
        f"mean load diff. {m['mean_load_diff_mw']:.2f} MW; RMSE {m['load_profile_rmse_mw']:.1f} MW",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color=NEUTRAL,
    )

    plot_smoothed(axes[1], pre, "pressure_mpa", PID_COLOR, "PID pressure", smooth)
    plot_smoothed(axes[1], post, "pressure_mpa", RETROFIT_COLOR, "Post pressure", smooth)
    axes[1].plot(pre["relative_min"], rolling(pre["pressure_setpoint_mpa"], smooth), color=PID_COLOR, lw=0.9, ls=":")
    axes[1].plot(
        post["relative_min"],
        rolling(post["pressure_setpoint_mpa"], smooth),
        color=RETROFIT_COLOR,
        lw=0.9,
        ls=":",
    )
    axes[1].set_ylabel("Pressure (MPa)")
    axes[1].text(0.01, 0.90, "dotted: sliding-pressure setpoint", transform=axes[1].transAxes, fontsize=7, color=NEUTRAL, va="top")

    plot_smoothed(axes[2], pre, "pressure_error_mpa", PID_COLOR, "Historical PID", smooth)
    plot_smoothed(axes[2], post, "pressure_error_mpa", RETROFIT_COLOR, "Post-retrofit", smooth)
    axes[2].axhline(0.0, color="#7f7f7f", lw=0.7)
    axes[2].set_ylabel("Tracking error (MPa)")
    axes[2].text(
        0.98,
        0.90,
        f"std: {m['pressure_error_std_pre_mpa']:.3f} -> {m['pressure_error_std_post_mpa']:.3f} MPa\n"
        f"reduction: {100*m['pressure_error_std_reduction_fraction']:.1f}%",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#bdbdbd", "lw": 0.5},
    )

    axes[2].set_xlabel("Elapsed time in matched historian window (min)")

    pre_abs_x, pre_abs_y = ecdf(pre["pressure_error_mpa"].abs())
    post_abs_x, post_abs_y = ecdf(post["pressure_error_mpa"].abs())
    axes[3].plot(pre_abs_x, pre_abs_y, color=PID_COLOR, lw=1.35, label="Historical PID")
    axes[3].plot(post_abs_x, post_abs_y, color=RETROFIT_COLOR, lw=1.35, label="Post-retrofit")
    axes[3].axvline(m["pressure_error_abs_p95_pre_mpa"], color=PID_COLOR, lw=0.9, ls=":")
    axes[3].axvline(m["pressure_error_abs_p95_post_mpa"], color=RETROFIT_COLOR, lw=0.9, ls=":")
    axes[3].set_xlabel("|Tracking error| (MPa)")
    axes[3].set_ylabel("Empirical CDF")
    axes[3].legend(loc="lower right", frameon=False, fontsize=7)
    axes[3].text(
        0.04,
        0.92,
        "p95: %.3f -> %.3f MPa"
        % (m["pressure_error_abs_p95_pre_mpa"], m["pressure_error_abs_p95_post_mpa"]),
        transform=axes[3].transAxes,
        fontsize=7,
        color=NEUTRAL,
        va="top",
    )

    box = axes[4].boxplot(
        [pre["fuel_per_mw"].dropna(), post["fuel_per_mw"].dropna()],
        positions=[1, 2],
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "lw": 1.1},
        whiskerprops={"color": "#666666", "lw": 0.8},
        capprops={"color": "#666666", "lw": 0.8},
        boxprops={"edgecolor": "#4d4d4d", "lw": 0.8},
    )
    for patch, color in zip(box["boxes"], [PID_COLOR, RETROFIT_COLOR], strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    axes[4].plot([1], [m["fuel_per_mw_mean_pre"]], marker="D", color=PID_COLOR, ms=4.5, mec="white", mew=0.6)
    axes[4].plot([2], [m["fuel_per_mw_mean_post"]], marker="D", color=RETROFIT_COLOR, ms=4.5, mec="white", mew=0.6)
    axes[4].set_xticks([1, 2])
    axes[4].set_xticklabels(["Historical\nPID", "Post-\nretrofit"])
    axes[4].set_ylabel("Fuel/load")
    axes[4].text(
        0.05,
        0.92,
        "mean: %.3f -> %.3f"
        % (m["fuel_per_mw_mean_pre"], m["fuel_per_mw_mean_post"]),
        transform=axes[4].transAxes,
        fontsize=7,
        color=NEUTRAL,
        va="top",
    )

    for ax, label in zip(axes, labels, strict=True):
        panel_label(ax, label)
        ax.grid(True, axis="y", color="#d9d9d9", lw=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
    for ax in axes[:3]:
        ax.set_xlim(0.0, 120.0)
    axes[0].tick_params(labelbottom=False)
    axes[1].tick_params(labelbottom=False)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    if output_png:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if output_svg:
        output_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pre-csv",
        type=Path,
        default=Path("results/production_validation/raw/unit_660mw_historical_pre_match_20251105_1100_5s.csv"),
    )
    parser.add_argument(
        "--post-csv",
        type=Path,
        default=Path("results/production_validation/raw/unit_660mw_post_match_20260625_1230_5s.csv"),
    )
    parser.add_argument("--output-pdf", type=Path, default=Path("paper/figures/Figure_10_production_retrofit_evidence.pdf"))
    parser.add_argument("--output-png", type=Path, default=Path("paper/figures/Figure_10_production_retrofit_evidence.png"))
    parser.add_argument("--output-svg", type=Path, default=Path("paper/figures/Figure_10_production_retrofit_evidence.svg"))
    parser.add_argument("--metrics-json", type=Path, default=Path("results/production_validation/figure10_production_retrofit_metrics.json"))
    args = parser.parse_args()

    pre = load_window(args.pre_csv, "Historical PID")
    post = load_window(args.post_csv, "Post-retrofit")
    draw(pre, post, args.output_pdf, args.output_png, args.output_svg)
    result = metrics(pre, post)
    result["pre_csv_private"] = str(args.pre_csv)
    result["post_csv_private"] = str(args.post_csv)
    result["figure_pdf"] = str(args.output_pdf)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot high-load controller-export evidence without modifying source CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.mc_figure_style import apply_times_new_roman_style, normalize_svg_whitespace


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#6A3D9A"
GREY = "#666666"
WINDOW_COLORS = {"MW04": ORANGE, "MW05": BLUE, "MW06": GREEN}


def column(frame: pd.DataFrame, prefix: str) -> str:
    matches = [name for name in frame.columns if name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one column starting with {prefix!r}; got {matches}")
    return matches[0]


def numeric(frame: pd.DataFrame, prefix: str) -> pd.Series:
    return pd.to_numeric(frame[column(frame, prefix)], errors="raise")


def load_window(path: Path) -> tuple[str, pd.DataFrame]:
    window = path.name.split("_")[0]
    frame = pd.read_csv(path)
    frame["time"] = pd.to_datetime(frame[column(frame, "millisecondtimestamp/TS")])
    frame = frame.sort_values("time").reset_index(drop=True)
    frame["elapsed_min"] = (frame["time"] - frame["time"].iloc[0]).dt.total_seconds() / 60.0

    frame["load_mw"] = numeric(frame, "20CQTP_MW (")
    frame["separator_pressure_mpa"] = numeric(frame, "DCS2_20HAG10CP101 (")
    frame["main_pressure_mpa"] = numeric(frame, "DCS2_MAIN_PRESS (")
    frame["enthalpy_kjkg"] = numeric(frame, "DCS2_SEPARATOUT_ENTH (")
    frame["pressure_low_margin_mpa"] = frame["separator_pressure_mpa"] - numeric(frame, "PLO (")
    frame["main_pressure_upper_margin_mpa"] = numeric(frame, "PST_HI (") - frame["main_pressure_mpa"]
    frame["enthalpy_low_margin_kjkg"] = frame["enthalpy_kjkg"] - numeric(
        frame, "DCS2_SEPARATOROUT_DMDH/HLO")
    frame["enthalpy_high_margin_kjkg"] = numeric(
        frame, "DCS2_SEPARATOROUT_DSGH/HHI") - frame["enthalpy_kjkg"]
    frame["enthalpy_nearest_margin_kjkg"] = frame[
        ["enthalpy_low_margin_kjkg", "enthalpy_high_margin_kjkg"]].min(axis=1)
    power_error = frame["load_mw"] - numeric(frame, "DCS2_20LOAD_SET/NSET")
    frame["power_band_margin_mw"] = 50.0 - power_error.abs()
    frame["mg_plo"] = numeric(frame, "MG_PLO (")
    frame["min_hocbf_margin"] = numeric(frame, "MCBF (")
    frame["qp_time_ms"] = numeric(frame, "QP_MS (")
    frame["task_time_ms"] = numeric(frame, "TASK_MS (")
    frame["qp_status"] = frame[column(frame, "QP_ST (")].astype(str)
    frame["fallback_reason"] = frame[column(frame, "FB_RSN (")].astype(str)
    frame["active_constraint"] = frame[column(frame, "ACT_CSTR (")].astype(str)
    frame["fallback"] = numeric(frame, "FB (").astype(bool)
    saturation_columns = [column(frame, prefix) for prefix in ("SAT_FUEL (", "SAT_FW (", "SAT_TV (")]
    frame["saturation"] = frame[saturation_columns].apply(pd.to_numeric, errors="raise").astype(bool).any(axis=1)
    return window, frame


def summarize(window: str, frame: pd.DataFrame, source: Path) -> dict:
    recovered = frame["qp_status"].eq("optimal_recovered")
    return {
        "window": window,
        "source_file": source.name,
        "rows": int(len(frame)),
        "time_start": frame["time"].iloc[0].isoformat(),
        "time_end": frame["time"].iloc[-1].isoformat(),
        "load_range_mw": [float(frame["load_mw"].min()), float(frame["load_mw"].max())],
        "minimum_direct_margins": {
            "separator_pressure_low_mpa": float(frame["pressure_low_margin_mpa"].min()),
            "main_pressure_upper_mpa": float(frame["main_pressure_upper_margin_mpa"].min()),
            "enthalpy_nearest_kjkg": float(frame["enthalpy_nearest_margin_kjkg"].min()),
            "power_band_mw": float(frame["power_band_margin_mw"].min()),
        },
        "minimum_logged_hocbf_margin": float(frame["min_hocbf_margin"].min()),
        "qp_status_counts": {str(k): int(v) for k, v in frame["qp_status"].value_counts().items()},
        "recovered_records": int(recovered.sum()),
        "fallback_or_recovery_records": int(frame["fallback"].sum()),
        "saturation_records": int(frame["saturation"].sum()),
        "recovered_pressure_low_margin_range_mpa": (
            [float(frame.loc[recovered, "pressure_low_margin_mpa"].min()),
             float(frame.loc[recovered, "pressure_low_margin_mpa"].max())]
            if recovered.any() else None
        ),
        "recovered_mg_plo_range": (
            [float(frame.loc[recovered, "mg_plo"].min()),
             float(frame.loc[recovered, "mg_plo"].max())]
            if recovered.any() else None
        ),
        "qp_time_ms": {
            "median": float(frame["qp_time_ms"].median()),
            "p95": float(frame["qp_time_ms"].quantile(0.95)),
            "max": float(frame["qp_time_ms"].max()),
        },
        "task_time_ms": {
            "median": float(frame["task_time_ms"].median()),
            "p95": float(frame["task_time_ms"].quantile(0.95)),
            "max": float(frame["task_time_ms"].max()),
        },
    }


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontsize=10,
            fontweight="bold", ha="left", va="bottom")


def contrast_text_color(rgba: tuple[float, float, float, float]) -> str:
    """Choose black or white text from the rendered cell luminance."""
    rgb = np.asarray(rgba[:3], dtype=float)
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    luminance = float(np.dot(linear, [0.2126, 0.7152, 0.0722]))
    black_contrast = (luminance + 0.05) / 0.05
    white_contrast = 1.05 / (luminance + 0.05)
    return "black" if black_contrast >= white_contrast else "white"


def plot(windows: dict[str, pd.DataFrame], output_pdf: Path,
         output_png: Path, output_svg: Path) -> None:
    apply_times_new_roman_style(base_size=8.5)
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.25), constrained_layout=True)
    ax_load, ax_margin, ax_recovery, ax_runtime = axes.ravel()

    for name, frame in windows.items():
        ax_load.plot(frame["elapsed_min"], frame["load_mw"], lw=1.05,
                     color=WINDOW_COLORS[name], label=name)
    ax_load.axhspan(462, 660, color="#eeeeee", zorder=-2)
    ax_load.text(118, 645, "70–100% nameplate band", ha="right", va="top",
                 fontsize=7.4, color=GREY)
    ax_load.set(xlim=(0, 120), xlabel="Elapsed time (min)", ylabel="Active power (MW)",
                title="High-load operating coverage")
    ax_load.legend(frameon=False, ncols=3, loc="lower right")
    panel_label(ax_load, "a")

    margin_names = ["Sep. p\nlow\n(MPa)", "Main p\nupper\n(MPa)",
                    "Enthalpy\nnearest\n(kJ/kg)", "Power\nband\n(MW)"]
    margin_keys = ["pressure_low_margin_mpa", "main_pressure_upper_margin_mpa",
                   "enthalpy_nearest_margin_kjkg", "power_band_margin_mw"]
    raw = np.array([[windows[name][key].min() for key in margin_keys]
                    for name in windows], dtype=float)
    scaled = raw / np.maximum(raw.max(axis=0, keepdims=True), 1e-12)
    cmap = plt.get_cmap("YlGn")
    image = ax_margin.imshow(scaled, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for row, name in enumerate(windows):
        for col in range(raw.shape[1]):
            text_color = contrast_text_color(cmap(float(scaled[row, col])))
            ax_margin.text(col, row, f"{raw[row, col]:.3f}", ha="center", va="center",
                           fontsize=8.0, color=text_color, fontweight="bold")
    ax_margin.set_xticks(range(len(margin_names)), margin_names)
    ax_margin.set_yticks(range(len(windows)), list(windows))
    ax_margin.set_title("Minimum directly measured margins")
    ax_margin.tick_params(length=0)
    image.set_clim(0, 1)
    panel_label(ax_margin, "b")

    mw04 = windows["MW04"]
    recovered = mw04["qp_status"].eq("optimal_recovered")
    ax_recovery.plot(mw04["elapsed_min"], mw04["pressure_low_margin_mpa"],
                     color=BLUE, lw=1.0, label=r"Direct $PM-PLO$")
    ax_recovery.axhline(2.0, color=GREY, ls="--", lw=0.8, label="2.0 MPa guard")
    ax_recovery.scatter(mw04.loc[recovered, "elapsed_min"],
                        mw04.loc[recovered, "pressure_low_margin_mpa"],
                        color=ORANGE, s=8, zorder=4, label="Reduced-QP recovery")
    ax_recovery.set(xlim=(0, 120), xlabel="Elapsed time in MW04 (min)",
                    ylabel="Direct pressure margin (MPa)",
                    title="MW04 recovery diagnosis")
    ax_recovery_right = ax_recovery.twinx()
    ax_recovery_right.plot(mw04["elapsed_min"], mw04["mg_plo"], color=PURPLE,
                           lw=0.75, alpha=0.75, label=r"Diagnostic $MG_{PLO}$")
    ax_recovery_right.axhline(0, color=PURPLE, ls=":", lw=0.7)
    ax_recovery_right.set_ylabel(r"Pressure-low HOCBF margin $MG_{PLO}$")
    left_handles, left_labels = ax_recovery.get_legend_handles_labels()
    right_handles, right_labels = ax_recovery_right.get_legend_handles_labels()
    ax_recovery.legend(left_handles + right_handles, left_labels + right_labels,
                       frameon=True, facecolor="white", edgecolor="none",
                       framealpha=0.82, fontsize=6.9, loc="upper left", ncols=2)
    panel_label(ax_recovery, "c")

    labels = []
    values = []
    colors = []
    positions = []
    for index, (name, frame) in enumerate(windows.items()):
        labels.extend([f"{name}\nQP", f"{name}\ntask"])
        values.extend([frame["qp_time_ms"].to_numpy(), frame["task_time_ms"].to_numpy()])
        colors.extend([WINDOW_COLORS[name], WINDOW_COLORS[name]])
        positions.extend([index * 3, index * 3 + 1])
    boxes = ax_runtime.boxplot(values, positions=positions, widths=0.65, patch_artist=True,
                               showfliers=False, whis=(0, 100), medianprops={"color": "black"})
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.58)
    ax_runtime.axhline(1000, color=ORANGE, ls="--", lw=0.9, label="1000 ms task deadline")
    ax_runtime.set_yscale("log")
    ax_runtime.set_ylim(1, 1400)
    ax_runtime.set_xticks(positions, labels, rotation=25, ha="right")
    ax_runtime.tick_params(axis="x", labelsize=7.2)
    ax_runtime.set_ylabel("Execution time (ms, log scale)")
    ax_runtime.set_title("Controller timing on deployed CPU")
    ax_runtime.legend(frameon=True, facecolor="white", edgecolor="none",
                      framealpha=0.88, loc="upper right", fontsize=7.4)
    panel_label(ax_runtime, "d")

    for ax in (ax_load, ax_recovery, ax_runtime):
        ax.grid(True, color="#d9d9d9", lw=0.45, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    normalize_svg_whitespace(output_svg)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/production_validation/controller_exports_public"),
    )
    parser.add_argument("--output-pdf", type=Path,
                        default=Path("paper/figures/Figure_11_controller_log_validation.pdf"))
    parser.add_argument("--output-png", type=Path,
                        default=Path("paper/figures/Figure_11_controller_log_validation.png"))
    parser.add_argument("--output-svg", type=Path,
                        default=Path("paper/figures/Figure_11_controller_log_validation.svg"))
    parser.add_argument("--metrics", type=Path,
                        default=Path("results/production_validation/figure11_high_load_controller_metrics.json"))
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("MW0[456]*.csv"))
    if len(paths) != 3:
        raise SystemExit(f"Expected MW04--MW06 CSVs in {args.input_dir}; found {len(paths)}")
    windows = {}
    summaries = []
    for path in paths:
        name, frame = load_window(path)
        windows[name] = frame
        summaries.append(summarize(name, frame, path))
    plot(windows, args.output_pdf, args.output_png, args.output_svg)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps({"windows": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

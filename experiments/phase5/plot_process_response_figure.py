"""Plot the M&C process-response and safety-filter intervention figure."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.mc_figure_style import apply_times_new_roman_style

RESULTS_DIR = ROOT / "results" / "phase5"
FIGURE_DIR = ROOT / "paper" / "figures"
TRAJECTORY_JSON = RESULTS_DIR / "process_response_trajectories.json"
OUTPUT_BASENAME = "Figure_6_process_response"

DEFAULT_DISPLAY_START = 45
DEFAULT_DISPLAY_END = 135
DISPLAY_DT_SEC = 0.1
VIOLATION_TOL = 1e-2

OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}


@dataclass(frozen=True)
class MethodStyle:
    key: str
    label: str
    short_label: str
    color: str
    linestyle: str | tuple
    linewidth: float


METHOD_STYLES = [
    MethodStyle(
        key="hocbf",
        label="HOCBF (no GP)",
        short_label="HOCBF",
        color=OKABE_ITO["vermillion"],
        linestyle=(0, (4, 2)),
        linewidth=1.25,
    ),
    MethodStyle(
        key="gp_comparator",
        label="Comparator endpoint",
        short_label="Endpoint",
        color=OKABE_ITO["blue"],
        linestyle="-",
        linewidth=1.45,
    ),
    MethodStyle(
        key="gp_selected",
        label="RoCBF-SF tune-selected",
        short_label="Selected",
        color=OKABE_ITO["green"],
        linestyle=(0, (5, 2, 1.5, 2)),
        linewidth=1.45,
    ),
]


def _method_styles(data: dict) -> list[MethodStyle]:
    styles = []
    for base in METHOD_STYLES:
        method = data["methods"].get(base.key)
        if method is None:
            raise KeyError(f"Missing process-response method: {base.key}")
        if base.key == "gp_comparator":
            is_full_margin = "epsilon_kappa=1" in str(method["label"])
            short_label = r"$\epsilon_\kappa=1$" if is_full_margin else r"$\epsilon_\kappa=0$"
            label = (
                r"Full-margin endpoint ($\epsilon_\kappa=1$)"
                if is_full_margin
                else r"GP mean endpoint ($\epsilon_\kappa=0$)"
            )
        elif base.key == "gp_selected":
            label = r"RoCBF-SF tune-selected ($\epsilon_\kappa=0$)"
            short_label = base.short_label
        else:
            label = str(method["label"])
            short_label = base.short_label
        styles.append(
            MethodStyle(
                key=base.key,
                label=label,
                short_label=short_label,
                color=base.color,
                linestyle=base.linestyle,
                linewidth=base.linewidth,
            )
        )
    return styles


def configure_style() -> None:
    apply_times_new_roman_style(base_size=8)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.25,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        fontweight="bold",
    )


def _slice_state_time(data: dict, display_start: int, display_end: int) -> np.ndarray:
    t = np.asarray(data["metadata"]["time_state_s"], dtype=float)
    return (t >= display_start) & (t <= display_end)


def _slice_action_time(data: dict, display_start: int, display_end: int) -> np.ndarray:
    t = np.asarray(data["metadata"]["time_action_s"], dtype=float)
    return (t >= display_start) & (t < display_end)


def _display_grid(display_start: int, display_end: int,
                  dt: float = DISPLAY_DT_SEC) -> np.ndarray:
    return np.round(np.arange(display_start, display_end + 0.5 * dt, dt), 10)


def _interp_series(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Resample a series only when the stored data are coarser than the display grid."""
    return np.interp(t_dst, t_src, y_src)


def _native_marker_kwargs(color: str) -> dict:
    return {
        "marker": "o",
        "markersize": 2.2,
        "markerfacecolor": "white",
        "markeredgecolor": color,
        "markeredgewidth": 0.55,
        "linestyle": "None",
        "zorder": 4,
    }


def _nice_ylim(values: list[float], pad_frac: float = 0.08, min_pad: float = 0.1):
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    pad = max((vmax - vmin) * pad_frac, min_pad)
    return vmin - pad, vmax + pad


def _format_pct(value: float, *, compact: bool = False) -> str:
    if abs(value) < 0.005:
        return "0%"
    if abs(value - 100.0) < 0.005:
        return "100%"
    if value < 10.0:
        return f"{value:.1f}%" if compact else f"{value:.2f}%"
    return f"{value:.0f}%"


def _print_data_audit(data: dict) -> None:
    methods = _method_styles(data)
    print("Process-response trajectory audit")
    print(
        f"  native controller sample: {data['metadata'].get('dt_sec', 1.0):.1f} s; "
        f"plot grid: {DISPLAY_DT_SEC:.1f} s"
    )
    for spec in methods:
        method = data["methods"][spec.key]
        summary = _summary_with_common_tolerance(method)
        ranges = summary["ranges"]
        print(
            f"  {spec.key}: violation={summary['violation_pct']:.2f}% "
            f"QP={summary['qp_intervention_pct']:.1f}% "
            f"h={ranges['enthalpy_kj_kg'][0]:.2f}..{ranges['enthalpy_kj_kg'][1]:.2f} "
            f"p={ranges['pressure_mpa'][0]:.3f}..{ranges['pressure_mpa'][1]:.3f} "
            f"N={ranges['power_mw'][0]:.2f}..{ranges['power_mw'][1]:.2f}"
        )


def _violation_mask(method: dict) -> np.ndarray:
    constraints = np.column_stack(
        [np.asarray(values, dtype=float) for values in method["constraint_values"].values()]
    )
    return np.min(constraints, axis=1) < -VIOLATION_TOL


def _summary_with_common_tolerance(method: dict) -> dict:
    summary = dict(method["summary"])
    violation = _violation_mask(method)
    summary["n_violations"] = int(violation.sum())
    summary["violation_pct"] = float(100.0 * violation.mean())
    return summary


def plot_figure(data: dict, display_start: int = DEFAULT_DISPLAY_START,
                display_end: int = DEFAULT_DISPLAY_END) -> Path:
    configure_style()
    methods = _method_styles(data)

    if display_start < 0 or display_end <= display_start:
        raise ValueError("display interval must satisfy 0 <= start < end")

    t_state = np.asarray(data["metadata"]["time_state_s"], dtype=float)
    t_action = np.asarray(data["metadata"]["time_action_s"], dtype=float)
    state_mask = _slice_state_time(data, display_start, display_end)
    action_mask = _slice_action_time(data, display_start, display_end)
    t_state_plot = _display_grid(display_start, display_end)
    t_action_plot = _display_grid(display_start, display_end)

    fig = plt.figure(figsize=(7.2, 4.85), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.25, 1.0],
        width_ratios=[1.1, 1.2, 0.95],
    )
    ax_margin = fig.add_subplot(grid[0, :])
    ax_qp = fig.add_subplot(grid[1, 0])
    ax_events = fig.add_subplot(grid[1, 1])
    ax_summary = fig.add_subplot(grid[1, 2])

    margin_values: list[float] = []
    correction_values: list[float] = []
    full_event_rows: list[np.ndarray] = []
    min_margin_rows: list[float] = []

    for spec in methods:
        method = data["methods"][spec.key]
        margin = np.asarray(method["pressure_low_margin_mpa"], dtype=float)
        correction = np.asarray(method["qp_correction_norm"], dtype=float)
        correction_norm = correction / float(data["metadata"]["v_max"])
        margin_plot = _interp_series(t_state[state_mask], margin[state_mask], t_state_plot)
        correction_plot = _interp_series(
            t_action[action_mask], correction_norm[action_mask], t_action_plot
        )

        margin_values.extend(margin_plot.tolist())
        correction_values.extend(correction_plot.tolist())
        min_margin_rows.append(
            float(method["summary"]["ranges"]["pressure_low_margin_mpa"][0])
        )

        ax_margin.plot(
            t_state_plot,
            margin_plot,
            color=spec.color,
            linestyle=spec.linestyle,
            linewidth=spec.linewidth,
            label=spec.label,
        )
        ax_margin.plot(
            t_state[state_mask],
            margin[state_mask],
            color=spec.color,
            **_native_marker_kwargs(spec.color),
        )
        ax_qp.plot(
            t_action_plot,
            correction_plot,
            color=spec.color,
            linestyle=spec.linestyle,
            linewidth=spec.linewidth,
        )
        ax_qp.plot(
            t_action[action_mask],
            correction_norm[action_mask],
            color=spec.color,
            **_native_marker_kwargs(spec.color),
        )

    m_ymin, m_ymax = _nice_ylim(margin_values, pad_frac=0.08, min_pad=0.05)
    m_ymin = min(m_ymin, -0.05)
    m_ymax = max(m_ymax, 0.05)
    ax_margin.axhspan(m_ymin, 0.0, color="#F4A582", alpha=0.22, linewidth=0)
    ax_margin.axhline(0.0, color="0.15", linestyle="-", linewidth=0.9)
    ax_margin.text(
        0.985,
        0.0,
        r"$p_{\rm st}-p_{\min}=0$",
        transform=ax_margin.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=7,
        color="0.25",
    )
    ax_margin.text(
        0.015,
        0.08,
        "open markers: native 1 s samples",
        transform=ax_margin.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.7,
        color="0.30",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
    )
    ax_margin.set_ylim(m_ymin, m_ymax)
    ax_margin.set_xlim(display_start, display_end)
    ax_margin.set_ylabel(r"Pressure lower-bound margin $p_{\rm st}-p_{\min}$ (MPa)")
    ax_margin.set_xlabel(f"Time (s; diagnostic interval {display_start}--{display_end} s)")
    ax_margin.grid(axis="y", color="0.88", linewidth=0.5)
    ax_margin.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        handlelength=2.7,
        columnspacing=1.2,
    )
    panel_label(ax_margin, "a")

    q_ymin = min(0.0, min(correction_values) - 0.05)
    q_ymax = max(max(correction_values) + 0.08, 0.35)
    ax_qp.set_ylim(q_ymin, q_ymax)
    ax_qp.set_ylabel(r"QP correction $\|v_{\rm safe}-v_{\rm ref}\|_2/v_{\max}$")
    ax_qp.set_xlabel(f"Time (s; diagnostic interval {display_start}--{display_end} s)")
    ax_qp.grid(axis="y", color="0.88", linewidth=0.5)
    ax_qp.set_xlim(display_start, display_end)
    panel_label(ax_qp, "b")

    for spec in methods:
        method = data["methods"][spec.key]
        violation = _violation_mask(method)
        intervened = np.asarray(method["intervened"], dtype=bool)
        rejected = ~np.asarray(method["qp_accepted"], dtype=bool)
        status = np.zeros_like(violation, dtype=int)
        status[intervened] = 1
        status[violation] = 2
        status[rejected] = 3
        full_event_rows.append(status)

    event_matrix = np.vstack(full_event_rows)
    event_cmap = ListedColormap(
        ["#F7F7F7", "#9ECAE1", OKABE_ITO["vermillion"], "#666666"]
    )
    event_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], event_cmap.N)
    n_steps = int(data["metadata"]["n_steps"])
    dt_sec = float(data["metadata"].get("dt_sec", 1.0))
    total_time = n_steps * dt_sec
    ax_events.imshow(
        event_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=event_cmap,
        norm=event_norm,
        extent=[0, total_time, len(methods) - 0.5, -0.5],
    )
    ax_events.set_yticks(range(len(methods)))
    ax_events.set_yticklabels([spec.short_label for spec in methods])
    ax_events.set_xlabel("Rollout time (s)")
    ax_events.set_title("Full-rollout events", pad=2)
    ax_events.set_xticks([0, total_time / 3, 2 * total_time / 3, total_time])
    ax_events.set_xticklabels(["0", "100", "200", "300"])
    for idx, spec in enumerate(methods):
        summary = _summary_with_common_tolerance(data["methods"][spec.key])
        rejected_pct = 100.0 * (
            1.0 - np.mean(np.asarray(data["methods"][spec.key]["qp_accepted"], dtype=float))
        )
        ax_events.text(
            1.02,
            idx,
            f"{_format_pct(summary['violation_pct'], compact=True)} V; "
            f"{_format_pct(summary['qp_intervention_pct'], compact=True)} QP; "
            f"{_format_pct(rejected_pct, compact=True)} R",
            transform=ax_events.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=6.6,
            color="0.25",
        )
    ax_events.legend(
        handles=[
            Patch(facecolor="#9ECAE1", edgecolor="none", label="QP active"),
            Patch(facecolor=OKABE_ITO["vermillion"], edgecolor="none", label="violation"),
            Patch(facecolor="#666666", edgecolor="none", label="QP rejected / fallback"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=6.5,
        handlelength=1.0,
        columnspacing=0.7,
    )
    panel_label(ax_events, "c")

    y_positions = np.arange(len(methods))
    summary_min = float(min(min_margin_rows))
    summary_max = float(max(min_margin_rows))
    summary_pad = max(0.08 * (summary_max - summary_min), 0.05)
    summary_xlim = (
        min(summary_min - summary_pad, -0.05),
        max(summary_max + summary_pad, 0.05),
    )
    ax_summary.axvline(0.0, color="0.2", linewidth=0.8)
    ax_summary.axvspan(
        summary_xlim[0], 0.0, color="#F4A582", alpha=0.16, linewidth=0
    )
    for y_pos, spec, min_margin in zip(y_positions, methods, min_margin_rows):
        ax_summary.hlines(
            y_pos,
            min(0.0, min_margin),
            max(0.0, min_margin),
            color=spec.color,
            linewidth=1.6,
        )
        ax_summary.scatter(
            min_margin,
            y_pos,
            s=34,
            color=spec.color,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        viol = _summary_with_common_tolerance(data["methods"][spec.key])["violation_pct"]
        label = f"{_format_pct(viol, compact=True)} viol."
        ax_summary.text(
            summary_xlim[1] - 0.02 * (summary_xlim[1] - summary_xlim[0]),
            y_pos,
            label,
            ha="right",
            va="center",
            fontsize=6.8,
            color="0.25",
        )
    ax_summary.set_yticks(y_positions)
    ax_summary.set_yticklabels([spec.short_label for spec in methods])
    ax_summary.invert_yaxis()
    ax_summary.set_xlim(*summary_xlim)
    ax_summary.set_xlabel("Min. pressure margin (MPa)", fontsize=7, labelpad=2)
    ax_summary.set_title("Rollout summary", pad=2)
    ax_summary.grid(axis="x", color="0.88", linewidth=0.5)
    panel_label(ax_summary, "d")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base = FIGURE_DIR / OUTPUT_BASENAME
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".png"), dpi=300)
    plt.close(fig)

    with Image.open(base.with_suffix(".png")) as image:
        image.convert("L").save(FIGURE_DIR / f"{OUTPUT_BASENAME}_grayscale.png")

    print(f"Saved {base.with_suffix('.pdf')}")
    print(f"Saved {base.with_suffix('.png')}")
    return base.with_suffix(".pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=TRAJECTORY_JSON)
    parser.add_argument("--display-start", type=int, default=DEFAULT_DISPLAY_START)
    parser.add_argument("--display-end", type=int, default=DEFAULT_DISPLAY_END)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Missing {args.input}. Run collect_process_response_figure.py first."
        )
    with args.input.open() as f:
        data = json.load(f)
    _print_data_audit(data)
    plot_figure(data, display_start=args.display_start, display_end=args.display_end)


if __name__ == "__main__":
    main()

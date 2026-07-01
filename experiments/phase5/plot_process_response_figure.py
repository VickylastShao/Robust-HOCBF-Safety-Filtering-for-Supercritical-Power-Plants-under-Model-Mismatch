"""Plot the M&C process-response and safety-filter intervention figure."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "phase5"
FIGURE_DIR = ROOT / "paper" / "figures"
TRAJECTORY_JSON = RESULTS_DIR / "process_response_trajectories.json"
OUTPUT_BASENAME = "Figure_6_process_response"

DEFAULT_DISPLAY_STEPS = 10

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


METHODS = [
    MethodStyle(
        key="hocbf",
        label="HOCBF (no GP)",
        short_label="HOCBF",
        color=OKABE_ITO["vermillion"],
        linestyle=(0, (4, 2)),
        linewidth=1.25,
    ),
    MethodStyle(
        key="gp_k0",
        label=r"GP-HOCBF ($\epsilon_\kappa=0$)",
        short_label=r"$\epsilon_\kappa=0$",
        color=OKABE_ITO["blue"],
        linestyle="-",
        linewidth=1.45,
    ),
    MethodStyle(
        key="gp_k01",
        label=r"RoCBF-Net ($\epsilon_\kappa=0.1$)",
        short_label=r"$\epsilon_\kappa=0.1$",
        color=OKABE_ITO["green"],
        linestyle=(0, (5, 2, 1.5, 2)),
        linewidth=1.45,
    ),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
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


def _slice_state_time(data: dict, display_steps: int) -> np.ndarray:
    t = np.asarray(data["metadata"]["time_state_s"], dtype=float)
    return t <= display_steps


def _slice_action_time(data: dict, display_steps: int) -> np.ndarray:
    t = np.asarray(data["metadata"]["time_action_s"], dtype=float)
    return t < display_steps


def _nice_ylim(values: list[float], pad_frac: float = 0.08, min_pad: float = 0.1):
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    pad = max((vmax - vmin) * pad_frac, min_pad)
    return vmin - pad, vmax + pad


def _print_data_audit(data: dict) -> None:
    print("Process-response trajectory audit")
    for spec in METHODS:
        method = data["methods"][spec.key]
        summary = method["summary"]
        ranges = summary["ranges"]
        print(
            f"  {spec.key}: violation={summary['violation_pct']:.2f}% "
            f"QP={summary['qp_intervention_pct']:.1f}% "
            f"h={ranges['enthalpy_kj_kg'][0]:.2f}..{ranges['enthalpy_kj_kg'][1]:.2f} "
            f"p={ranges['pressure_mpa'][0]:.3f}..{ranges['pressure_mpa'][1]:.3f} "
            f"N={ranges['power_mw'][0]:.2f}..{ranges['power_mw'][1]:.2f}"
        )


def plot_figure(data: dict, display_steps: int = DEFAULT_DISPLAY_STEPS) -> Path:
    configure_style()

    bounds = data["metadata"]["bounds"]
    t_state = np.asarray(data["metadata"]["time_state_s"], dtype=float)
    t_action = np.asarray(data["metadata"]["time_action_s"], dtype=float)
    state_mask = _slice_state_time(data, display_steps)
    action_mask = _slice_action_time(data, display_steps)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 4.8),
        sharex=False,
        constrained_layout=True,
    )
    ax_p, ax_h, ax_margin, ax_qp = axes.ravel()

    pressure_values: list[float] = []
    enthalpy_values: list[float] = []
    margin_values: list[float] = []
    correction_values: list[float] = []

    for spec in METHODS:
        method = data["methods"][spec.key]
        outputs = method["outputs"]
        pressure = np.asarray(outputs["pressure_mpa"], dtype=float)
        enthalpy = np.asarray(outputs["enthalpy_kj_kg"], dtype=float)
        margin = np.asarray(method["enthalpy_margin_kj_kg"], dtype=float)
        correction = np.asarray(method["qp_correction_norm"], dtype=float)
        correction_norm = correction / float(data["metadata"]["v_max"])

        pressure_values.extend(pressure[state_mask].tolist())
        enthalpy_values.extend(enthalpy[state_mask].tolist())
        margin_values.extend(margin[state_mask].tolist())
        correction_values.extend(correction_norm[action_mask].tolist())

        ax_p.plot(
            t_state[state_mask],
            pressure[state_mask],
            color=spec.color,
            linestyle=spec.linestyle,
            linewidth=spec.linewidth,
            label=spec.label,
        )
        ax_h.plot(
            t_state[state_mask],
            enthalpy[state_mask],
            color=spec.color,
            linestyle=spec.linestyle,
            linewidth=spec.linewidth,
        )
        ax_margin.plot(
            t_state[state_mask],
            margin[state_mask],
            color=spec.color,
            linestyle=spec.linestyle,
            linewidth=spec.linewidth,
        )
        ax_qp.plot(
            t_action[action_mask],
            correction_norm[action_mask],
            color=spec.color,
            linestyle=spec.linestyle,
            linewidth=spec.linewidth,
            label=(
                f"{spec.short_label}, "
                f"{method['summary']['qp_intervention_pct']:.0f}% active"
            ),
        )

    p_low, p_high = bounds["pressure_mpa"]
    h_low, _ = bounds["enthalpy_kj_kg"]
    p_ymin, p_ymax = _nice_ylim(pressure_values, min_pad=0.15)
    ax_p.set_ylim(p_ymin, p_ymax)
    ax_p.text(
        0.02,
        0.95,
        rf"Pressure limits: {p_low:.0f} to {p_high:.0f} MPa",
        transform=ax_p.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="0.3",
    )
    ax_p.set_ylabel(r"Steam pressure $p_{st}$ (MPa)")
    ax_p.set_xlabel("Time (s)")
    ax_p.grid(axis="y", color="0.88", linewidth=0.5)
    ax_p.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        handlelength=2.7,
        columnspacing=1.2,
    )
    panel_label(ax_p, "a")

    h_ymin = min(min(enthalpy_values) - 4.0, h_low - 5.0)
    h_ymax = max(max(enthalpy_values) + 4.0, h_low + 35.0)
    ax_h.axhspan(h_ymin, h_low, color="#F4A582", alpha=0.22, linewidth=0)
    ax_h.axhline(h_low, color="0.15", linestyle=":", linewidth=1.0)
    ax_h.text(
        0.98,
        h_low + 0.8,
        r"$h_{\min}$",
        transform=ax_h.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=7,
        color="0.2",
    )
    ax_h.set_ylim(h_ymin, h_ymax)
    ax_h.set_ylabel(r"Separator enthalpy $h_m$ (kJ kg$^{-1}$)")
    ax_h.set_xlabel("Time (s)")
    ax_h.grid(axis="y", color="0.88", linewidth=0.5)
    panel_label(ax_h, "b")

    m_ymin = min(min(margin_values) - 3.0, -8.0)
    m_ymax = max(max(margin_values) + 3.0, 30.0)
    ax_margin.axhspan(m_ymin, 0.0, color="#F4A582", alpha=0.22, linewidth=0)
    ax_margin.axhline(0.0, color="0.15", linestyle="-", linewidth=0.9)
    ax_margin.set_ylim(m_ymin, m_ymax)
    ax_margin.set_ylabel(r"Enthalpy margin $h_m-h_{\min}$ (kJ kg$^{-1}$)")
    ax_margin.set_xlabel("Time (s)")
    ax_margin.grid(axis="y", color="0.88", linewidth=0.5)
    panel_label(ax_margin, "c")

    q_ymin = min(0.0, min(correction_values) - 0.05)
    q_ymax = max(max(correction_values) + 0.08, 0.35)
    ax_qp.set_ylim(q_ymin, q_ymax)
    ax_qp.set_ylabel(r"QP correction $\|v_{\rm safe}-v_{\rm ref}\|_2/v_{\max}$")
    ax_qp.set_xlabel("Time (s)")
    ax_qp.grid(axis="y", color="0.88", linewidth=0.5)
    ax_qp.legend(
        loc="upper right",
        frameon=False,
        handlelength=2.4,
        borderaxespad=0.2,
    )
    panel_label(ax_qp, "d")

    for ax in axes.ravel():
        ax.set_xlim(0, display_steps)

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
    parser.add_argument("--display-steps", type=int, default=DEFAULT_DISPLAY_STEPS)
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
    plot_figure(data, display_steps=args.display_steps)


if __name__ == "__main__":
    main()

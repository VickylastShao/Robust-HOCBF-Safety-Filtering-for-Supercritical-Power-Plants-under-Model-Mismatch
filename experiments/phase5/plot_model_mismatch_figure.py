"""Plot a Nature-style model-mismatch diagnostic figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "phase5"
FIGURE_DIR = ROOT / "paper" / "figures"
INPUT_JSON = RESULTS_DIR / "model_mismatch_diagnostic.json"
OUTPUT_BASENAME = "Figure_8_model_mismatch"

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


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "legend.fontsize": 5.4,
            "axes.linewidth": 0.5,
            "lines.linewidth": 0.85,
            "lines.markersize": 2.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
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
        -0.15,
        1.08,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        fontweight="bold",
    )


def _mask_time(t: np.ndarray, display_steps: int, *, include_endpoint: bool) -> np.ndarray:
    if include_endpoint:
        return t <= display_steps
    return t < display_steps


def _nice_ylim(values: list[float], pad_frac: float = 0.08, min_pad: float = 0.05):
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    pad = max((vmax - vmin) * pad_frac, min_pad)
    return vmin - pad, vmax + pad


def _print_data_audit(data: dict, display_steps: int) -> None:
    meta = data["metadata"]
    coverage = meta["coverage"]
    print("Model-mismatch diagnostic audit")
    print(f"  scenario: {meta['scenario']}")
    print(
        f"  GP: N={meta['gp_n_training_points']}, "
        f"beta={meta['gp_beta']:.3f}, gamma_N={meta['gp_gamma_N']:.3f}"
    )
    print(
        f"  enthalpy violation over full rollout: "
        f"{meta['enthalpy_violation_pct']:.2f}% "
        f"(first step={meta['first_enthalpy_violation_step']})"
    )
    for key in ("p_m", "h_m"):
        c = coverage[key]
        print(
            f"  {key} GP-UCB coverage: {c['coverage_pct']:.1f}% "
            f"(max ratio={c['max_ratio']:.3f}, mean ratio={c['mean_ratio']:.3f})"
        )
    print(f"  display window: 0-{display_steps} s")


def plot_figure(data: dict, display_steps: int = DEFAULT_DISPLAY_STEPS) -> Path:
    configure_style()

    meta = data["metadata"]
    traj = data["trajectory"]
    beta = float(meta["gp_beta"])
    bounds = meta["bounds"]
    h_low = float(bounds["enthalpy_kj_kg"][0])

    t_state = np.asarray(meta["time_state_s"], dtype=float)
    t_action = np.asarray(meta["time_action_s"], dtype=float)
    state_mask = _mask_time(t_state, display_steps, include_endpoint=True)
    action_mask = _mask_time(t_action, display_steps, include_endpoint=False)

    true_margin = np.asarray(traj["enthalpy_margin_true"], dtype=float)
    nominal_one_step = np.asarray(
        traj["enthalpy_margin_nominal_one_step"], dtype=float
    )
    true_one_step = np.asarray(traj["enthalpy_margin_true_one_step"], dtype=float)
    nominal_margin_plot = np.concatenate(([true_margin[0]], nominal_one_step))
    true_margin_plot = np.concatenate(([true_margin[0]], true_one_step))

    actual_p = np.asarray(traj["actual_residual"]["p_m"], dtype=float)
    actual_h = np.asarray(traj["actual_residual"]["h_m"], dtype=float)
    mu_p = np.asarray(traj["gp_mu"]["p_m"], dtype=float)
    mu_h = np.asarray(traj["gp_mu"]["h_m"], dtype=float)
    bound_p = np.asarray(traj["gp_bound"]["p_m"], dtype=float)
    bound_h = np.asarray(traj["gp_bound"]["h_m"], dtype=float)
    ratio_p = np.asarray(traj["normalized_abs_error"]["p_m"], dtype=float)
    ratio_h = np.asarray(traj["normalized_abs_error"]["h_m"], dtype=float)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 4.65),
        constrained_layout=True,
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # Panel a: one-step response mismatch.
    panel_a_values = (
        nominal_margin_plot[state_mask].tolist()
        + true_margin_plot[state_mask].tolist()
        + [0.0]
    )
    y_min, y_max = _nice_ylim(panel_a_values, min_pad=2.0)
    ax_a.axhspan(y_min, 0.0, color="#F4A582", alpha=0.20, linewidth=0)
    ax_a.axhline(0.0, color="0.15", linestyle=":", linewidth=0.7)
    ax_a.plot(
        t_state[state_mask],
        nominal_margin_plot[state_mask],
        color=OKABE_ITO["blue"],
        linestyle=(0, (4, 2)),
        linewidth=0.9,
        label="Nominal one-step prediction",
    )
    ax_a.plot(
        t_state[state_mask],
        true_margin_plot[state_mask],
        color=OKABE_ITO["vermillion"],
        linestyle="-",
        linewidth=0.95,
        label="True S3 plant response",
    )
    ax_a.set_xlim(0, display_steps)
    ax_a.set_ylim(y_min, y_max)
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel(r"Enthalpy margin $h_m-h_{\min}$ (kJ kg$^{-1}$)")
    ax_a.text(
        0.03,
        0.90,
        r"identical $x_k, v_k$ pairs",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=5.4,
        color="0.35",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.6),
    )
    ax_a.text(
        0.03,
        0.08,
        r"unsafe",
        transform=ax_a.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.4,
        color="0.35",
    )
    ax_a.grid(axis="y", color="0.90", linewidth=0.35)
    ax_a.text(
        display_steps - 0.15,
        nominal_margin_plot[int(display_steps)] + 0.55,
        "Nominal model",
        ha="right",
        va="bottom",
        fontsize=5.5,
        color=OKABE_ITO["blue"],
    )
    ax_a.text(
        display_steps - 2.0,
        true_margin_plot[int(display_steps)] + 2.1,
        "True S3 plant",
        ha="right",
        va="bottom",
        fontsize=5.5,
        color=OKABE_ITO["vermillion"],
    )
    panel_label(ax_a, "a")

    def residual_panel(ax, actual, mu, bound, ylabel, color_actual, color_gp):
        mask = action_mask
        ax.fill_between(
            t_action[mask],
            (mu - bound)[mask],
            (mu + bound)[mask],
            color=color_gp,
            alpha=0.18,
            linewidth=0,
            label=r"GP mean $\pm\,\beta\sigma$",
        )
        ax.plot(
            t_action[mask],
            actual[mask],
            color=color_actual,
            linestyle="-",
            linewidth=0.9,
            label="Actual residual",
        )
        ax.plot(
            t_action[mask],
            mu[mask],
            color=color_gp,
            linestyle=(0, (4, 2)),
            linewidth=0.85,
            label="GP posterior mean",
        )
        ax.axhline(0.0, color="0.75", linewidth=0.45)
        ax.set_xlim(0, display_steps)
        values = (
            actual[mask].tolist()
            + (mu - bound)[mask].tolist()
            + (mu + bound)[mask].tolist()
            + [0.0]
        )
        ax.set_ylim(*_nice_ylim(values, min_pad=0.4))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="0.90", linewidth=0.35)

    residual_panel(
        ax_b,
        actual_p,
        mu_p,
        bound_p,
        r"Pressure residual $\Delta f_p$ (MPa s$^{-1}$)",
        OKABE_ITO["vermillion"],
        OKABE_ITO["blue"],
    )
    ax_b.legend(
        loc="lower right",
        frameon=False,
        handlelength=2.6,
        borderaxespad=0.15,
    )
    panel_label(ax_b, "b")

    residual_panel(
        ax_c,
        actual_h,
        mu_h,
        bound_h,
        r"Enthalpy residual $\Delta f_h$ (kJ kg$^{-1}$ s$^{-1}$)",
        OKABE_ITO["vermillion"],
        OKABE_ITO["green"],
    )
    ax_c.legend(
        loc="upper right",
        frameon=False,
        handlelength=2.6,
        borderaxespad=0.15,
    )
    panel_label(ax_c, "c")

    # Panel d: normalized posterior residual coverage. A small display floor
    # avoids hiding zero-valued residual errors on the log scale.
    ratio_floor = 1e-4
    ratio_p_plot = np.maximum(ratio_p, ratio_floor)
    ratio_h_plot = np.maximum(ratio_h, ratio_floor)
    ax_d.axhspan(1.0, 1.25, color="#F4A582", alpha=0.16, linewidth=0)
    ax_d.axhline(
        1.0,
        color="0.15",
        linestyle=":",
        linewidth=0.7,
        label="GP-UCB bound",
    )
    ax_d.plot(
        t_action[action_mask],
        ratio_p_plot[action_mask],
        color=OKABE_ITO["blue"],
        linestyle=(0, (4, 2)),
        linewidth=0.9,
        label=r"Pressure residual / bound",
    )
    ax_d.plot(
        t_action[action_mask],
        ratio_h_plot[action_mask],
        color=OKABE_ITO["green"],
        linestyle="-",
        linewidth=0.9,
        label=r"Enthalpy residual / bound",
    )
    ax_d.set_xlim(0, display_steps)
    ax_d.set_yscale("log")
    ax_d.set_ylim(ratio_floor, 1.25)
    ax_d.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    ax_d.set_xlabel("Time (s)")
    ax_d.set_ylabel("Normalized residual error")
    ax_d.grid(axis="y", which="major", color="0.90", linewidth=0.35)
    ax_d.legend(
        loc="upper right",
        frameon=False,
        handlelength=2.6,
        borderaxespad=0.15,
    )
    panel_label(ax_d, "d")

    for ax in axes.ravel():
        ax.tick_params(width=0.5, length=2.5, pad=1.5)

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
    parser.add_argument("--input", type=Path, default=INPUT_JSON)
    parser.add_argument("--display-steps", type=int, default=DEFAULT_DISPLAY_STEPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Missing {args.input}. Run collect_model_mismatch_figure.py first."
        )
    with args.input.open() as f:
        data = json.load(f)
    _print_data_audit(data, args.display_steps)
    plot_figure(data, display_steps=args.display_steps)


if __name__ == "__main__":
    main()

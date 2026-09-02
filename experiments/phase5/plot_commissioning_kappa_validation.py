"""Plot tune/test evidence for the commissioning robustness factor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.mc_figure_style import apply_times_new_roman_style, normalize_svg_whitespace


DEFAULT_INPUT = (
    ROOT
    / "results"
    / "phase5_ccs7_kappa_20260902"
    / "selection_summary.json"
)
DEFAULT_OUTPUT = ROOT / "paper" / "figures" / "Figure_2"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GREY = "#777777"


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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        fontweight="bold",
    )


def plot(summary: dict, output: Path) -> None:
    configure_style()
    rows = summary["rows"]
    kappa = np.asarray([row["epsilon_kappa"] for row in rows], dtype=float)
    violation = 100.0 * np.asarray([row["violation_rate"] for row in rows])
    rejection = 100.0 * np.asarray([row["qp_rejection_rate"] for row in rows])
    max_violation = 100.0 * np.asarray(
        [row["maximum_seed_violation_rate"] for row in rows]
    )
    max_rejection = 100.0 * np.asarray(
        [row["maximum_seed_qp_rejection_rate"] for row in rows]
    )
    tune_x = np.arange(len(kappa), dtype=float)
    selected = float(summary["selected_epsilon_kappa"])
    selected_idx = int(np.flatnonzero(kappa == selected)[0])
    holdout = summary["holdout_test"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), constrained_layout=True)
    ax_tune, ax_test = axes

    ax_tune.axhspan(0.0, 0.5, color="#E8F3EE", alpha=0.85, linewidth=0)
    ax_tune.axhline(1.0, color=BLUE, linestyle=":", linewidth=0.9)
    ax_tune.axhline(0.5, color=ORANGE, linestyle=":", linewidth=0.9)
    ax_tune.plot(
        tune_x,
        violation,
        color=BLUE,
        marker="o",
        markersize=4.0,
        label="Pooled violation rate",
    )
    ax_tune.plot(
        tune_x,
        rejection,
        color=ORANGE,
        marker="s",
        markersize=3.8,
        label="Pooled QP rejection rate",
    )
    ax_tune.plot(
        tune_x,
        max_violation,
        color=BLUE,
        linestyle="--",
        alpha=0.55,
        label="Maximum seed violation rate",
    )
    ax_tune.plot(
        tune_x,
        max_rejection,
        color=ORANGE,
        linestyle="--",
        alpha=0.55,
        label="Maximum seed QP rejection rate",
    )
    ax_tune.axvline(selected_idx, color=GREEN, linestyle="-.", linewidth=1.0)
    selected_row = next(row for row in rows if row["epsilon_kappa"] == selected)
    ax_tune.annotate(
        rf"selected $\epsilon_\kappa={selected:.2f}$",
        xy=(selected_idx, 100.0 * selected_row["violation_rate"]),
        xytext=(1.65, 5.0),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color=GREEN, linewidth=0.8),
        color=GREEN,
        fontsize=7,
    )
    ax_tune.set_xlabel(r"Robustness factor $\epsilon_\kappa$")
    ax_tune.set_ylabel("Controller-sample rate (%)")
    tune_samples = int(rows[0]["total_samples"])
    ax_tune.set_title(
        f"Tune seeds 0–2 ({tune_samples:,} samples per setting)", pad=3)
    ax_tune.set_xlim(-0.25, len(kappa) - 0.75)
    ax_tune.set_ylim(0.0, max(18.0, 1.08 * max(max_violation.max(), max_rejection.max())))
    ax_tune.set_xticks(tune_x)
    ax_tune.set_xticklabels([f"{value:g}" for value in kappa])
    ax_tune.grid(axis="y", color="0.88", linewidth=0.5)
    ax_tune.legend(loc="upper left", frameon=False, ncol=1, handlelength=2.2)
    panel_label(ax_tune, "a")

    per_seed = holdout.get("per_seed")
    if not per_seed:
        raise ValueError("Holdout summary must include per_seed results")
    seed_labels = [f"Seed {item['seed']}" for item in per_seed] + ["Pooled"]
    seed_violation = [
        100.0 * item["violation_rate"] for item in per_seed
    ] + [100.0 * holdout["violation_rate"]]
    seed_rejection = [
        100.0 * item["qp_rejection_rate"] for item in per_seed
    ] + [100.0 * holdout["qp_rejection_rate"]]
    x = np.arange(len(seed_labels), dtype=float)
    width = 0.34
    ax_test.axhspan(0.0, 0.5, color="#E8F3EE", alpha=0.85, linewidth=0)
    bars_v = ax_test.bar(
        x - width / 2,
        seed_violation,
        width,
        color=BLUE,
        label="Violation rate",
    )
    bars_r = ax_test.bar(
        x + width / 2,
        seed_rejection,
        width,
        color=ORANGE,
        label="QP rejection rate",
    )
    ax_test.axhline(1.0, color=BLUE, linestyle=":", linewidth=0.9)
    ax_test.axhline(0.5, color=ORANGE, linestyle=":", linewidth=0.9)
    for bars in (bars_v, bars_r):
        for bar in bars:
            value = bar.get_height()
            ax_test.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.035,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.7,
            )
    ax_test.set_xticks(x)
    ax_test.set_xticklabels(seed_labels)
    ax_test.set_ylabel("Controller-sample rate (%)")
    ax_test.set_title(
        (rf"Held-out test at fixed $\epsilon_\kappa={selected:g}$ "
         f"({int(holdout['total_samples']):,} samples)"),
        pad=3,
    )
    ax_test.set_ylim(0.0, 1.1)
    ax_test.grid(axis="y", color="0.88", linewidth=0.5)
    ax_test.legend(loc="upper right", frameon=False)
    ax_test.text(
        0.98,
        0.37,
        "No retuning after holdout",
        transform=ax_test.transAxes,
        color=GREY,
        fontsize=7,
        ha="right",
        va="center",
    )
    panel_label(ax_test, "b")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"))
    svg_path = output.with_suffix(".svg")
    fig.savefig(svg_path)
    normalize_svg_whitespace(svg_path)
    fig.savefig(output.with_suffix(".png"), dpi=600)
    plt.close(fig)

    with Image.open(output.with_suffix(".png")) as image:
        image.convert("L").save(output.parent / f"{output.name}_grayscale.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = json.loads(args.input.read_text())
    plot(summary, args.output)
    print(f"Saved {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

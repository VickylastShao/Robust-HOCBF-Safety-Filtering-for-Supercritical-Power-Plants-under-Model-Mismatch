#!/usr/bin/env python3
"""Render the controlled GP quantity/quality sensitivity diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.mc_figure_style import apply_times_new_roman_style


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#6A3D9A"
QUALITY_COLORS = {0.0: BLUE, 0.05: ORANGE, 0.10: PURPLE}


def load_records(results_dir: Path) -> list[dict]:
    records = []
    for path in sorted(results_dir.glob("n*_q*_seed*.json")):
        payload = json.loads(path.read_text())
        payload["_path"] = str(path)
        records.append(payload)
    if len(records) != 45:
        raise ValueError(f"Expected 45 sensitivity records; found {len(records)}")
    return records


def aggregate(records: list[dict]) -> dict[tuple[int, float], dict]:
    groups: dict[tuple[int, float], list[dict]] = defaultdict(list)
    for record in records:
        key = (int(record["training_sample_size"]),
               float(record["contamination_fraction"]))
        groups[key].append(record)

    summary = {}
    for key, rows in groups.items():
        nrmse = [float(np.mean(row["prediction"]["nrmse_by_validation_sd"]))
                 for row in rows]
        coverage = [float(np.mean(row["prediction"]["predictive_interval_95_coverage"]))
                    for row in rows]
        closed_loop = {}
        for mode in ("epsilon_kappa_0", "commissioned"):
            violations = sum(row["closed_loop"][mode]["violation_count"] for row in rows)
            samples = sum(row["closed_loop"][mode]["total_samples"] for row in rows)
            rejected = sum(row["closed_loop"][mode]["qp_infeasible_count"] for row in rows)
            attempts = sum(row["closed_loop"][mode]["qp_attempt_count"] for row in rows)
            closed_loop[mode] = {
                "violation_count": int(violations),
                "total_samples": int(samples),
                "violation_rate": float(violations / samples),
                "rejected_count": int(rejected),
                "attempts": int(attempts),
                "rejected_rate": float(rejected / attempts),
            }
        summary[key] = {
            "seeds": sorted(int(row["seed"]) for row in rows),
            "nrmse_mean": float(np.mean(nrmse)),
            "nrmse_sd": float(np.std(nrmse)),
            "coverage_mean": float(np.mean(coverage)),
            "coverage_sd": float(np.std(coverage)),
            "closed_loop": closed_loop,
        }
    return summary


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.04, label, transform=ax.transAxes, fontsize=10,
            fontweight="bold", ha="left", va="bottom")


def plot(summary: dict[tuple[int, float], dict], pdf: Path,
         png: Path, svg: Path, commissioned_kappa: float) -> None:
    apply_times_new_roman_style(base_size=8.5)
    sample_sizes = [100, 250, 500]
    qualities = [0.0, 0.05, 0.10]
    quality_labels = {0.0: "Clean", 0.05: "5% target corruption",
                      0.10: "10% target corruption"}
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.1), constrained_layout=True)
    ax_error, ax_coverage, ax_clean, ax_corrupt = axes.ravel()

    for quality in qualities:
        means = [summary[(n, quality)]["nrmse_mean"] for n in sample_sizes]
        sds = [summary[(n, quality)]["nrmse_sd"] for n in sample_sizes]
        ax_error.errorbar(sample_sizes, means, yerr=sds, marker="o", capsize=2.5,
                          lw=1.15, color=QUALITY_COLORS[quality],
                          label=quality_labels[quality])
    ax_error.set_yscale("log")
    ax_error.set(xlabel="GP training transitions", ylabel="Held-out NRMSE (log scale)",
                 title="Residual-prediction error")
    ax_error.set_xticks(sample_sizes)
    ax_error.legend(frameon=False, fontsize=7.2)
    panel_label(ax_error, "a")

    for quality in qualities:
        means = [100 * summary[(n, quality)]["coverage_mean"] for n in sample_sizes]
        sds = [100 * summary[(n, quality)]["coverage_sd"] for n in sample_sizes]
        ax_coverage.errorbar(sample_sizes, means, yerr=sds, marker="o", capsize=2.5,
                             lw=1.15, color=QUALITY_COLORS[quality],
                             label=quality_labels[quality])
    ax_coverage.axhline(95, color="#666666", ls="--", lw=0.8)
    ax_coverage.set(xlabel="GP training transitions", ylabel="Empirical 95% interval coverage (%)",
                    title="Held-out interval coverage", ylim=(90, 101))
    ax_coverage.set_xticks(sample_sizes)
    panel_label(ax_coverage, "b")

    width = 0.34
    positions = np.arange(len(sample_sizes))
    k0_viol = [100 * summary[(n, 0.0)]["closed_loop"]["epsilon_kappa_0"]["violation_rate"]
               for n in sample_sizes]
    k01_viol = [100 * summary[(n, 0.0)]["closed_loop"]["commissioned"]["violation_rate"]
                for n in sample_sizes]
    k01_reject = [100 * summary[(n, 0.0)]["closed_loop"]["commissioned"]["rejected_rate"]
                  for n in sample_sizes]
    ax_clean.bar(positions - width / 2, k0_viol, width, color=BLUE,
                 label=r"$\epsilon_\kappa=0$: violation")
    ax_clean.bar(positions + width / 2, k01_viol, width, color=GREEN,
                 label=rf"$\epsilon_\kappa={commissioned_kappa:g}$: violation")
    ax_clean.scatter(positions + width / 2, k01_reject, marker="x", s=35,
                     color=ORANGE,
                     label=rf"$\epsilon_\kappa={commissioned_kappa:g}$: QP rejected")
    ax_clean.set_xticks(positions, [str(n) for n in sample_sizes])
    ax_clean.set(xlabel="Clean GP training transitions", ylabel="Rate across 2500 samples (%)",
                 title="Closed-loop clean-data sensitivity")
    ax_clean.legend(frameon=False, fontsize=7.0)
    panel_label(ax_clean, "c")

    matrix = np.array([
        [100 * summary[(n, q)]["closed_loop"]["commissioned"]["rejected_rate"]
         for n in sample_sizes]
        for q in (0.05, 0.10)
    ])
    im = ax_corrupt.imshow(matrix, cmap="OrRd", vmin=0, vmax=100, aspect="auto")
    for row, quality in enumerate((0.05, 0.10)):
        for col, n in enumerate(sample_sizes):
            outcome = summary[(n, quality)]["closed_loop"]["commissioned"]
            ax_corrupt.text(col, row,
                            f"V {100*outcome['violation_rate']:.1f}%\nR {100*outcome['rejected_rate']:.1f}%",
                            ha="center", va="center", fontsize=8.0,
                            color="white" if matrix[row, col] > 55 else "black")
    ax_corrupt.set_xticks(range(len(sample_sizes)), [str(n) for n in sample_sizes])
    ax_corrupt.set_yticks([0, 1], ["5%", "10%"])
    ax_corrupt.set(xlabel="GP training transitions", ylabel="Injected target corruption",
                   title=rf"$\epsilon_\kappa={commissioned_kappa:g}$ under corrupted targets")
    ax_corrupt.tick_params(length=0)
    panel_label(ax_corrupt, "d")
    cbar = fig.colorbar(im, ax=ax_corrupt, fraction=0.046, pad=0.03)
    cbar.set_label("QP rejection rate (%)")

    for ax in (ax_error, ax_coverage, ax_clean):
        ax.grid(True, color="#d9d9d9", lw=0.45, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)


def serializable(summary: dict[tuple[int, float], dict]) -> dict:
    return {
        f"n{n}_q{quality:.2f}": value
        for (n, quality), value in sorted(summary.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path,
                        default=Path("results/phase5_gp_data_sensitivity_20260831"))
    parser.add_argument("--output-pdf", type=Path,
                        default=Path("paper/figures/Figure_GP_data_sensitivity.pdf"))
    parser.add_argument("--output-png", type=Path,
                        default=Path("paper/figures/Figure_GP_data_sensitivity.png"))
    parser.add_argument("--output-svg", type=Path,
                        default=Path("paper/figures/Figure_GP_data_sensitivity.svg"))
    parser.add_argument("--summary", type=Path,
                        default=Path("results/phase5_gp_data_sensitivity_20260831/summary.json"))
    args = parser.parse_args()
    records = load_records(args.results_dir)
    kappas = {float(row["commissioned_epsilon_kappa"]) for row in records}
    if len(kappas) != 1:
        raise ValueError(f"Expected one commissioned epsilon_kappa; found {sorted(kappas)}")
    commissioned_kappa = kappas.pop()
    summary = aggregate(records)
    plot(summary, args.output_pdf, args.output_png, args.output_svg,
         commissioned_kappa)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(serializable(summary), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

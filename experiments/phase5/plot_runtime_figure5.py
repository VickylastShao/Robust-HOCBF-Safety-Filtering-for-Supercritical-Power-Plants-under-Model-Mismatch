"""Regenerate the M&C supplementary runtime comparison figure.

The figure is intentionally method-neutral: the manuscript now treats RoCBF-SF
as a safety-filter overlay rather than a PPO-specific contribution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.mc_figure_style import apply_times_new_roman_style  # noqa: E402


def main() -> None:
    apply_times_new_roman_style(base_size=9.0)

    labels = [
        "Proposal\nonly",
        "Projection QP\n(no JIT)",
        "Safety-filter\npipeline",
        "NMPC\nSLSQP",
        "Distilled\nsurrogate",
    ]
    times_ms = [10.0, 578.0, 25.0, 254.0, 1.8]
    colors = ["#7A7F86", "#D56B5F", "#3B6EA8", "#B7791F", "#3E8B5B"]

    fig, ax = plt.subplots(figsize=(6.6, 3.25))
    x = range(len(labels))
    bars = ax.bar(x, times_ms, width=0.68, color=colors, edgecolor="#2F3337", linewidth=0.6)

    ax.set_yscale("log")
    ax.set_ylim(1.0, 800.0)
    ax.set_ylabel("Per-step computation time (ms, log scale)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title("Online computation time comparison", pad=6)
    ax.grid(axis="y", which="major", color="#D9DEE3", linewidth=0.6)
    ax.grid(axis="y", which="minor", color="#EEF1F4", linewidth=0.4)
    ax.set_axisbelow(True)

    for idx, (bar, value) in enumerate(zip(bars, times_ms)):
        label = f"{value:.0f} ms" if value >= 10 else f"{value:.1f} ms"
        if idx == 2:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 0.78,
                label,
                ha="center",
                va="top",
                fontsize=8,
                fontweight="bold",
                color="white",
            )
            continue
        offset = 8 if idx == 2 else 5
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    # The pipeline bar reports a 25 ms median with 27 ms p95 in the stored timing audit.
    ax.plot([2 - 0.24, 2 + 0.24], [27.0, 27.0], color="#1F2933", linewidth=1.0)
    ax.text(2.0, 35.0, "p95=27 ms", ha="center", va="bottom", fontsize=7)

    ratio = times_ms[3] / times_ms[2]
    ax.text(
        2.55,
        70,
        f"{ratio:.1f}x faster\nthan NMPC",
        ha="center",
        va="center",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#B8C0C8", "linewidth": 0.5},
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out_dir = ROOT / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "Figure_5.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "Figure_5.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

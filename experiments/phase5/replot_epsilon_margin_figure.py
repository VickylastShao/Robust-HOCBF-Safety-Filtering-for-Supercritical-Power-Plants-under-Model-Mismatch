"""Replot Figure_3 epsilon-margin data with stable Times New Roman fonts.

The original epsilon-margin calculation is expensive because it depends on a
scenario-specific GP. The checked-in SVG already contains the vetted curve and
histogram data. This script reads those vector coordinates back, maps them to
data coordinates, and redraws the figure through the Matplotlib PDF backend so
DOCX/PDF conversion does not introduce abnormal letter spacing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.mc_figure_style import apply_times_new_roman_style


SOURCE_SVG = ROOT / "paper" / "figures" / "Figure_3.svg"
OUTPUT_BASE = ROOT / "paper" / "figures" / "Figure_3"


def _path_points(svg: str, group_id: str) -> np.ndarray:
    pattern = rf'<g id="{re.escape(group_id)}">.*?<path d="(.*?)"'
    match = re.search(pattern, svg, flags=re.S)
    if not match:
        raise ValueError(f"Cannot find SVG path for {group_id}")
    coords = re.findall(r"[ML]\s+([-0-9.]+)\s+([-0-9.]+)", match.group(1))
    return np.asarray([(float(x), float(y)) for x, y in coords], dtype=float)


def _patch_rect(svg: str, group_id: str) -> tuple[float, float, float, float]:
    pattern = rf'<g id="{re.escape(group_id)}">.*?<path d="(.*?)"'
    match = re.search(pattern, svg, flags=re.S)
    if not match:
        raise ValueError(f"Cannot find SVG patch for {group_id}")
    coords = re.findall(r"[ML]\s+([-0-9.]+)\s+([-0-9.]+)", match.group(1))
    pts = np.asarray([(float(x), float(y)) for x, y in coords], dtype=float)
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def _left_x(x_svg: np.ndarray) -> np.ndarray:
    tick_x_2660 = 69.704783
    tick_x_2680 = 104.458646
    scale = (tick_x_2680 - tick_x_2660) / 20.0
    return 2660.0 + (x_svg - tick_x_2660) / scale


def _left_y(y_svg: np.ndarray) -> np.ndarray:
    # Log-y transform inferred from two annotated points in the source SVG.
    y_svg_mean, y_data_mean = 103.215350, 2.41
    y_svg_min, y_data_min = 190.343384, 0.48
    a = (y_svg_mean - y_svg_min) / (np.log10(y_data_mean) - np.log10(y_data_min))
    b = y_svg_mean - a * np.log10(y_data_mean)
    return 10.0 ** ((y_svg - b) / a)


def _right_x(x_svg: np.ndarray) -> np.ndarray:
    tick_x_2 = 314.607418
    tick_x_4 = 346.362705
    scale = (tick_x_4 - tick_x_2) / 2.0
    return 2.0 + (x_svg - tick_x_2) / scale


def _right_y(y_svg: np.ndarray) -> np.ndarray:
    tick_y_0 = 215.785373
    tick_y_02 = 175.107769
    scale = (tick_y_0 - tick_y_02) / 0.2
    return (tick_y_0 - y_svg) / scale


def _load_source_data(svg_path: Path) -> dict[str, np.ndarray]:
    svg = svg_path.read_text(encoding="utf-8")
    eps_h_pts = _path_points(svg, "line2d_30")
    eps_p_pts = _path_points(svg, "line2d_31")
    kde_pts = _path_points(svg, "line2d_55")

    bars = []
    for idx in range(9, 22):
        x0, y0, x1, _y1 = _patch_rect(svg, f"patch_{idx}")
        bars.append((_right_x(np.array([x0]))[0], _right_x(np.array([x1]))[0], _right_y(np.array([y0]))[0]))

    return {
        "h_x": _left_x(eps_h_pts[:, 0]),
        "h_y": _left_y(eps_h_pts[:, 1]),
        "p_x": _left_x(eps_p_pts[:, 0]),
        "p_y": _left_y(eps_p_pts[:, 1]),
        "kde_x": _right_x(kde_pts[:, 0]),
        "kde_y": _right_y(kde_pts[:, 1]),
        "bars": np.asarray(bars, dtype=float),
    }


def configure_style() -> None:
    apply_times_new_roman_style(base_size=8.5)
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "figure.dpi": 300,
        }
    )


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.10,
        1.06,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        fontweight="bold",
    )


def plot_figure(svg_path: Path = SOURCE_SVG, output_base: Path = OUTPUT_BASE) -> Path:
    data = _load_source_data(svg_path)
    configure_style()

    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.55),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
        constrained_layout=True,
    )

    purple = "#756BB1"
    blue = "#2C7FB8"
    red = "#D62728"

    ax_l.plot(data["h_x"], data["h_y"], color=purple, linewidth=1.8,
              label=r"enthalpy low margin $\epsilon_h$")
    ax_l.plot(data["p_x"], data["p_y"], color=blue, linewidth=1.25, alpha=0.82,
              label=r"pressure low margin $\epsilon_p$ (CV=71%)")
    ax_l.axhline(2.41, color=red, linestyle=(0, (4, 3)), linewidth=1.0,
                 label=r"mean $\epsilon_h=2.41$")
    ax_l.axvline(2698.0, color="0.35", linestyle=(0, (2, 2)), linewidth=0.9)
    ax_l.scatter([2698.0, 2749.0], [0.48, 8.89], color=purple,
                 edgecolor="white", linewidth=0.55, s=24, zorder=5)
    ax_l.annotate("0.48", xy=(2698.0, 0.48), xytext=(2687.5, 0.33),
                  arrowprops=dict(arrowstyle="-", lw=0.6, color="0.35"),
                  fontsize=7.4, ha="center")
    ax_l.annotate("8.89", xy=(2749.0, 8.89), xytext=(2733.0, 9.7),
                  arrowprops=dict(arrowstyle="-", lw=0.6, color="0.35"),
                  fontsize=7.4, ha="center")
    ax_l.text(2701.0, 1.02, r"$h_m^0=2698$", fontsize=7.5, color="#6B7280")
    ax_l.text(2733.0, 0.38, r"CV($\epsilon_h$) = 105%",
              fontsize=7.8, bbox=dict(facecolor="white", edgecolor="#E5E7EB", alpha=0.95, pad=1.6))
    ax_l.set_yscale("log")
    ax_l.set_xlim(2647, 2754)
    ax_l.set_ylim(0.28, 11.0)
    ax_l.set_title("State-dependent robustness margin under S3", pad=4)
    ax_l.set_xlabel(r"Separator enthalpy $h_m$ (kJ kg$^{-1}$)")
    ax_l.set_ylabel(r"Robustness margin $\epsilon(x)$")
    ax_l.grid(axis="both", color="#E5E7EB", linewidth=0.5, alpha=0.65)
    ax_l.legend(loc="upper left", frameon=False, handlelength=2.2)
    panel_label(ax_l, "a")

    bars = data["bars"]
    ax_r.bar(
        bars[:, 0],
        bars[:, 2],
        width=bars[:, 1] - bars[:, 0],
        align="edge",
        color=purple,
        alpha=0.34,
        edgecolor="white",
        linewidth=0.6,
        label="grid samples",
    )
    ax_r.plot(data["kde_x"], data["kde_y"], color=purple, linewidth=1.8, label="KDE")
    ax_r.axvline(2.41, color=red, linestyle=(0, (4, 3)), linewidth=1.0, label="mean")
    ax_r.text(
        0.97,
        0.82,
        "$\\mu=2.41$\n$\\sigma=2.54$\nCV=105%",
        transform=ax_r.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="#E5E7EB", alpha=0.95, pad=2.0),
    )
    ax_r.set_title("Distribution over operating range", pad=4)
    ax_r.set_xlabel(r"$\epsilon_h$")
    ax_r.set_ylabel("Density")
    ax_r.set_xlim(0.55, 9.25)
    ax_r.set_ylim(0.0, 0.95)
    ax_r.grid(axis="y", color="#E5E7EB", linewidth=0.5, alpha=0.8)
    ax_r.legend(loc="center right", frameon=False)
    panel_label(ax_r, "b")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_name(output_base.name + "_replotted.svg"))
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    plt.close(fig)
    with Image.open(output_base.with_suffix(".png")) as image:
        image.convert("L").save(output_base.with_name(output_base.name + "_grayscale.png"))
    print(f"Saved {output_base.with_suffix('.pdf')}")
    return output_base.with_suffix(".pdf")


def main() -> None:
    plot_figure()


if __name__ == "__main__":
    main()

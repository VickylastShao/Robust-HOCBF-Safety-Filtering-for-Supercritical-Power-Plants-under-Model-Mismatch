"""Generate Figure 2: S3 failure mechanism and kappa calibration.

The figure focuses on the enthalpy bottleneck rather than all process states:

1. Representative S3 enthalpy trajectories for HOCBF, GP-HOCBF k=0, and
   GP-HOCBF k=0.1.
2. The corresponding enthalpy safety margin, h_m - h_min.
3. Seed-level S3 violation rates from the kappa sweep.

Run on a GPU host for trajectory generation. Re-running locally without JAX is
supported after the trajectory JSON has been generated.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "phase5"
FIGURE_DIR = ROOT / "paper" / "figures"
TRAJECTORY_JSON = RESULTS_DIR / "figure2_mechanism_trajectories.json"
KAPPA_SUMMARY_JSON = RESULTS_DIR / "figure2_s3_kappa_summary.json"

LOAD_RATIO = 1.0
N_STEPS = 300
DISPLAY_STEPS = 10
H_LOW = 2670.0
H_HIGH = 2830.0

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
class MethodSpec:
    key: str
    label: str
    color: str
    linestyle: str
    linewidth: float


METHODS = [
    MethodSpec(
        key="hocbf",
        label="HOCBF (no GP)",
        color=OKABE_ITO["vermillion"],
        linestyle=(0, (4, 2)),
        linewidth=1.3,
    ),
    MethodSpec(
        key="gp_k0",
        label=r"GP-HOCBF ($\epsilon_\kappa=0$)",
        color=OKABE_ITO["blue"],
        linestyle="-",
        linewidth=1.5,
    ),
    MethodSpec(
        key="gp_k01",
        label=r"GP-HOCBF ($\epsilon_\kappa=0.1$)",
        color=OKABE_ITO["green"],
        linestyle=(0, (5, 2, 1.5, 2)),
        linewidth=1.5,
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
            "lines.linewidth": 1.3,
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


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise RuntimeError(
            f"{TRAJECTORY_JSON} does not exist and JAX is unavailable. "
            "Run this script on gpu205/gpu206 first, then rerun locally for plotting."
        ) from exc
    return jax, jnp


def generate_trajectories(n_steps: int = N_STEPS, force: bool = False) -> dict:
    """Generate representative trajectories under S3 coupled perturbation."""
    if TRAJECTORY_JSON.exists() and not force:
        with TRAJECTORY_JSON.open() as f:
            return json.load(f)

    warnings.filterwarnings("ignore")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")
    sys.path.insert(0, str(ROOT))

    jax, jnp = _require_jax()

    from rocbf.qp.diff_qp import DifferentiableQP
    from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
    from envs.ccs.constraints import CCSConstraints5th
    from experiments.phase5.methods_5th import (
        _make_hocbf_5th,
        _make_robust_hocbf_5th,
        _pretrain_gp_5th,
    )

    dynamics = USCCSDynamics5th(load_ratio=LOAD_RATIO)
    x0, u0 = dynamics.equilibrium(LOAD_RATIO)
    constraint = CCSConstraints5th(
        p_bounds=(13.0, 24.0),
        h_bounds=(H_LOW, H_HIGH),
        power_deviation=50.0,
        power_target=1000.0,
    )

    print("Building HOCBF (no GP)...", flush=True)
    hocbf = _make_hocbf_5th(dynamics, constraint, u0, use_phi_scaled_g=True)

    print("Pretraining scenario-specific GP for S3 (N=500)...", flush=True)
    gp = _pretrain_gp_5th(
        LOAD_RATIO,
        n_pretrain=500,
        key=jax.random.key(42),
        scenario="coupled",
        scenario_specific=True,
    )

    print("Building GP-HOCBF filters (k=0 and k=0.1)...", flush=True)
    gp_k0 = _make_robust_hocbf_5th(
        dynamics,
        constraint,
        gp,
        u0,
        use_mean_correction=True,
        epsilon_kappa=0.0,
        use_phi_scaled_g=True,
    )
    gp_k01 = _make_robust_hocbf_5th(
        dynamics,
        constraint,
        gp,
        u0,
        use_mean_correction=True,
        epsilon_kappa=0.1,
        use_phi_scaled_g=True,
    )

    qp = DifferentiableQP(v_max=5.0, scale_constraints=True)
    env = UncertainUSCCSDynamics5th(
        load_ratio=LOAD_RATIO, uncertainty_scenario="coupled"
    )

    def rollout(hocbf_obj, label: str) -> dict:
        x = x0.copy()
        out = {
            "h_m": [],
            "h_margin": [],
            "pressure": [],
            "power": [],
            "violation": [],
        }
        for _ in range(n_steps):
            v_rl = jnp.zeros(3)
            a_mat, b_vec = hocbf_obj.qp_matrices(x)
            v_safe = qp.solve_with_rl_action(
                v_rl, a_mat, b_vec, differentiable=False
            )
            if isinstance(v_safe, tuple):
                v_safe = v_safe[0]
            next_x = env.step_stabilized_phi_scaled(x, jnp.asarray(v_safe))
            cv = constraint.check_all(next_x)
            violated = any(float(v) < 0 for v in cv.values())

            h_val = float(next_x[2])
            p_st = float(next_x[1] - 0.13 * next_x[1] ** 0.882)
            out["h_m"].append(h_val)
            out["h_margin"].append(h_val - H_LOW)
            out["pressure"].append(p_st)
            out["power"].append(float(next_x[3]))
            out["violation"].append(bool(violated))
            x = next_x

        n_viol = int(sum(out["violation"]))
        out["n_violations"] = n_viol
        out["violation_pct"] = 100.0 * n_viol / n_steps
        print(
            f"  {label}: {n_viol}/{n_steps} violations "
            f"({out['violation_pct']:.2f}%)",
            flush=True,
        )
        return out

    trajectories = {
        "metadata": {
            "scenario": "S3: Coupled",
            "load_ratio": LOAD_RATIO,
            "n_steps": n_steps,
            "gp_pretrain_points": 500,
            "gp_seed": 42,
            "rl_action": "zero deviation action",
            "h_bounds": [H_LOW, H_HIGH],
        },
        "methods": {
            "hocbf": rollout(hocbf, "HOCBF (no GP)"),
            "gp_k0": rollout(gp_k0, "GP-HOCBF k=0"),
            "gp_k01": rollout(gp_k01, "GP-HOCBF k=0.1"),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with TRAJECTORY_JSON.open("w") as f:
        json.dump(trajectories, f, indent=2)
    print(f"Saved trajectory data: {TRAJECTORY_JSON}", flush=True)
    return trajectories


def load_kappa_seed_data() -> tuple[list[dict], list[dict]]:
    """Load seed-level kappa sweep data for S3 coupled perturbation."""
    seed_rows = []
    pattern = re.compile(r"kappa([0-9.]+)_s3_coupled_seed(\d+)\.json$")
    for path in sorted((RESULTS_DIR / "kappa_sweep").glob("kappa*_s3_coupled_seed*.json")):
        match = pattern.match(path.name)
        if not match:
            continue
        kappa = float(match.group(1))
        seed = int(match.group(2))
        with path.open() as f:
            result = json.load(f)
        value = result.get("violation_rate", [np.nan])
        violation_rate = value[0] if isinstance(value, list) else value
        seed_rows.append(
            {
                "kappa": kappa,
                "seed": seed,
                "violation_pct": float(violation_rate) * 100.0,
            }
        )

    if not seed_rows:
        raise FileNotFoundError(
            f"No S3 kappa sweep files found under {RESULTS_DIR / 'kappa_sweep'}"
        )

    summary_rows = []
    for kappa in sorted({row["kappa"] for row in seed_rows}):
        values = [
            row["violation_pct"] for row in seed_rows if row["kappa"] == kappa
        ]
        summary_rows.append(
            {
                "kappa": kappa,
                "n": len(values),
                "mean_violation_pct": mean(values),
                "sd_violation_pct": stdev(values) if len(values) > 1 else 0.0,
                "seed_values_pct": values,
            }
        )

    with KAPPA_SUMMARY_JSON.open("w") as f:
        json.dump({"seed_rows": seed_rows, "summary": summary_rows}, f, indent=2)
    return seed_rows, summary_rows


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.075,
        1.08,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        fontweight="bold",
    )


def plot_figure(trajectories: dict, seed_rows: list[dict], summary_rows: list[dict]) -> None:
    configure_style()

    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.08, 1.0])
    ax_traj = fig.add_subplot(grid[0, :])
    ax_margin = fig.add_subplot(grid[1, 0])
    ax_kappa = fig.add_subplot(grid[1, 1])

    total_steps = trajectories["metadata"]["n_steps"]
    t = np.arange(total_steps)
    all_h = []
    all_margin = []
    for spec in METHODS:
        data = trajectories["methods"][spec.key]
        h_vals = np.asarray(data["h_m"], dtype=float)
        margins = np.asarray(data["h_margin"], dtype=float)
        all_h.extend(h_vals.tolist())
        all_margin.extend(margins.tolist())

        ax_traj.plot(
            t,
            h_vals,
            color=spec.color,
            linestyle=spec.linestyle,
            linewidth=spec.linewidth,
            label=spec.label,
        )
        ax_margin.plot(
            t,
            margins,
            color=spec.color,
            linestyle=spec.linestyle,
            linewidth=spec.linewidth,
        )

    y_min = min(min(all_h) - 3.0, H_LOW - 5.0)
    y_max = max(max(all_h) + 3.0, H_LOW + 25.0)
    ax_traj.axhspan(y_min, H_LOW, color="#F4A582", alpha=0.22, linewidth=0)
    ax_traj.axhline(H_LOW, color="0.15", linestyle=":", linewidth=1.1)
    ax_traj.text(
        0.995,
        H_LOW + 0.6,
        r"$h_{\min}=2670$",
        transform=ax_traj.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=7,
        color="0.2",
    )
    ax_traj.set_ylim(y_min, y_max)
    ax_traj.set_xlim(0, DISPLAY_STEPS)
    ax_traj.set_xticks(np.arange(0, DISPLAY_STEPS + 1, 5))
    ax_traj.set_ylabel(r"Separator enthalpy $h_m$ (kJ kg$^{-1}$)")
    ax_traj.set_xlabel(
        f"Time step (first {DISPLAY_STEPS} of {total_steps}-step rollout)"
    )
    ax_traj.grid(axis="y", color="0.88", linewidth=0.5)
    ax_traj.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        handlelength=2.7,
        columnspacing=1.6,
    )
    panel_label(ax_traj, "a")

    m_min = min(min(all_margin) - 2.0, -5.0)
    m_max = max(max(all_margin) + 2.0, 20.0)
    ax_margin.axhspan(m_min, 0.0, color="#F4A582", alpha=0.22, linewidth=0)
    ax_margin.axhline(0.0, color="0.15", linestyle="-", linewidth=0.9)
    ax_margin.set_ylim(m_min, m_max)
    ax_margin.set_xlim(0, DISPLAY_STEPS)
    ax_margin.set_xticks(np.arange(0, DISPLAY_STEPS + 1, 5))
    ax_margin.set_ylabel(r"Enthalpy margin $h_m-h_{\min}$ (kJ kg$^{-1}$)")
    ax_margin.set_xlabel(
        f"Time step (first {DISPLAY_STEPS} of {total_steps}-step rollout)"
    )
    ax_margin.grid(axis="y", color="0.88", linewidth=0.5)
    panel_label(ax_margin, "b")

    kappas = [row["kappa"] for row in summary_rows]
    means = [row["mean_violation_pct"] for row in summary_rows]
    ax_kappa.axhspan(0, 1.0, color=OKABE_ITO["green"], alpha=0.10, linewidth=0)
    ax_kappa.plot(
        kappas,
        means,
        color="0.25",
        linestyle="-",
        linewidth=0.9,
        marker="o",
        markersize=3.5,
        label="mean",
    )

    for row in seed_rows:
        same_k = [r for r in seed_rows if r["kappa"] == row["kappa"]]
        same_k = sorted(same_k, key=lambda x: x["seed"])
        idx = [r["seed"] for r in same_k].index(row["seed"])
        jitter = 0.018 * (idx - (len(same_k) - 1) / 2.0)
        ax_kappa.scatter(
            row["kappa"] + jitter,
            row["violation_pct"],
            s=18,
            color=OKABE_ITO["blue"],
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )

    for row in summary_rows:
        kappa = row["kappa"]
        val = row["mean_violation_pct"]
        ax_kappa.hlines(
            val,
            kappa - 0.035,
            kappa + 0.035,
            colors="0.1",
            linewidth=1.1,
            zorder=4,
        )

    ax_kappa.annotate(
        r"$\kappa=0.1$ restores zero violation",
        xy=(0.1, 0.0),
        xytext=(0.24, 12.0),
        fontsize=7,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", lw=0.7, color="0.25"),
    )
    ax_kappa.text(
        0.98,
        0.07,
        "<1% safety threshold",
        transform=ax_kappa.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color=OKABE_ITO["green"],
    )
    ax_kappa.set_xlim(-0.04, 1.04)
    ax_kappa.set_ylim(-1.5, max(45.0, max(means) + 4.0))
    ax_kappa.set_xticks(kappas)
    ax_kappa.set_xticklabels(["0", "0.1", "0.3", "0.5", "1.0"])
    ax_kappa.set_xlabel(r"Robustness scaling $\epsilon_\kappa$")
    ax_kappa.set_ylabel("S3 violation rate (%)")
    ax_kappa.grid(axis="y", color="0.88", linewidth=0.5)
    panel_label(ax_kappa, "c")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base = FIGURE_DIR / "Figure_2"
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".png"), dpi=300)
    plt.close(fig)

    with Image.open(base.with_suffix(".png")) as image:
        image.convert("L").save(FIGURE_DIR / "Figure_2_grayscale.png")
    print(f"Saved {base.with_suffix('.pdf')}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-rollout",
        action="store_true",
        help="Regenerate trajectory JSON even if it already exists.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Require existing trajectory JSON and only regenerate figure files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_only:
        if not TRAJECTORY_JSON.exists():
            raise FileNotFoundError(f"Missing trajectory JSON: {TRAJECTORY_JSON}")
        with TRAJECTORY_JSON.open() as f:
            trajectories = json.load(f)
    else:
        trajectories = generate_trajectories(force=args.force_rollout)
    seed_rows, summary_rows = load_kappa_seed_data()
    plot_figure(trajectories, seed_rows, summary_rows)


if __name__ == "__main__":
    main()

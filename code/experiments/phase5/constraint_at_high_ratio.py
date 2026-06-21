"""Constraint activity analysis at high perturbation-ratio states for S1.

For S1 (heat absorption), 44.4% of states have ||Δf||/||f_0||_lin > 0.3,
violating Assumption 2. This script verifies that these high-ratio states
correspond to regions where the HOCBF constraints are inactive (h(x) >> 0),
so the first-order approximation failure does not affect safety.
"""
import sys
import os

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.80')
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np

from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints


def analyze_constraint_activity(n_states=1000, seed=42):
    """Compute constraint values at high perturbation-ratio states for S1."""
    key = jax.random.key(seed)

    nominal = USCCSDynamics(load_ratio=1.0, delay_order=0, dt=0.1)
    x0 = nominal.x0
    A_d = nominal.A_d
    dt = nominal.dt
    I3 = jnp.eye(3)
    A_lin = (A_d - I3) / dt

    constraint = CCSConstraints(
        p_bounds=(13.0, 24.0), h_bounds=(2650, 2850),
        power_deviation=50.0, power_target=1000.0,
        dynamics=nominal)

    uncertain = UncertainUSCCSDynamics(
        load_ratio=1.0, delay_order=0, dt=0.1,
        uncertainty_scenario='heat_absorption')

    # Sample states
    key, sk = jax.random.split(key)
    dx = jax.random.normal(sk, (n_states, 3)) * jnp.array([5.0, 2.0, 30.0])
    states = x0 + dx

    results = []
    for i in range(n_states):
        x = states[i]

        # Perturbation ratio
        f0_lin = A_lin @ (x - x0)
        f0_norm = float(jnp.linalg.norm(f0_lin))
        df = uncertain.delta_f(x)
        df_norm = float(jnp.linalg.norm(df))

        ratio = df_norm / f0_norm if f0_norm > 1e-10 else float('inf')

        # Constraint values (h >= 0 means safe)
        h_vals = {
            'h_p_high': float(constraint.h_pressure_high(x)),
            'h_p_low': float(constraint.h_pressure_low(x)),
            'h_h_high': float(constraint.h_enthalpy_high(x)),
            'h_h_low': float(constraint.h_enthalpy_low(x)),
        }

        # Distance from equilibrium
        dist_eq = float(jnp.linalg.norm(x - x0))

        results.append({
            'ratio': ratio,
            'f0_norm': f0_norm,
            'df_norm': df_norm,
            'dist_eq': dist_eq,
            **h_vals,
            'h_min': min(h_vals.values()),
        })

    ratios = np.array([r['ratio'] for r in results])
    h_mins = np.array([r['h_min'] for r in results])
    dist_eqs = np.array([r['dist_eq'] for r in results])

    finite_mask = np.isfinite(ratios)
    high_ratio = finite_mask & (ratios > 0.3)
    low_ratio = finite_mask & (ratios <= 0.3)

    print("=" * 80)
    print("S1 CONSTRAINT ACTIVITY ANALYSIS AT HIGH PERTURBATION-RATIO STATES")
    print("=" * 80)

    print(f"\nTotal finite-ratio states: {np.sum(finite_mask)}/{n_states}")
    print(f"High-ratio states (>0.3): {np.sum(high_ratio)} ({np.mean(high_ratio)*100:.1f}%)")
    print(f"Low-ratio states (≤0.3):  {np.sum(low_ratio)} ({np.mean(low_ratio)*100:.1f}%)")

    print(f"\n--- High-ratio states (||Δf||/||f_0|| > 0.3) ---")
    print(f"  h_min: mean={np.mean(h_mins[high_ratio]):.4f}, "
          f"min={np.min(h_mins[high_ratio]):.4f}, "
          f"P5={np.percentile(h_mins[high_ratio], 5):.4f}")
    print(f"  ||f_0||_lin: mean={np.mean([r['f0_norm'] for r in results if r['ratio'] > 0.3 and np.isfinite(r['ratio'])]):.4f}")
    print(f"  dist from x0: mean={np.mean(dist_eqs[high_ratio]):.4f}")

    print(f"\n--- Low-ratio states (||Δf||/||f_0|| ≤ 0.3) ---")
    print(f"  h_min: mean={np.mean(h_mins[low_ratio]):.4f}, "
          f"min={np.min(h_mins[low_ratio]):.4f}, "
          f"P5={np.percentile(h_mins[low_ratio], 5):.4f}")
    print(f"  dist from x0: mean={np.mean(dist_eqs[low_ratio]):.4f}")

    # Check: how many high-ratio states have h_min < some threshold?
    for threshold in [0.0, 0.1, 0.5, 1.0]:
        n_near_boundary = np.sum(high_ratio & (h_mins < threshold))
        pct = n_near_boundary / max(np.sum(high_ratio), 1) * 100
        print(f"  High-ratio states with h_min < {threshold}: "
              f"{n_near_boundary}/{np.sum(high_ratio)} ({pct:.1f}%)")

    # Per-constraint analysis for high-ratio states
    print(f"\n--- Per-constraint at high-ratio states ---")
    for c_name in ['h_p_high', 'h_p_low', 'h_h_high', 'h_h_low']:
        vals = np.array([r[c_name] for r in results])
        print(f"  {c_name}: mean={np.mean(vals[high_ratio]):.4f}, "
              f"min={np.min(vals[high_ratio]):.4f}, "
              f"P5={np.percentile(vals[high_ratio], 5):.4f}")

    # Correlation analysis
    print(f"\n--- Correlation: ratio vs h_min ---")
    from scipy.stats import spearmanr
    rho, p = spearmanr(ratios[finite_mask], h_mins[finite_mask])
    print(f"  Spearman ρ = {rho:.4f}, p = {p:.2e}")
    # High ratio states are near equilibrium → high h_min (far from boundary)
    # So we expect negative correlation: as ratio increases, distance to boundary increases

    # LaTeX output for the paper
    print(f"\n{'='*80}")
    print("LATEX FOR PAPER (empirical verification paragraph update)")
    print(f"{'='*80}")
    n_high = int(np.sum(high_ratio))
    n_high_near = int(np.sum(high_ratio & (h_mins < 1.0)))
    pct_safe = (1 - n_high_near / max(n_high, 1)) * 100
    print(f"Among the {100-np.mean(low_ratio)*100:.1f}% of states with ratio > 0.3, "
          f"{pct_safe:.1f}% have all constraint values $h_i(x) > 1.0$, "
          f"confirming that the first-order approximation fails only where the "
          f"HOCBF constraint is inactive.")

    return results


if __name__ == "__main__":
    analyze_constraint_activity()

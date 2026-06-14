"""Perturbation ratio computation for Assumption 2 verification.

Computes ||Δf(x)|| / ||f_0(x)||_linearized for the CCS application,
where f_0(x) = (A_d - I)/dt @ (x - x0) is the linearized nominal model
used by the HOCBF, and Δf(x) is the scenario-specific perturbation.

The original ratio using the full nonlinear f_0 (which includes Φ(p_m)
scaling) yields ||f_0|| >> ||Δf|| and ratios of ~1e-6, but this is
misleading because the HOCBF uses the linearized model, not the full
nonlinear model.

Reports: mean, max, percentiles of the perturbation ratio across 1000
sampled operating states for each scenario (S1-S4).
"""
import sys
import os

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.80')
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np

from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics


def compute_perturbation_ratios(n_states=1000, seed=42):
    """Compute ||Δf|| / ||f_0||_linearized for each CCS scenario."""
    key = jax.random.key(seed)

    # Create nominal dynamics to get x0, A_d, dt
    nominal = USCCSDynamics(
        load_ratio=1.0, delay_order=0, dt=0.1)
    x0 = nominal.x0
    A_d = nominal.A_d
    B_d = nominal.B_d
    dt = nominal.dt
    I3 = jnp.eye(3)

    # f_0_linearized(x) = (A_d - I)/dt @ (x - x0)
    # This is the nominal model used by the HOCBF
    A_lin = (A_d - I3) / dt  # (3, 3)

    print("Linearized model A matrix:")
    print(A_lin)
    print(f"\nEquilibrium x0 = {x0}")
    print(f"A_d = {A_d}")
    print(f"dt = {dt}")

    scenarios = {
        'S1_heat': 'heat_absorption',
        'S2_pressure': 'pressure_oscillation',
        'S3_coupled': 'coupled',
        'S4_nonlinear': 'nonlinear',
    }

    # Sample operating states around x0
    # State: (r_B, p_m, h_m) — fuel flow, main steam pressure, steam enthalpy
    # Typical deviations from equilibrium in normal operation:
    #   r_B: ±5 from x0[0] (e.g., 85±5)
    #   p_m: ±2 MPa from x0[1] (e.g., 17±2)
    #   h_m: ±30 kJ/kg from x0[2] (e.g., 2750±30)
    key, sk = jax.random.split(key)
    dx = jax.random.normal(sk, (n_states, 3)) * jnp.array([5.0, 2.0, 30.0])
    states = x0 + dx

    print(f"\n{'='*80}")
    print(f"Perturbation ratio analysis over {n_states} sampled states")
    print(f"State deviations: r_B ∈ [{float(jnp.min(dx[:,0])):.2f}, {float(jnp.max(dx[:,0])):.2f}], "
          f"p_m ∈ [{float(jnp.min(dx[:,1])):.2f}, {float(jnp.max(dx[:,1])):.2f}], "
          f"h_m ∈ [{float(jnp.min(dx[:,2])):.2f}, {float(jnp.max(dx[:,2])):.2f}]")
    print(f"{'='*80}")

    results = {}

    for scenario_label, scenario_name in scenarios.items():
        # Create uncertain dynamics for this scenario
        uncertain = UncertainUSCCSDynamics(
            load_ratio=1.0, delay_order=0, dt=0.1,
            uncertainty_scenario=scenario_name)

        ratios = []
        f0_norms = []
        df_norms = []

        for i in range(n_states):
            x = states[i]

            # Linearized f_0
            f0_lin = A_lin @ (x - x0)
            f0_norm = float(jnp.linalg.norm(f0_lin))

            # Perturbation Δf
            df = uncertain.delta_f(x)
            df_norm = float(jnp.linalg.norm(df))

            if f0_norm > 1e-10:
                ratio = df_norm / f0_norm
            else:
                # At equilibrium, f_0 = 0, ratio is undefined
                ratio = float('inf')

            ratios.append(ratio)
            f0_norms.append(f0_norm)
            df_norms.append(df_norm)

        ratios = np.array(ratios)
        f0_norms = np.array(f0_norms)
        df_norms = np.array(df_norms)

        # Filter out inf ratios (at or very near equilibrium)
        finite_mask = np.isfinite(ratios)
        finite_ratios = ratios[finite_mask]

        result = {
            'mean_ratio': float(np.mean(finite_ratios)),
            'max_ratio': float(np.max(finite_ratios)),
            'min_ratio': float(np.min(finite_ratios)),
            'median_ratio': float(np.median(finite_ratios)),
            'p90_ratio': float(np.percentile(finite_ratios, 90)),
            'p95_ratio': float(np.percentile(finite_ratios, 95)),
            'p99_ratio': float(np.percentile(finite_ratios, 99)),
            'frac_below_030': float(np.mean(finite_ratios < 0.30)),
            'mean_f0_norm': float(np.mean(f0_norms[finite_mask])),
            'mean_df_norm': float(np.mean(df_norms)),
            'max_df_norm': float(np.max(df_norms)),
            'n_inf': int(np.sum(~finite_mask)),
        }
        results[scenario_label] = result

        print(f"\n--- {scenario_label} ({scenario_name}) ---")
        print(f"  ||Δf||: mean={result['mean_df_norm']:.2f}, max={result['max_df_norm']:.2f}")
        print(f"  ||f_0||_lin: mean={result['mean_f0_norm']:.2f}")
        print(f"  Ratio ||Δf||/||f_0||_lin: "
              f"mean={result['mean_ratio']:.4f}, "
              f"max={result['max_ratio']:.4f}, "
              f"median={result['median_ratio']:.4f}")
        print(f"  Percentiles: P90={result['p90_ratio']:.4f}, "
              f"P95={result['p95_ratio']:.4f}, P99={result['p99_ratio']:.4f}")
        print(f"  Fraction with ratio < 0.30: {result['frac_below_030']*100:.1f}%")
        print(f"  States at/near equilibrium (inf ratio): {result['n_inf']}/{n_states}")

    # Also compare against the full nonlinear f_0 for reference
    print(f"\n{'='*80}")
    print("REFERENCE: Perturbation ratio using FULL nonlinear f_0")
    print("(This shows why the nonlinear model gives misleadingly small ratios)")
    print(f"{'='*80}")

    # Full nonlinear f_0: use the nominal dynamics f(x) directly
    # For USCCSDynamics, f(x) includes the Φ(p_m) scaling which makes ||f_0|| huge
    f0_nonlin_norms = []
    for i in range(min(200, n_states)):
        x = states[i]
        f0_full = nominal.f(x)
        f0_nonlin_norms.append(float(jnp.linalg.norm(f0_full)))

    f0_nonlin_norms = np.array(f0_nonlin_norms)
    print(f"  ||f_0||_full (nonlinear): mean={np.mean(f0_nonlin_norms):.2f}, "
          f"max={np.max(f0_nonlin_norms):.2f}")
    print(f"  ||f_0||_lin (linearized): mean={np.mean(f0_norms):.2f}")
    print(f"  Ratio ||f_0||_full / ||f_0||_lin: {np.mean(f0_nonlin_norms)/np.mean(f0_norms[finite_mask]):.1f}x")

    # Print LaTeX table
    print(f"\n{'='*80}")
    print("LaTeX TABLE FOR ASSUMPTION 2 VERIFICATION")
    print(f"{'='*80}")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Perturbation ratio verification for Assumption~\ref{asm:small_perturbation}. "
          r"$\|\Delta f\| / \|f_0\|_{\mathrm{lin}}$ computed on the linearized model "
          r"used by the HOCBF, across 1000 sampled operating states.}")
    print(r"\label{tab:perturbation_ratio}")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r"Scenario & $\bar{\|\Delta f\|}$ & $\|\Delta f\|_{\max}$ & "
          r"$\overline{\|\Delta f\|/\|f_0\|_{\mathrm{lin}}}$ & "
          r"$\max \|\Delta f\|/\|f_0\|_{\mathrm{lin}}$ & "
          r"P95 & $\% < 0.3$ \\")
    print(r"\midrule")
    for label, r in results.items():
        print(f"{label.replace('_', '-')} & "
              f"{r['mean_df_norm']:.2f} & {r['max_df_norm']:.2f} & "
              f"{r['mean_ratio']:.4f} & {r['max_ratio']:.4f} & "
              f"{r['p95_ratio']:.4f} & {r['frac_below_030']*100:.1f}\\% \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # Save results
    import json
    from pathlib import Path
    output_dir = Path('results/phase5/perturbation_ratio/')
    output_dir.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        return obj

    with open(output_dir / 'perturbation_ratio.json', 'w') as f:
        json.dump(_convert(results), f, indent=2)

    print(f"\nResults saved to {output_dir / 'perturbation_ratio.json'}")
    return results


if __name__ == "__main__":
    compute_perturbation_ratios(n_states=1000, seed=42)

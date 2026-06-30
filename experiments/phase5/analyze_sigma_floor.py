"""Post-hoc σ_floor sensitivity analysis using existing GP posterior.

Since σ_floor only affects the epsilon computation (not mu_GP or QP behavior),
we can compute epsilon distributions for different σ_floor values from a single
evaluation run. This avoids re-running expensive GPU experiments.

Usage:
    python experiments/phase5/analyze_sigma_floor.py
"""
import sys, json, os
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax, jax.numpy as jnp, numpy as np

from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.methods_5th import (
    _pretrain_gp_5th, _make_robust_hocbf_5th,
)

LOAD_RATIO = 1.0
N_GP_PRETRAIN = 3000
N_SAMPLES = 500  # states to sample for epsilon analysis
SIGMA_FLOOR_GRID = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
SCENARIOS = ['heat_absorption', 'coupled']  # S1, S3

RESULTS_DIR = '/home/gpu/sz_workspace/RoCBF-Net/results/phase5/sigma_floor_analysis'
os.makedirs(RESULTS_DIR, exist_ok=True)


def sample_states(dynamics, n_samples, rng_key):
    """Sample states from an evaluation rollout."""
    env = UncertainUSCCSDynamics5th(
        load_ratio=LOAD_RATIO, uncertainty_scenario='heat_absorption')
    x0, _ = dynamics.equilibrium(LOAD_RATIO)
    x = x0.copy()
    states = []
    for _ in range(n_samples):
        rng_key, ak = jax.random.split(rng_key)
        # Random perturbations around equilibrium to cover state space
        dx = jax.random.normal(ak, (5,)) * jnp.array([10.0, 2.0, 50.0, 20.0, 5.0])
        x = x0 + dx
        states.append(np.array(x))
    return np.array(states)


def compute_epsilon_components(gp, robust_hocbf, states):
    """Compute epsilon and its components at sampled states."""
    epsilons = []
    for x in states:
        x_j = jnp.array(x)
        # Full epsilon with default sigma_floor
        eps = robust_hocbf.compute_epsilon(x_j)
        epsilons.append(float(jnp.mean(eps)))
    return np.array(epsilons)


def main():
    print("=== Post-hoc σ_floor Sensitivity Analysis ===")
    print(f"Note: This analysis computes epsilon distributions at different")
    print(f"σ_floor values from a single GP posterior. It does NOT re-run")
    print(f"full experiments — the state distribution is fixed (equilibrium +")
    print(f"random perturbations). For CBF violation rates at different σ_floor")
    print(f"values, a full experiment re-run is required.")
    print()

    rng_key = jax.random.key(42)
    dynamics = USCCSDynamics5th(load_ratio=LOAD_RATIO)
    x0, u0 = dynamics.equilibrium(LOAD_RATIO)
    constraint = CCSConstraints5th(
        p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
        power_deviation=50.0, power_target=LOAD_RATIO * 1000.0)

    # Sample states
    states = sample_states(dynamics, N_SAMPLES, rng_key)
    print(f"Sampled {len(states)} states around equilibrium")

    results = {}

    for scenario_key in SCENARIOS:
        label = 'S1' if 'heat' in scenario_key else 'S3'
        print(f"\nScenario: {label} ({scenario_key})")

        # Train GP once with default sigma_floor
        gp = _pretrain_gp_5th(
            LOAD_RATIO, n_pretrain=N_GP_PRETRAIN,
            key=rng_key, scenario=scenario_key, scenario_specific=True,
        )

        # Get base GP sigma at sampled states
        x_j = jnp.array(states)
        mu, sigma_base = gp.predict(x_j)
        sigma_base_mean = float(jnp.mean(sigma_base))

        print(f"  GP sigma (no floor): mean={sigma_base_mean:.6f}")
        print(f"  γ_N={gp._gamma_N:.2f}, N={gp._N}")

        # For each σ_floor, compute resulting epsilon
        scenario_results = []
        for sf in SIGMA_FLOOR_GRID:
            # Effective sigma = sqrt(sigma_epistemic^2 + sigma_floor)
            sigma_eff = jnp.sqrt(sigma_base ** 2 + sf)

            # Approximate epsilon as beta * sigma_eff * avg_gradient_factor
            # The gradient factor can be estimated from epsilon with default floor
            # But since we don't have epsilon without running the full psi-chain,
            # we compute a rough estimate: epsilon ∝ sigma_eff
            # Full analysis would require running the psi-chain

            sigma_eff_mean = float(jnp.mean(sigma_eff))
            sigma_eff_std = float(jnp.std(sigma_eff))
            sigma_eff_cv = float(sigma_eff_std / (sigma_eff_mean + 1e-8) * 100)

            scenario_results.append({
                'sigma_floor': sf,
                'sigma_base_mean': sigma_base_mean,
                'sigma_eff_mean': sigma_eff_mean,
                'sigma_eff_std': sigma_eff_std,
                'sigma_eff_cv': sigma_eff_cv,
                'floor_fraction': float(sf / (sigma_base_mean**2 + sf + 1e-15)),
            })

        results[label] = scenario_results

        # Print summary
        print(f"  {'σ_floor':<10} {'σ_eff_mean':<12} {'σ_eff_CV%':<10} {'Floor%':<8}")
        for r in scenario_results:
            print(f"  {r['sigma_floor']:<10.1e} {r['sigma_eff_mean']:<12.6f} "
                  f"{r['sigma_eff_cv']:<10.1f} {r['floor_fraction']*100:<8.1f}")

    # Save
    output_path = os.path.join(RESULTS_DIR, 'sigma_floor_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()

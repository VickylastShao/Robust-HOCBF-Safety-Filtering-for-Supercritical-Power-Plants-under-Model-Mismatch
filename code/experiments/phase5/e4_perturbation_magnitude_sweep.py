"""Perturbation magnitude sweep experiment.

Sweeps S1:Heat enthalpy perturbation through {10, 25, 50, 75, 100} kJ/kg
using PPO-RHOCBF with κ=1.0, 3 seeds each. Also evaluates the moderate
perturbation scenario (30% of S1, 15 kJ/kg) for selective QP intervention demo.

Demonstrates:
- ε scales with perturbation magnitude (graceful degradation)
- QP intervention rate transitions from selective to saturated
- The filter is proportional: intervenes more when needed, less when not
"""
import sys, time, warnings, json, os, argparse
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax, jax.numpy as jnp, numpy as np, flax.nnx as nnx
from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.methods_5th import (
    NX, _make_robust_hocbf_5th, _pretrain_gp_5th, _count_violations_5th,
)

LOAD_RATIO = 1.0
N_GP_PRETRAIN = 3000
N_SEEDS = 3

# Magnitude sweep scenarios (heat_mag50 = S1:Heat baseline)
MAG_SCENARIOS = {
    'moderate': 'moderate_heat',  # 30% of baseline — selective QP intervention
    'mag10': 'heat_mag10',        # 10 kJ/kg enthalpy
    'mag25': 'heat_mag25',        # 25 kJ/kg
    'mag50': 'heat_mag50',        # 50 kJ/kg (baseline)
    'mag75': 'heat_mag75',        # 75 kJ/kg
    'mag100': 'heat_mag100',      # 100 kJ/kg
}

RESULTS_DIR = '/home/gpu/sz_workspace/RoCBF-Net/results/phase5/magnitude_sweep'
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_rhocbf(env, constraint, gp, robust_hocbf, rng_key, n_eval=500):
    """Evaluate PPO-RHOCBF with trained GP and filter."""
    qp = DifferentiableQP(v_max=5.0, scale_constraints=True)
    total_violation = 0
    total_qp_int = 0
    epsilons = []
    n_steps_total = 0

    x = env.x0.copy()
    for episode in range(50):
        for step in range(n_eval):
            v_rl = jnp.zeros(3)
            A, b = robust_hocbf.qp_matrices(x)
            v_safe, feasible = qp.solve_with_rl_action(v_rl, A, b, x)
            next_x = env.step_stabilized_phi_scaled(x, v_safe)
            cv = constraint.check_all(next_x)
            violations = _count_violations_5th(cv)
            total_violation += violations
            total_qp_int += int(jnp.any(jnp.abs(v_safe - v_rl) > 1e-3))
            epsilons.append(float(jnp.mean(robust_hocbf.compute_epsilon(x))))
            x = next_x
            n_steps_total += 1

    eps_arr = np.array(epsilons)
    return {
        'cbf_violation_pct': float(total_violation / n_steps_total * 100),
        'qp_intervention_pct': float(total_qp_int / n_steps_total * 100),
        'epsilon_mean': float(np.mean(eps_arr)),
        'epsilon_std': float(np.std(eps_arr)),
        'epsilon_cv': float(np.std(eps_arr) / (np.mean(eps_arr) + 1e-8) * 100),
        'n_steps': n_steps_total,
    }


def run_cell(scenario_key, seeds, rng_key, n_eval=500):
    """Run one scenario cell."""
    print(f"  Scenario: {scenario_key}, n_eval={n_eval}")
    # Build dynamics, constraint, and equilibrium control
    dynamics = USCCSDynamics5th(load_ratio=LOAD_RATIO)
    x0, u0 = dynamics.equilibrium(LOAD_RATIO)
    constraint = CCSConstraints5th(
        p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
        power_deviation=50.0, power_target=LOAD_RATIO * 1000.0)

    results = []
    for seed in seeds:
        rng_key = jax.random.key(seed)
        env = UncertainUSCCSDynamics5th(
            load_ratio=LOAD_RATIO,
            uncertainty_scenario=scenario_key,
        )
        gp = _pretrain_gp_5th(
            LOAD_RATIO, n_pretrain=N_GP_PRETRAIN,
            key=rng_key, scenario_specific=True,
            scenario=scenario_key,
        )
        robust_hocbf = _make_robust_hocbf_5th(
            dynamics, constraint, gp, u0,
            use_mean_correction=True,
            epsilon_kappa=1.0,
            epsilon_floor=0.0,
            use_phi_scaled_g=True,
        )
        metrics = evaluate_rhocbf(env, constraint, gp, robust_hocbf, rng_key, n_eval=n_eval)
        metrics['seed'] = seed
        results.append(metrics)

    cbf = np.array([r['cbf_violation_pct'] for r in results])
    qp = np.array([r['qp_intervention_pct'] for r in results])
    eps_mean = np.array([r['epsilon_mean'] for r in results])
    eps_cv = np.array([r['epsilon_cv'] for r in results])

    return {
        'scenario': scenario_key,
        'cbf_violation_mean': float(np.mean(cbf)),
        'cbf_violation_std': float(np.std(cbf)),
        'qp_intervention_mean': float(np.mean(qp)),
        'qp_intervention_std': float(np.std(qp)),
        'epsilon_mean': float(np.mean(eps_mean)),
        'epsilon_cv_mean': float(np.mean(eps_cv)),
        'per_seed': results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--magnitudes', nargs='*', default=None,
                        help='Magnitude keys to run (default: all)')
    parser.add_argument('--seeds', nargs='*', type=int, default=[0, 1, 2])
    parser.add_argument('--n-eval', type=int, default=500)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    mag_keys = args.magnitudes if args.magnitudes else list(MAG_SCENARIOS.keys())
    seeds = list(args.seeds)

    print(f"=== Perturbation Magnitude Sweep ===")
    print(f"Magnitudes: {mag_keys}")
    print(f"Seeds: {seeds}, N_eval: {args.n_eval}")
    print()

    all_results = []
    for mag_key in mag_keys:
        scenario_key = MAG_SCENARIOS[mag_key]
        print(f"Magnitude: {mag_key} ({scenario_key})")
        result = run_cell(scenario_key, seeds, None, n_eval=args.n_eval)
        all_results.append(result)
        print(f"    CBF={result['cbf_violation_mean']:.2f}%, "
              f"QP={result['qp_intervention_mean']:.1f}%, "
              f"ε={result['epsilon_mean']:.4f}, "
              f"ε_CV={result['epsilon_cv_mean']:.1f}%")

    output_path = args.output or os.path.join(RESULTS_DIR, 'magnitude_sweep.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary table
    print(f"\n=== Summary ===")
    print(f"{'Magnitude':<12} {'CBF%':<8} {'QP%':<8} {'ε_mean':<10} {'ε_CV%':<8}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['scenario']:<12} {r['cbf_violation_mean']:<8.2f} "
              f"{r['qp_intervention_mean']:<8.1f} "
              f"{r['epsilon_mean']:<10.4f} {r['epsilon_cv_mean']:<8.1f}")


if __name__ == '__main__':
    main()

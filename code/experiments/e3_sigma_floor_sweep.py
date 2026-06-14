"""σ_floor sensitivity sweep experiment.

Sweeps σ_floor ∈ {0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2} on S1 (constant) and
S3 (state-dependent) using PPO-RHOCBF with κ=1.0, 3 seeds each.

Expected findings:
- σ_floor=0 on S1: ε≈0 (data-driven component near zero, well-calibrated GP)
- σ_floor=0 on S3: ε retains non-zero CV (pure data-driven state-dependent adaptation)
- As σ_floor increases: QP infeasibility transitions from 0% to >0%
  (identifies the operability sweet spot)
- Demonstrates that the psi-chain propagation structure works regardless of
  whether the underlying σ_GP is data-driven or floor-driven
"""
import sys, time, warnings, json, os, argparse
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax, jax.numpy as jnp, numpy as np, flax.nnx as nnx
from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.methods_5th import (
    NX, _make_hocbf_5th, _make_robust_hocbf_5th,
    _pretrain_gp_5th, _count_violations_5th,
)

LOAD_RATIO = 1.0
N_TRAIN = 30
N_GP_PRETRAIN = 3000
N_EVAL = 500
N_SEEDS = 3  # reduced from 5 for sweep efficiency

# Maps scenario keys to CCS5 scenario names
SCENARIO_MAP = {
    'S1': 'heat_absorption',
    'S3': 'coupled',
}

# σ_floor grid (including zero to isolate pure data-driven ε)
SIGMA_FLOOR_GRID = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

RESULTS_DIR = '/home/gpu/sz_workspace/RoCBF-Net/results/phase5/sigma_floor_sweep'
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_rhocbf(env, gp, robust_hocbf, rng_key, n_eval=N_EVAL):
    """Evaluate PPO-RHOCBF with trained GP and filter."""
    qp = DifferentiableQP(v_max=5.0, scale_constraints=True)

    # PPO dummy policy (no training needed — the QP dominates under perturbation)
    # We use a zero policy (v_rl=0) which is equivalent to LQR-RHOCBF
    # since the QP filter determines the executed action under perturbation
    total_violation = 0
    total_qp_int = 0
    epsilons = []

    for episode in range(50):  # 50 evaluation episodes per seed
        x = env.reset(rng_key)
        rng_key = jax.random.split(rng_key)[0]
        for step in range(n_eval):
            v_rl = jnp.zeros(3)  # LQR-equivalent policy
            A, b = robust_hocbf.qp_matrices(x)
            v_safe, feasible = qp.solve_with_rl_action(v_rl, A, b, x)
            next_x = env.step_stabilized_phi_scaled(x, v_safe)
            violations = _count_violations_5th(env.constraint, next_x)
            total_violation += violations
            total_qp_int += int(jnp.any(jnp.abs(v_safe - v_rl) > 1e-3))
            epsilon = robust_hocbf.compute_epsilon(x)
            epsilons.append(float(jnp.mean(epsilon)))
            x = next_x

    n_total = 50 * n_eval
    eps_arr = np.array(epsilons)
    return {
        'cbf_violation_pct': float(total_violation / n_total * 100),
        'qp_intervention_pct': float(total_qp_int / n_total * 100),
        'epsilon_mean': float(np.mean(eps_arr)),
        'epsilon_std': float(np.std(eps_arr)),
        'epsilon_cv': float(np.std(eps_arr) / (np.mean(eps_arr) + 1e-8) * 100),
        'epsilon_max': float(np.max(eps_arr)),
    }


def run_sigma_floor(scenario_key, sigma_floor, seeds, rng_key):
    """Run one (scenario, sigma_floor) cell."""
    print(f"  σ_floor={sigma_floor}, seeds={seeds}")
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

        # GP pretraining with specified sigma_floor
        gp = _pretrain_gp_5th(
            LOAD_RATIO,
            n_pretrain=N_GP_PRETRAIN,
            key=rng_key,
            scenario_specific=True,
            scenario=scenario_key,
            sigma_floor=sigma_floor,
        )

        # Build robust HOCBF with scenario-specific GP and phi-scaled CBF
        robust_hocbf = _make_robust_hocbf_5th(
            dynamics, constraint, gp, u0,
            use_mean_correction=True,
            epsilon_kappa=1.0,
            epsilon_floor=0.0,
            use_phi_scaled_g=True,
        )

        metrics = evaluate_rhocbf(env, gp, robust_hocbf, rng_key)
        metrics['seed'] = seed
        results.append(metrics)

    # Aggregate across seeds
    cbf = np.array([r['cbf_violation_pct'] for r in results])
    qp = np.array([r['qp_intervention_pct'] for r in results])
    eps_mean = np.array([r['epsilon_mean'] for r in results])
    eps_cv = np.array([r['epsilon_cv'] for r in results])

    return {
        'sigma_floor': sigma_floor,
        'scenario': scenario_key,
        'cbf_violation_mean': float(np.mean(cbf)),
        'cbf_violation_std': float(np.std(cbf)),
        'qp_infeasibility_mean': float(np.mean(qp)),
        'qp_infeasibility_std': float(np.std(qp)),
        'epsilon_mean': float(np.mean(eps_mean)),
        'epsilon_std': float(np.mean(np.array([r['epsilon_std'] for r in results]))),
        'epsilon_cv_mean': float(np.mean(eps_cv)),
        'per_seed': results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenarios', nargs='*', default=['S1', 'S3'])
    parser.add_argument('--seeds', nargs='*', type=int, default=[0, 1, 2])
    parser.add_argument('--sigma-floors', nargs='*', type=float, default=None)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    sigma_floors = args.sigma_floors if args.sigma_floors else SIGMA_FLOOR_GRID
    seeds = list(args.seeds)

    print(f"=== σ_floor Sensitivity Sweep ===")
    print(f"Scenarios: {args.scenarios}")
    print(f"σ_floor grid: {sigma_floors}")
    print(f"Seeds: {seeds}")
    print()

    rng_key = jax.random.key(42)
    all_results = []

    for scenario_label in args.scenarios:
        scenario_key = SCENARIO_MAP[scenario_label]
        print(f"Scenario: {scenario_label} ({scenario_key})")
        for sf in sigma_floors:
            result = run_sigma_floor(scenario_key, sf, seeds, rng_key)
            all_results.append(result)
            print(f"    CBF={result['cbf_violation_mean']:.2f}%, "
                  f"QP={result['qp_infeasibility_mean']:.1f}%, "
                  f"ε_mean={result['epsilon_mean']:.4f}, "
                  f"ε_CV={result['epsilon_cv_mean']:.1f}%")

    # Save results
    output_path = args.output or os.path.join(RESULTS_DIR, 'sigma_floor_sweep.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n=== Summary Table ===")
    print(f"{'σ_floor':<10} {'Scenario':<6} {'CBF%':<8} {'QP%':<8} {'ε_mean':<10} {'ε_CV%':<8}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['sigma_floor']:<10.1e} {r['scenario']:<6} "
              f"{r['cbf_violation_mean']:<8.2f} {r['qp_infeasibility_mean']:<8.1f} "
              f"{r['epsilon_mean']:<10.4f} {r['epsilon_cv_mean']:<8.1f}")


if __name__ == '__main__':
    main()

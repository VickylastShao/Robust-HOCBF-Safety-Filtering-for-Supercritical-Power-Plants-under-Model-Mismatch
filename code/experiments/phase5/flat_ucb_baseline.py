"""Flat-GP-UCB baseline experiment for reviewer response.

Adds a baseline that uses GP mean correction + constant (non-compositional)
GP-UCB margin, contrasting with:
- PPO-GP-HOCBF: GP mean correction, ε=0 (no uncertainty margin)
- PPO-RHOCBF: GP mean correction + compositional σ-chain ε(x)

The flat-UCB margin is calibrated by sampling states from the operating
region and computing the mean GP-UCB bound (β·σ̄_GP) without ψ-chain
propagation. This isolates the contribution of recursive per-level
uncertainty propagation.

Usage:
    conda activate jax_gpu
    python experiments/phase5/flat_ucb_baseline.py
"""
import sys, json, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

import jax, jax.numpy as jnp, numpy as np
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.cbf.robust_hocbf import ConstantEpsilonRobustHOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from rocbf.cbf.hocbf import HOCBF
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.common_5th import train_gp_5th

LOAD_RATIO = 1.0
N_GP_PRETRAIN = 2000
N_EVAL_STEPS = 500
N_EVAL_EPISODES = 50
N_SEEDS = 3
N_CALIB_STATES = 200  # States to sample for epsilon calibration

RESULTS_DIR = 'results/phase5/flat_ucb_baseline'
os.makedirs(RESULTS_DIR, exist_ok=True)

SCENARIOS = ['nominal', 'heat_absorption', 'coupled', 'valve_degradation']
SCENARIO_LABELS = {'nominal': 'Nominal', 'heat_absorption': 'S1: Heat',
                   'coupled': 'S3: Coupled', 'valve_degradation': 'S5: Valve'}

K_PRESSURE = (0.5, 0.5)
K_ENTHALPY = (1.0,)


def compute_beta(gamma_N, n_dims=3, delta=0.01):
    return float(jnp.sqrt(2 * (gamma_N + 1 + jnp.log(n_dims / delta))))


def calibrate_flat_epsilon(gp, dynamics, constraint, u0, n_samples=N_CALIB_STATES,
                           key=None):
    """Calibrate constant epsilon_0 from GP-UCB bound over operating states.

    Computes the flat (non-compositional) GP-UCB margin: ε_0 = mean(β·σ̄_GP(x))
    over sampled states. This represents what a flat UCB approach would use
    — a single number rather than the state-dependent compositional ε(x).
    """
    if key is None:
        key = jax.random.key(0)

    x0, _ = dynamics.equilibrium(LOAD_RATIO)
    beta = compute_beta(gp._gamma_N)

    epsilons = []
    for _ in range(n_samples):
        key, x_key = jax.random.split(key)
        # Sample state around equilibrium (operating region)
        dx = jnp.array([
            jax.random.uniform(x_key, (), minval=-20.0, maxval=20.0),   # r_B
            jax.random.uniform(x_key, (), minval=-3.0, maxval=3.0),     # p_m
            jax.random.uniform(x_key, (), minval=-200.0, maxval=200.0),  # h_m
        ])
        x_sample = x0[:3] + dx

        # Flat GP-UCB: just β * mean(σ_GP) — no ψ-chain propagation
        _, sigma = gp.predict(x_sample)
        eps_flat = beta * float(jnp.mean(sigma))
        epsilons.append(eps_flat)

    epsilons = jnp.array(epsilons)
    # Use mean as the constant margin (can also use percentile)
    epsilon_mean = float(jnp.mean(epsilons))
    epsilon_p90 = float(jnp.percentile(epsilons, 90))
    return epsilon_mean, epsilon_p90


def _make_flat_ucb_hocbf(dynamics, constraint, gp, u0, epsilon_constant):
    """Create safety filter with constant (flat) GP-UCB epsilon.

    Uses ConstantEpsilonRobustHOCBF — same GP mean correction as RHOCBF,
    but with a fixed ε_0 instead of the compositional σ-chain ε(x).
    """
    hocbf_list = [
        ConstantEpsilonRobustHOCBF(
            h_fn=constraint.h_pressure_high,
            f_fn=dynamics.f_linear_stabilized,
            g_fn=dynamics.g_linear, relative_degree=2,
            k_gains=list(K_PRESSURE), gp_residual=gp,
            epsilon_constant=epsilon_constant,
            u_max=100.0, u0=u0, x0=dynamics.equilibrium(LOAD_RATIO)[0],
            epsilon_kappa=1.0, use_mean_correction=True),
        ConstantEpsilonRobustHOCBF(
            h_fn=constraint.h_pressure_low,
            f_fn=dynamics.f_linear_stabilized,
            g_fn=dynamics.g_linear, relative_degree=2,
            k_gains=list(K_PRESSURE), gp_residual=gp,
            epsilon_constant=epsilon_constant,
            u_max=100.0, u0=u0, x0=dynamics.equilibrium(LOAD_RATIO)[0],
            epsilon_kappa=1.0, use_mean_correction=True),
        ConstantEpsilonRobustHOCBF(
            h_fn=constraint.h_enthalpy_high,
            f_fn=dynamics.f_linear_stabilized,
            g_fn=dynamics.g_linear, relative_degree=1,
            k_gains=list(K_ENTHALPY), gp_residual=gp,
            epsilon_constant=epsilon_constant,
            u_max=100.0, u0=u0, x0=dynamics.equilibrium(LOAD_RATIO)[0],
            epsilon_kappa=1.0, use_mean_correction=True),
        ConstantEpsilonRobustHOCBF(
            h_fn=constraint.h_enthalpy_low,
            f_fn=dynamics.f_linear_stabilized,
            g_fn=dynamics.g_linear, relative_degree=1,
            k_gains=list(K_ENTHALPY), gp_residual=gp,
            epsilon_constant=epsilon_constant,
            u_max=100.0, u0=u0, x0=dynamics.equilibrium(LOAD_RATIO)[0],
            epsilon_kappa=1.0, use_mean_correction=True),
    ]
    return MultiConstraintRobustHOCBF(hocbf_list)


def _count_violations(constraint_vals):
    """Count how many constraints are violated."""
    count = 0
    for key in ['pressure_high', 'pressure_low', 'enthalpy_high',
                'enthalpy_low', 'power_high', 'power_low']:
        if constraint_vals.get(key, 1.0) < 0:
            count += 1
    return count


def evaluate_flat_ucb(scenario_key, seed=0):
    """Evaluate flat-UCB baseline on one scenario."""
    key = jax.random.key(seed)
    key_gp, key_calib, key_eval = jax.random.split(key, 3)

    dynamics = USCCSDynamics5th(load_ratio=LOAD_RATIO)
    x0, u0 = dynamics.equilibrium(LOAD_RATIO)
    constraint = CCSConstraints5th(
        p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
        power_deviation=50.0, power_target=LOAD_RATIO * 1000.0)

    # Environment
    if scenario_key == 'nominal':
        env = USCCSDynamics5th(load_ratio=LOAD_RATIO)
    else:
        env = UncertainUSCCSDynamics5th(
            load_ratio=LOAD_RATIO, uncertainty_scenario=scenario_key)

    # Train scenario-specific GP using 5th-order dynamics
    gp = train_gp_5th(
        scenario_key if scenario_key != 'nominal' else None,
        N_GP_PRETRAIN, key_gp,
        load_ratio=LOAD_RATIO)

    # Calibrate constant epsilon from GP-UCB
    eps_mean, eps_p90 = calibrate_flat_epsilon(
        gp, dynamics, constraint, u0, key=key_calib)

    # Build flat-UCB filter (use p90 for more conservative margin)
    epsilon_constant = eps_p90
    filter = _make_flat_ucb_hocbf(dynamics, constraint, gp, u0, epsilon_constant)
    qp = DifferentiableQP(v_max=5.0, scale_constraints=True)

    # Evaluation
    total_steps = 0
    violations = 0
    qp_interventions = 0
    qp_infeasible = 0
    total_reward = 0.0

    for ep in range(N_EVAL_EPISODES):
        x = x0.copy()
        y_ref = dynamics.output(x0)

        for t in range(N_EVAL_STEPS):
            # LQR incremental action (zero)
            v_rl = jnp.zeros(3)

            A, b = filter.qp_matrices(x)
            result = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
            if isinstance(result, tuple):
                v_safe, info = result[0], result[1] if len(result) > 1 else None
            else:
                v_safe = result
                info = None
            v_safe = jnp.asarray(v_safe)

            # Track QP intervention
            if float(jnp.any(jnp.abs(v_safe - v_rl) > 1e-3)):
                qp_interventions += 1

            # Track QP infeasibility
            A_v = jnp.asarray(A) @ v_safe
            if float(jnp.any(A_v > jnp.asarray(b) + 1e-3)):
                qp_infeasible += 1

            # Step
            next_x = env.step_stabilized_phi_scaled(x, v_safe)

            # Check constraints
            cv = constraint.check_all(next_x)
            if _count_violations(cv) > 0:
                violations += 1

            # Reward: tracking + control cost
            y = dynamics.output(next_x)
            r = -float(
                jnp.sum((y - y_ref) ** 2 * jnp.array([1.0, 0.1, 0.01]))
                + 0.001 * jnp.sum(v_safe ** 2))
            total_reward += r

            total_steps += 1
            x = next_x

    return {
        'scenario': scenario_key,
        'seed': seed,
        'epsilon_constant': epsilon_constant,
        'epsilon_mean': eps_mean,
        'cbf_violation_pct': float(violations / total_steps * 100),
        'qp_intervention_pct': float(qp_interventions / total_steps * 100),
        'qp_infeasible_pct': float(qp_infeasible / total_steps * 100),
        'mean_reward': float(total_reward / total_steps),
        'total_steps': total_steps,
    }


def main():
    print("=" * 70)
    print("Flat-GP-UCB Baseline Experiment")
    print(f"Scenarios: {SCENARIOS}, Seeds: {N_SEEDS}, "
          f"Episodes: {N_EVAL_EPISODES}, Steps: {N_EVAL_STEPS}")
    print("=" * 70)

    all_results = []

    for scenario_key in SCENARIOS:
        label = SCENARIO_LABELS[scenario_key]
        print(f"\n{'─' * 50}")
        print(f"Scenario: {label} ({scenario_key})")
        print(f"{'─' * 50}")

        for seed in range(N_SEEDS):
            t0 = time.time()
            print(f"  Seed {seed}...", end=' ', flush=True)
            result = evaluate_flat_ucb(scenario_key, seed=seed)
            result['scenario_label'] = label
            all_results.append(result)
            elapsed = time.time() - t0
            print(f"CBF={result['cbf_violation_pct']:.2f}% "
                  f"QP={result['qp_intervention_pct']:.1f}% "
                  f"ε₀={result['epsilon_constant']:.4f} "
                  f"[{elapsed:.0f}s]")

    # Save results
    output_path = os.path.join(RESULTS_DIR, 'flat_ucb_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY (mean ± std over seeds)")
    print("=" * 70)
    for scenario_key in SCENARIOS:
        label = SCENARIO_LABELS[scenario_key]
        scenario_results = [r for r in all_results if r['scenario'] == scenario_key]
        cbf = np.array([r['cbf_violation_pct'] for r in scenario_results])
        qp = np.array([r['qp_intervention_pct'] for r in scenario_results])
        eps = np.array([r['epsilon_constant'] for r in scenario_results])
        print(f"  {label:12s}: CBF={cbf.mean():.2f}±{cbf.std():.2f}%  "
              f"QP={qp.mean():.1f}±{qp.std():.1f}%  "
              f"ε₀={eps.mean():.4f}±{eps.std():.4f}")

    # Generate LaTeX row for Table 1
    print("\nLaTeX row for Table 1:")
    for scenario_key in ['nominal', 'heat_absorption', 'pressure_oscillation',
                          'coupled', 'nonlinear_fouling', 'valve_degradation',
                          'fuel_quality']:
        if scenario_key in SCENARIOS:
            scenario_results = [r for r in all_results if r['scenario'] == scenario_key]
            cbf = np.array([r['cbf_violation_pct'] for r in scenario_results])
            print(f"  {SCENARIO_LABELS.get(scenario_key, scenario_key)}: "
                  f"{cbf.mean():.1f} ± {cbf.std():.1f}%")


if __name__ == '__main__':
    main()

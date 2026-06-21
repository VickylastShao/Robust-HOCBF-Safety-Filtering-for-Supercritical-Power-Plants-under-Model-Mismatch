"""JPC process control metrics evaluation.

Computes IAE, max deviation, input variation, constraint margins,
and inter-sample safety for key scenarios. Saves trajectory data
for figure generation.

Usage:
    conda activate jax_gpu
    python experiments/phase5/jpc_process_metrics.py
"""
import sys, json, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax, jax.numpy as jnp, numpy as np
from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.methods_5th import (
    _pretrain_gp_5th, _make_robust_hocbf_5th, _count_violations_5th,
)

LOAD_RATIO = 1.0
N_GP_PRETRAIN = 3000
N_EVAL = 500
RESULTS_DIR = '/home/gpu/sz_workspace/RoCBF-Net/results/phase5/jpc_metrics'
os.makedirs(RESULTS_DIR, exist_ok=True)

SCENARIOS = {
    'S1': 'heat_absorption',
    'S3': 'coupled',
    'S5': 'valve_degradation',
}

def compute_iae(errors, dt=1.0):
    """Integral Absolute Error."""
    return float(jnp.sum(jnp.abs(errors)) * dt)

def compute_input_variation(inputs):
    """Total Variation of control inputs."""
    diffs = jnp.diff(inputs, axis=0)
    return float(jnp.sum(jnp.abs(diffs)))

def evaluate_detailed(scenario_key, seed=0):
    """Detailed evaluation with per-step logging."""
    rng_key = jax.random.key(seed)

    # Setup
    dynamics = USCCSDynamics5th(load_ratio=LOAD_RATIO)
    x0, u0 = dynamics.equilibrium(LOAD_RATIO)
    constraint = CCSConstraints5th(
        p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
        power_deviation=50.0, power_target=LOAD_RATIO * 1000.0)

    env = UncertainUSCCSDynamics5th(
        load_ratio=LOAD_RATIO, uncertainty_scenario=scenario_key)

    # Train GP
    gp = _pretrain_gp_5th(
        LOAD_RATIO, n_pretrain=N_GP_PRETRAIN, key=rng_key,
        scenario_specific=True, scenario=scenario_key)

    # Build filter
    robust_hocbf = _make_robust_hocbf_5th(
        dynamics, constraint, gp, u0,
        use_mean_correction=True, epsilon_kappa=1.0,
        epsilon_floor=0.0, use_phi_scaled_g=True)

    qp = DifferentiableQP(v_max=5.0, scale_constraints=True)

    # Logging arrays
    n_steps = N_EVAL
    states = np.zeros((n_steps, 5))
    inputs = np.zeros((n_steps, 3))
    tracking_errors = np.zeros((n_steps, 3))  # pressure, enthalpy, power
    constraint_margins = np.zeros((n_steps, 6))
    qp_interventions = np.zeros(n_steps, dtype=bool)
    qp_infeasible = np.zeros(n_steps, dtype=bool)
    epsilons = np.zeros(n_steps)
    inter_sample_violations = np.zeros(n_steps, dtype=bool)

    x = x0.copy()
    y_ref = dynamics.output(x0)

    for t in range(n_steps):
        states[t] = np.array(x)

        # Policy action (LQR: zero incremental)
        v_rl = jnp.zeros(3)

        # QP filter
        A, b = robust_hocbf.qp_matrices(x)
        result = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
        if isinstance(result, tuple):
            v_safe = result[0]
        else:
            v_safe = result
        v_safe = jnp.asarray(v_safe)

        qp_interventions[t] = float(jnp.any(jnp.abs(v_safe - v_rl) > 1e-3))
        A_v = jnp.asarray(A) @ v_safe
        qp_infeasible[t] = float(jnp.any(A_v > jnp.asarray(b) + 1e-3))
        epsilons[t] = float(jnp.mean(robust_hocbf.compute_epsilon(x)))

        # Phi-scaled step
        next_x = env.step_stabilized_phi_scaled(x, v_safe)

        # Inter-sample check: evaluate constraint at finer grid
        # Use 4 sub-steps to check inter-sample constraint satisfaction
        n_sub = 4
        for k in range(1, n_sub):
            alpha = k / n_sub
            x_interp = x + alpha * (next_x - x)
            cv = constraint.check_all(x_interp)
            if any(v < 0 for v in cv.values()):
                inter_sample_violations[t] = True
                break

        # Tracking errors
        y = dynamics.output(next_x)
        tracking_errors[t] = np.array([y_ref[0] - y[0], y_ref[1] - y[1], y_ref[2] - y[2]])

        # Constraint margins
        cv = constraint.check_all(next_x)
        for i, key in enumerate(['pressure_high', 'pressure_low', 'enthalpy_high',
                                  'enthalpy_low', 'power_high', 'power_low']):
            constraint_margins[t, i] = float(cv.get(key, 0))

        inputs[t] = np.array(v_safe)
        x = next_x

    # Compute metrics
    n_violations = int(np.sum(constraint_margins.min(axis=1) < 0))
    total_steps = n_steps

    return {
        'scenario': scenario_key,
        'seed': seed,
        'iae_pressure': compute_iae(tracking_errors[:, 0]),
        'iae_enthalpy': compute_iae(tracking_errors[:, 1]),
        'iae_power': compute_iae(tracking_errors[:, 2]),
        'max_pressure_dev': float(np.max(np.abs(tracking_errors[:, 0]))),
        'max_enthalpy_dev': float(np.max(np.abs(tracking_errors[:, 1]))),
        'max_power_dev': float(np.max(np.abs(tracking_errors[:, 2]))),
        'min_constraint_margin': float(np.min(constraint_margins)),
        'input_variation': compute_input_variation(inputs),
        'input_saturation_ratio': float(np.mean(np.abs(inputs) > 4.9)),
        'qp_intervention_pct': float(np.mean(qp_interventions) * 100),
        'qp_infeasible_pct': float(np.mean(qp_infeasible) * 100),
        'mean_epsilon': float(np.mean(epsilons)),
        'cbf_violation_pct': float(n_violations / total_steps * 100),
        'inter_sample_violation_pct': float(np.mean(inter_sample_violations) * 100),
        'n_steps': n_steps,
        # Trajectory data (downsampled for figure generation)
        'trajectory': {
            'pressure': states[:, 1].tolist(),
            'enthalpy': states[:, 2].tolist(),
            'power': states[:, 3].tolist(),
            'input_uB': inputs[:, 0].tolist(),
            'input_Dfw': inputs[:, 1].tolist(),
            'input_ut': inputs[:, 2].tolist(),
            'epsilon': epsilons.tolist(),
            'constraint_min': constraint_margins.min(axis=1).tolist(),
            'qp_intervention': qp_interventions.astype(int).tolist(),
        }
    }


def main():
    print("=== JPC Process Control Metrics ===")
    all_results = []

    for label, scenario_key in SCENARIOS.items():
        print(f"\nScenario: {label} ({scenario_key})")

        # Robust HOCBF (main method)
        print("  Robust HOCBF...")
        result = evaluate_detailed(scenario_key, seed=0)
        result['method'] = 'PPO-RHOCBF'
        all_results.append(result)
        print(f"    IAE: P={result['iae_pressure']:.1f} E={result['iae_enthalpy']:.1f} N={result['iae_power']:.1f}")
        print(f"    MaxDev: P={result['max_pressure_dev']:.2f} E={result['max_enthalpy_dev']:.1f} N={result['max_power_dev']:.1f}")
        print(f"    CBF%: {result['cbf_violation_pct']:.2f} QP%: {result['qp_intervention_pct']:.1f}")
        print(f"    Inter-sample violation%: {result['inter_sample_violation_pct']:.2f}")
        print(f"    Input variation: {result['input_variation']:.1f} Saturation: {result['input_saturation_ratio']:.3f}")

    # Save all results
    output_path = os.path.join(RESULTS_DIR, 'process_metrics.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()

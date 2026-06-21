"""Power constraint violation decomposition and mitigation.

MF-4 evidence: power constraint has relative degree 0 (algebraic function
of state x2 and input u3), so CBF theory cannot enforce it. Two complementary
mechanisms are tested:

1. Reward shaping: r_power = -λ_power * max(0, |N_e - N_target| - Δ_N)²
   Tests λ_power ∈ {0.0, 0.01, 0.1, 1.0}
2. QP input constraint: adds linear bounds on v[2] derived from power limits
   via the relationship N_e = 0.00055 * (u3/100) * f(x2).

Produces per-constraint-type violation rates and Wilson CIs.
"""
import json
import time
import sys
import os
from pathlib import Path

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.80')
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import yaml

from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.gp.gp_residual import GPResidual
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from experiments.phase4.methods import (
    _make_robust_hocbf, _pretrain_gp, _rollout_with_qp,
    _rollout_no_qp, _collect_gp_data, _count_violations,
    CBF_PROTECTED,
)
from experiments.phase4.statistics import wilson_ci, format_violation_with_ci


def _compute_power_input_bounds(dynamics, x, power_target, power_deviation):
    """Compute v[2] bounds from power constraint for a given state.

    N_e = 0.00055 * (u3/100) * f(x2) = 0.0000055 * u3 * f(x2)
    ⟹ u3_min = (N_target - Δ_N) / (0.0000055 * f(x2))
       u3_max = (N_target + Δ_N) / (0.0000055 * f(x2))

    Since u3 = u0[2] + K[2,:] @ (x0 - x) + v[2], the bounds on v[2] are:
       v2_min = u3_min - u0[2] - K[2,:] @ (x0 - x)
       v2_max = u3_max - u0[2] - K[2,:] @ (x0 - x)

    Returns (v2_min, v2_max) or None if f(x2) ≈ 0.
    """
    fp = dynamics.fluid_property(x[1])
    if abs(float(fp)) < 1e-6:
        return None

    coeff = 0.0000055 * fp
    u3_min = (power_target - power_deviation) / coeff
    u3_max = (power_target + power_deviation) / coeff

    # Clamp to physical bounds
    u3_min = max(u3_min, dynamics.u_bounds[2][0])
    u3_max = min(u3_max, dynamics.u_bounds[2][1])

    # Convert to deviation control bounds
    u_base_3 = dynamics._u0[2] + dynamics._K[2, :] @ (dynamics._x0 - x)
    v2_min = u3_min - u_base_3
    v2_max = u3_max - u_base_3

    return (float(v2_min), float(v2_max))


def _rollout_with_power_constraint(
        model, dynamics, multi_hocbf, qp_solver, constraint, x0, u0,
        key, n_steps=500, lambda_power=0.0, use_input_constraint=False,
        jit_qp_fn=None):
    """Rollout with power constraint mitigation.

    Adds:
    - Power penalty to reward: r_power = -λ_power * max(0, |N_e - N_target| - Δ_N)²
    - Optional QP input constraint on v[2]
    """
    x = x0
    total_reward = 0.0
    violations_all = 0
    cbf_violations = 0
    power_violations = 0
    pressure_violations = 0
    enthalpy_violations = 0
    qp_times = []

    for t in range(n_steps):
        key, action_key = jax.random.split(key)
        v_rl, _, _ = model.get_action(x[:3], action_key)

        # QP safety filter
        if jit_qp_fn is not None:
            A, b = jit_qp_fn(x[:3])
        else:
            A, b = multi_hocbf.qp_matrices(x[:3])

        # Add power input constraint on v[2]
        if use_input_constraint:
            bounds = _compute_power_input_bounds(
                dynamics, x[:3],
                constraint.power_target, constraint.power_deviation)
            if bounds is not None:
                v2_min, v2_max = bounds
                # Row: v[2] <= v2_max  →  [0, 0, 1] v <= v2_max
                A_power_upper = jnp.array([[0.0, 0.0, 1.0]])
                b_power_upper = jnp.array([v2_max])
                # Row: -v[2] <= -v2_min  →  [0, 0, -1] v <= -v2_min
                A_power_lower = jnp.array([[0.0, 0.0, -1.0]])
                b_power_lower = jnp.array([-v2_min])
                A = jnp.concatenate([A, A_power_upper, A_power_lower], axis=0)
                b = jnp.concatenate([b, b_power_upper, b_power_lower])

        t0 = time.perf_counter()
        v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
        v_max = qp_solver.v_max if qp_solver.v_max else 10.0
        v_safe = jnp.clip(v_safe, -v_max, v_max)
        qp_times.append((time.perf_counter() - t0) * 1000)

        next_x = dynamics.step_stabilized(x[:3], v_safe)
        u_total = dynamics.compute_total_control(x[:3], v_safe)
        constraint_vals = constraint.check_all(next_x, u_total)

        # Reward with optional power penalty
        y = dynamics.output(next_x, u_total)
        y0 = dynamics.output(x0, u0)
        reward = (
            -1.0 * (y[0] - y0[0]) ** 2
            - 0.001 * (y[1] - y0[1]) ** 2
            - 0.01 * (y[2] - y0[2]) ** 2
            - 0.0001 * jnp.sum(v_safe ** 2)
        )

        # Power reward shaping
        if lambda_power > 0:
            N_e = y[2]
            power_dev = jnp.abs(N_e - constraint.power_target)
            power_excess = jnp.maximum(0.0, power_dev - constraint.power_deviation)
            reward = reward - lambda_power * power_excess ** 2

        total_reward += float(reward)

        # Per-type violation counting
        if _count_violations(constraint_vals, protected_only=False):
            violations_all += 1
        if _count_violations(constraint_vals, protected_only=True):
            cbf_violations += 1
        # Power
        if constraint_vals.get('power_high', 1.0) < 0 or constraint_vals.get('power_low', 1.0) < 0:
            power_violations += 1
        # Pressure
        if constraint_vals.get('pressure_high', 1.0) < 0 or constraint_vals.get('pressure_low', 1.0) < 0:
            pressure_violations += 1
        # Enthalpy
        if constraint_vals.get('enthalpy_high', 1.0) < 0 or constraint_vals.get('enthalpy_low', 1.0) < 0:
            enthalpy_violations += 1

        x = next_x

    return {
        'total_violations': violations_all,
        'cbf_violations': cbf_violations,
        'power_violations': power_violations,
        'pressure_violations': pressure_violations,
        'enthalpy_violations': enthalpy_violations,
        'n_steps': n_steps,
        'reward': total_reward,
        'solve_time_ms': np.mean(qp_times) if qp_times else 0,
    }


def _rollout_no_qp_with_power_penalty(
        model, dynamics, constraint, x0, u0,
        key, n_steps=500, lambda_power=0.0):
    """Rollout without QP but with power reward shaping (for training)."""
    rollout = {'obs': [], 'actions': [], 'rewards': [],
               'log_probs': [], 'values': [], 'dones': []}

    x = x0
    total_reward = 0.0

    for t in range(n_steps):
        key, action_key = jax.random.split(key)
        v_rl, log_prob, value = model.get_action(x[:3], action_key)

        next_x = dynamics.step_stabilized(x[:3], v_rl)
        u_total = dynamics.compute_total_control(x[:3], v_rl)

        y = dynamics.output(next_x, u_total)
        y0 = dynamics.output(x0, u0)
        reward = (
            -1.0 * (y[0] - y0[0]) ** 2
            - 0.001 * (y[1] - y0[1]) ** 2
            - 0.01 * (y[2] - y0[2]) ** 2
            - 0.0001 * jnp.sum(v_rl ** 2)
        )

        if lambda_power > 0:
            N_e = y[2]
            power_dev = jnp.abs(N_e - constraint.power_target)
            power_excess = jnp.maximum(0.0, power_dev - constraint.power_deviation)
            reward = reward - lambda_power * power_excess ** 2

        rollout['obs'].append(x[:3])
        rollout['actions'].append(v_rl)
        rollout['rewards'].append(reward)
        rollout['log_probs'].append(log_prob)
        rollout['values'].append(value)
        rollout['dones'].append(jnp.float32(0.0))

        total_reward += float(reward)
        x = next_x

    for k in ['obs', 'actions', 'rewards', 'log_probs', 'values', 'dones']:
        rollout[k] = jnp.stack(rollout[k])

    return rollout, total_reward


def train_with_power_shaping(model, trainer, train_dyn, constraint, x0, u0,
                              lambda_power=0.0, n_episodes=200, n_steps=200,
                              seed=42):
    """Train PPO with power reward shaping."""
    key = jax.random.key(seed)
    reward_history = []

    for ep in range(n_episodes):
        key, rollout_key = jax.random.split(key)
        rollout, ep_reward = _rollout_no_qp_with_power_penalty(
            model, train_dyn, constraint, x0, u0, rollout_key,
            n_steps=n_steps, lambda_power=lambda_power)

        if rollout['obs'].shape[0] < 2:
            continue

        reward_history.append(ep_reward)
        advantages, returns = compute_gae(
            rollout['rewards'], rollout['values'], rollout['dones'])
        batch = {
            'obs': rollout['obs'],
            'actions': rollout['actions'],
            'old_log_probs': rollout['log_probs'],
            'advantages': advantages,
            'returns': returns,
        }
        for _ in range(trainer.epochs):
            trainer.train_step(batch)

        if (ep + 1) % 50 == 0:
            avg_r = np.mean(reward_history[-50:])
            print(f"  [λ={lambda_power}] Ep {ep+1}: r={ep_reward:.1f} (avg50={avg_r:.1f})", flush=True)

    return model, reward_history


def evaluate_method(model, safety_layer, qp_solver, dynamics, constraint,
                    x0, u0, lambda_power=0.0, use_input_constraint=False,
                    n_episodes=50, n_steps=500, seed=42, jit_qp_fn=None):
    """Evaluate with full per-type violation decomposition."""
    key = jax.random.key(seed)
    all_power_viol = []
    all_pressure_viol = []
    all_enthalpy_viol = []
    all_cbf_viol = []
    all_total_viol = []
    all_rewards = []
    all_solve_times = []

    for ep in range(n_episodes):
        key, ep_key = jax.random.split(key)
        result = _rollout_with_power_constraint(
            model, dynamics, safety_layer, qp_solver, constraint,
            x0, u0, ep_key, n_steps=n_steps,
            lambda_power=lambda_power,
            use_input_constraint=use_input_constraint,
            jit_qp_fn=jit_qp_fn)

        n = result['n_steps']
        all_power_viol.append(result['power_violations'] / n)
        all_pressure_viol.append(result['pressure_violations'] / n)
        all_enthalpy_viol.append(result['enthalpy_violations'] / n)
        all_cbf_viol.append(result['cbf_violations'] / n)
        all_total_viol.append(result['total_violations'] / n)
        all_rewards.append(result['reward'])
        all_solve_times.append(result['solve_time_ms'])
        if (ep + 1) % 10 == 0:
            print(f"    Eval ep {ep+1}/{n_episodes}: power_viol={result['power_violations']/n:.4f}, "
                  f"cbf={result['cbf_violations']/n:.4f}", flush=True)

    return {
        'power_violation_rate': np.mean(all_power_viol),
        'pressure_violation_rate': np.mean(all_pressure_viol),
        'enthalpy_violation_rate': np.mean(all_enthalpy_viol),
        'cbf_violation_rate': np.mean(all_cbf_viol),
        'total_violation_rate': np.mean(all_total_viol),
        'reward': np.mean(all_rewards),
        'solve_time_ms': np.mean(all_solve_times),
        'power_viol_counts': [int(r * n_steps) for r in all_power_viol],
        'pressure_viol_counts': [int(r * n_steps) for r in all_pressure_viol],
        'enthalpy_viol_counts': [int(r * n_steps) for r in all_enthalpy_viol],
        'cbf_viol_counts': [int(r * n_steps) for r in all_cbf_viol],
        'total_viol_counts': [int(r * n_steps) for r in all_total_viol],
        'n_steps': n_steps,
        'n_episodes': n_episodes,
    }


def run_power_mitigation(n_episodes=200, n_steps=200, n_seeds=5,
                         scenario='pressure_oscillation',
                         n_eval_episodes=30, n_eval_steps=300):
    """Run power constraint mitigation experiment.

    Tests 6 conditions:
    1. Baseline: λ=0, no input constraint
    2. λ=0.01, no input constraint
    3. λ=0.1, no input constraint
    4. λ=1.0, no input constraint
    5. λ=0, with input constraint (QP bounds on v[2])
    6. λ=0.1, with input constraint (combined approach)

    Uses S2 (pressure_oscillation) scenario where power violation is 94.6%.
    """
    output_dir = 'results/phase5/power_mitigation/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open('configs/phase4.yaml') as f:
        config = yaml.safe_load(f)

    hocbf_cfg = config['hocbf']
    method_cfg = config['methods_config']['rocbf_net']

    conditions = [
        {'label': 'baseline', 'lambda_power': 0.0, 'use_input_constraint': False},
        {'label': 'lambda_0.01', 'lambda_power': 0.01, 'use_input_constraint': False},
        {'label': 'lambda_0.1', 'lambda_power': 0.1, 'use_input_constraint': False},
        {'label': 'lambda_1.0', 'lambda_power': 1.0, 'use_input_constraint': False},
        {'label': 'input_constraint', 'lambda_power': 0.0, 'use_input_constraint': True},
        {'label': 'combined', 'lambda_power': 0.1, 'use_input_constraint': True},
    ]

    all_results = {}

    for seed in range(n_seeds):
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        key = jax.random.key(seed)
        base_dyn = USCCSDynamics(delay_order=0, load_ratio=1.0)
        x0, u0 = base_dyn.equilibrium(1.0)
        u0_arr = base_dyn._u0
        constraint = CCSConstraints(
            p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
            power_deviation=50.0, power_target=1000.0,
            dynamics=base_dyn)
        train_dyn = UncertainUSCCSDynamics(
            delay_order=0, load_ratio=1.0, uncertainty_scenario=scenario)

        # Pre-train GP
        key, gp_key = jax.random.split(key)
        gp = _pretrain_gp(1.0, 0, n_pretrain=2000, key=gp_key,
                          sigma_floor=1e-4, scenario=scenario,
                          scenario_specific=True)

        k_p = tuple(hocbf_cfg['pressure_k_gains'])
        k_h = tuple(hocbf_cfg['enthalpy_k_gains'])
        u_max = hocbf_cfg['u_max']

        safety_layer = _make_robust_hocbf(
            base_dyn, constraint, gp, u0_arr, epsilon_kappa=1.0,
            k_pressure=k_p, k_enthalpy=k_h, u_max=u_max,
            use_mean_correction=True, epsilon_floor=0.0)
        qp_solver = DifferentiableQP(v_max=5.0)

        # JIT-compile qp_matrices for ~55x speedup in evaluation
        jit_qp_fn = jax.jit(safety_layer.qp_matrices)
        _ = jit_qp_fn(x0[:3])  # warm up JIT compilation
        print(f"  JIT qp_matrices compiled", flush=True)

        seed_results = {}

        for cond in conditions:
            label = cond['label']
            lam = cond['lambda_power']
            use_ic = cond['use_input_constraint']

            print(f"\n--- Condition: {label} (λ={lam}, input_constraint={use_ic}) ---")

            # Train with reward shaping
            model = ActorCritic(n_obs=3, n_act=3, hidden_dim=128, rngs=nnx.Rngs(0))
            trainer = PPOTrainer(model, lr=1e-4)
            model, rewards = train_with_power_shaping(
                model, trainer, train_dyn, constraint, x0, u0,
                lambda_power=lam, n_episodes=n_episodes,
                n_steps=n_steps, seed=seed)

            # Evaluate with QP + optional input constraint
            # Use train_dyn (uncertain) to measure violations under S2 scenario
            eval_result = evaluate_method(
                model, safety_layer, qp_solver, train_dyn, constraint,
                x0, u0, lambda_power=0.0,  # eval reward is standard
                use_input_constraint=use_ic,
                n_episodes=n_eval_episodes, n_steps=n_eval_steps, seed=seed+100,
                jit_qp_fn=jit_qp_fn)

            seed_results[label] = eval_result
            print(f"  {label}: power_viol={eval_result['power_violation_rate']:.4f}, "
                  f"cbf_viol={eval_result['cbf_violation_rate']:.4f}, "
                  f"pressure_viol={eval_result['pressure_violation_rate']:.4f}, "
                  f"enthalpy_viol={eval_result['enthalpy_violation_rate']:.4f}")

        all_results[f'seed_{seed}'] = seed_results

    # Save results
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

    with open(f'{output_dir}power_mitigation.json', 'w') as f:
        json.dump(_convert(all_results), f, indent=2)

    # Print summary table
    print(f"\n{'='*90}")
    print("POWER CONSTRAINT MITIGATION SUMMARY")
    print(f"{'='*90}")
    print(f"{'Condition':<20} {'Power Viol':<15} {'CBF Viol':<15} "
          f"{'Pressure':<15} {'Enthalpy':<15} {'Reward':<10}")
    print("-" * 90)

    for cond in conditions:
        label = cond['label']
        power_viols = [all_results[f'seed_{s}'][label]['power_violation_rate']
                       for s in range(n_seeds)]
        cbf_viols = [all_results[f'seed_{s}'][label]['cbf_violation_rate']
                     for s in range(n_seeds)]
        pressure_viols = [all_results[f'seed_{s}'][label]['pressure_violation_rate']
                          for s in range(n_seeds)]
        enthalpy_viols = [all_results[f'seed_{s}'][label]['enthalpy_violation_rate']
                          for s in range(n_seeds)]
        rewards = [all_results[f'seed_{s}'][label]['reward']
                   for s in range(n_seeds)]

        # Wilson CI for power violation
        total_power_steps = sum(
            sum(all_results[f'seed_{s}'][label].get('power_viol_counts', [0]*n_eval_episodes))
            for s in range(n_seeds))
        n_total = n_seeds * n_eval_episodes * n_eval_steps
        power_rate, power_lo, power_hi = wilson_ci(total_power_steps, n_total)

        print(f"{label:<20} {power_rate*100:.2f}% [{power_lo*100:.2f},{power_hi*100:.2f}]  "
              f"{np.mean(cbf_viols)*100:.2f}%       "
              f"{np.mean(pressure_viols)*100:.2f}%       "
              f"{np.mean(enthalpy_viols)*100:.2f}%     "
              f"{np.mean(rewards):.0f}")

    print(f"{'='*90}")

    # Print LaTeX-formatted table
    print(f"\n--- LaTeX Table ---")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Power constraint violation rates under S2 (pressure oscillation)}")
    print(r"\label{tab:power_mitigation}")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"Method & Power Viol. & CBF Viol. & Pressure & Enthalpy & Reward \\")
    print(r"\midrule")

    for cond in conditions:
        label = cond['label']
        total_power_steps = sum(
            sum(all_results[f'seed_{s}'][label].get('power_viol_counts', [0]*n_eval_episodes))
            for s in range(n_seeds))
        n_total = n_seeds * n_eval_episodes * n_eval_steps
        power_fmt = format_violation_with_ci(total_power_steps, n_total)

        cbf_viols = [all_results[f'seed_{s}'][label]['cbf_violation_rate']
                     for s in range(n_seeds)]
        rewards = [all_results[f'seed_{s}'][label]['reward']
                   for s in range(n_seeds)]

        print(f"{label} & {power_fmt} & {np.mean(cbf_viols)*100:.2f}\\% & "
              f"{np.mean([all_results[f'seed_{s}'][label]['pressure_violation_rate'] for s in range(n_seeds)])*100:.2f}\\% & "
              f"{np.mean([all_results[f'seed_{s}'][label]['enthalpy_violation_rate'] for s in range(n_seeds)])*100:.2f}\\% & "
              f"{np.mean(rewards):.0f} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=200)
    parser.add_argument('--n_steps', type=int, default=200)
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--scenario', type=str, default='pressure_oscillation')
    parser.add_argument('--n_eval_episodes', type=int, default=30)
    parser.add_argument('--n_eval_steps', type=int, default=300)
    args = parser.parse_args()

    results = run_power_mitigation(
        n_episodes=args.n_episodes,
        n_steps=args.n_steps,
        n_seeds=args.n_seeds,
        scenario=args.scenario,
        n_eval_episodes=args.n_eval_episodes,
        n_eval_steps=args.n_eval_steps,
    )

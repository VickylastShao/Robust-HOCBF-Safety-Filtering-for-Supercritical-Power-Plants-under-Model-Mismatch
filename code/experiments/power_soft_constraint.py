"""Power constraint mitigation via soft constraint in QP objective.

Instead of adding power as a hard constraint (which conflicts with CBF),
we modify the QP objective to prefer actions that also satisfy the power
constraint while keeping CBF as hard constraints.

QP formulation:
  min ||u - u_rl||^2 + w_power * (v[2] - v2_power)^2
  s.t. A u ≤ b  (CBF constraints only)

where v2_power is the v[2] value closest to power compliance.

This is equivalent to:
  min u^T P u + q^T u   s.t. A u ≤ b
where P = I + w_power * e3 e3^T  (e3 = [0,0,1])
      q = -u_rl - w_power * v2_power * e3

When CBF constraints don't conflict with power, the solution naturally
satisfies both. When they conflict, CBF takes priority (hard constraint),
but the power-aware objective finds the CBF-feasible action closest to
power compliance.
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


def _compute_power_preferred_v2(dynamics, x, power_target, power_deviation):
    """Compute the v[2] value that makes power output closest to target.

    N_e = 0.00055 * (u3/100) * f(x2) = 0.0000055 * u3 * f(x2)
    For N_e = power_target: u3_target = power_target / (0.0000055 * f(x2))

    Returns v2_target = u3_target - u0[2] - K[2,:] @ (x0 - x)
    or None if f(x2) ≈ 0.
    """
    fp = dynamics.fluid_property(x[1])
    if abs(float(fp)) < 1e-6:
        return None

    coeff = 0.0000055 * fp
    u3_target = power_target / coeff

    # Clamp to physical bounds
    u3_target = np.clip(u3_target, dynamics.u_bounds[2][0], dynamics.u_bounds[2][1])

    # Convert to deviation control
    u_base_3 = dynamics._u0[2] + dynamics._K[2, :] @ (dynamics._x0 - x)
    v2_target = u3_target - u_base_3

    return float(v2_target)


def _solve_power_aware_qp(qp_solver, u_rl, A, b, v2_power, w_power,
                           differentiable=False):
    """Solve QP with power-aware objective.

    min ||u - u_rl||^2 + w_power * (v[2] - v2_power)^2
    s.t. A u ≤ b

    Expanding:
      P = I + w_power * e3 e3^T  where e3 = [0,0,1]
      q = -u_rl - w_power * v2_power * e3

    This is a standard QP that qpax/scipy can solve.
    """
    n = u_rl.shape[0]

    # Build modified P and q
    P = np.eye(n)
    P[2, 2] += w_power  # Add power weight to v[2] dimension

    q = np.array(-u_rl, dtype=np.float64)
    q[2] -= w_power * v2_power  # Pull v[2] toward power compliance

    if differentiable:
        P_jax = jnp.array(P)
        q_jax = jnp.array(q)
        u_star = qp_solver.solve_primal(P_jax, q_jax, A, b)
        u_star = jnp.where(jnp.isfinite(u_star), u_star, jnp.zeros(n))
        if qp_solver.v_max is not None and qp_solver.v_max > 0:
            u_star = jnp.clip(u_star, -qp_solver.v_max, qp_solver.v_max)
        return u_star
    else:
        # Use scipy for robust non-differentiable solve
        from scipy.optimize import minimize as sp_minimize

        A_np = np.array(A, dtype=np.float64)
        b_np = np.array(b, dtype=np.float64)

        def obj(v):
            dv = v - np.array(u_rl)
            power_term = w_power * (v[2] - v2_power) ** 2
            return float(dv @ dv + power_term)

        def obj_grad(v):
            dv = v - np.array(u_rl)
            grad = 2.0 * dv
            grad[2] += 2.0 * w_power * (v[2] - v2_power)
            return grad

        constraints_scipy = {
            'type': 'ineq',
            'fun': lambda v: b_np - A_np @ v,
            'jac': lambda v: -A_np,
        }

        bounds = None
        if qp_solver.v_max is not None and qp_solver.v_max > 0:
            bounds = [(-float(qp_solver.v_max), float(qp_solver.v_max))] * n

        result = sp_minimize(
            obj, np.array(u_rl, dtype=np.float64), method='SLSQP',
            jac=obj_grad, constraints=constraints_scipy, bounds=bounds,
            options={'ftol': 1e-12, 'maxiter': 500})

        u_star = jnp.array(result.x)
        if not jnp.all(jnp.isfinite(u_star)):
            u_star = jnp.zeros(n)
        if qp_solver.v_max is not None and qp_solver.v_max > 0:
            u_star = jnp.clip(u_star, -qp_solver.v_max, qp_solver.v_max)
        return u_star


def _rollout_power_aware(
        model, dynamics, multi_hocbf, qp_solver, constraint, x0, u0,
        key, n_steps=500, w_power=0.0, jit_qp_fn=None):
    """Rollout with power-aware QP objective.

    When w_power > 0, the QP objective includes a penalty for
    deviating from the power-compliant v[2] value, pulling the
    solution toward power compliance while maintaining CBF safety.
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

        # CBF constraint matrices
        if jit_qp_fn is not None:
            A, b = jit_qp_fn(x[:3])
        else:
            A, b = multi_hocbf.qp_matrices(x[:3])

        t0 = time.perf_counter()

        if w_power > 0:
            # Power-aware QP
            v2_power = _compute_power_preferred_v2(
                dynamics, x[:3], constraint.power_target, constraint.power_deviation)
            if v2_power is not None:
                v_safe = _solve_power_aware_qp(
                    qp_solver, v_rl, A, b, v2_power, w_power,
                    differentiable=False)
            else:
                v_safe, _ = qp_solver.solve_with_rl_action(
                    v_rl, A, b, differentiable=False)
        else:
            v_safe, _ = qp_solver.solve_with_rl_action(
                v_rl, A, b, differentiable=False)

        v_max = qp_solver.v_max if qp_solver.v_max else 10.0
        v_safe = jnp.clip(v_safe, -v_max, v_max)
        qp_times.append((time.perf_counter() - t0) * 1000)

        next_x = dynamics.step_stabilized(x[:3], v_safe)
        u_total = dynamics.compute_total_control(x[:3], v_safe)
        constraint_vals = constraint.check_all(next_x, u_total)

        # Standard reward (no power shaping in eval)
        y = dynamics.output(next_x, u_total)
        y0 = dynamics.output(x0, u0)
        reward = (
            -1.0 * (y[0] - y0[0]) ** 2
            - 0.001 * (y[1] - y0[1]) ** 2
            - 0.01 * (y[2] - y0[2]) ** 2
            - 0.0001 * jnp.sum(v_safe ** 2)
        )
        total_reward += float(reward)

        # Per-type violation counting
        if _count_violations(constraint_vals, protected_only=False):
            violations_all += 1
        if _count_violations(constraint_vals, protected_only=True):
            cbf_violations += 1
        if constraint_vals.get('power_high', 1.0) < 0 or constraint_vals.get('power_low', 1.0) < 0:
            power_violations += 1
        if constraint_vals.get('pressure_high', 1.0) < 0 or constraint_vals.get('pressure_low', 1.0) < 0:
            pressure_violations += 1
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


def evaluate_power_aware(model, safety_layer, qp_solver, dynamics, constraint,
                         x0, u0, w_power=0.0, n_episodes=30, n_steps=300,
                         seed=42, jit_qp_fn=None):
    """Evaluate with power-aware QP objective."""
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
        result = _rollout_power_aware(
            model, dynamics, safety_layer, qp_solver, constraint,
            x0, u0, ep_key, n_steps=n_steps, w_power=w_power,
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
            print(f"    Eval ep {ep+1}/{n_episodes}: "
                  f"power_viol={result['power_violations']/n:.4f}, "
                  f"cbf={result['cbf_violations']/n:.4f}", flush=True)

    return {
        'power_violation_rate': np.mean(all_power_viol),
        'pressure_violation_rate': np.mean(all_pressure_viol),
        'enthalpy_violation_rate': np.mean(all_enthalpy_viol),
        'cbf_violation_rate': np.mean(all_cbf_viol),
        'total_violation_rate': np.mean(all_total_viol),
        'reward': np.mean(all_rewards),
        'solve_time_ms': np.mean(all_solve_times),
        'w_power': w_power,
        'n_steps': n_steps,
        'n_episodes': n_episodes,
    }


def run_power_soft_constraint(n_seeds=5, scenario='pressure_oscillation',
                               n_eval_episodes=30, n_eval_steps=300):
    """Run power soft constraint experiment.

    Compares multiple QP objective configurations:
    1. Baseline: standard min ||u - u_rl||^2 (w_power=0)
    2. Input constraint: hard power bounds on v[2]
    3. Soft constraint w_power=0.1
    4. Soft constraint w_power=1.0
    5. Soft constraint w_power=10.0
    6. Soft constraint w_power=100.0
    7. Combined: input constraint + soft w_power=1.0
    """
    output_dir = 'results/phase5/power_soft_constraint/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open('configs/phase4.yaml') as f:
        config = yaml.safe_load(f)

    hocbf_cfg = config['hocbf']

    conditions = [
        {'label': 'baseline', 'type': 'standard', 'w_power': 0.0, 'use_input_constraint': False},
        {'label': 'input_constraint', 'type': 'hard', 'w_power': 0.0, 'use_input_constraint': True},
        {'label': 'soft_w0.1', 'type': 'soft', 'w_power': 0.1, 'use_input_constraint': False},
        {'label': 'soft_w1.0', 'type': 'soft', 'w_power': 1.0, 'use_input_constraint': False},
        {'label': 'soft_w10', 'type': 'soft', 'w_power': 10.0, 'use_input_constraint': False},
        {'label': 'soft_w100', 'type': 'soft', 'w_power': 100.0, 'use_input_constraint': False},
        {'label': 'combined_soft_w1', 'type': 'soft', 'w_power': 1.0, 'use_input_constraint': True},
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

        # Pre-train GP (scenario-specific for S2)
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

        # Train standard PPO (decoupled, no power shaping)
        model = ActorCritic(n_obs=3, n_act=3, hidden_dim=128, rngs=nnx.Rngs(0))
        trainer = PPOTrainer(model, lr=1e-4)

        # Train on uncertain dynamics
        key, train_key = jax.random.split(key)
        for ep in range(200):
            train_key, rollout_key = jax.random.split(train_key)
            rollout, ep_reward, _, _, _ = _rollout_no_qp(
                model, train_dyn, constraint, x0, u0, rollout_key, n_steps=200)
            if rollout['obs'].shape[0] < 2:
                continue
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
                print(f"  Training ep {ep+1}: r={ep_reward:.1f}", flush=True)

        # JIT-compile qp_matrices
        jit_qp_fn = jax.jit(safety_layer.qp_matrices)
        _ = jit_qp_fn(x0[:3])
        print(f"  JIT compiled", flush=True)

        seed_results = {}

        for cond in conditions:
            label = cond['label']
            print(f"\n--- Condition: {label} ---")

            if cond['type'] == 'soft':
                # Soft constraint QP
                eval_result = evaluate_power_aware(
                    model, safety_layer, qp_solver, train_dyn, constraint,
                    x0, u0, w_power=cond['w_power'],
                    n_episodes=n_eval_episodes, n_steps=n_eval_steps,
                    seed=seed+100, jit_qp_fn=jit_qp_fn)
            elif cond['type'] == 'hard':
                # Hard input constraint (from existing code)
                from experiments.phase5.power_constraint_mitigation import (
                    evaluate_method as evaluate_hard)
                eval_result = evaluate_hard(
                    model, safety_layer, qp_solver, train_dyn, constraint,
                    x0, u0, lambda_power=0.0, use_input_constraint=True,
                    n_episodes=n_eval_episodes, n_steps=n_eval_steps,
                    seed=seed+100, jit_qp_fn=jit_qp_fn)
            else:
                # Baseline (standard QP)
                eval_result = evaluate_power_aware(
                    model, safety_layer, qp_solver, train_dyn, constraint,
                    x0, u0, w_power=0.0,
                    n_episodes=n_eval_episodes, n_steps=n_eval_steps,
                    seed=seed+100, jit_qp_fn=jit_qp_fn)

            seed_results[label] = eval_result
            print(f"  {label}: power_viol={eval_result['power_violation_rate']*100:.2f}%, "
                  f"cbf_viol={eval_result['cbf_violation_rate']*100:.2f}%, "
                  f"reward={eval_result['reward']:.0f}")

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

    with open(f'{output_dir}power_soft_constraint.json', 'w') as f:
        json.dump(_convert(all_results), f, indent=2)

    # Print summary
    print(f"\n{'='*90}")
    print("POWER SOFT CONSTRAINT SUMMARY")
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

        print(f"{label:<20} {np.mean(power_viols)*100:.2f}±{np.std(power_viols)*100:.2f}%  "
              f"{np.mean(cbf_viols)*100:.2f}%       "
              f"{np.mean(pressure_viols)*100:.2f}%       "
              f"{np.mean(enthalpy_viols)*100:.2f}%     "
              f"{np.mean(rewards):.0f}")

    print(f"{'='*90}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--scenario', type=str, default='pressure_oscillation')
    parser.add_argument('--n_eval_episodes', type=int, default=30)
    parser.add_argument('--n_eval_steps', type=int, default=300)
    args = parser.parse_args()

    results = run_power_soft_constraint(
        n_seeds=args.n_seeds,
        scenario=args.scenario,
        n_eval_episodes=args.n_eval_episodes,
        n_eval_steps=args.n_eval_steps,
    )

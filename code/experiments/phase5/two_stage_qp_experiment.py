"""Two-stage QP for power constraint mitigation.

Phase 1: Standard CBF QP → safe set (A, b)
Phase 2: Replace RL action v[2] with power-preferred value, then project
         onto CBF safe set via the same QP.

The key insight: instead of adding power as a hard constraint (which conflicts
with CBF) or modifying the QP objective (which compromises CBF safety), we
simply replace the reference action's v[2] component with the power-compliant
value. The CBF QP then finds the nearest safe action that is already biased
toward power compliance. When CBF and power are compatible, both are satisfied;
when they conflict, CBF takes priority (hard constraint) but the solution is
the CBF-feasible action closest to power compliance.
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
    _make_robust_hocbf, _pretrain_gp, _rollout_no_qp,
    _count_violations, CBF_PROTECTED,
)
from experiments.phase4.statistics import wilson_ci, format_violation_with_ci
from experiments.phase5.power_soft_constraint import _compute_power_preferred_v2


def _compute_power_input_bounds(dynamics, x, power_target, power_deviation):
    """Compute v[2] bounds from power constraint for a given state."""
    fp = dynamics.fluid_property(x[1])
    if abs(float(fp)) < 1e-6:
        return None

    coeff = 0.0000055 * fp
    u3_min = (power_target - power_deviation) / coeff
    u3_max = (power_target + power_deviation) / coeff

    u3_min = max(u3_min, dynamics.u_bounds[2][0])
    u3_max = min(u3_max, dynamics.u_bounds[2][1])

    u_base_3 = dynamics._u0[2] + dynamics._K[2, :] @ (dynamics._x0 - x)
    v2_min = u3_min - u_base_3
    v2_max = u3_max - u_base_3

    return (float(v2_min), float(v2_max))


def _rollout_eval(model, dynamics, base_dyn, safety_layer, qp_solver, constraint,
                  x0, u0, key, n_steps=300, mode='standard', jit_qp_fn=None):
    """Evaluate with specified QP mode.

    Modes:
      - 'standard': min ||v - v_rl||^2 s.t. CBF constraints
      - 'input_constraint': hard power bounds on v[2] + CBF constraints
      - 'two_stage': replace v_rl[2] with power-preferred value + CBF constraints
    """
    x = x0
    total_reward = 0.0
    power_viols = 0
    cbf_viols = 0
    pressure_viols = 0
    enthalpy_viols = 0

    for t in range(n_steps):
        key, action_key = jax.random.split(key)
        v_rl, _, _ = model.get_action(x[:3], action_key)

        if jit_qp_fn is not None:
            A, b = jit_qp_fn(x[:3])
        else:
            A, b = safety_layer.qp_matrices(x[:3])

        if mode == 'two_stage':
            v2_power = _compute_power_preferred_v2(
                base_dyn, x[:3], constraint.power_target, constraint.power_deviation)
            if v2_power is not None:
                v_ref = jnp.array([v_rl[0], v_rl[1], v2_power])
            else:
                v_ref = v_rl
            v_safe, _ = qp_solver.solve_with_rl_action(
                v_ref, A, b, differentiable=False)

        elif mode == 'input_constraint':
            bounds = _compute_power_input_bounds(
                dynamics, x[:3], constraint.power_target, constraint.power_deviation)
            if bounds is not None:
                v2_min, v2_max = bounds
                A_pu = jnp.array([[0.0, 0.0, 1.0]])
                b_pu = jnp.array([v2_max])
                A_pl = jnp.array([[0.0, 0.0, -1.0]])
                b_pl = jnp.array([-v2_min])
                A = jnp.concatenate([A, A_pu, A_pl], axis=0)
                b = jnp.concatenate([b, b_pu, b_pl])
            v_safe, _ = qp_solver.solve_with_rl_action(
                v_rl, A, b, differentiable=False)

        else:  # standard
            v_safe, _ = qp_solver.solve_with_rl_action(
                v_rl, A, b, differentiable=False)

        v_max = qp_solver.v_max if qp_solver.v_max else 5.0
        v_safe = jnp.clip(v_safe, -v_max, v_max)

        next_x = dynamics.step_stabilized(x[:3], v_safe)
        u_total = dynamics.compute_total_control(x[:3], v_safe)
        constraint_vals = constraint.check_all(next_x, u_total)

        y = dynamics.output(next_x, u_total)
        y0 = dynamics.output(x0, u0)
        reward = (
            -1.0 * (y[0] - y0[0]) ** 2
            - 0.001 * (y[1] - y0[1]) ** 2
            - 0.01 * (y[2] - y0[2]) ** 2
            - 0.0001 * jnp.sum(v_safe ** 2)
        )
        total_reward += float(reward)

        if _count_violations(constraint_vals, protected_only=True):
            cbf_viols += 1
        if constraint_vals.get('power_high', 1.0) < 0 or constraint_vals.get('power_low', 1.0) < 0:
            power_viols += 1
        if constraint_vals.get('pressure_high', 1.0) < 0 or constraint_vals.get('pressure_low', 1.0) < 0:
            pressure_viols += 1
        if constraint_vals.get('enthalpy_high', 1.0) < 0 or constraint_vals.get('enthalpy_low', 1.0) < 0:
            enthalpy_viols += 1

        x = next_x

    return {
        'power_violations': power_viols,
        'cbf_violations': cbf_viols,
        'pressure_violations': pressure_viols,
        'enthalpy_violations': enthalpy_viols,
        'n_steps': n_steps,
        'reward': total_reward,
    }


def run_two_stage_experiment(n_seeds=5, scenario='pressure_oscillation',
                              n_eval_episodes=30, n_eval_steps=300):
    """Run two-stage QP experiment with 5 seeds."""
    output_dir = 'results/phase5/two_stage_qp/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open('configs/phase4.yaml') as f:
        config = yaml.safe_load(f)

    hocbf_cfg = config['hocbf']

    modes = [
        ('standard', 'Baseline (standard QP)'),
        ('input_constraint', 'Input constraint (hard)'),
        ('two_stage', 'Two-stage QP'),
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

        # Pre-train GP (scenario-specific)
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

        # Train PPO (decoupled)
        model = ActorCritic(n_obs=3, n_act=3, hidden_dim=128, rngs=nnx.Rngs(0))
        trainer = PPOTrainer(model, lr=1e-4)

        for ep in range(200):
            key, rollout_key = jax.random.split(key)
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
                print(f"  Train ep {ep+1}: r={ep_reward:.0f}", flush=True)

        # JIT-compile qp_matrices
        jit_qp_fn = jax.jit(safety_layer.qp_matrices)
        _ = jit_qp_fn(x0[:3])
        print(f"  JIT compiled", flush=True)

        seed_results = {}

        for mode, desc in modes:
            print(f"\n--- {desc} ---")

            all_power = []
            all_cbf = []
            all_pressure = []
            all_enthalpy = []
            all_rewards = []

            eval_key = jax.random.key(seed + 100)
            for ep in range(n_eval_episodes):
                eval_key, ep_key = jax.random.split(eval_key)
                result = _rollout_eval(
                    model, train_dyn, base_dyn, safety_layer, qp_solver,
                    constraint, x0, u0, ep_key, n_steps=n_eval_steps,
                    mode=mode, jit_qp_fn=jit_qp_fn)

                n = result['n_steps']
                all_power.append(result['power_violations'] / n)
                all_cbf.append(result['cbf_violations'] / n)
                all_pressure.append(result['pressure_violations'] / n)
                all_enthalpy.append(result['enthalpy_violations'] / n)
                all_rewards.append(result['reward'])

                if (ep + 1) % 10 == 0:
                    print(f"    Ep {ep+1}: power={result['power_violations']/n*100:.1f}%", flush=True)

            seed_results[mode] = {
                'power_violation_rate': float(np.mean(all_power)),
                'cbf_violation_rate': float(np.mean(all_cbf)),
                'pressure_violation_rate': float(np.mean(all_pressure)),
                'enthalpy_violation_rate': float(np.mean(all_enthalpy)),
                'reward': float(np.mean(all_rewards)),
                'power_viol_std': float(np.std(all_power)),
                'reward_std': float(np.std(all_rewards)),
                'n_episodes': n_eval_episodes,
                'n_steps': n_eval_steps,
            }

            print(f"  {mode}: power={np.mean(all_power)*100:.1f}±{np.std(all_power)*100:.1f}%, "
                  f"cbf={np.mean(all_cbf)*100:.1f}%, reward={np.mean(all_rewards):.0f}")

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

    with open(f'{output_dir}two_stage_qp.json', 'w') as f:
        json.dump(_convert(all_results), f, indent=2)

    # Print summary
    print(f"\n{'='*90}")
    print("TWO-STAGE QP SUMMARY (mean ± std over seeds)")
    print(f"{'='*90}")
    print(f"{'Mode':<25} {'Power Viol':<20} {'CBF Viol':<15} {'Reward':<15}")
    print("-" * 75)

    for mode, desc in modes:
        powers = [all_results[f'seed_{s}'][mode]['power_violation_rate'] for s in range(n_seeds)]
        cbfs = [all_results[f'seed_{s}'][mode]['cbf_violation_rate'] for s in range(n_seeds)]
        rewards = [all_results[f'seed_{s}'][mode]['reward'] for s in range(n_seeds)]

        print(f"{desc:<25} {np.mean(powers)*100:.1f}±{np.std(powers)*100:.1f}%     "
              f"{np.mean(cbfs)*100:.1f}%       "
              f"{np.mean(rewards):.0f}±{np.std(rewards):.0f}")

    print(f"{'='*90}")

    # LaTeX table
    print(f"\n--- LaTeX Table ---")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Power Constraint Violation under S2: Pressure with Two-Stage QP (mean $\pm$ std over 5 seeds, 30 episodes $\times$ 300 steps each). All methods use the Robust HOCBF safety filter.}")
    print(r"\label{tab:power_two_stage}")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"Strategy & Power & CBF & Pressure & Enthalpy & Reward \\")
    print(r"\midrule")

    for mode, desc in modes:
        powers = [all_results[f'seed_{s}'][mode]['power_violation_rate'] for s in range(n_seeds)]
        cbfs = [all_results[f'seed_{s}'][mode]['cbf_violation_rate'] for s in range(n_seeds)]
        pressures = [all_results[f'seed_{s}'][mode]['pressure_violation_rate'] for s in range(n_seeds)]
        enthalpies = [all_results[f'seed_{s}'][mode]['enthalpy_violation_rate'] for s in range(n_seeds)]
        rewards = [all_results[f'seed_{s}'][mode]['reward'] for s in range(n_seeds)]

        power_str = f"${np.mean(powers)*100:.1f}{{\\pm}}{np.std(powers)*100:.1f}\\%$"
        cbf_str = f"${np.mean(cbfs)*100:.1f}\\%$"
        press_str = f"${np.mean(pressures)*100:.1f}\\%$"
        ent_str = f"${np.mean(enthalpies)*100:.1f}\\%$"
        rew_str = f"${np.mean(rewards):.0f}{{\\pm}}{np.std(rewards):.0f}$"

        label = desc.replace('(', '').replace(')', '').strip()
        print(f"{label} & {power_str} & {cbf_str} & {press_str} & {ent_str} & {rew_str} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--scenario', type=str, default='pressure_oscillation')
    parser.add_argument('--n_eval_episodes', type=int, default=30)
    parser.add_argument('--n_eval_steps', type=int, default=300)
    args = parser.parse_args()

    results = run_two_stage_experiment(
        n_seeds=args.n_seeds,
        scenario=args.scenario,
        n_eval_episodes=args.n_eval_episodes,
        n_eval_steps=args.n_eval_steps,
    )

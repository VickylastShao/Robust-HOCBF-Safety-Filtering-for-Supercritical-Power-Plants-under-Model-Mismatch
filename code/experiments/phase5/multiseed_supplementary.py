"""Supplementary multi-seed experiments for PPO-HOCBF and PPO-GP-HOCBF.

C4 fix: PPO-HOCBF and PPO-GP-HOCBF were originally run with 1 seed;
other methods have 5 seeds. This script runs the missing seeds.

Target:
  - PPO-HOCBF: 5 seeds × 6 conditions (missing ~23 runs)
  - PPO-GP-HOCBF: 5 seeds × S1:Heat (missing 4 runs)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np
import yaml
import flax.nnx as nnx
import json

from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.gp.gp_residual import GPResidual
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from experiments.phase4.methods import (
    METHODS, METHOD_LABELS,
    _make_ccs_env, _make_hocbf, _make_robust_hocbf, _pretrain_gp,
    _rollout_with_qp, _rollout_no_qp, _collect_gp_data, _count_violations,
)
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints

CONDITIONS = ['nominal', 's1_heat', 's2_pressure', 's3_coupled',
              's4_nonlinear', 'load_following']

CONDITION_SCENARIO_MAP = {
    'nominal': None,
    's1_heat': 'heat_absorption',
    's2_pressure': 'pressure_oscillation',
    's3_coupled': 'coupled',
    's4_nonlinear': 'nonlinear',
    'load_following': None,
}

# CBF-protected constraint names
CBF_PROTECTED = {'pressure_low', 'pressure_high', 'enthalpy_low', 'enthalpy_high'}

N_SEEDS = 5
RESULTS_DIR = 'results/phase4/'


def load_config():
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'phase4.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_single(method_name, condition, seed, config):
    """Run one experiment and return metrics."""
    method_cfg = config.get('methods_config', {}).get(method_name, {})
    hocbf_cfg = config.get('hocbf', {})
    gp_cfg = config.get('gp', {})
    train_cfg = config.get('training', {})
    eval_cfg = config.get('evaluation', {})

    load_ratio = 1.0
    delay_order = 0
    scenario = CONDITION_SCENARIO_MAP.get(condition)

    # Setup dynamics and constraint
    if scenario is not None:
        dynamics, constraint = _make_ccs_env(load_ratio, delay_order, scenario)
    else:
        dynamics, constraint = _make_ccs_env(load_ratio, delay_order)

    x0, u0 = dynamics.equilibrium(load_ratio)
    key = jax.random.key(seed)

    # Pre-train GP if needed
    gp = None
    if method_name == 'ppo_gp_hocbf':
        use_scenario_gp = method_cfg.get('scenario_specific_gp', False)
        if use_scenario_gp:
            gp = _pretrain_gp(
                load_ratio, delay_order,
                n_pretrain=method_cfg.get('n_pretrain', gp_cfg.get('n_pretrain', 3000)),
                key=key,
                sigma_floor=method_cfg.get('sigma_floor', gp_cfg.get('sigma_floor', None)),
                scenario=scenario, scenario_specific=True)
        else:
            gp = _pretrain_gp(
                load_ratio, delay_order,
                n_pretrain=method_cfg.get('n_pretrain', gp_cfg.get('n_pretrain', 3000)),
                key=key,
                sigma_floor=method_cfg.get('sigma_floor', gp_cfg.get('sigma_floor', None)))

    # Initialize method
    train_fn = METHODS[method_name]
    model, trainer, safety_layer = train_fn(
        method_cfg, dynamics, constraint, key, gp=gp)

    qp_solver = DifferentiableQP(v_max=5.0)

    # Training loop (decoupled: PPO without QP)
    n_episodes = train_cfg.get('max_episodes', 200)
    n_steps = train_cfg.get('n_steps', 200)
    reward_history = []

    for ep in range(n_episodes):
        # Use deployment scenario for training dynamics
        if scenario is not None:
            train_dyn = UncertainUSCCSDynamics(
                delay_order=delay_order, load_ratio=load_ratio,
                uncertainty_scenario=scenario)
        else:
            train_dyn = dynamics

        key, rollout_key = jax.random.split(key)
        rollout, ep_reward, _, _, _ = _rollout_no_qp(
            model, train_dyn, constraint, x0, u0, rollout_key, n_steps=n_steps)

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
            avg_r = np.mean(reward_history[-50:]) if len(reward_history) >= 50 else np.mean(reward_history)
            print(f"  Ep {ep+1}: r={ep_reward:.1f} (avg50={avg_r:.1f})", flush=True)

    # Evaluation with QP safety filter
    n_eval = eval_cfg.get('n_episodes', 50)
    n_eval_steps = eval_cfg.get('n_steps', 500)
    violation_rates = []
    cbf_violation_rates = []
    rewards = []

    for ep in range(n_eval):
        key, ep_key = jax.random.split(key)
        x = x0
        violations = 0
        cbf_violations = 0
        ep_reward = 0.0

        for t in range(n_eval_steps):
            key, action_key = jax.random.split(key)
            v_rl, _, _ = model.get_action(x[:3], action_key)

            if safety_layer is not None:
                A, b = safety_layer.qp_matrices(x[:3])
                v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
            else:
                v_safe = v_rl

            v_safe = jnp.clip(v_safe, -5.0, 5.0)
            next_x = dynamics.step_stabilized(x[:3], v_safe)
            u_total = dynamics.compute_total_control(x[:3], v_safe)
            constraint_vals = constraint.check_all(next_x, u_total)

            if _count_violations(constraint_vals, protected_only=False):
                violations += 1
            if _count_violations(constraint_vals, protected_only=True):
                cbf_violations += 1

            y = dynamics.output(next_x, u_total)
            y0 = dynamics.output(x0, u0)
            reward = (
                -1.0 * (y[0] - y0[0]) ** 2
                - 0.001 * (y[1] - y0[1]) ** 2
                - 0.01 * (y[2] - y0[2]) ** 2
                - 0.0001 * jnp.sum(v_safe ** 2)
            )
            ep_reward += float(reward)
            x = next_x

        violation_rates.append(violations / n_eval_steps)
        cbf_violation_rates.append(cbf_violations / n_eval_steps)
        rewards.append(ep_reward)

    return {
        'violation_rate': (float(np.mean(violation_rates)), float(np.std(violation_rates))),
        'cbf_violation_rate': (float(np.mean(cbf_violation_rates)), float(np.std(cbf_violation_rates))),
        'cumulative_reward': (float(np.mean(rewards)), float(np.std(rewards))),
    }


def save_result(result, method_name, condition, seed):
    """Save result to JSON."""
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    path = Path(RESULTS_DIR) / f'{method_name}_{condition}_seed{seed}.json'

    def _convert(obj):
        if isinstance(obj, (jnp.ndarray,)):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(path, 'w') as f:
        json.dump(_convert(result), f, indent=2)


def main():
    config = load_config()

    # Determine which runs are needed
    runs_needed = []

    # PPO-HOCBF: 5 seeds × 6 conditions
    for condition in CONDITIONS:
        for seed in range(N_SEEDS):
            path = Path(RESULTS_DIR) / f'ppo_hocbf_{condition}_seed{seed}.json'
            if not path.exists():
                runs_needed.append(('ppo_hocbf', condition, seed))

    # PPO-GP-HOCBF: 5 seeds × S1:Heat
    for seed in range(N_SEEDS):
        path = Path(RESULTS_DIR) / f'ppo_gp_hocbf_s1_heat_seed{seed}.json'
        if not path.exists():
            runs_needed.append(('ppo_gp_hocbf', 's1_heat', seed))

    print(f"Total runs needed: {len(runs_needed)}")
    for method, cond, seed in runs_needed:
        print(f"  {method} | {cond} | seed={seed}")

    # Run experiments
    completed = 0
    failed = 0
    for method_name, condition, seed in runs_needed:
        label = METHOD_LABELS.get(method_name, method_name)
        print(f"\n{'='*60}")
        print(f"[{completed+1}/{len(runs_needed)}] {label} | {condition} | seed={seed}")
        print(f"{'='*60}")

        t_start = time.time()
        try:
            result = run_single(method_name, condition, seed, config)
            save_result(result, method_name, condition, seed)
            elapsed = time.time() - t_start
            vr = result['violation_rate']
            cbf_vr = result['cbf_violation_rate']
            print(f"  → total_viol={vr[0]:.4f}±{vr[1]:.4f}, "
                  f"cbf_viol={cbf_vr[0]:.4f}±{cbf_vr[1]:.4f}, "
                  f"reward={result['cumulative_reward'][0]:.1f}, "
                  f"time={elapsed:.1f}s")
            completed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n=== Done: {completed} completed, {failed} failed ===")


if __name__ == "__main__":
    main()

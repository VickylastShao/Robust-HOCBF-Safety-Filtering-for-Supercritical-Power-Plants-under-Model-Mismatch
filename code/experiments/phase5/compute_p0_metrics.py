"""P0-3 & P0-4: Compute physical violation rates and QP intervention rates.

This script runs targeted evaluation on key methods and conditions to collect:
1. Physical constraint violation rates per constraint type (pressure, enthalpy, power)
2. CBF violation rates (pressure + enthalpy only)
3. QP intervention rate (fraction of steps where u_safe ≠ u_rl)

Only runs the most critical method-scenario combinations:
- S1:Heat (core ablation): PPO-HOCBF, PPO-GP-HOCBF, PPO-RHOCBF, RoCBF-Net
- S2:Pressure (measurement-only): PPO-RHOCBF, RoCBF-Net
- Nominal: PPO-RHOCBF, RoCBF-Net (baseline)

Total: ~4 methods × 2-3 conditions × 5 seeds = ~40-60 runs (not 240)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np
import yaml
import json

from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.gp.gp_residual import GPResidual
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from experiments.phase4.methods import (
    METHODS, METHOD_LABELS,
    _make_ccs_env, _make_robust_hocbf, _pretrain_gp,
    _rollout_no_qp, _collect_gp_data,
)
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints

CONDITION_SCENARIO_MAP = {
    'nominal': None,
    's1_heat': 'heat_absorption',
    's2_pressure': 'pressure_oscillation',
    's3_coupled': 'coupled',
    's4_nonlinear': 'nonlinear',
}

# CBF-protected constraint names
CBF_PROTECTED = {'pressure_low', 'pressure_high', 'enthalpy_low', 'enthalpy_high'}

# Key methods to evaluate (skip PPO, PPO-Lagr, NMPC, PPO-CBF as they don't use QP filter)
KEY_METHODS = ['ppo_hocbf', 'ppo_gp_hocbf', 'ppo_rhocbf', 'rocbf_net']
KEY_CONDITIONS = ['s1_heat', 's2_pressure']  # Core conditions for P0-3/P0-4

N_SEEDS = 5
N_EVAL_EPISODES = 50
N_EVAL_STEPS = 500
N_TRAIN_EPISODES = 200
N_TRAIN_STEPS = 200

RESULTS_DIR = 'results/p0_metrics/'


def load_config():
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'phase4.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate_with_intervention(model, dynamics, multi_hocbf, qp_solver, constraint,
                               x0, u0, key, n_steps=500, jit_qp_fn=None):
    """Rollout with QP safety filter, tracking u_rl vs u_safe for intervention rate.

    Returns detailed per-step metrics including intervention indicator and
    per-constraint violation breakdown.
    """
    x = x0
    total_reward = 0.0

    # Per-step tracking
    intervention_steps = 0  # steps where u_safe ≠ u_rl
    total_steps = 0

    # Per-constraint-type violation tracking
    per_type = {
        'pressure': {'physical_violations': 0, 'cbf_violations': 0, 'steps': 0},
        'enthalpy': {'physical_violations': 0, 'cbf_violations': 0, 'steps': 0},
        'power':    {'physical_violations': 0, 'cbf_violations': 0, 'steps': 0},
    }

    # Overall tracking
    any_physical_violation = 0
    any_cbf_violation = 0

    for t in range(n_steps):
        key, action_key = jax.random.split(key)
        v_rl, _, _ = model.get_action(x[:3], action_key)

        # QP safety filter
        if jit_qp_fn is not None:
            A, b = jit_qp_fn(x[:3])
        else:
            A, b = multi_hocbf.qp_matrices(x[:3])

        v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
        v_safe = jnp.clip(v_safe, -5.0, 5.0)

        # Check if QP intervened
        if not jnp.allclose(v_rl, v_safe, atol=1e-4):
            intervention_steps += 1
        total_steps += 1

        # Step dynamics
        next_x = dynamics.step_stabilized(x[:3], v_safe)
        u_total = dynamics.compute_total_control(x[:3], v_safe)

        # Check constraints
        constraint_vals = constraint.check_all(next_x, u_total)

        # Per-constraint-type violations
        any_physical = False
        any_cbf = False
        for ctype in ('pressure', 'enthalpy', 'power'):
            type_keys = [k for k in constraint_vals if ctype in k]
            physical_viol = any(constraint_vals[k] < 0 for k in type_keys)
            cbf_viol = any(constraint_vals[k] < 0 for k in type_keys if k in CBF_PROTECTED)
            if physical_viol:
                per_type[ctype]['physical_violations'] += 1
                any_physical = True
            if cbf_viol:
                per_type[ctype]['cbf_violations'] += 1
                any_cbf = True
            per_type[ctype]['steps'] += 1

        if any_physical:
            any_physical_violation += 1
        if any_cbf:
            any_cbf_violation += 1

        # Reward
        y = dynamics.output(next_x, u_total)
        y0 = dynamics.output(x0, u0)
        reward = (
            -1.0 * (y[0] - y0[0]) ** 2
            - 0.001 * (y[1] - y0[1]) ** 2
            - 0.01 * (y[2] - y0[2]) ** 2
            - 0.0001 * jnp.sum(v_safe ** 2)
        )
        total_reward += float(reward)
        x = next_x

    return {
        'intervention_rate': intervention_steps / max(total_steps, 1),
        'intervention_steps': intervention_steps,
        'total_steps': total_steps,
        'physical_violation_rate': any_physical_violation / max(total_steps, 1),
        'cbf_violation_rate': any_cbf_violation / max(total_steps, 1),
        'per_constraint_type': {
            ctype: {
                'physical_violation_rate': info['physical_violations'] / max(info['steps'], 1),
                'cbf_violation_rate': info['cbf_violations'] / max(info['steps'], 1),
                'physical_violation_count': info['physical_violations'],
                'cbf_violation_count': info['cbf_violations'],
                'total_steps': info['steps'],
            }
            for ctype, info in per_type.items()
        },
        'cumulative_reward': total_reward,
    }


def evaluate_no_qp(model, dynamics, constraint, x0, u0, key, n_steps=500):
    """Rollout without QP filter, tracking constraint violations."""
    x = x0
    total_reward = 0.0
    total_steps = 0

    per_type = {
        'pressure': {'physical_violations': 0, 'steps': 0},
        'enthalpy': {'physical_violations': 0, 'steps': 0},
        'power':    {'physical_violations': 0, 'steps': 0},
    }
    any_physical_violation = 0
    any_cbf_violation = 0

    for t in range(n_steps):
        key, action_key = jax.random.split(key)
        v_rl, _, _ = model.get_action(x[:3], action_key)

        next_x = dynamics.step_stabilized(x[:3], v_rl)
        u_total = dynamics.compute_total_control(x[:3], v_rl)

        constraint_vals = constraint.check_all(next_x, u_total)

        any_physical = False
        any_cbf = False
        for ctype in ('pressure', 'enthalpy', 'power'):
            type_keys = [k for k in constraint_vals if ctype in k]
            physical_viol = any(constraint_vals[k] < 0 for k in type_keys)
            cbf_viol = any(constraint_vals[k] < 0 for k in type_keys if k in CBF_PROTECTED)
            if physical_viol:
                per_type[ctype]['physical_violations'] += 1
                any_physical = True
            if cbf_viol:
                any_cbf = True
            per_type[ctype]['steps'] += 1

        if any_physical:
            any_physical_violation += 1
        if any_cbf:
            any_cbf_violation += 1

        y = dynamics.output(next_x, u_total)
        y0 = dynamics.output(x0, u0)
        reward = (
            -1.0 * (y[0] - y0[0]) ** 2
            - 0.001 * (y[1] - y0[1]) ** 2
            - 0.01 * (y[2] - y0[2]) ** 2
            - 0.0001 * jnp.sum(v_rl ** 2)
        )
        total_reward += float(reward)
        total_steps += 1
        x = next_x

    return {
        'physical_violation_rate': any_physical_violation / max(total_steps, 1),
        'cbf_violation_rate': any_cbf_violation / max(total_steps, 1),
        'per_constraint_type': {
            ctype: {
                'physical_violation_rate': info['physical_violations'] / max(info['steps'], 1),
                'physical_violation_count': info['physical_violations'],
                'total_steps': info['steps'],
            }
            for ctype, info in per_type.items()
        },
        'cumulative_reward': total_reward,
    }


def run_single(method_name, condition, seed, config):
    """Train + evaluate a single method-condition-seed combination."""
    method_cfg = config.get('methods_config', {}).get(method_name, {})
    gp_cfg = config.get('gp', {})

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
    if method_name in ('ppo_gp_hocbf', 'ppo_rhocbf', 'rocbf_net'):
        use_scenario_gp = method_cfg.get('scenario_specific_gp', method_name == 'rocbf_net')
        if use_scenario_gp:
            gp = _pretrain_gp(
                load_ratio, delay_order,
                n_pretrain=method_cfg.get('n_pretrain', gp_cfg.get('n_pretrain', 2000)),
                key=key,
                sigma_floor=method_cfg.get('sigma_floor', gp_cfg.get('sigma_floor', None)),
                scenario=scenario, scenario_specific=True)
        else:
            gp = _pretrain_gp(
                load_ratio, delay_order,
                n_pretrain=method_cfg.get('n_pretrain', gp_cfg.get('n_pretrain', 2000)),
                key=key,
                sigma_floor=method_cfg.get('sigma_floor', gp_cfg.get('sigma_floor', None)))

    # Initialize method
    train_fn = METHODS[method_name]
    model, trainer, safety_layer = train_fn(
        method_cfg, dynamics, constraint, key, gp=gp)

    qp_solver = DifferentiableQP(v_max=5.0)
    has_qp = safety_layer is not None

    # Training loop (same as run_experiment.py)
    for ep in range(N_TRAIN_EPISODES):
        if scenario is not None:
            train_dyn = UncertainUSCCSDynamics(
                delay_order=delay_order, load_ratio=load_ratio,
                uncertainty_scenario=scenario)
        else:
            train_dyn = dynamics

        key, rollout_key = jax.random.split(key)
        rollout, ep_reward, _, _, _ = _rollout_no_qp(
            model, train_dyn, constraint, x0, u0, rollout_key, n_steps=N_TRAIN_STEPS)

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

        # Online GP update for RoCBF-Net
        if method_name == 'rocbf_net' and (ep + 1) % 50 == 0 and gp is not None:
            key, gp_key = jax.random.split(key)
            env_gp = UncertainUSCCSDynamics(
                delay_order=delay_order, load_ratio=load_ratio,
                uncertainty_scenario=scenario)
            key, data_key = jax.random.split(gp_key)
            X_new, Y_new = _collect_gp_data(env_gp, 200, key=data_key)
            gp.incremental_update(X_new, Y_new, reoptimize_hyperparams=False)
            # Rebuild safety layer
            safety_layer = _make_robust_hocbf(
                dynamics, constraint, gp, u0,
                epsilon_kappa=method_cfg.get('epsilon_kappa', 1.0),
                k_pressure=tuple(method_cfg.get('pressure_k_gains', (0.5, 0.5))),
                k_enthalpy=tuple(method_cfg.get('enthalpy_k_gains', (1.0,))),
                u_max=method_cfg.get('u_max', 100.0),
                use_mean_correction=method_cfg.get('use_mean_correction', True),
                epsilon_floor=method_cfg.get('epsilon_floor', 0.0))

        if (ep + 1) % 50 == 0:
            print(f"    Ep {ep+1}: r={ep_reward:.1f}", flush=True)

    # JIT-compile QP matrices function
    jit_qp_fn = None
    if has_qp:
        try:
            jit_qp_fn = jax.jit(safety_layer.qp_matrices)
            _ = jit_qp_fn(x0[:3])  # warm up
        except Exception:
            jit_qp_fn = None

    # Evaluation: collect metrics across episodes
    all_results = []
    for ep in range(N_EVAL_EPISODES):
        key, ep_key = jax.random.split(key)
        if has_qp:
            result = evaluate_with_intervention(
                model, dynamics, safety_layer, qp_solver, constraint,
                x0, u0, ep_key, n_steps=N_EVAL_STEPS, jit_qp_fn=jit_qp_fn)
        else:
            result = evaluate_no_qp(
                model, dynamics, constraint, x0, u0, ep_key, n_steps=N_EVAL_STEPS)
        all_results.append(result)

    # Aggregate across episodes
    aggregated = aggregate_results(all_results, has_qp)
    return aggregated


def aggregate_results(results, has_qp):
    """Aggregate per-episode results into summary statistics."""
    if has_qp:
        intervention_rates = [r['intervention_rate'] for r in results]
        physical_rates = [r['physical_violation_rate'] for r in results]
        cbf_rates = [r['cbf_violation_rate'] for r in results]
        rewards = [r['cumulative_reward'] for r in results]

        per_type_agg = {}
        for ctype in ('pressure', 'enthalpy', 'power'):
            phys_rates = [r['per_constraint_type'][ctype]['physical_violation_rate'] for r in results]
            cbf_v_rates = [r['per_constraint_type'][ctype].get('cbf_violation_rate', 0) for r in results]
            per_type_agg[ctype] = {
                'physical_violation_rate': (float(np.mean(phys_rates)), float(np.std(phys_rates))),
                'cbf_violation_rate': (float(np.mean(cbf_v_rates)), float(np.std(cbf_v_rates))),
            }

        return {
            'intervention_rate': (float(np.mean(intervention_rates)), float(np.std(intervention_rates))),
            'physical_violation_rate': (float(np.mean(physical_rates)), float(np.std(physical_rates))),
            'cbf_violation_rate': (float(np.mean(cbf_rates)), float(np.std(cbf_rates))),
            'cumulative_reward': (float(np.mean(rewards)), float(np.std(rewards))),
            'per_constraint_type': per_type_agg,
        }
    else:
        physical_rates = [r['physical_violation_rate'] for r in results]
        cbf_rates = [r['cbf_violation_rate'] for r in results]
        rewards = [r['cumulative_reward'] for r in results]

        per_type_agg = {}
        for ctype in ('pressure', 'enthalpy', 'power'):
            phys_rates = [r['per_constraint_type'][ctype]['physical_violation_rate'] for r in results]
            per_type_agg[ctype] = {
                'physical_violation_rate': (float(np.mean(phys_rates)), float(np.std(phys_rates))),
            }

        return {
            'physical_violation_rate': (float(np.mean(physical_rates)), float(np.std(physical_rates))),
            'cbf_violation_rate': (float(np.mean(cbf_rates)), float(np.std(cbf_rates))),
            'cumulative_reward': (float(np.mean(rewards)), float(np.std(rewards))),
            'per_constraint_type': per_type_agg,
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

    # Build list of runs needed
    runs = []
    for method_name in KEY_METHODS:
        for condition in KEY_CONDITIONS:
            for seed in range(N_SEEDS):
                path = Path(RESULTS_DIR) / f'{method_name}_{condition}_seed{seed}.json'
                if not path.exists():
                    runs.append((method_name, condition, seed))

    print(f"Total runs needed: {len(runs)}")
    for method, cond, seed in runs:
        label = METHOD_LABELS.get(method, method)
        print(f"  {label} | {cond} | seed={seed}")

    # Run experiments
    completed = 0
    failed = 0
    all_results = {}  # (method, condition) -> list of seed results

    for method_name, condition, seed in runs:
        label = METHOD_LABELS.get(method_name, method_name)
        print(f"\n{'='*60}")
        print(f"[{completed+1}/{len(runs)}] {label} | {condition} | seed={seed}")
        print(f"{'='*60}")

        t_start = time.time()
        try:
            result = run_single(method_name, condition, seed, config)
            save_result(result, method_name, condition, seed)
            elapsed = time.time() - t_start

            key = (method_name, condition)
            if key not in all_results:
                all_results[key] = []
            all_results[key].append(result)

            # Print summary
            if 'intervention_rate' in result:
                ir = result['intervention_rate']
                print(f"  → intervention={ir[0]:.4f}±{ir[1]:.4f}", end='')
            pvr = result['physical_violation_rate']
            cvr = result['cbf_violation_rate']
            print(f"  physical_viol={pvr[0]:.4f}±{pvr[1]:.4f}, "
                  f"cbf_viol={cvr[0]:.4f}±{cvr[1]:.4f}, "
                  f"time={elapsed:.1f}s")
            completed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Print aggregated results
    print(f"\n{'='*80}")
    print("AGGREGATED RESULTS")
    print(f"{'='*80}")
    for (method, cond), seed_results in sorted(all_results.items()):
        label = METHOD_LABELS.get(method, method)
        print(f"\n{label} | {cond}:")

        # Aggregate across seeds
        if 'intervention_rate' in seed_results[0]:
            irs = [r['intervention_rate'][0] for r in seed_results]
            print(f"  Intervention rate: {np.mean(irs):.4f} ± {np.std(irs):.4f}")

        pvr = [r['physical_violation_rate'][0] for r in seed_results]
        cvr = [r['cbf_violation_rate'][0] for r in seed_results]
        print(f"  Physical violation: {np.mean(pvr):.4f} ± {np.std(pvr):.4f}")
        print(f"  CBF violation: {np.mean(cvr):.4f} ± {np.std(cvr):.4f}")

        for ctype in ('pressure', 'enthalpy', 'power'):
            pt = [r['per_constraint_type'][ctype]['physical_violation_rate'][0] for r in seed_results]
            print(f"  {ctype} physical: {np.mean(pt):.4f} ± {np.std(pt):.4f}")
            if 'cbf_violation_rate' in seed_results[0]['per_constraint_type'][ctype]:
                ct = [r['per_constraint_type'][ctype]['cbf_violation_rate'][0] for r in seed_results]
                print(f"  {ctype} CBF: {np.mean(ct):.4f} ± {np.std(ct):.4f}")

    print(f"\n=== Done: {completed} completed, {failed} failed ===")


if __name__ == "__main__":
    main()

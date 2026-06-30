"""P0-3 & P0-4: Quick computation of physical violation rates and QP intervention rates.

Optimized: 1 seed, 10 eval episodes, 200 eval steps per method-condition.
Only covers S1:Heat (core scenario where Δf ≠ 0).

Expected runtime: ~5 min per method × 4 methods = ~20 min total.
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

CBF_PROTECTED = {'pressure_low', 'pressure_high', 'enthalpy_low', 'enthalpy_high'}

# Minimal config for quick results
SEED = 42
N_EVAL_EPISODES = 10
N_EVAL_STEPS = 200
N_TRAIN_EPISODES = 200
N_TRAIN_STEPS = 200

# Key method-scenario combos
COMBOS = [
    ('ppo_hocbf', 's1_heat'),
    ('ppo_gp_hocbf', 's1_heat'),
    ('ppo_rhocbf', 's1_heat'),
    ('rocbf_net', 's1_heat'),
    ('ppo_hocbf', 's2_pressure'),
    ('ppo_rhocbf', 's2_pressure'),
    ('rocbf_net', 's2_pressure'),
    ('ppo_rhocbf', 'nominal'),
    ('rocbf_net', 'nominal'),
]

CONDITION_SCENARIO_MAP = {
    'nominal': None,
    's1_heat': 'heat_absorption',
    's2_pressure': 'pressure_oscillation',
}


def load_config():
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'phase4.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate_with_intervention(model, dynamics, multi_hocbf, qp_solver, constraint,
                               x0, u0, key, n_steps=200, jit_qp_fn=None):
    """Rollout with QP safety filter, tracking u_rl vs u_safe."""
    x = x0
    intervention_steps = 0
    total_steps = 0
    per_type = {
        'pressure': {'physical_violations': 0, 'cbf_violations': 0, 'steps': 0},
        'enthalpy': {'physical_violations': 0, 'cbf_violations': 0, 'steps': 0},
        'power':    {'physical_violations': 0, 'cbf_violations': 0, 'steps': 0},
    }
    any_physical_violation = 0
    any_cbf_violation = 0

    for t in range(n_steps):
        key, action_key = jax.random.split(key)
        v_rl, _, _ = model.get_action(x[:3], action_key)

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

        next_x = dynamics.step_stabilized(x[:3], v_safe)
        u_total = dynamics.compute_total_control(x[:3], v_safe)
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
                per_type[ctype]['cbf_violations'] += 1
                any_cbf = True
            per_type[ctype]['steps'] += 1

        if any_physical:
            any_physical_violation += 1
        if any_cbf:
            any_cbf_violation += 1
        x = next_x

    return {
        'intervention_rate': intervention_steps / max(total_steps, 1),
        'physical_violation_rate': any_physical_violation / max(total_steps, 1),
        'cbf_violation_rate': any_cbf_violation / max(total_steps, 1),
        'per_constraint_type': {
            ctype: {
                'physical_violation_rate': info['physical_violations'] / max(info['steps'], 1),
                'cbf_violation_rate': info['cbf_violations'] / max(info['steps'], 1),
            }
            for ctype, info in per_type.items()
        },
    }


def run_single(method_name, condition, config):
    """Train + evaluate a single method-condition combination (1 seed)."""
    method_cfg = config.get('methods_config', {}).get(method_name, {})
    gp_cfg = config.get('gp', {})

    load_ratio = 1.0
    delay_order = 0
    scenario = CONDITION_SCENARIO_MAP.get(condition)

    if scenario is not None:
        dynamics, constraint = _make_ccs_env(load_ratio, delay_order, scenario)
    else:
        dynamics, constraint = _make_ccs_env(load_ratio, delay_order)

    x0, u0 = dynamics.equilibrium(load_ratio)
    key = jax.random.key(SEED)

    # Pre-train GP
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

    # Training loop
    t0 = time.time()
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
            safety_layer = _make_robust_hocbf(
                dynamics, constraint, gp, u0,
                epsilon_kappa=method_cfg.get('epsilon_kappa', 1.0),
                k_pressure=tuple(method_cfg.get('pressure_k_gains', (0.5, 0.5))),
                k_enthalpy=tuple(method_cfg.get('enthalpy_k_gains', (1.0,))),
                u_max=method_cfg.get('u_max', 100.0),
                use_mean_correction=method_cfg.get('use_mean_correction', True),
                epsilon_floor=method_cfg.get('epsilon_floor', 0.0))

    train_time = time.time() - t0

    # JIT-compile QP
    jit_qp_fn = None
    if safety_layer is not None:
        try:
            jit_qp_fn = jax.jit(safety_layer.qp_matrices)
            _ = jit_qp_fn(x0[:3])
        except Exception:
            jit_qp_fn = None

    # Evaluation
    all_results = []
    for ep in range(N_EVAL_EPISODES):
        key, ep_key = jax.random.split(key)
        result = evaluate_with_intervention(
            model, dynamics, safety_layer, qp_solver, constraint,
            x0, u0, ep_key, n_steps=N_EVAL_STEPS, jit_qp_fn=jit_qp_fn)
        all_results.append(result)

    # Aggregate
    intervention_rates = [r['intervention_rate'] for r in all_results]
    physical_rates = [r['physical_violation_rate'] for r in all_results]
    cbf_rates = [r['cbf_violation_rate'] for r in all_results]

    per_type_agg = {}
    for ctype in ('pressure', 'enthalpy', 'power'):
        phys = [r['per_constraint_type'][ctype]['physical_violation_rate'] for r in all_results]
        cbf = [r['per_constraint_type'][ctype]['cbf_violation_rate'] for r in all_results]
        per_type_agg[ctype] = {
            'physical_violation_rate_mean': float(np.mean(phys)),
            'physical_violation_rate_std': float(np.std(phys)),
            'cbf_violation_rate_mean': float(np.mean(cbf)),
            'cbf_violation_rate_std': float(np.std(cbf)),
        }

    return {
        'intervention_rate_mean': float(np.mean(intervention_rates)),
        'intervention_rate_std': float(np.std(intervention_rates)),
        'physical_violation_rate_mean': float(np.mean(physical_rates)),
        'physical_violation_rate_std': float(np.std(physical_rates)),
        'cbf_violation_rate_mean': float(np.mean(cbf_rates)),
        'cbf_violation_rate_std': float(np.std(cbf_rates)),
        'per_constraint_type': per_type_agg,
        'train_time_s': train_time,
    }


def main():
    config = load_config()

    all_results = {}
    for i, (method_name, condition) in enumerate(COMBOS):
        label = METHOD_LABELS.get(method_name, method_name)
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(COMBOS)}] {label} | {condition}")
        print(f"{'='*60}")

        t_start = time.time()
        try:
            result = run_single(method_name, condition, config)
            elapsed = time.time() - t_start

            all_results[(method_name, condition)] = result

            print(f"  → intervention={result['intervention_rate_mean']:.4f}±{result['intervention_rate_std']:.4f}", flush=True)
            print(f"     physical_viol={result['physical_violation_rate_mean']:.4f}±{result['physical_violation_rate_std']:.4f}", flush=True)
            print(f"     cbf_viol={result['cbf_violation_rate_mean']:.4f}±{result['cbf_violation_rate_std']:.4f}", flush=True)
            for ctype in ('pressure', 'enthalpy', 'power'):
                pt = result['per_constraint_type'][ctype]
                print(f"     {ctype}: physical={pt['physical_violation_rate_mean']:.4f}, cbf={pt['cbf_violation_rate_mean']:.4f}", flush=True)
            print(f"     time={elapsed:.1f}s (train={result['train_time_s']:.1f}s)", flush=True)

        except Exception as e:
            print(f"  ✗ FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()

    # Summary table
    print(f"\n{'='*100}")
    print("SUMMARY TABLE")
    print(f"{'='*100}")
    print(f"{'Method':<20} {'Condition':<15} {'Interv.':<10} {'Phys.Viol':<12} {'CBF Viol':<12} {'P(press)':<10} {'P(enthal)':<10} {'P(power)':<10}")
    print("-" * 100)
    for (method_name, condition), result in sorted(all_results.items()):
        label = METHOD_LABELS.get(method_name, method_name)
        ir = result['intervention_rate_mean']
        pv = result['physical_violation_rate_mean']
        cv = result['cbf_violation_rate_mean']
        pp = result['per_constraint_type']['pressure']['physical_violation_rate_mean']
        pe = result['per_constraint_type']['enthalpy']['physical_violation_rate_mean']
        pw = result['per_constraint_type']['power']['physical_violation_rate_mean']
        print(f"{label:<20} {condition:<15} {ir:<10.4f} {pv:<12.4f} {cv:<12.4f} {pp:<10.4f} {pe:<10.4f} {pw:<10.4f}")

    # Save results
    results_dir = Path('results/p0_metrics/')
    results_dir.mkdir(parents=True, exist_ok=True)
    output = {}
    for (method_name, condition), result in all_results.items():
        key = f"{method_name}_{condition}"
        output[key] = result
    with open(results_dir / 'quick_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_dir / 'quick_results.json'}")


if __name__ == "__main__":
    main()

"""Compute P0 metrics on 5th-order CCS model.

This script evaluates the core safety comparison on the 5th-order CCS model
where ALL constraints (including power) are CBF-enforceable (m >= 1).

Key validation points:
1. Power constraint violation drops from ~95% (3rd-order, rd=0) to 0% (5th-order, m=1)
2. CBF violation patterns: PPO-HOCBF catastrophic, PPO-RHOCBF/RoCBF-Net 0%
3. QP intervention rates under different scenarios
4. Per-constraint violation breakdown

Usage:
    conda activate jax_gpu
    cd /home/gpu/sz_workspace/RoCBF-Net
    python experiments/phase5/compute_p0_metrics_5th.py
"""
import json
import os
import sys
import time
import numpy as np

import jax
import jax.numpy as jnp
import flax.nnx as nnx

# Add project root
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.gp.gp_residual import GPResidual
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th

# Import 5th-order methods
from experiments.phase5.methods_5th import (
    NX, SCENARIOS, SCENARIO_LABELS,
    _make_ccs_env_5th, _make_hocbf_5th, _make_robust_hocbf_5th,
    _pretrain_gp_5th, _collect_gp_data_5th,
    _rollout_with_qp_5th, _rollout_no_qp_5th,
    _count_violations_5th, CBF_PROTECTED_5TH,
    METHODS_5TH, METHOD_LABELS,
    train_ppo_5th, train_ppo_hocbf_5th, train_ppo_gp_hocbf_5th,
    train_ppo_rhocbf_5th, train_rocbf_net_5th,
)

# ===== Configuration =====
LOAD_RATIOS = [0.75, 1.0]
CONDITION_MAP = {
    'nominal': None,
    's1_heat': 'heat_absorption',
    's2_pressure': 'pressure_oscillation',
    's3_coupled': 'coupled',
    's4_nonlinear': 'nonlinear',
    's5_valve': 'valve_degradation',
    's6_fuel': 'fuel_quality',
}

# Priority methods for quick validation
PRIORITY_METHODS = ['ppo_hocbf', 'ppo_gp_hocbf', 'ppo_rhocbf', 'rocbf_net']
ALL_METHODS = ['ppo', 'ppo_hocbf', 'ppo_gp_hocbf', 'ppo_rhocbf', 'rocbf_net']

N_STEPS = 300
N_SEEDS = 3
LOAD_RATIO = 0.75  # Primary test condition

RESULTS_DIR = '/home/gpu/sz_workspace/RoCBF-Net/results/p0_metrics_5th'
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_method_5th(method_name, scenario, load_ratio, seed, n_steps=300):
    """Evaluate a single method/scenario/seed combination on 5th-order CCS."""
    key = jax.random.key(seed)
    dynamics, constraint = _make_ccs_env_5th(load_ratio, scenario=scenario)
    x0, u0 = dynamics.equilibrium(load_ratio)

    config = {
        'hidden_dim': 128,
        'lr': 1e-4,
        'epochs': 4,
        'minibatch_size': 64,
        'epsilon_kappa': 1.0,
        'u_max': 100.0,
        'use_mean_correction': True,
        'epsilon_floor': 0.0,
        'scenario': scenario,
        'scenario_specific_gp': True,
    }

    # Pre-train GP for methods that need it
    gp = None
    if method_name in ('ppo_gp_hocbf', 'ppo_rhocbf', 'rocbf_net'):
        gp_key, key = jax.random.split(key)
        gp = _pretrain_gp_5th(load_ratio, n_pretrain=3000, key=gp_key,
                               scenario=scenario, scenario_specific=True)

    # Create model and safety layer
    train_fn = METHODS_5TH[method_name]
    model_key, key = jax.random.split(key)
    model, trainer, safety_layer = train_fn(config, dynamics, constraint, model_key, gp=gp)

    # Train PPO (short training for evaluation)
    print(f"  Training {method_name} (seed={seed})...")
    for ep in range(10):  # 10 episodes of training
        rollout_key, key = jax.random.split(key)
        if safety_layer is not None:
            # With QP safety filter
            qp_solver = DifferentiableQP(v_max=10.0)
            rollout, total_reward, violations, cbf_violations, qp_times = _rollout_with_qp_5th(
                model, dynamics, safety_layer, qp_solver, constraint,
                x0, u0, rollout_key, n_steps=100)
        else:
            rollout, total_reward, violations, cbf_violations, _ = _rollout_no_qp_5th(
                model, dynamics, constraint, x0, u0, rollout_key, n_steps=100)

        # PPO update
        if rollout['obs'].shape[0] > 1:
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

    # ===== Evaluation rollout =====
    eval_key, key = jax.random.split(key)
    per_step_violations = {
        'pressure_high': [], 'pressure_low': [],
        'enthalpy_high': [], 'enthalpy_low': [],
        'power_high': [], 'power_low': [],
    }
    total_violations = 0
    cbf_violations = 0
    total_reward = 0.0
    qp_interventions = 0
    total_steps = 0
    qp_times_eval = []

    x = x0
    for t in range(n_steps):
        eval_key, action_key = jax.random.split(eval_key)
        v_rl, log_prob, value = model.get_action(x[:NX], action_key)

        if safety_layer is not None:
            # QP safety filter
            t0 = time.perf_counter()
            A, b = safety_layer.qp_matrices(x[:NX])
            qp_solver = DifferentiableQP(v_max=10.0)
            v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
            v_safe = jnp.clip(v_safe, -10.0, 10.0)
            qp_time = (time.perf_counter() - t0) * 1000
            qp_times_eval.append(qp_time)

            # Check if QP modified the action
            if jnp.any(jnp.abs(v_safe - v_rl) > 1e-3):
                qp_interventions += 1
        else:
            v_safe = v_rl

        next_x = dynamics.step_stabilized(x[:NX], v_safe)
        constraint_vals = constraint.check_all(next_x)

        # Per-constraint tracking
        for cname in per_step_violations:
            if cname in constraint_vals:
                per_step_violations[cname].append(float(constraint_vals[cname] < 0))

        # Overall violation tracking
        if _count_violations_5th(constraint_vals, protected_only=False):
            total_violations += 1
        if _count_violations_5th(constraint_vals, protected_only=True):
            cbf_violations += 1

        # Reward
        y = dynamics.output(next_x)
        y0 = dynamics.output(x0)
        reward = (
            -1.0 * (y[0] - y0[0]) ** 2
            - 0.001 * (y[1] - y0[1]) ** 2
            - 0.01 * (y[2] - y0[2]) ** 2
            - 0.0001 * jnp.sum(v_safe ** 2)
        )
        total_reward += float(reward)
        total_steps += 1
        x = next_x

    # Compute per-constraint violation rates
    per_constraint_rates = {}
    for cname, vals in per_step_violations.items():
        per_constraint_rates[cname] = sum(vals) / len(vals) * 100 if vals else 0.0

    result = {
        'method': method_name,
        'method_label': METHOD_LABELS[method_name],
        'scenario': scenario,
        'load_ratio': load_ratio,
        'seed': seed,
        'n_steps': n_steps,
        'total_violation_rate': total_violations / total_steps * 100,
        'cbf_violation_rate': cbf_violations / total_steps * 100,
        'total_reward': total_reward,
        'qp_intervention_rate': qp_interventions / total_steps * 100 if safety_layer else 0.0,
        'mean_qp_time_ms': np.mean(qp_times_eval) if qp_times_eval else 0.0,
        'per_constraint_violation': per_constraint_rates,
    }
    return result


def run_priority_experiments():
    """Run priority experiments: 4 methods × 3 conditions × 3 seeds."""
    results = []
    priority_conditions = ['nominal', 's1_heat', 's2_pressure']

    for condition_name in priority_conditions:
        scenario = CONDITION_MAP[condition_name]
        print(f"\n{'='*60}")
        print(f"Condition: {condition_name} (scenario={scenario})")
        print(f"{'='*60}")

        for method_name in PRIORITY_METHODS:
            print(f"\n  Method: {METHOD_LABELS[method_name]}")
            for seed in range(N_SEEDS):
                print(f"    Seed {seed}...", end=" ", flush=True)
                t0 = time.time()
                try:
                    result = evaluate_method_5th(
                        method_name, scenario, LOAD_RATIO, seed, n_steps=N_STEPS)
                    elapsed = time.time() - t0
                    print(f"CBF viol: {result['cbf_violation_rate']:.1f}%, "
                          f"Total viol: {result['total_violation_rate']:.1f}%, "
                          f"QP interv: {result['qp_intervention_rate']:.1f}%, "
                          f"({elapsed:.1f}s)")
                    results.append(result)
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    return results


def run_full_experiments():
    """Run full experiments: 5 methods × 7 conditions × 3 seeds."""
    results = []
    for condition_name, scenario in CONDITION_MAP.items():
        print(f"\n{'='*60}")
        print(f"Condition: {condition_name} (scenario={scenario})")
        print(f"{'='*60}")

        for method_name in ALL_METHODS:
            print(f"\n  Method: {METHOD_LABELS[method_name]}")
            for seed in range(N_SEEDS):
                print(f"    Seed {seed}...", end=" ", flush=True)
                t0 = time.time()
                try:
                    result = evaluate_method_5th(
                        method_name, scenario, LOAD_RATIO, seed, n_steps=N_STEPS)
                    elapsed = time.time() - t0
                    print(f"CBF viol: {result['cbf_violation_rate']:.1f}%, "
                          f"Total viol: {result['total_violation_rate']:.1f}%, "
                          f"Power viol: "
                          f"high={result['per_constraint_violation'].get('power_high', 0):.1f}% "
                          f"low={result['per_constraint_violation'].get('power_low', 0):.1f}%, "
                          f"({elapsed:.1f}s)")
                    results.append(result)
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    return results


def summarize_results(results):
    """Summarize results by method × condition."""
    from collections import defaultdict
    summary = defaultdict(lambda: defaultdict(list))

    for r in results:
        key = (r['method'], r['scenario'])
        summary[key]['cbf_violation'].append(r['cbf_violation_rate'])
        summary[key]['total_violation'].append(r['total_violation_rate'])
        summary[key]['qp_intervention'].append(r['qp_intervention_rate'])
        summary[key]['reward'].append(r['total_reward'])
        for cname in r['per_constraint_violation']:
            summary[key][f'viol_{cname}'].append(r['per_constraint_violation'][cname])

    # Print summary table
    print("\n" + "="*80)
    print("5TH-ORDER CCS RESULTS SUMMARY")
    print("="*80)

    header = f"{'Method':<20} {'Condition':<15} {'CBF Viol%':>10} {'Total Viol%':>12} {'QP Int%':>10} {'Power Hi%':>10} {'Power Lo%':>10}"
    print(header)
    print("-"*80)

    for method_name in ALL_METHODS:
        for condition_name, scenario in CONDITION_MAP.items():
            key = (method_name, scenario)
            if key in summary:
                cbf = np.mean(summary[key]['cbf_violation'])
                total = np.mean(summary[key]['total_violation'])
                qp = np.mean(summary[key]['qp_intervention'])
                p_hi = np.mean(summary[key].get('viol_power_high', [0]))
                p_lo = np.mean(summary[key].get('viol_power_low', [0]))
                print(f"{METHOD_LABELS[method_name]:<20} {condition_name:<15} "
                      f"{cbf:>10.1f} {total:>12.1f} {qp:>10.1f} {p_hi:>10.1f} {p_lo:>10.1f}")

    return dict(summary)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Run full experiments (all methods × all conditions)')
    parser.add_argument('--priority', action='store_true', help='Run priority experiments (4 methods × 3 conditions)')
    args = parser.parse_args()

    if args.full:
        results = run_full_experiments()
    elif args.priority:
        results = run_priority_experiments()
    else:
        # Default: run priority experiments
        print("Running priority experiments (4 methods × 3 conditions × 3 seeds)")
        print("Use --full for all methods × all conditions")
        results = run_priority_experiments()

    # Save raw results
    results_file = os.path.join(RESULTS_DIR, 'quick_results_5th.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to {results_file}")

    # Summarize
    summary = summarize_results(results)

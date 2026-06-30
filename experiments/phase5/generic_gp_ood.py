"""Phase 5 (revision): Generic GP and OOD deployment experiment.

Motivation
----------
The main experiments (Table~\\ref{tab:cbf_violation}) train the GP on data from
the *deployment scenario* (scenario-specific GP). A reviewer concern is that
this requires knowing the deployment environment a priori, and that a generic
GP (trained on nominal or mixed data) may not provide sufficient coverage
under OOD perturbation structures.

This script evaluates three GP-training strategies under all Φ-scaled
scenarios, providing the data for a new table or paragraph that explicitly
addresses the generic-GP question:

  (a) Scenario-specific GP  — trained on deployment-scenario data (main result)
  (b) Generic (mixed) GP    — trained on Nominal + S1 + S2 + S4 data
  (c) Mismatched (OOD) GP   — trained on S1 data, deployed under S3 (and vice versa)

Each configuration is evaluated over 5 seeds × 500 steps with the same
PPO-RHOCBF pipeline (epsilon_kappa=1.0, Φ-scaled rollout).

Expected outcomes (based on existing ablation data):
  - Scenario-specific GP: 0% CBF violation on all scenarios (main result)
  - Generic GP: 0% on S1/S2/S4 (well-covered), higher violation on S3 (OOD)
  - Mismatched GP: elevated violation due to incorrect mean correction,
    but ε(x) should still provide *some* protection if σ_GP is large enough

Usage
-----
    conda activate jax_gpu
    cd /home/gpu/sz_workspace/RoCBF-Net
    PYTHONPATH=. python experiments/phase5/generic_gp_ood.py \\
        --seeds 0 1 2 3 4 --n_eval 500

    # Quick test:
    PYTHONPATH=. python experiments/phase5/generic_gp_ood.py \\
        --seeds 0 --n_eval 50 --conditions S1 S3

Outputs
-------
    results/p0_metrics_5th_phi_scaled/generic_gp_ood.json
    Console summary with Wilson CIs
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.80')
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_enable_command_buffer=')

sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.methods_5th import (
    NX, _make_robust_hocbf_5th, _pretrain_gp_5th,
    _rollout_no_qp_5th, _count_violations_5th,
)

LOAD_RATIO = 1.0
N_TRAIN = 30
N_GP_PRETRAIN = 3000
N_EVAL = 500
RESULTS_DIR = '/home/gpu/sz_workspace/RoCBF-Net/results/p0_metrics_5th_phi_scaled'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Deployment conditions
DEPLOY_CONDITIONS = {
    'Nominal': None,
    'S1:Heat': 'heat_absorption',
    'S2:Pressure': 'pressure_oscillation',
    'S3:Coupled': 'coupled',
    'S4:Nonlinear': 'nonlinear',
    'S5:Valve': 'valve_degradation',
    'S6:Fuel': 'fuel_quality',
}

# GP training strategies
GP_STRATEGIES = {
    'scenario_specific': 'Scenario-specific GP (trained on deployment data)',
    'generic_mixed': 'Generic GP (trained on Nominal+S1+S2+S4 data)',
    'ood_s1_to_s3': 'OOD GP (trained on S1, deployed under S3)',
    'ood_s3_to_s1': 'OOD GP (trained on S3, deployed under S1)',
}


def _build_gp(strategy, deploy_scenario, load_ratio, n_pretrain, seed):
    """Construct a GP according to the specified training strategy.

    Returns
    -------
    gp : GPResidual
    gp_type : str — human-readable label
    """
    key = jax.random.key(seed * 100 + 7)

    if strategy == 'scenario_specific':
        gp = _pretrain_gp_5th(
            load_ratio, n_pretrain=n_pretrain, key=key,
            scenario=deploy_scenario, scenario_specific=True)
        return gp, 'scenario_specific'

    elif strategy == 'generic_mixed':
        # Train on Nominal + S1 + S2 + S4 (no S3, S5, S6)
        gp = _pretrain_gp_5th(
            load_ratio, n_pretrain=n_pretrain, key=key,
            scenario_specific=False)  # uses SCENARIOS[:4]
        return gp, 'generic_mixed'

    elif strategy == 'ood_s1_to_s3':
        # Train on S1 data, deploy under S3
        gp = _pretrain_gp_5th(
            load_ratio, n_pretrain=n_pretrain, key=key,
            scenario='heat_absorption', scenario_specific=True)
        return gp, 'ood_s1_to_s3'

    elif strategy == 'ood_s3_to_s1':
        # Train on S3 data, deploy under S1
        gp = _pretrain_gp_5th(
            load_ratio, n_pretrain=n_pretrain, key=key,
            scenario='coupled', scenario_specific=True)
        return gp, 'ood_s3_to_s1'

    else:
        raise ValueError(f"Unknown GP strategy: {strategy}")


def evaluate_gp_strategy(strategy, deploy_scenario, seed, n_train, n_eval,
                         load_ratio, n_gp_pretrain):
    """Evaluate PPO-RHOCBF with a specific GP training strategy."""
    dynamics = USCCSDynamics5th(load_ratio=load_ratio)
    x0, u0 = dynamics.equilibrium(load_ratio)
    constraint = CCSConstraints5th(
        p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
        power_deviation=50.0, power_target=load_ratio * 1000.0)

    model = ActorCritic(n_obs=NX, n_act=3, hidden_dim=128, rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=1e-4, epochs=4, minibatch_size=64)

    # Build GP according to strategy
    gp, gp_type = _build_gp(strategy, deploy_scenario, load_ratio,
                             n_gp_pretrain, seed)

    # Build Robust HOCBF with ε(x), κ=1.0
    safety_layer = _make_robust_hocbf_5th(
        dynamics, constraint, gp, u0,
        epsilon_kappa=1.0, use_mean_correction=True, use_phi_scaled_g=True)

    # Decoupled PPO training (no QP)
    key = jax.random.key(seed)
    for ep in range(n_train):
        key, rk = jax.random.split(key)
        rollout, ep_r, _, _, _ = _rollout_no_qp_5th(
            model, dynamics, constraint, x0, u0, rk, n_steps=100)
        if rollout['obs'].shape[0] > 1:
            adv, ret = compute_gae(rollout['rewards'], rollout['values'],
                                   rollout['dones'])
            batch = {'obs': rollout['obs'], 'actions': rollout['actions'],
                     'old_log_probs': rollout['log_probs'],
                     'advantages': adv, 'returns': ret}
            for _ in range(trainer.epochs):
                trainer.train_step(batch)

    # Evaluation on uncertain dynamics
    if deploy_scenario is not None:
        uncertain = UncertainUSCCSDynamics5th(
            load_ratio=load_ratio, uncertainty_scenario=deploy_scenario)
    else:
        uncertain = dynamics

    qp_solver = DifferentiableQP(v_max=10.0)
    cbf_viols = 0
    power_viols = 0
    qp_interventions = 0
    total_reward = 0.0
    x = x0[:NX].copy()
    key = jax.random.key(seed + 1000)

    for t in range(n_eval):
        key, ak = jax.random.split(key)
        v_rl, _, _ = model.get_action(x, ak)

        A, b = safety_layer.qp_matrices(x)
        v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b,
                                                    differentiable=False)
        v_safe = jnp.clip(v_safe, -10.0, 10.0)
        if jnp.any(jnp.abs(v_safe - v_rl) > 1e-3):
            qp_interventions += 1

        next_x = uncertain.step_stabilized_phi_scaled(x, v_safe)
        cv = constraint.check_all(next_x)

        if _count_violations_5th(cv, protected_only=True):
            cbf_viols += 1
        if cv.get('power_high', 1) < 0 or cv.get('power_low', 1) < 0:
            power_viols += 1

        y = dynamics.output(next_x)
        y0 = dynamics.output(x0)
        reward = (-1.0 * (y[0] - y0[0]) ** 2 - 0.001 * (y[1] - y0[1]) ** 2
                  - 0.01 * (y[2] - y0[2]) ** 2 - 0.0001 * jnp.sum(v_safe ** 2))
        total_reward += float(reward)
        x = next_x

    return {
        'cbf_violation': cbf_viols / n_eval * 100,
        'power_violation': power_viols / n_eval * 100,
        'qp_intervention': qp_interventions / n_eval * 100,
        'total_reward': total_reward,
        'gp_type': gp_type,
        'gp_strategy': strategy,
    }


def run_generic_gp_ood(seeds, n_eval, n_train, n_gp_pretrain,
                       conditions, strategies):
    all_results = []

    print(f"{'Strategy':<22} {'Deploy':<14} {'CBF%':>8} {'Pwr%':>6} "
          f"{'QP%':>6} {'Reward':>10}")
    print('-' * 75)

    for cond_name, scenario in conditions:
        for strategy in strategies:
            # OOD strategies only make sense for specific deployment scenarios
            if strategy == 'ood_s1_to_s3' and scenario != 'coupled':
                continue
            if strategy == 'ood_s3_to_s1' and scenario != 'heat_absorption':
                continue

            t0 = time.time()
            results = []
            for seed in seeds:
                try:
                    r = evaluate_gp_strategy(
                        strategy, scenario, seed,
                        n_train=n_train, n_eval=n_eval,
                        load_ratio=LOAD_RATIO,
                        n_gp_pretrain=n_gp_pretrain)
                    r['deploy_condition'] = cond_name
                    r['seed'] = seed
                    results.append(r)
                    all_results.append(r)
                except Exception as e:
                    print(f"  ERROR {strategy} {cond_name} seed={seed}: {e}")
                    import traceback; traceback.print_exc()

            if not results:
                continue

            elapsed = time.time() - t0
            avg_cbf = np.mean([r['cbf_violation'] for r in results])
            std_cbf = (np.std([r['cbf_violation'] for r in results])
                       if len(results) > 1 else 0.0)
            avg_pwr = np.mean([r['power_violation'] for r in results])
            avg_qp = np.mean([r['qp_intervention'] for r in results])
            avg_r = np.mean([r['total_reward'] for r in results])

            print(f"{strategy:<22} {cond_name:<14} {avg_cbf:>5.1f}±{std_cbf:<4.1f} "
                  f"{avg_pwr:>6.1f} {avg_qp:>6.1f} {avg_r:>10.1f}  [{elapsed:.0f}s]")
            sys.stdout.flush()

    # Save results
    out_path = os.path.join(RESULTS_DIR, 'generic_gp_ood.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Wilson CI summary
    from experiments.phase4.statistics import format_violation_with_ci
    print(f"\n{'='*80}")
    print("EPISODE-LEVEL WILSON CI SUMMARY")
    print(f"{'='*80}")
    print(f"{'Strategy':<22} {'Deploy':<14} {'CBF Viol':<25}")
    print('-' * 65)
    for cond_name, scenario in conditions:
        for strategy in strategies:
            if strategy == 'ood_s1_to_s3' and scenario != 'coupled':
                continue
            if strategy == 'ood_s3_to_s1' and scenario != 'heat_absorption':
                continue
            cell = [r for r in all_results
                    if r['deploy_condition'] == cond_name
                    and r['gp_strategy'] == strategy]
            if not cell:
                continue
            n_viol = sum(1 for r in cell if r['cbf_violation'] > 0)
            n_total = len(cell)
            ci_str = format_violation_with_ci(n_viol, n_total)
            print(f"{strategy:<22} {cond_name:<14} {ci_str:<25}")
    print('=' * 65)

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--n_eval', type=int, default=500)
    parser.add_argument('--n_train', type=int, default=N_TRAIN)
    parser.add_argument('--n_gp_pretrain', type=int, default=N_GP_PRETRAIN)
    parser.add_argument('--conditions', nargs='+', default=None,
                        help='Deployment conditions (e.g. S1 S3 S5)')
    args = parser.parse_args()

    if args.conditions:
        cond_aliases = {
            'Nominal': 'Nominal', 'N': 'Nominal',
            'S1': 'S1:Heat', 'S1:Heat': 'S1:Heat',
            'S2': 'S2:Pressure', 'S2:Pressure': 'S2:Pressure',
            'S3': 'S3:Coupled', 'S3:Coupled': 'S3:Coupled',
            'S4': 'S4:Nonlinear', 'S4:Nonlinear': 'S4:Nonlinear',
            'S5': 'S5:Valve', 'S5:Valve': 'S5:Valve',
            'S6': 'S6:Fuel', 'S6:Fuel': 'S6:Fuel',
        }
        conditions = []
        for c in args.conditions:
            alias = cond_aliases.get(c, c)
            if alias in DEPLOY_CONDITIONS:
                conditions.append((alias, DEPLOY_CONDITIONS[alias]))
    else:
        conditions = list(DEPLOY_CONDITIONS.items())

    strategies = ['scenario_specific', 'generic_mixed']
    # Add OOD strategies only if S1 and S3 are in the condition set
    cond_names = {c[0] for c in conditions}
    if 'S1:Heat' in cond_names:
        strategies.append('ood_s3_to_s1')
    if 'S3:Coupled' in cond_names:
        strategies.append('ood_s1_to_s3')

    run_generic_gp_ood(
        seeds=args.seeds,
        n_eval=args.n_eval,
        n_train=args.n_train,
        n_gp_pretrain=args.n_gp_pretrain,
        conditions=conditions,
        strategies=strategies,
    )

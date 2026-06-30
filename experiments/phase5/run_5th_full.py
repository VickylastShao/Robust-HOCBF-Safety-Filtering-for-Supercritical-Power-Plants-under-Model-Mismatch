"""Comprehensive 5th-order CCS safety comparison.

Full experiment protocol:
- 5 methods × 7 conditions × 5 seeds
- Proper training (30 episodes) and GP pretraining (3000 points)
- Mixed GP for PPO-GP-HOCBF and PPO-RHOCBF
- Scenario-specific GP for RoCBF-Net
- 500 evaluation steps

Usage:
    conda activate jax_gpu
    cd /home/gpu/sz_workspace/RoCBF-Net
    python experiments/phase5/run_5th_full.py
    python experiments/phase5/run_5th_full.py --methods ppo ppo_hocbf ppo_gp_hocbf ppo_rhocbf
    python experiments/phase5/run_5th_full.py --conditions S1 S5 S6
"""
import sys
import time
import warnings
import json
import os
import argparse

warnings.filterwarnings('ignore')
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
    NX, _make_ccs_env_5th, _make_hocbf_5th, _make_robust_hocbf_5th,
    _pretrain_gp_5th, _rollout_no_qp_5th, _count_violations_5th, CBF_PROTECTED_5TH,
    SCENARIOS, SCENARIO_LABELS,
)

# ===== Configuration =====
LOAD_RATIO = 0.75
N_TRAIN = 30        # Training episodes (was 15 in quick run)
N_GP_PRETRAIN = 3000  # GP pretraining points (was 1500)
N_EVAL = 500        # Evaluation steps (was 300)
N_SEEDS = 5         # Number of random seeds

RESULTS_DIR = '/home/gpu/sz_workspace/RoCBF-Net/results/p0_metrics_5th'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Condition mapping
ALL_CONDITIONS = {
    'Nominal': None,
    'S1:Heat': 'heat_absorption',
    'S2:Pressure': 'pressure_oscillation',
    'S3:Coupled': 'coupled',
    'S4:Nonlinear': 'nonlinear',
    'S5:Valve': 'valve_degradation',
    'S6:Fuel': 'fuel_quality',
}

# Short aliases for command-line
CONDITION_ALIASES = {
    'Nominal': 'Nominal', 'nominal': 'Nominal', 'N': 'Nominal',
    'S1': 'S1:Heat', 'S1:Heat': 'S1:Heat', 'heat': 'S1:Heat',
    'S2': 'S2:Pressure', 'S2:Pressure': 'S2:Pressure', 'pressure': 'S2:Pressure',
    'S3': 'S3:Coupled', 'S3:Coupled': 'S3:Coupled', 'coupled': 'S3:Coupled',
    'S4': 'S4:Nonlinear', 'S4:Nonlinear': 'S4:Nonlinear', 'nonlinear': 'S4:Nonlinear',
    'S5': 'S5:Valve', 'S5:Valve': 'S5:Valve', 'valve': 'S5:Valve',
    'S6': 'S6:Fuel', 'S6:Fuel': 'S6:Fuel', 'fuel': 'S6:Fuel',
}

METHOD_ALIASES = {
    'ppo': 'ppo', 'PPO': 'ppo',
    'hocbf': 'ppo_hocbf', 'ppo_hocbf': 'ppo_hocbf', 'PPO-HOCBF': 'ppo_hocbf',
    'gp_hocbf': 'ppo_gp_hocbf', 'ppo_gp_hocbf': 'ppo_gp_hocbf', 'PPO-GP-HOCBF': 'ppo_gp_hocbf',
    'rhocbf': 'ppo_rhocbf', 'ppo_rhocbf': 'ppo_rhocbf', 'PPO-RHOCBF': 'ppo_rhocbf',
    'rocbf': 'rocbf_net', 'rocbf_net': 'rocbf_net', 'RoCBF-Net': 'rocbf_net',
}


def evaluate(method_name, scenario, seed, n_train=N_TRAIN, n_eval=N_EVAL,
             load_ratio=LOAD_RATIO):
    """Train PPO and evaluate with/without safety filter on 5th-order CCS.

    GP configuration follows the paper:
    - PPO-GP-HOCBF: mixed GP (trained on all scenarios), ε_kappa=0.0
    - PPO-RHOCBF: mixed GP (trained on all scenarios), ε_kappa=1.0
    - RoCBF-Net: scenario-specific GP, ε_kappa=1.0
    """
    dynamics, constraint = _make_ccs_env_5th(load_ratio, scenario=scenario)
    x0, u0 = dynamics.equilibrium(load_ratio)

    model = ActorCritic(n_obs=NX, n_act=3, hidden_dim=128, rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=1e-4, epochs=4, minibatch_size=64)

    gp = None
    safety_layer = None
    gp_type = None

    if method_name == 'ppo_hocbf':
        safety_layer = _make_hocbf_5th(dynamics, constraint, u0)

    elif method_name == 'ppo_gp_hocbf':
        # Mixed GP: trained on data from all scenarios (S1-S4)
        gp = _pretrain_gp_5th(load_ratio, n_pretrain=N_GP_PRETRAIN,
                               key=jax.random.key(seed * 100 + 42),
                               scenario_specific=False)
        safety_layer = _make_robust_hocbf_5th(
            dynamics, constraint, gp, u0,
            epsilon_kappa=0.0, use_mean_correction=True)
        gp_type = 'mixed'

    elif method_name == 'ppo_rhocbf':
        # Mixed GP: trained on data from all scenarios (S1-S4)
        gp = _pretrain_gp_5th(load_ratio, n_pretrain=N_GP_PRETRAIN,
                               key=jax.random.key(seed * 100 + 42),
                               scenario_specific=False)
        safety_layer = _make_robust_hocbf_5th(
            dynamics, constraint, gp, u0,
            epsilon_kappa=1.0, use_mean_correction=True)
        gp_type = 'mixed'

    elif method_name == 'rocbf_net':
        # Scenario-specific GP
        gp = _pretrain_gp_5th(load_ratio, n_pretrain=N_GP_PRETRAIN,
                               key=jax.random.key(seed * 100 + 42),
                               scenario=scenario, scenario_specific=True)
        safety_layer = _make_robust_hocbf_5th(
            dynamics, constraint, gp, u0,
            epsilon_kappa=1.0, use_mean_correction=True)
        gp_type = 'scenario_specific'

    # ===== Training (decoupled: no QP filter) =====
    key = jax.random.key(seed)
    for ep in range(n_train):
        key, rk = jax.random.split(key)
        rollout, ep_r, _, _, _ = _rollout_no_qp_5th(
            model, dynamics, constraint, x0, u0, rk, n_steps=100)
        if rollout['obs'].shape[0] > 1:
            adv, ret = compute_gae(rollout['rewards'], rollout['values'], rollout['dones'])
            batch = {'obs': rollout['obs'], 'actions': rollout['actions'],
                     'old_log_probs': rollout['log_probs'], 'advantages': adv, 'returns': ret}
            for _ in range(trainer.epochs):
                trainer.train_step(batch)

    # ===== Evaluation with QP filter =====
    qp_solver = DifferentiableQP(v_max=10.0)
    cbf_viols = 0
    power_viols = 0
    qp_interventions = 0
    per_constraint = {'pressure_high': 0, 'pressure_low': 0,
                      'enthalpy_high': 0, 'enthalpy_low': 0,
                      'power_high': 0, 'power_low': 0}
    total_reward = 0.0
    x = x0

    for t in range(n_eval):
        key, ak = jax.random.split(key)
        v_rl, _, _ = model.get_action(x[:NX], ak)

        if safety_layer is not None:
            A, b = safety_layer.qp_matrices(x[:NX])
            v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
            v_safe = jnp.clip(v_safe, -10.0, 10.0)
            if jnp.any(jnp.abs(v_safe - v_rl) > 1e-3):
                qp_interventions += 1
        else:
            v_safe = v_rl

        next_x = dynamics.step_stabilized(x[:NX], v_safe)
        cv = constraint.check_all(next_x)

        for cname in per_constraint:
            if cname in cv and cv[cname] < 0:
                per_constraint[cname] += 1

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
        'qp_intervention': qp_interventions / n_eval * 100 if safety_layer else 0.0,
        'total_reward': total_reward,
        'per_constraint': {k: v / n_eval * 100 for k, v in per_constraint.items()},
        'gp_type': gp_type,
        'n_train': n_train,
        'n_eval': n_eval,
        'gp_pretrain': N_GP_PRETRAIN if gp else 0,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Methods to evaluate (ppo, ppo_hocbf, ppo_gp_hocbf, ppo_rhocbf, rocbf_net)')
    parser.add_argument('--conditions', nargs='+', default=None,
                        help='Conditions to evaluate (Nominal, S1-S6)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Seeds to run (default: 0-4)')
    parser.add_argument('--n-train', type=int, default=N_TRAIN)
    parser.add_argument('--n-eval', type=int, default=N_EVAL)
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (default: safety_comparison_full.json)')
    args = parser.parse_args()

    # Resolve methods
    if args.methods:
        methods = [METHOD_ALIASES.get(m, m) for m in args.methods]
    else:
        methods = ['ppo', 'ppo_hocbf', 'ppo_gp_hocbf', 'ppo_rhocbf', 'rocbf_net']

    # Resolve conditions
    if args.conditions:
        conditions = [(CONDITION_ALIASES.get(c, c), ALL_CONDITIONS.get(CONDITION_ALIASES.get(c, c)))
                      for c in args.conditions]
    else:
        conditions = list(ALL_CONDITIONS.items())

    # Resolve seeds
    seeds = args.seeds if args.seeds else list(range(N_SEEDS))

    # Output file
    output_file = args.output or 'safety_comparison_full.json'

    print(f"{'Method':<18} {'Cond':<14} {'CBF%':>6} {'Pwr%':>6} {'QP%':>6} "
          f"{'p_hi':>5} {'p_lo':>5} {'h_hi':>5} {'h_lo':>5} {'N_hi':>5} {'N_lo':>5}")
    print('-' * 90)

    all_results = []
    t0_all = time.time()

    for cond_name, scenario in conditions:
        for method in methods:
            t0 = time.time()
            results = []
            for seed in seeds:
                try:
                    r = evaluate(method, scenario, seed,
                                 n_train=args.n_train, n_eval=args.n_eval)
                    r['method'] = method
                    r['condition'] = cond_name
                    r['seed'] = seed
                    results.append(r)
                    all_results.append(r)
                except Exception as e:
                    print(f"  ERROR {method} {cond_name} seed={seed}: {e}")
                    import traceback
                    traceback.print_exc()

            if not results:
                continue

            elapsed = time.time() - t0
            avg_cbf = np.mean([r['cbf_violation'] for r in results])
            std_cbf = np.std([r['cbf_violation'] for r in results]) if len(results) > 1 else 0.0
            avg_pwr = np.mean([r['power_violation'] for r in results])
            avg_qp = np.mean([r['qp_intervention'] for r in results])
            avg_pc = {}
            for cname in results[0]['per_constraint']:
                avg_pc[cname] = np.mean([r['per_constraint'][cname] for r in results])

            print(f"{method:<18} {cond_name:<14} {avg_cbf:>5.1f}±{std_cbf:<4.1f} {avg_pwr:>6.1f} {avg_qp:>6.1f} "
                  f"{avg_pc['pressure_high']:>5.1f} {avg_pc['pressure_low']:>5.1f} "
                  f"{avg_pc['enthalpy_high']:>5.1f} {avg_pc['enthalpy_low']:>5.1f} "
                  f"{avg_pc['power_high']:>5.1f} {avg_pc['power_low']:>5.1f}  [{elapsed:.0f}s]")
            sys.stdout.flush()

    print(f"\nTotal time: {time.time() - t0_all:.0f}s")

    # Save results
    results_file = os.path.join(RESULTS_DIR, output_file)
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_file}")

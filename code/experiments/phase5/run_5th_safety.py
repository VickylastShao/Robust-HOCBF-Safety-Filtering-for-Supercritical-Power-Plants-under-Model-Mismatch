"""5th-order CCS safety comparison: 4 methods × 3 conditions × 3 seeds.

Writes results to results/p0_metrics_5th/safety_comparison.json
"""
import sys
import time
import warnings
import json
import os

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
)

LOAD_RATIO = 0.75
N_EVAL = 300
N_TRAIN = 15  # Reduced from 30 for faster runs
N_GP_PRETRAIN = 1500  # Reduced from 3000
RESULTS_DIR = '/home/gpu/sz_workspace/RoCBF-Net/results/p0_metrics_5th'
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate(method_name, scenario, seed, n_train=N_TRAIN, n_eval=300):
    """Train PPO and evaluate with/without safety filter."""
    dynamics, constraint = _make_ccs_env_5th(LOAD_RATIO, scenario=scenario)
    x0, u0 = dynamics.equilibrium(LOAD_RATIO)

    model = ActorCritic(n_obs=NX, n_act=3, hidden_dim=128, rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=1e-4, epochs=4, minibatch_size=64)

    gp = None
    safety_layer = None
    if method_name == 'ppo_hocbf':
        safety_layer = _make_hocbf_5th(dynamics, constraint, u0)
    elif method_name == 'ppo_gp_hocbf':
        gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=N_GP_PRETRAIN, key=jax.random.key(seed*100+42),
                               scenario=scenario, scenario_specific=True)
        safety_layer = _make_robust_hocbf_5th(dynamics, constraint, gp, u0,
                                               epsilon_kappa=0.0, use_mean_correction=True)
    elif method_name in ('ppo_rhocbf', 'rocbf_net'):
        gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=N_GP_PRETRAIN, key=jax.random.key(seed*100+42),
                               scenario=scenario, scenario_specific=True)
        safety_layer = _make_robust_hocbf_5th(dynamics, constraint, gp, u0,
                                               epsilon_kappa=1.0, use_mean_correction=True)

    # Training (decoupled: no QP filter)
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

    # Evaluation with QP filter
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
    }


if __name__ == '__main__':
    conditions = [
        ('Nominal', None),
        ('S1:Heat', 'heat_absorption'),
        ('S2:Pressure', 'pressure_oscillation'),
    ]
    methods = ['ppo', 'ppo_hocbf', 'ppo_gp_hocbf', 'ppo_rhocbf']

    print(f"{'Method':<18} {'Cond':<12} {'CBF%':>6} {'Pwr%':>6} {'QP%':>6} "
          f"{'p_hi':>5} {'p_lo':>5} {'h_hi':>5} {'h_lo':>5} {'N_hi':>5} {'N_lo':>5}")
    print('-' * 80)

    all_results = []
    t0_all = time.time()
    for cond_name, scenario in conditions:
        for method in methods:
            t0 = time.time()
            results = []
            for seed in range(3):
                r = evaluate(method, scenario, seed)
                r['method'] = method
                r['condition'] = cond_name
                r['seed'] = seed
                results.append(r)
                all_results.append(r)
            elapsed = time.time() - t0
            avg_cbf = np.mean([r['cbf_violation'] for r in results])
            avg_pwr = np.mean([r['power_violation'] for r in results])
            avg_qp = np.mean([r['qp_intervention'] for r in results])
            avg_pc = {}
            for cname in results[0]['per_constraint']:
                avg_pc[cname] = np.mean([r['per_constraint'][cname] for r in results])

            print(f"{method:<18} {cond_name:<12} {avg_cbf:>6.1f} {avg_pwr:>6.1f} {avg_qp:>6.1f} "
                  f"{avg_pc['pressure_high']:>5.1f} {avg_pc['pressure_low']:>5.1f} "
                  f"{avg_pc['enthalpy_high']:>5.1f} {avg_pc['enthalpy_low']:>5.1f} "
                  f"{avg_pc['power_high']:>5.1f} {avg_pc['power_low']:>5.1f}  [{elapsed:.0f}s]")
            sys.stdout.flush()

    print(f"\nTotal time: {time.time() - t0_all:.0f}s")

    # Save results
    results_file = os.path.join(RESULTS_DIR, 'safety_comparison.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_file}")

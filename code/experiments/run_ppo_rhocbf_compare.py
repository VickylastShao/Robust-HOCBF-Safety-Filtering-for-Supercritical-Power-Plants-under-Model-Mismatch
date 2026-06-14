#!/usr/bin/env python
"""Quick PPO-RHOCBF evaluation for LQR comparison.
Uses trained PPO policy with RHOCBF safety filter on S1:Heat.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax, jax.numpy as jnp, numpy as np, flax.nnx as nnx
from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.methods_5th import (
    NX, _make_robust_hocbf_5th, _pretrain_gp_5th,
    _rollout_no_qp_5th, _count_violations_5th,
)

N_STEPS = 300
N_SEEDS = 5

def train_ppo(dynamics, constraint, x0, u0, seed=0):
    model = ActorCritic(n_obs=NX, n_act=3, hidden_dim=128, rngs=nnx.Rngs(seed))
    trainer = PPOTrainer(model, lr=1e-4, epochs=4, minibatch_size=64)
    key = jax.random.key(seed * 100)
    for ep in range(20):
        key, rk = jax.random.split(key)
        rollout, ep_r, _, _, _ = _rollout_no_qp_5th(model, dynamics, constraint, x0, u0, rk, n_steps=100)
        if rollout['obs'].shape[0] > 1:
            adv, ret = compute_gae(rollout['rewards'], rollout['values'], rollout['dones'])
            batch = {'obs': rollout['obs'], 'actions': rollout['actions'],
                     'old_log_probs': rollout['log_probs'], 'advantages': adv, 'returns': ret}
            for _ in range(trainer.epochs):
                trainer.train_step(batch)
    return model


def eval_ppo_rhocbf(model, dynamics, constraint, multi_rhocbf, x0, u0, n_steps, seed=0):
    qp = DifferentiableQP(v_max=10.0)
    x = x0[:NX].copy()
    total_reward = 0.0
    cbf_viols = 0
    qp_interventions = 0
    y0 = dynamics.output(x0)

    for t in range(n_steps):
        key = jax.random.fold_in(jax.random.key(seed * 1000 + t), t)
        v_rl, _, _ = model.get_action(x, key)

        A, b = multi_rhocbf.qp_matrices(x)
        v_safe, _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
        v_safe = jnp.clip(v_safe, -10.0, 10.0)
        if jnp.any(jnp.abs(v_safe - v_rl) > 1e-3):
            qp_interventions += 1

        next_x = dynamics.step_stabilized(x, v_safe)
        cv = constraint.check_all(next_x)
        if _count_violations_5th(cv, protected_only=True):
            cbf_viols += 1

        y = dynamics.output(next_x)
        reward = (-1.0*(y[0]-y0[0])**2 - 0.001*(y[1]-y0[1])**2
                  - 0.01*(y[2]-y0[2])**2 - 0.0001*jnp.sum(v_safe**2))
        total_reward += float(reward)
        x = next_x

    return total_reward, cbf_viols / n_steps * 100, qp_interventions / n_steps * 100


# Run PPO+RHOCBF on S1:Heat
scenario = 'heat_absorption'
print(f'PPO+RHOCBF evaluation: S1:Heat, {N_SEEDS} seeds, {N_STEPS} steps', flush=True)

rewards = []
cbf_rates = []
qp_rates = []

for seed in range(N_SEEDS):
    key = jax.random.key(seed * 100 + 42)
    dynamics = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=1.0, uncertainty_scenario=scenario)
    constraint = CCSConstraints5th(p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
                                    power_deviation=50.0, power_target=1000.0)
    x0, u0 = dynamics.equilibrium(1.0)

    # Train PPO
    model = train_ppo(dynamics, constraint, x0, u0, seed=seed)

    # GP + Robust HOCBF
    key, gp_key = jax.random.split(key)
    gp = _pretrain_gp_5th(1.0, key=gp_key, scenario=scenario, scenario_specific=True)
    multi_rhocbf = _make_robust_hocbf_5th(dynamics, constraint, gp, u0, use_mean_correction=True)

    # Evaluate
    reward, cbf_viol, qp_int = eval_ppo_rhocbf(model, dynamics, constraint, multi_rhocbf,
                                                  x0, u0, N_STEPS, seed=seed)
    rewards.append(reward)
    cbf_rates.append(cbf_viol)
    qp_rates.append(qp_int)
    print(f'  Seed {seed}: reward={reward:.1f}, cbf_viol={cbf_viol:.1f}%, QP_int={qp_int:.1f}%', flush=True)

print(f'\nPPO+RHOCBF: reward={np.mean(rewards):.1f}+-{np.std(rewards):.1f}, '
      f'cbf_viol={np.mean(cbf_rates):.1f}%, QP_int={np.mean(qp_rates):.1f}%', flush=True)

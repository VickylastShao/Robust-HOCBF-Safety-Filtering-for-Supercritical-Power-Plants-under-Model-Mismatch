#!/usr/bin/env python
"""PPO+RHOCBF vs LQR+RHOCBF comparison on all scenarios."""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax, jax.numpy as jnp, numpy as np, flax.nnx as nnx
from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.methods_5th import (
    NX, _make_robust_hocbf_5th, _pretrain_gp_5th,
    _rollout_no_qp_5th, _count_violations_5th,
    SCENARIOS, SCENARIO_LABELS,
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


def eval_method(model, dynamics, constraint, multi_rhocbf, x0, u0, n_steps, seed=0):
    """Evaluate PPO+RHOCBF or LQR+RHOCBF."""
    qp = DifferentiableQP(v_max=10.0)
    x = x0[:NX].copy()
    total_reward = 0.0
    cbf_viols = 0
    qp_interventions = 0
    y0 = dynamics.output(x0)
    use_lqr = model is None

    for t in range(n_steps):
        key = jax.random.fold_in(jax.random.key(seed * 1000 + t), t)
        if use_lqr:
            v_rl = jnp.zeros(3)
        else:
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


# Run comparison on all scenarios
all_results = {}
for si, scenario in enumerate(SCENARIOS):
    label = SCENARIO_LABELS[si]
    print(f'\n{"="*70}', flush=True)
    print(f'Scenario: {label}', flush=True)
    print(f'{"="*70}', flush=True)

    ppo_rewards, ppo_cbfs, ppo_qps = [], [], []
    lqr_rewards, lqr_cbfs, lqr_qps = [], [], []

    for seed in range(N_SEEDS):
        key = jax.random.key(seed * 100 + 42)
        if scenario is not None:
            dynamics = UncertainUSCCSDynamics5th(dt=1.0, load_ratio=1.0, uncertainty_scenario=scenario)
        else:
            dynamics = USCCSDynamics5th(dt=1.0, load_ratio=1.0)
        constraint = CCSConstraints5th(p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
                                        power_deviation=50.0, power_target=1000.0)
        x0, u0 = dynamics.equilibrium(1.0)

        # GP + Robust HOCBF
        key, gp_key = jax.random.split(key)
        gp = _pretrain_gp_5th(1.0, key=gp_key, scenario=scenario, scenario_specific=True)
        multi_rhocbf = _make_robust_hocbf_5th(dynamics, constraint, gp, u0, use_mean_correction=True)

        # Train PPO
        model = train_ppo(dynamics, constraint, x0, u0, seed=seed)

        # Evaluate PPO+RHOCBF
        r, c, q = eval_method(model, dynamics, constraint, multi_rhocbf, x0, u0, N_STEPS, seed)
        ppo_rewards.append(r); ppo_cbfs.append(c); ppo_qps.append(q)

        # Evaluate LQR+RHOCBF
        r, c, q = eval_method(None, dynamics, constraint, multi_rhocbf, x0, u0, N_STEPS, seed)
        lqr_rewards.append(r); lqr_cbfs.append(c); lqr_qps.append(q)

        print(f'  Seed {seed}: PPO reward={ppo_rewards[-1]:.1f} qp={ppo_qps[-1]:.1f}% | '
              f'LQR reward={lqr_rewards[-1]:.1f} qp={lqr_qps[-1]:.1f}%', flush=True)

    all_results[label] = {
        'PPO': {'reward': np.mean(ppo_rewards), 'reward_std': np.std(ppo_rewards),
                'cbf': np.mean(ppo_cbfs), 'qp': np.mean(ppo_qps)},
        'LQR': {'reward': np.mean(lqr_rewards), 'reward_std': np.std(lqr_rewards),
                'cbf': np.mean(lqr_cbfs), 'qp': np.mean(lqr_qps)},
    }
    print(f'  PPO: reward={np.mean(ppo_rewards):.1f}+-{np.std(ppo_rewards):.1f}, '
          f'cbf={np.mean(ppo_cbfs):.1f}%, qp={np.mean(ppo_qps):.1f}%', flush=True)
    print(f'  LQR: reward={np.mean(lqr_rewards):.1f}+-{np.std(lqr_rewards):.1f}, '
          f'cbf={np.mean(lqr_cbfs):.1f}%, qp={np.mean(lqr_qps):.1f}%', flush=True)

# Summary table
print(f'\n\n{"="*90}', flush=True)
print('Summary: PPO+RHOCBF vs LQR+RHOCBF', flush=True)
print(f'{"="*90}', flush=True)
print(f'{"Scenario":<15} {"PPO Reward":>14} {"LQR Reward":>14} {"PPO CBF%":>10} {"LQR CBF%":>10} {"PPO QP%":>10} {"LQR QP%":>10}', flush=True)
print('-'*83, flush=True)
for label, r in all_results.items():
    print(f'{label:<15} {r["PPO"]["reward"]:>8.1f}+-{r["PPO"]["reward_std"]:<4.1f} '
          f'{r["LQR"]["reward"]:>8.1f}+-{r["LQR"]["reward_std"]:<4.1f} '
          f'{r["PPO"]["cbf"]:>8.1f}% {r["LQR"]["cbf"]:>8.1f}% '
          f'{r["PPO"]["qp"]:>8.1f}% {r["LQR"]["qp"]:>8.1f}%', flush=True)

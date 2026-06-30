"""Run S1:Heat with MC=True + k_h=3.0, using QP filter during BOTH training and evaluation.

This is the correct approach: the policy must learn to cooperate with the
mean-corrected QP safety filter during training.
"""
import sys
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import json
import time
import yaml
import os
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.gp.gp_residual import GPResidual
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from experiments.phase4.methods import (_make_ccs_env, _make_robust_hocbf,
                                         _pretrain_gp, _rollout_with_qp,
                                         _rollout_no_qp, _count_violations)

with open('configs/phase4.yaml') as f:
    config = yaml.safe_load(f)

hocbf_cfg = config['hocbf']
method_cfg = config['methods_config']['rocbf_net'].copy()
training_cfg = config['training']
eval_cfg = config['evaluation']

seed = 0
condition = 's1_heat'
scenario = 'heat_absorption'
key = jax.random.key(seed)

# Setup
dynamics, constraint = _make_ccs_env(1.0, 0, scenario)
x0, u0 = dynamics.equilibrium(1.0)

# Pre-train scenario-specific GP
print("Pre-training scenario-specific GP...")
key, gp_key = jax.random.split(key)
gp = _pretrain_gp(1.0, 0, n_pretrain=method_cfg.get('n_pretrain', 2000),
                   key=gp_key, sigma_floor=1e-4,
                   scenario=scenario, scenario_specific=True)
print(f"GP: n_points={gp.n_training_points}")

# Build safety layer
k_p = tuple(hocbf_cfg['pressure_k_gains'])
k_h = tuple(hocbf_cfg['enthalpy_k_gains'])
u_max = hocbf_cfg['u_max']
safety_layer = _make_robust_hocbf(dynamics, constraint, gp, u0,
                                   epsilon_kappa=1.0, k_pressure=k_p,
                                   k_enthalpy=k_h, u_max=u_max,
                                   use_mean_correction=True, epsilon_floor=0.0)

# Initialize model
model = ActorCritic(n_obs=3, n_act=3, hidden_dim=method_cfg.get('hidden_dim', 128),
                    rngs=nnx.Rngs(0))
trainer = PPOTrainer(model, lr=method_cfg.get('lr', 1e-4))
qp_solver = DifferentiableQP(v_max=5.0)

# Training loop WITH QP filter
n_episodes = 20  # Quick test: fewer episodes
n_steps = training_cfg.get('n_steps', 200)
gp_update_interval = method_cfg.get('gp_update_interval', 50)

reward_history = []
epsilon_log = []

# Training dynamics (uncertain)
train_dyn = UncertainUSCCSDynamics(delay_order=0, load_ratio=1.0,
                                   uncertainty_scenario=scenario)

print(f"\nTraining with QP filter: {n_episodes} episodes x {n_steps} steps")
t_start = time.time()

for ep in range(n_episodes):
    key, rollout_key = jax.random.split(key)

    # Training rollout WITH QP filter
    rollout, ep_reward, violations, cbf_violations, qp_times = _rollout_with_qp(
        model, train_dyn, safety_layer, qp_solver, constraint,
        x0, u0, rollout_key, n_steps, use_scipy=True)

    if rollout['obs'].shape[0] < 2:
        continue

    reward_history.append(ep_reward)

    # PPO update
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
        loss = trainer.train_step(batch)

    # GP update
    if (ep + 1) % gp_update_interval == 0:
        key, gp_key = jax.random.split(key)
        from experiments.phase4.methods import _collect_gp_data
        key, data_key = jax.random.split(gp_key)
        X_new, Y_new = _collect_gp_data(train_dyn, 200, key=data_key)
        gp.incremental_update(X_new, Y_new, reoptimize_hyperparams=False)
        safety_layer = _make_robust_hocbf(dynamics, constraint, gp, u0,
                                           epsilon_kappa=1.0, k_pressure=k_p,
                                           k_enthalpy=k_h, u_max=u_max,
                                           use_mean_correction=True, epsilon_floor=0.0)

        # Log epsilon
        eps_vals = [float(h.compute_epsilon(x0[:3])) for h in safety_layer.robust_hocbf_list]
        mu, sigma = gp.predict(x0[:3].reshape(1, -1))
        epsilon_log.append({
            'episode': ep + 1,
            'n_gp_points': gp.n_training_points,
            'epsilon_total': sum(eps_vals),
            'mu_gp_mean': float(jnp.mean(mu)),
            'sigma_gp_mean': float(jnp.mean(sigma)),
        })

    if (ep + 1) % 1 == 0:
        elapsed = time.time() - t_start
        avg_qp = np.mean(qp_times) if qp_times else 0
        print(f"  Ep {ep+1}: r={ep_reward:.1f}, viol={violations}/{n_steps}, "
              f"avg_qp={avg_qp:.0f}ms, elapsed={elapsed:.0f}s", flush=True)

train_time = time.time() - t_start
print(f"\nTraining done in {train_time:.0f}s ({train_time/60:.1f}min)")

# Evaluation
print("\nEvaluating...")
n_eval = eval_cfg.get('n_episodes', 3)
n_eval_steps = eval_cfg.get('n_steps', 200)

all_violation_rates = []
all_rewards = []

for ep in range(n_eval):
    key, ep_key = jax.random.split(key)
    rollout, ep_reward, violations, cbf_violations, qp_times = _rollout_with_qp(
        model, dynamics, safety_layer, qp_solver, constraint,
        x0, u0, ep_key, n_eval_steps, use_scipy=True)
    all_violation_rates.append(violations / n_eval_steps)
    all_rewards.append(ep_reward)

violation_rate = np.mean(all_violation_rates)
cumulative_reward = np.mean(all_rewards)

print(f"\nResults:")
print(f"  Violation rate: {violation_rate:.4f}")
print(f"  Cumulative reward: {cumulative_reward:.1f}")
print(f"  Epsilon log: {epsilon_log}")

# Save
output_dir = 'results/phase5_mc_true_qp_train'
os.makedirs(output_dir, exist_ok=True)
result = {
    'violation_rate': violation_rate,
    'cumulative_reward': cumulative_reward,
    'train_episodes': n_episodes,
    'train_time_s': train_time,
    'epsilon_log': epsilon_log,
    'config': {'use_mean_correction': True, 'enthalpy_k_gains': list(k_h),
               'scenario_specific_gp': True, 'train_with_qp': True},
}
with open(os.path.join(output_dir, 'rocbf_net_s1_heat_seed0.json'), 'w') as f:
    json.dump(result, f, indent=2)

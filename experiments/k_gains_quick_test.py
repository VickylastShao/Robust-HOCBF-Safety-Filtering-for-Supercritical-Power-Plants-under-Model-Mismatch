"""Quick test: k_p=[2.0,2.0] with MC=True + scenario-specific GP on S1:Heat.

This is the most promising configuration: large k_gains give b_min=5.357,
providing plenty of QP headroom while MC corrects the drift direction.
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
                                         _rollout_no_qp)

with open('configs/phase4.yaml') as f:
    config = yaml.safe_load(f)

training_cfg = config['training']
eval_cfg = config['evaluation']

seed = 0
scenario = 'heat_absorption'

# Configuration: k_p=[2.0,2.0], k_h=[3.0], MC=True, scenario-specific GP
k_p = (2.0, 2.0)
k_h = (3.0,)
mc = True
spec_gp = True
eps_floor = 0.0

print("Configuration: k_p=[2.0,2.0], k_h=[3.0], MC=True, scenario-specific GP")

key = jax.random.key(seed)

dynamics, constraint = _make_ccs_env(1.0, 0, scenario)
x0, u0 = dynamics.equilibrium(1.0)

# Pre-train GP
key, gp_key = jax.random.split(key)
gp = _pretrain_gp(1.0, 0, n_pretrain=2000, key=gp_key,
                   sigma_floor=1e-4, scenario=scenario,
                   scenario_specific=True)

# Build safety layer
safety_layer = _make_robust_hocbf(dynamics, constraint, gp, u0,
                                   epsilon_kappa=1.0,
                                   k_pressure=k_p,
                                   k_enthalpy=k_h,
                                   u_max=100.0,
                                   use_mean_correction=mc,
                                   epsilon_floor=eps_floor)

A, b = safety_layer.qp_matrices(x0[:3])
eps_vals = [float(h.compute_epsilon(x0[:3])) for h in safety_layer.robust_hocbf_list]
print(f"Initial: b_min={float(b.min()):.3f}, eps_total={sum(eps_vals):.3f}, feasible={bool(b.min()>0)}")

# Initialize model
model = ActorCritic(n_obs=3, n_act=3, hidden_dim=128, rngs=nnx.Rngs(seed))
trainer = PPOTrainer(model, lr=1e-4)
qp_solver = DifferentiableQP(v_max=5.0)

# Training
n_episodes = 200
n_steps = training_cfg.get('n_steps', 200)
gp_update_interval = 50

reward_history = []
epsilon_log = []

train_dyn = UncertainUSCCSDynamics(delay_order=0, load_ratio=1.0,
                                   uncertainty_scenario=scenario)

t_start = time.time()

for ep in range(n_episodes):
    key, rollout_key = jax.random.split(key)

    rollout, ep_reward, violations, _, _ = _rollout_no_qp(
        model, train_dyn, constraint, x0, u0, rollout_key, n_steps)

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
        loss = trainer.train_step(batch)

    if (ep + 1) % gp_update_interval == 0:
        key, gp_key = jax.random.split(key)
        from experiments.phase4.methods import _collect_gp_data
        key, data_key = jax.random.split(gp_key)
        env_gp = UncertainUSCCSDynamics(delay_order=0, load_ratio=1.0,
                                         uncertainty_scenario=scenario)
        X_new, Y_new = _collect_gp_data(env_gp, 200, key=data_key)
        gp.incremental_update(X_new, Y_new, reoptimize_hyperparams=False)

        safety_layer = _make_robust_hocbf(dynamics, constraint, gp, u0,
                                           epsilon_kappa=1.0,
                                           k_pressure=k_p,
                                           k_enthalpy=k_h,
                                           u_max=100.0,
                                           use_mean_correction=mc,
                                           epsilon_floor=eps_floor)

        eps_vals = [float(h.compute_epsilon(x0[:3])) for h in safety_layer.robust_hocbf_list]
        mu, sigma = gp.predict(x0[:3].reshape(1, -1))
        epsilon_log.append({
            'episode': ep + 1,
            'n_gp_points': gp.n_training_points,
            'epsilon_total': sum(eps_vals),
            'epsilon_per_constraint': eps_vals,
            'sigma_gp_mean': float(jnp.mean(sigma)),
            'sigma_gp_max': float(jnp.max(sigma)),
        })

    if (ep + 1) % 50 == 0:
        elapsed = time.time() - t_start
        print(f"  Ep {ep+1}: r={ep_reward:.1f}, elapsed={elapsed:.0f}s", flush=True)

train_time = time.time() - t_start
print(f"\nTraining done in {train_time:.0f}s")

# Evaluation
n_eval = 3
n_eval_steps = 200

all_violation_rates = []
all_cbf_violation_rates = []
all_rewards = []

for ep in range(n_eval):
    key, ep_key = jax.random.split(key)
    rollout, ep_reward, violations, cbf_violations, _ = _rollout_with_qp(
        model, dynamics, safety_layer, qp_solver, constraint,
        x0, u0, ep_key, n_eval_steps, use_scipy=True)
    all_violation_rates.append(violations / n_eval_steps)
    all_cbf_violation_rates.append(cbf_violations / n_eval_steps)
    all_rewards.append(ep_reward)
    print(f"  Eval ep {ep+1}: viol={violations}/{n_eval_steps}, cbf_viol={cbf_violations}, r={ep_reward:.1f}")

violation_rate = np.mean(all_violation_rates)
cbf_violation_rate = np.mean(all_cbf_violation_rates)
cumulative_reward = np.mean(all_rewards)

print(f"\nResults:")
print(f"  Violation rate: {violation_rate:.4f}")
print(f"  CBF violation: {cbf_violation_rate:.4f}")
print(f"  Reward: {cumulative_reward:.1f}")
if epsilon_log:
    print(f"  Epsilon: {epsilon_log[0]['epsilon_total']:.3f} → {epsilon_log[-1]['epsilon_total']:.3f}")

# Save
output_dir = 'results/k_gains_ablation'
os.makedirs(output_dir, exist_ok=True)
result = {
    'label': 'k2_mc_true_spec',
    'config': {'k_p': list(k_p), 'k_h': list(k_h), 'use_mean_correction': mc,
               'scenario_specific_gp': spec_gp, 'epsilon_floor': eps_floor},
    'violation_rate': [float(np.mean(all_violation_rates)), float(np.std(all_violation_rates))],
    'cbf_violation_rate': [float(np.mean(all_cbf_violation_rates)), float(np.std(all_cbf_violation_rates))],
    'cumulative_reward': [float(np.mean(all_rewards)), float(np.std(all_rewards))],
    'train_time_s': train_time,
    'epsilon_log': epsilon_log,
}
with open(os.path.join(output_dir, 'k2_mc_true_spec_s1_heat.json'), 'w') as f:
    json.dump(result, f, indent=2)

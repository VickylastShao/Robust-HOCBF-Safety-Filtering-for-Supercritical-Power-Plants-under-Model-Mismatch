"""Ablation: test different epsilon_floor values with MC=False + mixed GP.

The Phase 5 v3 configuration (MC=False + mixed GP) produces:
- PPO-RHOCBF: 6.2% violation, constant epsilon=2.56
- RoCBF-Net: 20.2% violation, epsilon drops 2.76→0.53

Goal: find epsilon_floor that lets RoCBF-Net keep advantage of online
GP adaptation while maintaining safety margin.
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
scenario = 'heat_absorption'
condition = 's1_heat'

# Test different epsilon_floor values
epsilon_floors = [0.0, 0.5, 1.0, 1.5, 2.0]

k_p = tuple(hocbf_cfg['pressure_k_gains'])
k_h = tuple(hocbf_cfg['enthalpy_k_gains'])
u_max = hocbf_cfg['u_max']

results = {}

for eps_floor in epsilon_floors:
    print(f"\n{'='*60}")
    print(f"Testing epsilon_floor = {eps_floor}")
    print(f"{'='*60}")

    key = jax.random.key(seed)

    # Setup dynamics
    dynamics, constraint = _make_ccs_env(1.0, 0, scenario)
    x0, u0 = dynamics.equilibrium(1.0)

    # Pre-train mixed GP (same as Phase 5 v3)
    key, gp_key = jax.random.split(key)
    gp = _pretrain_gp(1.0, 0, n_pretrain=method_cfg.get('n_pretrain', 2000),
                       key=gp_key, sigma_floor=1e-4)

    # Build safety layer with MC=False + epsilon_floor
    safety_layer = _make_robust_hocbf(dynamics, constraint, gp, u0,
                                       epsilon_kappa=1.0, k_pressure=k_p,
                                       k_enthalpy=k_h, u_max=u_max,
                                       use_mean_correction=False,
                                       epsilon_floor=eps_floor)

    # Initialize model
    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=128, rngs=nnx.Rngs(seed))
    trainer = PPOTrainer(model, lr=1e-4)
    qp_solver = DifferentiableQP(v_max=5.0)

    # Training loop (no QP, same as run_experiment.py)
    n_episodes = 200
    n_steps = training_cfg.get('n_steps', 200)
    gp_update_interval = method_cfg.get('gp_update_interval', 50)

    reward_history = []
    epsilon_log = []

    # Training dynamics (uncertain)
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

        # GP update for RoCBF-Net
        if (ep + 1) % gp_update_interval == 0:
            key, gp_key = jax.random.split(key)
            from experiments.phase4.methods import _collect_gp_data
            key, data_key = jax.random.split(gp_key)
            env_gp = UncertainUSCCSDynamics(delay_order=0, load_ratio=1.0,
                                             uncertainty_scenario=scenario)
            X_new, Y_new = _collect_gp_data(env_gp, 200, key=data_key)
            gp.incremental_update(X_new, Y_new, reoptimize_hyperparams=False)

            # Rebuild safety layer
            safety_layer = _make_robust_hocbf(dynamics, constraint, gp, u0,
                                               epsilon_kappa=1.0, k_pressure=k_p,
                                               k_enthalpy=k_h, u_max=u_max,
                                               use_mean_correction=False,
                                               epsilon_floor=eps_floor)

            # Log epsilon
            eps_vals = [float(h.compute_epsilon(x0[:3])) for h in safety_layer.robust_hocbf_list]
            mu, sigma = gp.predict(x0[:3].reshape(1, -1))
            epsilon_log.append({
                'episode': ep + 1,
                'n_gp_points': gp.n_training_points,
                'epsilon_total': sum(eps_vals),
                'sigma_gp_mean': float(jnp.mean(sigma)),
                'sigma_gp_max': float(jnp.max(sigma)),
            })

        if (ep + 1) % 50 == 0:
            elapsed = time.time() - t_start
            print(f"  Ep {ep+1}: r={ep_reward:.1f}, elapsed={elapsed:.0f}s", flush=True)

    train_time = time.time() - t_start
    print(f"Training done in {train_time:.0f}s")

    # Evaluation
    n_eval = eval_cfg.get('n_episodes', 3)
    n_eval_steps = eval_cfg.get('n_steps', 200)

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

    violation_rate = np.mean(all_violation_rates)
    cbf_violation_rate = np.mean(all_cbf_violation_rates)
    cumulative_reward = np.mean(all_rewards)

    result = {
        'epsilon_floor': eps_floor,
        'violation_rate': [float(np.mean(all_violation_rates)), float(np.std(all_violation_rates))],
        'cbf_violation_rate': [float(np.mean(all_cbf_violation_rates)), float(np.std(all_cbf_violation_rates))],
        'cumulative_reward': [float(np.mean(all_rewards)), float(np.std(all_rewards))],
        'train_time_s': train_time,
        'epsilon_log': epsilon_log,
    }
    results[str(eps_floor)] = result

    print(f"\n  Results for epsilon_floor={eps_floor}:")
    print(f"    Violation rate: {violation_rate:.4f}")
    print(f"    CBF violation: {cbf_violation_rate:.4f}")
    print(f"    Reward: {cumulative_reward:.1f}")
    if epsilon_log:
        print(f"    Epsilon: {epsilon_log[0]['epsilon_total']:.3f} → {epsilon_log[-1]['epsilon_total']:.3f}")

# Save all results
output_dir = 'results/epsilon_floor_ablation'
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 's1_heat_ablation.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for eps_floor in epsilon_floors:
    r = results[str(eps_floor)]
    eps_start = r['epsilon_log'][0]['epsilon_total'] if r['epsilon_log'] else 'N/A'
    eps_end = r['epsilon_log'][-1]['epsilon_total'] if r['epsilon_log'] else 'N/A'
    print(f"  floor={eps_floor}: viol={r['violation_rate'][0]:.4f}, "
          f"cbf_viol={r['cbf_violation_rate'][0]:.4f}, "
          f"reward={r['cumulative_reward'][0]:.1f}, "
          f"eps: {eps_start}→{eps_end}")

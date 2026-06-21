"""Reproduce S2 experiment with full 200-episode training + detailed eval logging."""
import sys
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import yaml
from pathlib import Path

from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.gp.gp_residual import GPResidual
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from experiments.phase4.methods import (
    _make_ccs_env, _make_robust_hocbf, _pretrain_gp,
    _collect_gp_data, _rollout_no_qp, _rollout_with_qp,
    CBF_PROTECTED, _count_violations, _make_hocbf,
)

CBF_PROTECTED_SET = CBF_PROTECTED

# Load config
config_path = Path("/home/gpu/sz_workspace/RoCBF-Net/configs/phase4.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

# Setup
seed = 0
key = jax.random.key(seed)
scenario = "pressure_oscillation"
dynamics, constraint = _make_ccs_env(1.0, 0, scenario)
x0, u0 = dynamics.equilibrium(1.0)

method_cfg = config.get('methods_config', {}).get('rocbf_net', {})
hocbf_cfg = config.get('hocbf', {})
gp_cfg = config.get('gp', {})
train_cfg = config.get('training', {})
eval_cfg = config.get('evaluation', {})

print(f"Equilibrium: x0[:3] = {x0[:3]}")
print(f"  u0 = {u0}")

c0 = constraint.check_all(x0[:3], u0)
print(f"  Constraints at equilibrium:")
for k, v in c0.items():
    print(f"    {k}: {v:.4f}")

# Pre-train GP (scenario-specific)
key, gp_key = jax.random.split(key)
gp = _pretrain_gp(1.0, 0, n_pretrain=2000, key=gp_key,
                   sigma_floor=1e-4, scenario_specific=True)

# Build safety layer
safety_layer = _make_robust_hocbf(dynamics, constraint, gp, u0,
                                   epsilon_kappa=1.0,
                                   k_pressure=(0.5, 0.5),
                                   k_enthalpy=(1.0,),
                                   u_max=100.0,
                                   use_mean_correction=True,
                                   epsilon_floor=0.0)

qp_solver = DifferentiableQP(v_max=5.0)

# Initialize model
model = ActorCritic(n_obs=3, n_act=3, hidden_dim=128, rngs=nnx.Rngs(0))
trainer = PPOTrainer(model, lr=1e-4, epochs=4, minibatch_size=64)

# Training loop (same as run_single)
n_episodes = train_cfg.get('max_episodes', 200)
n_steps = train_cfg.get('n_steps', 200)
gp_update_interval = method_cfg.get('gp_update_interval', 50)

print(f"\nTraining for {n_episodes} episodes...")
reward_history = []
epsilon_log = []

for ep in range(n_episodes):
    # For rocbf_net with scenario_specific_gp, train on deployment scenario
    train_dyn = UncertainUSCCSDynamics(
        delay_order=0, load_ratio=1.0,
        uncertainty_scenario=scenario)

    key, rollout_key = jax.random.split(key)
    rollout, ep_reward, violations, cbf_viol, _ = _rollout_no_qp(
        model, train_dyn, constraint, x0, u0, rollout_key, n_steps)

    if rollout['obs'].shape[0] < 2:
        continue

    reward_history.append(ep_reward)

    # PPO update
    advantages, returns = compute_gae(rollout['rewards'], rollout['values'], rollout['dones'])
    batch = {
        'obs': rollout['obs'],
        'actions': rollout['actions'],
        'old_log_probs': rollout['log_probs'],
        'advantages': advantages,
        'returns': returns,
    }
    for _ in range(trainer.epochs):
        loss = trainer.train_step(batch)

    # GP update every 50 episodes
    if (ep + 1) % gp_update_interval == 0 and gp is not None:
        key, gp_key = jax.random.split(key)
        env_gp = UncertainUSCCSDynamics(
            delay_order=0, load_ratio=1.0,
            uncertainty_scenario=scenario)
        key, data_key = jax.random.split(gp_key)
        X_new, Y_new = _collect_gp_data(env_gp, 200, key=data_key)
        gp.incremental_update(X_new, Y_new, reoptimize_hyperparams=False)

        # Rebuild safety layer
        safety_layer = _make_robust_hocbf(dynamics, constraint, gp, u0,
                                           epsilon_kappa=1.0,
                                           k_pressure=(0.5, 0.5),
                                           k_enthalpy=(1.0,),
                                           u_max=100.0,
                                           use_mean_correction=True,
                                           epsilon_floor=0.0)

        # Log epsilon
        eps_vals = []
        for hocbf in safety_layer.robust_hocbf_list:
            try:
                eps_vals.append(float(hocbf.compute_epsilon(x0[:3])))
            except:
                pass
        if eps_vals:
            mu, sigma = gp.predict(x0[:3].reshape(1, -1))
            print(f"  Ep {ep+1}: reward={ep_reward:.1f}, eps={sum(eps_vals):.4f}, "
                  f"mu_gp={float(jnp.mean(mu)):.4f}, sigma_gp={float(jnp.mean(sigma)):.6f}, "
                  f"gp_pts={gp.n_training_points}")
            epsilon_log.append({
                'episode': ep + 1,
                'epsilon_total': sum(eps_vals),
                'epsilon_per_constraint': eps_vals,
                'mu_gp_mean': float(jnp.mean(mu)),
                'sigma_gp_mean': float(jnp.mean(sigma)),
            })

    if (ep + 1) % 50 == 0 and (ep + 1) % gp_update_interval != 0:
        print(f"  Ep {ep+1}: reward={ep_reward:.1f}")

# Evaluate with QP
print(f"\n{'='*60}")
print("Evaluating with QP safety filter (3 episodes, 200 steps each)")
print(f"{'='*60}")

all_violation_rates = []
all_cbf_violation_rates = []

for eval_ep in range(3):
    key, ep_key = jax.random.split(key)
    rollout, ep_reward, violations, cbf_violations, qp_times = _rollout_with_qp(
        model, dynamics, safety_layer, qp_solver, constraint,
        x0, u0, ep_key, n_steps=200, use_scipy=True)

    n_actual = rollout['obs'].shape[0]
    vr = violations / max(n_actual, 1)
    cvr = cbf_violations / max(n_actual, 1)
    all_violation_rates.append(vr)
    all_cbf_violation_rates.append(cvr)

    print(f"  Eval ep {eval_ep}: violations={violations}/{n_actual} ({vr:.1%}), "
          f"cbf_violations={cbf_violations}/{n_actual} ({cvr:.1%}), reward={ep_reward:.1f}")

    # Show min barrier value
    if rollout['constraint_vals']:
        min_b = min(min(float(v) for v in cv.values()) for cv in rollout['constraint_vals'])
        print(f"    min_barrier = {min_b:.4f}")

print(f"\nOverall: total_viol={np.mean(all_violation_rates):.1%}±{np.std(all_violation_rates):.1%}, "
      f"cbf_viol={np.mean(all_cbf_violation_rates):.1%}±{np.std(all_cbf_violation_rates):.1%}")

# Detailed trace of FIRST CBF violation
print(f"\n{'='*60}")
print("Detailed trace: finding first CBF violation")
print(f"{'='*60}")

key, trace_key = jax.random.split(key)
x = x0
for t in range(200):
    key, action_key = jax.random.split(key)
    v_rl, log_prob, value = model.get_action(x[:3], action_key)

    A, b = safety_layer.qp_matrices(x[:3])
    v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)

    next_x = dynamics.step_stabilized(x[:3], v_safe)
    u_total = dynamics.compute_total_control(x[:3], v_safe)
    c = constraint.check_all(next_x, u_total)

    cbf_violated = {k: v for k, v in c.items() if v < 0 and k in CBF_PROTECTED_SET}

    if cbf_violated or t < 3 or t == 199:
        dv = v_safe - v_rl
        print(f"  t={t}: x={x[:3]}, v_rl={v_rl}, v_safe={v_safe}, Δv={dv}")
        print(f"         b={b}, constraints={dict((k,f'{v:.4f}') for k,v in c.items())}")
        if cbf_violated:
            print(f"         *** CBF VIOLATED: {cbf_violated} ***")
            if t > 5:  # Found first violation after initial steps, stop
                break

    x = next_x

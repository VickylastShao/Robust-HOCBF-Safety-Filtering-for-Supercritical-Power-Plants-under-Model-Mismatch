"""Quick test: RoCBF-Net with epsilon_floor=0.3, 1 seed, S1:Heat, 50 episodes.

Validates that:
1. epsilon_floor keeps QP feasible during online GP updates
2. ε decreases over training (validates paper claim)
3. Violation rate is reasonable
"""
import sys
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import json
import time
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.gp.gp_residual import GPResidual
from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from experiments.phase4.methods import (
    _make_ccs_env, _make_robust_hocbf, _pretrain_gp,
    _rollout_with_qp, _rollout_no_qp, _collect_gp_data, SCENARIOS,
)


def main():
    load_ratio = 1.0
    delay_order = 0
    scenario = 'heat_absorption'
    epsilon_floor = 0.3
    sigma_floor = 1e-4

    dynamics, constraint = _make_ccs_env(load_ratio, delay_order, scenario)
    x0, u0 = dynamics.equilibrium(load_ratio)

    # Pre-train GP
    key = jax.random.key(42)
    gp = _pretrain_gp(load_ratio, delay_order, n_pretrain=2000, key=key, sigma_floor=sigma_floor)

    # Create safety layer with epsilon_floor
    safety_layer = _make_robust_hocbf(
        dynamics, constraint, gp, u0,
        epsilon_kappa=1.0, k_pressure=[0.5, 0.5],
        k_enthalpy=[2.0], u_max=100.0, use_mean_correction=False,
        epsilon_floor=epsilon_floor)

    # Log initial epsilon
    eps_vals = []
    for hocbf in safety_layer.robust_hocbf_list:
        eps_vals.append(float(hocbf.compute_epsilon(x0[:3])))
    eps_total = sum(eps_vals)
    print(f"Initial epsilon (N={gp.n_training_points}): total={eps_total:.4f}, per={eps_vals}")

    # Create PPO model
    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=128, rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=1e-4, epochs=4, minibatch_size=64)
    qp_solver = DifferentiableQP(v_max=5.0)

    # Training loop
    n_episodes = 50
    gp_update_interval = 50
    n_steps = 200
    epsilon_log = []

    print(f"\nTraining {n_episodes} episodes on S1:Heat with epsilon_floor={epsilon_floor}")
    print("=" * 70)

    for ep in range(n_episodes):
        key, scenario_key = jax.random.split(key)
        scenario_idx = jax.random.randint(scenario_key, (), 0, len(SCENARIOS))
        train_scenario = SCENARIOS[int(scenario_idx)]
        train_dyn = UncertainUSCCSDynamics(
            delay_order=delay_order, load_ratio=load_ratio,
            uncertainty_scenario=train_scenario)

        key, rollout_key = jax.random.split(key)
        rollout, ep_reward, violations, _ = _rollout_no_qp(
            model, train_dyn, constraint, x0, u0, rollout_key, n_steps)

        if rollout['obs'].shape[0] < 2:
            continue

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
            X_new_list, Y_new_list = [], []
            for sc in SCENARIOS:
                env_gp = UncertainUSCCSDynamics(
                    delay_order=delay_order, load_ratio=load_ratio,
                    uncertainty_scenario=sc)
                key, data_key = jax.random.split(gp_key)
                X_new, Y_new = _collect_gp_data(env_gp, 200, key=data_key)
                X_new_list.append(X_new)
                Y_new_list.append(Y_new)
            X_new = jnp.concatenate(X_new_list, axis=0)
            Y_new = jnp.concatenate(Y_new_list, axis=0)
            gp.incremental_update(X_new, Y_new, reoptimize_hyperparams=False)

            # Log epsilon
            eps_vals = []
            for hocbf in safety_layer.robust_hocbf_list:
                try:
                    eps_vals.append(float(hocbf.compute_epsilon(x0[:3])))
                except Exception:
                    pass
            eps_total = sum(eps_vals)
            epsilon_log.append({
                'episode': ep + 1,
                'n_gp_points': gp.n_training_points,
                'epsilon_total': eps_total,
                'epsilon_per_constraint': eps_vals,
            })
            print(f"  GP UPDATE at ep {ep+1}: eps_total={eps_total:.4f}, "
                  f"per={[f'{e:.3f}' for e in eps_vals]}, N={gp.n_training_points}")

        if (ep + 1) % 10 == 0:
            print(f"  Ep {ep+1}: reward={ep_reward:.1f}")

    # Evaluation with QP filter
    print("\n" + "=" * 70)
    print("EVALUATION (with QP safety filter)")
    print("=" * 70)

    n_eval = 3
    n_eval_steps = 200
    all_violations = []
    all_rewards = []

    for eval_ep in range(n_eval):
        key, ep_key = jax.random.split(key)
        rollout, ep_reward, violations, qp_times = _rollout_with_qp(
            model, dynamics, safety_layer, qp_solver, constraint,
            x0, u0, ep_key, n_eval_steps, use_scipy=True)
        n_actual = rollout['obs'].shape[0]
        vr = violations / max(n_actual, 1)
        all_violations.append(vr)
        all_rewards.append(ep_reward)
        print(f"  Eval ep {eval_ep+1}: violation_rate={vr*100:.2f}%, reward={ep_reward:.1f}")

    mean_vr = sum(all_violations) / len(all_violations) * 100
    mean_r = sum(all_rewards) / len(all_rewards)
    print(f"\n  MEAN: violation_rate={mean_vr:.2f}%, reward={mean_r:.1f}")

    # Compare with no-floor baseline
    print("\n" + "=" * 70)
    print("COMPARISON: epsilon_floor=0.0 (no floor)")
    print("=" * 70)

    key2 = jax.random.key(42)
    gp2 = _pretrain_gp(load_ratio, delay_order, n_pretrain=2000, key=key2, sigma_floor=sigma_floor)
    safety_layer2 = _make_robust_hocbf(
        dynamics, constraint, gp2, u0,
        epsilon_kappa=1.0, k_pressure=[0.5, 0.5],
        k_enthalpy=[2.0], u_max=100.0, use_mean_correction=False,
        epsilon_floor=0.0)
    model2 = ActorCritic(n_obs=3, n_act=3, hidden_dim=128, rngs=nnx.Rngs(0))
    trainer2 = PPOTrainer(model2, lr=1e-4, epochs=4, minibatch_size=64)

    # Quick training: same 50 episodes
    epsilon_log2 = []
    for ep in range(n_episodes):
        key2, scenario_key = jax.random.split(key2)
        scenario_idx = jax.random.randint(scenario_key, (), 0, len(SCENARIOS))
        train_scenario = SCENARIOS[int(scenario_idx)]
        train_dyn = UncertainUSCCSDynamics(
            delay_order=delay_order, load_ratio=load_ratio,
            uncertainty_scenario=train_scenario)

        key2, rollout_key = jax.random.split(key2)
        rollout, ep_reward, violations, _ = _rollout_no_qp(
            model2, train_dyn, constraint, x0, u0, rollout_key, n_steps)
        if rollout['obs'].shape[0] < 2:
            continue
        advantages, returns = compute_gae(
            rollout['rewards'], rollout['values'], rollout['dones'])
        batch = {
            'obs': rollout['obs'],
            'actions': rollout['actions'],
            'old_log_probs': rollout['log_probs'],
            'advantages': advantages,
            'returns': returns,
        }
        for _ in range(trainer2.epochs):
            loss = trainer2.train_step(batch)

        if (ep + 1) % gp_update_interval == 0:
            key2, gp_key = jax.random.split(key2)
            X_new_list, Y_new_list = [], []
            for sc in SCENARIOS:
                env_gp = UncertainUSCCSDynamics(
                    delay_order=delay_order, load_ratio=load_ratio,
                    uncertainty_scenario=sc)
                key2, data_key = jax.random.split(gp_key)
                X_new, Y_new = _collect_gp_data(env_gp, 200, key=data_key)
                X_new_list.append(X_new)
                Y_new_list.append(Y_new)
            X_new = jnp.concatenate(X_new_list, axis=0)
            Y_new = jnp.concatenate(Y_new_list, axis=0)
            gp2.incremental_update(X_new, Y_new, reoptimize_hyperparams=False)

            eps_vals = []
            for hocbf in safety_layer2.robust_hocbf_list:
                try:
                    eps_vals.append(float(hocbf.compute_epsilon(x0[:3])))
                except Exception:
                    pass
            eps_total = sum(eps_vals)
            epsilon_log2.append({
                'episode': ep + 1,
                'epsilon_total': eps_total,
            })
            print(f"  GP UPDATE at ep {ep+1}: eps_total={eps_total:.4f}")

    # Evaluate no-floor
    for eval_ep in range(n_eval):
        key2, ep_key = jax.random.split(key2)
        rollout, ep_reward, violations, qp_times = _rollout_with_qp(
            model2, dynamics, safety_layer2, qp_solver, constraint,
            x0, u0, ep_key, n_eval_steps, use_scipy=True)
        n_actual = rollout['obs'].shape[0]
        vr = violations / max(n_actual, 1)
        all_violations.append(vr)

    mean_vr2 = sum(all_violations[3:]) / 3 * 100
    print(f"\n  No-floor MEAN: violation_rate={mean_vr2:.2f}%")
    print(f"  With-floor MEAN: violation_rate={mean_vr:.2f}%")

    # Print epsilon trajectory comparison
    print("\n" + "=" * 70)
    print("EPSILON TRAJECTORY SUMMARY")
    print("=" * 70)
    print(f"  With floor={epsilon_floor}:")
    for entry in epsilon_log:
        print(f"    Ep {entry['episode']}: eps={entry['epsilon_total']:.4f}, "
              f"per={[f'{e:.3f}' for e in entry['epsilon_per_constraint']]}")
    print(f"  Without floor:")
    for entry in epsilon_log2:
        print(f"    Ep {entry['episode']}: eps={entry['epsilon_total']:.4f}")


if __name__ == "__main__":
    main()

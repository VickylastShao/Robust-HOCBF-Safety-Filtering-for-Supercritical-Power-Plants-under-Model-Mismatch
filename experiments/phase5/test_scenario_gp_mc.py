"""Diagnostic: test scenario-specific GP + MC=True with noise_variance floor.

This is the theoretically correct configuration:
- Scenario-specific GP: μ_GP matches the actual operating scenario
- MC=True: uses f̂ = f0 + μ_GP (correct drift)
- noise_variance floor: σ includes model misspecification margin
- Online GP updates: gradually reduce ε as more data is collected

Key question: does ε decrease over time while maintaining safety?

Config: s1_heat, noise_variance=1e-4 (gives σ_floor ≈ 0.01, ε ≈ 0.42)
"""
import sys
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import yaml

from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.gp.gp_residual import GPResidual
from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from envs.ccs.agc_schedule import AGCSchedule
from experiments.phase4.methods import (
    _make_ccs_env, _make_robust_hocbf, _rollout_with_qp, _rollout_no_qp,
    SCENARIOS,
)


def pretrain_gp_scenario(dynamics, n_pretrain=2000, noise_variance=1e-4, key=None):
    """Pre-train GP on a single scenario's data."""
    if key is None:
        key = jax.random.key(42)
    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)
    X_list, Y_list = [], []
    x = x0[:3]
    for _ in range(n_pretrain):
        key, v_key = jax.random.split(key)
        v = jnp.array([
            jax.random.uniform(v_key, (), minval=-2.0, maxval=2.0),
            jax.random.uniform(v_key, (), minval=-5.0, maxval=5.0),
            jax.random.uniform(v_key, (), minval=-1.0, maxval=1.0),
        ])
        x_next = dynamics.step_stabilized(x, v)
        x_pred = dynamics._x0 + dynamics._A_d @ (x - dynamics._x0) + dynamics._B_d @ v
        residual = (x_next - x_pred) / dynamics.dt
        X_list.append(x)
        Y_list.append(residual)
        if jnp.any(jnp.abs(x_next - x0[:3]) > jnp.array([30.0, 5.0, 300.0])):
            key, reset_key = jax.random.split(key)
            x = x0[:3] + jnp.array([5.0, 0.5, 50.0]) * jax.random.normal(reset_key, (3,))
        else:
            x = x_next
    X = jnp.stack(X_list)
    Y = jnp.stack(Y_list)
    gp = GPResidual(n_dims=3, noise_variance=noise_variance)
    gp.fit(X, Y)
    return gp


def collect_gp_data_scenario(dynamics, n_transitions=200, key=None):
    """Collect GP training data from a single scenario's dynamics."""
    if key is None:
        key = jax.random.key(0)
    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)
    X_list, Y_list = [], []
    x = x0[:3]
    for _ in range(n_transitions):
        key, v_key = jax.random.split(key)
        v = jnp.array([
            jax.random.uniform(v_key, (), minval=-2.0, maxval=2.0),
            jax.random.uniform(v_key, (), minval=-5.0, maxval=5.0),
            jax.random.uniform(v_key, (), minval=-1.0, maxval=1.0),
        ])
        x_next = dynamics.step_stabilized(x, v)
        x_pred = dynamics._x0 + dynamics._A_d @ (x - dynamics._x0) + dynamics._B_d @ v
        residual = (x_next - x_pred) / dynamics.dt
        X_list.append(x)
        Y_list.append(residual)
        if jnp.any(jnp.abs(x_next - x0[:3]) > jnp.array([30.0, 5.0, 300.0])):
            key, reset_key = jax.random.split(key)
            x = x0[:3] + jnp.array([5.0, 0.5, 50.0]) * jax.random.normal(reset_key, (3,))
        else:
            x = x_next
    return jnp.stack(X_list), jnp.stack(Y_list)


def compute_epsilon_at_state(gp, multi_hocbf, x):
    """Compute epsilon at a given state."""
    mu_gp, sigma_gp = gp.predict(x[:3])
    beta = GPResidual.compute_beta(gp.n_dims, gp.n_training_points, delta=0.01)
    epsilons = []
    for hocbf in multi_hocbf.robust_hocbf_list:
        try:
            eps = float(hocbf.compute_epsilon(x[:3]))
            epsilons.append(eps)
        except Exception:
            epsilons.append(float('nan'))
    eps_total = sum(e for e in epsilons if e == e)
    sigma_mean = float(jnp.mean(sigma_gp))
    sigma_max = float(jnp.max(sigma_gp))
    return {
        'epsilon_total': eps_total,
        'sigma_gp_mean': sigma_mean,
        'sigma_gp_max': sigma_max,
        'n_gp_points': gp.n_training_points,
        'beta': float(beta),
    }


def run_experiment(method_name, noise_variance=1e-4, use_mc=True, seed=0,
                   do_online_gp=True, condition='s1_heat'):
    """Run one experiment with scenario-specific GP + MC."""
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'phase4.yaml'
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    CONDITION_SCENARIO_MAP = {
        'nominal': None,
        's1_heat': 'heat_absorption',
        's2_pressure': 'pressure_oscillation',
        's3_coupled': 'coupled',
        's4_nonlinear': 'nonlinear',
    }

    scenario = CONDITION_SCENARIO_MAP.get(condition)
    load_ratio = 1.0
    delay_order = 0

    if scenario is not None:
        dynamics, constraint = _make_ccs_env(load_ratio, delay_order, scenario)
    else:
        dynamics, constraint = _make_ccs_env(load_ratio, delay_order)

    x0, u0 = dynamics.equilibrium(load_ratio)
    key = jax.random.key(seed)

    # Pre-train GP on scenario-specific data
    gp = pretrain_gp_scenario(dynamics, n_pretrain=2000,
                              noise_variance=noise_variance, key=key)
    print(f"  GP pre-trained on {condition}: N={gp.n_training_points}, noise_var={noise_variance}")

    # Log initial epsilon
    k_p = [0.5, 0.5]
    k_h = [2.0]
    u_max = 100.0

    safety_layer = _make_robust_hocbf(
        dynamics, constraint, gp, u0,
        epsilon_kappa=1.0, k_pressure=k_p,
        k_enthalpy=k_h, u_max=u_max, use_mean_correction=use_mc)

    eps_info = compute_epsilon_at_state(gp, safety_layer, x0)
    print(f"  Initial epsilon: {eps_info}")

    # Initialize model and train
    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=128, rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=1e-4, epochs=4, minibatch_size=64)
    qp_solver = DifferentiableQP(v_max=5.0)

    n_episodes = 200
    n_steps = 200
    gp_update_interval = 50
    epsilon_log = []

    for ep in range(n_episodes):
        # Training with random scenario sampling
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

        # Online GP update
        if (do_online_gp and method_name == 'rocbf_net' and
                (ep + 1) % gp_update_interval == 0 and gp is not None):
            key, gp_key = jax.random.split(key)
            X_new, Y_new = collect_gp_data_scenario(dynamics, 200, key=gp_key)
            gp.incremental_update(X_new, Y_new, reoptimize_hyperparams=False)
            safety_layer = _make_robust_hocbf(
                dynamics, constraint, gp, u0,
                epsilon_kappa=1.0, k_pressure=k_p,
                k_enthalpy=k_h, u_max=u_max, use_mean_correction=use_mc)

        # Log epsilon
        if (ep + 1) % 10 == 0:
            eps_info = compute_epsilon_at_state(gp, safety_layer, x0)
            eps_info['episode'] = ep + 1
            epsilon_log.append(eps_info)

        if (ep + 1) % 50 == 0:
            gp_info = f", GP_N={gp.n_training_points}" if gp else ""
            print(f"  [{method_name}|{condition}] Ep {ep+1}: r={ep_reward:.1f}{gp_info}")

    # Evaluation
    all_violation_rates = []
    all_rewards = []
    for ep in range(3):
        key, ep_key = jax.random.split(key)
        rollout, ep_reward, violations, qp_times = _rollout_with_qp(
            model, dynamics, safety_layer, qp_solver, constraint,
            x0, u0, ep_key, 200, use_scipy=True)
        n_actual = rollout['obs'].shape[0]
        all_violation_rates.append(violations / max(n_actual, 1))
        all_rewards.append(ep_reward)

    vr = (float(np.mean(all_violation_rates)), float(np.std(all_violation_rates)))
    rw = (float(np.mean(all_rewards)), float(np.std(all_rewards)))

    print(f"\n  RESULT: violation={vr[0]:.4f}+/-{vr[1]:.4f}, reward={rw[0]:.1f}")
    if epsilon_log:
        first_eps = epsilon_log[0]['epsilon_total']
        last_eps = epsilon_log[-1]['epsilon_total']
        print(f"  EPSILON: {first_eps:.4f} -> {last_eps:.4f} "
              f"(change: {(last_eps/first_eps - 1)*100:.1f}%)")

    return {
        'violation_rate': vr,
        'reward': rw,
        'epsilon_log': epsilon_log,
        'final_gp_n': gp.n_training_points if gp else 0,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='rocbf_net')
    parser.add_argument('--noise-variance', type=float, default=1e-4)
    parser.add_argument('--no-mc', action='store_true')
    parser.add_argument('--no-online-gp', action='store_true')
    parser.add_argument('--condition', default='s1_heat')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Method: {args.method}, noise_var={args.noise_variance}, "
          f"MC={not args.no_mc}, online_GP={not args.no_online_gp}")
    print(f"{'='*60}")

    result = run_experiment(
        args.method,
        noise_variance=args.noise_variance,
        use_mc=not args.no_mc,
        do_online_gp=not args.no_online_gp,
        condition=args.condition,
        seed=args.seed,
    )

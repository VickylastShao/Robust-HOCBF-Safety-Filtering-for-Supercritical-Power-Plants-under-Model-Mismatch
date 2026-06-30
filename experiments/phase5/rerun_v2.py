"""Phase 5: Re-run with scenario-specific GP + mean correction.

Key fix: The GP should be trained on the evaluation scenario's data only
(not mixed scenarios), matching real deployment where the GP observes data
from the actual plant operating condition. Both PPO-RHOCBF and RoCBF-Net
use mean correction; the only difference is online GP updates.

This makes the ablation table consistent with the code:
- PPO-RHOCBF: GP mean correction ✓, robustness margin ✓, online GP ✗
- RoCBF-Net:  GP mean correction ✓, robustness margin ✓, online GP ✓

Usage:
    conda run -n jax_gpu python experiments/phase5/rerun_v2.py \
        --methods rocbf_net ppo_rhocbf \
        --conditions s1_heat \
        --seeds 0 \
        --log-epsilon
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

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
)

CONDITION_SCENARIO_MAP = {
    'nominal': None,
    's1_heat': 'heat_absorption',
    's2_pressure': 'pressure_oscillation',
    's3_coupled': 'coupled',
    's4_nonlinear': 'nonlinear',
    'load_following': None,
}


def _pretrain_gp_scenario(dynamics, n_pretrain=2000, key=None):
    """Pre-train GP on a single scenario's data.

    Unlike _pretrain_gp from methods.py (which mixes all scenarios),
    this trains on the evaluation scenario only, matching real deployment
    where the GP observes data from the actual plant.
    """
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
        u = dynamics.compute_total_control(x, v)
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
    gp = GPResidual(n_dims=3, noise_variance=1e-4)
    gp.fit(X, Y)
    return gp


def _collect_gp_data_scenario(dynamics, n_transitions=200, key=None):
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
        u = dynamics.compute_total_control(x, v)
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


def compute_epsilon_at_state(gp, multi_hocbf, x, dynamics, constraint, u_max=100.0):
    """Compute epsilon(x) at a given state for logging."""
    mu_gp, sigma_gp = gp.predict(x[:3])
    beta = GPResidual.compute_beta(gp.n_dims, gp.n_training_points, delta=0.01)

    # Compute epsilon from each constraint
    epsilons = []
    for hocbf in multi_hocbf.robust_hocbf_list:
        try:
            eps = float(hocbf.compute_epsilon(x[:3]))
            epsilons.append(eps)
        except Exception:
            epsilons.append(float('nan'))

    epsilon_total = sum(e for e in epsilons if e == e)  # skip NaN
    sigma_mean = float(jnp.mean(sigma_gp))
    sigma_max = float(jnp.max(sigma_gp))

    return {
        'beta': float(beta),
        'sigma_gp_mean': sigma_mean,
        'sigma_gp_max': sigma_max,
        'sigma_gp_per_dim': [float(s) for s in sigma_gp],
        'n_gp_points': gp.n_training_points,
        'mu_gp_mean': float(jnp.mean(jnp.abs(mu_gp))),
        'mu_gp_per_dim': [float(m) for m in mu_gp],
        'epsilon_per_constraint': epsilons,
        'epsilon_total': epsilon_total,
    }


def run_single(method_name, condition, seed, config, log_epsilon=False):
    """Run one experiment with scenario-specific GP + mean correction."""
    cfg = config
    train_cfg = cfg.get('training', {})
    eval_cfg = cfg.get('evaluation', {})
    method_cfg = cfg.get('methods_config', {}).get(method_name, {})
    hocbf_cfg = cfg.get('hocbf', {})

    load_ratio = 1.0
    delay_order = 0
    scenario = CONDITION_SCENARIO_MAP.get(condition)

    # Create evaluation dynamics (scenario-specific)
    if scenario is not None:
        eval_dynamics, constraint = _make_ccs_env(load_ratio, delay_order, scenario)
    else:
        eval_dynamics, constraint = _make_ccs_env(load_ratio, delay_order)

    x0, u0 = eval_dynamics.equilibrium(load_ratio)

    agc_schedule = None
    if condition == 'load_following':
        agc_cfg = cfg.get('agc_schedule', {})
        agc_schedule = AGCSchedule(
            base_load=agc_cfg.get('base_load', 1000.0),
            ramp_rate=agc_cfg.get('ramp_rate', 5.0),
            regulation_amp=agc_cfg.get('regulation_amp', 20.0),
            regulation_period=agc_cfg.get('regulation_period', 300.0),
        )

    key = jax.random.key(seed)

    # Pre-train GP on evaluation scenario's data (NOT mixed scenarios)
    gp = None
    gp_cfg = cfg.get('gp', {})
    if method_name in ('ppo_rhocbf', 'rocbf_net'):
        gp = _pretrain_gp_scenario(
            eval_dynamics,
            n_pretrain=method_cfg.get('n_pretrain', gp_cfg.get('n_pretrain', 2000)),
            key=key)
        print(f"  GP pre-trained on {condition} data: N={gp.n_training_points}", flush=True)

    # Initialize model
    model = ActorCritic(n_obs=3, n_act=3,
                        hidden_dim=method_cfg.get('hidden_dim', 128),
                        rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=method_cfg.get('lr', 1e-4),
                         epochs=method_cfg.get('epochs', 4),
                         minibatch_size=method_cfg.get('minibatch_size', 64))

    # Both PPO-RHOCBF and RoCBF-Net use mean correction with scenario-specific GP
    # The only difference is online GP updates (RoCBF-Net has them, PPO-RHOCBF doesn't)
    use_mc = True  # Both use mean correction
    k_p = hocbf_cfg.get('pressure_k_gains', [0.5, 0.5])
    k_h = hocbf_cfg.get('enthalpy_k_gains', [2.0])
    u_max = hocbf_cfg.get('u_max', 100.0)
    epsilon_kappa = method_cfg.get('epsilon_kappa', 1.0)

    safety_layer = _make_robust_hocbf(
        eval_dynamics, constraint, gp, u0,
        epsilon_kappa=epsilon_kappa, k_pressure=k_p,
        k_enthalpy=k_h, u_max=u_max, use_mean_correction=use_mc)

    qp_solver = DifferentiableQP(v_max=5.0)

    # Training loop
    n_episodes = train_cfg.get('max_episodes', 200)
    n_steps = train_cfg.get('n_steps', 200)
    eval_every = train_cfg.get('eval_every', 50)
    n_eval = eval_cfg.get('n_episodes', 3)
    actual_eval_steps = eval_cfg.get('n_steps', 200)
    gp_update_interval = method_cfg.get('gp_update_interval', 50)

    reward_history = []
    epsilon_log = []

    # Training dynamics: use evaluation scenario for consistency
    # (In real deployment, the plant operates under a specific condition)
    train_dynamics = eval_dynamics

    for ep in range(n_episodes):
        key, rollout_key = jax.random.split(key)
        rollout, ep_reward, violations, _ = _rollout_no_qp(
            model, train_dynamics, constraint, x0, u0, rollout_key, n_steps)

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

        # Online GP update (for RoCBF-Net only)
        if (method_name == 'rocbf_net' and
                (ep + 1) % gp_update_interval == 0 and gp is not None):
            key, gp_key = jax.random.split(key)
            # Collect data from the evaluation scenario (NOT all scenarios)
            X_new, Y_new = _collect_gp_data_scenario(
                eval_dynamics, n_transitions=500, key=gp_key)
            gp.incremental_update(X_new, Y_new)

            # Rebuild safety layer with updated GP
            safety_layer = _make_robust_hocbf(
                eval_dynamics, constraint, gp, u0,
                epsilon_kappa=epsilon_kappa, k_pressure=k_p,
                k_enthalpy=k_h, u_max=u_max, use_mean_correction=use_mc)

        # Log epsilon every 10 episodes
        if log_epsilon and (ep + 1) % 10 == 0 and gp is not None:
            try:
                eps_info = compute_epsilon_at_state(
                    gp, safety_layer, x0, eval_dynamics, constraint)
                eps_info['episode'] = ep + 1
                epsilon_log.append(eps_info)
            except Exception as e:
                print(f"  [Epsilon log error at ep {ep+1}]: {e}", flush=True)
                eps_info = {
                    'beta': float(GPResidual.compute_beta(
                        gp.n_dims, gp.n_training_points, delta=0.01)),
                    'n_gp_points': gp.n_training_points,
                    'episode': ep + 1,
                    'error': str(e),
                }
                epsilon_log.append(eps_info)

        # Periodic logging
        if (ep + 1) % eval_every == 0:
            gp_info = f", GP_N={gp.n_training_points}" if gp else ""
            print(f"  [{method_name}|{condition}|seed={seed}] "
                  f"Ep {ep+1}: r={ep_reward:.1f}{gp_info}", flush=True)

    # Evaluation with QP safety filter
    all_violation_rates = []
    all_rewards = []

    for ep in range(n_eval):
        key, ep_key = jax.random.split(key)
        rollout, ep_reward, violations, qp_times = _rollout_with_qp(
            model, eval_dynamics, safety_layer, qp_solver, constraint,
            x0, u0, ep_key, actual_eval_steps, agc_schedule=agc_schedule,
            use_scipy=True)
        n_actual = rollout['obs'].shape[0]
        all_violation_rates.append(violations / max(n_actual, 1))
        all_rewards.append(ep_reward)

    # Final epsilon log
    if log_epsilon and gp is not None:
        eps_info = compute_epsilon_at_state(
            gp, safety_layer, x0, eval_dynamics, constraint)
        eps_info['episode'] = n_episodes
        epsilon_log.append(eps_info)

    result = {
        'violation_rate': (float(np.mean(all_violation_rates)),
                          float(np.std(all_violation_rates))),
        'cumulative_reward': (float(np.mean(all_rewards)),
                             float(np.std(all_rewards))),
        'n_training_episodes': len(reward_history),
        'reward_history': reward_history[-100:],
        'epsilon_log': epsilon_log if log_epsilon else [],
        'final_gp_n_points': gp.n_training_points if gp else 0,
        'config': {
            'use_mean_correction': use_mc,
            'epsilon_kappa': epsilon_kappa,
            'gp_scenario_specific': True,
            'condition': condition,
        },
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='Phase 5 v2: scenario-specific GP + mean correction')
    parser.add_argument('--methods', nargs='+', default=['rocbf_net', 'ppo_rhocbf'])
    parser.add_argument('--conditions', nargs='+', default=['s1_heat'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--log-epsilon', action='store_true', default=True)
    parser.add_argument('--results-dir', default='results/phase5_v2/')
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent.parent / 'configs' / 'phase4.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.methods) * len(args.conditions) * len(args.seeds)
    count = 0

    for method_name in args.methods:
        for condition in args.conditions:
            for seed in args.seeds:
                count += 1
                print(f"\n[{count}/{total}] {method_name} | {condition} | seed={seed}",
                      flush=True)

                try:
                    t0 = time.time()
                    result = run_single(method_name, condition, seed, config,
                                       log_epsilon=args.log_epsilon)
                    elapsed = time.time() - t0

                    # Save result
                    out_path = results_dir / f'{method_name}_{condition}_seed{seed}.json'
                    def _convert(obj):
                        if isinstance(obj, (jnp.ndarray,)):
                            return obj.tolist()
                        if isinstance(obj, tuple):
                            return list(obj)
                        if isinstance(obj, dict):
                            return {k: _convert(v) for k, v in obj.items()}
                        if isinstance(obj, list):
                            return [_convert(v) for v in obj]
                        return obj
                    with open(out_path, 'w') as f:
                        json.dump(_convert(result), f, indent=2)

                    vr = result['violation_rate']
                    gp_n = result['final_gp_n_points']
                    print(f"  -> violation={vr[0]:.4f}+/-{vr[1]:.4f}, "
                          f"reward={result['cumulative_reward'][0]:.1f}, "
                          f"GP_N={gp_n}, time={elapsed:.0f}s", flush=True)

                except Exception as e:
                    print(f"  FAILED: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

    print(f"\n=== Done: {count}/{total} experiments ===", flush=True)


if __name__ == "__main__":
    main()

"""Optimized Phase 4 experiment runner.

Key optimization: PPO, PPO-HOCBF, PPO-GP-HOCBF, PPO-RHOCBF all use
identical PPO training (no QP filter during training). We train PPO once
per (condition, seed) and reuse the trained model with different safety
layers at evaluation time.

Only PPO-Lagrangian and RoCBF-Net need separate training:
- PPO-Lagrangian adds cost terms to PPO loss
- RoCBF-Net does online GP updates during training

Training runs: 3 × 6 × 5 = 90 (instead of 7 × 6 × 5 = 210)
Evaluation runs: 8 × 6 × 5 = 240 (unchanged)
"""
import json
import sys
import time
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import yaml
from pathlib import Path

from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.gp.gp_residual import GPResidual
from rocbf.baselines.ppo_lagrangian import PPOTrainerLagrangian, compute_step_costs
from rocbf.baselines.nmpc import NMPCController
from rocbf.baselines.ppo_cbf import make_first_order_cbf
from rocbf.cbf.hocbf import HOCBF
from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintHOCBF, MultiConstraintRobustHOCBF
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from envs.ccs.agc_schedule import AGCSchedule
from experiments.phase4.methods import (
    METHODS, METHOD_LABELS, SCENARIOS, SCENARIO_LABELS,
    _make_ccs_env, _make_hocbf, _make_robust_hocbf, _pretrain_gp,
    _rollout_with_qp, _rollout_no_qp,
)

CONDITIONS = ['nominal', 's1_heat', 's2_pressure', 's3_coupled',
              's4_nonlinear', 'load_following']

CONDITION_SCENARIO_MAP = {
    'nominal': None,
    's1_heat': 'heat_absorption',
    's2_pressure': 'pressure_oscillation',
    's3_coupled': 'coupled',
    's4_nonlinear': 'nonlinear',
    'load_following': None,
}

# Methods that can reuse PPO's trained model (identical training loop)
PPO_REUSE_METHODS = {'ppo', 'ppo_cbf', 'ppo_hocbf', 'ppo_gp_hocbf', 'ppo_rhocbf'}


def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'phase4.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


def _train_ppo(config, dynamics, constraint, x0, u0, key, n_episodes, n_steps):
    """Train a PPO model. Returns (model, reward_history)."""
    train_cfg = config.get('training', {})
    method_cfg = config.get('methods_config', {}).get('ppo', {})

    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=method_cfg.get('hidden_dim', 128),
                        rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=method_cfg.get('lr', 1e-4),
                         epochs=method_cfg.get('epochs', 4),
                         minibatch_size=method_cfg.get('minibatch_size', 64))
    load_ratio = dynamics._load_ratio
    delay_order = dynamics.delay_order
    reward_history = []

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

    return model, trainer, reward_history


def _train_ppo_lagr(config, dynamics, constraint, x0, u0, key, n_episodes, n_steps):
    """Train PPO-Lagrangian model. Returns (model, trainer, reward_history)."""
    method_cfg = config.get('methods_config', {}).get('ppo_lagr', {})

    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=method_cfg.get('hidden_dim', 128),
                        rngs=nnx.Rngs(0))
    trainer = PPOTrainerLagrangian(
        model, lr=method_cfg.get('lr', 1e-4),
        cost_limit=method_cfg.get('cost_limit', 0.0),
        lagrangian_lr=method_cfg.get('lagrangian_lr', 0.01),
        epochs=method_cfg.get('epochs', 4),
        minibatch_size=method_cfg.get('minibatch_size', 64))
    load_ratio = dynamics._load_ratio
    delay_order = dynamics.delay_order
    reward_history = []

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
        if 'constraint_vals' in rollout:
            batch['costs'] = compute_step_costs(rollout['constraint_vals'])
        for _ in range(trainer.epochs):
            loss = trainer.train_step(batch)

    return model, trainer, reward_history


def _train_rocbf_net(config, dynamics, constraint, x0, u0, key, gp,
                      n_episodes, n_steps):
    """Train RoCBF-Net model (PPO + online GP updates)."""
    method_cfg = config.get('methods_config', {}).get('rocbf_net', {})

    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=method_cfg.get('hidden_dim', 128),
                        rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=method_cfg.get('lr', 1e-4),
                         epochs=method_cfg.get('epochs', 4),
                         minibatch_size=method_cfg.get('minibatch_size', 64))
    load_ratio = dynamics._load_ratio
    delay_order = dynamics.delay_order
    gp_update_interval = method_cfg.get('gp_update_interval', 50)
    reward_history = []

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

        # Online GP update
        if (ep + 1) % gp_update_interval == 0 and gp is not None:
            key, gp_key = jax.random.split(key)
            X_new_list, Y_new_list = [], []
            for sc in SCENARIOS:
                env_gp = UncertainUSCCSDynamics(
                    delay_order=delay_order, load_ratio=load_ratio,
                    uncertainty_scenario=sc)
                key, data_key = jax.random.split(gp_key)
                from experiments.phase4.methods import _collect_gp_data
                X_new, Y_new = _collect_gp_data(env_gp, 200, key=data_key)
                X_new_list.append(X_new)
                Y_new_list.append(Y_new)
            X_new = jnp.concatenate(X_new_list, axis=0)
            Y_new = jnp.concatenate(Y_new_list, axis=0)
            gp.incremental_update(X_new, Y_new)

    return model, trainer, reward_history


def _build_safety_layer(method_name, dynamics, constraint, gp, u0,
                         method_cfg, hocbf_cfg):
    """Build safety layer for a method."""
    k_p = tuple(hocbf_cfg.get('pressure_k_gains', [0.5, 0.5]))
    k_h = tuple(hocbf_cfg.get('enthalpy_k_gains', [2.0]))
    u_max = hocbf_cfg.get('u_max', 100.0)

    if method_name == 'ppo_cbf':
        return make_first_order_cbf(constraint, dynamics, u0)
    elif method_name == 'ppo_hocbf':
        return _make_hocbf(dynamics, constraint, u0, k_p, k_h)
    elif method_name == 'ppo_gp_hocbf':
        return _make_robust_hocbf(dynamics, constraint, gp, u0,
                                   epsilon_kappa=0.0, k_pressure=k_p,
                                   k_enthalpy=k_h, u_max=u_max,
                                   use_mean_correction=True)
    elif method_name == 'ppo_rhocbf':
        return _make_robust_hocbf(dynamics, constraint, gp, u0,
                                   epsilon_kappa=1.0, k_pressure=k_p,
                                   k_enthalpy=k_h, u_max=u_max,
                                   use_mean_correction=False)
    elif method_name == 'rocbf_net':
        kappa = method_cfg.get('epsilon_kappa', 1.0)
        return _make_robust_hocbf(dynamics, constraint, gp, u0,
                                   epsilon_kappa=kappa, k_pressure=k_p,
                                   k_enthalpy=k_h, u_max=u_max,
                                   use_mean_correction=False)
    return None


def _rmse(values, target):
    import numpy as np
    arr = np.array(values)
    return float(np.sqrt(np.mean((arr - target) ** 2)))


def _mean_std(values):
    import numpy as np
    arr = np.array(values, dtype=float)
    return (float(np.mean(arr)), float(np.std(arr)))


def evaluate_method(model, safety_layer, qp_solver, dynamics, constraint,
                    x0, u0, key, method_name, agc_schedule=None,
                    n_episodes=3, n_steps=200, hocbf_cfg=None,
                    method_cfg=None, gp=None):
    """Evaluate a trained policy with given safety layer."""
    all_violation_rates = []
    all_rewards = []
    all_tracking = {'pressure': [], 'enthalpy': [], 'power': []}
    all_control_costs = []
    all_min_barrier = []
    all_online_times = []

    is_nmpc = method_name == 'nmpc'

    for ep in range(n_episodes):
        key, ep_key = jax.random.split(key)

        if is_nmpc:
            nmpc = safety_layer
            x = x0
            violations = 0
            ep_reward = 0.0
            barrier_vals = []
            pressures, enthalpies, powers = [], [], []
            ctrl_costs = []
            online_times = []

            for t in range(n_steps):
                t0 = time.perf_counter()
                if agc_schedule is not None:
                    target_load = agc_schedule.get_reference(float(t))
                    x_ref, u_target = dynamics.equilibrium(target_load / 1000.0)
                    y_ref = dynamics.output(x_ref, u_target)
                    v_opt = nmpc.compute_action(x, y_ref)
                else:
                    v_opt = nmpc.compute_action(x)
                online_times.append((time.perf_counter() - t0) * 1000)

                next_x = dynamics.step_stabilized(x[:3], v_opt)
                u_total = dynamics.compute_total_control(x[:3], v_opt)

                constraint_vals = constraint.check_all(next_x, u_total)
                if any(v < 0 for v in constraint_vals.values()):
                    violations += 1

                y = dynamics.output(next_x, u_total)
                if agc_schedule is not None:
                    target_load = agc_schedule.get_reference(float(t))
                    x_ref, u_target = dynamics.equilibrium(target_load / 1000.0)
                    y_ref = dynamics.output(x_ref, u_target)
                else:
                    y_ref = dynamics.output(x0, u0)
                reward = (
                    -1.0 * (y[0] - y_ref[0]) ** 2
                    - 0.001 * (y[1] - y_ref[1]) ** 2
                    - 0.01 * (y[2] - y_ref[2]) ** 2
                    - 0.0001 * jnp.sum(v_opt ** 2)
                )
                ep_reward += float(reward)

                pressures.append(float(y[0]))
                enthalpies.append(float(y[1]))
                powers.append(float(y[2]))
                ctrl_costs.append(float(jnp.sum(v_opt ** 2)))
                barrier_vals.append(min(float(v) for v in constraint_vals.values()))

                x = next_x

            all_violation_rates.append(violations / n_steps)
            all_rewards.append(ep_reward)
            all_min_barrier.append(min(barrier_vals) if barrier_vals else 0.0)
            all_control_costs.append(sum(ctrl_costs))
            all_online_times.extend(online_times)

            y0 = dynamics.output(x0, u0)
            all_tracking['pressure'].append(_rmse(pressures, float(y0[0])))
            all_tracking['enthalpy'].append(_rmse(enthalpies, float(y0[1])))
            all_tracking['power'].append(_rmse(powers, float(y0[2])))

        else:
            has_qp = safety_layer is not None and method_name not in ('ppo', 'ppo_lagr')

            if has_qp:
                rollout, ep_reward, violations, qp_times = _rollout_with_qp(
                    model, dynamics, safety_layer, qp_solver, constraint,
                    x0, u0, ep_key, n_steps, agc_schedule=agc_schedule,
                    use_scipy=True)
                all_online_times.extend(qp_times)
            else:
                rollout, ep_reward, violations, _ = _rollout_no_qp(
                    model, dynamics, constraint, x0, u0, ep_key, n_steps,
                    agc_schedule=agc_schedule)

            n_actual = rollout['obs'].shape[0]
            all_violation_rates.append(violations / max(n_actual, 1))
            all_rewards.append(ep_reward)

            y0 = dynamics.output(x0, u0)
            pressures, enthalpies, powers = [], [], []
            for t in range(n_actual):
                obs = rollout['obs'][t]
                v = rollout['actions'][t]
                u_total = dynamics.compute_total_control(obs, v)
                y = dynamics.output(obs, u_total)
                pressures.append(float(y[0]))
                enthalpies.append(float(y[1]))
                powers.append(float(y[2]))

            all_tracking['pressure'].append(_rmse(pressures, float(y0[0])))
            all_tracking['enthalpy'].append(_rmse(enthalpies, float(y0[1])))
            all_tracking['power'].append(_rmse(powers, float(y0[2])))

            ctrl_cost = float(jnp.sum(rollout['actions'] ** 2))
            all_control_costs.append(ctrl_cost)

            if 'constraint_vals' in rollout and rollout['constraint_vals']:
                min_barriers = []
                for cv in rollout['constraint_vals']:
                    min_barriers.append(min(float(v) for v in cv.values()))
                all_min_barrier.append(min(min_barriers) if min_barriers else 0.0)
            else:
                all_min_barrier.append(0.0)

    return {
        'violation_rate': _mean_std(all_violation_rates),
        'cumulative_reward': _mean_std(all_rewards),
        'tracking_rmse': {
            'pressure': _mean_std(all_tracking['pressure']),
            'enthalpy': _mean_std(all_tracking['enthalpy']),
            'power': _mean_std(all_tracking['power']),
        },
        'control_cost': _mean_std(all_control_costs),
        'min_barrier_value': _mean_std(all_min_barrier),
        'online_time_ms': _mean_std(all_online_times) if all_online_times else (0.0, 0.0),
    }


def result_exists(method_name, condition, seed, results_dir='results/phase4/'):
    path = Path(results_dir) / f'{method_name}_{condition}_seed{seed}.json'
    return path.exists()


def save_result(result, method_name, condition, seed, results_dir='results/phase4/'):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    path = Path(results_dir) / f'{method_name}_{condition}_seed{seed}.json'

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

    with open(path, 'w') as f:
        json.dump(_convert(result), f, indent=2)


def run_optimized(config_path=None, methods=None, conditions=None,
                   seeds=None, results_dir='results/phase4/'):
    """Run all experiments with policy sharing optimization.

    Training groups:
    1. PPO (shared by PPO, PPO-CBF, PPO-HOCBF, PPO-GP-HOCBF, PPO-RHOCBF)
    2. PPO-Lagrangian (separate training due to cost terms)
    3. RoCBF-Net (separate training due to online GP updates)
    4. NMPC (no training)
    """
    config = load_config(config_path)

    if methods is None:
        methods = config.get('methods', list(METHODS.keys()))
    if conditions is None:
        conditions = config.get('conditions', CONDITIONS)
    if seeds is None:
        seeds = list(range(config.get('seeds', 5)))

    train_cfg = config.get('training', {})
    eval_cfg = config.get('evaluation', {})
    hocbf_cfg = config.get('hocbf', {})
    gp_cfg = config.get('gp', {})

    n_episodes = train_cfg.get('max_episodes', 500)
    n_steps = train_cfg.get('n_steps', 200)

    # Group methods by training requirement
    needs_ppo_training = [m for m in methods if m in PPO_REUSE_METHODS]
    needs_lagr_training = 'ppo_lagr' if 'ppo_lagr' in methods else None
    needs_rocbf_training = 'rocbf_net' if 'rocbf_net' in methods else None
    needs_nmpc = 'nmpc' if 'nmpc' in methods else None

    total = len(methods) * len(conditions) * len(seeds)
    count = 0
    start_time = time.time()

    for condition in conditions:
        scenario = CONDITION_SCENARIO_MAP.get(condition)

        for seed in seeds:
            # Skip if all methods for this (condition, seed) already have results
            all_done = all(
                result_exists(m, condition, seed, results_dir)
                for m in methods
            )
            if all_done:
                count += len(methods)
                print(f"  [SKIP] All methods done for {condition} seed={seed}", flush=True)
                continue

            key = jax.random.key(seed)
            load_ratio = 1.0
            delay_order = 0

            # Setup dynamics and constraint for this condition
            if scenario is not None:
                dynamics, constraint = _make_ccs_env(load_ratio, delay_order, scenario)
            else:
                dynamics, constraint = _make_ccs_env(load_ratio, delay_order)
            x0, u0 = dynamics.equilibrium(load_ratio)

            # AGC schedule for load-following condition
            agc_schedule = None
            if condition == 'load_following':
                agc_cfg = config.get('agc_schedule', {})
                agc_schedule = AGCSchedule(
                    base_load=agc_cfg.get('base_load', 1000.0),
                    ramp_rate=agc_cfg.get('ramp_rate', 5.0),
                    regulation_amp=agc_cfg.get('regulation_amp', 20.0),
                    regulation_period=agc_cfg.get('regulation_period', 300.0),
                )

            # Pre-train GP if needed
            gp = None
            gp_methods = {'ppo_gp_hocbf', 'ppo_rhocbf', 'rocbf_net'} & set(methods)
            if gp_methods:
                gp_n_pretrain = gp_cfg.get('n_pretrain', 2000)
                t0 = time.time()
                gp = _pretrain_gp(load_ratio, delay_order, n_pretrain=gp_n_pretrain, key=key)
                print(f"  [GP pretrain] {time.time()-t0:.1f}s", flush=True)

            # === Group 1: Train PPO (shared across PPO-based methods) ===
            ppo_needed = any(
                not result_exists(m, condition, seed, results_dir)
                for m in needs_ppo_training
            )
            ppo_model = None
            ppo_trainer = None
            ppo_reward_history = None

            if needs_ppo_training and ppo_needed:
                t0 = time.time()
                ppo_model, ppo_trainer, ppo_reward_history = _train_ppo(
                    config, dynamics, constraint, x0, u0, key, n_episodes, n_steps)
                print(f"  [PPO train] {time.time()-t0:.1f}s", flush=True)

                # Evaluate all PPO-based methods
                for method_name in needs_ppo_training:
                    if result_exists(method_name, condition, seed, results_dir):
                        count += 1
                        print(f"  [SKIP] {method_name} | {condition} | s={seed}", flush=True)
                        continue
                    count += 1
                    label = METHOD_LABELS.get(method_name, method_name)

                    method_cfg = config.get('methods_config', {}).get(method_name, {})
                    actual_n_steps = eval_cfg.get('n_steps', 200)

                    # Build safety layer for this method
                    safety_layer = _build_safety_layer(
                        method_name, dynamics, constraint, gp, u0,
                        method_cfg, hocbf_cfg)
                    qp_solver = DifferentiableQP(v_max=5.0)

                    key, eval_key = jax.random.split(key)
                    result = evaluate_method(
                        ppo_model, safety_layer, qp_solver, dynamics, constraint,
                        x0, u0, eval_key, method_name, agc_schedule=agc_schedule,
                        n_episodes=eval_cfg.get('n_episodes', 3),
                        n_steps=actual_n_steps,
                        hocbf_cfg=hocbf_cfg, method_cfg=method_cfg, gp=gp)

                    result['convergence_episode'] = n_episodes
                    result['n_training_episodes'] = n_episodes
                    result['reward_history'] = ppo_reward_history[-100:] if len(ppo_reward_history) > 100 else ppo_reward_history

                    save_result(result, method_name, condition, seed, results_dir)
                    vr = result['violation_rate']
                    print(f"  [{count}/{total}] {label:20s} | {condition:15s} | s={seed}: "
                          f"viol={vr[0]:.4f}±{vr[1]:.4f}, reward={result['cumulative_reward'][0]:.1f}",
                          flush=True)

            # === Group 2: Train PPO-Lagrangian ===
            if needs_lagr_training and not result_exists('ppo_lagr', condition, seed, results_dir):
                t0 = time.time()
                lagr_model, lagr_trainer, lagr_reward_history = _train_ppo_lagr(
                    config, dynamics, constraint, x0, u0, key, n_episodes, n_steps)
                print(f"  [PPO-Lagr train] {time.time()-t0:.1f}s", flush=True)

                count += 1
                label = METHOD_LABELS['ppo_lagr']
                actual_n_steps = eval_cfg.get('n_steps', 200)
                qp_solver = DifferentiableQP(v_max=5.0)

                key, eval_key = jax.random.split(key)
                result = evaluate_method(
                    lagr_model, None, qp_solver, dynamics, constraint,
                    x0, u0, eval_key, 'ppo_lagr', agc_schedule=agc_schedule,
                    n_episodes=eval_cfg.get('n_episodes', 3),
                    n_steps=actual_n_steps,
                    hocbf_cfg=hocbf_cfg, gp=gp)

                result['convergence_episode'] = n_episodes
                result['n_training_episodes'] = n_episodes
                result['reward_history'] = lagr_reward_history[-100:] if len(lagr_reward_history) > 100 else lagr_reward_history

                save_result(result, 'ppo_lagr', condition, seed, results_dir)
                vr = result['violation_rate']
                print(f"  [{count}/{total}] {label:20s} | {condition:15s} | s={seed}: "
                      f"viol={vr[0]:.4f}±{vr[1]:.4f}", flush=True)

            # === Group 3: Train RoCBF-Net ===
            if needs_rocbf_training and not result_exists('rocbf_net', condition, seed, results_dir):
                t0 = time.time()
                rocbf_model, rocbf_trainer, rocbf_reward_history = _train_rocbf_net(
                    config, dynamics, constraint, x0, u0, key, gp, n_episodes, n_steps)
                print(f"  [RoCBF-Net train] {time.time()-t0:.1f}s", flush=True)

                count += 1
                label = METHOD_LABELS['rocbf_net']
                method_cfg = config.get('methods_config', {}).get('rocbf_net', {})
                actual_n_steps = eval_cfg.get('n_steps', 200)

                safety_layer = _build_safety_layer(
                    'rocbf_net', dynamics, constraint, gp, u0, method_cfg, hocbf_cfg)
                qp_solver = DifferentiableQP(v_max=5.0)

                key, eval_key = jax.random.split(key)
                result = evaluate_method(
                    rocbf_model, safety_layer, qp_solver, dynamics, constraint,
                    x0, u0, eval_key, 'rocbf_net', agc_schedule=agc_schedule,
                    n_episodes=eval_cfg.get('n_episodes', 3),
                    n_steps=actual_n_steps,
                    hocbf_cfg=hocbf_cfg, method_cfg=method_cfg, gp=gp)

                result['convergence_episode'] = n_episodes
                result['n_training_episodes'] = n_episodes
                result['reward_history'] = rocbf_reward_history[-100:] if len(rocbf_reward_history) > 100 else rocbf_reward_history

                save_result(result, 'rocbf_net', condition, seed, results_dir)
                vr = result['violation_rate']
                print(f"  [{count}/{total}] {label:20s} | {condition:15s} | s={seed}: "
                      f"viol={vr[0]:.4f}±{vr[1]:.4f}", flush=True)

            # === Group 4: NMPC (no training) ===
            if needs_nmpc and not result_exists('nmpc', condition, seed, results_dir):
                count += 1
                label = METHOD_LABELS['nmpc']
                nmpc_cfg = config.get('methods_config', {}).get('nmpc', {})
                nmpc = NMPCController(
                    dynamics=dynamics, constraint=constraint,
                    horizon=nmpc_cfg.get('horizon', 10), Q=None, R=None)
                actual_n_steps = eval_cfg.get('n_steps', 200)
                qp_solver = DifferentiableQP(v_max=5.0)

                key, eval_key = jax.random.split(key)
                result = evaluate_method(
                    None, nmpc, qp_solver, dynamics, constraint,
                    x0, u0, eval_key, 'nmpc', agc_schedule=agc_schedule,
                    n_episodes=eval_cfg.get('n_episodes', 3),
                    n_steps=actual_n_steps,
                    hocbf_cfg=hocbf_cfg, gp=None)

                result['convergence_episode'] = 0
                result['n_training_episodes'] = 0
                result['reward_history'] = []

                save_result(result, 'nmpc', condition, seed, results_dir)
                vr = result['violation_rate']
                print(f"  [{count}/{total}] {label:20s} | {condition:15s} | s={seed}: "
                      f"viol={vr[0]:.4f}±{vr[1]:.4f}", flush=True)

            # Progress report
            elapsed = time.time() - start_time
            if count > 0:
                eta = elapsed / count * (total - count)
                print(f"  Progress: {count}/{total}, elapsed: {elapsed/60:.1f}min, "
                      f"ETA: {eta/60:.1f}min", flush=True)

    print(f"\n=== Done: {count}/{total} experiments in {(time.time()-start_time)/60:.1f}min ===",
          flush=True)


if __name__ == "__main__":
    run_optimized()

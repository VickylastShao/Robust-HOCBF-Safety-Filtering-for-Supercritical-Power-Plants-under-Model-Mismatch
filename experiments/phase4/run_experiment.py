"""Phase 4: 8 methods × 6 conditions × 10 seeds.

Saves results to results/phase4/ as JSON per (method, condition, seed).

Conditions:
- nominal: standard CCS dynamics
- s1_heat: heat absorption perturbation
- s2_pressure: pressure oscillation perturbation
- s3_coupled: coupled perturbation
- s4_nonlinear: nonlinear perturbation
- load_following: AGC load-following schedule

Methods:
- ppo, ppo_lagr, nmpc, ppo_cbf, ppo_hocbf, ppo_gp_hocbf, ppo_rhocbf, rocbf_net
"""
import json
import sys
import time

sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

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
    CBF_PROTECTED,
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


def load_config(config_path=None):
    """Load Phase 4 configuration."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'phase4.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_single(method_name: str, condition: str, seed: int, config: dict):
    """Run one experiment: train + evaluate, return metrics dict.

    Metrics:
    - violation_rate: fraction of steps with any h_i < 0
    - cumulative_reward: total episode reward
    - tracking_rmse: {pressure, enthalpy, power} RMSE
    - control_cost: Σ‖u - u0‖²
    - min_barrier_value: min h_i across trajectory
    - online_time_ms: average per-step computation time
    - gp_inference_time_ms: GP predict time (if applicable)
    - convergence_episode: episode where reward stabilizes
    """
    cfg = config
    train_cfg = cfg.get('training', {})
    eval_cfg = cfg.get('evaluation', {})
    method_cfg = cfg.get('methods_config', {}).get(method_name, {})

    load_ratio = 1.0
    delay_order = 0

    # Determine scenario for this condition
    scenario = CONDITION_SCENARIO_MAP.get(condition)

    # Setup dynamics and constraint
    if scenario is not None:
        dynamics, constraint = _make_ccs_env(load_ratio, delay_order, scenario)
    else:
        dynamics, constraint = _make_ccs_env(load_ratio, delay_order)

    x0, u0 = dynamics.equilibrium(load_ratio)

    # AGC schedule for load-following condition
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

    # Pre-train GP if needed by method
    gp = None
    hocbf_cfg = cfg.get('hocbf', {})
    gp_cfg = cfg.get('gp', {})
    if method_name in ('ppo_gp_hocbf', 'ppo_rhocbf', 'rocbf_net'):
        # Methods with scenario-specific GP get accurate mean + small sigma
        # (controlled by scenario_specific config flag, default True for rocbf_net)
        use_scenario_gp = method_cfg.get('scenario_specific_gp',
                                          method_name == 'rocbf_net')
        if use_scenario_gp:
            gp = _pretrain_gp(
                load_ratio, delay_order,
                n_pretrain=method_cfg.get('n_pretrain', gp_cfg.get('n_pretrain', 3000)),
                key=key,
                sigma_floor=method_cfg.get('sigma_floor', gp_cfg.get('sigma_floor', None)),
                scenario=scenario, scenario_specific=True)
        else:
            gp = _pretrain_gp(
                load_ratio, delay_order,
                n_pretrain=method_cfg.get('n_pretrain', gp_cfg.get('n_pretrain', 3000)),
                key=key,
                sigma_floor=method_cfg.get('sigma_floor', gp_cfg.get('sigma_floor', None)))

    # Initialize method
    train_fn = METHODS[method_name]
    model, trainer, safety_layer = train_fn(
        method_cfg, dynamics, constraint, key, gp=gp)

    qp_solver = DifferentiableQP(v_max=5.0)
    is_nmpc = method_name == 'nmpc'

    # Training loop
    n_episodes = train_cfg.get('max_episodes', 5000)
    n_steps = train_cfg.get('n_steps', 300)
    eval_every = train_cfg.get('eval_every', 100)
    n_eval = train_cfg.get('n_eval', 5)
    convergence_window = train_cfg.get('convergence_window', 50)
    convergence_threshold = train_cfg.get('convergence_threshold', 0.05)
    min_episodes = train_cfg.get('min_episodes', 500)
    gp_update_interval = method_cfg.get('gp_update_interval', 50)

    reward_history = []
    epsilon_log = []
    convergence_episode = n_episodes
    best_model_state = None

    if is_nmpc:
        # NMPC: no training loop, just evaluate directly
        convergence_episode = 0
    else:
        for ep in range(n_episodes):
            # Sample training scenario:
            # - RoCBF-Net / scenario-specific GP methods: deployment scenario
            # - Other methods: random scenario (mixed training)
            use_scenario_gp = method_cfg.get('scenario_specific_gp',
                                              method_name == 'rocbf_net')
            if use_scenario_gp:
                train_scenario = scenario  # deployment scenario only
            else:
                key, scenario_key = jax.random.split(key)
                scenario_idx = jax.random.randint(scenario_key, (), 0, len(SCENARIOS))
                train_scenario = SCENARIOS[int(scenario_idx)]

            if train_scenario is not None:
                train_dyn = UncertainUSCCSDynamics(
                    delay_order=delay_order, load_ratio=load_ratio,
                    uncertainty_scenario=train_scenario)
            else:
                train_dyn = dynamics

            key, rollout_key = jax.random.split(key)

            # Training rollout: always without QP filter (fast).
            # The safety filter is only applied at evaluation time.
            # PPO doesn't use QP gradients, so training without the filter
            # is equivalent but much faster.
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

            # Add costs for PPO-Lagrangian
            if method_name == 'ppo_lagr' and 'constraint_vals' in rollout:
                batch['costs'] = compute_step_costs(rollout['constraint_vals'])

            for _ in range(trainer.epochs):
                loss = trainer.train_step(batch)

            # Slow timescale: GP update + epsilon logging
            if (ep + 1) % gp_update_interval == 0 and gp is not None:
                # RoCBF-Net: online GP update with deployment scenario data
                if method_name == 'rocbf_net':
                    key, gp_key = jax.random.split(key)
                    # Collect new data from deployment scenario only
                    env_gp = UncertainUSCCSDynamics(
                        delay_order=delay_order, load_ratio=load_ratio,
                        uncertainty_scenario=scenario)
                    key, data_key = jax.random.split(gp_key)
                    from experiments.phase4.methods import _collect_gp_data
                    X_new, Y_new = _collect_gp_data(env_gp, 200, key=data_key)
                    gp.incremental_update(X_new, Y_new, reoptimize_hyperparams=False)

                    # Rebuild safety layer after GP update
                    safety_layer = _rebuild_safety(method_name, dynamics, constraint,
                                                    gp, u0, method_cfg, hocbf_cfg)

                # Log epsilon for any GP-based method
                if safety_layer is not None and hasattr(safety_layer, 'robust_hocbf_list'):
                    eps_vals = []
                    for hocbf in safety_layer.robust_hocbf_list:
                        try:
                            eps_vals.append(float(hocbf.compute_epsilon(x0[:3])))
                        except Exception:
                            pass
                    if eps_vals:
                        log_entry = {
                            'episode': ep + 1,
                            'n_gp_points': gp.n_training_points,
                            'epsilon_total': sum(eps_vals),
                            'epsilon_per_constraint': eps_vals,
                        }
                        # Add GP diagnostics
                        if gp is not None:
                            try:
                                mu, sigma = gp.predict(x0[:3].reshape(1, -1))
                                log_entry['sigma_gp_mean'] = float(jnp.mean(sigma))
                                log_entry['sigma_gp_max'] = float(jnp.max(sigma))
                                log_entry['mu_gp_mean'] = float(jnp.mean(mu))
                                log_entry['beta'] = float(gp.compute_beta())
                            except Exception:
                                pass
                        epsilon_log.append(log_entry)

            # Convergence check
            if (ep + 1 >= min_episodes and
                    len(reward_history) >= convergence_window):
                recent = reward_history[-convergence_window:]
                if len(recent) > convergence_window:
                    prev_avg = sum(recent[:-1]) / (len(recent) - 1)
                    curr_avg = sum(recent) / len(recent)
                    if abs(curr_avg - prev_avg) / (abs(prev_avg) + 1e-8) < convergence_threshold:
                        convergence_episode = ep + 1
                        break

            # Periodic logging
            if (ep + 1) % eval_every == 0:
                print(f"  [{method_name}|{condition}|seed={seed}] "
                      f"Ep {ep+1}: r={ep_reward:.1f}", flush=True)

    # Evaluation
    actual_n_steps = eval_cfg.get('n_steps', 100)
    if is_nmpc:
        actual_n_steps = eval_cfg.get('n_steps_nmpc', 50)
    elif condition == 'load_following':
        actual_n_steps = eval_cfg.get('load_following_steps', 200)

    eval_results = evaluate(model, trainer, safety_layer, qp_solver,
                            dynamics, constraint, x0, u0, key,
                            n_episodes=eval_cfg.get('n_episodes', 3),
                            n_steps=actual_n_steps,
                            agc_schedule=agc_schedule,
                            method_name=method_name,
                            condition=condition,
                            method_cfg=method_cfg,
                            hocbf_cfg=hocbf_cfg,
                            gp=gp,
                            is_nmpc=is_nmpc)

    eval_results['convergence_episode'] = convergence_episode
    eval_results['n_training_episodes'] = convergence_episode if is_nmpc else len(reward_history)
    eval_results['reward_history'] = reward_history[-100:] if len(reward_history) > 100 else reward_history
    eval_results['epsilon_log'] = epsilon_log

    return eval_results


def _rebuild_safety(method_name, dynamics, constraint, gp, u0, method_cfg, hocbf_cfg):
    """Rebuild safety layer for training dynamics."""
    k_p = hocbf_cfg.get('pressure_k_gains', [0.5, 0.5])
    k_h = hocbf_cfg.get('enthalpy_k_gains', [1.0])
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
        kappa = method_cfg.get('epsilon_kappa', 1.0)
        use_mc = method_cfg.get('use_mean_correction', True)
        epsilon_floor = method_cfg.get('epsilon_floor', 0.0)
        return _make_robust_hocbf(dynamics, constraint, gp, u0,
                                   epsilon_kappa=kappa, k_pressure=k_p,
                                   k_enthalpy=k_h, u_max=u_max,
                                   use_mean_correction=use_mc,
                                   epsilon_floor=epsilon_floor)
    elif method_name == 'rocbf_net':
        kappa = method_cfg.get('epsilon_kappa', 1.0)
        epsilon_floor = method_cfg.get('epsilon_floor', 0.0)
        use_mc = method_cfg.get('use_mean_correction', True)
        return _make_robust_hocbf(dynamics, constraint, gp, u0,
                                   epsilon_kappa=kappa, k_pressure=k_p,
                                   k_enthalpy=k_h, u_max=u_max,
                                   use_mean_correction=use_mc,
                                   epsilon_floor=epsilon_floor)
    return None


def evaluate(model, trainer, safety_layer, qp_solver,
             dynamics, constraint, x0, u0, key,
             n_episodes=10, n_steps=300,
             agc_schedule=None, method_name='ppo',
             condition='nominal', method_cfg=None,
             hocbf_cfg=None, gp=None, is_nmpc=False):
    """Evaluate a trained policy across episodes, compute metrics."""
    # JIT-compile QP matrices function for fast evaluation.
    # Without JIT, each qp_matrices call takes ~466ms (eager mode);
    # with JIT, each call takes ~1ms after one-time compilation.
    jit_qp_fn = None
    if safety_layer is not None and method_name not in ('ppo', 'ppo_lagr', 'nmpc'):
        try:
            jit_qp_fn = jax.jit(safety_layer.qp_matrices)
            _ = jit_qp_fn(x0[:3])  # warm up JIT compilation
        except Exception:
            jit_qp_fn = None  # fall back to non-JIT if compilation fails

    all_violation_rates = []
    all_cbf_violation_rates = []
    all_rewards = []
    all_tracking = {'pressure': [], 'enthalpy': [], 'power': []}
    all_control_costs = []
    all_min_barrier = []
    all_online_times = []
    # Per-constraint-type violation counts (for MF-3/MF-4 decomposition)
    per_type_violations = {
        'pressure': {'count': 0, 'steps': 0},
        'enthalpy': {'count': 0, 'steps': 0},
        'power': {'count': 0, 'steps': 0},
    }

    if hocbf_cfg is None:
        hocbf_cfg = {}
    if method_cfg is None:
        method_cfg = {}

    for ep in range(n_episodes):
        key, ep_key = jax.random.split(key)

        if is_nmpc:
            # NMPC evaluation
            nmpc = safety_layer  # NMPCController stored as safety_layer
            x = x0
            violations = 0
            cbf_violations = 0
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

                # Step with stabilized dynamics
                next_x = dynamics.step_stabilized(x[:3], v_opt)
                u_total = dynamics.compute_total_control(x[:3], v_opt)

                constraint_vals = constraint.check_all(next_x, u_total)
                if any(v < 0 for v in constraint_vals.values()):
                    violations += 1
                if any(v < 0 for k, v in constraint_vals.items()
                       if k in CBF_PROTECTED):
                    cbf_violations += 1
                # Per-constraint-type violation tracking
                for ctype in ('pressure', 'enthalpy', 'power'):
                    if any(v < 0 for k, v in constraint_vals.items() if ctype in k):
                        per_type_violations[ctype]['count'] += 1
                    per_type_violations[ctype]['steps'] += 1

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
            all_cbf_violation_rates.append(cbf_violations / n_steps)
            all_rewards.append(ep_reward)
            all_min_barrier.append(min(barrier_vals))
            all_control_costs.append(sum(ctrl_costs))
            all_online_times.extend(online_times)

            y0 = dynamics.output(x0, u0)
            all_tracking['pressure'].append(_rmse(pressures, float(y0[0])))
            all_tracking['enthalpy'].append(_rmse(enthalpies, float(y0[1])))
            all_tracking['power'].append(_rmse(powers, float(y0[2])))

        else:
            # RL-based methods
            has_qp = safety_layer is not None and method_name not in ('ppo', 'ppo_lagr')

            if has_qp:
                rollout, ep_reward, violations, cbf_violations, qp_times = _rollout_with_qp(
                    model, dynamics, safety_layer, qp_solver, constraint,
                    x0, u0, ep_key, n_steps, agc_schedule=agc_schedule,
                    use_scipy=True, jit_qp_fn=jit_qp_fn)
                all_online_times.extend(qp_times)
            else:
                rollout, ep_reward, violations, cbf_violations, _ = _rollout_no_qp(
                    model, dynamics, constraint, x0, u0, ep_key, n_steps,
                    agc_schedule=agc_schedule)

            n_actual = rollout['obs'].shape[0]
            all_violation_rates.append(violations / max(n_actual, 1))
            all_cbf_violation_rates.append(cbf_violations / max(n_actual, 1))
            all_rewards.append(ep_reward)

            # Tracking RMSE
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

            # Control cost
            ctrl_cost = float(jnp.sum(rollout['actions'] ** 2))
            all_control_costs.append(ctrl_cost)

            # Min barrier value
            if 'constraint_vals' in rollout and rollout['constraint_vals']:
                min_barriers = []
                for cv in rollout['constraint_vals']:
                    min_barriers.append(min(float(v) for v in cv.values()))
                    # Per-constraint-type violation tracking
                    for ctype in ('pressure', 'enthalpy', 'power'):
                        if any(v < 0 for k, v in cv.items() if ctype in k):
                            per_type_violations[ctype]['count'] += 1
                        per_type_violations[ctype]['steps'] += 1
                all_min_barrier.append(min(min_barriers) if min_barriers else 0.0)
            else:
                all_min_barrier.append(0.0)

    # Compute per-type violation rates
    per_type_rates = {}
    for ctype, info in per_type_violations.items():
        if info['steps'] > 0:
            per_type_rates[ctype] = {
                'violation_rate': info['count'] / info['steps'],
                'violation_count': info['count'],
                'total_steps': info['steps'],
            }
        else:
            per_type_rates[ctype] = {
                'violation_rate': 0.0,
                'violation_count': 0,
                'total_steps': 0,
            }

    return {
        'violation_rate': _mean_std(all_violation_rates),
        'cbf_violation_rate': _mean_std(all_cbf_violation_rates),
        'cumulative_reward': _mean_std(all_rewards),
        'tracking_rmse': {
            'pressure': _mean_std(all_tracking['pressure']),
            'enthalpy': _mean_std(all_tracking['enthalpy']),
            'power': _mean_std(all_tracking['power']),
        },
        'control_cost': _mean_std(all_control_costs),
        'min_barrier_value': _mean_std(all_min_barrier),
        'online_time_ms': _mean_std(all_online_times) if all_online_times else (0.0, 0.0),
        'per_constraint_type': per_type_rates,
    }


def _rmse(values, target):
    """Compute RMSE of values vs target."""
    import numpy as np
    arr = np.array(values)
    return float(np.sqrt(np.mean((arr - target) ** 2)))


def _mean_std(values):
    """Return (mean, std) of a list."""
    import numpy as np
    arr = np.array(values, dtype=float)
    return (float(np.mean(arr)), float(np.std(arr)))


def save_result(result, method_name, condition, seed, results_dir='results/phase4/'):
    """Save result to JSON file."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    path = Path(results_dir) / f'{method_name}_{condition}_seed{seed}.json'

    # Convert non-serializable types
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


def run_all(config_path=None, methods=None, conditions=None,
            seeds=None, results_dir='results/phase4/'):
    """Run all experiments: 8 methods × 6 conditions × 10 seeds."""
    config = load_config(config_path)

    if methods is None:
        methods = config.get('methods', list(METHODS.keys()))
    if conditions is None:
        conditions = config.get('conditions', CONDITIONS)
    if seeds is None:
        seeds = list(range(config.get('seeds', 10)))

    total = len(methods) * len(conditions) * len(seeds)
    count = 0

    for method_name in methods:
        for condition in conditions:
            for seed in seeds:
                count += 1
                label = METHOD_LABELS.get(method_name, method_name)

                # Skip if result already exists
                result_path = Path(results_dir) / f'{method_name}_{condition}_seed{seed}.json'
                if result_path.exists():
                    print(f"\n[{count}/{total}] SKIP {label} | {condition} | seed={seed} (exists)",
                          flush=True)
                    continue

                print(f"\n[{count}/{total}] {label} | {condition} | seed={seed}",
                      flush=True)

                try:
                    result = run_single(method_name, condition, seed, config)
                    save_result(result, method_name, condition, seed, results_dir)
                    vr = result['violation_rate']
                    print(f"  → violation_rate={vr[0]:.4f}±{vr[1]:.4f}, "
                          f"reward={result['cumulative_reward'][0]:.1f}", flush=True)
                except Exception as e:
                    print(f"  ✗ FAILED: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

    print(f"\n=== Done: {count}/{total} experiments completed ===", flush=True)


if __name__ == "__main__":
    run_all()

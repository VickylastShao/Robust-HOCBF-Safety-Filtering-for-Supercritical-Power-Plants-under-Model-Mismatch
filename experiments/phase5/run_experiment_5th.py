"""Phase 5: Full 5th-order CCS fair comparison — 8 methods × 8 conditions × 5 seeds.

Uses 5th-order dynamics (USCCSDynamics5th, nx=5) with Φ-scaled nonlinear
rollout and 6 CBF constraints (including power, now CBF-enforceable at m=1).

Conditions (8): Nominal, S1-S6, Load Following
Methods (8): PPO, PPO-Lagrangian, NMPC, PPO-CBF, PPO-HOCBF,
              PPO-GP-HOCBF, PPO-RHOCBF, RoCBF-Net

Results saved to results/phase5/ as JSON per (method, condition, seed).

Usage:
    cd /home/gpu/sz_workspace/RoCBF-Net
    . .venv/bin/activate
    python experiments/phase5/run_experiment_5th.py
    python experiments/phase5/run_experiment_5th.py --methods ppo_rhocbf --conditions nominal --seeds 42
"""
import sys
import time
import json
import argparse
import warnings
import os

warnings.filterwarnings('ignore')

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import yaml
from pathlib import Path

from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.gp.gp_residual import GPResidual
from rocbf.baselines.ppo_lagrangian import PPOTrainerLagrangian, compute_step_costs
from rocbf.baselines.nmpc_5th import NMPCController5th
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from envs.ccs.agc_schedule import AGCSchedule
from experiments.phase5.methods_5th import (
    NX, N_GP_DIMS,
    METHODS_5TH, METHOD_LABELS,
    SCENARIOS, SCENARIO_LABELS,
    _make_ccs_env_5th, _make_hocbf_5th, _make_robust_hocbf_5th,
    _pretrain_gp_5th,
    _rollout_with_qp_5th, _rollout_no_qp_5th, _rollout_lqr_5th,
    _count_violations_5th, CBF_PROTECTED_5TH,
    make_lqr_rhocbf_5th,
)

# ---------- Condition definitions ----------

CONDITIONS = ['nominal', 's1_heat', 's2_pressure', 's3_coupled',
              's4_nonlinear', 's5_valve', 's6_fuel', 'load_following']

CONDITION_SCENARIO_MAP = {
    'nominal': None,
    's1_heat': 'heat_absorption',
    's2_pressure': 'pressure_oscillation',
    's3_coupled': 'coupled',
    's3_weak': 'coupled_weak',
    's3_midstrong': 'coupled_midstrong',
    's3_strong': 'coupled_strong',
    's4_nonlinear': 'nonlinear',
    's5_valve': 'valve_degradation',
    's6_fuel': 'fuel_quality',
    'load_following': None,
}

CONDITION_LABELS = {
    'nominal': 'Nominal', 's1_heat': 'S1:Heat', 's2_pressure': 'S2:Pressure',
    's3_coupled': 'S3:Med', 's3_weak': 'S3:Weak', 's3_midstrong': 'S3:MStr', 's3_strong': 'S3:Strong',
    's4_nonlinear': 'S4:Nonlinear',
    's5_valve': 'S5:Valve', 's6_fuel': 'S6:Fuel', 'load_following': 'LoadFol',
}


# ---------- Config ----------

def load_config(config_path=None):
    """Load Phase 5 configuration."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'phase5.yaml'
    if not os.path.exists(config_path):
        # Fall back to defaults if no config file
        return _default_config()
    with open(config_path) as f:
        return yaml.safe_load(f)


def _default_config():
    return {
        'phase': 5,
        'seeds': 5,
        'conditions': CONDITIONS,
        'methods': list(METHODS_5TH.keys()),
        'training': {
            'n_steps': 200, 'max_episodes': 200, 'min_episodes': 50,
            'eval_every': 50, 'n_eval': 3, 'convergence_window': 30,
            'convergence_threshold': 0.05,
        },
        'evaluation': {
            'n_episodes': 50, 'n_steps': 500, 'load_following_steps': 500,
        },
        'methods_config': {
            'ppo': {'hidden_dim': 128, 'lr': 1e-4},
            'ppo_lagr': {'hidden_dim': 128, 'lr': 1e-4, 'cost_limit': 0.0, 'lagrangian_lr': 0.01},
            'nmpc': {'horizon': 10},
            'ppo_cbf': {'hidden_dim': 128, 'lr': 1e-4},
            'ppo_hocbf': {'hidden_dim': 128, 'lr': 1e-4, 'pressure_k_gains': [0.5, 0.5], 'enthalpy_k_gains': [1.0], 'power_k_gains': [1.0]},
            'ppo_gp_hocbf': {'hidden_dim': 128, 'lr': 1e-4, 'n_pretrain': 500, 'sigma_floor': 1e-4, 'scenario_specific_gp': True},
            'ppo_rhocbf': {'hidden_dim': 128, 'lr': 1e-4, 'epsilon_kappa': 1.0, 'n_pretrain': 500, 'sigma_floor': 1e-4, 'scenario_specific_gp': True, 'use_mean_correction': True, 'pressure_k_gains': [0.5, 0.5], 'enthalpy_k_gains': [1.0], 'power_k_gains': [1.0], 'u_max': 100.0},
            'rocbf_net': {'hidden_dim': 128, 'lr': 1e-4, 'epsilon_kappa': 1.0, 'epsilon_floor': 0.0, 'use_mean_correction': True, 'scenario_specific_gp': True, 'gp_update_interval': 50, 'n_pretrain': 500, 'sigma_floor': 1e-4, 'pressure_k_gains': [0.5, 0.5], 'enthalpy_k_gains': [1.0], 'power_k_gains': [1.0], 'u_max': 100.0},
        },
        'hocbf': {'pressure_k_gains': [0.5, 0.5], 'enthalpy_k_gains': [1.0], 'power_k_gains': [1.0], 'u_max': 100.0},
        'gp': {'noise_variance': 1e-4, 'sigma_floor': 1e-4, 'n_pretrain': 500},
        'agc_schedule': {'base_load': 1000.0, 'ramp_rate': 5.0, 'regulation_amp': 20.0, 'regulation_period': 300.0},
    }


# ---------- Run single experiment ----------

def run_single(method_name, condition, seed, config):
    """Run one experiment: train + evaluate, return metrics dict."""
    cfg = config
    train_cfg = cfg.get('training', {})
    eval_cfg = cfg.get('evaluation', {})
    method_cfg = cfg.get('methods_config', {}).get(method_name, {})
    hocbf_cfg = cfg.get('hocbf', {})

    load_ratio = 1.0
    scenario = CONDITION_SCENARIO_MAP.get(condition)

    # Setup 5th-order dynamics and constraint
    dynamics, constraint = _make_ccs_env_5th(load_ratio, scenario)
    x0, u0 = dynamics.equilibrium(load_ratio)

    # AGC schedule for load-following
    agc_schedule = None
    if condition == 'load_following':
        agc_cfg = cfg.get('agc_schedule', {})
        agc_schedule = AGCSchedule(
            base_load=agc_cfg.get('base_load', 1000.0),
            ramp_rate=agc_cfg.get('ramp_rate', 5.0),
            regulation_amp=agc_cfg.get('regulation_amp', 20.0),
            regulation_period=agc_cfg.get('regulation_period', 300.0))

    key = jax.random.key(seed)

    # Pre-train GP if needed
    gp = None
    if method_name in ('ppo_gp_hocbf', 'ppo_rhocbf', 'rocbf_net'):
        use_scenario_gp = method_cfg.get('scenario_specific_gp',
                                          method_name in ('ppo_rhocbf', 'rocbf_net'))
        gp = _pretrain_gp_5th(
            load_ratio,
            n_pretrain=method_cfg.get('n_pretrain', 500),
            key=key,
            sigma_floor=method_cfg.get('sigma_floor', 1e-4),
            scenario=scenario,
            scenario_specific=use_scenario_gp)

    # Initialize method
    train_fn = METHODS_5TH[method_name]
    model, trainer, safety_layer = train_fn(
        method_cfg, dynamics, constraint, key, gp=gp)

    qp_solver = DifferentiableQP(v_max=5.0)
    is_nmpc = method_name == 'nmpc'
    is_lqr = method_name == 'lqr'

    # --- Training loop ---
    n_episodes = train_cfg.get('max_episodes', 200)
    n_steps = train_cfg.get('n_steps', 200)
    eval_every = train_cfg.get('eval_every', 50)
    convergence_window = train_cfg.get('convergence_window', 30)
    convergence_threshold = train_cfg.get('convergence_threshold', 0.05)
    min_episodes = train_cfg.get('min_episodes', 50)
    gp_update_interval = method_cfg.get('gp_update_interval', 50)

    reward_history = []
    epsilon_log = []
    convergence_episode = n_episodes

    if is_nmpc:
        convergence_episode = 0
    elif not is_lqr:
        for ep in range(n_episodes):
            # Sample training scenario
            use_scenario_gp = method_cfg.get('scenario_specific_gp',
                                              method_name in ('ppo_rhocbf', 'rocbf_net'))
            if use_scenario_gp:
                train_scenario = scenario
            else:
                key, scenario_key = jax.random.split(key)
                scenario_idx = jax.random.randint(scenario_key, (), 0, len(SCENARIOS))
                train_scenario = SCENARIOS[int(scenario_idx)]

            if train_scenario is not None:
                train_dyn = UncertainUSCCSDynamics5th(
                    dt=1.0, load_ratio=load_ratio,
                    uncertainty_scenario=train_scenario)
            else:
                train_dyn = dynamics

            key, rollout_key = jax.random.split(key)
            rollout, ep_reward, violations, _, _ = _rollout_no_qp_5th(
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

            if method_name == 'ppo_lagr' and 'constraint_vals' in rollout:
                batch['costs'] = compute_step_costs(rollout['constraint_vals'])

            for _ in range(trainer.epochs):
                loss = trainer.train_step(batch)

            # Online GP update for RoCBF-Net
            if (ep + 1) % gp_update_interval == 0 and gp is not None:
                if method_name == 'rocbf_net':
                    key, gp_key = jax.random.split(key)
                    from experiments.phase5.common_5th import collect_gp_data_5th as _collect
                    env_gp = UncertainUSCCSDynamics5th(
                        dt=1.0, load_ratio=load_ratio,
                        uncertainty_scenario=scenario)
                    key, data_key = jax.random.split(gp_key)
                    X_new, Y_new = _collect(env_gp, 200, key=data_key,
                                             load_ratio=load_ratio)
                    gp.incremental_update(X_new, Y_new, reoptimize_hyperparams=False)
                    # Rebuild safety layer with updated GP
                    k_p = tuple(method_cfg.get('pressure_k_gains', (0.5, 0.5)))
                    k_h = tuple(method_cfg.get('enthalpy_k_gains', (1.0,)))
                    k_n = tuple(method_cfg.get('power_k_gains', (1.0,)))
                    safety_layer = _make_robust_hocbf_5th(
                        dynamics, constraint, gp, u0,
                        epsilon_kappa=method_cfg.get('epsilon_kappa', 1.0),
                        k_pressure=k_p, k_enthalpy=k_h, k_power=k_n,
                        u_max=method_cfg.get('u_max', 100.0),
                        use_mean_correction=method_cfg.get('use_mean_correction', True),
                        epsilon_floor=method_cfg.get('epsilon_floor', 0.0),
                        use_phi_scaled_g=True)

                # Log epsilon values
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
                        if gp is not None:
                            try:
                                mu, sigma = gp.predict(x0[:3].reshape(1, -1))
                                log_entry['sigma_gp_mean'] = float(jnp.mean(sigma))
                                log_entry['sigma_gp_max'] = float(jnp.max(sigma))
                            except Exception:
                                pass
                        epsilon_log.append(log_entry)

            # Convergence check
            if (ep + 1 >= min_episodes and len(reward_history) >= convergence_window):
                recent = reward_history[-convergence_window:]
                if len(recent) > 1:
                    prev_avg = sum(recent[:-1]) / len(recent[:-1])
                    curr_avg = sum(recent) / len(recent)
                    if prev_avg != 0 and abs(curr_avg - prev_avg) / abs(prev_avg) < convergence_threshold:
                        convergence_episode = ep + 1
                        break

            if (ep + 1) % eval_every == 0:
                print(f"  [{method_name}|{condition}|seed={seed}] "
                      f"Ep {ep+1}: r={ep_reward:.1f}", flush=True)

    # --- Evaluation ---
    actual_n_steps = eval_cfg.get('n_steps', 500)
    if is_nmpc:
        actual_n_steps = eval_cfg.get('n_steps_nmpc', 50)
    elif condition == 'load_following':
        actual_n_steps = eval_cfg.get('load_following_steps', 500)

    eval_results = _evaluate(
        model, trainer, safety_layer, qp_solver,
        dynamics, constraint, x0, u0, key,
        n_episodes=eval_cfg.get('n_episodes', 50),
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


# ---------- Evaluation ----------

def _evaluate(model, trainer, safety_layer, qp_solver,
              dynamics, constraint, x0, u0, key,
              n_episodes=10, n_steps=300,
              agc_schedule=None, method_name='ppo',
              condition='nominal', method_cfg=None,
              hocbf_cfg=None, gp=None, is_nmpc=False):
    """Evaluate a trained policy across episodes, compute metrics."""
    # JIT-compile QP matrices function for fast evaluation
    jit_qp_fn = None
    if safety_layer is not None and method_name not in ('ppo', 'ppo_lagr', 'nmpc', 'lqr'):
        try:
            jit_qp_fn = jax.jit(safety_layer.qp_matrices)
            _ = jit_qp_fn(x0[:NX])  # warm up (full 5D state)
        except Exception:
            jit_qp_fn = None

    all_violation_rates = []
    all_cbf_violation_rates = []
    all_rewards = []
    all_tracking = {'pressure': [], 'enthalpy': [], 'power': []}
    all_control_costs = []
    all_min_barrier = []
    all_online_times = []
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
            nmpc = safety_layer  # NMPCController5th stored as safety_layer
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
                    y_ref = dynamics.output(x_ref)
                    v_opt = nmpc.compute_action(x, y_ref)
                else:
                    v_opt = nmpc.compute_action(x)
                online_times.append((time.perf_counter() - t0) * 1000)

                next_x = dynamics.step_stabilized_phi_scaled(x[:NX], v_opt)
                constraint_vals = constraint.check_all(next_x)

                if any(v < 0 for v in constraint_vals.values()):
                    violations += 1
                if any(v < 0 for k, v in constraint_vals.items()
                       if k in CBF_PROTECTED_5TH):
                    cbf_violations += 1

                for ctype in ('pressure', 'enthalpy', 'power'):
                    if any(v < 0 for k, v in constraint_vals.items() if ctype in k):
                        per_type_violations[ctype]['count'] += 1
                    per_type_violations[ctype]['steps'] += 1

                y = dynamics.output(next_x)
                if agc_schedule is not None:
                    target_load = agc_schedule.get_reference(float(t))
                    x_ref, u_target = dynamics.equilibrium(target_load / 1000.0)
                    y_ref = dynamics.output(x_ref)
                else:
                    y_ref = dynamics.output(x0)
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
            all_min_barrier.append(min(barrier_vals) if barrier_vals else 0.0)
            all_control_costs.append(sum(ctrl_costs))
            all_online_times.extend(online_times)

            y0 = dynamics.output(x0)
            all_tracking['pressure'].append(_rmse(pressures, float(y0[0])))
            all_tracking['enthalpy'].append(_rmse(enthalpies, float(y0[1])))
            all_tracking['power'].append(_rmse(powers, float(y0[2])))

        else:
            has_qp = safety_layer is not None and method_name not in ('ppo', 'ppo_lagr', 'lqr')

            if has_qp:
                rollout, ep_reward, violations, cbf_violations, qp_times = _rollout_with_qp_5th(
                    model, dynamics, safety_layer, qp_solver, constraint,
                    x0, u0, ep_key, n_steps, jit_qp_fn=jit_qp_fn)
                all_online_times.extend(qp_times)
            else:
                rollout, ep_reward, violations, cbf_violations, _ = _rollout_no_qp_5th(
                    model, dynamics, constraint, x0, u0, ep_key, n_steps)

            n_actual = rollout['obs'].shape[0]
            all_violation_rates.append(violations / max(n_actual, 1))
            all_cbf_violation_rates.append(cbf_violations / max(n_actual, 1))
            all_rewards.append(ep_reward)

            y0 = dynamics.output(x0)
            pressures, enthalpies, powers = [], [], []
            for t in range(n_actual):
                obs = rollout['obs'][t]
                v = rollout['actions'][t]
                y = dynamics.output(obs)
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
                    for ctype in ('pressure', 'enthalpy', 'power'):
                        if any(v < 0 for k, v in cv.items() if ctype in k):
                            per_type_violations[ctype]['count'] += 1
                        per_type_violations[ctype]['steps'] += 1
                all_min_barrier.append(min(min_barriers) if min_barriers else 0.0)
            else:
                all_min_barrier.append(0.0)

    per_type_rates = {}
    for ctype, info in per_type_violations.items():
        if info['steps'] > 0:
            per_type_rates[ctype] = {
                'violation_rate': info['count'] / info['steps'],
                'violation_count': info['count'],
                'total_steps': info['steps'],
            }
        else:
            per_type_rates[ctype] = {'violation_rate': 0.0, 'violation_count': 0, 'total_steps': 0}

    result = {
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
    return result


# ---------- LQR-RHOCBF evaluation (no training) ----------

def run_lqr_rhocbf(condition, seed, config):
    """Evaluate LQR-RHOCBF: LQR controller (v=0) + Robust HOCBF filter.

    This is the controller-agnostic baseline — no PPO training needed.
    """
    cfg = config
    eval_cfg = cfg.get('evaluation', {})
    hocbf_cfg = cfg.get('hocbf', {})
    method_cfg = cfg.get('methods_config', {}).get('ppo_rhocbf', {})

    load_ratio = 1.0
    scenario = CONDITION_SCENARIO_MAP.get(condition)

    dynamics, constraint = _make_ccs_env_5th(load_ratio, scenario)
    x0, u0 = dynamics.equilibrium(load_ratio)

    key = jax.random.key(seed)

    # Pretrain GP
    gp = _pretrain_gp_5th(
        load_ratio,
        n_pretrain=method_cfg.get('n_pretrain', 500),
        key=key,
        sigma_floor=method_cfg.get('sigma_floor', 1e-4),
        scenario=scenario,
        scenario_specific=True)

    k_p = tuple(method_cfg.get('pressure_k_gains', (0.5, 0.5)))
    k_h = tuple(method_cfg.get('enthalpy_k_gains', (1.0,)))
    k_n = tuple(method_cfg.get('power_k_gains', (1.0,)))

    safety_layer = make_lqr_rhocbf_5th(
        dynamics, constraint, gp, u0,
        epsilon_kappa=method_cfg.get('epsilon_kappa', 1.0),
        epsilon_floor=method_cfg.get('epsilon_floor', 0.0),
        k_pressure=k_p, k_enthalpy=k_h, k_power=k_n,
        u_max=method_cfg.get('u_max', 100.0),
        use_mean_correction=method_cfg.get('use_mean_correction', True))

    qp_solver = DifferentiableQP(v_max=5.0)

    # JIT QP matrices
    try:
        jit_qp_fn = jax.jit(safety_layer.qp_matrices)
        _ = jit_qp_fn(x0[:3])
    except Exception:
        jit_qp_fn = None

    # Evaluate over episodes
    n_episodes = eval_cfg.get('n_episodes', 50)
    n_steps = eval_cfg.get('n_steps', 500)
    if condition == 'load_following':
        n_steps = eval_cfg.get('load_following_steps', 500)

    all_violation_rates = []
    all_cbf_violation_rates = []
    all_online_times = []

    for ep in range(n_episodes):
        violations, cbf_violations, qp_times = _rollout_lqr_5th(
            dynamics, safety_layer, qp_solver, constraint,
            x0, u0, key, n_steps, jit_qp_fn=jit_qp_fn)
        all_violation_rates.append(violations / n_steps)
        all_cbf_violation_rates.append(cbf_violations / n_steps)
        all_online_times.extend(qp_times)

    return {
        'violation_rate': _mean_std(all_violation_rates),
        'cbf_violation_rate': _mean_std(all_cbf_violation_rates),
        'online_time_ms': _mean_std(all_online_times),
    }


# ---------- Helpers ----------

def _rmse(values, target):
    arr = np.array(values)
    return float(np.sqrt(np.mean((arr - target) ** 2)))


def _mean_std(values):
    arr = np.array(values, dtype=float)
    return (float(np.mean(arr)), float(np.std(arr)))


def _convert(obj):
    """Convert jax/numpy types for JSON serialization."""
    if isinstance(obj, (jnp.ndarray,)):
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, dict):
        return {k: _convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert(v) for v in obj]
    return obj


# ---------- Save / Load ----------

def save_result(result, method_name, condition, seed, results_dir='results/phase5/'):
    """Save result to JSON file."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    path = Path(results_dir) / f'{method_name}_{condition}_seed{seed}.json'
    with open(path, 'w') as f:
        json.dump(_convert(result), f, indent=2)


# ---------- Main entry point ----------

def run_all(config_path=None, methods=None, conditions=None,
            seeds=None, results_dir='results/phase5/'):
    """Run all experiments: methods × conditions × seeds."""
    config = load_config(config_path)

    if methods is None:
        methods = config.get('methods', list(METHODS_5TH.keys()))
    if conditions is None:
        conditions = config.get('conditions', CONDITIONS)
    if seeds is None:
        seeds = list(range(config.get('seeds', 5)))

    total = len(methods) * len(conditions) * len(seeds)
    count = 0

    for method_name in methods:
        for condition in conditions:
            for seed in seeds:
                count += 1
                label = METHOD_LABELS.get(method_name, method_name)
                cond_label = CONDITION_LABELS.get(condition, condition)

                # Skip if already exists
                result_path = Path(results_dir) / f'{method_name}_{condition}_seed{seed}.json'
                if result_path.exists():
                    print(f"\n[{count}/{total}] SKIP {label} | {cond_label} | seed={seed} (exists)",
                          flush=True)
                    continue

                print(f"\n[{count}/{total}] {label} | {cond_label} | seed={seed}",
                      flush=True)

                try:
                    if method_name == 'lqr':
                        result = run_lqr_rhocbf(condition, seed, config)
                    else:
                        result = run_single(method_name, condition, seed, config)
                    save_result(result, method_name, condition, seed, results_dir)
                    vr = result['violation_rate']
                    print(f"  → violation_rate={vr[0]:.4f}±{vr[1]:.4f}, "
                          f"reward={result.get('cumulative_reward', (0.0,))[0]:.1f}", flush=True)
                except Exception as e:
                    print(f"  ✗ FAILED: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

    print(f"\n=== Done: {count}/{total} experiments completed ===", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 5: 5th-order CCS fair comparison')
    parser.add_argument('--methods', nargs='*', default=None,
                        help='Methods to run (default: all 8)')
    parser.add_argument('--conditions', nargs='*', default=None,
                        help='Conditions to run (default: all 8)')
    parser.add_argument('--seeds', type=int, nargs='*', default=None,
                        help='Seeds to run (default: 0-4)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML')
    parser.add_argument('--results-dir', type=str, default='results/phase5/',
                        help='Results directory')
    args = parser.parse_args()

    run_all(config_path=args.config, methods=args.methods,
            conditions=args.conditions, seeds=args.seeds,
            results_dir=args.results_dir)

"""Phase 5 (revision): symmetric NMPC vs RoCBF-Net time-varying rerun.

Motivation
----------
The original ``timevarying_experiment.py`` evaluates NMPC with only 5 episodes
(``n_episodes_nmpc = min(n_episodes, 5)``) while RoCBF-Net is evaluated with
20 episodes. This produces an asymmetric Wilson confidence interval (NMPC's
0/5 vs RoCBF-Net's 0/20 → CI width 52.2% vs 16.1%) and is flagged by the
reviewer panel as a fairness concern.

This script reruns NMPC, NMPC+GP, RoCBF-Net (fixed GP), and RoCBF-Net
(online GP) under the SAME ``n_episodes`` budget (default 20) on the same
S5/S6/S7 time-varying scenarios. The result is a fully symmetric comparison
matrix where every cell in Table~\\ref{tab:timevarying} is supported by
identical statistical power.

Usage
-----
    conda activate jax_gpu
    cd /home/gpu/sz_workspace/RoCBF-Net
    PYTHONPATH=. python experiments/phase5/timevarying_symmetric_rerun.py \\
        --n_episodes 20 --n_steps 300 --max_train 500 --min_train 200

Outputs
-------
    results/phase5/timevarying_symmetric.json — full per-method per-scenario
        violation/reward/solve-time arrays with ``n_episodes`` per cell.
    Console — formatted Wilson-CI summary table ready to drop into
        paper/sections/experiments.tex Table 4 (tab:timevarying).
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.80')
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_enable_command_buffer=')

sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np

from rocbf.baselines.nmpc import NMPCController
from rocbf.baselines.nmpc_gp import NMPCGPController
from rocbf.cbf.robust_hocbf import RobustHOCBF  # noqa: F401 (registered downstream)
from envs.ccs.dynamics import USCCSDynamics
from envs.ccs.constraints import CCSConstraints
from experiments.phase4.methods import _make_robust_hocbf, _pretrain_gp
from experiments.phase5.timevarying_experiment import (
    TimeVaryingDynamics, TIMEVARYING_SCENARIOS, check_hocbf_constraints,
    train_rocbf_net_phase4, evaluate_method,
)


def evaluate_nmpc_symmetric(nmpc_ctrl, dynamics, constraint, x0, u0,
                            n_steps, n_episodes, label):
    """Symmetric NMPC evaluator — same schema as evaluate_method('nmpc')."""
    results = {'violation_rates': [], 'hocbf_violation_rates': [],
               'rewards': [], 'solve_times': []}

    for ep in range(n_episodes):
        dynamics.reset_step_count()
        # Reset NMPC internal disturbance estimator per episode
        if hasattr(nmpc_ctrl, '_d_x'):
            nmpc_ctrl._d_x = np.zeros(nmpc_ctrl.n_x)
        if hasattr(nmpc_ctrl, '_prev_x'):
            nmpc_ctrl._prev_x = None
        if hasattr(nmpc_ctrl, '_prev_v'):
            nmpc_ctrl._prev_v = None
        if hasattr(nmpc_ctrl, '_prev_solution'):
            nmpc_ctrl._prev_solution = None

        x = x0
        violations = 0
        hocbf_violations = 0
        ep_reward = 0.0
        step_times = []

        for t in range(n_steps):
            t0 = time.perf_counter()
            v = nmpc_ctrl.compute_action(x[:3])
            step_times.append((time.perf_counter() - t0) * 1000)

            next_x = dynamics.step_stabilized(x[:3], v)
            u_total = dynamics.compute_total_control(x[:3], v)

            constraint_vals = constraint.check_all(next_x, u_total)
            if any(val < 0 for val in constraint_vals.values()):
                violations += 1

            hocbf_vals = check_hocbf_constraints(constraint, next_x)
            if any(val < 0 for val in hocbf_vals.values()):
                hocbf_violations += 1

            y = dynamics.output(next_x, u_total)
            y_ref = dynamics.output(x0, u0)
            reward = (
                -1.0 * (y[0] - y_ref[0]) ** 2
                - 0.001 * (y[1] - y_ref[1]) ** 2
                - 0.01 * (y[2] - y_ref[2]) ** 2
                - 0.0001 * jnp.sum(v ** 2)
            )
            ep_reward += float(reward)
            x = next_x

        results['violation_rates'].append(violations / n_steps)
        results['hocbf_violation_rates'].append(hocbf_violations / n_steps)
        results['rewards'].append(ep_reward)
        results['solve_times'].append(float(np.mean(step_times)))

        if (ep + 1) % 5 == 0:
            avg_v = np.mean(results['hocbf_violation_rates']) * 100
            print(f"    [{label}] ep {ep+1}/{n_episodes}  "
                  f"hocbf-viol so far: {avg_v:.2f}%", flush=True)

    return results


def run_symmetric_rerun(n_episodes=20, n_steps=300, max_train=500,
                        min_train=200, seed=42, max_gp=3000):
    output_dir = 'results/phase5/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    load_ratio = 1.0
    delay_order = 0

    base_dyn = USCCSDynamics(delay_order=delay_order, load_ratio=load_ratio)
    x0, u0 = base_dyn.equilibrium(load_ratio)
    u0_arr = base_dyn._u0
    constraint = CCSConstraints(
        p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
        power_deviation=50.0, power_target=load_ratio * 1000.0,
        dynamics=base_dyn)

    print("=" * 70, flush=True)
    print("Phase 5 symmetric rerun: 4 controllers × 3 scenarios × "
          f"{n_episodes} episodes", flush=True)
    print("=" * 70, flush=True)

    # ---- Train RoCBF-Net (shared between fixed/online configurations) ----
    print("\nTraining RoCBF-Net (phase4 pipeline)...", flush=True)
    model, safety_layer, qp_solver, gp = train_rocbf_net_phase4(
        constraint, x0, u0, load_ratio, delay_order,
        max_episodes=max_train, min_episodes=min_train, seed=seed)
    safety_layer_fixed = safety_layer
    gp_state = copy.deepcopy(gp)

    # ---- Pre-train GP for NMPC+GP (same pipeline) ----
    print("\nPre-training GP for NMPC+GP baseline...", flush=True)
    key = jax.random.key(seed)
    nmpc_gp_seed_key = jax.random.split(key)[0]
    nmpc_gp_base = _pretrain_gp(load_ratio, delay_order,
                                n_pretrain=max_gp,
                                key=nmpc_gp_seed_key)

    all_results = {}
    for scenario_name, perturbation_fn in TIMEVARYING_SCENARIOS.items():
        print(f"\n{'='*70}", flush=True)
        print(f"Scenario: {scenario_name}  (n_episodes={n_episodes}, "
              f"n_steps={n_steps})", flush=True)
        print(f"{'='*70}", flush=True)

        dyn_nmpc = TimeVaryingDynamics(
            delay_order=delay_order, load_ratio=load_ratio,
            perturbation_fn=perturbation_fn)
        dyn_nmpc_gp = TimeVaryingDynamics(
            delay_order=delay_order, load_ratio=load_ratio,
            perturbation_fn=perturbation_fn)
        dyn_rocbf_fixed = TimeVaryingDynamics(
            delay_order=delay_order, load_ratio=load_ratio,
            perturbation_fn=perturbation_fn)
        dyn_rocbf_online = TimeVaryingDynamics(
            delay_order=delay_order, load_ratio=load_ratio,
            perturbation_fn=perturbation_fn)

        # --- NMPC (symmetric: full n_episodes, not capped at 5) ---
        print(f"  [1/4] NMPC ({n_episodes} eps)...", flush=True)
        nmpc_ctrl = NMPCController(dyn_nmpc, constraint, horizon=20, alpha=0.5)
        nmpc_results = evaluate_nmpc_symmetric(
            nmpc_ctrl, dyn_nmpc, constraint, x0, u0,
            n_steps=n_steps, n_episodes=n_episodes, label='NMPC')

        # --- NMPC + GP (already symmetric in source) ---
        print(f"  [2/4] NMPC+GP ({n_episodes} eps)...", flush=True)
        gp_for_nmpc = copy.deepcopy(nmpc_gp_base)
        nmpc_gp_ctrl = NMPCGPController(
            dyn_nmpc_gp, constraint, gp_for_nmpc,
            horizon=20, use_constant_correction=True)
        nmpc_gp_results = evaluate_nmpc_symmetric(
            nmpc_gp_ctrl, dyn_nmpc_gp, constraint, x0, u0,
            n_steps=n_steps, n_episodes=n_episodes, label='NMPC+GP')

        # --- RoCBF-Net (fixed GP) ---
        print(f"  [3/4] RoCBF-Net fixed-GP ({n_episodes} eps)...", flush=True)
        jit_qp_fn_fixed = jax.jit(safety_layer_fixed.qp_matrices)
        _ = jit_qp_fn_fixed(x0[:3])
        rocbf_fixed_results = evaluate_method(
            model, safety_layer_fixed, qp_solver, dyn_rocbf_fixed, constraint,
            x0, u0, gp=None, base_dyn=base_dyn, u0_arr=u0_arr,
            method_name='rocbf_net',
            n_steps=n_steps, n_episodes=n_episodes, seed=seed,
            jit_qp_fn=jit_qp_fn_fixed)

        # --- RoCBF-Net (online GP with ε-floor) ---
        print(f"  [4/4] RoCBF-Net online-GP ({n_episodes} eps)...", flush=True)
        gp_online = copy.deepcopy(gp_state)
        fixed_eps_mean = float(np.mean(
            rocbf_fixed_results.get('epsilon_log', [40.0])))
        epsilon_floor_val = fixed_eps_mean * 0.9
        safety_layer_online = _make_robust_hocbf(
            base_dyn, constraint, gp_online, u0_arr, epsilon_kappa=1.0,
            use_mean_correction=True, epsilon_floor=epsilon_floor_val)
        jit_qp_fn_online = jax.jit(safety_layer_online.qp_matrices)
        _ = jit_qp_fn_online(x0[:3])
        rocbf_online_results = evaluate_method(
            model, safety_layer_online, qp_solver, dyn_rocbf_online, constraint,
            x0, u0, gp=gp_online, base_dyn=base_dyn, u0_arr=u0_arr,
            method_name='rocbf_net',
            n_steps=n_steps, n_episodes=n_episodes, seed=seed,
            jit_qp_fn=jit_qp_fn_online,
            epsilon_floor=epsilon_floor_val)

        all_results[scenario_name] = {
            'nmpc': nmpc_results,
            'nmpc_gp': nmpc_gp_results,
            'rocbf_net_fixed': rocbf_fixed_results,
            'rocbf_net_online': rocbf_online_results,
            'n_episodes': n_episodes,
            'n_steps': n_steps,
        }

        # Per-scenario quick summary
        nmpc_vr = np.mean(nmpc_results['hocbf_violation_rates']) * 100
        nmpc_gp_vr = np.mean(nmpc_gp_results['hocbf_violation_rates']) * 100
        rocbf_fixed_vr = np.mean(rocbf_fixed_results['hocbf_violation_rates']) * 100
        rocbf_online_vr = np.mean(rocbf_online_results['hocbf_violation_rates']) * 100
        print(f"\n  Scenario {scenario_name} symmetric (n_episodes={n_episodes}):",
              flush=True)
        print(f"    NMPC              hocbf-viol = {nmpc_vr:6.2f}%", flush=True)
        print(f"    NMPC+GP           hocbf-viol = {nmpc_gp_vr:6.2f}%", flush=True)
        print(f"    RoCBF-Net fixed   hocbf-viol = {rocbf_fixed_vr:6.2f}%", flush=True)
        print(f"    RoCBF-Net online  hocbf-viol = {rocbf_online_vr:6.2f}%", flush=True)

    # ---- Save results ----
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    out_path = f'{output_dir}timevarying_symmetric.json'
    with open(out_path, 'w') as f:
        json.dump(_convert(all_results), f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    # ---- Wilson-CI summary table ----
    from experiments.phase4.statistics import format_violation_with_ci

    print(f"\n{'='*110}", flush=True)
    print(f"SYMMETRIC SUMMARY (n_episodes={n_episodes} across all methods, "
          "Wilson 95% CI on episode-level any-violation rate)", flush=True)
    print(f"{'='*110}", flush=True)
    header = (f"{'Scenario':<14}{'NMPC':<22}{'NMPC+GP':<22}"
              f"{'RoCBF-Net (fixed)':<24}{'RoCBF-Net (online)':<24}")
    print(header, flush=True)
    print('-' * 110, flush=True)
    for sc_name in TIMEVARYING_SCENARIOS:
        d = all_results[sc_name]
        cells = []
        for key in ('nmpc', 'nmpc_gp', 'rocbf_net_fixed', 'rocbf_net_online'):
            vr = np.array(d[key]['hocbf_violation_rates'])
            cells.append(format_violation_with_ci(
                int(np.sum(vr > 0)), len(vr)))
        print(f"{sc_name:<14}{cells[0]:<22}{cells[1]:<22}"
              f"{cells[2]:<24}{cells[3]:<24}", flush=True)
    print('=' * 110, flush=True)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_episodes', type=int, default=20,
                        help='Episodes per (method, scenario) cell '
                             '(symmetric across all methods).')
    parser.add_argument('--n_steps', type=int, default=300,
                        help='Steps per episode.')
    parser.add_argument('--max_train', type=int, default=500,
                        help='Max RoCBF-Net PPO training episodes.')
    parser.add_argument('--min_train', type=int, default=200,
                        help='Min RoCBF-Net PPO training episodes.')
    parser.add_argument('--max_gp', type=int, default=3000,
                        help='GP pre-training samples (NMPC+GP and RoCBF-Net).')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_symmetric_rerun(
        n_episodes=args.n_episodes,
        n_steps=args.n_steps,
        max_train=args.max_train,
        min_train=args.min_train,
        max_gp=args.max_gp,
        seed=args.seed,
    )

"""Phase 5: NMPC + GP-mean correction baseline (simplified cautious MPC).

Adds the "GP-augmented NMPC" baseline column to Table~\\ref{tab:timevarying}
by evaluating NMPCGPController on the same S5/S6/S7 time-varying scenarios
already used by the standard NMPC baseline.

Pipeline:
1. Pre-train GP using the same _pretrain_gp(load_ratio, delay_order=0)
   pipeline as RoCBF-Net (3000 samples, 5 mixed scenarios).
2. For each scenario S5/S6/S7, instantiate NMPCGPController with the
   pre-trained GP and evaluate for n_episodes × n_steps.
3. Save results to results/phase5/timevarying_nmpc_gp.json with same
   schema as the NMPC entries in timevarying_results.json so the new
   column can be merged into the LaTeX table.

This is the simplified Hewing-2020 cautious MPC variant: GP provides
mean correction only, no σ_GP propagation through the prediction horizon.
The fixed GP mirrors RoCBF-Net's fixed-GP configuration and preserves
the PAC-Bayes guarantee scope.

Usage:
    PYTHONPATH=. python experiments/phase5/timevarying_nmpc_gp.py \\
        --n_episodes 5 --n_steps 300 --max_train 500
"""
import json
import sys
import os
import time
import copy
from pathlib import Path

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.80')
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_enable_command_buffer=')

sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import jax
import jax.numpy as jnp
import numpy as np

from rocbf.baselines.nmpc_gp import NMPCGPController
from envs.ccs.dynamics import USCCSDynamics
from envs.ccs.constraints import CCSConstraints
from experiments.phase4.methods import _pretrain_gp
from experiments.phase5.timevarying_experiment import (
    TimeVaryingDynamics, TIMEVARYING_SCENARIOS, check_hocbf_constraints,
)


def evaluate_nmpc_gp(nmpc_gp, dynamics, constraint, x0, u0,
                     n_steps=300, n_episodes=5):
    """Evaluate NMPCGPController on a time-varying scenario.

    Mirrors evaluate_method(method_name='nmpc') from timevarying_experiment
    so results JSON has identical schema.
    """
    results = {'violation_rates': [], 'hocbf_violation_rates': [],
               'rewards': [], 'solve_times': [], 'epsilon_log': []}

    for ep in range(n_episodes):
        dynamics.reset_step_count()
        # Reset NMPC internal state per episode (fresh disturbance estimate)
        nmpc_gp._d_x = np.zeros(nmpc_gp.n_x)
        nmpc_gp._prev_x = None
        nmpc_gp._prev_v = None
        nmpc_gp._prev_solution = None
        x = x0
        violations = 0
        hocbf_violations = 0
        ep_reward = 0.0
        step_times = []

        for t in range(n_steps):
            t0 = time.perf_counter()
            v = nmpc_gp.compute_action(x[:3])
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

    return results


def run_nmpc_gp_experiment(n_episodes=5, n_steps=300, max_train=3000,
                            seed=42):
    """Run NMPC+GP on S5/S6/S7 time-varying scenarios."""
    output_dir = 'results/phase5/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    load_ratio = 1.0
    delay_order = 0

    base_dyn = USCCSDynamics(delay_order=delay_order, load_ratio=load_ratio)
    x0, u0 = base_dyn.equilibrium(load_ratio)
    constraint = CCSConstraints(
        p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
        power_deviation=50.0, power_target=load_ratio * 1000.0,
        dynamics=base_dyn)

    # Step 1: Pre-train GP using the SAME pipeline as RoCBF-Net
    # to ensure fair comparison (identical GP information content).
    print("=" * 60, flush=True)
    print("Pre-training GP (mixed scenarios, 3000 samples)...", flush=True)
    print("=" * 60, flush=True)
    key = jax.random.key(seed)
    gp = _pretrain_gp(load_ratio, delay_order, n_pretrain=max_train, key=key)
    print(f"GP pre-trained: n_data={gp.n_training_points}, "
          f"gamma_N={float(gp._gamma_N):.2f}", flush=True)

    all_results = {}

    # Step 2: Evaluate NMPC+GP on each time-varying scenario
    for scenario_name, perturbation_fn in TIMEVARYING_SCENARIOS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Scenario: {scenario_name}", flush=True)
        print(f"{'='*60}", flush=True)

        dynamics = TimeVaryingDynamics(
            delay_order=delay_order, load_ratio=load_ratio,
            perturbation_fn=perturbation_fn)

        # Fresh GP copy per scenario (no online updates, but isolated state)
        gp_scenario = copy.deepcopy(gp)
        nmpc_gp = NMPCGPController(
            dynamics, constraint, gp_scenario,
            horizon=20, use_constant_correction=True)

        print(f"  Evaluating NMPC+GP ({n_episodes} eps, {n_steps} steps)...",
              flush=True)
        results = evaluate_nmpc_gp(
            nmpc_gp, dynamics, constraint, x0, u0,
            n_steps=n_steps, n_episodes=n_episodes)
        vr = np.mean(results['hocbf_violation_rates']) * 100
        full_vr = np.mean(results['violation_rates']) * 100
        st = np.mean(results['solve_times'])
        print(f"  NMPC+GP hocbf-violation: {vr:.2f}%  "
              f"full-violation: {full_vr:.2f}%  "
              f"avg-solve: {st:.1f} ms", flush=True)

        all_results[scenario_name] = {'nmpc_gp': results}

    # Save results
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

    out_path = f'{output_dir}timevarying_nmpc_gp.json'
    with open(out_path, 'w') as f:
        json.dump(_convert(all_results), f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    # Summary table
    print(f"\n{'='*80}", flush=True)
    print("SUMMARY: NMPC+GP (simplified cautious MPC) on time-varying scenarios",
          flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Scenario':<20} {'HOCBF Viol.%':<15} {'Full Viol.%':<15} "
          f"{'Avg Solve (ms)':<15}", flush=True)
    print("-" * 80, flush=True)
    for sc_name in TIMEVARYING_SCENARIOS:
        r = all_results[sc_name]['nmpc_gp']
        vr = np.mean(r['hocbf_violation_rates']) * 100
        full_vr = np.mean(r['violation_rates']) * 100
        st = np.mean(r['solve_times'])
        print(f"{sc_name:<20} {vr:<15.2f} {full_vr:<15.2f} {st:<15.1f}",
              flush=True)
    print(f"{'='*80}", flush=True)

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=5)
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--max_train', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_nmpc_gp_experiment(
        n_episodes=args.n_episodes,
        n_steps=args.n_steps,
        max_train=args.max_train,
        seed=args.seed,
    )

"""Systematic inter-sample safety analysis for reviewer response.

Extends the existing jpc_process_metrics.py inter-sample check to cover:
- All 6 static perturbation scenarios (S1-S6) + Nominal
- 5 seeds per scenario
- New metrics: maximum inter-sample excursion magnitude, violation duration

Uses 4 sub-step interpolation per sampling interval (ZOH, Ts=1s).
Self-contained 5th-order GP training via common_5th module.

Optimized: JIT-precompiles qp_matrices (pure JAX) to avoid eager-mode
overhead (~500ms/call → ~1-5ms/call). scipy SLSQP stays outside JIT.

Usage:
    conda activate jax_gpu
    XLA_PYTHON_CLIENT_PREALLOCATE=false python experiments/phase5/inter_sample_analysis.py
"""
import sys, json, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

# Force unbuffered output for progress visibility
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import jax, jax.numpy as jnp, numpy as np
from functools import partial
from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase4.methods import _make_robust_hocbf
from experiments.phase5.common_5th import train_gp_5th

LOAD_RATIO = 1.0
N_GP_PRETRAIN = 500
N_EVAL = 200           # 200 steps × 4 sub-steps = 800 constraint checks per seed
N_SUB_STEPS = 4
N_SEEDS = 5
RESULTS_DIR = 'results/phase5/inter_sample'
os.makedirs(RESULTS_DIR, exist_ok=True)

SCENARIOS = {
    'Nominal':       None,
    'S1: Heat':      'heat_absorption',
    'S2: Pressure':  'pressure_oscillation',
    'S3: Coupled':   'coupled',
    'S4: Nonlinear': 'nonlinear_fouling',
    'S5: Valve':     'valve_degradation',
    'S6: Fuel':      'fuel_quality',
}


def evaluate_inter_sample(scenario_key, seed=0):
    """Detailed inter-sample safety evaluation for one scenario x seed.

    Optimized: JIT-compiles robust_hocbf.qp_matrices once per seed.
    The compilation (~10-30s) is amortized over N_EVAL steps, saving
    ~500ms per call vs eager mode.
    """
    t_total = time.time()
    rng_key = jax.random.key(seed)

    dynamics = USCCSDynamics5th(load_ratio=LOAD_RATIO)
    x0, u0 = dynamics.equilibrium(LOAD_RATIO)
    constraint = CCSConstraints5th(
        p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
        power_deviation=50.0, power_target=LOAD_RATIO * 1000.0)

    # Environment (nominal or perturbed)
    if scenario_key is not None:
        env = UncertainUSCCSDynamics5th(
            load_ratio=LOAD_RATIO, uncertainty_scenario=scenario_key)
    else:
        env = USCCSDynamics5th(load_ratio=LOAD_RATIO)

    # Train GP
    t0 = time.time()
    gp = train_gp_5th(scenario_key, N_GP_PRETRAIN, rng_key,
                      load_ratio=LOAD_RATIO)
    t_gp = time.time() - t0
    print(f" GP[{t_gp:.0f}s]", end='', flush=True)

    # Build filter
    t0 = time.time()
    robust_hocbf = _make_robust_hocbf(
        dynamics, constraint, gp, u0,
        use_mean_correction=True, epsilon_kappa=1.0,
        epsilon_floor=0.0)
    qp = DifferentiableQP(v_max=5.0, scale_constraints=True)
    t_build = time.time() - t0
    print(f" build[{t_build:.0f}s]", end='', flush=True)

    # JIT-precompile qp_matrices.
    # The compositional epsilon(x) involves multiple jax.grad calls through
    # the psi-chain + GP prediction. JIT compilation takes 10-30s but reduces
    # per-call latency from ~500ms (eager) to ~1-5ms (JIT).
    @jax.jit
    def _jit_qp_mats(x):
        return robust_hocbf.qp_matrices(x)

    t0 = time.time()
    _ = _jit_qp_mats(x0)  # trigger compilation
    t_compile = time.time() - t0
    print(f" JIT[{t_compile:.0f}s]", end='', flush=True)

    # Evaluation
    inter_sample_violations = np.zeros(N_EVAL, dtype=bool)
    max_excursions = np.zeros(N_EVAL)
    violation_duration = np.zeros(N_EVAL, dtype=int)
    cbf_violations = np.zeros(N_EVAL, dtype=bool)

    x = x0.copy()
    t_eval_start = time.time()

    for t in range(N_EVAL):
        v_rl = jnp.zeros(3)

        # QP filter: JIT-compiled qp_matrices + scipy SLSQP (~70ms/step total)
        A, b = _jit_qp_mats(x)
        result = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
        v_safe = jnp.asarray(result[0] if isinstance(result, tuple) else result)

        # Step with perturbation
        next_x = env.step_stabilized_phi_scaled(x, v_safe)

        # Check constraint at sample point
        cv_sample = constraint.check_all(next_x)
        if any(v < 0 for v in cv_sample.values()):
            cbf_violations[t] = True

        # Inter-sample check: evaluate constraints at finer sub-step grid
        max_excursion_this_step = 0.0
        n_sub_violations = 0

        for k in range(1, N_SUB_STEPS + 1):
            alpha = k / N_SUB_STEPS
            x_interp = x + alpha * (next_x - x)
            cv = constraint.check_all(x_interp)

            for key, val in cv.items():
                if val < 0:
                    excursion = float(-val)
                    max_excursion_this_step = max(max_excursion_this_step, excursion)

            if any(v < 0 for v in cv.values()):
                n_sub_violations += 1

        inter_sample_violations[t] = (n_sub_violations > 0)
        max_excursions[t] = max_excursion_this_step
        violation_duration[t] = n_sub_violations

        x = next_x

        # Progress every 50 steps
        if (t + 1) % 50 == 0:
            pct_done = (t + 1) / N_EVAL * 100
            elapsed = time.time() - t_eval_start
            ms_per_step = 1000 * elapsed / (t + 1)
            print(f" {t+1}/{N_EVAL}[{ms_per_step:.0f}ms]", end='', flush=True)

    t_eval = time.time() - t_eval_start
    t_total_elapsed = time.time() - t_total
    print(f" done eval[{t_eval:.0f}s] total[{t_total_elapsed:.0f}s]", flush=True)

    return {
        'scenario': scenario_key if scenario_key else 'nominal',
        'seed': seed,
        'cbf_violation_pct': float(np.mean(cbf_violations) * 100),
        'inter_sample_violation_pct': float(np.mean(inter_sample_violations) * 100),
        'max_excursion': float(np.max(max_excursions)),
        'mean_excursion': float(np.mean(max_excursions[max_excursions > 0])) if np.any(max_excursions > 0) else 0.0,
        'max_violation_duration': int(np.max(violation_duration)),
        'mean_violation_duration': float(np.mean(violation_duration[violation_duration > 0])) if np.any(violation_duration > 0) else 0.0,
        'steps_with_inter_violation': int(np.sum(inter_sample_violations)),
        'total_steps': N_EVAL,
        'n_sub_steps': N_SUB_STEPS,
        'timing_gp_s': t_gp,
        'timing_jit_s': t_compile,
        'timing_eval_s': t_eval,
    }


def main():
    print("=" * 70)
    print("Systematic Inter-Sample Safety Analysis (OPTIMIZED)")
    print(f"Scenarios: {len(SCENARIOS)}, Seeds: {N_SEEDS}, "
          f"Steps: {N_EVAL}, Sub-steps: {N_SUB_STEPS}")
    print("JIT: qp_matrices compiled once per seed (amortized over {N_EVAL} steps)")
    print("=" * 70)

    all_results = []

    for label, scenario_key in SCENARIOS.items():
        print(f"\n{'─' * 50}")
        print(f"Scenario: {label}")
        print(f"{'─' * 50}")

        scenario_results = []
        for seed in range(N_SEEDS):
            print(f"  Seed {seed}...", end='', flush=True)
            t0 = time.time()
            result = evaluate_inter_sample(scenario_key, seed=seed)
            result['scenario_label'] = label
            scenario_results.append(result)
            all_results.append(result)
            elapsed = time.time() - t0
            print(f"  → CBF={result['cbf_violation_pct']:.2f}% "
                  f"Inter={result['inter_sample_violation_pct']:.2f}% "
                  f"MaxExc={result['max_excursion']:.4f} [{elapsed:.0f}s total]")

        # Scenario summary
        cbf_arr = np.array([r['cbf_violation_pct'] for r in scenario_results])
        inter_arr = np.array([r['inter_sample_violation_pct'] for r in scenario_results])
        exc_arr = np.array([r['max_excursion'] for r in scenario_results])
        jit_times = np.array([r['timing_jit_s'] for r in scenario_results])
        eval_times = np.array([r['timing_eval_s'] for r in scenario_results])

        print(f"  Summary: CBF={cbf_arr.mean():.2f}±{cbf_arr.std():.2f}% "
              f"Inter={inter_arr.mean():.2f}±{inter_arr.std():.2f}% "
              f"MaxExc={exc_arr.mean():.4f}±{exc_arr.std():.4f}")
        print(f"  Timing:  JIT={jit_times.mean():.0f}±{jit_times.std():.0f}s  "
              f"Eval={eval_times.mean():.0f}±{eval_times.std():.0f}s")

    # Save full results
    output_path = os.path.join(RESULTS_DIR, 'inter_sample_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {output_path}")

    # =================================================================
    # LaTeX Table S8
    # =================================================================
    print("\n" + "=" * 70)
    print("LATEX TABLE S8 (copy to supplementary.tex)")
    print("=" * 70)

    scenario_agg = {}
    for r in all_results:
        lbl = r['scenario_label']
        if lbl not in scenario_agg:
            scenario_agg[lbl] = {k: [] for k in ['cbf', 'inter', 'exc', 'dur']}
        scenario_agg[lbl]['cbf'].append(r['cbf_violation_pct'])
        scenario_agg[lbl]['inter'].append(r['inter_sample_violation_pct'])
        scenario_agg[lbl]['exc'].append(r['max_excursion'])
        scenario_agg[lbl]['dur'].append(r['max_violation_duration'])

    pct = r"\%"
    for label in SCENARIOS:
        cbf_m = np.mean(scenario_agg[label]['cbf'])
        cbf_s = np.std(scenario_agg[label]['cbf'])
        inter_m = np.mean(scenario_agg[label]['inter'])
        inter_s = np.std(scenario_agg[label]['inter'])
        exc_m = np.mean(scenario_agg[label]['exc'])
        exc_s = np.std(scenario_agg[label]['exc'])
        dur_m = np.mean(scenario_agg[label]['dur'])
        dur_s = np.std(scenario_agg[label]['dur'])
        print(f'        {label} & ${cbf_m:.2f}\\pm{cbf_s:.2f}$ & '
              f'${inter_m:.2f}\\pm{inter_s:.2f}$ & '
              f'${exc_m:.4f}\\pm{exc_s:.4f}$ & '
              f'${dur_m:.1f}\\pm{dur_s:.1f}$ \\\\')

    # Summary
    all_inter = [r['inter_sample_violation_pct'] for r in all_results]
    all_cbf = [r['cbf_violation_pct'] for r in all_results]
    print(f"\nGlobal summary ({len(all_results)} runs):")
    print(f"  CBF violation:       {np.mean(all_cbf):.3f} ± {np.std(all_cbf):.3f}%")
    print(f"  Inter-sample viol.:  {np.mean(all_inter):.3f} ± {np.std(all_inter):.3f}%")


if __name__ == '__main__':
    main()

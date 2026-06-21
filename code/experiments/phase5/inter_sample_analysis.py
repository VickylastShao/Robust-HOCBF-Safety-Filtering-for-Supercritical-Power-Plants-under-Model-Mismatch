"""Systematic inter-sample safety analysis for reviewer response.

Extends the existing jpc_process_metrics.py inter-sample check to cover:
- All 6 static perturbation scenarios (S1-S6) + Nominal
- 5 seeds per scenario
- New metrics: maximum inter-sample excursion magnitude, violation duration

Uses 4 sub-step interpolation per sampling interval (ZOH, Ts=1s).
Self-contained 5th-order GP training via common_5th module.

Usage:
    conda activate jax_gpu
    XLA_PYTHON_CLIENT_PREALLOCATE=false python experiments/phase5/inter_sample_analysis.py
"""
import sys, json, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

import jax, jax.numpy as jnp, numpy as np
from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase4.methods import _make_robust_hocbf
from experiments.phase5.common_5th import train_gp_5th

LOAD_RATIO = 1.0
N_GP_PRETRAIN = 2000
N_EVAL = 500
N_SUB_STEPS = 4        # Interpolation resolution per sampling interval
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

CONSTRAINT_NAMES = [
    'pressure_high', 'pressure_low',
    'enthalpy_high', 'enthalpy_low',
    'power_high', 'power_low',
]


def evaluate_inter_sample(scenario_key, seed=0):
    """Detailed inter-sample safety evaluation for one scenario x seed."""
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

    # Train GP using 5th-order dynamics
    gp = train_gp_5th(scenario_key, N_GP_PRETRAIN, rng_key,
                      load_ratio=LOAD_RATIO)

    robust_hocbf = _make_robust_hocbf(
        dynamics, constraint, gp, u0,
        use_mean_correction=True, epsilon_kappa=1.0,
        epsilon_floor=0.0)
    qp = DifferentiableQP(v_max=5.0, scale_constraints=True)

    # Logging
    inter_sample_violations = np.zeros(N_EVAL, dtype=bool)
    max_excursions = np.zeros(N_EVAL)  # Max constraint violation depth per step
    violation_duration = np.zeros(N_EVAL, dtype=int)  # Consecutive sub-step violations
    cbf_violations = np.zeros(N_EVAL, dtype=bool)     # At sample points

    x = x0.copy()

    for t in range(N_EVAL):
        # Policy action (LQR: zero incremental = equilibrium tracking)
        v_rl = jnp.zeros(3)

        # QP filter
        A, b = robust_hocbf.qp_matrices(x)
        result = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
        if isinstance(result, tuple):
            v_safe = result[0]
        else:
            v_safe = result
        v_safe = jnp.asarray(v_safe)

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

            # Track worst-case constraint violation depth
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
    }


def main():
    print("=" * 70)
    print("Systematic Inter-Sample Safety Analysis")
    print(f"Scenarios: {len(SCENARIOS)}, Seeds: {N_SEEDS}, "
          f"Steps: {N_EVAL}, Sub-steps: {N_SUB_STEPS}")
    print("=" * 70)

    all_results = []

    for label, scenario_key in SCENARIOS.items():
        print(f"\n{'─' * 50}")
        print(f"Scenario: {label}")
        print(f"{'─' * 50}")

        scenario_results = []
        for seed in range(N_SEEDS):
            t0 = time.time()
            print(f"  Seed {seed}...", end=' ', flush=True)
            result = evaluate_inter_sample(scenario_key, seed=seed)
            result['scenario_label'] = label
            scenario_results.append(result)
            all_results.append(result)
            elapsed = time.time() - t0
            print(f"CBF={result['cbf_violation_pct']:.2f}% "
                  f"Inter={result['inter_sample_violation_pct']:.2f}% "
                  f"MaxExc={result['max_excursion']:.4f} [{elapsed:.0f}s]")

        # Aggregate across seeds
        cbf_arr = np.array([r['cbf_violation_pct'] for r in scenario_results])
        inter_arr = np.array([r['inter_sample_violation_pct'] for r in scenario_results])
        exc_arr = np.array([r['max_excursion'] for r in scenario_results])

        print(f"  → Mean: CBF={cbf_arr.mean():.2f}±{cbf_arr.std():.2f}% "
              f"Inter={inter_arr.mean():.2f}±{inter_arr.std():.2f}% "
              f"MaxExc={exc_arr.mean():.4f}±{exc_arr.std():.4f}")

    # Save full results
    output_path = os.path.join(RESULTS_DIR, 'inter_sample_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {output_path}")

    # Generate LaTeX table
    print("\n" + "=" * 70)
    print("LATEX TABLE S8 (for Supplementary Material)")
    print("=" * 70)

    # Aggregate by scenario
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
    print(r"\begin{table}[htbp]")
    print(r"    \centering")
    print(f"    \\caption{{Systematic inter-sample constraint satisfaction analysis under "
          f"zero-order hold ($T_s=1$\\,s, {N_SUB_STEPS} sub-step interpolation per sampling "
          f"interval, {N_SEEDS} seeds $\\times$ {N_EVAL} steps per scenario). "
          f"``Max.\\ excursion'': maximum constraint violation depth at any sub-step "
          f"(in physical units of the most violated constraint).}}")
    print(r"    \label{tab:inter_sample}")
    print(r"    \footnotesize")
    print(r"    \begin{tabular}{lcccc}")
    print(r"        \toprule")
    print(f"        Scenario & CBF viol.\\ ({pct}) & Inter-sample viol.\\ ({pct}) & "
          f"Max.\\ excursion & Max.\\ sub-step dur. \\\\")
    print(r"        \midrule")

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

    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"\end{table}")

    # Summary
    all_inter = [r['inter_sample_violation_pct'] for r in all_results]
    all_cbf = [r['cbf_violation_pct'] for r in all_results]
    print(f"\nGlobal summary ({len(all_results)} runs):")
    print(f"  CBF violation:       {np.mean(all_cbf):.3f} ± {np.std(all_cbf):.3f}%")
    print(f"  Inter-sample viol.:  {np.mean(all_inter):.3f} ± {np.std(all_inter):.3f}%")


if __name__ == '__main__':
    main()

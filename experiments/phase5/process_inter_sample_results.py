"""Process inter-sample analysis results and generate LaTeX Table S8 rows.

Usage:
    python experiments/phase5/process_inter_sample_results.py
"""
import json, os, sys
import numpy as np

RESULTS_PATH = 'results/phase5/inter_sample/inter_sample_analysis.json'
SCENARIO_ORDER = [
    'Nominal', 'S1: Heat', 'S2: Pressure', 'S3: Coupled',
    'S4: Nonlinear', 'S5: Valve', 'S6: Fuel'
]


def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"Error: {RESULTS_PATH} not found")
        sys.exit(1)

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    print(f"Loaded {len(results)} results from {RESULTS_PATH}")

    # Aggregate by scenario
    scenario_data = {}
    for r in results:
        lbl = r.get('scenario_label', r.get('scenario', 'unknown'))
        if lbl not in scenario_data:
            scenario_data[lbl] = {
                'cbf': [], 'inter': [], 'exc': [], 'dur': [],
                'mean_exc': [], 'max_dur': [], 'steps_with_viol': [],
                'total_steps': r.get('total_steps', 200),
                'n_sub_steps': r.get('n_sub_steps', 4),
            }
        sd = scenario_data[lbl]
        sd['cbf'].append(r['cbf_violation_pct'])
        sd['inter'].append(r['inter_sample_violation_pct'])
        sd['exc'].append(r['max_excursion'])
        sd['dur'].append(r.get('max_violation_duration', 0))
        sd['mean_exc'].append(r.get('mean_excursion', 0))
        sd['max_dur'].append(r.get('max_violation_duration', 0))
        sd['steps_with_viol'].append(r.get('steps_with_inter_violation', 0))

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for lbl in SCENARIO_ORDER:
        if lbl not in scenario_data:
            continue
        sd = scenario_data[lbl]
        print(f"  {lbl:16s}: CBF={np.mean(sd['cbf']):.2f}±{np.std(sd['cbf']):.2f}%  "
              f"Inter={np.mean(sd['inter']):.2f}±{np.std(sd['inter']):.2f}%  "
              f"MaxExc={np.mean(sd['exc']):.4f}±{np.std(sd['exc']):.4f}  "
              f"Dur={np.mean(sd['dur']):.1f}±{np.std(sd['dur']):.1f}")

    # Count scenarios with any inter-sample violations
    any_viol = sum(1 for lbl in SCENARIO_ORDER if lbl in scenario_data
                   and np.mean(scenario_data[lbl]['inter']) > 0)
    zero_viol = sum(1 for lbl in SCENARIO_ORDER if lbl in scenario_data
                    and np.mean(scenario_data[lbl]['inter']) == 0)

    print(f"\n  Scenarios with zero inter-sample violations: {zero_viol}")
    print(f"  Scenarios with any inter-sample violations: {any_viol}")

    # =================================================================
    # LaTeX Table S8
    # =================================================================
    print("\n" + "=" * 80)
    print("LATEX TABLE S8 ROWS (copy to supplementary.tex)")
    print("=" * 80)

    for lbl in SCENARIO_ORDER:
        if lbl not in scenario_data:
            print(f"        {lbl} & -- & -- & -- & -- \\\\")
            continue
        sd = scenario_data[lbl]
        cbf_m = np.mean(sd['cbf'])
        cbf_s = np.std(sd['cbf'])
        inter_m = np.mean(sd['inter'])
        inter_s = np.std(sd['inter'])
        exc_m = np.mean(sd['exc'])
        exc_s = np.std(sd['exc'])
        dur_m = np.mean(sd['dur'])
        dur_s = np.std(sd['dur'])

        # Bold zero-violation rows
        bf = r'\mathbf' if cbf_m == 0 and inter_m == 0 else ''
        bf_end = '}' if bf else ''

        if bf:
            print(f'        {lbl} & ${bf}{{{cbf_m:.2f}\\pm{cbf_s:.2f}}}$ & '
                  f'${bf}{{{inter_m:.2f}\\pm{inter_s:.2f}}}$ & '
                  f'${bf}{{{exc_m:.4f}\\pm{exc_s:.4f}}}$ & '
                  f'${bf}{{{dur_m:.1f}\\pm{dur_s:.1f}}}$ \\\\')
        else:
            print(f'        {lbl} & ${cbf_m:.2f}\\pm{cbf_s:.2f}$ & '
                  f'${inter_m:.2f}\\pm{inter_s:.2f}$ & '
                  f'${exc_m:.4f}\\pm{exc_s:.4f}$ & '
                  f'${dur_m:.1f}\\pm{dur_s:.1f}$ \\\\')

    # Global summary
    all_cbf = []
    all_inter = []
    for lbl in SCENARIO_ORDER:
        if lbl in scenario_data:
            all_cbf.extend(scenario_data[lbl]['cbf'])
            all_inter.extend(scenario_data[lbl]['inter'])

    n_seeds = len(results) // max(1, len([s for s in SCENARIO_ORDER if s in scenario_data]))
    # Get config from first scenario that has data
    ref_sd = next(iter(scenario_data.values()))
    print(f"\n  Configuration: {len(scenario_data)} scenarios x "
          f"{n_seeds} seeds x {ref_sd['total_steps']} steps x "
          f"{ref_sd['n_sub_steps']} sub-steps")
    print(f"  Global CBF violation:       {np.mean(all_cbf):.3f} ± {np.std(all_cbf):.3f}%")
    print(f"  Global inter-sample viol.:  {np.mean(all_inter):.3f} ± {np.std(all_inter):.3f}%")

    # =================================================================
    # Discussion paragraph guidance
    # =================================================================
    print("\n" + "=" * 80)
    print("DISCUSSION PARAGRAPH GUIDANCE")
    print("=" * 80)

    max_inter = max(np.mean(scenario_data[lbl]['inter'])
                    for lbl in SCENARIO_ORDER if lbl in scenario_data
                    and lbl != 'Nominal')
    max_exc_scenario = max(
        (lbl for lbl in SCENARIO_ORDER if lbl in scenario_data and lbl != 'Nominal'),
        key=lambda l: np.mean(scenario_data[l]['exc']))

    if max_inter == 0:
        print("ALL ZERO: The filter provides perfect inter-sample safety across all scenarios.")
        print("Keep existing discussion text emphasizing controller-agnostic safety guarantee.")
    elif max_inter < 5:
        print(f"LOW (<5%): Max inter-sample violation {max_inter:.1f}%.")
        print("Mostly consistent with ZOH claim. Minor caveats for specific scenarios.")
    else:
        print(f"SIGNIFICANT (>{max_inter:.1f}%): Inter-sample violations exist.")
        print(f"Largest excursions in {max_exc_scenario}.")
        print("Need to distinguish: LQR-only vs PPO-augmented, 5th-order vs 3rd-order.")
        print("Recommend explicitly noting LQR baseline limitation vs trained PPO.")


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Run LQR+RHOCBF experiment across all scenarios."""
import sys
sys.stdout.reconfigure(line_buffering=True)

from rocbf.baselines.lqr_rhocbf import run_lqr_rhocbf_experiment

scenarios = [None, 'heat_absorption', 'pressure_oscillation', 'coupled', 'nonlinear']
labels = {None: 'Nominal', 'heat_absorption': 'S1:Heat',
          'pressure_oscillation': 'S2:Pressure',
          'coupled': 'S3:Coupled', 'nonlinear': 'S4:Nonlinear'}

all_results = {}
for scenario in scenarios:
    label = labels.get(scenario, str(scenario))
    print(f'\n{"="*60}', flush=True)
    print(f'LQR + Robust-HOCBF: {label}', flush=True)
    print(f'{"="*60}', flush=True)
    result = run_lqr_rhocbf_experiment(
        load_ratio=1.0, scenario=scenario, n_seeds=5, n_steps=300)
    agg = result['aggregate']
    print(f'Summary: reward={agg["reward_mean"]:.1f}+-{agg["reward_std"]:.1f}, '
          f'cbf_viol={agg["cbf_violation_rate_mean"]:.1%}+-{agg["cbf_violation_rate_std"]:.1%}, '
          f'QP_int={agg["qp_intervention_rate_mean"]:.1%}+-{agg["qp_intervention_rate_std"]:.1%}',
          flush=True)
    all_results[label] = agg

print('\n\n' + '='*80, flush=True)
print('LQR + Robust-HOCBF Summary Table', flush=True)
print('='*80, flush=True)
print(f'{"Scenario":<15} {"Reward":>12} {"CBF Viol":>12} {"QP Int":>12}', flush=True)
print('-'*51, flush=True)
for label, agg in all_results.items():
    print(f'{label:<15} {agg["reward_mean"]:>8.1f}+-{agg["reward_std"]:<3.1f} '
          f'{agg["cbf_violation_rate_mean"]:>8.1%}+-{agg["cbf_violation_rate_std"]:<3.1%} '
          f'{agg["qp_intervention_rate_mean"]:>8.1%}+-{agg["qp_intervention_rate_std"]:<3.1%}',
          flush=True)

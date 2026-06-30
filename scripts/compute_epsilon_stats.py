#!/usr/bin/env python3
"""Compute aggregate statistics from epsilon ablation experiments.
Reads results/phase5/epsilon_ablation/epsilon_ablation.json and computes
mean±std across seeds for each (GP type, epsilon mode) combination.
"""

import json
import sys
import numpy as np

def main():
    json_path = 'results/phase5/epsilon_ablation/epsilon_ablation.json'
    with open(json_path) as f:
        data = json.load(f)
    
    # Metrics to aggregate
    metrics = ['cbf_violation_rate', 'violation_rate', 'qp_infeasible_rate', 'mean_epsilon', 'reward']
    
    # Collect data per (gp_type, epsilon_mode) across seeds
    results = {}
    for gp_key, seeds in data.items():
        gp_type = gp_key.replace('gp_', '')
        for seed_key, modes in seeds.items():
            seed = int(seed_key.replace('seed_', ''))
            for mode, values in modes.items():
                key = (gp_type, mode)
                if key not in results:
                    results[key] = {m: [] for m in metrics}
                for m in metrics:
                    if m in values:
                        results[key][m].append(values[m])
    
    # Compute mean±std
    print(f"{'GP Type':<20} {'ε Mode':<15} {'CBF Viol':>12} {'Total Viol':>12} {'QP Infeas':>12} {'Mean ε':>10} {'Reward':>12}")
    print("-" * 95)
    
    for (gp_type, mode), values in sorted(results.items()):
        n_seeds = len(values['cbf_violation_rate'])
        if n_seeds < 2:
            # Not enough seeds for std
            cbf = values['cbf_violation_rate'][0] * 100
            total = values['violation_rate'][0] * 100
            qp = values['qp_infeasible_rate'][0] * 100
            eps = values['mean_epsilon'][0]
            rwd = values['reward'][0]
            print(f"{gp_type:<20} {mode:<15} {cbf:>11.2f}% {total:>11.3f}% {qp:>11.2f}% {eps:>10.3f} {rwd:>12.1f}")
        else:
            cbf_mean = np.mean(values['cbf_violation_rate']) * 100
            cbf_std = np.std(values['cbf_violation_rate']) * 100
            total_mean = np.mean(values['violation_rate']) * 100
            total_std = np.std(values['violation_rate']) * 100
            qp_mean = np.mean(values['qp_infeasible_rate']) * 100
            qp_std = np.std(values['qp_infeasible_rate']) * 100
            eps_mean = np.mean(values['mean_epsilon'])
            eps_std = np.std(values['mean_epsilon'])
            rwd_mean = np.mean(values['reward'])
            rwd_std = np.std(values['reward'])
            
            print(f"{gp_type:<20} {mode:<15} {cbf_mean:>5.2f}±{cbf_std:.2f}% {total_mean:>5.3f}±{total_std:.3f}% {qp_mean:>5.2f}±{qp_std:.2f}% {eps_mean:>5.3f}±{eps_std:.3f} {rwd_mean:>7.1f}±{rwd_std:.1f}")
    
    # Print summary for LaTeX table
    print("\n\n=== LaTeX Table Format ===")
    print(f"{'GP':<15} {'ε':<15} {'CBF':>10} {'Total':>10} {'QP':>10} {'ε̄':>10}")
    
    # Order for paper
    order = [
        ('scenario_specific', 'compositional'),
        ('scenario_specific', 'constant_mean'),
        ('scenario_specific', 'constant_max'),
        ('scenario_specific', 'no_epsilon'),
    ]
    
    for gp_type, mode in order:
        key = (gp_type, mode)
        if key in results:
            values = results[key]
            n_seeds = len(values['cbf_violation_rate'])
            cbf_vals = [v * 100 for v in values['cbf_violation_rate']]
            total_vals = [v * 100 for v in values['violation_rate']]
            qp_vals = [v * 100 for v in values['qp_infeasible_rate']]
            eps_vals = values['mean_epsilon']
            
            def fmt(vals, pct=True):
                m, s = np.mean(vals), np.std(vals)
                if s < 0.005:
                    return f"{m:.2f}"
                else:
                    return f"{m:.2f}±{s:.2f}"
            
            cbf_str = fmt(cbf_vals)
            total_str = fmt(total_vals)
            qp_str = fmt(qp_vals)
            eps_str = fmt(eps_vals, pct=False)
            
            print(f"{gp_type:<15} {mode:<15} {cbf_str:>10} {total_str:>10} {qp_str:>10} {eps_str:>10}")

if __name__ == '__main__':
    main()

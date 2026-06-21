"""E1: Sparse GP n_samples sweep for triple integrator (R4 P0-5 + DA CRITICAL #1).

Addresses Devil's Advocate cherry-picking concern: triple integrator sparse GP
with n=50 was a deliberately constructed scenario. This experiment shows the
ε(x) advantage is robust across n ∈ {25, 50, 100, 200} and multiple seeds.

Output: results/phase5/e1_sparse_gp_n_sweep/{n_samples}.json
"""
import json
import sys
import os
import time
from pathlib import Path

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.50')
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

# Reuse the existing single-experiment driver
from experiments.phase5.m3_sparse_gp_demo import run_sparse_gp_experiment

OUTPUT_DIR = Path('results/phase5/e1_sparse_gp_n_sweep')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sweep grid
N_SAMPLES_GRID = [25, 50, 100, 200]
N_SEEDS = 5
UNCERTAINTY_SCALE = 0.5  # locked from m3_sparse_gp_demo final config

summary = {}

for n_samples in N_SAMPLES_GRID:
    print(f"\n{'='*80}\nE1 sweep: n_samples = {n_samples} (5 seeds)\n{'='*80}")
    t0 = time.time()
    res = run_sparse_gp_experiment(
        uncertainty_scale=UNCERTAINTY_SCALE,
        n_sparse_samples=n_samples,
        n_seeds=N_SEEDS,
        scenario='nonlinear',
    )
    elapsed = time.time() - t0

    # Pull per-seed summaries
    per_seed = {}
    for seed in range(N_SEEDS):
        k = f'seed_{seed}'
        if k not in res:
            continue
        per_seed[k] = {
            'epsilon_stats': res[k].get('epsilon_stats', {}),
        }
        for mode in ['compositional', 'constant_mean', 'constant_max', 'no_epsilon']:
            if mode in res[k]:
                m = res[k][mode]
                per_seed[k][mode] = {
                    'violation_rate': m.get('violation_rate'),
                    'cbf_violation_rate': m.get('cbf_violation_rate'),
                    'qp_infeasible_rate': m.get('qp_infeasible_rate'),
                    'qp_intervention_rate': m.get('qp_intervention_rate'),
                    'near_constraint_fraction': m.get('near_constraint_fraction'),
                    'epsilon_mean': m.get('epsilon_mean'),
                }

    summary[f'n_{n_samples}'] = {
        'n_samples': n_samples,
        'n_seeds': N_SEEDS,
        'uncertainty_scale': UNCERTAINTY_SCALE,
        'per_seed': per_seed,
        'elapsed_seconds': elapsed,
    }

    # Save incremental
    with open(OUTPUT_DIR / f'n_{n_samples}.json', 'w') as f:
        json.dump(summary[f'n_{n_samples}'], f, indent=2)

    print(f"  n={n_samples} done in {elapsed:.0f}s")

with open(OUTPUT_DIR / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nE1 complete. Results in {OUTPUT_DIR}")

# Print compact table for paper
print("\n{:>6} | {:>10} | {:>14} | {:>14} | {:>14} | {:>14}".format(
    "n", "mode", "viol%(mean±std)", "cbf%(mean±std)", "qp_inf%(mean)", "ε̄(mean)"))
print("-" * 90)
import numpy as np
for n_samples in N_SAMPLES_GRID:
    seed_data = summary[f'n_{n_samples}']['per_seed']
    for mode in ['compositional', 'constant_mean', 'constant_max', 'no_epsilon']:
        vals = []
        cbfs = []
        qpinf = []
        eps = []
        for sk in seed_data:
            if mode in seed_data[sk]:
                vals.append(seed_data[sk][mode]['violation_rate'] * 100)
                cbfs.append(seed_data[sk][mode]['cbf_violation_rate'] * 100)
                qpinf.append(seed_data[sk][mode]['qp_infeasible_rate'] * 100)
                eps.append(seed_data[sk][mode]['epsilon_mean'])
        if vals:
            print("{:>6} | {:>10} | {:>5.2f}±{:>5.2f} | {:>5.2f}±{:>5.2f} | {:>5.2f}±{:>5.2f} | {:>5.4f}".format(
                n_samples, mode,
                float(np.mean(vals)), float(np.std(vals)),
                float(np.mean(cbfs)), float(np.std(cbfs)),
                float(np.mean(qpinf)), float(np.std(qpinf)),
                float(np.mean(eps)),
            ))

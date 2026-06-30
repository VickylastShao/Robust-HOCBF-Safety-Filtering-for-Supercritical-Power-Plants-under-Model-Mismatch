"""Epsilon-Kappa Sensitivity Sweep.

Sweeps ε_κ across [0.0, 0.1, 0.3, 0.5, 1.0] for 3 key conditions
(S2:Pressure, S3:Coupled, S4:Nonlinear) to demonstrate that:
  - Optimal κ differs by uncertainty type
  - κ=0 is best for additive uncertainty (S2, S4)
  - κ>0 is needed for state-dependent uncertainty (S3)
  - κ=1.0 is overly conservative for all conditions

Uses PPO-RHOCBF (static GP) for clean κ isolation.

Usage:
  python experiments/phase5/run_kappa_sweep.py
  python experiments/phase5/run_kappa_sweep.py --conditions s3_coupled --kappas 0.0 0.3
  python experiments/phase5/run_kappa_sweep.py --conditions s2_pressure --seeds 0
"""

import argparse, json, os, sys, time
from pathlib import Path

import jax
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.phase5.run_experiment_5th import (
    run_single, load_config, save_result,
    CONDITION_LABELS,
)
from experiments.phase5.methods_5th import METHOD_LABELS

# ---- Configuration ----

DEFAULT_CONDITIONS = ['s3_weak', 's3_coupled', 's3_strong']
DEFAULT_KAPPAS = [0.0, 0.1, 0.3, 0.5, 1.0]
DEFAULT_SEEDS = [0, 1, 2]
RESULTS_SUBDIR = 'kappa_sweep'

CONDITION_LABELS_EXTRA = {
    's3_weak': 'S3: Weak (γ=0.5)',
    's3_coupled': 'S3: Medium (γ=1.0)',
    's3_strong': 'S3: Strong (γ=2.0)',
}


def run_kappa_sweep(conditions=None, kappas=None, seeds=None,
                    results_dir='results/phase5/'):
    """Run ε_κ sensitivity sweep across specified conditions and κ values."""
    conditions = conditions or DEFAULT_CONDITIONS
    kappas = kappas or DEFAULT_KAPPAS
    seeds = seeds or DEFAULT_SEEDS

    base_config = load_config()
    out_dir = Path(results_dir) / RESULTS_SUBDIR
    os.makedirs(out_dir, exist_ok=True)

    total = len(conditions) * len(kappas) * len(seeds)
    count = 0
    results_summary = {}

    print(f"=== Epsilon-Kappa Sensitivity Sweep ===")
    print(f"Conditions: {[CONDITION_LABELS.get(c,c) for c in conditions]}")
    print(f"κ values:   {kappas}")
    print(f"Seeds:      {seeds}")
    print(f"Total:      {total} experiments")
    print(f"Output:     {out_dir}")
    print(f"Method:     PPO-RHOCBF (static GP, variable ε_κ)")
    print(f"=" * 70)

    for condition in conditions:
        for kappa in kappas:
            for seed in seeds:
                count += 1

                # Unique filename for kappa-sweep results
                fname = f'kappa{kappa}_{condition}_seed{seed}.json'
                result_path = out_dir / fname

                # Skip if exists
                if result_path.exists():
                    print(f"\n[{count}/{total}] SKIP κ={kappa} | "
                          f"{CONDITION_LABELS.get(condition, condition)} | "
                          f"seed={seed} (exists)", flush=True)
                    continue

                print(f"\n[{count}/{total}] κ={kappa} | "
                      f"{CONDITION_LABELS.get(condition, condition)} | seed={seed}",
                      flush=True)

                # Deep copy config and inject epsilon_kappa
                import copy
                config = copy.deepcopy(base_config)
                if 'methods_config' not in config:
                    config['methods_config'] = {}
                if 'ppo_rhocbf' not in config['methods_config']:
                    config['methods_config']['ppo_rhocbf'] = {}
                config['methods_config']['ppo_rhocbf']['epsilon_kappa'] = kappa

                t0 = time.time()
                try:
                    result = run_single('ppo_rhocbf', condition, seed, config)

                    # Save with kappa-specific name
                    with open(result_path, 'w') as f:
                        json.dump(result, f)

                    vr = result.get('violation_rate', (float('nan'), float('nan')))
                    elapsed = time.time() - t0
                    print(f"  → violation_rate={vr[0]:.4f}±{vr[1]:.4f}, "
                          f"reward={result.get('cumulative_reward', (float('nan'),))[0]:.1f}, "
                          f"time={elapsed:.0f}s", flush=True)

                    key = f"{kappa}_{condition}"
                    if key not in results_summary:
                        results_summary[key] = []
                    results_summary[key].append(vr[0])

                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"  ✗ FAILED after {elapsed:.0f}s: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

    # Print summary
    print(f"\n=== Kappa Sweep Summary ===")
    for condition in conditions:
        print(f"\n  {CONDITION_LABELS.get(condition, condition)} "
              f"({CONDITION_LABELS_EXTRA.get(condition, '')}):")
        for kappa in kappas:
            key = f"{kappa}_{condition}"
            if key in results_summary:
                vals = results_summary[key]
                print(f"    κ={kappa:.1f}: mean={np.mean(vals)*100:.2f}% "
                      f"±{np.std(vals)*100:.2f}% (n={len(vals)})")
            else:
                print(f"    κ={kappa:.1f}: no results")

    print(f"\n=== Done: {count}/{total} experiments ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='ε_κ Sensitivity Sweep for RoCBF-Net')
    parser.add_argument('--conditions', nargs='*', default=None,
                        help=f'Conditions to sweep (default: {DEFAULT_CONDITIONS})')
    parser.add_argument('--kappas', type=float, nargs='*', default=None,
                        help=f'κ values to test (default: {DEFAULT_KAPPAS})')
    parser.add_argument('--seeds', type=int, nargs='*', default=None,
                        help=f'Seeds to run (default: {DEFAULT_SEEDS})')
    parser.add_argument('--results-dir', type=str, default='results/phase5/',
                        help='Base results directory')
    args = parser.parse_args()

    run_kappa_sweep(
        conditions=args.conditions,
        kappas=args.kappas,
        seeds=args.seeds,
        results_dir=args.results_dir)

"""Phase 4 rerun: RoCBF-Net (with fixed GP update + epsilon_floor) vs PPO-RHOCBF.

Runs on key conditions: s1_heat, s2_pressure, s3_coupled, s4_nonlinear, nominal.
Saves results to results/phase4_rerun/ with epsilon_log.
"""
import sys
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import json
import time
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from experiments.phase4.run_experiment import run_single, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True,
                        choices=['rocbf_net', 'ppo_rhocbf', 'both'])
    parser.add_argument('--condition', type=str, default='all',
                        choices=['all', 'nominal', 's1_heat', 's2_pressure',
                                 's3_coupled', 's4_nonlinear', 'load_following'])
    parser.add_argument('--seed', type=int, default=-1,
                        help='Single seed to run (-1 for all 0-4)')
    parser.add_argument('--output-dir', type=str,
                        default='results/phase4_rerun')
    args = parser.parse_args()

    config = load_config()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ['rocbf_net', 'ppo_rhocbf'] if args.method == 'both' else [args.method]
    conditions = (['nominal', 's1_heat', 's2_pressure', 's3_coupled', 's4_nonlinear']
                  if args.condition == 'all' else [args.condition])
    seeds = list(range(5)) if args.seed < 0 else [args.seed]

    print(f"Phase 4 Rerun: methods={methods}, conditions={conditions}, seeds={seeds}")
    print(f"Output: {output_dir}")
    print(f"Config: epsilon_floor={config['methods_config']['rocbf_net'].get('epsilon_floor', 0.0)}")
    print()

    for method_name in methods:
        for condition in conditions:
            for seed in seeds:
                result_file = output_dir / f"{method_name}_{condition}_seed{seed}.json"
                if result_file.exists():
                    print(f"  SKIP (exists): {result_file.name}")
                    continue

                print(f"\n{'='*60}")
                print(f"  {method_name} | {condition} | seed={seed}")
                print(f"{'='*60}")
                t0 = time.time()

                try:
                    result = run_single(method_name, condition, seed, config)

                    # Save result
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2, default=str)

                    elapsed = time.time() - t0
                    vr = result.get('violation_rate', [None, None])
                    eps_log = result.get('epsilon_log', [])
                    print(f"  DONE in {elapsed/60:.1f}min: violation={vr[0]*100:.2f}%±{vr[1]*100:.2f}%")
                    if eps_log:
                        print(f"  Epsilon: {eps_log[0]['epsilon_total']:.4f} → {eps_log[-1]['epsilon_total']:.4f}")
                except Exception as e:
                    print(f"  FAILED: {e}")
                    import traceback
                    traceback.print_exc()

    print("\nAll done!")


if __name__ == "__main__":
    main()

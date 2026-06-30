"""Focused GP ablation experiment: PPO-RHOCBF with scenario-specific GP + MC=True.

This produces the data for Table gp_ablation in the paper.
Compares:
  - PPO-RHOCBF + mixed GP + MC=False (existing phase4 data: 7.97%)
  - PPO-RHOCBF + scenario-specific GP + MC=True (this experiment)
  - RoCBF-Net + scenario-specific GP + MC=True + online (this experiment)

Usage:
    conda run -n jax_gpu python experiments/phase5/run_gp_ablation.py
"""
import sys
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import json
from pathlib import Path

from experiments.phase4.run_experiment import run_single, save_result, load_config


def main():
    config = load_config()

    results_dir = 'results/phase5_gp_ablation'
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # GP ablation: only s1_heat (the key scenario for the paper)
    condition = 's1_heat'
    seeds = [0, 1, 2]

    # Config 1: PPO-RHOCBF with scenario-specific GP + MC=True (no online updates)
    # Config 2: RoCBF-Net with scenario-specific GP + MC=True + online updates
    methods = ['ppo_rhocbf', 'rocbf_net']

    total = len(methods) * len(seeds)
    count = 0

    for method_name in methods:
        for seed in seeds:
            count += 1
            result_path = Path(results_dir) / f'{method_name}_{condition}_seed{seed}.json'
            if result_path.exists():
                print(f"[{count}/{total}] SKIP {method_name}|{condition}|seed={seed} (exists)")
                continue

            print(f"\n{'='*60}")
            print(f"  {method_name} | {condition} | seed={seed}")
            print(f"{'='*60}")

            try:
                result = run_single(method_name, condition, seed, config)
                save_result(result, method_name, condition, seed, results_dir)
                vr = result['violation_rate']
                print(f"  DONE: violation={vr[0]*100:.2f}%±{vr[1]*100:.2f}%")
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print("GP ABLATION RESULTS (S1: Heat)")
    print(f"{'='*60}")
    for method_name in methods:
        results = []
        for seed in seeds:
            path = Path(results_dir) / f'{method_name}_{condition}_seed{seed}.json'
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                    results.append(data['violation_rate'][0])
        if results:
            import numpy as np
            arr = np.array(results)
            print(f"  {method_name:15s}: {arr.mean()*100:.2f}%±{arr.std()*100:.2f}% (n={len(results)})")

    # Also print phase4 baseline for comparison
    print(f"\n  Phase4 baseline (mixed GP + MC=False):")
    baseline_dir = Path('results/phase4')
    baseline_results = []
    for seed in range(5):
        path = baseline_dir / f'ppo_rhocbf_{condition}_seed{seed}.json'
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                baseline_results.append(data['violation_rate'][0])
    if baseline_results:
        import numpy as np
        arr = np.array(baseline_results)
        print(f"  ppo_rhocbf    : {arr.mean()*100:.2f}%±{arr.std()*100:.2f}% (n={len(baseline_results)}, mixed GP+MC=False)")

    print("\nDone!")


if __name__ == "__main__":
    main()

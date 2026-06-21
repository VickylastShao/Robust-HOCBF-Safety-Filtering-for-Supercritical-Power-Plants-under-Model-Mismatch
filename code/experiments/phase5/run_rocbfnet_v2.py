"""Phase 5: Rerun RoCBF-Net with scenario-specific GP + MC=True.

Validates the corrected configuration before full experiment suite.
"""
import sys
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import json
from pathlib import Path

from experiments.phase4.run_experiment import run_single, save_result, load_config


def main():
    config = load_config()

    # Focus on RoCBF-Net and PPO-RHOCBF
    methods = ['rocbf_net', 'ppo_rhocbf']
    conditions = ['nominal', 's1_heat', 's2_pressure', 's3_coupled', 's4_nonlinear', 'load_following']
    seeds = [0, 1, 2]

    results_dir = 'results/phase5_rocbfnet_v2'
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    print(f"Phase 5 v2: methods={methods}, conditions={conditions}, seeds={seeds}")
    print(f"Output: {results_dir}")
    print(f"Config: use_mean_correction=True, scenario-specific GP\n")

    total = len(methods) * len(conditions) * len(seeds)
    count = 0

    for method_name in methods:
        for condition in conditions:
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
    print("SUMMARY")
    print(f"{'='*60}")
    for method_name in methods:
        for condition in conditions:
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
                print(f"  {method_name:15s} | {condition:15s}: "
                      f"{arr.mean()*100:.2f}%±{arr.std()*100:.2f}% "
                      f"(n={len(results)})")

    print("\nAll done!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run critical missing baseline experiments sequentially.

Priority order:
1. ppo_hocbf on mismatch scenarios (S2-S4) — show catastrophic failure
2. ppo_rhocbf on mismatch scenarios (S2-S4) + nominal — show robustness works
3. rocbf_net load_following seeds 1-4 — complete the dataset
"""
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

from experiments.phase4.run_experiment import run_single, save_result, load_config

# Experiments to run in order (method, condition, seed)
EXPERIMENTS = [
    # PPO-HOCBF on mismatch scenarios — catastrophic failure expected
    ("ppo_hocbf", "s2_pressure", 0),
    ("ppo_hocbf", "s3_coupled", 0),
    ("ppo_hocbf", "s4_nonlinear", 0),
    ("ppo_hocbf", "nominal", 0),
    ("ppo_hocbf", "s1_heat", 1),  # 2nd seed for S1

    # PPO-RHOCBF on mismatch scenarios — should show 0% CBF violation
    ("ppo_rhocbf", "s2_pressure", 0),
    ("ppo_rhocbf", "s3_coupled", 0),
    ("ppo_rhocbf", "s4_nonlinear", 0),
    ("ppo_rhocbf", "nominal", 0),
    ("ppo_rhocbf", "load_following", 0),

    # RoCBF-Net load_following completion
    ("rocbf_net", "load_following", 1),
    ("rocbf_net", "load_following", 2),
    ("rocbf_net", "load_following", 3),
    ("rocbf_net", "load_following", 4),
]


def main():
    results_dir = Path("results/phase4")
    config = load_config()

    total = len(EXPERIMENTS)
    completed = 0
    skipped = 0
    failed = 0

    for i, (method, condition, seed) in enumerate(EXPERIMENTS):
        result_path = results_dir / f"{method}_{condition}_seed{seed}.json"
        if result_path.exists():
            print(f"\n[{i+1}/{total}] SKIP {method} | {condition} | seed={seed} (exists)")
            skipped += 1
            continue

        print(f"\n[{i+1}/{total}] RUNNING {method} | {condition} | seed={seed}")
        t0 = time.time()
        try:
            result = run_single(method, condition, seed, config)
            save_result(result, method, condition, seed)
            elapsed = time.time() - t0

            cvr = result.get("cbf_violation_rate", [None, None])
            tvr = result["violation_rate"]
            reward = result["cumulative_reward"][0]
            print(f"  -> cbf_viol={cvr[0]*100:.2f}%, total_viol={tvr[0]*100:.2f}%, "
                  f"reward={reward:.1f}, time={elapsed/60:.1f}min")
            completed += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAILED: {e} ({elapsed/60:.1f}min)")
            failed += 1
            import traceback
            traceback.print_exc()

    print(f"\n=== Done: {completed} completed, {skipped} skipped, {failed} failed ===")


if __name__ == "__main__":
    main()

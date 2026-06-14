"""Run PPO-RHOCBF S2-S4 additional seeds (1-4) for M4 revision.

This addresses the reviewer concern that PPO-RHOCBF S2-S4 only had 1 seed.
"""
import sys
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

from experiments.phase4.run_experiment import run_single, save_result, load_config

CONDITIONS_SEEDS = {
    's2_pressure': [1, 2, 3, 4],
    's3_coupled': [1, 2, 3, 4],
    's4_nonlinear': [1, 2, 3, 4],
}

def main():
    config = load_config()
    total = sum(len(seeds) for seeds in CONDITIONS_SEEDS.values())
    count = 0

    for condition, seeds in CONDITIONS_SEEDS.items():
        for seed in seeds:
            count += 1
            print(f"\n[{count}/{total}] ppo_rhocbf | {condition} | seed={seed}", flush=True)
            try:
                result = run_single('ppo_rhocbf', condition, seed, config)
                save_result(result, 'ppo_rhocbf', condition, seed)
                vr = result['violation_rate']
                cvr = result['cbf_violation_rate']
                print(f"  → total_viol={vr[0]:.4f}±{vr[1]:.4f}, "
                      f"cbf_viol={cvr[0]:.4f}±{cvr[1]:.4f}", flush=True)
            except Exception as e:
                print(f"  ✗ FAILED: {e}", flush=True)
                import traceback
                traceback.print_exc()

    print(f"\n=== Done: {count}/{total} experiments completed ===", flush=True)

if __name__ == "__main__":
    main()

"""Run S1:Heat experiment with MC=True + k_h=3.0 for RoCBF-Net and PPO-RHOCBF."""
import sys
sys.path.insert(0, "/home/gpu/sz_workspace/RoCBF-Net")

import json
import time
import yaml
import os

from experiments.phase4.run_experiment import run_single, load_config

config = load_config('configs/phase4.yaml')

output_dir = 'results/phase5_mc_true'
os.makedirs(output_dir, exist_ok=True)

seed = 0
condition = 's1_heat'

for method in ['rocbf_net', 'ppo_rhocbf']:
    print(f"\n{'='*60}")
    print(f"Running {method} | {condition} | seed={seed}")
    print(f"Config: MC=True, k_h=3.0, scenario_specific_gp=True")
    print(f"{'='*60}")

    t0 = time.time()
    result = run_single(method, condition, seed, config)
    elapsed = time.time() - t0

    # Save results
    fname = f"{method}_{condition}_seed{seed}.json"
    with open(os.path.join(output_dir, fname), 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Violation rate: {result.get('violation_rate')}")
    print(f"Cumulative reward: {result.get('cumulative_reward')}")
    if 'eps_log' in result:
        print(f"Epsilon log ({len(result['eps_log'])} entries):")
        for entry in result['eps_log'][-3:]:
            print(f"  {entry}")

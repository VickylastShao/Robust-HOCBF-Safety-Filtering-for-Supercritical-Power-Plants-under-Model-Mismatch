#!/bin/bash
# Phase 4 full experiment runner: 8 methods × 6 conditions × 5 seeds
# Uses corrected k_h=[1.0] configuration
# Run from /home/gpu/sz_workspace/RoCBF-Net/

cd /home/gpu/sz_workspace/RoCBF-Net/

# Run methods in order of importance
# Priority 1: RoCBF-Net and PPO-RHOCBF (core comparison)
# Priority 2: Other CBF-based methods
# Priority 3: Baseline methods (PPO, PPO-Lagr, NMPC)

echo "Starting Phase 4 experiments with k_h=[1.0] fix"
echo "Start time: $(date)"

# Run all methods × all conditions × 5 seeds
# Using run_experiment.py which skips existing results
conda run -n jax_gpu python3 -c "
import sys
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')
from experiments.phase4.run_experiment import run_all
run_all(
    methods=['rocbf_net', 'ppo_rhocbf', 'ppo_gp_hocbf', 'ppo_hocbf',
             'ppo_cbf', 'ppo_lagr', 'ppo', 'nmpc'],
    conditions=['s1_heat', 's2_pressure', 's3_coupled', 's4_nonlinear',
                'nominal', 'load_following'],
    seeds=[0, 1, 2, 3, 4],
    results_dir='results/phase4/'
)
"

echo "End time: $(date)"
echo "Phase 4 experiments complete!"

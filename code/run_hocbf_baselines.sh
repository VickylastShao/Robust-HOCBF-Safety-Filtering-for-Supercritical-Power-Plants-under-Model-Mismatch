#!/bin/bash
# Run HOCBF-based baseline experiments with k_h=[1.0]
# Methods: ppo_hocbf, ppo_gp_hocbf, ppo_rhocbf
# Scenarios: nominal, s1_heat, s2_pressure, s3_coupled, s4_nonlinear, load_following
# Seeds: 0-4

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate jax_gpu

cd /home/gpu/sz_workspace/RoCBF-Net

METHODS="ppo_rhocbf ppo_hocbf ppo_gp_hocbf"
CONDITIONS="nominal s1_heat s2_pressure s3_coupled s4_nonlinear load_following"

python3 -c "
import sys
sys.path.insert(0, '.')
from experiments.phase4.run_experiment import run_all
import yaml

config = yaml.safe_load(open('configs/phase4.yaml'))
# Override to only run HOCBF methods
config['methods'] = ['ppo_rhocbf', 'ppo_hocbf', 'ppo_gp_hocbf']
config['seeds'] = 5
config['conditions'] = ['nominal', 's1_heat', 's2_pressure', 's3_coupled', 's4_nonlinear', 'load_following']

run_all(config=config, results_dir='results/phase4/')
"

#!/bin/bash
# Full 5th-order fair comparison sweep
# 8 methods × 8 conditions × 5 seeds = 320 experiments
# Log: experiments/phase5/sweep_5th.log

set -e
cd /mnt/c/Users/vicks/codex-home/RoCBF-Net
LOGFILE="experiments/phase5/sweep_5th.log"
echo "=== Phase 5 Full Sweep Started: $(date) ===" | tee -a "$LOGFILE"
echo "Methods: ppo, ppo_lagr, nmpc, ppo_cbf, ppo_hocbf, ppo_gp_hocbf, ppo_rhocbf, rocbf_net" | tee -a "$LOGFILE"
echo "Conditions: nominal, s1_heat, s2_pressure, s3_coupled, s4_nonlinear, s5_valve, s6_fuel, load_following" | tee -a "$LOGFILE"
echo "Seeds: 0-4 (5 per combo)" | tee -a "$LOGFILE"
echo "Total: 320 experiments" | tee -a "$LOGFILE"

XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
  python experiments/phase5/run_experiment_5th.py \
  --results-dir results/phase5/ \
  2>&1 | tee -a "$LOGFILE"

echo "=== Phase 5 Full Sweep Completed: $(date) ===" | tee -a "$LOGFILE"

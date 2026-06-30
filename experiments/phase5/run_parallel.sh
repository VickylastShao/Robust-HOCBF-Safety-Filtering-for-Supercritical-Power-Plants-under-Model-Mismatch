#!/bin/bash
# Phase 5 Parallel Runner — 3 independent processes
# Each process runs a subset of methods on ALL conditions × ALL seeds
# GPU target: LAN RTX 4090 servers under /home/gpu/sz_workspace/RoCBF-Net
# Each process uses a capped JAX memory fraction (MEM_FRACTION=0.25)
#
# Watch progress:  tail -f experiments/phase5/parallel_*.log
# Count completed: ls results/phase5/*.json | wc -l

set -e
cd /home/gpu/sz_workspace/RoCBF-Net
PYTHON="${PYTHON:-/home/gpu/sz_workspace/RoCBF-Net/.venv/bin/python}"
RESULTS_DIR="results/phase5"
LOG_DIR="experiments/phase5"

echo "=== Phase 5 Parallel Sweep Started: $(date) ==="
echo "Group 1 (light):  ppo ppo_lagr nmpc          — est. 44h"
echo "Group 2 (medium): ppo_cbf ppo_hocbf ppo_gp_hocbf — est. 61h"
echo "Group 3 (heavy):  ppo_rhocbf rocbf_net        — est. 50h"
echo "Wall time: ~2.5 days (vs ~7 days serial)"
echo ""

# Group 1: PPO (remaining) + PPO-Lagrangian + NMPC
nohup env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
  "$PYTHON" experiments/phase5/run_experiment_5th.py \
  --methods ppo ppo_lagr nmpc \
  --results-dir "$RESULTS_DIR" \
  > "$LOG_DIR/parallel_group1.log" 2>&1 &
PID1=$!
echo "Group 1 PID: $PID1"

# Group 2: PPO-CBF + PPO-HOCBF + PPO-GP-HOCBF
nohup env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
  "$PYTHON" experiments/phase5/run_experiment_5th.py \
  --methods ppo_cbf ppo_hocbf ppo_gp_hocbf \
  --results-dir "$RESULTS_DIR" \
  > "$LOG_DIR/parallel_group2.log" 2>&1 &
PID2=$!
echo "Group 2 PID: $PID2"

# Group 3: PPO-RHOCBF + RoCBF-Net
nohup env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
  "$PYTHON" experiments/phase5/run_experiment_5th.py \
  --methods ppo_rhocbf rocbf_net \
  --results-dir "$RESULTS_DIR" \
  > "$LOG_DIR/parallel_group3.log" 2>&1 &
PID3=$!
echo "Group 3 PID: $PID3"

echo ""
echo "All 3 groups launched. Monitor:"
echo "  tail -f $LOG_DIR/parallel_group1.log"
echo "  tail -f $LOG_DIR/parallel_group2.log"
echo "  tail -f $LOG_DIR/parallel_group3.log"
echo ""
echo "Kill all:  kill $PID1 $PID2 $PID3"
echo "=== Launched: $(date) ==="

# Wait for all to complete (optional — comment out to detach)
# wait $PID1 $PID2 $PID3
# echo "=== All groups completed: $(date) ==="

#!/bin/bash
# Phase 5 Kappa Sensitivity Sweep — 3-way parallel
# Conditions: s2_pressure s3_coupled s4_nonlinear
# Kappas: 0.0 0.1 0.3 0.5 1.0
# Seeds: 0 1 2
# Total: 3 × 5 × 3 = 45 experiments, ~12h wall time
#
# Uses python -u + PYTHONUNBUFFERED=1 to prevent log buffering issues.

set -e
cd /mnt/c/Users/vicks/codex-home/RoCBF-Net
LOG_DIR="experiments/phase5"
RESULTS_DIR="results/phase5"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
export PYTHONUNBUFFERED=1

echo "=== Kappa Sweep Started: $(date) ==="
echo "Sweeping ε_κ ∈ {0.0, 0.1, 0.3, 0.5, 1.0} on S2, S3, S4"
echo ""

NO_FIFO=""
echo "Testing if unbuffer is available..."
if command -v unbuffer &> /dev/null; then
  echo "  unbuffer available, using it for log output"
  NO_FIFO="unbuffer"
elif command -v stdbuf &> /dev/null; then
  echo "  stdbuf available, using it for log output"
  NO_FIFO="stdbuf -oL -eL"
else
  echo "  using python -u (should suffice)"
fi

# Group 1: S2 (Pressure) — all 5 kappas × 3 seeds = 15 experiments
if [ -n "$NO_FIFO" ] && [ "$NO_FIFO" != "unbuffer" ]; then
  nohup env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 PYTHONUNBUFFERED=1 \
    $NO_FIFO python -u experiments/phase5/run_kappa_sweep.py \
    --conditions s2_pressure \
    --kappas 0.0 0.1 0.3 0.5 1.0 \
    --seeds 0 1 2 \
    --results-dir "$RESULTS_DIR" \
    > "$LOG_DIR/kappa_group1.log" 2>&1 &
else
  nohup env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 PYTHONUNBUFFERED=1 \
    python -u experiments/phase5/run_kappa_sweep.py \
    --conditions s2_pressure \
    --kappas 0.0 0.1 0.3 0.5 1.0 \
    --seeds 0 1 2 \
    --results-dir "$RESULTS_DIR" \
    > "$LOG_DIR/kappa_group1.log" 2>&1 &
fi
PID1=$!
echo "Group 1 (S2:Pressure) PID: $PID1"

# Group 2: S3 (Coupled) — all 5 kappas × 3 seeds = 15 experiments
if [ -n "$NO_FIFO" ] && [ "$NO_FIFO" != "unbuffer" ]; then
  nohup env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 PYTHONUNBUFFERED=1 \
    $NO_FIFO python -u experiments/phase5/run_kappa_sweep.py \
    --conditions s3_coupled \
    --kappas 0.0 0.1 0.3 0.5 1.0 \
    --seeds 0 1 2 \
    --results-dir "$RESULTS_DIR" \
    > "$LOG_DIR/kappa_group2.log" 2>&1 &
else
  nohup env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 PYTHONUNBUFFERED=1 \
    python -u experiments/phase5/run_kappa_sweep.py \
    --conditions s3_coupled \
    --kappas 0.0 0.1 0.3 0.5 1.0 \
    --seeds 0 1 2 \
    --results-dir "$RESULTS_DIR" \
    > "$LOG_DIR/kappa_group2.log" 2>&1 &
fi
PID2=$!
echo "Group 2 (S3:Coupled) PID: $PID2"

# Group 3: S4 (Nonlinear) — all 5 kappas × 3 seeds = 15 experiments
if [ -n "$NO_FIFO" ] && [ "$NO_FIFO" != "unbuffer" ]; then
  nohup env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 PYTHONUNBUFFERED=1 \
    $NO_FIFO python -u experiments/phase5/run_kappa_sweep.py \
    --conditions s4_nonlinear \
    --kappas 0.0 0.1 0.3 0.5 1.0 \
    --seeds 0 1 2 \
    --results-dir "$RESULTS_DIR" \
    > "$LOG_DIR/kappa_group3.log" 2>&1 &
else
  nohup env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 PYTHONUNBUFFERED=1 \
    python -u experiments/phase5/run_kappa_sweep.py \
    --conditions s4_nonlinear \
    --kappas 0.0 0.1 0.3 0.5 1.0 \
    --seeds 0 1 2 \
    --results-dir "$RESULTS_DIR" \
    > "$LOG_DIR/kappa_group3.log" 2>&1 &
fi
PID3=$!
echo "Group 3 (S4:Nonlinear) PID: $PID3"

echo ""
echo "Monitor:"
echo "  tail -f $LOG_DIR/kappa_group1.log"
echo "  tail -f $LOG_DIR/kappa_group2.log"
echo "  tail -f $LOG_DIR/kappa_group3.log"
echo ""
echo "Kill all: kill $PID1 $PID2 $PID3"
echo "=== Launched: $(date) ==="

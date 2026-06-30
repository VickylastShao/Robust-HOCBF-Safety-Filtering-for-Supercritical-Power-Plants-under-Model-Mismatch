#!/bin/bash
# Serial runner for remaining kappa sweep experiments
# Run after K1 (S2) finishes, or alongside it with MEM_FRACTION=0.50
#
# Strategy: Run S3 then S4 serially, one at a time, full GPU access.
# Each experiment ~40 min × 15 = ~10h per condition.
# Total: ~20h for both.

set -e
cd /mnt/c/Users/vicks/codex-home/RoCBF-Net
LOG_DIR="experiments/phase5"
RESULTS_DIR="results/phase5"
CONDA_ENV="jax_gpu"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
export PYTHONUNBUFFERED=1

echo "=== Kappa Serial Runner (S2→S3→S4): $(date) ==="
echo "Strategy: Run one condition at a time, full GPU speed (~40 min/exp)"
echo "Remaining: ~30 experiments × 40 min ≈ 20 hours"
echo ""

for COND in s2_pressure s3_coupled s4_nonlinear; do
    case $COND in
        s2_pressure) LABEL="S2: Pressure (additive)" ;;
        s3_coupled)  LABEL="S3: Coupled (state-dependent)" ;;
        s4_nonlinear) LABEL="S4: Nonlinear (additive)" ;;
    esac

    echo "============================================================"
    echo "--- $LABEL ---"
    echo "============================================================"
    echo ""

    python -u experiments/phase5/run_kappa_sweep.py \
      --conditions $COND \
      --kappas 0.0 0.1 0.3 0.5 1.0 \
      --seeds 0 1 2 \
      --results-dir "$RESULTS_DIR" \
      2>&1 | tee "$LOG_DIR/kappa_${COND}_serial.log"

    echo ""
    echo "--- $LABEL Done: $(date) ---"
    echo ""
done

echo "=== All kappa sweep done: $(date) ==="

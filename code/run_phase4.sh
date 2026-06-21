#!/bin/bash
# Phase 4 Full Experiments: 8 methods × 6 conditions × 5 seeds
# Run sequentially by method for monitoring
# Estimated total: ~16 hours

cd /home/gpu/sz_workspace/RoCBF-Net
export PYTHONPATH=/home/gpu/sz_workspace/RoCBF-Net:$PYTHONPATH

LOG_DIR="results/phase4/logs"
mkdir -p "$LOG_DIR"

echo "=== Phase 4 Experiments Started: $(date) ==="

METHODS=("nmpc" "ppo" "ppo_lagr" "ppo_cbf" "ppo_hocbf" "ppo_gp_hocbf" "ppo_rhocbf" "rocbf_net")

for METHOD in "${METHODS[@]}"; do
    echo ""
    echo "=== Starting method: $METHOD at $(date) ==="
    /home/gpu/miniconda3/envs/jax_gpu/bin/python -c "
from experiments.phase4.run_experiment import run_all, load_config
config = load_config()
run_all(methods=['$METHOD'], conditions=None, seeds=None)
" 2>&1 | tee "$LOG_DIR/${METHOD}_$(date +%Y%m%d_%H%M%S).log"
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "=== WARNING: Method $METHOD exited with code $EXIT_CODE ==="
    fi
    echo "=== Finished method: $METHOD at $(date) ==="

    COUNT=$(ls results/phase4/*.json 2>/dev/null | wc -l)
    echo "=== Total results so far: $COUNT ==="
done

echo ""
echo "=== Phase 4 All Experiments Completed: $(date) ==="
COUNT=$(ls results/phase4/*.json 2>/dev/null | wc -l)
echo "=== Total result files: $COUNT / 240 expected ==="

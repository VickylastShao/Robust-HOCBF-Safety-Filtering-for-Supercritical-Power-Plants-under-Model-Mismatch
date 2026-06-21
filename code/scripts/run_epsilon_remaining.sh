#!/bin/bash
# Run remaining epsilon ablation experiments for scenario_specific GP
# Sequential execution to avoid GPU OOM

cd /home/gpu/sz_workspace/RoCBF-Net

GP_TYPE="scenario_specific"
SEEDS=(0 1 2 3 4)
MODES=("no_epsilon" "constant_mean" "constant_max" "compositional")

# Track which experiments are already done in JSON
JSON_FILE="results/phase5/epsilon_ablation/epsilon_ablation.json"

for SEED in "${SEEDS[@]}"; do
    for MODE in "${MODES[@]}"; do
        # Check if already completed in JSON
        if python3 -c "
import json, sys
try:
    d = json.load(open('${JSON_FILE}'))
    # Check new format
    gp_key = 'gp_${GP_TYPE}'
    seed_key = 'seed_${SEED}'
    if gp_key in d and seed_key in d[gp_key] and '${MODE}' in d[gp_key][seed_key]:
        sys.exit(0)  # already done
    # Check old format (seed_0 only)
    if 'seed_${SEED}' in d and '${MODE}' in d['seed_${SEED}']:
        sys.exit(0)  # already done (old format)
    sys.exit(1)  # not done
except:
    sys.exit(1)  # not done
" 2>/dev/null; then
            echo "=== SKIP: already done: gp=$GP_TYPE mode=$MODE seed=$SEED ==="
            continue
        fi
        
        echo "=== Running: gp_type=$GP_TYPE mode=$MODE seed=$SEED ==="
        conda run -n jax_gpu python experiments/phase5/epsilon_ablation.py \
            --gp_type "$GP_TYPE" \
            --mode "$MODE" \
            --seed "$SEED" \
            --n_eval_episodes 50 \
            --n_eval_steps 500 \
            --n_episodes 50 \
            --n_steps 200 \
            2>&1 | tail -3
        echo "--- Done: gp_type=$GP_TYPE mode=$MODE seed=$SEED ---"
        
        # Brief pause between runs
        sleep 5
    done
done

echo "=== All remaining experiments completed ==="

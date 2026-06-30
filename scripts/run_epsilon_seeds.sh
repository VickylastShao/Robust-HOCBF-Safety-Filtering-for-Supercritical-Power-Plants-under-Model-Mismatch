#!/bin/bash
# Run epsilon ablation seeds 1-4 for scenario_specific GP
# All 4 epsilon modes: compositional, constant_mean, constant_max, no_epsilon
# Uses --seed flag for single-config mode

cd /home/gpu/sz_workspace/RoCBF-Net

GP_TYPE="scenario_specific"
MODES=("compositional" "constant_mean" "constant_max" "no_epsilon")
SEEDS=(1 2 3 4)

for SEED in "${SEEDS[@]}"; do
    for MODE in "${MODES[@]}"; do
        echo "=== Running: gp_type=$GP_TYPE mode=$MODE seed=$SEED ==="
        conda run -n jax_gpu python experiments/phase5/epsilon_ablation.py \
            --gp_type "$GP_TYPE" \
            --mode "$MODE" \
            --seed "$SEED" \
            --n_eval_episodes 50 \
            --n_eval_steps 500 \
            --n_episodes 50 \
            --n_steps 200 \
            2>&1 | tail -5
        echo "--- Done: gp_type=$GP_TYPE mode=$MODE seed=$SEED ---"
    done
done

echo "=== All seeds completed ==="

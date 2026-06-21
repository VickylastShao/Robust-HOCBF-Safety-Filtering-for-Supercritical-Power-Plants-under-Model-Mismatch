#!/bin/bash
# Phase 2 experiments driver — uses existing run_5th_phi_scaled.py infrastructure.
# Run from repo root after: conda activate jax_gpu

set -e
REPO=/home/gpu/sz_workspace/RoCBF-Net
cd $REPO

OUTDIR=$REPO/results/phase5/revision
mkdir -p $OUTDIR

# ============================================================================
# Experiment 1: σ_floor sensitivity sweep (σ ∈ {0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2})
# ============================================================================
echo "=== E1: σ_floor sensitivity sweep ==="
# Run the existing epsilon analysis to get epsilon values with default sigma_floor
# For now, we verify the existing results are correct and compute additional
# sigma_floor analysis offline from the GP posterior.
#
# The key insight: sigma_floor can be varied in post-processing of the GP
# posterior, since ε = β * σ_GP * ψ_chain_gradient. We compute this from
# the existing GP model without re-running full experiments.
echo "E1: Using existing GP posterior for sigma_floor analysis (see analyze_sigma_floor.py)"

# ============================================================================
# Experiment 2: Moderate perturbation (selective QP intervention demo)
# ============================================================================
echo "=== E2: Moderate perturbation scenario ==="
python experiments/phase5/run_5th_phi_scaled.py \
    --methods ppo_rhocbf \
    --conditions moderate_heat \
    --seeds 0 1 2 3 4 \
    --n-eval 500 \
    --output $OUTDIR/moderate_perturbation.json \
    2>&1 | tee $OUTDIR/moderate_perturbation.log

# ============================================================================
# Experiment 3: Perturbation magnitude sweep
# ============================================================================
echo "=== E3: Perturbation magnitude sweep ==="
for mag in mag10 mag25 mag50 mag75 mag100; do
    echo "  Magnitude: $mag"
    python experiments/phase5/run_5th_phi_scaled.py \
        --methods ppo_rhocbf \
        --conditions $mag \
        --seeds 0 1 2 \
        --n-eval 500 \
        --output $OUTDIR/magnitude_${mag}.json \
        2>&1 | tail -5
done

# ============================================================================
# Experiment 4: Longer evaluation episodes (2000 steps)
# ============================================================================
echo "=== E4: Longer evaluation episodes (2000 steps) ==="
for cond in S1 S3; do
    echo "  Condition: $cond, n_eval=2000"
    python experiments/phase5/run_5th_phi_scaled.py \
        --methods ppo_rhocbf \
        --conditions $cond \
        --seeds 0 1 2 \
        --n-eval 2000 \
        --output $OUTDIR/long_eval_${cond}.json \
        2>&1 | tail -5
done

echo "=== ALL DONE ==="
ls -lh $OUTDIR/*.json
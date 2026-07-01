# Phase 5: 5th-Order CCS Fair Comparison — Complete Asset Inventory

> Historical note (2026-07-01): this file records an intermediate Phase 5 asset snapshot and is kept as process evidence. For the current M&C submission artifact, use `README.md`, `REPRODUCIBILITY.md`, `ARTIFACT_MANIFEST.md`, and `scripts/check_repro_artifacts.py` as the authoritative repository entry points. Some details below, including target-venue language and hardware notes, are superseded.

**Generated:** 2026-06-28
**Status:** Complete (361/365 experiments, 98.9% success rate)
**Purpose:** Fair comparison of all 8 methods on 5th-order CCS dynamics for IEEE TAC manuscript revision

---

## 1. Experimental Scale

| Batch | Scope | Successful | Failed | Time |
|-------|-------|-----------|--------|------|
| Main sweep | 8 methods × 8 conditions × 5 seeds = 320 | 319 | 1 (RoCBF-Net S1 seed1: GP Cholesky) | ~115h (3-way parallel, then serial) |
| κ sensitivity sweep | 3 conditions × 5 κ values × 3 seeds = 45 | 42 | 3 (all seed=2 hangs: S3 κ=1.0, S4 κ=0.5, S4 κ=1.0) | ~80h (serial, GPU contention) |
| **Total** | **365** | **361** | **4** | **~8 days wall time** |

### Hardware
- **GPU:** NVIDIA RTX 3080 Ti (12GB VRAM), WSL2 passthrough
- **CPU:** AMD Ryzen (29GB RAM)
- **JAX config:** `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.25`

---

## 2. Source Files Created/Modified

### New Files
| File | Purpose | Lines |
|------|---------|-------|
| `experiments/phase5/methods_5th.py` | 5th-order method factory (8 methods) | ~520 |
| `experiments/phase5/run_experiment_5th.py` | Main experiment runner + CLI | ~688 |
| `experiments/phase5/run_kappa_sweep.py` | ε_κ sensitivity sweep runner | ~140 |
| `experiments/phase5/plot_kappa_sweep.py` | κ-sweep plotting + LaTeX table generator | ~350 |
| `experiments/phase5/analyze_results_5th.py` | Full analysis: violation tables, reward tables, per-constraint breakdown, LaTeX, plots | ~350 |
| `experiments/phase5/run_parallel.sh` | 3-way parallel launcher for main sweep | ~60 |
| `experiments/phase5/run_kappa_parallel.sh` | 3-way parallel launcher for κ sweep | ~80 |
| `experiments/phase5/run_kappa_serial.sh` | Serial launcher for κ sweep (S2→S3→S4) | ~50 |
| `configs/phase5.yaml` | Phase 5 experiment configuration | ~92 |

### Modified Files
| File | Changes |
|------|---------|
| `paper/sections_mc/experimental.tex` | Table 1 replaced with 8-method × 7-condition real data; narrative rewritten with 5 findings; new §2.4 κ sensitivity section added |
| `rocbf/cbf/multi_hocbf.py` | (Indirectly verified — 5th-order compatible, no changes needed) |

### Key Technical Fixes
- **`x0` parameter:** `_make_robust_hocbf_5th` now passes `x0` to all 6 `RobustHOCBF` constructors, fixing `op_norm_estimate` warnings
- **JIT compilation:** `jax.jit(multi_hocbf.qp_matrices)` achieves 2570× speedup (1103ms → 0.4ms per call)
- **5th-order state handling:** `step_stabilized(x[:5])` (was `x[:3]`); Actor `n_obs=5` (was `n_obs=3`); LQR gain K shape (3,5) (was (3,3))
- **Power constraints:** 5th-order enables power CBF at m=1 (was m=0 in 3rd-order, not CBF-enforceable)
- **GP dimension:** `GPResidual(n_dims=3)` on core states (r_B, p_m, h_m); extended states (N_e, τ_f) get zero-padded σ

---

## 3. Methods Compared (8 Methods)

| # | Method Key | Label | Safety Layer | GP | ε_κ | Training |
|---|-----------|-------|--------------|----|-----|----------|
| 1 | `ppo` | PPO (no safety) | None | No | — | PPO |
| 2 | `ppo_lagr` | PPO-Lagrangian | Dual descent penalty | No | — | PPO + Lagrangian |
| 3 | `nmpc` | NMPC ($N{=}5$, SLSQP) | Horizon optimization | No | — | None (optimization) |
| 4 | `ppo_cbf` | PPO-CBF (1st-order) | CBF, $m{=}1$ for all | No | — | PPO (no CBF in training) |
| 5 | `ppo_hocbf` | PPO-HOCBF ($m{=}2,1,1$) | HOCBF, correct m | No | — | PPO (no CBF in training) |
| 6 | `ppo_gp_hocbf` | PPO-GP-HOCBF ($\epsilon_\kappa{=}0$) | HOCBF + GP mean correction | Static GP | 0.0 | PPO (no CBF in training) |
| 7 | `ppo_rhocbf` | PPO-RHOCBF ($\epsilon_\kappa{=}1.0$) | HOCBF + GP + compositional ε | Static scenario GP | 1.0 | PPO (no CBF in training) |
| 8 | `rocbf_net` | **RoCBF-Net (Ours)** | HOCBF + GP + ε + online adaptation | Online GP | 1.0 | PPO + incremental GP update |

### Critical Design Note
All PPO-based methods use `_rollout_no_qp_5th` during training — the Actor learns **without** the safety filter. The CBF/QP safety layer is applied **only during evaluation**. This creates a training-evaluation gap: the Actor never learns to produce actions that are safe under the QP projection.

---

## 4. Conditions (8 Scenarios)

| Condition | Label | Uncertainty Type | Dynamics Model | Notes |
|-----------|-------|-----------------|----------------|-------|
| `nominal` | Nominal | None | 5th-order nominal | Baseline |
| `s1_heat` | S1: Heat | Additive (constant Δf) | 5th-order + S1 | Coal quality drop |
| `s2_pressure` | S2: Pressure | Additive (oscillation) | 5th-order + S2 | Feedwater pump cycling |
| `s3_coupled` | S3: Coupled | **State-dependent** ($\delta \propto x$) | 5th-order + S3 | Positive feedback |
| `s4_nonlinear` | S4: Nonlinear | Additive (quadratic) | 5th-order + S4 | Heat exchanger degradation |
| `s5_valve` | S5: Valve | Additive (valve degradation) | 5th-order + S5 | **5th-order only** (affects N_e, τ_f) |
| `s6_fuel` | S6: Fuel | Additive (fuel quality) | 5th-order + S6 | **5th-order only** |
| `load_following` | LoadFol | None (AGC ramping) | 5th-order nominal | 750-1000MW AGC schedule |

### Constraints (6 CBFs)
- **Pressure high** ($m{=}2$): $p_m \leq 24$ MPa
- **Pressure low** ($m{=}2$): $p_m \geq 13$ MPa
- **Enthalpy high** ($m{=}1$): $h_m \leq 2830$ kJ/kg
- **Enthalpy low** ($m{=}1$): $h_m \geq 2670$ kJ/kg
- **Power high** ($m{=}1$): $N_e \leq P_{\text{ref}} + 50$ MW
- **Power low** ($m{=}1$): $N_e \geq P_{\text{ref}} - 50$ MW

---

## 5. Key Experimental Results

### 5.1 Main Violation Table (%)

| Method | Nom. | S1 | S2 | S3 | S4 | S5 | S6 | LoadFol |
|--------|------|-----|-----|-----|-----|-----|-----|-----|
| PPO (no safety) | 0.08 | 99.92 | 99.53 | 99.59 | 99.05 | 99.12 | 99.83 | 0.08 |
| PPO-Lagrangian | 0.08 | 99.92 | 99.53 | 99.59 | 99.05 | 99.12 | 99.83 | 0.08 |
| PPO-CBF (1st-order) | 0.08 | 99.92 | 99.53 | 99.59 | 99.05 | 99.12 | 99.83 | 0.08 |
| PPO-HOCBF (correct m) | 0.00 | 99.92 | 99.53 | 99.59 | 99.05 | 99.12 | 99.83 | 0.00 |
| PPO-GP-HOCBF ($\epsilon_\kappa{=}0$) | 0.00 | 0.12 | 0.04 | 39.65 | 0.04 | 0.15 | 0.18 | 0.00 |
| PPO-RHOCBF ($\epsilon_\kappa{=}1.0$) | 0.00 | 1.38 | 99.43 | 16.96 | 32.19 | 0.00 | 0.00 | 0.00 |
| **RoCBF-Net ($\epsilon_\kappa{=}1.0$)** | 0.00 | 18.78 | 99.40 | 48.58 | 49.16 | 0.00 | 0.00 | 0.00 |
| NMPC ($N{=}5$, SLSQP) | 0.00 | 0.08 | 0.08 | 0.08 | 0.04 | 0.08 | 0.08 | 0.00 |

### 5.2 ε_κ Sensitivity Sweep

| κ | S2 (Additive) | S3 (State-Dependent) | S4 (Additive) |
|-----|---------------|----------------------|---------------|
| 0.0 | 0.00% ✅ | 33.47% ❌ | 0.00% ✅ |
| 0.1 | 0.00% ✅ | 0.00% ✅ | 0.00% ✅ |
| 0.3 | 0.00% ✅ | 0.10% ✅ | 0.00% ✅ |
| 0.5 | 99.36% ❌ | 0.25% ✅ | 0.10% ✅ |
| 1.0 | 99.42% ❌ | 40.30% ❌ | 46.83% ❌ |

**Optimal κ:**
- S2 (additive): κ* = 0.0 (κ ≤ 0.3 safe)
- S3 (state-dependent): κ* = 0.1 (κ ∈ [0.1, 0.5] safe)
- S4 (additive): κ* = 0.0 (κ ≤ 0.5 safe, unlike S2)

### 5.3 Per-Constraint Decomposition
- **Pressure (m=2):** 0.00% violation across ALL methods and ALL conditions
- **Power (m=1):** 0.00% violation across ALL methods and ALL conditions
- **Enthalpy (m=1):** Accounts for 100% of all observed violations — the sole safety bottleneck

### 5.4 Reward Comparison
- NMPC achieves the best reward (0 to -887, depending on condition)
- PPO-GP-HOCBF ($\epsilon_\kappa{=}0$) achieves better reward than PPO-RHOCBF ($\epsilon_\kappa{=}1.0$) on all conditions
- The safety-performance trade-off is real: κ=0.3 gives better reward than κ=0 on S2 (-15167 vs -15484)

### 5.5 Convergence
- All methods converge at exactly 50 episodes (minimum training episodes)
- `min_episodes=50`, `convergence_threshold=0.05`, `convergence_window=30`
- Training is reward-only; CBF is not involved during training

---

## 6. Core Findings (Paper Narrative)

### Finding 1: Safety layers alone fail under uncertainty
PPO, PPO-Lagrangian, and PPO-CBF all achieve identical 99%+ violation on any uncertainty condition. Soft constraints (Lagrangian) and incorrect relative degree (1st-order CBF) offer zero protection.

### Finding 2: GP mean correction is the dominant safety mechanism
PPO-GP-HOCBF ($\epsilon_\kappa{=}0$) reduces violation by **>500×** (99.9% → ≤0.18%) on 6 of 7 uncertainty conditions — without any robustness margin. GP residual modeling, not the compositional ε, is the primary empirical safety driver.

### Finding 3: HOCBF with correct relative degrees is necessary but insufficient
PPO-HOCBF (correct $m{=}2,1,1$, no GP) achieves 0% on nominal but still 99%+ under uncertainty. Mathematical structure alone cannot compensate for model error.

### Finding 4: ε_κ=1.0 is overly conservative
The theoretical worst-case bound overly constrains the QP, forcing constraint-dropping fallback and paradoxically increasing violations. PPO-RHOCBF ($\epsilon_\kappa{=}1.0$) performs WORSE than PPO-GP-HOCBF ($\epsilon_\kappa{=}0$) on S2 (99.43% vs 0.04%) and S4 (32.19% vs 0.04%).

### Finding 5: Different scenarios need different ε_κ
- **Additive uncertainty:** $\epsilon_\kappa \in [0, 0.3]$ — GP mean correction alone suffices
- **State-dependent uncertainty:** $\epsilon_\kappa \in [0.1, 0.5]$ — margin is necessary (κ=0 fails at 33.5%) but should not be maximal
- **κ=1.0 is never optimal** in practice

### Finding 6: NMPC is the optimal benchmark
NMPC achieves ≤0.08% violation across ALL conditions — the gold standard. Learning-based methods currently underperform due to the training-evaluation gap (CBF not used during PPO training).

### Finding 7: Enthalpy is the sole bottleneck
All violations are enthalpy violations. Pressure constraints (m=2) are never violated. This reflects both tighter enthalpy bounds and the direct impact of heat-absorption uncertainty.

---

## 7. Paper Narrative Structure

```
Layer 1: PPO (no safety)              → 99%+ violation on uncertainty
Layer 2: + Lagrangian (soft)          → 99%+ (no improvement)
Layer 3: + 1st-order CBF (wrong m)    → 99%+ (no improvement)
Layer 4: + HOCBF (correct m, no GP)   → 99%+ (necessary but insufficient)
    ═══════════════════════════════════  BREAK LINE ═══════════════════
Layer 5: + GP mean correction (ε_κ=0) → 0.04-0.18% on 6/7 conditions  ← MAIN CONTRIBUTION
Layer 6: + Compositional ε (ε_κ=1.0)  → Degrades on additive conditions
Layer 7: RoCBF-Net (online GP + ε)    → Architecture; needs κ tuning
```

**Key insight:** GP > ε. The framework RoCBF-Net(κ) is the architectural contribution; the κ-sensitivity data shows that κ must be tuned per uncertainty type, not fixed at the theoretical worst-case bound.

---

## 8. Figures Generated

| Figure | Path | Description |
|--------|------|-------------|
| Violation bar chart | `results/phase5/figures/violation_bar_5th.png` | 8 methods × 8 conditions grouped bars |
| Violation heatmap | `results/phase5/figures/violation_heatmap_5th.png` | Color-coded violation matrix |
| κ sensitivity curves | `results/phase5/figures/kappa_sensitivity.png` | 3-condition κ sweep with optimal annotations |
| κ sensitivity (paper) | `paper/figures/kappa_sensitivity.png` | Copy for LaTeX inclusion |

---

## 9. LaTeX Assets

| Asset | Location | Status |
|-------|----------|--------|
| Updated Table 1 (violation) | `paper/sections_mc/experimental.tex` §2.2 | ✅ 8 methods, real data |
| New §2.4 κ sensitivity | `paper/sections_mc/experimental.tex` | ✅ With figure + recommendations |
| Rewritten narrative | `paper/sections_mc/experimental.tex` §2.2 | ✅ 5 findings, honest |
| κ recommendation table | Generated by `plot_kappa_sweep.py` | ✅ Manual LaTeX in output |
| Comprehensive violation table | Generated by `analyze_results_5th.py` | ✅ 8×8 LaTeX format |

---

## 10. Results File Inventory

| Directory | Files | Description |
|-----------|-------|-------------|
| `results/phase5/` | 322 JSON | Main sweep results (319 experiments + 3 analysis files) |
| `results/phase5/kappa_sweep/` | 42 JSON | κ sensitivity sweep results |
| `results/phase5/figures/` | 7 PNG | Generated plots |

### Result JSON Schema
Each result file contains:
```json
{
  "violation_rate": [mean, std],
  "cbf_violation_rate": [mean, std],
  "cumulative_reward": [mean, std],
  "tracking_rmse": {"pressure": [m,s], "enthalpy": [m,s], "power": [m,s]},
  "control_cost": [mean, std],
  "min_barrier_value": [mean, std],
  "online_time_ms": [mean, std],
  "per_constraint_type": {
    "pressure": {"violation_rate": float, "violation_count": int, "total_steps": int},
    "enthalpy": {...}, "power": {...}
  },
  "convergence_episode": int,
  "n_training_episodes": int,
  "reward_history": [...],
  "epsilon_log": [...]
}
```

---

## 11. Known Limitations & Future Work

1. **Training-evaluation gap:** CBF not used during PPO training (`_rollout_no_qp_5th`). Actor learns unsafe actions; QP can't always fix them. **Future work: CBF-aware training.**
2. **Enthalpy bottleneck:** All violations are enthalpy violations. Tighter enthalpy bounds or higher-gain enthalpy CBF needed.
3. **S3 (state-dependent) remains hard:** No learning method achieves <0.1% on S3 with κ=0. Even GP mean correction leaves 39.65%.
4. **Seed=2 instability:** Seed=2 causes training hangs on multiple conditions (S3 κ=1.0, S4 κ=0.5, S4 κ=1.0). Investigation needed.
5. **ε_κ needs per-condition tuning:** No single κ works for all conditions. Runtime κ adaptation (κ as function of GP uncertainty) is a natural extension.
6. **GPU contention:** 3-way JAX parallel on single GPU causes >2× slowdown. Serial execution recommended for future sweeps.

---

## 12. Reproducibility Commands

```bash
# Full main sweep
python experiments/phase5/run_experiment_5th.py --methods ppo ppo_lagr nmpc ppo_cbf ppo_hocbf ppo_gp_hocbf ppo_rhocbf rocbf_net

# Single method, single condition
python experiments/phase5/run_experiment_5th.py --methods ppo_gp_hocbf --conditions s3_coupled --seeds 0 1 2

# κ sensitivity sweep
python experiments/phase5/run_kappa_sweep.py --conditions s2_pressure s3_coupled s4_nonlinear --kappas 0.0 0.1 0.3 0.5 1.0

# Analysis
python experiments/phase5/analyze_results_5th.py
python experiments/phase5/plot_kappa_sweep.py
```

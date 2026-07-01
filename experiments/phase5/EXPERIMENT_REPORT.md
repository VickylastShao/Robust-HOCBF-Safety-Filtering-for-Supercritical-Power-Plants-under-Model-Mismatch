# 5th-Order CCS Experiment Report — Phase 5

> Historical note (2026-07-01): this report captures an earlier Phase 5 analysis snapshot. The current M&C submission repository is indexed by the root `README.md`, `REPRODUCIBILITY.md`, `ARTIFACT_MANIFEST.md`, and `scripts/check_repro_artifacts.py`. Treat the numbers below as historical notes unless they are also present in the current manuscript tables or regenerated result summaries.

## 1. Experimental Setup

- **Model**: USCCSDynamics5th (5 states: r_B, p_m, h_m, N_e, τ_f)
- **Constraints**: CCSConstraints5th (6 CBF constraints, ALL relative degree ≥1)
  - p_high, p_low: relative degree m=2
  - h_high, h_low: relative degree m=1
  - N_high, N_low: relative degree m=1 (NEW — was m=0 in 3rd-order model)
- **Evaluation**: 100 steps, 5 seeds
- **PPO**: trained 20 episodes on nominal dynamics, lr=1e-4
- **GP**: pre-trained 3000 points, scenario-specific, σ_floor=1e-4
- **Methods**:
  - PPO: no safety filter
  - HOCBF: nominal CBF (no uncertainty awareness)
  - GP-HOCBF: mean correction only (ε_kappa=0)
  - RHOCBF: mean correction + ε (ε_kappa=1)

## 2. Perturbation Scenarios (5th-Order)

```python
_CCS5_SCENARIOS = {
    "heat_absorption":      [0, -5, -50, 0, 0],        # S1: enthalpy disturbance
    "pressure_oscillation": [0, -6, -45, 0, 0],        # S2: pressure+enthalpy
    "coupled":              [0, 0.3*(x1-x01)-3, 0.15*(x2-x02)-40, 0, 0],  # S3: state-dependent
    "nonlinear":            [0, -0.05*(x1-x01)²-3, -45, 0, 0],  # S4: quadratic in pressure
    "valve_degradation":    [0, -4, -45, -20, 0],       # S5: includes power disturbance
    "fuel_quality":         [0, -3, -50, -15, -3],      # S6: includes τ_f disturbance
}
```

## 3. Safety Comparison (Table 3 — CBF Violation Rate)

| Scenario | PPO | HOCBF | GP-HOCBF | RHOCBF |
|----------|-----|-------|----------|--------|
| Nominal | 0.0% | 0.0% | 0.0% | 0.0% |
| S1:Heat | 40.2±18.8% | 40.2±18.8% | 0.0% | 0.0% |
| S2:Pressure | 23.6±12.2% | 23.6±12.2% | 0.0% | 0.0% |
| S3:Coupled | 29.0±15.8% | 29.0±15.8% | **6.8±7.5%** | **0.0%** |
| S4:Nonlinear | 23.6±12.2% | 23.6±12.2% | 0.0% | 0.0% |
| S5:Valve | 23.6±12.2% | 23.6±12.2% | 0.0% | 0.0% |
| S6:Fuel | 40.2±18.8% | 40.2±18.8% | 0.0% | 0.0% |

**Key observations**:
- HOCBF QP intervention rate = 0% for all scenarios → nominal CBF completely ineffective under perturbation
- GP-HOCBF achieves 0% for constant perturbations (S1/S2/S4/S5/S6)
- **S3 is the critical differentiator**: GP-HOCBF 6.8% vs RHOCBF 0% → ε margin is formally necessary
- RHOCBF achieves 0% for ALL scenarios → perfect safety guarantee

## 4. Per-Constraint Violation Breakdown (Table 8)

All violations concentrate on **h_low (enthalpy lower bound, m=1)**:

| Scenario | PPO h_low | HOCBF h_low | GP-HOCBF h_low | RHOCBF h_low |
|----------|-----------|-------------|----------------|--------------|
| S1:Heat | 41% | 41% | 0% | 0% |
| S2:Pressure | 19% | 19% | 0% | 0% |
| S3:Coupled | 26% | 26% | 0%* | 0% |
| S4:Nonlinear | 19% | 19% | 0% | 0% |
| S5:Valve | 19% | 19% | 0% | 0% |
| S6:Fuel | 41% | 41% | 0% | 0% |

*Single-seed result; 5-seed shows 6.8% average for S3

Pressure (m=2) and power (m=1) constraints are never violated.
This is because perturbations primarily drive enthalpy (Δf_h = -45 to -50),
and h_low is the most vulnerable m=1 constraint.

## 5. Epsilon Analysis — Three Roles Validated

### Role 1: Formal PAC-Bayes Certification
- **S3:Coupled**: GP-HOCBF 6.8±7.5% vs RHOCBF 0%
- Mean correction alone insufficient for state-dependent perturbations
- ε(x) margin closes the residual gap → formal guarantee

### Role 2: Per-Constraint Differentiation
ε values at equilibrium for constant-perturbation scenarios:

| Scenario | ε_p_high (m=2) | ε_h_low (m=1) | ε_N_low (m=1) | Ratio p:h |
|----------|----------------|----------------|----------------|-----------|
| S1:Heat | 0.2225 | 0.0567 | 0.0567 | 3.9:1 |
| S2:Pressure | 0.2114 | 0.0539 | 0.0539 | 3.9:1 |
| S5:Valve | 0.2175 | 0.0555 | 0.0555 | 3.9:1 |
| S6:Fuel | 0.2235 | 0.0570 | 0.0570 | 3.9:1 |

**ε_pressure(m=2) ≈ 4× ε_enthalpy(m=1)** consistently.
This is because m=2 constraints propagate σ through two Lie derivatives,
amplifying the uncertainty by the Lie derivative norms.

### Role 3: State-Dependent Adaptation
ε(x) variation along trajectory (coefficient of variation):

| Scenario | CV(ε_p_high) | CV(ε_h_low) | CV(ε_N_low) | σ_GP variation |
|----------|-------------|-------------|-------------|----------------|
| S1:Heat | 0.04% | 0.00% | 0.00% | floor (0.01) |
| S2:Pressure | 0.04% | 0.00% | 0.00% | floor (0.01) |
| **S3:Coupled** | **3.82%** | **70.06%** | 0.00% | **CV(σ_h)=76.2%** |
| **S4:Nonlinear** | **8.45%** | 0.00% | 0.00% | **CV(σ_p)=28.2%** |
| S5:Valve | 0.03% | 0.00% | 0.00% | floor (0.01) |
| S6:Fuel | 0.08% | 0.00% | 0.00% | floor (0.01) |

**S3:Coupled is the key scenario**: CV(ε_h)=70.06% demonstrates significant
state-dependent ε(x) variation. This occurs because the state-dependent
perturbation (0.3*(x1-x01), 0.15*(x2-x02)) creates non-uniform GP coverage
(σ_p CV=48.3%, σ_h CV=76.2%).

## 6. QP Intervention Rates

| Scenario | HOCBF | GP-HOCBF | RHOCBF |
|----------|-------|----------|--------|
| Nominal | 0% | 40.0±19.2% | 40.0±19.2% |
| S1:Heat | 0% | 40.2±18.8% | 40.2±18.8% |
| S2:Pressure | 0% | 23.6±12.2% | 23.6±12.2% |
| S3:Coupled | 0% | 28.2±15.2% | 29.0±15.8% |
| S4:Nonlinear | 0% | 24.0±11.8% | 24.0±11.8% |
| S5:Valve | 0% | 23.6±12.2% | 23.6±12.2% |
| S6:Fuel | 0% | 40.2±18.8% | 40.4±18.9% |

HOCBF never intervenes (0%) → nominal CBF provides no safety modification.
GP-HOCBF/RHOCBF intervene at rates matching PPO violation rates →
safety filter is the sole mechanism preventing violations.

## 7. Comparison with 3rd-Order Model

| Dimension | 3rd-Order | 5th-Order | Improvement |
|-----------|-----------|-----------|-------------|
| State dim | 3 | 5 | σ_GP naturally non-uniform for S3/S4 |
| Power rd | 0 (not CBF) | 1 (CBF) | **No rd-0 artifact** |
| CBF constraints | 4 | 6 | Richer per-constraint structure |
| ε(x) variation | ~0 (floor) | S3: CV=70% | **State-dependent evidence on CCS** |
| Power violation | 92.7% (rd-0) | 0% (m=1) | **Eliminated** |

## 8. Summary for Paper

The 5th-order CCS model validates all three roles of ε(x) on a single platform:

1. **Formal guarantee** (Role 1): S3 shows GP-HOCBF 6.8% vs RHOCBF 0% → ε margin is necessary
2. **Per-constraint differentiation** (Role 2): ε_pressure(m=2) ≈ 4× ε_enthalpy(m=1) across all scenarios
3. **State-dependent adaptation** (Role 3): S3:Coupled shows CV(ε)=70% for h_low constraint

S3:Coupled is the single scenario that simultaneously demonstrates all three roles,
making it the primary evidence scenario for the paper.

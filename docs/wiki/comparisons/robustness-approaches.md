---
title: Comparison — Robustness Approaches for CBFs
tags: [comparison, robustness, uncertainty]
sources: [cohen2022, das2024, choi2021]
updated: 2026-05-19
---

## Uncertainty Model

| Method | Uncertainty Type | Dynamics Form | Structural Assumption |
|--------|-----------------|---------------|----------------------|
| HO-RaCBF [[cohen2022]] | Parametric | $\dot{x}=f+Y\theta+gu$ | Matching condition (IRD=DRD) |
| Estimator CBF [[das2024]] | Unmodeled | $\dot{x}=\hat{f}+\hat{g}u+\Delta$ | Lipschitz $\Delta$, bounded |
| CBVF [[choi2021]] | Disturbance | $\dot{x}=f+gu+d$ | $\|d\| \leq d_{\max}$ |
| **RoCBF-Net** | **Learned** | $\dot{x}=\hat{f}+\hat{g}u+\Delta_{\text{GP}}$ | **GP prior, data-driven** |

## Safety Guarantee Type

| Method | Guarantee | Conservatism |
|--------|-----------|-------------|
| HO-RaCBF | Asymptotic: $\nu(t) \to 0$ → exact constraint recovery | Moderate (shrinks with $\nu$) |
| Estimator CBF | Exponential: $\|e(t)\|$ converges exponentially | Moderate (shrinks with $\|e\|$) |
| CBVF | Worst-case: deterministic for all $d \in \mathcal{D}$ | High (accounts for worst case) |
| **RoCBF-Net** | **Probabilistic**: $1-\delta$ confidence via GP posterior | **Adaptive** (data-dependent) |

## Computational Cost

| Method | Offline | Online | Scalability |
|--------|---------|--------|-------------|
| HO-RaCBF | None | QP + $\nu$ update | $O(n_u^2)$ QP |
| Estimator CBF | None | QP + observer | $O(n_u^2)$ QP + $O(n_x)$ observer |
| CBVF | HJ grid solve | CBVF-QP | **Curse of dimensionality** (offline) |
| **RoCBF-Net** | GP training (optional) | QP + Kalman GP step | $O(d^3)$ Kalman + $O(n_u^2)$ QP |

## What RoCBF-Net Takes From Each

- **From Cohen 2022**: Robust buffer concept $\|L_Y \psi\| \nu$ → adapted to $\beta \|L_{\hat{g}} \psi\| \sigma_*$; ψ-chain propagation of uncertainty
- **From Das 2024**: Uncertainty estimator structure → replaced by GP posterior; HOCBF extension pattern (Corollary 1) → direct template for our robust HOCBF
- **From Choi 2021**: Motivation for less conservative approach → GP provides data-adaptive uncertainty instead of worst-case

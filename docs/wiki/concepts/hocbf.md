---
title: HOCBF — High-Order Control Barrier Functions
tags: [hocbf, cbf, safety, relative-degree]
sources: [xiao2019, cohen2022, das2024]
updated: 2026-05-19
---

## Definition

A function $h: \mathbb{R}^n \to \mathbb{R}$ is a **High-Order CBF (HOCBF)** of relative degree $m$ for system $\dot{x}=f(x)+g(x)u$ if there exist class-$\mathcal{K}$ functions $\alpha_1, \ldots, \alpha_m$ such that the $\psi$-chain:

$$\psi_0 = h, \quad \psi_i = \dot{\psi}_{i-1} + \alpha_i(\psi_{i-1}), \quad i=1,\ldots,m$$

satisfies $\psi_0(x) \geq 0 \Rightarrow \psi_1(x) \geq 0 \Rightarrow \cdots \Rightarrow \psi_m(x) \geq 0$, and the control constraint $\psi_m(x) \geq 0$ is feasible.

The resulting safety constraint on $u$ is:

$$L_f^m h + \sum_{j=0}^{m-1} \binom{m}{j} L_f^j [\alpha_{m-j} \circ \cdots \circ \alpha_1(h)] + L_g L_f^{m-1} h \cdot u \geq 0$$

## Special Case: m=2 with Linear Class-K

With $\alpha_1(r) = k_1 r$, $\alpha_2(r) = k_2 r$, $k_1, k_2 > 0$:

$$-L_g L_f h \cdot u \leq L_f^2 h + (k_1 + k_2) L_f h + k_1 k_2 h$$

This is the formula we use in Phase 1. The right-hand side defines the $S$ function:

$$S = L_f^2 h + (k_1 + k_2) L_f h + k_1 k_2 h$$

The $\psi$-chain for $m=2$:
- $\psi_0 = h$
- $\psi_1 = L_f h + k_1 h$
- $\psi_2 = L_f \psi_1 + k_2 \psi_1 = L_f^2 h + (k_1+k_2)L_f h + k_1 k_2 h + L_g L_f h \cdot u$

## Variants Across Papers

| Paper | Variant | Key Difference |
|-------|---------|---------------|
| [[xiao2019]] | Standard HOCBF | Original definition, deterministic, no uncertainty |
| [[cohen2022]] | HO-RaCBF | Adds robustness buffer $\|L_Y \psi_{r-1}\| \nu(t)$ for parametric uncertainty |
| [[das2024]] | Robust HOCBF (Corollary 1) | Adds uncertainty estimator $\hat{\Delta}$ and error bound $\|e(t)\|$ for unmodeled dynamics |

## Robustness Extensions

### HO-RaCBF (Cohen & Belta 2022)
For $\dot{x} = f(x) + Y(x)\theta + g(x)u$ with unknown $\theta$:

$$L_f \psi_{r-1} + L_Y \psi_{r-1} \hat{\theta} + L_g \psi_{r-1} u \geq -\alpha_r(\psi_{r-1}) + \|L_Y \psi_{r-1}\| \nu(t)$$

where $\nu(t)$ decays exponentially via concurrent learning.

### Robust HOCBF (Das & Burdick 2024)
For $\dot{x} = \hat{f}(x) + \hat{g}(x)u + \Delta(x,u)$ with unmodeled $\Delta$:

$$L_f^m h + L_{\hat{g}} L_f^{m-1} h \cdot u + \frac{\partial L_f^{m-1} h}{\partial x} \hat{\Delta} - \left\|\frac{\partial L_f^{m-1} h}{\partial x}\right\| \|e(t)\| + O(h) \geq -\alpha_m(\psi_{m-1})$$

where $\hat{\Delta}(t) = \Lambda x - \xi(t)$ is the uncertainty estimator and $e(t) = \Delta(t) - \hat{\Delta}(t)$.

## Implementation Notes for RoCBF-Net

- **Phase 1**: Implement standard HOCBF with $m=2$, linear class-K, factory-closure ψ-chain
- **Phase 2**: Replace Das's $\hat{\Delta}$ with GP posterior mean, replace $\|e(t)\|$ with GP posterior variance-based bound
- **Key design choice**: Use ψ-chain formulation for `qp_matrices()` rather than recursive recomputation

## Open Questions

1. How to compose multiple HOCBF constraints (multiple safety specifications)?
2. Feasibility guarantees when $L_g L_f^{m-1} h = 0$ (loss of relative degree)?
3. Optimal class-K selection — heuristic (Xiao) vs learned (Ma) vs adaptive (BarrierNet)?
4. How does GP posterior variance propagate through the ψ-chain for $m > 2$?

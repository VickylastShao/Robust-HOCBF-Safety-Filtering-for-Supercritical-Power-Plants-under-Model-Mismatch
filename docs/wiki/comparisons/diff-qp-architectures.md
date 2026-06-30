---
title: Comparison — Differentiable QP Architectures
tags: [comparison, differentiable-qp, architecture]
sources: [barriernet, ma2022]
updated: 2026-05-19
---

## CBF Type

| Method | CBF Type | Relative Degree | Class-K Functions |
|--------|----------|----------------|-------------------|
| BarrierNet [[barriernet]] | HOCBF | Arbitrary $m$ | Fixed linear $k_i$, softened with penalty $p_i$ |
| Ma 2022 [[ma2022]] | ECBF | Arbitrary $r$ | Learned via α-net (eigenvalue parameterization) |

## QP Formulation

| Method | Objective | Constraint | Slack |
|--------|-----------|-----------|-------|
| BarrierNet | $\min \|u - u_{\text{nom}}\|^2$ | Softened: $\psi_i \geq -p_i(z)$ | Implicit via $p_i$ |
| Ma 2022 | $\min \|u - u_{\text{nom}}\|^2 + \rho s^2$ | Hard: $\psi_r \geq 0$ | Explicit slack $s \geq 0$ |

## Gradient Computation

| Method | Technique | Framework |
|--------|-----------|-----------|
| BarrierNet | KKT implicit differentiation (manual derivation) | Custom JAX/PyTorch |
| Ma 2022 | cvxpylayers (automatic) | cvxpy + PyTorch |

## Learnable Parameters

| Method | What is Learned | Input |
|--------|----------------|-------|
| BarrierNet | Penalty $p_i(z; \theta)$ via upstream NN | Environment features $z$ |
| Ma 2022 | Class-K gains $k_i(z; \theta)$ via α-net | Environment features $z$ |

## Safety During Training

| Method | Approach | Risk |
|--------|----------|------|
| BarrierNet | Softened constraint ($p_i > 0$ allows violation) | Medium — violation bounded by $p_i$ |
| Ma 2022 | Slack variable with penalty in loss | High — slack allows arbitrary violation during training |

## What RoCBF-Net Adopts

- **From BarrierNet**: KKT implicit differentiation approach (more explicit than cvxpylayers); trainable penalty concept
- **From Ma 2022**: Learnable class-K gains (α-net concept); eigenvalue parameterization for stability
- **New in RoCBF-Net**: GP uncertainty feeds into QP constraint; gradients flow through GP predictions AND QP solution

## Implementation Decision

For RoCBF-Net Phase 1:
- Use **optimistix** (JAX-native QP solver) for forward pass
- Implement **KKT implicit differentiation** (BarrierNet style) for backward pass
- Train **learnable class-K gains** $k_i(z; \theta)$ (Ma 2022 style) with eigenvalue parameterization
- Add **softened penalty** $p_i(z; \theta)$ (BarrierNet style) for training feasibility

Both learnable components are outputs of a shared upstream network that takes environment/state features as input.

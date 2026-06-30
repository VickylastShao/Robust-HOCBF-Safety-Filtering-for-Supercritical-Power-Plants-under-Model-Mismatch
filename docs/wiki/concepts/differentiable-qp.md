---
title: Differentiable QP Layers
tags: [differentiable-qp, kkt, cvxpylayers, gradient]
sources: [barriernet, ma2022]
updated: 2026-05-19
---

## Definition

A differentiable QP layer solves a quadratic program in the forward pass and computes gradients through the KKT conditions in the backward pass, enabling end-to-end training of safety-critical controllers.

## Standard QP Formulation (CBF-QP)

$$\min_u \frac{1}{2} u^T H u + c^T u \quad \text{s.t.} \quad A u \leq b$$

For HOCBF: $H = I$, $c = -u_{\text{nom}}$, $A = -L_g L_f^{m-1} h$, $b = L_f^m h + \sum_{j=0}^{m-1} \ldots$ (from ψ-chain)

## Variants Across Papers

| Paper | Variant | Key Difference |
|-------|---------|---------------|
| [[barriernet]] | Softened HOCBF-QP + KKT implicit differentiation | Trainable penalty $p_i(z)$ from upstream NN; softened constraint $\psi_i \geq -p_i$; gradients via KKT linear system |
| [[ma2022]] | ECBF-QP + cvxpylayers | Learned class-K gains via α-net; slack variable for training-time feasibility; cvxpylayers for auto-diff |

## BarrierNet Approach

**Forward pass**: Upstream network produces penalty $p_i(z; \theta)$, then solve:

$$\min_u \|u - u_{\text{nom}}\|^2 \quad \text{s.t.} \quad L_g \psi_{m-1} u \geq -L_f \psi_{m-1} - \alpha_m(\psi_{m-1}) + p_m$$

**Backward pass**: Differentiate through KKT conditions. At optimal $(u^*, \lambda^*)$:

$$\begin{bmatrix} H & A^T \\ \text{diag}(\lambda^*) A & \text{diag}(A u^* - b) \end{bmatrix} \begin{bmatrix} d_u \\ d_\lambda \end{bmatrix} = \begin{bmatrix} d_c \\ d_b \end{bmatrix}$$

Then chain rule to upstream parameters: $\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial u^*} \frac{\partial u^*}{\partial b} \frac{\partial b}{\partial p} \frac{\partial p}{\partial \theta}$

## Ma 2022 Approach

**Alpha-net**: Learns eigenvalues $p_i > 0$ of the companion matrix, which define the class-K gains. ReLU + bias trick: $p_i = \text{ReLU}(w_i^T z + b_i^0) + b_i$ ensures $p_i \geq b_i > 0$.

**Slack variable**: Adds $s \geq 0$ to QP with penalty in loss to allow training-time violations while maintaining safety at convergence.

**Differentiation**: Uses cvxpylayers (Agrawal et al. 2019) for automatic differentiation through the QP.

## Implementation Notes for RoCBF-Net

- **Phase 1**: Implement differentiable QP layer using `optimistix` (JAX-native) or `cvxpylayers`
- **Gradient computation**: KKT implicit differentiation (BarrierNet style) — more explicit and controllable
- **Trainable parameters**: Penalty parameters $p_i$ (from BarrierNet) and class-K gains $k_i$ (from Ma 2022)
- **Phase 2 extension**: GP uncertainty parameters feed into the QP constraint; gradients flow through both GP predictions and QP solution

## Open Questions

1. `optimistix` vs `cvxpylayers` — which is more stable for JAX/Flax NNX?
2. How to handle infeasible QP during training? (Ma uses slack, BarrierNet uses softened constraint)
3. Numerical stability of KKT system when active set changes?
4. Batch QP solving on GPU — can we vectorize across environments?

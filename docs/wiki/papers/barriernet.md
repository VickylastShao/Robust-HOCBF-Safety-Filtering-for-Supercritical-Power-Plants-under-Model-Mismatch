---
title: BarrierNet — Differentiable Safety-Guaranteed Neural Network Layer
authors: [Wei Xiao, Ramin Hasani, Xiao Li, Daniela Rus]
year: 2021
venue: arXiv:2111.11277v1 (CSAIL MIT)
tags: [barriernet, differentiable-qp, safety-layer, neural-network, HOCBF, KKT-gradients]
sources: [barriernet]
updated: 2026-05-19
---

## One-Line Summary

BarrierNet embeds a differentiable Higher-Order Control Barrier Function (HOCBF) quadratic program as a trainable neural-network layer, guaranteeing safety while learning environment-adaptive penalty parameters $p_i(z)$ that soften the CBF constraints without sacrificing forward invariance.

## Problem Setting

### System Model

Affine control system:

$$\dot{x} = f(x) + g(x)\,u$$

where $x \in \mathbb{R}^n$, $f: \mathbb{R}^n \to \mathbb{R}^n$ and $g: \mathbb{R}^n \to \mathbb{R}^{n \times q}$ are locally Lipschitz, and $u \in \mathcal{U} \subset \mathbb{R}^q$.

### Uncertainty Model

No explicit stochastic uncertainty model. Conservativeness of the HOCBF under increasing model uncertainty is addressed structurally by making the class $\mathcal{K}$ function gains trainable and environment-dependent. Lipschitz continuity of the penalty functions $p_i(z)$ is required (enforced via smooth activations such as sigmoid in the upstream network).

### Control Objective (Problem 1)

Given:
1. Known affine dynamics Eq. (1)
2. A nominal controller $f^*(x) = u^*$
3. Safety constraints $b_j(x) \geq 0,\; j \in S$
4. Control bounds $u_{\min} \leq u \leq u_{\max}$
5. A neural network controller $f(x|\theta) = u$ parameterized by $\theta$

Find optimal parameters:

$$\theta^* = \arg\min_\theta \; \mathbb{E}_x\bigl[\,l\bigl(f^*(x),\, f(x|\theta)\bigr)\bigr]$$

while guaranteeing all safety constraints and control bounds. This is a **policy distillation with safety guarantees** formulation.

## Key Contributions

1. **Trainable safety layer (BarrierNet):** Proposes a novel, interpretable, end-to-end trainable neural-network layer built on HOCBFs that provides safety guarantees for general control problems with arbitrary relative degree.

2. **Softening HOCBF constraints without losing safety:** Replaces the hard class $\mathcal{K}$ function gains in HOCBFs with differentiable, environment-dependent penalty functions $p_i(z)$, addressing the over-conservativeness of standard CBFs while retaining formal safety guarantees.

3. **Environment-adaptive CBF parameters:** The penalty functions $p_i(z)$ and cost parameters ($H$, $F$) depend on observations $z$ (environment features), making safety constraints adaptable to changing environments — trained from data rather than hand-designed auxiliary dynamics (as required by AdaCBF).

## Core Formulation

### Softened HOCBF Sequence

For a safety constraint $b(x) \geq 0$ with relative degree $m$, define $\psi_0(x,z) := b(x)$ and the modified CBF sequence:

$$\psi_i(x, z) := \dot{\psi}_{i-1}(x, z) + p_i(z)\,\alpha_i\bigl(\psi_{i-1}(x, z)\bigr), \quad i \in \{1, \ldots, m\}$$

where $\alpha_i(\cdot)$ are $(m-i)$-th order differentiable class $\mathcal{K}$ functions, and $p_i: \mathbb{R}^d \to \mathbb{R}_{>0}$ are **trainable penalty functions** output by the previous network layer. This is analogous to AdaCBF but **trainable** and without the need to design auxiliary dynamics for $p_i$.

The softened HOCBF constraint becomes:

$$L_f^m b(x) + \bigl[L_g L_f^{m-1} b(x)\bigr]u + O(b(x), z) + p_m(z)\,\alpha_m\bigl(\psi_{m-1}(x, z)\bigr) \geq 0$$

where:

$$O(b(x), z) = \sum_{i=1}^{m-1} L_f^{m-i}\bigl(\alpha_{m-i} \circ \psi_{m-i-1}\bigr)(x, z)$$

### BarrierNet QP Layer (Definition 7)

Each BarrierNet neuron solves:

$$u^*(t) = \arg\min_{u(t)} \;\frac{1}{2}\,u(t)^T H(z|\theta_h)\,u(t) + F^T(z|\theta_f)\,u(t)$$

subject to:

$$L_f^m b_j(x) + \bigl[L_g L_f^{m-1} b_j(x)\bigr]u + O(b_j(x), z|\theta_p) + p_m(z|\theta_{p_m})\,\alpha_m\bigl(\psi_{m-1}(x, z|\theta_p)\bigr) \geq 0, \quad j \in S$$

$$u_{\min} \leq u \leq u_{\max}$$

where:
- $H(z|\theta_h) \in \mathbb{R}^{q \times q}$ — trainable cost matrix (can depend on observations or be a standalone parameter)
- $F(z|\theta_f) \in \mathbb{R}^q$ — reference control (output of previous network layers)
- $\theta_p = (\theta_{p_1}, \ldots, \theta_{p_m})$ — trainable parameters for penalty functions
- Trainable parameters: $\theta = \{\theta_h, \theta_f, \theta_p\}$

$F(z|\theta_f)$ is interpreted as a reference control that the QP tracks while enforcing safety. This enables tracking arbitrary nominal controllers rather than just minimizing control effort.

### Gradient Flow Through KKT Conditions (Implicit Differentiation)

Gradients flow through the QP solution via the KKT conditions, following the OptNet approach (Amos & Kolter, 2017). Define:

$$G_j = -L_g L_f^{m-1} b_j(x), \qquad h_j = L_f^m b_j(x) + O(b_j(x), z) + p_m(z)\,\alpha_m(\psi_{m-1}(x, z))$$

Let $G$ and $h$ be the concatenation over $j \in S$, and $\lambda$ be the dual variables on the HOCBF constraints. The loss gradients with respect to BarrierNet parameters are:

$$\frac{\partial H}{\partial} \ell = \frac{1}{2}\bigl(d_u\, u^{*T} + u^*\, d_u^T\bigr), \qquad \frac{\partial F}{\partial} \ell = d_u$$

$$\frac{\partial G}{\partial} \ell = D(\lambda^*)\,(d_\lambda\, u^{*T} + \lambda^*\, d_u^T), \qquad \frac{\partial h}{\partial} \ell = -D(\lambda^*)\, d_\lambda$$

where $D(\cdot)$ creates a diagonal matrix from a vector, and $d_u$, $d_\lambda$ are obtained by solving the linear system:

$$\begin{bmatrix} H & G^T D(\lambda^*) \\ G\, D(Gu^* - h) & 0 \end{bmatrix}^{-1} \begin{bmatrix} d_u \\ d_\lambda \end{bmatrix} = \begin{bmatrix} \frac{\partial \ell}{\partial u^*} \\ 0 \end{bmatrix}$$

This is derived by taking the Lagrangian of the QP and differentiating the KKT conditions. Since $G$ is determined by the HOCBF structure (not directly trainable), $\frac{\partial G}{\partial}\ell$ is not applicable in BarrierNet. However, gradients propagate to penalty parameters via:

$$\frac{\partial p_i}{\partial} \ell = \frac{\partial h_j}{\partial} \ell \cdot \frac{\partial h_j}{\partial p_i}, \quad i \in \{1,\ldots,m\},\; j \in S$$

where $\frac{\partial h_j}{\partial p_i}$ is obtained by taking the partial derivative of $h_j$ with respect to $p_i$.

## Theoretical Results

### Safety Guarantee (Theorem 8)

> If $p_i(z),\; i \in \{1, \ldots, m\}$ are Lipschitz continuous, then a BarrierNet composed by neurons as in Definition 7 guarantees the safety of system (1).

**Proof sketch:** Lipschitz continuity of $p_i(z)$ ensures each $\psi_i(x,z)$ in the softened sequence (Eq. 6) is a valid CBF (by Theorem 5 / HOCBF theory). Starting from $\psi_m(x,z) \geq 0$ (the non-Lie-derivative form of each HOCBF constraint), $\psi_{m-1}(x,z) \geq 0$ is guaranteed by Theorem 5. Recursively, $\psi_0(x,z) \geq 0$ is guaranteed, and since $b(x) = \psi_0(x,z)$, system (1) is safety guaranteed.

### Forward Invariance (Inherited from HOCBF, Theorem 5)

Given a HOCBF $b(x)$ with associated sets $C_1, \ldots, C_m$ defined by:

$$C_i := \{x \in \mathbb{R}^n : \psi_{i-1}(x) \geq 0\}, \quad i \in \{1, \ldots, m\}$$

if $x(0) \in C_1 \cap \cdots \cap C_m$, then any Lipschitz continuous controller satisfying the HOCBF constraint renders $C_1 \cap \cdots \cap C_m$ forward invariant.

### Feasibility

The paper does not provide explicit feasibility guarantees for the BarrierNet QP. Feasibility depends on the interplay between control bounds and the HOCBF constraints. The softening via $p_i(z)$ can improve feasibility by reducing the conservativeness of the constraints (allowing the QP to find solutions that a standard HOCBF QP might reject), but formal feasibility analysis is not provided.

### Adaptivity (Remark 9)

The HOCBF constraints are softened by trainable penalty functions without losing safety guarantees. Since penalty functions are environment-dependent (features from upstream networks), the CBF constraints adapt to changing environments, enabling the BarrierNet to generate safe controls while avoiding over-conservativeness.

## Algorithm / Implementation

### Forward Pass

1. Input observations $z$ (environment features) and system state $x$ (feedback).
2. Upstream network (e.g., FC layers with sigmoid activations) outputs penalty functions $p_i(z|\theta_{p_i})$, reference control $F(z|\theta_f)$, and cost matrix $H(z|\theta_h)$.
3. Construct the QP from Definition 7 using current $x$, $z$, and network outputs.
4. Solve the QP (pointwise at each time step $t = k\Delta t + t_0$) to obtain safe control $u^*$.

### Backward Pass

1. Compute loss $\ell$ (similarity measure between $u^*$ and nominal control $u^*$).
2. Compute $\frac{\partial \ell}{\partial u^*}$.
3. Solve the linear system (Eq. 13) for $d_u$ and $d_\lambda$.
4. Compute gradients $\frac{\partial H}{\partial}\ell$, $\frac{\partial F}{\partial}\ell$, $\frac{\partial h}{\partial}\ell$ via Eq. (11).
5. Propagate to penalty parameters: $\frac{\partial p_i}{\partial}\ell = \frac{\partial h_j}{\partial}\ell \cdot \frac{\partial h_j}{\partial p_i}$.
6. Apply chain rule through upstream network to $\theta_h, \theta_f, \theta_p$.

### Training Procedure (Algorithm 1)

1. **Construct** softened HOCBFs by Eq. (6) for each safety requirement.
2. **Construct** the BarrierNet by Eq. (9).
3. **Collect** training data using a nominal controller (e.g., OCBF, optimal controller).
4. **Initialize** BarrierNet parameters $\theta$, number of epochs, and learning rate $\gamma$.
5. **Loop** over epochs:
   - Forward: Solve QP (9), compute loss $\ell$.
   - Backward: Compute $\frac{\partial}{\partial \theta}\ell$ using Eq. (11).
   - Update: $\theta \leftarrow \theta - \gamma\,\frac{\partial}{\partial \theta}\ell$.
6. **Return** optimal parameters $\theta$.

### Design Choices

- Upstream network activations: sigmoid (or other continuously differentiable functions) to ensure Lipschitz continuity of $p_i(z)$.
- $\dot{p}_1(x)$ is set to $0$ in the discretized QP formulation (2D/3D navigation cases).
- Class $\mathcal{K}$ functions $\alpha_i$ are chosen as linear functions in all experiments.

## Limitations

1. **Fixed number of safety constraints at training time:** The number of constraints in a BarrierNet must be defined during training. Time-varying constraint counts (e.g., dynamic obstacle counts) remain a challenge. A workaround is to define more constraints than necessary and enable them only when required.

2. **Inter-sampling effects:** BarrierNet is solved in discrete time (data fed at discrete intervals). Safety between sampling times is not formally guaranteed, especially near the safety set boundary. The Lyapunov property of HOCBFs ensures the system stays close to the boundary if violated, but does not prevent violation. A potential remedy is the data-driven event-triggered framework (Xiao et al., 2021a).

3. **No explicit feasibility guarantees:** The QP may become infeasible if control bounds conflict with safety constraints, and no formal feasibility certificate is provided.

4. **Known dynamics assumed:** The affine dynamics $f(x)$ and $g(x)$ must be known a priori to compute the Lie derivatives in the HOCBF constraints. Learning dynamics simultaneously is identified as future work.

5. **Relative degree and HOCBF structure must be specified:** The relative degree $m$ and the choice of class $\mathcal{K}$ functions are design choices, not learned.

## Relevance to RoCBF-Net

| Aspect | BarrierNet | RoCBF-Net Connection |
|--------|-----------|---------------------|
| **Differentiable QP layer** | Core contribution: HOCBF-QP as a differentiable layer | Directly adopted as the architectural pattern for the safety layer in RoCBF-Net |
| **KKT-based gradient computation** | Implicit differentiation through KKT conditions (Eq. 11, 13) | Adopted for backpropagating through the CBF-QP safety layer |
| **Trainable penalty parameters $p_i(z)$** | Environment-adaptive softening of CBF constraints | Relevant to the learning phase where RoCBF may learn adaptive parameters |
| **Policy distillation formulation** | Problem 1: match nominal controller subject to safety | Relevant if RoCBF-Net uses expert/reference controller supervision |
| **HOCBF for arbitrary relative degree** | Handles relative degree $m > 1$ | Potentially applicable if RoCBF-Net deals with high-relative-degree constraints |

**Phase relevance:** The differentiable QP layer architecture and KKT gradient computation are most relevant to the **safety layer / control synthesis phase** of RoCBF-Net, where a CBF-based QP must be embedded as a differentiable module within a learning pipeline.

**What we adopt:**
- The differentiable QP layer pattern (QP as a neural network layer with forward = QP solve, backward = KKT implicit differentiation)
- The KKT linear system for computing gradients through the QP (Eq. 13)
- The concept of trainable parameters within the CBF constraints (penalty functions $p_i(z)$)

**What we may differ on:**
- RoCBF-Net may use robust CBFs (addressing bounded uncertainty) rather than the standard HOCBF formulation
- The dynamics model and uncertainty handling may differ (RoCBF-Net likely considers robust/uncertain dynamics)

## Cross-References

- [[xiao2021hocbf]] — Xiao & Belta, "High Order Control Barrier Functions," IEEE TAC, 2021. Foundation for the HOCBF formulation used in BarrierNet.
- [[ames2017]] — Ames et al., "Control Barrier Function Based Quadratic Programs for Safety Critical Systems," IEEE TAC, 2017. Original CBF-QP framework.
- [[amos2017optnet]] — Amos & Kolter, "OptNet: Differentiable Optimization as a Layer in Neural Networks," ICML, 2017. Differentiable QP layer that BarrierNet builds upon for gradient computation.
- [[xiao2021adacbf]] — Xiao et al., "Adaptive Control Barrier Functions," IEEE TAC, 2021b. AdaCBF: the precursor to BarrierNet's trainable penalty approach; requires hand-designed auxiliary dynamics.
- [[xiao2021ocbf]] — Xiao, Cassandras & Belta, "Bridging the Gap between Optimal Trajectory Planning and Safety-Critical Control," Automatica, 2021c. OCBF controller used for training data generation.
- [[xiao2021event]] — Xiao et al., "Event-triggered Safety-critical Control for Systems with Unknown Dynamics," CDC, 2021a. Potential approach for addressing the inter-sampling limitation.
- [[pereira2020]] — Pereira et al., "Safe Optimal Control using Stochastic Barrier Functions and Deep Forward-Backward SDEs," CoRL, 2020. DFB baseline: non-trainable CBF-QP safety filter compared against in experiments.
- [[taylor2020]] — Taylor et al., "Learning for Safety-critical Control with Control Barrier Functions," L4DC, 2020a. CBF + learning baseline.

---
title: HO-RaCBF — High-Order Robust Adaptive CBFs
authors: [Max H. Cohen, Calin Belta]
year: 2022
venue: American Control Conference (ACC 2022)
tags: [hocbf, robust, adaptive, parametric-uncertainty, ra-cbf]
sources: [cohen2022]
updated: 2026-05-19
---

## One-Line Summary

Introduces High-Order Robust Adaptive Control Barrier Functions (HO-RaCBFs) that extend the adaptive CBF paradigm to safety constraints with arbitrary relative degree, using a robustness buffer that decays exponentially via concurrent learning, paired with Exponentially Stabilizing Adaptive Control Lyapunov Functions (ES-aCLFs) for stability.

## Problem Setting

Consider a nonlinear control-affine system with **parametric uncertainty**:

$$\dot{x} = f(x) + Y(x)\theta + g(x)u, \quad x \in \mathbb{R}^n, \; u \in \mathcal{U} \subseteq \mathbb{R}^m$$

where:
- $f: \mathbb{R}^n \to \mathbb{R}^n$ and $g: \mathbb{R}^n \to \mathbb{R}^{n \times m}$ are known locally Lipschitz drift and control vector fields
- $Y: \mathbb{R}^n \to \mathbb{R}^{n \times p}$ is a known locally Lipschitz regression matrix
- $\theta \in \mathbb{R}^p$ is a constant vector of **uncertain parameters** (unknown but belonging to a known convex polytope $\Theta \subset \mathbb{R}^p$)
- It is assumed that $f(0) = 0$ and $Y(0) = 0$ so that the origin is an equilibrium of the unforced system

The **control objective** is twofold:
1. **Safety**: Render the safe set $\mathcal{C} = \{x \in \mathbb{R}^n \mid h(x) \ge 0\}$ forward invariant, where $h$ has **high relative degree** $r$ with respect to the system (i.e., $L_g L_f^{i-1} h(x) \equiv 0$ for $1 \le i \le r-1$ and $L_g L_f^{r-1} h(x) \neq 0$)
2. **Stability**: Exponentially stabilize the origin (or the safe set from outside)

## Key Contributions

1. **HO-RaCBF definition (Def. 3)**: First extension of the adaptive CBF paradigm from relative degree one to arbitrary relative degree under mild structural assumptions on the uncertainty
2. **Robustness buffer with exponential decay**: Introduces $\|L_Y \psi_{r-1}(x)\| \nu(t)$ as a safety margin that accounts for all possible parameter realizations and shrinks to zero as concurrent learning identifies the true parameters
3. **Zeroing CBF robustness inheritance (Corollary 1)**: Unlike prior aCBF formulations, HO-RaCBFs guarantee not just forward invariance but also **asymptotic stability** of the safe set when starting outside it — a key practical robustness property
4. **ES-aCLF definition (Def. 4)**: A novel class of adaptive CLF that leverages the same concurrent learning data to guarantee **exponential** convergence of both the state and parameter estimation error, going beyond the uniformly ultimately bounded guarantees of prior work
5. **Unified QP framework**: Safety (HO-RaCBF) and stability (ES-aCLF) are combined in a single quadratic program with affine constraints on the control input

## Core Formulation

### HOCBF Preliminaries

For a function $h: \mathbb{R}^n \to \mathbb{R}$ with relative degree $r$ and a collection of extended class $\mathcal{K}$ functions $\{\alpha_i\}_{i=1}^{r}$, define:

$$\psi_0(x) := h(x), \quad \psi_i(x) := \dot{\psi}_{i-1}(x) + \alpha_i(\psi_{i-1}(x)), \quad i \in \{1, \ldots, r-1\}$$

$$\psi_r(x, u) := \dot{\psi}_{r-1}(x, u) + \alpha_r(\psi_{r-1}(x))$$

with sets $\mathcal{C}_i := \{x \in \mathbb{R}^n \mid \psi_{i-1}(x) \ge 0\}$.

### Matching Condition (Assumption 1)

If $h$ has relative degree $r$ with respect to the control (i.e., $L_g L_f^{i-1} h(x) \equiv 0$ for $1 \le i \le r-1$ and $L_g L_f^{r-1} h(x) \neq 0$), then the uncertainty must not appear before the control in the Lie derivative chain:

$$L_Y L_f^{i-1} h(x) \equiv 0 \quad \text{for all } 1 \le i \le r-1, \qquad L_Y L_f^{r-1} h(x) \neq 0$$

This is analogous to the **matching condition** from adaptive control — the uncertain parameters enter the system through the same channel as the control input. Lagrangian mechanical systems (where $h$ depends only on configuration) typically satisfy this.

### Concurrent Learning and the Bound $\nu(t)$

Integrating the system dynamics over $[t - \Delta T, t]$ yields an equivalent data-driven representation:

$$\Delta x(t) = F(t) + \mathcal{Y}(t)\theta + G(t)$$

where $F(t) = \int_{t-\Delta T}^{t} f(x(\tau))\,d\tau$, $\mathcal{Y}(t) = \int_{t-\Delta T}^{t} Y(x(\tau))\,d\tau$, $G(t) = \int_{t-\Delta T}^{t} g(x(\tau))u(\tau)\,d\tau$.

Given a history stack $\mathcal{H} = \{(t_j, x_j, x_j^-, u_j)\}_{j=1}^{M}$, define:

$$\Lambda(t) := \sum_{j=1}^{M} \mathcal{Y}_j^\top \mathcal{Y}_j, \qquad \lambda(t) := \lambda_{\min}(\Lambda(t))$$

**Parameter update law**:

$$\dot{\hat{\theta}} = \gamma \sum_{j=1}^{M} \mathcal{Y}_j^\top (\Delta x_j - F_j - \mathcal{Y}_j \hat{\theta} - G_j)$$

where $\gamma \in \mathbb{R}_{>0}$ is an adaptation gain.

**Key bound (Lemma 1)**: Under Assumption 2 ($\theta \in \Theta$, a known convex polytope) and $\hat{\theta}(0) \in \Theta$, the parameter estimation error $\tilde{\theta} = \theta - \hat{\theta}$ satisfies:

$$\|\tilde{\theta}(t)\| \le \nu(t) := \|\tilde{\vartheta}\| \, e^{-\gamma \int_0^t \lambda(\tau)\,d\tau}$$

where $\tilde{\vartheta} \in \mathbb{R}^p$ is the maximum possible estimation error over $\Theta$ (computed via linear programs). If there exists $T$ such that $\lambda(t) > 0$ for all $t > T$, then $\nu(t) \to 0$ exponentially — the parameter estimates converge to their true values.

### HO-RaCBF Definition (Definition 3)

The function $h$ is a **High Order Robust Adaptive Control Barrier Function** of order $r$ for the uncertain system on an open set $D \supset \bigcap_{i=1}^{r} \mathcal{C}_i$ if $h$ has relative degree $r$ and there exists a suitable choice of $\{\alpha_i\}_{i=1}^{r}$ such that for all $x \in D$, $\theta \in \Theta$, and $t \in I$:

$$\sup_{u \in \mathcal{U}} \big\{L_f \psi_{r-1}(x) + L_Y \psi_{r-1}(x)\theta + L_g \psi_{r-1}(x)u\big\} \ge -\alpha_r(\psi_{r-1}(x)) + \|L_Y \psi_{r-1}(x)\| \nu(t)$$

The **robustness buffer** $\|L_Y \psi_{r-1}(x)\| \nu(t)$ accounts for the worst-case parameter realization given the current estimation error bound $\nu(t)$. This buffer:
- Is initially nonzero (accounting for all possible $\theta \in \Theta$)
- Shrinks as concurrent learning identifies the true parameters ($\nu(t) \to 0$)
- Reduces conservatism over time without ever sacrificing safety

### HO-RaCBF-QP Constraint

The set of safe control values is:

$$\hat{K}_{\text{cbf}}(x, \hat{\theta}, t) := \big\{u \in \mathcal{U} \mid L_f \psi_{r-1}(x) + L_Y \psi_{r-1}(x)\hat{\theta} + L_g \psi_{r-1}(x)u + \alpha_r(\psi_{r-1}(x)) - \|L_Y \psi_{r-1}(x)\| \nu(t) \ge 0\big\}$$

The **HO-RaCBF-QP** computes a minimally invasive safe controller:

$$\min_{u \in \mathcal{U}} \frac{1}{2} \|u - k_d(x, \hat{\theta}, t)\|^2$$

$$\text{s.t.} \quad L_f \psi_{r-1}(x) + L_Y \psi_{r-1}(x)\hat{\theta} + L_g \psi_{r-1}(x)u \ge -\alpha_r(\psi_{r-1}(x)) + \|L_Y \psi_{r-1}(x)\| \nu(t)$$

where $k_d$ is a nominal desired policy.

### ES-aCLF Definition (Definition 4)

A continuously differentiable positive definite function $V: \mathbb{R}^n \to \mathbb{R}_{\ge 0}$ is an **Exponentially Stabilizing Adaptive Control Lyapunov Function** for the uncertain system if there exist positive constants $c_1, c_2, c_3 \in \mathbb{R}_{>0}$ such that for all $x \in \mathbb{R}^n$ and $\theta \in \mathbb{R}^p$:

$$c_1 \|x\|^2 \le V(x) \le c_2 \|x\|^2$$

$$\inf_{u \in \mathcal{U}} \big\{L_f V(x) + L_Y V(x)\theta + L_g V(x)u\big\} \le -c_3 V(x)$$

The ES-aCLF update law (extending the CLF component with concurrent learning):

$$\dot{\hat{\theta}} = \Gamma L_Y V(x)^\top + \gamma \Gamma \sum_{j=1}^{M} \mathcal{Y}_j^\top (\Delta x_j - F_j - \mathcal{Y}_j \hat{\theta} - G_j)$$

where $\Gamma \in \mathbb{R}^{p \times p}$ is a positive definite gain matrix.

**Note (Remark 2)**: The parameter update laws for HO-RaCBF and ES-aCLF differ — if combined in a single QP, separate parameter estimates must be maintained, though the same history stack data can be shared.

### ES-aCLF-QP

$$\min_{u \in \mathcal{U}} \frac{1}{2} u^\top u$$

$$\text{s.t.} \quad L_f V(x) + L_Y V(x)\hat{\theta} + L_g V(x)u \le -c_3 V(x)$$

## Theoretical Results

### Safety Theorem (Theorem 2)

Consider the uncertain system, a safe set $\mathcal{C}$ defined by $h$, and let $h$ be a HO-RaCBF on $D$. Provided Assumptions 1--2 hold and the estimated parameters are updated according to the concurrent learning law, then **any controller** $u = k(x, \hat{\theta}, t)$ locally Lipschitz in $(x, \hat{\theta})$ and piecewise continuous in $t$ satisfying $k(x, \hat{\theta}, t) \in \hat{K}_{\text{cbf}}(x, \hat{\theta}, t)$ **renders** $\bigcap_{i=1}^{r} \mathcal{C}_i$ **forward invariant**.

**Proof sketch**: Under the matching condition (Assumption 1), the HOCBF condition for the true system becomes $\psi_r(x, u) = L_f \psi_{r-1} + L_Y \psi_{r-1} \hat{\theta} + L_Y \psi_{r-1} \tilde{\theta} + L_g \psi_{r-1} u + \alpha_r(\psi_{r-1})$. Using $L_Y \psi_{r-1} \tilde{\theta} \ge -\|L_Y \psi_{r-1}\|\|\tilde{\theta}\|$ and the bound $\|\tilde{\theta}\| \le \nu(t)$, enforcing the HO-RaCBF constraint ensures $\psi_r \ge 0$ for all possible $\theta$.

### Asymptotic Stability of the Safe Set (Corollary 1)

If the conditions of Theorem 2 hold and $x(0) \in D \setminus \bigcap_{i=1}^{r} \mathcal{C}_i$, then any controller satisfying $k(x, \hat{\theta}, t) \in \hat{K}_{\text{cbf}}(x, \hat{\theta}, t)$ also guarantees **asymptotic stability** of $\bigcap_{i=1}^{r} \mathcal{C}_i$ (provided the closed-loop dynamics are forward complete). This is a key practical advantage: starting from an unsafe state, the controller drives the system back to safety.

### Stability Theorem (Theorem 3)

Under the conditions of Lemma 2, if there exists a time $T$ and a positive constant $\lambda \in \mathbb{R}_{>0}$ such that $\lambda(t) \ge \lambda$ for all $t \in [T, \infty)$ (finite excitation condition), then for all $t \in [T, \infty)$, the composite state $z = [x^\top, \tilde{\theta}^\top]^\top$ **exponentially converges** to the origin at rate proportional to:

$$\eta_3 := \min\{\gamma\lambda, c_1 c_3\}$$

in the sense that:

$$\|z(t)\| \le \sqrt{\frac{\eta_2}{\eta_1}} \|z(T)\| \, e^{-\frac{\eta_3}{2\eta_2}(t - T)}$$

where $\eta_1 = \min\{c_1, \frac{1}{2}\lambda_{\min}(\Gamma^{-1})\}$ and $\eta_2 = \max\{c_2, \frac{1}{2}\lambda_{\max}(\Gamma^{-1})\}$.

### Feasibility

The QP (11) is feasible at any point where the HO-RaCBF condition holds — by definition, $\sup_{u \in \mathcal{U}} \{\ldots\} \ge 0$ implies the feasible set $\hat{K}_{\text{cbf}}(x, \hat{\theta}, t)$ is nonempty. The robustness buffer $\|L_Y \psi_{r-1}\|\nu(t)$ makes the constraint more restrictive than the nominal HOCBF, which can reduce feasibility; however, as $\nu(t) \to 0$ the constraint relaxes to the nominal HOCBF condition. The ES-aCLF constraint can be relaxed (with a slack variable) to guarantee joint QP feasibility.

## Algorithm / Implementation

### Concurrent Learning for $\theta$ Estimation

1. **Data recording**: Maintain a history stack $\mathcal{H} = \{(t_j, x_j, x_j^-, u_j)\}_{j=1}^{M}$ of $M$ input-output data points sampled along the trajectory. Use singular-value-maximizing algorithms (Chowdhary & Johnson, 2011) to ensure $\lambda(t)$ is nondecreasing.
2. **Parameter update (HO-RaCBF)**: $\dot{\hat{\theta}} = \gamma \sum_{j=1}^{M} \mathcal{Y}_j^\top (\Delta x_j - F_j - \mathcal{Y}_j \hat{\theta} - G_j)$
3. **Parameter update (ES-aCLF)**: $\dot{\hat{\theta}} = \Gamma L_Y V(x)^\top + \gamma \Gamma \sum_{j=1}^{M} \mathcal{Y}_j^\top (\Delta x_j - F_j - \mathcal{Y}_j \hat{\theta} - G_j)$
4. Separate parameter estimates are maintained for HO-RaCBF and ES-aCLF, but share the same history stack data

### $\nu(t)$ Update Law

Compute at each time step:

$$\nu(t) = \|\tilde{\vartheta}\| \, e^{-\gamma \int_0^t \lambda(\tau)\,d\tau}$$

where $\|\tilde{\vartheta}\|$ is precomputed from the convex polytope $\Theta$ via linear programs, and $\lambda(t) = \lambda_{\min}(\Lambda(t))$ is updated when new data is added to the history stack.

### QP Formulation

At each time step, solve:

$$\min_{u \in \mathcal{U}} \frac{1}{2} \|u - k_d(x, \hat{\theta}, t)\|^2$$

$$\text{s.t.} \quad L_f \psi_{r-1}(x) + L_Y \psi_{r-1}(x)\hat{\theta} + L_g \psi_{r-1}(x)u \ge -\alpha_r(\psi_{r-1}(x)) + \|L_Y \psi_{r-1}(x)\| \nu(t)$$

where $k_d$ is obtained from the ES-aCLF-QP (or the two constraints are combined in a single QP with the ES-aCLF constraint relaxed via a slack variable).

## Limitations

1. **Matching condition required (Assumption 1)**: The uncertain parameters must enter at the same relative degree as the control ($L_Y L_f^{i-1} h \equiv 0$ for $i < r$, $L_Y L_f^{r-1} h \neq 0$). If uncertainty appears at a lower relative degree than the control, the resulting constraints on $u$ are no longer affine, breaking the QP framework.
2. **Parametric uncertainty only**: The approach assumes the uncertainty can be expressed as $Y(x)\theta$ with known regression matrix $Y$ and constant unknown parameters $\theta$. It does not handle unmodeled dynamics, state-dependent nonlinear uncertainties, or time-varying parameters.
3. **No unmodeled dynamics / nonparametric uncertainty**: Unlike GP-based approaches, the method cannot account for uncertainty that does not admit a linear-in-parameters representation. The authors identify this as a direction for future work.
4. **Convex polytope assumption (Assumption 2)**: The parameter set $\Theta$ must be a known convex polytope, and initial estimates must satisfy $\hat{\theta}(0) \in \Theta$. This requires prior knowledge of parameter bounds.
5. **Finite excitation requirement**: Exponential convergence of $\nu(t) \to 0$ and the stability guarantees of Theorem 3 require $\lambda(t) > 0$ (sufficiently rich data). Without this, the controller remains safe but may be overly conservative and only achieves asymptotic (not exponential) stability.
6. **Single barrier function**: The formulation as stated applies to safe sets defined by a single barrier function. Multiple constraints can be handled by including multiple CBF constraints in the QP, but formal combination of multiple barrier functions requires additional techniques (smooth min/max, nonsmooth analysis).

## Relevance to RoCBF-Net

### Phase 2 Robustness Injection

The HO-RaCBF formulation is the **primary theoretical inspiration** for the robust HOCBF layer in RoCBF-Net. The core idea of adding a state-dependent robustness buffer to the HOCBF constraint that accounts for model mismatch and shrinks as uncertainty is reduced is directly adopted.

### What We Adopt

- **Robust buffer concept**: The term $\|L_Y \psi_{r-1}(x)\| \nu(t)$ that adds a safety margin to the HOCBF constraint is the direct precursor to our compositional robustness margin $\epsilon(x)$. Both serve the same purpose: guarantee safety under model uncertainty by tightening the CBF constraint.
- **Uncertainty propagation through the $\psi$-chain**: Cohen & Belta's observation that parametric uncertainty affects the highest-order Lie derivative term (under the matching condition) motivates our recursive perturbation analysis through the full $\psi$-chain.
- **QP-based safety filter**: The minimally invasive QP formulation (minimizing deviation from a nominal policy subject to a robust CBF constraint) is structurally identical to our differentiable QP layer.
- **Asymptotic stability of the safe set (Corollary 1)**: The result that the HO-RaCBF controller drives the system back to safety from outside the safe set is a desirable property we aim to preserve.

### What We Modify

- **Replace parametric assumption with GP uncertainty**: Instead of $Y(x)\theta$ with concurrent learning, we model the residual as $\Delta f(x) \sim \mathcal{GP}(\mu_{GP}, k_{GP})$. This removes the matching condition (Assumption 1) and the parametric structure requirement, allowing us to handle general nonlinear model mismatch.
- **Probabilistic vs. deterministic guarantees**: Cohen & Belta achieve deterministic forward invariance because $\nu(t)$ is a deterministic bound on $\|\tilde{\theta}\|$. Our GP-based $\epsilon(x) = \sigma_{\text{total}}(x)$ yields **probabilistic** safety (forward invariance with probability $\ge 1 - \delta$) because the GP posterior variance is a probabilistic bound on the residual.
- **Compositional uncertainty aggregation**: Instead of the single-term buffer $\|L_Y \psi_{r-1}\| \nu(t)$ (which only appears at the highest-order derivative under the matching condition), we compute $\epsilon(x)$ by propagating GP uncertainty through the full $\psi$-chain, accounting for $\Delta f$ perturbation at every level.
- **End-to-end differentiability**: The concurrent learning update law in Cohen & Belta is not differentiable w.r.t. policy parameters. In RoCBF-Net, the GP uncertainty $\epsilon(x)$ is differentiable (via the mean function), enabling gradient backpropagation through the safety layer.
- **Explicit policy distillation**: After training with the differentiable QP safety layer, RoCBF-Net distills the Actor+QP mapping into a standalone network for real-time inference, eliminating the need for online QP solving entirely.

## Cross-References

- [[xiao2021]] — HOCBF definition used as the baseline for high relative degree safety constraints
- [[taylor2020]] — Adaptive CBF (aCBF) for relative degree one; the starting point that Cohen & Belta extend to high relative degree
- [[isaly2021]] — Adaptive safety with multiple barrier functions using integral concurrent learning; shares the concurrent learning technique and the $\nu(t)$ bound
- [[lopez2021]] — Robust adaptive CBFs using set-membership identification; another concurrent-learning-based aCBF for relative degree one
- [[jankovic2018]] — Robust CBFs with deterministic uncertainty bounds; related to the robust buffer idea
- [[xu2015]] — Robustness of CBFs (zeroing CBFs); the theoretical basis for Corollary 1's asymptotic stability result
- [[tan2021]] — High-order barrier functions with robustness considerations; complementary approach to robust HOCBF
- [[parikh2019]] — Integral concurrent learning; the data-driven parameter estimation technique underlying $\nu(t)$
- [[das2024]] — GP-based robust CBF; the alternative uncertainty model that RoCBF-Net adopts instead of parametric concurrent learning

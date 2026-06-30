# RoCBF-Net: Robust Differentiable High-Order CBF for Explicit Safe RL in Energy Systems

**Date**: 2026-05-19
**Status**: Draft
**Target Venue**: IEEE Transactions on Automatic Control (TAC)
**Methodology**: Theory-driven progressive (Phase 1→5)
**Revision**: v8 — fixes from 7th self-review (5 issues)

---

## 1. Project Overview

### 1.1 Core Thesis

Fuse **High-Order Control Barrier Functions (HOCBF)** with **robustness against model mismatch** via Gaussian Process residual learning, and make the entire safety layer **end-to-end differentiable** through a differentiable QP layer. Validate on a supercritical power plant Coordinated Control System (CCS), demonstrating **probabilistic safety guarantees** (forward invariance with probability $\ge 1-\delta$) under parameter drift while achieving microsecond-level inference via explicit policy distillation.

**Key qualifier**: The safety guarantee is **probabilistic, not deterministic**. Under the GP-calibrated uncertainty bound $\epsilon(x)$, the safe set is forward invariant with probability at least $1-\delta$ for a user-specified $\delta > 0$. In practice, with conservative $\beta$ calibration, we target $\delta \le 0.01$ (empirical violation rate $< 1\%$). This distinction is critical and will be stated explicitly throughout the paper.

### 1.2 English Title (Draft)

*Robust Differentiable High-Order Control Barrier Functions for Explicit Safe Reinforcement Learning in Energy and Power Systems*

### 1.3 Key Contributions

1. **Robust HOCBF** (Section 3.1-3.2): HOCBF with compositional GP-based uncertainty compensation that accounts for $\Delta f$ propagation through the full $\psi$-chain, guaranteeing forward invariance with probability $\ge 1-\delta$ under bounded model mismatch
2. **Differentiable QP Layer** (Section 3.3): Implicit differentiation through KKT conditions enables gradient backpropagation from safety constraints to RL policy
3. **Explicit Policy Distillation** (Section 4): Post-training, distill the Actor+QP joint mapping into a standalone neural network via safety-weighted behavior cloning, achieving O(1) inference complexity without online QP solving
4. **CCS Validation** (Section 5-6): First application of Robust Differentiable HOCBF to supercritical boiler-turbine coordinated control, with systematic robustness evaluation under realistic uncertainty scenarios

---

## 2. Mathematical Foundation

This section defines the core mathematical objects and their relationships. All subsequent phases implement these definitions.

### 2.1 System Model

Consider a control-affine system:

$$\dot{x} = f(x) + g(x)u, \quad x \in \mathbb{R}^n, \; u \in \mathcal{U} \subset \mathbb{R}^m$$

where $f: \mathbb{R}^n \to \mathbb{R}^n$ and $g: \mathbb{R}^n \to \mathbb{R}^{n \times m}$ are locally Lipschitz.

**True system vs. nominal model**: The true dynamics include an unknown residual:

$$\dot{x} = \underbrace{f_0(x) + g_0(x)u}_{\text{nominal model}} + \underbrace{\Delta f(x) + \Delta g(x)u}_{\text{unknown residual}}$$

where $f_0, g_0$ are the nominal (design) model, and $\Delta f, \Delta g$ represent model mismatch.

**Assumption 1 ($\Delta g = 0$)**: We assume $\Delta g(x) = 0$ for all $x \in \mathcal{X}$.

**Rationale and scope**:
- In CCS, the **primary uncertainty source** is in the state dynamics: fuel combustion efficiency variations, heat transfer degradation (fouling), and coal quality drift. These affect $f$ but not $g$.
- The control input matrix $g(x)$ encodes the actuator authority mapping (valve opening → flow rate). Under normal operation, this relationship is well-characterized by the actuator calibration curves.
- **When $\Delta g \neq 0$** (valve aging, dead zones, saturation): This falls under **matched uncertainty** — it can be partially absorbed into the control design via robust control techniques (e.g., sliding mode). Extending Robust HOCBF to handle $\Delta g \neq 0$ is identified as **future work** (Section VII of the paper).
- **Practical justification**: CCS literature on robust coordinated control (Tan et al., Fang et al.) consistently treats actuator uncertainty as a secondary effect compared to process uncertainty. Our experiments (Section VI) will verify that the nominal $g_0$ remains adequate under the tested perturbation scenarios.

**Assumption 2 (Bounded $\Delta f$)**: The residual satisfies $|\Delta f_j(x)| \le \bar{\Delta}_j$ for all $x \in \mathcal{X}$, $j = 1, \ldots, n$.

**Note on bound estimation**: The bounds $\bar{\Delta}_j$ are not assumed known a priori. Instead, they are estimated from the GP posterior via the PAC-Bayes guarantee (Section 2.5.3): $\bar{\Delta}_j(x) = \beta \cdot \sigma_{GP,j}(x)$, which satisfies $|\Delta f_j(x) - \mu_{GP,j}(x)| \le \bar{\Delta}_j(x)$ with probability $\ge 1 - \delta/n$ (bounding the posterior residual after mean correction). This replaces the classical assumption of known deterministic bounds with data-driven probabilistic bounds, consistent with our GP methodology. The compositional aggregation (Section 2.5.2) propagates these per-dimension bounds through the $\psi$-chain to obtain $\epsilon(x)$.

### 2.2 High-Order CBF (HOCBF)

**Definition (HOCBF, following Xiao et al. 2021)**: Let $h: \mathbb{R}^n \to \mathbb{R}$ be a continuously differentiable function defining the safe set $\mathcal{C} = \{x : h(x) \ge 0\}$. If the relative degree of $h$ with respect to the system is $m$, define a sequence of functions:

$$\psi_0(x) = h(x)$$
$$\psi_1(x) = \dot{\psi}_0(x) + \alpha_1(\psi_0(x))$$
$$\psi_2(x) = \dot{\psi}_1(x) + \alpha_2(\psi_1(x))$$
$$\vdots$$
$$\psi_{m-1}(x) = \dot{\psi}_{m-2}(x) + \alpha_{m-1}(\psi_{m-2}(x))$$

where $\alpha_i$ ($i=1,\ldots,m-1$) are class-$\mathcal{K}$ functions (we use linear $\alpha_i(r) = k_i r$ with $k_i > 0$). The gains $k_i$ control the responsiveness of the safety filter: larger $k_i$ makes the safe set $\mathcal{C}$ more conservative (the $\psi$-chain tightens faster) but may reduce QP feasibility. For $m=2$ systems, we recommend $k_1 \in [1, 5]$ and $k_2 \in [1, 5]$ as starting points, tuned via grid search on a validation set. For CCS, initial values are $k_1 = 2$, $k_2 = 2$ (moderate conservatism, adjusted in Phase 3 based on feasibility analysis).

**HOCBF constraint**: A function $h$ is an HOCBF of order $m$ if there exists a class-$\mathcal{K}$ function $\alpha_m$ such that for all $x$ with $\psi_i(x) \ge 0$ for all $i = 0, \ldots, m-1$:

$$L_f^m h(x) + L_g L_f^{m-1} h(x) u + \underbrace{\sum_{i=0}^{m-1} L_f^i \big[\alpha_{m-i} \circ \psi_{m-1-i}\big](x)}_{\mathcal{S}(x)} \ge 0 \tag{1}$$

**Standard QP form**: Rearranging (1) into the standard inequality $A(x)u \le b(x)$:

$$\underbrace{-L_g L_f^{m-1} h(x)}_{A(x)} u \le \underbrace{L_f^m h(x) + \mathcal{S}(x)}_{b(x)} \tag{2}$$

since $L_g L_f^{m-1} h(x)$ is a row vector (for a single constraint), $A(x) \in \mathbb{R}^{1 \times m}$ and $b(x) \in \mathbb{R}$. For $p$ simultaneous constraints, $A(x) \in \mathbb{R}^{p \times m}$ and $b(x) \in \mathbb{R}^p$. The equivalence between (1) and (2) is the **sign convention bridge** — the QP constraint $\le$ form is derived directly from the HOCBF $\ge 0$ form by moving $L_g L_f^{m-1} h(x) u$ to the RHS and negating.

**Key property**: The Lie derivative chain $L_f^k h$ requires computing derivatives up to order $m$. For CCS, we expect $m=2$ for main steam pressure constraints (fuel input → energy release → pressure change) and potentially $m=3$ for temperature constraints.

### 2.3 Robust HOCBF

**Challenge**: The Lie derivatives $L_f^k h$ and the intermediate $\psi_i$ functions depend on the true $f$, but we only know $f_0$. With model mismatch $\Delta f$, **all levels of the $\psi$-chain are affected**, not just the highest-order Lie derivative.

#### 2.3.1 Uncertainty Propagation Through the $\psi$-Chain

Define the **nominal $\psi$-chain** (computed with $f_0$):

$$\psi_0^0(x) = h(x), \quad \psi_i^0(x) = L_{f_0} \psi_{i-1}^0(x) + \alpha_i(\psi_{i-1}^0(x))$$

and the **true $\psi$-chain** (with the actual $f = f_0 + \Delta f$):

$$\psi_0(x) = h(x), \quad \psi_i(x) = L_f \psi_{i-1}(x) + \alpha_i(\psi_{i-1}(x))$$

Define the **perturbation at level $i$**: $\delta_i(x) = \psi_i(x) - \psi_i^0(x)$.

**Recursive perturbation analysis** (for linear $\alpha_i(r) = k_i r$):

- $\delta_0(x) = 0$ (since $\psi_0 = \psi_0^0 = h$, independent of $f$)
- $\delta_1(x) = L_{\Delta f} h(x)$ (the residual Lie derivative acts on $h$)
- For $i \ge 2$: $\delta_i(x) = L_{\Delta f} \psi_{i-1}^0(x) + L_{f_0} \delta_{i-1}(x) + k_i \delta_{i-1}(x) + \mathcal{O}(\delta_{i-1}^2)$

The last equation uses the decomposition $L_f \psi_{i-1} = L_{f_0} \psi_{i-1} + L_{\Delta f} \psi_{i-1}$ and $\psi_{i-1} = \psi_{i-1}^0 + \delta_{i-1}$, with the cross term $L_{\Delta f} \delta_{i-1}$ absorbed into $\mathcal{O}(\delta^2)$ (second-order in the perturbation, negligible when $\Delta f$ is small relative to $f_0$).

**Assumption 3 (Small perturbation)**: The first-order approximation is valid when $\|\Delta f\| / \|f_0\| \le \rho_{max}$ in the operating region, where $\rho_{max}$ is the maximum relative perturbation. The $\mathcal{O}(\delta^2)$ term is bounded by $c \cdot \rho_{max}^2 \cdot \|f_0\|^2$ for a system-dependent constant $c$ (related to the Lipschitz constant of $f_0$ and $\nabla \psi_i^0$). For CCS, the maximum perturbation is 30% (coal quality drift), giving $\rho_{max} \le 0.3$. The second-order residual is then bounded by $c \cdot 0.09 \cdot \|f_0\|^2$, which is typically an order of magnitude smaller than the first-order term. We verify this empirically in Phase 2 by comparing the first-order bound $\epsilon(x)$ against the true perturbation $\Delta(x, u)$: if the ratio $\epsilon(x)/|\Delta(x,u)|$ remains in $[1, 2]$ for $\ge 90\%$ of states, the first-order approximation is adequate.

**First-order perturbation bound** (dropping $\mathcal{O}(\delta^2)$):

$$|\delta_i(x)| \le |L_{\Delta f} \psi_{i-1}^0(x)| + (|L_{f_0}|_{\text{op}} + k_i) \cdot |\delta_{i-1}(x)| \tag{3}$$

where $|L_{f_0}|_{\text{op}}$ denotes the local operator norm (Lipschitz constant) of the mapping $\phi \mapsto L_{f_0} \phi$ restricted to the function space spanned by $\psi_{i-1}^0$ and its perturbation $\delta_{i-1}$. Concretely, for a differentiable function $\phi: \mathbb{R}^n \to \mathbb{R}$, the Lie derivative $L_{f_0} \phi(x) = \nabla \phi(x)^\top f_0(x)$ is Lipschitz in $\phi$ with constant $|L_{f_0}|_{\text{op}} = \sup_{x \in \mathcal{X}} \|f_0(x)\| \cdot L_{\nabla \phi}$, where $L_{\nabla \phi}$ is the Lipschitz constant of $\nabla \phi$. In practice, we estimate $|L_{f_0}|_{\text{op}}$ numerically by evaluating $\|f_0(x)\| \cdot \|\nabla^2 \psi_{i-1}^0(x)\|_F$ over a grid of operating states and taking the maximum.

#### 2.3.2 Compositional Robustness Margin $\epsilon(x)$

The true HOCBF constraint is:

$$L_f^m h + L_g L_f^{m-1} h \cdot u + \mathcal{S}(x) \ge 0$$

The nominal HOCBF constraint (computed with $f_0$) is:

$$L_{f_0}^m h + L_g L_{f_0}^{m-1} h \cdot u + \mathcal{S}_0(x) \ge 0$$

The **total constraint perturbation** is $\Delta(x, u) = $ (true constraint) $-$ (nominal constraint). Under $\Delta g = 0$ and linear $\alpha_i$:

$$\Delta(x, u) = \underbrace{[L_f^m h - L_{f_0}^m h]}_{\text{highest-order Lie derivative gap}} + \underbrace{[L_g L_f^{m-1} h - L_g L_{f_0}^{m-1} h] \cdot u}_{\text{$\Delta f$-induced coupling coefficient gap}} + \underbrace{[\mathcal{S}(x) - \mathcal{S}_0(x)]}_{\text{accumulated }\psi\text{-chain gap}} \tag{4}$$

Since $\mathcal{S} - \mathcal{S}_0$ depends on $\delta_i$ at all levels, and the Lie derivative gaps also depend on the $\delta_i$ perturbations, we express the total bound recursively.

**Definition (Compositional robustness margin)**:

$$\epsilon(x) = \sigma_{\text{total}}(x) \tag{5}$$

where $\sigma_{\text{total}}(x)$ aggregates GP uncertainties through the $\psi$-chain (Section 2.5.2). Note that $\beta$ is already incorporated in each $\sigma_i$ via eqs (10)-(11), so it is NOT applied again at this level — applying $\beta$ twice would make $\epsilon$ too conservative by a factor of $\beta \approx 2\text{-}4$.

**Robust HOCBF constraint** (in $\ge 0$ form):

$$L_{f_0}^m h(x) + L_g L_{f_0}^{m-1} h(x) u + \mathcal{S}_0(x) - \epsilon(x) \ge 0 \tag{6}$$

**Sign convention rationale**: The robust constraint subtracts $\epsilon(x)$ to make the nominal requirement *more conservative*. Intuitively, the true constraint value equals $C_0(x,u) + \Delta(x,u)$, where $C_0$ is the nominal value and $|\Delta| \le \epsilon$. The worst case is $\Delta = -\epsilon$ (perturbation reduces the constraint value), so we require $C_0 - \epsilon \ge 0$ to ensure $C_0 + \Delta \ge 0$ even in the worst case. This convention is consistent with Taylor et al. (2020, ISSf-CBF) and Jankovic (2018). The opposite sign ($+\epsilon$) would *relax* the constraint, which is unsafe.

**Robust HOCBF constraint** (in QP $\le$ form):

$$-L_g L_{f_0}^{m-1} h(x) \cdot u \le L_{f_0}^m h(x) + \mathcal{S}_0(x) - \epsilon(x) \tag{7}$$

i.e., $A_0(x) u \le b_0(x) - \epsilon(x)$, where $A_0 = -L_g L_{f_0}^{m-1} h$ and $b_0 = L_{f_0}^m h + \mathcal{S}_0$. **After GP mean correction** (§2.5.1), these become $\hat{A}_0 = -L_g L_{\hat{f}}^{m-1} h$ and $\hat{b}_0 = L_{\hat{f}}^m h + \hat{\mathcal{S}}_0$; the QP formulation (§2.4) uses $\hat{A}_0, \hat{b}_0$ throughout.

#### 2.3.3 Probabilistic Safety Guarantee

**Theorem 3 (Forward invariance under uncertainty)**: Under Assumptions 1-2, if the compositional robustness margin $\epsilon(x) = \sigma_{\text{total}}(x)$ constructed via (10)-(12) satisfies $\epsilon(x) \ge |\Delta(x, u)|$ for all $u$ in the feasible set $\mathcal{U}$ and all $x$ in the operating region, then the safe set $\mathcal{C}$ is forward invariant with probability $\ge 1 - \delta$, where $\delta$ is determined by the GP calibration quality and $\beta$ (which is embedded in each $\sigma_i$ via eqs (10)-(11), so $\epsilon = \sigma_{\text{total}}$ already accounts for $\beta$). Note that $\epsilon(x)$ bounds the perturbation for *any* feasible $u$ (using $u_{max} = \sup_{u \in \mathcal{U}} \|u\|$ in eq (12)), so the condition $\epsilon(x) \ge |\Delta(x, u^*(x))|$ for the actual QP solution $u^*(x) \in \mathcal{U}$ is automatically satisfied.

**Proof sketch**: The proof proceeds in three steps:
1. **GP calibration**: By the PAC-Bayes concentration inequality (Srinivas et al., 2010), $|\Delta f_j(x) - \mu_{GP,j}(x)| \le \beta \cdot \sigma_{GP,j}(x)$ holds simultaneously for all $x$ with probability $\ge 1 - \delta/n$ (where $\mu_{GP,j}$ is the GP posterior mean, consistent with the mean-corrected formulation in §2.5.1).
2. **Uncertainty propagation**: By the recursive bound (3), the per-level perturbations $|\delta_i(x)|$ are bounded by functions of the GP uncertainties $\sigma_{GP,j}(x)$, which in turn bound $\sigma_{\text{total}}(x)$.
3. **Forward invariance**: When $\epsilon(x) \ge |\Delta(x, u^*(x))|$ holds, the nominal constraint minus $\epsilon$ accounts for the worst-case perturbation, so the true constraint (1) is satisfied whenever (6) is satisfied (since $C_0 - \epsilon \ge 0$ and $|\Delta| \le \epsilon$ imply $C_0 + \Delta \ge 0$). The standard HOCBF forward invariance result then applies.

The complete proof will be provided in the paper, with explicit construction of the recursive bounds and the $\sigma_{\text{total}}$ aggregation.

#### 2.3.4 Special Case: $m = 2$ (Double Integrator and CCS Pressure Constraint)

For relative degree $m = 2$ (the primary case in this work), the perturbation analysis simplifies:

- $\delta_0 = 0$
- $\delta_1(x) = L_{\Delta f} h(x) = \nabla h(x)^\top \Delta f(x)$
- The constraint perturbation is:

$$\Delta(x, u) = \underbrace{\nabla(\nabla h^\top f_0)^\top \Delta f + (\nabla h^\top \Delta f)^\top \nabla f_0 + \text{h.o.t.}}_{\text{approx } L_f^2 h - L_{f_0}^2 h} + \underbrace{(\nabla h^\top \Delta f)^\top g \cdot u}_{\text{$\Delta f$-induced coupling coefficient gap}} + k_2 \delta_1(x)$$

For the double integrator with $h(x) = x^2 - r^2$:
- $\delta_1 = 2x \cdot \Delta f_1(x,v)$ (where $\Delta f_1$ is the position-component residual)
- Under the first-order approximation (Assumption 3), dropping $\mathcal{O}(\delta^2)$ terms:

$$\Delta(x, u) \approx 2 \Delta f_1 \cdot v + 2 \Delta f_1 \cdot u + k_2 \cdot 2x \cdot \Delta f_1$$

where the $2x \cdot (\partial \Delta f_1 / \partial v) \cdot \Delta f_2$ term and the $2\Delta f_1^2/x$ term from the full expansion are both $\mathcal{O}(\delta^2)$ (second-order in $\Delta f$) and are dropped under the first-order approximation. The $(\partial \Delta f_1 / \partial v) \cdot \Delta f_2$ term is second-order because it involves a product of $\Delta f$ with a derivative of $\Delta f$ — while the recursive bound (3) propagates first-order perturbations through the $\psi$-chain, any term that involves $\nabla(\Delta f) \cdot \Delta f$ falls outside the first-order framework. Similarly, the $\Delta f_1^2/x$ term is undefined at $x=0$, confirming it must be excluded.

In practice, the dominant terms are the linear ones in $\Delta f$, and $\epsilon(x)$ needs to bound their aggregate effect.

### 2.4 Differentiable QP Layer

**QP formulation**: At each time step, the safety-corrected control is obtained by solving:

$$u^* = \arg\min_{u} \frac{1}{2} \|u - u_{RL}\|^2 \quad \text{s.t.} \quad \hat{A}_0(x)u \le \hat{b}_0(x) - \epsilon(x) \tag{8}$$

where $u_{RL} = \pi_{actor}(x)$ is the RL policy's raw output, $\hat{A}_0 = -L_g L_{\hat{f}}^{m-1} h$ and $\hat{b}_0 = L_{\hat{f}}^m h + \hat{\mathcal{S}}_0$ are computed using the **mean-corrected** $\hat{f} = f_0 + \mu_{GP}$ (see §2.5.1), and $\epsilon(x)$ is the robustness margin from (5). Before GP training (Phase 1), $\mu_{GP} = 0$ and $\hat{A}_0 = A_0$, $\hat{b}_0 = b_0$ reduce to the nominal model.

**Sign convention**: The QP constraint $\hat{A}_0 u \le \hat{b}_0 - \epsilon$ is equivalent to the Robust HOCBF constraint $L_{\hat{f}}^m h + L_g L_{\hat{f}}^{m-1} h \cdot u + \hat{\mathcal{S}}_0 - \epsilon \ge 0$ via (7). The subtraction of $\epsilon$ tightens the QP feasible set, ensuring safety even under worst-case perturbation.

**Implicit differentiation via KKT**: The QP solution $u^*$ satisfies the KKT conditions:

$$u^* - u_{RL} + \hat{A}_0(x)^\top \lambda^* = 0$$
$$\text{diag}(\lambda^*) (\hat{A}_0(x)u^* - \hat{b}_0(x) + \epsilon(x)) = 0$$
$$\hat{A}_0(x)u^* \le \hat{b}_0(x) - \epsilon(x), \quad \lambda^* \ge 0$$

where $\lambda^*$ are the dual variables. The gradient $\partial u^* / \partial \theta$ (where $\theta$ are Actor parameters) is obtained by differentiating through the KKT system:

$$\begin{bmatrix} I & \hat{A}_0^\top \\ \text{diag}(\lambda^*)\hat{A}_0 & \text{diag}(\hat{A}_0 u^*-\hat{b}_0+\epsilon) \end{bmatrix} \begin{bmatrix} \partial u^*/\partial\theta \\ \partial\lambda^*/\partial\theta \end{bmatrix} = \begin{bmatrix} \partial u_{RL}/\partial\theta \\ 0 \end{bmatrix} \tag{9}$$

This linear system is solvable when the QP is strictly complementary (non-degenerate), which we ensure via perturbation regularization (add $\gamma > 0$ to the diagonal of the bottom-right block).

### 2.5 GP Residual Learning

#### 2.5.1 Per-Dimension GP Model

**Residual definition**: $\Delta f(x) = f_{true}(x) - f_0(x)$, where $f_{true}$ is the unknown true dynamics and $f_0$ is the nominal model.

**Independent GP per dimension**: We place an independent GP prior on each component of $\Delta f$:

$$\Delta f_j(x) \sim \mathcal{GP}\big(0, k(x, x')\big), \quad j = 1, \ldots, n$$

with Matérn-5/2 kernel $k$. Given a dataset $\mathcal{D} = \{(x_i, \Delta \hat{f}_i)\}_{i=1}^N$ (residuals estimated from state transitions), the GP posterior provides for each dimension $j$:

- **Mean**: $\mu_{GP,j}(x) = k_*(x)^\top (K + \sigma_n^2 I)^{-1} \mathbf{y}_j$
- **Variance**: $\sigma_{GP,j}^2(x) = k(x,x) - k_*(x)^\top (K + \sigma_n^2 I)^{-1} k_*(x)$

**GP mean correction (mean-adjusted residual)**: The PAC-Bayes concentration inequality (Srinivas et al., 2010) bounds the *posterior residual*, not the raw residual:

$$|\Delta f_j(x) - \mu_{GP,j}(x)| \le \beta \cdot \sigma_{GP,j}(x) \quad \text{w.p. } \ge 1 - \delta/n$$

We leverage this by defining the **mean-corrected nominal model**:

$$\hat{f}(x) = f_0(x) + \mu_{GP}(x)$$

The remaining uncertainty is $\Delta \hat{f}_j(x) = \Delta f_j(x) - \mu_{GP,j}(x)$, bounded by $\beta \cdot \sigma_{GP,j}(x)$. The Robust HOCBF uses $\hat{f}$ (instead of $f_0$) for computing the $\psi$-chain, and $\epsilon(x)$ only needs to bound the *remaining* uncertainty $\Delta \hat{f}$, which is typically much smaller than $\Delta f$. This makes $\epsilon$ tighter and the QP less conservative.

**Implementation**: In the QP, replace $A_0, b_0$ (computed with $f_0$) with $\hat{A}_0, \hat{b}_0$ (computed with $\hat{f} = f_0 + \mu_{GP}$). The $\epsilon(x)$ computation uses the same formulas (10)-(12) but with $\hat{\psi}_i^0$ (the $\psi$-chain computed with $\hat{f}$) instead of $\psi_i^0$ (computed with $f_0$). Since $|\Delta \hat{f}_j| \le \beta \sigma_{GP,j}$ and $|\Delta f_j|$ could be much larger, the mean correction significantly reduces $\epsilon$.

**Data collection strategy**: GP training data is collected in two phases:
1. **Offline pre-training phase**: Run $N_{pre}$ episodes with a random exploratory policy, record $(x_t, x_{t+1}, u_t)$ tuples, compute residuals $\Delta \hat{f}(x_t) = \frac{x_{t+1} - x_t}{\Delta t} - f_0(x_t) - g_0(x_t)u_t$
2. **Online update phase**: During RL training, periodically (every $M$ episodes) update the GP with new residual data from the current policy's trajectories. Use incremental Cholesky updates to avoid full recomputation.

**GP hyperparameter optimization**: The Matérn-5/2 kernel has two hyperparameters per dimension: length scale $\ell_j$ and signal variance $\sigma_{s,j}^2$, plus the observation noise variance $\sigma_n^2$ (shared across dimensions for simplicity). Optimization strategy:
- **Method**: Type-II maximum likelihood (marginal likelihood optimization) via L-BFGS-B, maximizing $\log p(\mathbf{y}_j | X, \theta_j)$ for each dimension $j$ independently
- **Frequency**: Hyperparameters are re-optimized at each GP update epoch (every $M=50$ episodes), jointly with the posterior update. This is feasible because marginal likelihood evaluation is $O(N^2)$ with pre-computed Cholesky factors, and L-BFGS-B typically converges in 20-50 iterations
- **Initialization**: $\ell_j$ initialized to the median heuristic (median of pairwise distances in the current dataset); $\sigma_{s,j}^2$ initialized to the empirical variance of $\Delta \hat{f}_j$; $\sigma_n^2$ initialized to $10^{-4}$
- **Constraints**: $\ell_j \in [10^{-2}, 10^2]$ (log-space bounds prevent degenerate kernels); $\sigma_{s,j}^2 \in [10^{-6}, 10^2]$; $\sigma_n^2 \in [10^{-6}, 10^{-2}]$
- **Sparse GP**: When using inducing points ($Z=200$ for CCS), hyperparameters are optimized jointly with inducing point locations via the variational lower bound (Hensman et al., 2013). Inducing points initialized via k-means on the current dataset

#### 2.5.2 Aggregation: From Per-Dimension GP to Compositional $\sigma_{\text{total}}(x)$

The key question: each GP gives $\sigma_{GP,j}(x)$ for $\Delta \hat{f}_j$ (the posterior residual), but $\epsilon(x)$ must bound the **total effect** of $\Delta \hat{f}$ on the HOCBF constraint through the entire $\psi$-chain.

**Level-1 aggregation** ($\delta_1 = L_{\Delta \hat{f}} h = \nabla h^\top \Delta \hat{f}$):

Under the independent GP assumption, the aggregate uncertainty for $\delta_1$ is:

$$\sigma_1(x) = \beta \sqrt{\sum_{j=1}^{n} \left(\frac{\partial h}{\partial x_j}\right)^2 \sigma_{GP,j}^2(x)} \tag{10}$$

This is the standard error propagation formula: if $\Delta \hat{f}_j$ are independent with uncertainty $\beta \sigma_{GP,j}$ (bounding $|\Delta f_j - \mu_{GP,j}|$), then the uncertainty of their linear combination $\nabla h^\top \Delta \hat{f}$ is given by (10).

**Recursive aggregation** ($i \ge 2$): For the perturbation at level $i$:

$$\sigma_i(x) = \beta \sqrt{\sum_{j=1}^{n} \left(\frac{\partial \hat{\psi}_{i-1}^0}{\partial x_j}\right)^2 \sigma_{GP,j}^2(x)} + (|L_{\hat{f}}|_{\text{op}} + k_i) \cdot \sigma_{i-1}(x) \tag{11}$$

where $\hat{\psi}_{i-1}^0$ is the $\psi$-chain computed with the mean-corrected model $\hat{f} = f_0 + \mu_{GP}$ (see §2.5.1), and $|L_{\hat{f}}|_{\text{op}}$ is the corresponding operator norm. The first term bounds $|L_{\Delta \hat{f}} \hat{\psi}_{i-1}^0|$ (same error propagation as (10)), and the second term propagates the uncertainty from level $i-1$ (from the recursive bound (3)).

**Total aggregation** ($\sigma_{\text{total}}$): The constraint perturbation (4) includes contributions from all levels plus the highest-order Lie derivative gap. The total bound is:

$$\sigma_{\text{total}}(x) = \sigma_m(x) + \sum_{j=1}^{m-1} c_j \cdot \sigma_j(x) + \sigma_{\text{ctrl}}(x) \tag{12}$$

where:
- $\sigma_m(x)$ bounds $|L_{\Delta \hat{f}}^m h - L_{\hat{f}}^m h|$ (highest-order gap, using (11) with $i=m$)
- $c_j = \prod_{i=j+1}^{m-1} (|L_{\hat{f}}|_{\text{op},i} + k_i)$ is the accumulated gain from level $j$ to level $m-1$ through the recursive bound (3). For $m=2$ (our primary case), there is only $c_1 = 1$ (no recursion). For $m=3$, $c_1 = |L_{\hat{f}}|_{\text{op},2} + k_2$ and $c_2 = 1$.
- $\sigma_{\text{ctrl}}(x) = \beta \sqrt{\sum_{j} (\partial (L_g L_{\hat{f}}^{m-1} h) / \partial x_j)^2 \sigma_{GP,j}^2(x)} \cdot u_{max}$ bounds the $\Delta \hat{f}$-induced coupling coefficient gap (under $\Delta g = 0$, this is the effect of $\Delta \hat{f}$ on $L_g L_{\hat{f}}^{m-1} h$). Here $u_{max} = \sup_{u \in \mathcal{U}} \|u\|$ is the maximum control norm over the feasible set, which is known from the actuator saturation limits (e.g., $u_{max} = \sqrt{m}$ for $\mathcal{U} = [-1,1]^m$). This ensures $\epsilon(x) \ge |\Delta(x, u)|$ for *all* $u \in \mathcal{U}$, consistent with Theorem 3's condition.

**Practical computation**: All partial derivatives $\partial h / \partial x_j$, $\partial \hat{\psi}_i^0 / \partial x_j$, and the Lie derivative coefficients are computed via JAX autodiff (`jax.grad`, `jax.jacfwd`). The GP variances $\sigma_{GP,j}^2(x)$ are evaluated at the current state. The entire aggregation is a deterministic function of $x$ and the GP posterior, amenable to `jax.jit` compilation.

**Multi-constraint extension**: When multiple HOCBF constraints $h_1, \ldots, h_p$ exist simultaneously (e.g., CCS has $p=2$), each constraint has its own compositional robustness margin $\epsilon_i(x)$ computed independently using (10)-(12) with its own $\nabla h_i$, $\hat{\psi}_k^{0,i}$ chain, and relative degree $m_i$. The GP posterior variances $\sigma_{GP,j}^2(x)$ are shared across all constraints — the same $n$ per-dimension GPs provide uncertainty for all $\epsilon_i$ computations, with different Jacobian weights $(\partial h_i / \partial x_j)$ and $(\partial \hat{\psi}_k^{0,i} / \partial x_j)$ per constraint. In the QP, this yields $p$ separate Robust HOCBF rows:

$$\hat{A}_0^{(i)}(x) u \le \hat{b}_0^{(i)}(x) - \epsilon_i(x), \quad i = 1, \ldots, p$$

where $\hat{A}_0^{(i)} = -L_g L_{\hat{f}}^{m_i-1} h_i$ and $\hat{b}_0^{(i)} = L_{\hat{f}}^{m_i} h_i + \hat{\mathcal{S}}_{0,i}$ (using the mean-corrected $\hat{f} = f_0 + \mu_{GP}$).

#### 2.5.3 PAC-Bayes $\beta$ Calibration

$$\beta = \sqrt{2 \ln\big(\frac{nN^{1/2}}{\delta}\big)} \tag{13}$$

This $\beta$ follows from the PAC-Bayes concentration inequality for GP (Srinivas et al., 2010), guaranteeing that $|\Delta f_j(x) - \mu_{GP,j}(x)| \le \beta \cdot \sigma_{GP,j}(x)$ with probability $\ge 1-\delta/n$ for all $x$ in the compact domain. By the union bound, all $n$ dimensions satisfy their bounds simultaneously with probability $\ge 1 - \delta$. Note that the PAC-Bayes bound applies to the **posterior residual** $\Delta f_j - \mu_{GP,j}$ (not the raw residual $\Delta f_j$), consistent with the mean-corrected formulation in §2.5.1.

**Addressing the i.i.d. assumption**: The standard PAC-Bayes bound assumes i.i.d. data, but RL trajectories are time-correlated and non-stationary. We handle this via **$\beta$-mixing subsampling**:
- If the Markov chain induced by the policy is uniformly ergodic with mixing time $\tau_{mix}$, then subsampling every $\tau_{mix}$ steps yields approximately i.i.d. samples.
- In practice, we estimate $\tau_{mix}$ from the autocorrelation of the residual sequence, and subsample accordingly before GP training.
- Alternatively, the **batch acquisition protocol** (collect data from multiple independent episodes with different random seeds) provides i.i.d. samples across episodes. Since each episode starts from a random initial state, the first transition of each episode is i.i.d.
- For our experiments, we use the batch protocol: GP data consists of the first $K_{batch}$ transitions from each of $M_{episodes}$ episodes, ensuring i.i.d. coverage.

**Conservative choice**: For the experiments, we use $\delta = 0.01$ and report the empirical coverage. The $\beta$ value is fixed during each GP update epoch (not recomputed per step).

#### 2.5.4 Two-Timescale Protocol for GP Updates

**Problem**: GP online updates change $\epsilon(x) \to$ change the QP constraint surface $\to$ change the gradient paths for RL $\to$ non-stationary optimization landscape.

**Resolution**: Two-timescale separation.

- **Fast timescale (RL)**: Actor-critic updates every episode. The QP constraint $\hat{A}_0(x)u \le \hat{b}_0(x) - \epsilon(x)$ uses the **current fixed** $\epsilon(x)$ from the most recent GP snapshot.
- **Slow timescale (GP)**: GP is updated every $M = 50$ episodes. During a GP update:
  1. Collect new residual data from the most recent $M$ episodes
  2. Update the GP posterior (incremental Cholesky)
  3. Recompute $\beta$ via (13) with the new $N$
  4. $\epsilon(x)$ changes for subsequent RL episodes
- **Re-adaptation period**: After each GP update, allow $K = 10$ episodes for the RL policy to re-adapt to the new constraint surface. During this period, reduce the learning rate by $0.5\times$ to prevent destabilization. These $K$ episodes are counted within the next $M$-episode cycle (not in addition to it), so the effective schedule is: [GP update at episode $t$] → [10 re-adaptation episodes with $0.5\times$ LR] → [40 normal episodes] → [GP update at episode $t+M$].
- **Convergence argument**: As the GP accumulates more data, the posterior variance $\sigma_{GP,j}(x)$ decreases, but the PAC-Bayes factor $\beta$ increases with $N$ (eq (13)). Since $\epsilon(x) = \sigma_{\text{total}}(x)$ and each $\sigma_i$ contains a factor of $\beta$ (from eqs (10)-(11)), the net effect on $\epsilon(x)$ is not monotonic in general. However, in the asymptotic regime, $\sigma_{GP,j}(x) = O(N^{-1/2})$ (standard GP contraction rate) while $\beta = O(\sqrt{\ln N})$, so each $\sigma_i(x) = O(\sqrt{\ln N / N})$ and therefore $\epsilon(x) = O(\sqrt{\ln N / N}) \to 0$ as $N \to \infty$. We revise the convergence claim: $\epsilon(x)$ is **non-increasing in the asymptotic regime** (for sufficiently large $N$), and the RL optimization landscape changes less with each update, ensuring convergence in the limit. This is analogous to two-timescale stochastic approximation (Borkar, 2008), where the slower timescale's effect on the faster timescale vanishes asymptotically.

**Scalability for CCS**: As training progresses, the GP dataset grows ($N \sim 10^4$-$10^5$ transitions), making exact GP ($O(N^3)$) infeasible. Use **sparse GP** with $Z$ inducing points ($Z \ll N$), reducing complexity from $O(N^3)$ to $O(NZ^2)$. Target $Z = 200-500$ for CCS (5-dim input space). Inducing points selected via k-means clustering on collected data.

### 2.6 Explicit Policy Distillation

**Goal**: Replace the runtime Actor+QP pipeline with a single forward-pass network $\pi_{explicit}(x; \phi)$ for deployment.

**Distillation data**: Generate from the trained Actor+QP system:
- Collect trajectories $\{(x_t, u_t^* = \text{QP}(\pi_{actor}^*(x_t), x_t))\}$ along with **next states** $x_{t+1}$ from the environment
- This requires **environment interaction** during distillation (not purely offline), but only for data collection (no gradient through the environment)

**Distillation loss**:

$$\mathcal{L}_{distill}(\phi) = \mathbb{E}_{(x, u^*) \sim \mathcal{D}_{distill}} \big[ \| \pi_{explicit}(x; \phi) - u^* \|^2 \big] + \lambda_{safe} \cdot \mathbb{E}_{(x, x') \sim \mathcal{D}_{distill}} \big[ \sum_{i=1}^{p} \mathbb{1}[h_i(x') < 0] \cdot |h_i(x')|^2 \big] + \lambda_{margin} \cdot \mathbb{E}_{x \sim \mathcal{D}_{distill}} \big[ \sum_{i=1}^{p} \max(0, \epsilon_i(x) - C_i(x, \pi_{explicit}(x)))^2 \big] \tag{14}$$

where:
- **First term**: Standard behavior cloning MSE loss — the student matches the teacher's safe actions
- **Second term**: Safety penalty — penalizes any state transition where any safe set is exited. The sum is over $p$ HOCBF constraints (CCS has $p=2$). $x'$ is the next state from the **teacher's trajectory** (not a fresh rollout), so this term is computable from the distillation dataset without additional environment interaction
- **Third term**: Safety margin regularization — penalizes the student if its action would violate any HOCBF constraint with margin $\epsilon_i(x)$. Here $C_i(x, u) = L_{\hat{f}}^{m_i} h_i + L_g L_{\hat{f}}^{m_i-1} h_i \cdot u + \hat{\mathcal{S}}_{0,i}(x)$ is the mean-corrected constraint function value for constraint $i$, and the margin $\epsilon_i(x)$ ensures the student maintains at least the robust safety margin for each constraint. This term is **non-zero even on teacher data** (unlike the second term which is zero when the teacher is safe), providing a continuous gradient signal for safety.
- $\lambda_{safe} \gg 1$ (e.g., $\lambda_{safe} = 100$) ensures safety dominates when violations occur; $\lambda_{margin} = 10$ provides continuous safety-aware gradient

**Note on the safety terms**: The second term $\sum_i \mathbb{1}[h_i(x') < 0]$ is zero on teacher data (since the teacher is safe), providing only a "guard rail" against severe deviations. The third (margin) term is the primary safety-aware training signal — it directly penalizes the student when its action at state $x$ would produce any constraint value $C_i(x, \pi_{explicit}(x))$ below the robustness margin $\epsilon_i(x)$. This is computable without environment interaction (it depends only on $x$ and $\pi_{explicit}(x)$, not on $x'$).

**Post-distillation safety verification**: Run $N_{verify} = 100$ episodes with $\pi_{explicit}$; only accept if violation rate = 0%. If violations occur, increase $\lambda_{safe}$ by $10\times$ and re-distill. Maximum 3 re-distillation rounds; if still failing, analyze failure modes and adjust the student architecture.

**Theoretical safety guarantee for distillation**: Beyond empirical verification, we provide a formal argument. Let $\varepsilon_{distill} = \max_x \|\pi_{explicit}(x) - u^*(x)\|$ be the maximum distillation error (estimated from the distillation dataset). For each HOCBF constraint $i$, if the constraint function $C_i(x, u) = L_{\hat{f}}^{m_i} h_i + L_g L_{\hat{f}}^{m_i-1} h_i \cdot u + \hat{\mathcal{S}}_{0,i}$ is $L_{C_i}$-Lipschitz in $u$, then the constraint violation induced by distillation error is at most $L_{C_i} \cdot \varepsilon_{distill}$. If the teacher's constraint margin $C_i(x, u^*(x)) \ge \eta_i > 0$ (positive margin), then the student satisfies constraint $i$ whenever $\varepsilon_{distill} < \eta_i / L_{C_i}$. We estimate $L_{C_i}$ analytically (it equals $\|L_g L_{\hat{f}}^{m_i-1} h_i\|$) and verify $\varepsilon_{distill} < \min_i (\eta_i / L_{C_i})$ empirically. This provides a deterministic guarantee conditional on the estimated $\varepsilon_{distill}$, strengthening the purely empirical 100-episode verification.

**Architecture of $\pi_{explicit}$**: 3-layer MLP with 128-64-$m$ units, ReLU activations. This is intentionally compact for edge deployment (total parameters < 10K, enabling <1ms inference on CPU).

---

## 3. Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Core framework | JAX + Flax NNX | XLA compilation, `jit`/`vmap`, GPU parallelism on RTX 4090 |
| RL algorithms | PPO + SAC | Dual baselines; pure JAX/Flax implementation |
| Safety control | Robust HOCBF + Differentiable QP | JAX native autodiff for implicit differentiation |
| QP solver | qpax (JAX-native) | Differentiable QP with JAX autodiff integration |
| Uncertainty | GP residual learning (sparse GP) | Tight probabilistic uncertainty bounds; JAX implementation |
| Simulation | Pure JAX/NumPy | Custom CCS model + low-order validation environments |
| Figures | nature-figure (nature-skills) | Nature-grade matplotlib charts |
| Polishing | nature-polishing (nature-skills) | 25-rule academic English refinement |
| Hardware | Single RTX 4090 | Maximize JAX parallel training efficiency |

---

## 4. Repository Structure

```
RoCBF-Net/
├── rocbf/                       # Core library
│   ├── cbf/
│   │   ├── hocbf.py             # HOCBF: recursive construction, Lie derivatives via JAX autodiff
│   │   └── robust_hocbf.py      # Robust HOCBF: compositional ε from GP variance
│   ├── qp/
│   │   └── diff_qp.py           # Differentiable QP layer (qpax + KKT implicit diff)
│   ├── gp/
│   │   ├── gp_residual.py       # GP residual learning (JAX, Matérn-5/2 kernel)
│   │   └── sparse_gp.py         # Sparse GP with inducing points (for CCS scalability)
│   ├── rl/
│   │   ├── ppo.py               # PPO (Flax NNX)
│   │   └── sac.py               # SAC (Flax NNX, implemented in Phase 4)
│   └── policy/
│       ├── safe_policy.py       # Actor + QP projection wrapper (training-time)
│       └── distill.py           # Policy distillation: Actor+QP → explicit network
├── envs/                        # Simulation environments
│   ├── safe_navigation/         # Phase 1-2: Double integrator (JAX)
│   │   ├── dynamics.py          # Double integrator: ẋ=v, v̇=u
│   │   ├── constraints.py       # Circular/elliptical keep-out zones
│   │   └── env.py               # Gymnasium-compatible interface
│   └── ccs/                     # Phase 3: Supercritical CCS (JAX)
│       ├── dynamics.py           # Boiler-turbine coordinated dynamics (RK4)
│       ├── constraints.py        # Main steam pressure bounds, rate limit constraints
│       ├── uncertainty.py        # Coal quality drift + fouling simulation
│       └── env.py
├── experiments/                 # Experiment configs and scripts
│   ├── phase1_validation/
│   ├── phase2_robustness/
│   ├── phase3_ccs/
│   └── phase4_full/
├── baselines/                   # Baseline implementations
│   ├── pure_rl/                 # Pure PPO/SAC (no safety)
│   ├── hocbf_rl/               # Traditional HOCBF + RL (no robustness, no differentiable training)
│   └── nmpc/                    # Nonlinear MPC (CasADi-based, called via subprocess)
├── references/                  # Literature management
│   ├── papers/
│   │   ├── hocbf/
│   │   ├── robust_cbf/
│   │   ├── diff_qp/
│   │   ├── safe_rl/
│   │   ├── gp_uncertainty/
│   │   ├── ccs_modeling/
│   │   └── power_control/
│   ├── index.yaml               # Structured literature index
│   └── reading_notes/
├── paper/                       # Paper resources
│   ├── figures/                 # nature-figure generated charts
│   └── sections/                # Draft sections
├── tests/                       # Unit tests
├── docs/                        # Documentation
├── configs/                     # Experiment configs (YAML)
└── CLAUDE.md
```

---

## 5. Five-Phase Design

### Phase 1: Theoretical Foundation (Low-Order Validation)

**Goal**: Validate end-to-end differentiable training loop (HOCBF + Diff-QP + PPO) on a double integrator system with safety constraints.

**System**: Double integrator $\dot{x} = v$, $\dot{v} = u$, with state $[x, v]^\top \in \mathbb{R}^2$ and control $u \in [-u_{max}, u_{max}]$.

- **Relative degree**: $m = 2$ for position-based safety constraints $h(x) = x^2 - r^2 \ge 0$ (keep distance $\ge r$ from origin)
- **HOCBF chain**: $\psi_0 = h(x)$, $\psi_1 = \dot{h}(x) + k_1 h(x) = 2xv + k_1(x^2 - r^2)$
- **HOCBF constraint**: $2v^2 + 2xu + k_1 \cdot 2xv + k_2 \psi_1 \ge 0$ (affine in $u$)

**Theoretical Output**:
- Theorem 1: Recursive HOCBF construction for relative degree $m \ge 2$ systems with forward invariance proof (standard result, adapted to our QP formulation)
- Theorem 2: Gradient existence and correctness of the differentiable QP layer — prove that $\partial u^*/\partial u_{RL}$ exists and is well-defined when strict complementarity holds, with perturbation regularization ensuring this condition

**Code Modules**:
- `rocbf/cbf/hocbf.py` — HOCBF class: Lie derivative computation via JAX `jax.grad` and `jax.jacfwd`, constraint function $A_0(x), b_0(x)$ construction
- `rocbf/qp/diff_qp.py` — Differentiable QP layer using qpax; custom JVP rule for KKT implicit differentiation
- `rocbf/rl/ppo.py` — PPO implementation (Flax NNX, clipped objective, GAE)
- `envs/safe_navigation/` — Double integrator environment (JAX)

**Note**: SAC implementation (`rocbf/rl/sac.py`) is deferred to Phase 4 when both RL algorithms are needed for dual-baseline comparison. Phase 1-2 validates with PPO only.

**Validation Metrics**:
- Safety constraint zero-violation rate = 100% (under nominal model)
- PPO+HOCBF cumulative reward ≥ 90% of pure PPO reward
- QP gradient backpropagation: finite difference check with relative error < 1e-4

**Exit Criteria**:
- HOCBF constraints satisfied in all 100 evaluation episodes under nominal model
- Gradients flow correctly through QP layer (finite difference check passes)
- PPO + HOCBF achieves >90% of pure PPO reward while maintaining zero violation

---

### Phase 2: Robustness Injection

**Goal**: Introduce GP residual learning for Robust HOCBF; validate probabilistic safety guarantee under model mismatch.

#### 2.1 Multiple Uncertainty Scenarios for Double Integrator

To stress-test the compositional $\epsilon(x)$ bound under different $\Delta f$ structures, we define four scenarios of increasing complexity:

| Scenario | $\Delta f(x, v)$ | Physical Analogy | Perturbation Type |
|----------|-------------------|------------------|-------------------|
| S1: Damping | $\begin{bmatrix} 0 \\ 0.2v \end{bmatrix}$ | Viscous friction nonlinearity | Velocity-dependent, affects $\dot{v}$ only |
| S2: Periodic | $\begin{bmatrix} 0.1 \sin(2\pi x) \\ 0 \end{bmatrix}$ | Periodic external disturbance | Position-dependent, affects $\dot{x}$ only |
| S3: Coupled | $\begin{bmatrix} 0.1 \sin(2\pi x) \\ 0.2v \end{bmatrix}$ | Combined damping + disturbance | Both states affected |
| S4: Nonlinear | $\begin{bmatrix} 0.15x^2 \\ 0.3\cos(\pi v) \end{bmatrix}$ | Nonlinear spring + Coulomb-like friction | Strongly nonlinear, both states |

**Rationale for multiple scenarios**:
- S1 isolates $\Delta f$ to the velocity equation ($\dot{v}$) — tests if GP can learn a simple linear residual
- S2 isolates $\Delta f$ to the position equation ($\dot{x}$) — tests the $\psi$-chain propagation since $\Delta f_1$ affects $\delta_1 = L_{\Delta f} h = 2x \cdot \Delta f_1$
- S3 combines both — tests the multi-dimensional aggregation (10)
- S4 adds nonlinearity — tests GP's ability to capture complex residuals beyond linear models

**Why S2 is critical for the theory**: In S1 ($\Delta f_1 = 0, \Delta f_2 \ne 0$), the perturbation $\delta_1 = L_{\Delta f} h = 2x \cdot 0 + 0 = 0$, and the $\psi$-chain propagation is trivially zero at level 1. In S2 ($\Delta f_1 \ne 0, \Delta f_2 = 0$), $\delta_1 = 2x \cdot 0.1\sin(2\pi x) \ne 0$, which propagates to higher levels — this is the scenario that actually tests the compositional bound.

**Theoretical Output**:
- Theorem 3: Forward invariance guarantee of Robust HOCBF under bounded uncertainty — prove that if $\epsilon(x) \ge |\Delta(x, u)|$ for all $u$ in the feasible set $\mathcal{U}$ and all $x$ in the operating region, then the safe set is forward invariant with probability $\ge 1-\delta$
- Derivation: Per-dimension GP → compositional $\sigma_{\text{total}}(x)$ via (10)-(12) → $\epsilon(x) = \sigma_{\text{total}}(x)$ (note: $\beta$ is already in each $\sigma_i$, see eq (5) footnote)
- Corollary: For $m=2$ systems, the compositional bound simplifies to (Section 2.3.4), and $\sigma_{\text{total}}$ has a closed-form expression in terms of $\nabla h$, $\nabla \hat{\psi}_1^0$, and $\sigma_{GP,j}$

**Code Modules**:
- `rocbf/gp/gp_residual.py` — GP residual learning module (JAX, Matérn-5/2 kernel)
  - Offline data collection: 5000 transitions from random policy
  - Online update: every 50 training episodes, incorporate new residuals (two-timescale protocol)
- `rocbf/cbf/robust_hocbf.py` — Robust HOCBF: computes compositional $\epsilon(x)$ via (5)-(12)
- Uncertainty injection in `envs/safe_navigation/dynamics.py`: configurable $\Delta f$ from 4 scenarios above

**Validation Metrics**:
- Safety violation rate under model mismatch: Traditional HOCBF > 0% vs Robust HOCBF ≈ 0% (with probability $\ge 1-\delta$)
- GP residual prediction calibration: empirical coverage of $2\sigma$ interval should be ≥ 95%
- Tightness of $\epsilon(x)$: compare GP-based $\epsilon$ vs. oracle $\epsilon^*$ (computed with known $\Delta f$) — ratio $\epsilon/\epsilon^* < 2$ indicates a tight bound
- Per-scenario breakdown: violation rates for S1-S4 individually

**Exit Criteria**:
- Robust HOCBF maintains violation rate < 1% under all 4 perturbation scenarios (probabilistic guarantee)
- Traditional HOCBF violation rate > 10% under at least 2 of the 4 scenarios
- GP predictions are well-calibrated (empirical coverage within 90-98% of predicted intervals)
- Compositional $\epsilon(x)$ is within 2× of oracle bound in ≥ 90% of states

**Mapping to CCS**: The double integrator scenarios directly map to CCS uncertainty:
- S1 ($\Delta f_2 \propto v$) → CCS: unmodeled pressure dynamics proportional to pressure rate
- S2 ($\Delta f_1 \propto \sin(x)$) → CCS: periodic combustion oscillations affecting power output
- S3 (coupled) → CCS: combined coal quality drift + heat transfer degradation
- S4 (nonlinear) → CCS: nonlinear valve characteristics + complex thermal coupling

---

### Phase 3: CCS Scenario Deployment

**Goal**: Build supercritical CCS simulation environment; migrate validated theoretical framework.

#### 3.1 CCS State Space

The CCS state consists of five physical variables (no augmentation needed — see §3.4 for actuator rate constraint treatment):

$$x = [N_e, \; P_t, \; W_f, \; W_w, \; \mu_t]^\top \in \mathbb{R}^5$$

| Symbol | Description | Unit | Nominal Range |
|--------|------------|------|---------------|
| $N_e$ | Generator power output | MW | [300, 600] |
| $P_t$ | Main steam pressure | MPa | [16, 27] (supercritical) |
| $W_f$ | Fuel flow rate | kg/s | [30, 80] |
| $W_w$ | Feedwater flow rate | kg/s | [100, 300] |
| $\mu_t$ | Turbine valve opening | % | [0, 100] |

**Why no state augmentation**: The previous design augmented the state to $\mathbb{R}^7$ with $u_f^{prev}, u_t^{prev}$ to make actuator rate constraints "proper HOCBF." However, this is mathematically invalid: $h_2 = r_f^2 - ((u_f - u_f^{prev})/\Delta t)^2$ still depends on the control input $u_f$ (not just the state), so it is NOT a valid HOCBF function regardless of augmentation. Instead, actuator rate constraints are enforced as control input bounds directly in the QP (see §3.4).

#### 3.2 CCS Control Inputs

$$u = [u_f, \; u_w, \; u_t]^\top \in \mathcal{U}$$

| Symbol | Description | Range |
|--------|------------|-------|
| $u_f$ | Fuel command | $[-1, 1]$ (normalized) |
| $u_w$ | Feedwater command | $[-1, 1]$ (normalized) |
| $u_t$ | Turbine valve command | $[-1, 1]$ (normalized) |

#### 3.3 CCS Dynamics

$\dot{x} = f_0(x) + g_0(x)u$ with specific nonlinear coupling terms from boiler-turbine thermodynamics. The structure of the five state equations follows the Tan et al. coordinated control model:

1. **Power output** ($\dot{N}_e$): Generator power responds to turbine valve opening $\mu_t$ and main steam pressure $P_t$ through turbine dynamics. Dominant time constant $\sim$5-10s (steam chamber + rotor inertia). Simplified: $\dot{N}_e = f_{N_e}(N_e, P_t, \mu_t)$.

2. **Main steam pressure** ($\dot{P}_t$): Pressure dynamics determined by the energy balance between fuel combustion (heat input) and steam consumption (turbine + feedwater). Dominant time constant $\sim$30-60s (thermal inertia of the boiler). Simplified: $\dot{P}_t = f_{P_t}(P_t, N_e, W_f, W_w, \mu_t)$.

3. **Fuel flow rate** ($\dot{W}_f$): First-order lag from fuel command $u_f$ to actual fuel flow, representing feeder dynamics. Time constant $\sim$10-20s. Simplified: $\dot{W}_f = (-W_f + K_f \cdot u_f) / T_f$.

4. **Feedwater flow rate** ($\dot{W}_w$): First-order lag from feedwater command $u_w$ to actual flow, representing pump/valve dynamics. Time constant $\sim$1-5s. Simplified: $\dot{W}_w = (-W_w + K_w \cdot u_w) / T_w$.

5. **Turbine valve opening** ($\dot{\mu}_t$): First-order lag from valve command $u_t$ to actual opening, representing governor valve servo dynamics. Time constant $\sim$2-5s. Simplified: $\dot{\mu}_t = (-\mu_t + K_t \cdot u_t) / T_t$.

The first two equations ($\dot{N}_e$, $\dot{P}_t$) contain the essential nonlinear thermodynamic coupling and are where the uncertainty $\Delta f$ manifests (via $Q_{net}$ and $\kappa$ perturbations). The last three equations are actuator dynamics — linear first-order lags with well-characterized parameters. The relative degree analysis confirms: $h_1(P_t)$ has $m=2$ because $\dot{P}_t$ depends on $W_f$ (whose derivative $\dot{W}_f$ depends on $u_f$); $h_2(N_e)$ has $m=2$ because $\dot{N}_e$ depends on $\mu_t$ (whose derivative $\dot{\mu}_t$ depends on $u_t$).

The exact nonlinear functional forms and numerical parameters will be derived from the selected literature model (see §3.9 CCS Model Source). RK4 discretization with $\Delta t = 1$s.

#### 3.4 CCS Safety Constraints

The CCS has two types of safety constraints, handled differently in the QP formulation:

**Type 1: HOCBF constraints (state-based, enforce forward invariance)**:

| Constraint | HOCBF Function | Relative Degree | Type |
|-----------|---------------|-----------------|------|
| $\|P_t - P_t^{ref}\| \le \Delta P_{max}$ | $h_1(x) = \Delta P_{max}^2 - (P_t - P_t^{ref})^2$ | $m=2$ | State constraint (fuel→energy→pressure) |
| $\|N_e - N_e^{ref}\| \le \Delta N_{max}$ | $h_2(x) = \Delta N_{max}^2 - (N_e - N_e^{ref})^2$ | $m=2$ | State constraint (turbine→power) |

where $P_t^{ref}$ and $N_e^{ref}$ are the reference setpoints for main steam pressure and generator power output, respectively. These setpoints are determined by the unit load dispatch schedule:
- **Constant setpoint mode** (for regulation testing): $P_t^{ref} = 25$ MPa, $N_e^{ref} = 600$ MW (rated load)
- **Load-following mode** (for tracking testing): time-varying references from a predefined load profile (see §3.5)
- $\Delta P_{max} = 0.5$ MPa, $\Delta N_{max} = 30$ MW (safety margins from operational standards for supercritical units)

**Time-varying HOCBF treatment**: In load-following mode, $h_i$ becomes time-varying: $h_1(x, t) = \Delta P_{max}^2 - (P_t - P_t^{ref}(t))^2$. Standard HOCBF theory (Xiao et al., 2021) assumes time-invariant $h$, but the time-varying extension is straightforward:
- The $\psi$-chain acquires additional terms from $\partial h / \partial t$: $\psi_1(x, t) = L_f h + \partial h / \partial t + k_1 \psi_0$. For $h_1$, $\partial h / \partial t = 2(P_t - P_t^{ref}(t)) \cdot \dot{P}_t^{ref}(t)$.
- Since $P_t^{ref}(t)$ and $N_e^{ref}(t)$ are predefined reference profiles (piecewise constant with step changes), $\dot{P}_t^{ref} = 0$ almost everywhere (except at step transitions). At step transitions, we use a smooth ramp (linear interpolation over 10s) to ensure $\dot{P}_t^{ref}$ is well-defined and bounded.
- The $\partial h / \partial t$ term is treated as an additional known additive term in the HOCBF constraint — it shifts $\hat{b}_0$ but does not affect the $\psi$-chain structure or the compositional $\epsilon(x)$ computation (which bounds the uncertainty, not the known reference variation).
- Implementation: The time-varying terms are computed analytically from the reference profile and added to $\hat{b}_0(x, t)$ in the QP. This is consistent with the time-varying CBF formulation in Breeden & Panagou (2022).

These are proper HOCBF functions: they depend on the state $x$ (and time $t$ in load-following mode), have well-defined relative degrees, and require the compositional $\epsilon_i(x)$ robustness margin.

**Type 2: Control input constraints (rate limits, enforced as QP bounds)**:

The actuator rate constraints $\|u_j - u_j^{prev}\|/\Delta t \le r_j$ (for $j \in \{f, t\}$) are NOT valid HOCBF constraints because they depend on the control input $u_j$ (not just the state). Instead, they are enforced as linear constraints on $u$ in the QP:

$$u_j^{prev} - r_j \Delta t \le u_j \le u_j^{prev} + r_j \Delta t, \quad j \in \{f, t\}$$

where $u_j^{prev}$ is the control input from the previous time step (stored as a constant in the QP, not as a state variable). These are written in standard form as additional rows of the QP:

$$A_{rate} u \le b_{rate}, \quad A_{rate} \in \mathbb{R}^{4 \times 3}, \; b_{rate} \in \mathbb{R}^4$$

(4 rows for 2 actuators × 2 bounds each.)

**Why no rate constraint for $u_w$ (feedwater)**: In supercritical units, the feedwater control valve responds significantly faster than the fuel feeder and turbine governor valve. The feedwater system is driven by a variable-speed pump with response time $\sim$1-2s, well within the $\Delta t = 1$s control interval, making rate constraints unnecessary at our time scale. In contrast, the fuel feeder has mechanical inertia ($\sim$10-20s response time) and the turbine governor valve has steam-chamber dynamics ($\sim$5-10s), both requiring explicit rate limits. If a specific CCS model shows feedwater rate sensitivity, a rate constraint can be added (increasing $A_{rate}$ to $\mathbb{R}^{6 \times 3}$).

**Why this separation matters**: HOCBF constraints require the safety function to be state-only ($h: \mathbb{R}^n \to \mathbb{R}$) so that Lie derivatives and the $\psi$-chain are well-defined. Control input constraints (rate limits, saturation) are simply bounds on the feasible set of $u$ — they don't need the HOCBF framework and don't need robustness margins $\epsilon$. Mixing them with HOCBF constraints would require an invalid $h(u)$ and introduces unnecessary conservatism.

**Combined QP formulation**:

$$u^* = \arg\min_{u} \frac{1}{2} \|u - u_{RL}\|^2 \quad \text{s.t.} \quad \underbrace{\hat{A}_0(x)u \le \hat{b}_0(x) - \epsilon(x)}_{\text{Robust HOCBF (2 rows)}} \;,\; \underbrace{A_{rate} u \le b_{rate}}_{\text{Rate limits (4 rows)}} \;,\; \underbrace{u_{min} \le u \le u_{max}}_{\text{Saturation}}$$

where $\hat{A}_0 \in \mathbb{R}^{2 \times 3}$, $\hat{b}_0, \epsilon \in \mathbb{R}^2$, computed using the mean-corrected $\hat{f} = f_0 + \mu_{GP}$ (see §2.5.1), and each HOCBF constraint has its own $\epsilon_i(x)$ computed from the compositional bound (5)-(12) applied to $h_i$.

**Multi-constraint $\epsilon_i(x)$ definition**: Each HOCBF constraint $h_i$ has its own compositional $\epsilon_i(x)$ because the $\psi$-chain structure differs per constraint (different $\nabla h_i$, different relative degree). The GP posterior ($\sigma_{GP,j}$) is shared across all constraints — the same $n$ per-dimension GPs provide uncertainty for all $\epsilon_i$ computations, with different Jacobian weights $\partial h_i / \partial x_j$ and $\partial \hat{\psi}_k^{0,i} / \partial x_j$ per constraint.

#### 3.5 CCS Reward Function

The RL reward is a weighted multi-objective function balancing tracking performance, control effort, and safety:

$$r(x_t, u_t, x_{t+1}) = r_{track}(x_{t+1}) - \lambda_u \cdot r_{effort}(u_t) - \lambda_{safe} \cdot r_{violation}(x_{t+1})$$

where:

- **Tracking reward**: $r_{track} = -\big((P_t - P_t^{ref})^2 / \sigma_P^2 + (N_e - N_e^{ref})^2 / \sigma_N^2\big)$, where $\sigma_P = 0.5$ MPa and $\sigma_N = 30$ MW are the respective HOCBF margins ($\Delta P_{max}$, $\Delta N_{max}$). This normalization ensures both terms contribute equally when at the constraint boundary (pressure deviation $\Delta P \sim 0.5$ MPa and power deviation $\Delta N \sim 30$ MW would otherwise differ by ~2 orders of magnitude with unnormalized weights). Equivalently, $w_P = 1/\sigma_P^2 = 4.0$ and $w_N = 1/\sigma_N^2 \approx 0.0011$.

- **Control effort penalty**: $r_{effort} = \|u_t\|^2 = u_f^2 + u_w^2 + u_t^2$, with $\lambda_u = 0.01$. Penalizes aggressive control actions to promote smooth operation and reduce actuator wear.

- **Safety violation penalty**: $r_{violation} = \sum_{i=1}^{2} \max(0, -h_i(x_{t+1}))^2$, with $\lambda_{safe} = 10.0$. Quadratic penalty proportional to the square of constraint violation magnitude. This supplements (but does not replace) the HOCBF QP safety filter — the QP filter ensures forward invariance during training, while the reward penalty shapes the policy to avoid operating near constraint boundaries.

**Rationale for separate reward penalty and QP filter**: The QP filter provides hard safety guarantees at execution time, but the reward penalty encourages the policy to learn actions that are *intrinsically* safe (reducing QP intervention frequency and improving post-distillation safety). Without the reward penalty, the policy may learn actions that consistently push against the HOCBF boundary, making the QP filter active at every step and increasing distillation error.

#### 3.6 CCS Episode Structure

**Episode length**: $T_{ep} = 600$ steps ($\Delta t = 1$s, so 600s = 10 minutes of simulated operation). This is long enough to capture multiple load transitions while keeping training tractable.

**Termination conditions** (episode ends early if):
1. **Hard constraint violation**: $P_t < 16$ MPa or $P_t > 27$ MPa (physical safety limits beyond the HOCBF margin)
2. **State out of range**: Any state variable leaves its nominal range by more than 50% (e.g., $N_e < 150$ MW or $N_e > 750$ MW)
3. **Numerical divergence**: Any state becomes NaN or exceeds $10^6$ (numerical stability guard)

Early termination gives a terminal reward of $r_{terminal} = -1000$ (large negative reward to discourage reaching these states).

**Reference trajectory**: Two modes for different experimental purposes:

| Mode | Reference Profile | Purpose |
|------|-------------------|---------|
| Regulation | Constant $P_t^{ref} = 25$ MPa, $N_e^{ref} = 600$ MW | Test steady-state safety and disturbance rejection |
| Load-following | Step changes: 100%→80%→100%→90%→100% load at $t \in \{100, 200, 300, 400\}$s | Test tracking performance under load transitions |

In load-following mode, $P_t^{ref}(t)$ and $N_e^{ref}(t)$ scale proportionally with the load command (e.g., 80% load → $N_e^{ref} = 480$ MW, $P_t^{ref} = 20$ MPa approximately, following the boiler-turbine coordination curve). The exact coordination curve is derived from the CCS model's steady-state relationships.

**Training vs evaluation**: During training, episodes use the regulation mode (constant setpoint) for the first 80% of total training episodes, then switch to load-following for the final 20% (curriculum learning). Specifically, if total training is $E_{total}$ episodes, regulation runs for episodes $1$ to $\lfloor 0.8 \cdot E_{total} \rfloor$ and load-following runs for the remainder. **In both modes, the same uncertainty sampling protocol applies** (§3.7: 40% Nominal, 15% each for the other 5 conditions) — the reference mode and uncertainty condition are sampled independently at the start of each episode. Evaluation uses both modes, reporting separate metrics for each.

#### 3.7 CCS Uncertainty Scenarios (Quantified)

| Scenario | Parameter | Perturbation | Physical Meaning |
|----------|-----------|-------------|-----------------|
| Nominal | — | 0% | Design operating point |
| Coal quality drift | Fuel heating value $Q_{net}$ | -15% to -30% | Coal blending / inferior coal |
| Fouling | Heat transfer coeff. $\kappa$ | -20% to -40% | Ash deposition on heating surfaces |
| Combined | $Q_{net}$ + $\kappa$ | -15% & -20% | Realistic worst case |

These perturbations affect $f_0$ (not $g_0$), consistent with Assumption 1 ($\Delta g = 0$). Specifically:
- $Q_{net}$ reduction → $\Delta f$ in the fuel combustion term (reduces energy release rate)
- $\kappa$ reduction → $\Delta f$ in the heat transfer term (reduces steam generation efficiency)

**Uncertainty application protocol**:
1. **Per-episode application**: The perturbation level (e.g., -15% or -30% for coal drift) is **constant within each episode** but **varies across episodes**. This models a realistic scenario where operating conditions are stable within a control session but change between sessions (e.g., coal batch changes between shifts).
2. **Agent awareness**: The agent does **not** know the perturbation level — it must rely on the GP to learn the residual online. This is consistent with the Robust HOCBF formulation, where $\Delta f$ is unknown and compensated via $\epsilon(x)$.
3. **Training curriculum**: During training, perturbation conditions are sampled at the start of each episode from the full set of 6 conditions: {Nominal, Coal drift (-15%), Coal drift (-30%), Fouling (-20%), Fouling (-40%), Combined (-15% & -20%)}. The sampling distribution is 40% Nominal, 15% each for Coal drift (-15%), Coal drift (-30%), Fouling (-20%), Fouling (-40%), and 15% Combined, ensuring the policy and GP encounter the full range of perturbation types including the combined worst case.
4. **Evaluation protocol**: Each test condition uses a **fixed** perturbation level held constant across all evaluation episodes (e.g., all 100 episodes under "-15% coal drift" use exactly -15%). This ensures clean metric separation between conditions.
5. **GP data collection**: Residual data $\Delta \hat{f}(x) = (x_{t+1} - x_t)/\Delta t - f_0(x_t) - g_0(x_t)u_t$ naturally captures the applied perturbation. Since perturbation is constant within an episode, the GP learns a mapping from $x$ to the perturbed residual — no special perturbation labeling is needed.

#### 3.8 Code Modules

- `envs/ccs/dynamics.py` — Boiler-turbine coordinated dynamics (5-dim state, RK4, $\Delta t = 1$s)
- `envs/ccs/constraints.py` — Two HOCBF constraints ($h_1$: pressure, $h_2$: power) with auto-computed Lie derivatives; rate limit constraints as QP bounds
- `envs/ccs/uncertainty.py` — Parameterized perturbation of $Q_{net}$ and $\kappa$
- `envs/ccs/env.py` — Gymnasium-compatible interface, with `uncertainty_mode` parameter; stores $u^{prev}$ for rate limits
- `rocbf/gp/sparse_gp.py` — Sparse GP with inducing points (target $Z = 200$) for 5-dim state input

#### 3.9 CCS Model Source

**Primary choice: Tan et al. supercritical unit coordinated control model** — selected based on the following concrete criteria:

| Criterion | Tan et al. | Bell-Åström | Decision |
|-----------|-----------|-------------|----------|
| Supercritical (not drum) | ✅ Superficial unit model | ❌ Drum boiler | Tan — matches our target system |
| Published parameter set | ✅ Complete nonlinear ODE parameters in paper | ⚠️ Parameters scattered across multiple sources | Tan — self-contained |
| MIMO structure | ✅ 3-input 3-output (fuel, water, valve → power, pressure, temperature) | ✅ Same | Tie |
| Validated against plant data | ✅ 600MW supercritical unit | ✅ Various units | Tan — same class |
| Relative degree computable | ✅ Pressure: $m=2$ (fuel→energy→pressure); Power: $m=2$ (valve→flow→power) | ✅ Similar | Tie |
| Coordinate control focus | ✅ Explicit coordinated control structure | ⚠️ General boiler dynamics | Tan — direct application |

**Fallback**: If Tan et al.'s model proves difficult to implement (missing parameters, numerical instability), we fall back to the Bell-Åström model adapted for supercritical conditions (using the generalized state-space formulation from Åström & Bell 2000, Chapter 11).

**Decision timeline**: Finalize during Phase 3 Week 1 by implementing both models and comparing open-loop behavior. Accept Tan et al. if the trajectory matches published results within 10% error; otherwise switch to fallback.

#### 3.10 Key Challenges

- Lie derivative computation for 5-dim nonlinear system: use JAX `jax.grad`/`jax.jacfwd` for automatic computation
- Multi-constraint QP feasibility: add slack variables with heavy penalty to prevent infeasibility; prioritize pressure constraint ($h_1$) over power constraint ($h_2$)
- Sparse GP for 5-dim input: inducing point selection via k-means on collected data; $Z = 200$ inducing points

#### 3.11 Exit Criteria

- CCS environment reproduces expected dynamic behavior from literature (open-loop trajectory matches published results within 5% error)
- Both HOCBF constraints ($h_1, h_2$) on the 5-dim state are well-defined with correct relative degrees
- Environment runs at ≥1000 steps/sec on RTX 4090 (JAX jit-compiled)

---

### Phase 4: Full Experiments

**Baseline Comparison**:

| # | Method | Description | Expected Behavior |
|---|--------|------------|-------------------|
| 1 | Pure PPO | No safety constraint | No safety guarantee, high violation rate |
| 2 | Pure SAC | No safety constraint | Same as above |
| 3 | PPO + Traditional HOCBF | HOCBF without robustness, QP as post-hoc projection | Safe under nominal, violates under mismatch |
| 4 | SAC + Traditional HOCBF | Same as #3 with SAC | Same as above |
| 5 | NMPC | CasADi IPOPT solver, horizon $T=10$, same model as HOCBF | Safe but high inference latency (50-200ms/step) |
| 6 | PPO + Robust Diff-HOCBF (Ours) | Full pipeline: Robust HOCBF + diff-QP + PPO | Probabilistic safety + low latency |
| 7 | SAC + Robust Diff-HOCBF (Ours) | Full pipeline with SAC | Same as above |

**NMPC Implementation Details**:
- Solver: CasADi with IPOPT
- Prediction horizon: $T = 10$ steps ($\Delta t = 1$s, so 10s lookahead)
- Model: same nominal $f_0, g_0$ as HOCBF methods (fair comparison)
- Constraints: same safety constraints as HOCBF
- Called via `subprocess` from Python; timing measured wall-clock (excluding IPOPT first-solve warmup — the first solve is discarded as it includes JIT compilation and factorization overhead that would not occur in a deployed NMPC system with warm-starting)

**Experiment Dimensions**:

1. **Safety**: Constraint violation rate (% episodes with any violation), cumulative violation magnitude, maximum instantaneous violation
2. **Control Performance**: Tracking error (MSE/MAE for $P_t$ and $N_e$), overshoot (%), settling time (s)
3. **Robustness**: 6 operating conditions — Nominal, Coal drift (-15%), Coal drift (-30%), Fouling (-20%), Fouling (-40%), Combined (-15% & -20%)
4. **Inference Latency**: Explicit network vs NMPC single-step wall-clock time (ms), measured over 1000 steps
5. **Ablation Studies**:
   - A1: Remove GP residual → use fixed worst-case $\epsilon$ (conservative bound without learning)
   - A2: Remove differentiable training → use projection method (solve QP at execution only, no gradient through QP)
   - A3: Effect of GP mean correction — compare $f̂ = f_0 + μ_{GP}$ (mean-corrected) vs $f_0$ only (mean-ignored); quantifies how much tighter ε becomes when μ_GP is utilized
   - A4: Effect of $\beta$ scaling — compare $\beta$ values ($1.5, 2.0, 3.0$) on safety vs performance tradeoff

**Training Protocol**:
- 5 random seeds per method per condition
- Report mean ± std for all metrics
- Statistical significance: paired t-test or Wilcoxon signed-rank test
- Total: 7 methods × 6 conditions × 5 seeds = 210 training runs + 4 ablations × 6 conditions × 5 seeds = 120 ablation runs

**Exit Criteria**:
- Ours (methods 6-7) achieves <1% violation rate across all perturbation scenarios (probabilistic guarantee, $\delta \le 0.01$)
- Ours achieves ≥95% of pure RL tracking performance under nominal conditions
- Explicit policy inference < 1ms per step (vs NMPC 50-200ms)
- Distilled policy passes post-distillation safety verification (0% violation in $N_{verify}=100$ episodes)
- All ablation variants show statistically significant differences (p < 0.05)

---

### Phase 5: Paper Writing

**Target**: IEEE TAC (12-15 pages double-column)

**Paper Structure**:

| Section | Content | Key Theorems/Results |
|---------|---------|---------------------|
| I. Introduction | Safe RL + industrial control challenges + our contributions | — |
| II. Preliminaries | HOCBF definition (Def 1), GP regression (Def 2), RL basics | — |
| III. Robust Differentiable HOCBF | Sec 3.1: Robust HOCBF formulation + $\psi$-chain uncertainty propagation + Thm 3 (forward invariance with probability $1-\delta$). Sec 3.2: GP-based compositional $\epsilon(x)$ derivation + PAC-Bayes $\beta$. Sec 3.3: Differentiable QP + Thm 2 (gradient existence) | Thm 1 (adapted from Xiao et al. 2021), Thm 2 (new), Thm 3 (new) |
| IV. Explicit Safe Policy Learning | End-to-end training algorithm (Algorithm 1). Two-timescale GP update protocol. Policy distillation with safety-weighted loss. Post-distillation verification | — |
| V. Application to CCS | CCS model (5-dim state), state/control definitions, safety constraint design (2 HOCBFs + rate limit bounds), HOCBF chain computation, uncertainty scenarios, reward and episode design | — |
| VI. Experiments | 7 methods × 6 conditions × 5 seeds. Safety/performance/robustness/latency tables and figures. Ablation studies. Double integrator validation results (4 scenarios) | — |
| VII. Conclusion | Summary + future work (robustness to $\Delta g$, multi-agent extension, nonlinear $\alpha_i$) | — |

**Toolchain**:
- nature-figure: All experimental figures
  - Fig 1: System architecture diagram (Actor → Diff-QP → Safe Action)
  - Fig 2: Phase 1-2 validation results (2D trajectories + violation comparison across 4 scenarios)
  - Fig 3: CCS tracking performance comparison (multi-panel time series)
  - Fig 4: Safety violation comparison under perturbation (bar chart)
  - Fig 5: Inference latency comparison (bar chart: Ours vs NMPC)
  - Fig 6: Ablation study results (grouped bar or radar chart)
- nature-polishing: Final manuscript polish with 25-rule academic English refinement

---

## 6. Literature Categories

| Category | Coverage | Key References (to be populated) |
|----------|----------|----------------------------------|
| HOCBF | Recursive construction, forward invariance, class-$\mathcal{K}$ functions | Xiao et al. 2021 (IEEE TAC), Nguyen & Sreenath 2016 |
| Robust CBF | Uncertainty-aware CBF, input-to-state safety, $\psi$-chain propagation | Taylor et al. 2020 (ISSf), Jankovic 2018 |
| Differentiable Optimization | Implicit differentiation, QP layers, cvxpylayers/qpax | Amos & Kolter 2017, Agrawal et al. 2019 |
| Safe RL | CBF-RL integration, constrained RL, safe exploration | Cheng et al. 2019, Emam et al. 2021 |
| GP Uncertainty | GP regression, sparse GP, PAC-Bayes bounds, calibration | Srinivas et al. 2010, Rasmussen & Williams 2006 |
| CCS Modeling | Supercritical unit dynamics, coordinated control | Åström & Bell 2000, Tan et al. |
| Power Control | Energy system control, DCS/PLC, model mismatch | — |

---

## 7. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| QP infeasibility in multi-constraint CCS | Medium | High | Slack variables with $\lambda_{slack} \gg 1$ penalty; prioritize pressure constraint ($h_1$) over power constraint ($h_2$); rate limits as QP bounds (not HOCBF) reduce constraint dimensionality |
| GP scalability in 5-dim CCS input | Medium | Medium | Sparse GP with $Z=200$ inducing points; Cholesky factorization via JAX; fallback: deep kernel learning |
| JAX numerical instability in QP solver | Low | High | Perturbation regularization for strict complementarity; reference qpax; custom JVP rule with tolerance |
| CCS model fidelity insufficient for TAC | Low | Medium | Use well-cited models with published validation; compare open-loop trajectories against literature |
| Probabilistic vs deterministic safety gap | Medium | High | Explicitly state probabilistic guarantee ($1-\delta$) throughout paper; provide empirical evidence that $\delta < 0.01$; calibrate $\beta$ conservatively |
| Training time (330 runs total) | Medium | Low | JAX `vmap` for vectorized environments; prioritize key comparisons first; estimated ~4 days total on 4090 |
| Distillation safety gap | Low | Medium | Safety-weighted loss + post-distillation verification; re-distill with higher $\lambda_{safe}$ if violations detected |
| Compositional $\epsilon(x)$ too conservative | Medium | Medium | Compare with oracle bound; if $\epsilon/\epsilon^* > 3$, investigate adaptive $\beta$ or partitioned GP; ablation A4 quantifies tradeoff |
| Two-timescale GP update instability | Low | Medium | Re-adaptation period with reduced LR; monitor $\epsilon(x)$ change magnitude between updates; if $\|\epsilon_{new} - \epsilon_{old}\| / \|\epsilon_{old}\| > 0.5$, skip GP update |

---

## 8. Success Criteria

1. **Theoretical**: Three theorems with rigorous proofs — (T1) forward invariance of HOCBF, (T2) gradient existence of Diff-QP, (T3) **probabilistic** forward invariance of Robust HOCBF with explicit $\delta$ — accepted by TAC reviewers
2. **Safety**: Violation rate < 1% under model mismatch (Robust HOCBF with $\delta \le 0.01$), while traditional HOCBF violation rate > 10%
3. **Performance**: Control performance ≥ 95% of unconstrained RL baseline under nominal conditions
4. **Efficiency**: Explicit distilled policy inference < 1ms vs NMPC 50-200ms
5. **Robustness**: Safety maintained across all 6 operating conditions (nominal, coal drift -15%, coal drift -30%, fouling -20%, fouling -40%, combined)
6. **Reproducibility**: All experiments reproducible from code + configs; 5-seed statistical significance; open-source release

---

## Appendix A: Summary of Changes from Self-Review v7→v8

| Issue | Severity | Fix Applied |
|-------|----------|-------------|
| 1. Phase 2 Theoretical Output still used old condition $\epsilon(x) \ge |\Delta(x, u^*(x))|$ instead of updated "for all $u \in \mathcal{U}$" from §2.3.3 | HIGH | Updated Phase 2 Theoretical Output to match §2.3.3: "$\epsilon(x) \ge |\Delta(x, u)|$ for all $u$ in the feasible set $\mathcal{U}$" |
| 2. Eq (12) $\sigma_{\text{ctrl}}$ used $\|u_{RL}\|$ as upper bound for $\|u\|$, but Theorem 3 condition requires bound for all $u \in \mathcal{U}$ — $\|u_{RL}\|$ may be smaller than $u_{max}$ | MED | Changed $\|u_{RL}\|$ to $u_{max} = \sup_{u \in \mathcal{U}} \|u\|$ in eq (12) and Theorem 3 note; added explanation that $u_{max}$ is known from actuator saturation limits (e.g., $\sqrt{m}$ for $\mathcal{U} = [-1,1]^m$) |
| 3. Phase 5 Paper Structure listed Thm 1, 2, 3 without distinguishing new vs. adapted results | LOW | Annotated Thm 1 as "(adapted from Xiao et al. 2021)", Thm 2 as "(new)", Thm 3 as "(new)" |
| 4. §3.6 training mode did not specify whether uncertainty sampling differs between regulation and load-following modes | LOW | Added explicit note: same uncertainty sampling protocol (§3.7) applies in both modes; reference mode and uncertainty condition are sampled independently |
| 5. §2.2 class-$\mathcal{K}$ function gains $k_i$ had no selection criteria or recommended values | LOW | Added $k_i$ selection guidance: $k_1, k_2 \in [1, 5]$ for $m=2$; CCS initial values $k_1 = 2, k_2 = 2$; tuned via grid search |

---

## Appendix B: Summary of Changes from Self-Review v6→v7

| Issue | Severity | Fix Applied |
|-------|----------|-------------|
| 1. Paper Structure §5 still wrote "7 methods × 4 conditions" but actual count is 6 conditions (updated in v6) | HIGH | Changed to "7 methods × 6 conditions × 5 seeds" in Phase 5 Paper Structure table |
| 2. Success Criteria §8 wrote "4 uncertainty scenarios" but should be 6 operating conditions; also conflated double integrator S1-S4 with CCS conditions | HIGH | Changed to "6 operating conditions" with explicit listing (nominal, coal drift -15%, coal drift -30%, fouling -20%, fouling -40%, combined) |
| 3. §3.5 reward defined unnormalized form $w_P = w_N = 1.0$ then immediately replaced with normalized form — confusing | MED | Removed unnormalized form, directly presented normalized $r_{track} = -((P_t - P_t^{ref})^2 / \sigma_P^2 + (N_e - N_e^{ref})^2 / \sigma_N^2)$ with explanation |
| 4. §2.6 distillation loss eq (14) written for single constraint but CCS has $p=2$ HOCBF constraints | MED | Updated eq (14) to sum over $p$ constraints: $\sum_i \mathbb{1}[h_i(x')<0] |h_i(x')|^2$ and $\sum_i \max(0, \epsilon_i - C_i)^2$; updated all surrounding text and theoretical guarantee to multi-constraint form |
| 5. §3.7 training curriculum didn't include Combined condition during training; sampling protocol ambiguous | MED | Rewrote training curriculum: explicit sampling from all 6 conditions with 40% Nominal, 15% each for other 5 conditions |
| 6. §2.5.4 sparse GP justification "When $n \ge 5$" misleading — sparse GP needed for large $N$, not large $n$ | LOW | Changed to: sparse GP needed as dataset grows ($N \sim 10^4$-$10^5$), reducing $O(N^3)$ to $O(NZ^2)$; $Z=200-500$ for 5-dim input |
| 7. §2.3.3 Theorem 3 condition "$\epsilon(x) \ge |\Delta(x, u^*(x))|$" has circular dependency with $u^*$ depending on $\epsilon$ | LOW | Changed condition to "$\epsilon(x) \ge |\Delta(x, u)|$ for all $u \in \mathcal{U}$" and added note that $\epsilon$ bounds perturbation for any feasible $u$ (using $\|u_{RL}\|$ in eq (12)), so condition for actual $u^*$ is automatically satisfied |
| 8. Residue: §3.4 time-varying HOCBF "shifts $b_0$" should be "shifts $\hat{b}_0$"; §3.4 multi-constraint used $\psi_k^{0,i}$ should be $\hat{\psi}_k^{0,i}$ | LOW | Updated both references to use mean-corrected notation ($\hat{b}_0$, $\hat{\psi}_k^{0,i}$) |

---

## Appendix C: Summary of Changes from Self-Review v5→v6

| Issue | Severity | Fix Applied |
|-------|----------|-------------|
| 1. **CRITICAL**: Theorem 3 (§2.3.3) still wrote $\epsilon = \beta \cdot \sigma_{\text{total}}$, contradicting eq (5); §2.5.4 convergence argument also had $\epsilon = \beta \cdot \sigma_{\text{total}}$ | CRITICAL | Changed Theorem 3 from $\epsilon = \beta \cdot \sigma_{\text{total}}$ to $\epsilon = \sigma_{\text{total}}$ with note that $\beta$ is embedded in each $\sigma_i$; updated §2.5.4 convergence argument to explain each $\sigma_i = O(\sqrt{\ln N / N})$ so $\epsilon = O(\sqrt{\ln N / N})$ |
| 2. Time-varying $h_i$ in load-following mode not handled — standard HOCBF assumes time-invariant $h$ | HIGH | Added time-varying HOCBF treatment in §3.4: $\partial h / \partial t$ terms, piecewise-constant refs with smooth ramps, $\dot{P}_t^{ref} = 0$ a.e., additional known term in $\hat{b}_0(x,t)$, reference to Breeden & Panagou (2022) |
| 3. §2.5.3 PAC-Bayes statement said $|\Delta f_j| \le \beta \sigma_{GP,j}$ but should say $|\Delta f_j - \mu_{GP,j}| \le \beta \sigma_{GP,j}$ per GP mean correction; §2.3.3 proof sketch step 1 also affected | HIGH | Fixed PAC-Bayes statement in §2.5.3 to $|\Delta f_j - \mu_{GP,j}| \le \beta \sigma_{GP,j}$ with note about posterior residual; fixed Theorem 3 proof sketch step 1 to use $|\Delta f_j - \mu_{GP,j}|$; fixed Assumption 2 note to use $|\Delta f_j - \mu_{GP,j}|$ |
| 4. Reward weight dimension mismatch — pressure ~0.5 MPa vs power ~30 MW means power dominates by ~3 orders with $w_P = w_N = 1.0$ | HIGH | Added normalization: $r_{track} = -((P_t - P_t^{ref})^2 / \sigma_P^2 + (N_e - N_e^{ref})^2 / \sigma_N^2)$ where $\sigma_P = \Delta P_{max} = 0.5$ MPa, $\sigma_N = \Delta N_{max} = 30$ MW; equal contribution at constraint boundary |
| 5. QP notation inconsistency — §2.4 and §3.4 used $A_0, b_0$ (based on $f_0$) but §2.5.1 said should use $\hat{A}_0, \hat{b}_0$ (based on $\hat{f}$) | MED | Unified QP notation to $\hat{A}_0, \hat{b}_0$ throughout §2.4 (eq (8), KKT (9), implicit diff), §3.4 combined QP, and multi-constraint extension; added note that $\hat{A}_0 = A_0$ before GP training |
| 6. Eqs (10)-(12) used $\psi_{i-1}^0$ and $|L_{f_0}|_{\text{op}}$ but should use $\hat{\psi}_{i-1}^0$ and $|L_{\hat{f}}|_{\text{op}}$ after mean correction | MED | Updated eqs (10)-(12) and all surrounding text to use $\hat{\psi}_{i-1}^0$, $|L_{\hat{f}}|_{\text{op}}$, $\Delta \hat{f}$, $L_{\Delta \hat{f}}$; updated Practical computation and Multi-constraint extension accordingly |
| 7. Uncertainty scenarios don't specify application protocol | MED | Added detailed protocol in §3.7: per-episode constant perturbation, agent unaware, training curriculum (40% nominal / 30% mild / 30% severe), evaluation fixed per condition, GP data collection note |
| 8. No rate constraint for $u_w$ without explanation | MED | Added explanation: feedwater pump responds in ~1-2s (within $\Delta t = 1$s), much faster than fuel feeder (~10-20s) and turbine valve (~5-10s); can be added if model shows sensitivity |
| 9. "4 conditions" ambiguous — coal drift has 2 levels, fouling has 2 levels, unclear if 4 or 6 total | MED | Clarified as 6 conditions: Nominal, Coal drift (-15%), Coal drift (-30%), Fouling (-20%), Fouling (-40%), Combined; updated training run count to 330 total |
| 10. "n > 5" for sparse GP but CCS has exactly n=5 | LOW | Changed to $n \ge 5$ |
| 11. §3.3 CCS dynamics extremely vague ("exact equations will be derived") | LOW | Expanded to 5 numbered state equations with physical structure, dominant time constants, linearity classification, and relative degree confirmation |
| 12. "80% of training" undefined — episode count or total steps? | LOW | Defined as 80% of total training episodes: episodes $1$ to $\lfloor 0.8 \cdot E_{total} \rfloor$ use regulation, remainder use load-following |

---

## Appendix D: Summary of Changes from Self-Review v4→v5

| Issue | Severity | Fix Applied |
|-------|----------|-------------|
| 1. **CRITICAL**: $\beta$ double-counting — eq (5) used $\epsilon = \beta \cdot \sigma_{\text{total}}$ but each $\sigma_i$ already contains $\beta$ via eqs (10)-(11), making $\epsilon$ too conservative by factor ~2-4 | CRITICAL | Changed eq (5) from $\epsilon = \beta \cdot \sigma_{\text{total}}$ to $\epsilon = \sigma_{\text{total}}$; added footnote explaining $\beta$ is already in each $\sigma_i$; updated Phase 2 derivation reference |
| 2. **CRITICAL**: $h_2, h_3$ not valid HOCBF — rate constraint $h_2 = r_f^2 - ((u_f - u_f^{prev})/\Delta t)^2$ depends on control input, not just state | CRITICAL | Reverted state from $\mathbb{R}^7$ to $\mathbb{R}^5$ (no augmentation); rewrote §3.4: rate limits as linear QP bounds (4 rows), not HOCBF constraints; combined QP formulation with 2 HOCBF rows + 4 rate limit rows + saturation bounds; updated §3.8, §3.10, §3.11, Risk Assessment, Paper Structure |
| 3. GP mean $\mu_{GP}$ not utilized — only $\sigma_{GP}$ was used, wasting GP's predictive power | HIGH | Added GP mean correction subsection in §2.5.1: mean-corrected model $\hat{f} = f_0 + \mu_{GP}$; remaining uncertainty $\Delta \hat{f}_j = \Delta f_j - \mu_{GP,j}$ bounded by $\beta \sigma_{GP,j}$; tighter $\epsilon$ |
| 4. PAC-Bayes bound misapplied — should bound $|\Delta f_j - \mu_{GP,j}|$, not $|\Delta f_j|$ | HIGH | Fixed alongside HIGH-1: corrected PAC-Bayes statement from $|\Delta f_j| \le \beta \sigma_{GP,j}$ to $|\Delta f_j - \mu_{GP,j}| \le \beta \sigma_{GP,j}$; added implementation note for $\hat{A}_0, \hat{b}_0$ |
| 5. CCS reward function undefined | HIGH | Added §3.5: weighted multi-objective reward with tracking ($r_{track}$), control effort ($r_{effort}$), and safety violation ($r_{violation}$) terms; specified weights and rationale |
| 6. CCS episode structure undefined | HIGH | Added §3.6: episode length (600 steps/10min), termination conditions (hard limits, state range, divergence), reference trajectory modes (regulation vs load-following), training curriculum |
| 7. Ablation A3 invalid — cannot choose relative degree $m$ for a fixed physics system | HIGH | Replaced A3 from "compare $m=2$ vs $m=3$" to "effect of GP mean correction: $\hat{f} = f_0 + \mu_{GP}$ vs $f_0$ only" — quantifies how much tighter $\epsilon$ becomes with mean correction |
| 8. $|L_{f_0}|_{\text{op}}$ not precisely defined for TAC | MED | Expanded definition: local operator norm as Lipschitz constant of $\phi \mapsto L_{f_0} \phi$; concrete formula $|L_{f_0}|_{\text{op}} = \sup_x \|f_0(x)\| \cdot L_{\nabla \phi}$; numerical estimation via $\|f_0\| \cdot \|\nabla^2 \psi_{i-1}^0\|_F$ over state grid |
| 9. Distillation safety term $h(x')<0$ always 0 on teacher data | MED | Added third term to distillation loss (14): safety margin regularization $\max(0, \epsilon - C(x, \pi_{explicit}))^2$ that is non-zero even on teacher data; reorganized safety terms with $\lambda_{margin} = 10$ |
| 10. §2.3.4 $\partial \Delta f / \partial v$ term inconsistent with recursive bound (3) | MED | Removed $(\partial \Delta f_1 / \partial v) \cdot \Delta f_2$ term from first-order approximation; explained it is $\mathcal{O}(\delta^2)$ (product of $\Delta f$ with derivative of $\Delta f$); consistent with recursive bound which only propagates first-order perturbations |
| 11. Multi-constraint $\epsilon_i(x)$ formal definition incomplete | MED | Added multi-constraint extension paragraph in §2.5.2: each $h_i$ has independent $\epsilon_i(x)$ with its own $\nabla h_i$, $\psi_k^{0,i}$ chain; shared GP posterior with different Jacobian weights; QP has $p$ Robust HOCBF rows $A_0^{(i)} u \le b_0^{(i)} - \epsilon_i$ |
| 12. GP hyperparameter optimization strategy unspecified | MED | Added detailed strategy in §2.5.1: marginal likelihood (Type-II ML) via L-BFGS-B; frequency (every GP update epoch); initialization (median heuristic); bounds; sparse GP joint optimization with inducing points |
| 13. $P_t^{ref}, N_e^{ref}$ undefined | LOW | Added definition in §3.4: constant setpoint mode ($P_t^{ref} = 25$ MPa, $N_e^{ref} = 600$ MW) and load-following mode (time-varying from load profile); safety margins $\Delta P_{max} = 0.5$ MPa, $\Delta N_{max} = 30$ MW |
| 14. Two-timescale schedule $K=10$ within $M=50$ overlap unclear | LOW | Clarified: $K$ episodes counted within $M$-episode cycle (not additional); effective schedule: [GP update] → [10 re-adapt episodes @ 0.5× LR] → [40 normal episodes] → [next GP update] |
| 15. Eq (4) "control-direction gap" naming misleading under $\Delta g = 0$ | LOW | Renamed from "control-direction gap" to "$\Delta f$-induced coupling coefficient gap" — the term arises from $\Delta f$ affecting $L_g L_f^{m-1} h$, not from $\Delta g$ changing the control direction |

---

## Appendix E: Summary of Changes from Self-Review v3→v4

| Issue | Severity | Fix Applied |
|-------|----------|-------------|
| 1. **CRITICAL**: $\epsilon(x)$ sign error — eqs (6)-(8), KKT (9) used $+\epsilon$ (relaxed constraint) instead of $-\epsilon$ (conservative) | CRITICAL | Changed all instances: eq (6) $+\epsilon \to -\epsilon$; eq (7) $+\epsilon \to -\epsilon$; eq (8) $b_0+\epsilon \to b_0-\epsilon$; KKT eq (9) $-b_0-\epsilon \to -b_0+\epsilon$; all text references updated. Added sign convention rationale paragraph citing Taylor et al. 2020 and Jankovic 2018 |
| 2. First-order perturbation bound drops $\mathcal{O}(\delta^2)$ without quantitative justification | HIGH | Added Assumption 3 (Small perturbation) with explicit $\rho_{max}$ bound; quantified second-order residual as $c \cdot \rho_{max}^2 \cdot \|f_0\|^2$; stated CCS 30% perturbation gives $\rho_{max} \le 0.3$; added empirical verification criterion in Phase 2 |
| 3. Equation (12) coefficients $c_j$ undefined | HIGH | Defined $c_j = \prod_{i=j+1}^{m-1}(|L_{f_0}|_{\text{op},i} + k_i)$ as accumulated gain from level $j$ to $m-1$; gave explicit values for $m=2$ and $m=3$ |
| 4. $\sigma_{\text{ctrl}}(x)$ depends on $\|u^*\|$ (circular dependency) | HIGH | Replaced $\|u^*\|$ with $\|u_{RL}\|$ as upper bound; justified: $\|u_{RL}\| \ge \|u^*\|$ under typical QP corrections, avoids circularity |
| 5. PAC-Bayes $\beta$ assumes i.i.d. data but RL trajectories are time-correlated | HIGH | Added $\beta$-mixing subsampling protocol; batch acquisition from independent episodes; estimated $\tau_{mix}$ from autocorrelation |
| 6. Two-timescale convergence claim "$\epsilon$ decreases monotonically" is wrong | HIGH | Revised to "$\epsilon$ non-increasing in the asymptotic regime"; derived $\epsilon = O(\sqrt{\ln N / N}) \to 0$; explained $\sigma$ decreases but $\beta$ increases with $N$ |
| 7. Distillation provides only empirical safety guarantee | MED | Added theoretical argument: if $\varepsilon_{distill} < \eta / L_C$ (distillation error less than margin/Lipschitz ratio), deterministic safety guarantee holds; estimated $L_C = \|L_g L_{f_0}^{m-1} h\|$ analytically |
| 8. $h_2$ degeneracy at steady state ($L_g h_2 = 0$) | MED | Added 4-point analysis: (1) constraint in interior, (2) only degenerate at zero rate, (3) QP row becomes trivial, (4) added $\gamma_{reg}$ regularization for numerical stability |
| 9. Assumption 2 (known $\bar{\Delta}_j$) contradicts GP methodology | MED | Replaced "known bounds" with data-driven GP bounds: $\bar{\Delta}_j(x) = \beta \cdot \sigma_{GP,j}(x)$; added explanation that PAC-Bayes guarantee provides probabilistic bounds, consistent with GP framework |
| 10. Double integrator derivation contains $2\Delta f_1^2/x$ term undefined at $x=0$ | MED | Removed $2\Delta f_1^2/x$ term — identified as $\mathcal{O}(\delta^2)$ under first-order approximation; added explanation that term is undefined at $x=0$, confirming it must be excluded |
| 11. CCS model source undetermined | MED | Selected Tan et al. as primary choice with concrete comparison table against Bell-Åström; defined decision timeline (Phase 3 Week 1) and fallback plan |
| 12. NMPC timing should exclude IPOPT warmup | LOW | Added note: first solve discarded (includes JIT/factorization overhead); fair comparison with warm-started deployed NMPC |

---

## Appendix F: Summary of Changes from Self-Review v2→v3

| Issue | Severity | Fix Applied |
|-------|----------|-------------|
| 1. Core thesis "zero violation" vs probabilistic guarantee | HIGH | Rewrote Section 1.1 with explicit $\delta$ qualifier; added "Key qualifier" paragraph; updated all "zero violation" references to "probabilistic safety guarantee with $\delta \le 0.01$" |
| 2. $\Delta g=0$ rationale insufficient | HIGH | Added Assumption 1 with detailed rationale (3 paragraphs: primary CCS uncertainty source, matched uncertainty argument, future work for $\Delta g \ne 0$); added Assumption 2 (bounded $\Delta f$) |
| 3. $\epsilon(x)$ formula incomplete — undefined higher-order terms | HIGH | Added Section 2.3.1 (full $\psi$-chain uncertainty propagation with recursive analysis); Section 2.3.2 (compositional $\epsilon(x)$ definition); Section 2.3.4 ($m=2$ special case with explicit derivation) |
| 4. n-GP to scalar $\epsilon$ aggregation missing | HIGH | Added Section 2.5.2 with equations (10)-(12): per-level aggregation via error propagation, recursive aggregation, total $\sigma_{\text{total}}$ formula |
| 5. GP online update non-stationarity | HIGH | Added Section 2.5.4: two-timescale protocol (fast RL / slow GP), re-adaptation period, convergence argument; added to Risk Assessment |
| 6. $\psi$-chain computed with $f_0$ but true uses $f$ | HIGH | Resolved by Section 2.3.1: defined $\delta_i = \psi_i - \psi_i^0$, recursive perturbation bound (3), which feeds into compositional $\epsilon$ |
| 7. QP constraint sign inconsistency | MED | Added explicit derivation from HOCBF $\ge 0$ form (1) to QP $\le$ form (2) in Section 2.2; sign convention bridge paragraph; consistent notation $A_0, b_0$ throughout |
| 8. Phase 2 single $\Delta f$ scenario insufficient | MED | Added 4 uncertainty scenarios (S1-S4) with different $\Delta f$ structures; rationale for each; mapping to CCS |
| 9. CCS actuator rate constraints not proper HOCBF | MED | State augmentation: $x \in \mathbb{R}^5 \to \mathbb{R}^7$ with $u_f^{prev}, u_t^{prev}$; rate constraints become proper state constraints with $m=1$ |
| 10. Distillation loss requires environment interaction | MED | Clarified: teacher trajectory data provides $(x, u^*, x')$ triples; safety term uses teacher's $x'$ (approximation justified); added optional on-policy fine-tuning; post-distillation verification as safety net |

---
title: "GP-Kalman — Infinite-Dimensional Kalman Filtering for GP Regression"
authors: [Simo Särkkä, Jouni Hartikainen]
year: 2013
venue: AISTATS 2012 (JMLR W&CP 22)
tags: [gaussian-process, kalman-filtering, state-space, online-gp, spatio-temporal]
sources: [sarkka2013a]
updated: 2026-05-19
---

## One-Line Summary

Spatio-temporal Gaussian process regression can be reformulated as infinite-dimensional Kalman filtering and RTS smoothing, reducing computational cost from $O(T^3)$ to $O(T)$ in the number of time steps.

## Problem Setting

Gaussian process (GP) regression is a Bayesian nonparametric method where model functions are drawn from a GP prior $\text{GP}(m_0(\mathbf{x}), C_0(\mathbf{x}, \mathbf{x}'))$. In the spatio-temporal setting, the process $f(\mathbf{x}, t)$ depends on spatial input $\mathbf{x} \in \mathbb{R}^d$ and time $t$, with prior mean $m_0(\mathbf{x}, t)$ and covariance $C_0(\mathbf{x}, t; \mathbf{x}', t')$.

The core computational challenge is that direct GP regression over $M$ spatial measurements per time step and $T$ time points requires inverting the full joint covariance matrix, yielding a cost of $O(M^3 T^3)$. Even after marginalizing the spatial dimension, the temporal cost alone is $O(T^3)$, which is infeasible for long time series or streaming data.

The model considered is:

$$f(\mathbf{x}, t) \sim \text{GP}\bigl(m_0(\mathbf{x}, t),\; C_0(\mathbf{x}, t; \mathbf{x}', t')\bigr)$$

$$y_k = H_k f(\mathbf{x}, t_k) + e_k, \quad e_k \sim \mathcal{N}(0, \Sigma_k)$$

where $H_k$ are linear functionals mapping the infinite-dimensional field to finite-dimensional observations.

## Key Contributions

1. **Infinite-dimensional Kalman filter/smoother for GP regression**: Shows that spatio-temporal GP regression is formally equivalent to infinite-dimensional Kalman filtering and Rauch-Tung-Striebel (RTS) smoothing, providing the same posterior but at linear cost in time.
2. **Covariance-to-SDE conversion procedure**: Presents a systematic method for converting spatio-temporal covariance functions into infinite-dimensional stochastic (pseudo-)differential equations via spectral factorization in the temporal frequency variable.
3. **Separable covariance specialization**: Proves that when $C(\mathbf{x}, t) = C(\mathbf{x})\,C(t)$, the infinite-dimensional problem reduces to a finite-dimensional state-space model solvable with standard Kalman filters.
4. **Complexity reduction from $O(T^3)$ to $O(T)$**: The temporal scaling becomes linear in the number of time steps; spatial cost ranges from $O(MT)$ with sparse approximations to $O(M^3 T)$ without.
5. **Empirical validation**: Demonstrates accuracy on the Cressie & Huang nonseparable covariance and on real US precipitation/temperature data, showing 20%+ MSE reduction by including temporal dynamics.

## Core Formulation

### From GP covariance to infinite-dimensional SDEs

Given a stationary spatio-temporal covariance $C(\mathbf{x}, t)$, compute its spectral density $S(\boldsymbol{\omega}_x, \omega_t)$ via Fourier transform. The key step is spectral factorization with respect to $\omega_t$: find a rational transfer function

$$G(i\boldsymbol{\omega}_x, i\omega_t) = \frac{b_0(i\boldsymbol{\omega}_x)}{(i\omega_t)^N + a_{N-1}(i\boldsymbol{\omega}_x)(i\omega_t)^{N-1} + \cdots + a_0(i\boldsymbol{\omega}_x)}$$

such that $S(\boldsymbol{\omega}_x, \omega_t) \approx G(i\boldsymbol{\omega}_x, i\omega_t)\,G(-i\boldsymbol{\omega}_x, -i\omega_t)$. When the spectral density does not already have this form, a Taylor expansion of $1/S$ in $(i\omega_t)^2$ yields the polynomial coefficients.

Taking the inverse Fourier transform with respect to $\mathbf{x}$ then yields the infinite-dimensional stochastic evolution equation:

$$d\mathbf{f}(\mathbf{x}, t) = \begin{pmatrix} 0 & 1 & & \\ & \ddots & \ddots & \\ & & 0 & 1 \\ -\mathcal{A}_0 & -\mathcal{A}_1 & \cdots & -\mathcal{A}_{N-1} \end{pmatrix} \mathbf{f}(\mathbf{x}, t)\,dt + \begin{pmatrix} 0 \\ \vdots \\ 0 \\ 1 \end{pmatrix} dW(\mathbf{x}, t)$$

where $\mathcal{A}_j = \mathcal{F}_x^{-1}[a_j(i\boldsymbol{\omega}_x)]$ are linear operators (differential, pseudo-differential, or integro-differential) on the spatial domain, and $W(\mathbf{x}, t)$ is a Hilbert-space-valued Wiener process with diffusion operator $Q_c(\mathbf{x}, \mathbf{x}') = \mathcal{F}_x^{-1}[|b_0(i\boldsymbol{\omega}_x)|^2]$.

### State-space representation

The infinite-dimensional state-space model is:

$$d\mathbf{f}(\mathbf{x}, t) = \mathcal{A}\,\mathbf{f}(\mathbf{x}, t)\,dt + L\,dW(\mathbf{x}, t)$$

$$y_k = H_k\,\mathbf{f}(\mathbf{x}, t_k) + e_k$$

where $\mathcal{A}$ is an $s \times s$ matrix of spatial operators, $L \in \mathbb{R}^{s \times q}$, $H_k$ are functionals, and the prior is $\mathbf{f}(\mathbf{x}, t_0) \sim \text{GP}(m_0(\mathbf{x}), C_0(\mathbf{x}, \mathbf{x}'))$.

### Discrete-time transition (exact, not approximated)

$$\mathbf{f}(\mathbf{x}, t_k) = \mathcal{U}(\Delta t_k)\,\mathbf{f}(\mathbf{x}, t_{k-1}) + v_k(\mathbf{x})$$

where $\mathcal{U}(\Delta t) = \exp(\Delta t\,\mathcal{A})$ is the evolution operator (operator exponential) and $v_k(\mathbf{x}) \sim \text{GP}(0, Q(\mathbf{x}, \mathbf{x}'; \Delta t_k))$ with

$$Q(\mathbf{x}, \mathbf{x}'; \Delta t) = \int_0^{\Delta t} \mathcal{U}(\Delta t - \tau)\,L\,Q_c(\mathbf{x}, \mathbf{x}')\,L^T\,\mathcal{U}^*(\Delta t - \tau)\,d\tau.$$

### Kalman filtering equations for GP

**Filtering** (forward pass, $k = 1, \ldots, T$):

$$m_k^-(\mathbf{x}) = \mathcal{U}(\Delta t_k)\,m_{k-1}(\mathbf{x})$$

$$C_k^-(\mathbf{x}, \mathbf{x}') = \mathcal{U}(\Delta t_k)\,C_{k-1}(\mathbf{x}, \mathbf{x}')\,\mathcal{U}^*(\Delta t_k) + Q(\mathbf{x}, \mathbf{x}'; \Delta t_k)$$

$$m_k(\mathbf{x}) = m_k^-(\mathbf{x}) + C_k^-(\mathbf{x}, \mathbf{x}')\,H_k^*\,\bigl[H_k\,C_k^-(\mathbf{x}, \mathbf{x}')\,H_k^* + \Sigma_k\bigr]^{-1}(y_k - H_k\,m_k^-(\mathbf{x}))$$

$$C_k(\mathbf{x}, \mathbf{x}') = C_k^-(\mathbf{x}, \mathbf{x}') - C_k^-(\mathbf{x}, \mathbf{x}')\,H_k^*\,\bigl[H_k\,C_k^-(\mathbf{x}, \mathbf{x}')\,H_k^* + \Sigma_k\bigr]^{-1}\,H_k\,C_k^-(\mathbf{x}, \mathbf{x}')$$

**Smoothing** (backward pass, $k = T-1, \ldots, 0$):

$$G_k(\mathbf{x}) = C_k(\mathbf{x}, \mathbf{x}')\,\mathcal{U}^*(\Delta t_k)\,\bigl[C_{k+1}^-(\mathbf{x}, \mathbf{x}')\bigr]^{-1}$$

$$m_k^s(\mathbf{x}) = m_k(\mathbf{x}) + G_k\bigl(m_{k+1}^s(\mathbf{x}) - m_{k+1}^-(\mathbf{x})\bigr)$$

$$C_k^s(\mathbf{x}, \mathbf{x}') = C_k(\mathbf{x}, \mathbf{x}') + G_k\bigl(C_{k+1}^s(\mathbf{x}, \mathbf{x}') - C_{k+1}^-(\mathbf{x}, \mathbf{x}')\bigr)\,G_k^*(\mathbf{x})$$

The smoothed marginals give the GP posterior: $p(f(\mathbf{x}^*, t_k) \mid y_1, \ldots, y_T) = \mathcal{N}(m_k^s(\mathbf{x}^*),\; C_k^s(\mathbf{x}^*, \mathbf{x}^*))$.

### Complexity reduction

- **Direct GP**: $O(M^3 T^3)$ for the full spatio-temporal covariance matrix inversion.
- **Proposed Kalman/RTS**: $O(T)$ in the number of time steps; spatial cost depends on approximation: $O(MT)$ with sparse (FIC) approximations, $O(M^3 T)$ without.
- With $M = 50$ eigenfunctions and $T = 500$ time steps, the Kalman/RTS approach is orders of magnitude lighter than direct GP regression.

## Theoretical Results

### Equivalence of GP regression and Kalman smoothing

The infinite-dimensional Kalman smoother output $m_k^s(\mathbf{x})$, $C_k^s(\mathbf{x}, \mathbf{x}')$ is exactly the GP posterior mean and covariance conditioned on all observations $y_1, \ldots, y_T$. The Kalman filter alone gives the forward (causal) posterior; the RTS smoother is needed for the full posterior using all data. This extends the scalar-input result of Hartikainen and Sarkka (2010) to the spatio-temporal setting.

### Separable covariance functions yield finite-dimensional models

When $C(\mathbf{x}, t) = C(\mathbf{x})\,C(t)$, the spectral density factorizes as $S(\boldsymbol{\omega}_x, \omega_t) = S(\boldsymbol{\omega}_x)\,S(\omega_t)$, and the transfer function becomes:

$$G(i\boldsymbol{\omega}_x, i\omega_t) = \frac{b_0(i\boldsymbol{\omega}_x)}{(i\omega_t)^N + a_{N-1}(i\omega_t)^{N-1} + \cdots + a_0}$$

with $|b_0(i\boldsymbol{\omega}_x)|^2 = \text{const} \times S(\boldsymbol{\omega}_x)$ and constant $a_j$. The resulting SDE has the "diagonal" form where $\mathcal{A}$ is a constant matrix (not spatially varying operators), and the diffusion operator of $W(\mathbf{x}, t)$ is proportional to $C(\mathbf{x})$. Evaluating at a finite set of spatial points $\{\mathbf{x}_1, \ldots, \mathbf{x}_n\}$ reduces the model to a finite-dimensional state-space system solvable with standard Kalman filters.

### Exact discretization

The discrete-time transition model is the exact mild solution to the infinite-dimensional SDE (not an Euler-type approximation), ensuring no discretization error in the temporal propagation step.

## Algorithm / Implementation

### Conversion procedure (covariance function to SDE)

1. **Fourier transform**: Compute $S(\boldsymbol{\omega}_x, \omega_t) = \mathcal{F}[C(\mathbf{x}, t)]$.
2. **Rational approximation in $i\omega_t$**: If $S$ is not already rational, Taylor expand $1/S$ in $(i\omega_t)^2$ to obtain polynomial coefficients $a_j(i\boldsymbol{\omega}_x)$ and $b_0(i\boldsymbol{\omega}_x)$.
3. **Spectral factorization**: Factor with respect to $i\omega_t$ to select the causal (stable) transfer function $G(i\boldsymbol{\omega}_x, i\omega_t)$ with poles only in the upper half plane.
4. **Inverse Fourier transform**: Compute operators $\mathcal{A}_j = \mathcal{F}_x^{-1}[a_j(i\boldsymbol{\omega}_x)]$ and diffusion $Q_c(\mathbf{x}, \mathbf{x}') = \mathcal{F}_x^{-1}[|b_0(i\boldsymbol{\omega}_x)|^2]$.
5. **Assemble state-space**: Write the infinite-dimensional SDE with the companion-form matrix of operators $\mathcal{A}_0, \ldots, \mathcal{A}_{N-1}$.

### Kalman filter/smoother for GP (numerical implementation)

1. **Spatial discretization**: Project onto $M$ basis functions (eigenfunctions of the Laplacian, Galerkin/FEM basis, or point collocation). This converts operators to $M \times M$ matrices.
2. **Forward filter**: For $k = 1, \ldots, T$, apply the predict-update Kalman steps using the discretized operators.
3. **Backward smoother**: For $k = T-1, \ldots, 0$, apply the RTS smoothing pass.
4. **Prediction at test points**: Include test time points as measurement-less steps; evaluate the spatial basis at test locations $\mathbf{x}^*$ to read off marginals $m_k^s(\mathbf{x}^*)$, $C_k^s(\mathbf{x}^*, \mathbf{x}^*)$.
5. **Hyperparameter learning**: Marginal likelihood $p(y_1, \ldots, y_T) = \prod_{k=1}^T \mathcal{N}(y_k \mid H_k m_k^-(\mathbf{x}),\; H_k C_k^-(\mathbf{x}, \mathbf{x}') H_k^* + \Sigma_k)$ is available as a byproduct of filtering.

### Example: Cressie & Huang covariance

For the nonseparable covariance $C(\mathbf{x}, t) = \frac{\sigma^2}{(a^2 t^2 + 1)^{d/2}} \exp\!\bigl(-\frac{b^2 \|\mathbf{x}\|^2}{a^2 t^2 + 1}\bigr)$, the conversion yields the pseudo-differential evolution equation:

$$\frac{\partial}{\partial t}\begin{pmatrix} f \\ \dot{f} \end{pmatrix} = \begin{pmatrix} 0 & 1 \\ \sqrt{2}\,(a/b)^2 \nabla^2 & -2^{5/4}\cos(\pi/8)\,(a/b)\sqrt{-\nabla^2} \end{pmatrix}\begin{pmatrix} f \\ \dot{f} \end{pmatrix} + \begin{pmatrix} 0 \\ 1 \end{pmatrix} w(\mathbf{x}, t)$$

where $\nabla^2$ is the Laplacian and $\sqrt{-\nabla^2}$ is its fractional square root.

## Limitations

- **Approximation in conversion**: The spectral factorization step typically involves Taylor expansion or other polynomial approximations of the spectral density, introducing approximation error in the covariance function match.
- **Complicated operator equations**: For nonseparable covariances, the resulting SDEs involve pseudo-differential operators (e.g., fractional Laplacians) that require specialized numerical methods and are not always straightforward to implement.
- **Spatial discretization cost**: While linear in $T$, the spatial cost is $O(M^3 T)$ without sparse approximations, where $M$ is the number of spatial basis functions. Sparse approximations (FIC) reduce this to $O(M T)$ but introduce additional approximation.
- **Stationarity assumption**: The conversion procedure assumes stationary covariance functions; nonstationary kernels require additional treatment.
- **Boundary effects**: Basis function expansions on bounded domains introduce boundary effects that may distort the covariance structure near domain edges.

## Relevance to RoCBF-Net

This work is directly relevant to **Phase 2: online GP uncertainty estimation in a real-time control loop**:

- **$O(n)$ instead of $O(n^3)$**: The state-space formulation enables recursive Bayesian updates as new observations arrive, without re-inverting the full covariance matrix. Each new timestep requires only a Kalman predict-update step, making the approach suitable for real-time control.
- **Recursive Bayesian updates**: The Kalman filter naturally processes streaming data---the filtered state $m_k(\mathbf{x})$, $C_k(\mathbf{x}, \mathbf{x}')$ at time $k$ is the sufficient statistic for all past observations. This is ideal for online CBF safety filtering where uncertainty must be updated on-the-fly.
- **Separable covariance simplification**: For spatially structured control problems (e.g., obstacle fields with temporal evolution), separable kernels yield finite-dimensional state-space models that are trivially implementable with standard Kalman filter libraries.
- **Marginal likelihood for hyperparameter tuning**: The marginal likelihood is computed as a filtering byproduct, enabling online adaptation of GP hyperparameters (length scales, signal variance) within the control loop.
- **Connection to CBF safety**: The GP posterior uncertainty from the Kalman filter directly provides the confidence bounds needed for robust CBF constraints (e.g., chance-constrained CBF formulations that require $\sigma_f(\mathbf{x})$ estimates at control points).

## Cross-References

- [[rasmussen2006]] --- Standard GP regression textbook; provides the $O(n^3)$ baseline that this paper improves upon for temporal data.
- [[sarkka2013b]] --- Särkkä's later book "Bayesian Filtering and Smoothing"; general treatment of state-space methods including the finite-dimensional GP-to-Kalman conversion.
- [[hartikainen2010]] --- Predecessor paper by the same authors establishing the scalar-input GP-to-Kalman equivalence.
- [[lindgren2011]] --- Lindgren, Rue, and Lindström's SPDE approach to Gaussian fields; related but does not achieve linear-in-time scaling because the resulting models are not causal/Markovian in time.

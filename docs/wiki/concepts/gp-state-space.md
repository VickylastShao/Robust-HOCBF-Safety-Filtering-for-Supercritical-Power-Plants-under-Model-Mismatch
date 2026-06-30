---
title: GP State-Space Models
tags: [gaussian-process, state-space, kalman-filtering, online-inference]
sources: [rasmussen2006, sarkka2013a, sarkka2013b]
updated: 2026-05-19
---

## Definition

A GP state-space model converts a Gaussian process regression problem into an (infinite-dimensional) state-space form, enabling recursive Bayesian filtering with $O(n)$ complexity instead of the standard $O(n^3)$.

## Standard GP Regression

Given data $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$ with $y_i = f(x_i) + \epsilon_i$, the GP posterior is:

$$\mu_*(x) = k_*^T (K + \sigma_n^2 I)^{-1} y$$
$$\sigma_*^2(x) = k_{**} - k_*^T (K + \sigma_n^2 I)^{-1} k_*$$

Complexity: $O(n^3)$ for matrix inversion, $O(n^2)$ per prediction. Infeasible for online control.

## GP → State-Space Conversion (Särkkä)

**Key insight**: A stationary GP with covariance $C(x, x')$ can be represented as the solution of a stochastic differential equation (SDE). For temporal GP regression:

1. Compute spectral density $S(\omega)$ from covariance $C(\tau)$ via Fourier transform
2. Spectral factorization: $S(\omega) = G(i\omega) q_c G(-i\omega)$ where $q_c$ is the white noise intensity
3. Inverse Fourier transform yields the state-space SDE:
   $$\dot{z} = A z + L w(t)$$
   where $w(t)$ is white noise, $A$ is the state matrix from the spectral factorization

4. For **separable** covariance $C(\mathbf{x}, t) = C_{\mathbf{x}}(\mathbf{x}) C_t(t)$: the temporal part yields a **finite-dimensional** state-space model with constant matrices.

## Kalman Filter for GP

Once in state-space form, standard Kalman filtering applies:

**Predict**:
$$\hat{z}_{t|t-1} = F \hat{z}_{t-1|t-1}, \quad P_{t|t-1} = F P_{t-1|t-1} F^T + Q$$

**Update**:
$$K_t = P_{t|t-1} H^T (H P_{t|t-1} H^T + R)^{-1}$$
$$\hat{z}_{t|t} = \hat{z}_{t|t-1} + K_t (y_t - H \hat{z}_{t|t-1})$$
$$P_{t|t} = (I - K_t H) P_{t|t-1}$$

**Complexity**: $O(d^3)$ per step where $d$ is the state dimension (typically 1-3 for common kernels), independent of $n$.

## Common Kernel State-Space Representations

| Kernel | State Dimension | Drift Matrix $A$ |
|--------|----------------|-------------------|
| SE (Gaussian) | Infinite (needs truncation) | Approximate via Matérn limit |
| Matérn 1/2 (Exponential) | 1 | $A = -\lambda$ |
| Matérn 3/2 | 2 | $A = \begin{bmatrix} 0 & 1 \\ -\lambda^2 & -2\lambda \end{bmatrix}$ |
| Matérn 5/2 | 3 | $A = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ -\lambda^3 & -3\lambda^2 & -3\lambda \end{bmatrix}$ |

## Variants Across Papers

| Paper | Contribution |
|-------|-------------|
| [[rasmussen2006]] | Standard GP formulas, $O(n^3)$; sparse approximations (Ch. 8) for reducing cost |
| [[sarkka2013a]] | GP→infinite-dimensional Kalman filter; spectral factorization procedure; $O(T)$ temporal cost |
| [[sarkka2013b]] | Journal extension: systematic spectral factorization, explicit matrix examples, non-linear/non-Gaussian extensions |

## Implementation Notes for RoCBF-Net

- **Phase 2**: Use Matérn 5/2 kernel in state-space form (3D state) for online GP uncertainty estimation
- **Online update**: Kalman filter step each control iteration — $O(1)$ per step (constant state dimension)
- **Posterior mean** $\mu_*(x)$ → replaces $\hat{\Delta}$ in Das 2024's robust HOCBF
- **Posterior variance** $\sigma_*^2(x)$ → replaces $\|e(t)\|$ as uncertainty bound $\epsilon(x)$
- **ARD (Automatic Relevance Determination)**: Learn per-dimension length scales via marginal likelihood optimization

## Open Questions

1. How to handle multi-output GP (uncertainty in both $f$ and $g$)?
2. Hyperparameter learning online — marginal likelihood gradient via Kalman filter?
3. Non-stationary kernels for time-varying uncertainty?
4. Sparse approximation for large observation histories — forgetting factor?

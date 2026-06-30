---
title: GPML — Gaussian Processes for Machine Learning
authors: [Carl Edward Rasmussen, Christopher K. I. Williams]
year: 2006
venue: MIT Press
tags: [gaussian-process, regression, covariance, bayesian, textbook]
sources: [rasmussen2006]
updated: 2026-05-19
---

## One-Line Summary

The definitive textbook on Gaussian processes, providing the complete Bayesian framework for GP regression, covariance function design, hyperparameter learning, and sparse approximations — forming the theoretical foundation for using GPs to model unmodeled dynamics with calibrated uncertainty.

## Problem Setting

Given a training set $\mathcal{D} = \{(\mathbf{x}_i, y_i) \mid i = 1, \ldots, n\}$ where $\mathbf{x} \in \mathbb{R}^D$ is an input vector and $y \in \mathbb{R}$ is a noisy scalar output:

$$y = f(\mathbf{x}) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma_n^2)$$

The goal is to infer the latent function $f$ and make predictions at test inputs $\mathbf{x}_*$. Unlike parametric approaches that estimate point parameters, GP regression places a distribution over functions directly, yielding both a predictive mean and a calibrated predictive variance at every test point.

## Key Concepts

**Gaussian Process (Definition 2.1).** A Gaussian process is a collection of random variables, any finite number of which have a joint Gaussian distribution. A GP is fully specified by its mean function $m(\mathbf{x})$ and covariance function $k(\mathbf{x}, \mathbf{x}')$:

$$f(\mathbf{x}) \sim \mathcal{GP}\bigl(m(\mathbf{x}),\, k(\mathbf{x}, \mathbf{x}')\bigr)$$

where

$$m(\mathbf{x}) = \mathbb{E}[f(\mathbf{x})], \quad k(\mathbf{x}, \mathbf{x}') = \mathbb{E}\bigl[(f(\mathbf{x}) - m(\mathbf{x}))(f(\mathbf{x}') - m(\mathbf{x}'))\bigr]$$

**GP Prior.** Before observing data, the prior over function values at any set of points $\mathbf{X}_*$ is:

$$\mathbf{f}_* \sim \mathcal{N}\bigl(\mathbf{0},\, K(\mathbf{X}_*, \mathbf{X}_*)\bigr)$$

**GP Posterior.** After conditioning on noisy observations $(\mathbf{X}, \mathbf{y})$, the posterior is also a GP with updated mean and covariance functions (see Core Formulation below).

**Marginal Likelihood.** The probability of the data marginalized over all latent function values:

$$\log p(\mathbf{y} \mid \mathbf{X}) = -\frac{1}{2} \mathbf{y}^\top (K + \sigma_n^2 I)^{-1} \mathbf{y} - \frac{1}{2} \log|K + \sigma_n^2 I| - \frac{n}{2}\log 2\pi$$

This decomposes into a data-fit term (first), a complexity penalty (second), and a constant. It enables automatic model selection without cross-validation.

**Covariance Functions (Kernels).** The kernel encodes assumptions about function smoothness, length-scale, and periodicity. It is the single most important modeling choice in GP regression. Valid kernels must be positive semidefinite: $\int k(\mathbf{x}, \mathbf{x}') f(\mathbf{x}) f(\mathbf{x}') \, d\mu(\mathbf{x}) \, d\mu(\mathbf{x}') \geq 0$ for all $f \in L^2$.

## Core Formulation

### GP Posterior Predictive Distribution

The joint distribution of training targets $\mathbf{y}$ and test function values $\mathbf{f}_*$ under the prior is:

$$
\begin{bmatrix} \mathbf{y} \\ \mathbf{f}_* \end{bmatrix}
\sim \mathcal{N}\left(\mathbf{0},\,
\begin{bmatrix}
K + \sigma_n^2 I & K_* \\
K_*^\top & K_{**}
\end{bmatrix}\right)
$$

where $K = K(\mathbf{X}, \mathbf{X})$, $K_* = K(\mathbf{X}, \mathbf{X}_*)$, and $K_{**} = K(\mathbf{X}_*, \mathbf{X}_*)$. Conditioning yields the key predictive equations:

$$\bar{\mathbf{f}}_* = K_*^\top (K + \sigma_n^2 I)^{-1} \mathbf{y}$$

$$\operatorname{cov}(\mathbf{f}_*) = K_{**} - K_*^\top (K + \sigma_n^2 I)^{-1} K_*$$

For a single test point $\mathbf{x}_*$, with $\mathbf{k}_* = k(\mathbf{X}, \mathbf{x}_*)$:

$$\bar{f}_* = \mathbf{k}_*^\top (K + \sigma_n^2 I)^{-1} \mathbf{y} = \sum_{i=1}^{n} \alpha_i \, k(\mathbf{x}_i, \mathbf{x}_*)$$

where $\boldsymbol{\alpha} = (K + \sigma_n^2 I)^{-1} \mathbf{y}$.

$$\mathbb{V}[f_*] = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^\top (K + \sigma_n^2 I)^{-1} \mathbf{k}_*$$

The predictive variance is the prior variance minus a positive information term. It depends only on inputs, not on observed targets. The posterior mean is a linear predictor (linear combination of observations), also expressible as a linear combination of $n$ kernel functions centered on training points (representer theorem).

### Squared Exponential (SE) Kernel

$$k_{\text{SE}}(r) = \exp\!\left(-\frac{r^2}{2\ell^2}\right)$$

with characteristic length-scale $\ell$. The full noisy kernel is:

$$k_y(\mathbf{x}_p, \mathbf{x}_q) = \sigma_f^2 \exp\!\left(-\frac{|\mathbf{x}_p - \mathbf{x}_q|^2}{2\ell^2}\right) + \sigma_n^2 \delta_{pq}$$

Properties:
- Infinitely mean-square differentiable (very smooth sample functions)
- Spectral density: $S(s) = (2\pi\ell^2)^{D/2} \exp(-2\pi^2 \ell^2 s^2)$ (decays exponentially)
- Stein (1999) argues this smoothness is unrealistic for many physical processes
- Corresponds to a Bayesian linear model with infinitely many Gaussian basis functions
- The most widely used kernel in kernel machines

### Matern Class Kernel

$$k_{\text{Matern}}(r) = \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2\nu}\, r}{\ell}\right)^{\nu} K_{\nu}\!\left(\frac{\sqrt{2\nu}\, r}{\ell}\right)$$

where $K_\nu$ is a modified Bessel function, $\nu > 0$ controls smoothness, and $\ell$ is the length-scale. The process $f(\mathbf{x})$ is $k$-times MS differentiable if and only if $\nu > k$.

Special cases for half-integer $\nu = p + 1/2$:

**$\nu = 1/2$ (Ornstein-Uhlenbeck / exponential):**
$$k_{\nu=1/2}(r) = \exp\!\left(-\frac{r}{\ell}\right)$$
MS continuous but not MS differentiable. Very rough sample paths.

**$\nu = 3/2$:**
$$k_{\nu=3/2}(r) = \left(1 + \frac{\sqrt{3}\, r}{\ell}\right)\exp\!\left(-\frac{\sqrt{3}\, r}{\ell}\right)$$
Once MS differentiable.

**$\nu = 5/2$:**
$$k_{\nu=5/2}(r) = \left(1 + \frac{\sqrt{5}\, r}{\ell} + \frac{5 r^2}{3\ell^2}\right)\exp\!\left(-\frac{\sqrt{5}\, r}{\ell}\right)$$
Twice MS differentiable. Recommended as the most useful for practical applications — flexible enough to be realistic, simple enough to be distinguishable from SE with finite data.

As $\nu \to \infty$, the Matern kernel converges to the SE kernel.

### Kernel Construction Rules

New valid kernels can be constructed from existing ones (Sec. 4.2.4):

- **Sum:** $k_1 + k_2$ is valid (corresponds to independent additive processes)
- **Product:** $k_1 \cdot k_2$ is valid (corresponds to independent multiplicative processes)
- **Vertical rescaling:** $a(\mathbf{x})\, k(\mathbf{x}, \mathbf{x}')\, a(\mathbf{x}')$ is valid for any deterministic $a(\mathbf{x})$
- **Convolution:** $\int\!\!\int h(\mathbf{x}, \mathbf{z})\, k(\mathbf{z}, \mathbf{z}')\, h(\mathbf{x}', \mathbf{z}') \, d\mathbf{z} \, d\mathbf{z}'$ is valid

These rules enable composing structured kernels (e.g., SE trend + Matern residual + white noise).

### Hyperparameter Learning via Marginal Likelihood

The marginal likelihood automatically trades off data fit against model complexity. Gradients with respect to hyperparameters $\theta_j$ are:

$$\frac{\partial}{\partial \theta_j} \log p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\theta}) = \frac{1}{2} \operatorname{tr}\!\left(\bigl((K + \sigma_n^2 I)^{-1} \mathbf{y} \mathbf{y}^\top (K + \sigma_n^2 I)^{-1} - (K + \sigma_n^2 I)^{-1}\bigr) \frac{\partial K}{\partial \theta_j}\right)$$

This can be computed efficiently via Cholesky factorization (Algorithm 2.1). In practice, hyperparameters are found by conjugate gradient optimization of the log marginal likelihood.

The ARD (automatic relevance determination) parameterization uses a separate length-scale $\ell_d$ per input dimension:

$$k(\mathbf{x}_p, \mathbf{x}_q) = \sigma_f^2 \exp\!\left(-\frac{1}{2} \sum_{d=1}^{D} \frac{(x_{pd} - x_{qd})^2}{\ell_d^2}\right) + \sigma_n^2 \delta_{pq}$$

Large $\ell_d$ effectively removes input dimension $d$ from the model.

## Online/Sequential GP

The textbook does not provide a dedicated chapter on online/sequential GP updating. However, several relevant ideas appear:

**Csato and Opper (2002)** online sparse GP (Sec. 8.3.4): Training examples are processed sequentially. For each new input $\mathbf{x}$, compute novelty as the predictive variance:

$$\text{novelty}(\mathbf{x}) = k(\mathbf{x}, \mathbf{x}) - \mathbf{k}_m(\mathbf{x})^\top K_{mm}^{-1} \mathbf{k}_m(\mathbf{x})$$

If novelty is large, add the point to the active set $I$; otherwise add it to the remainder set $R$. If $|I|$ exceeds a budget, points can be deleted. This yields an $O(m^2)$ per-step update rather than $O(n^3)$.

**Bayesian Committee Machine (BCM)** (Sec. 8.3.5, Tresp 2000): Split the dataset into $p$ partitions. Combine predictions as:

$$\bigl[\operatorname{cov}_q(\mathbf{f}_* \mid \mathcal{D})\bigr]^{-1} = -(p-1) K_{**}^{-1} + \sum_{i=1}^{p} \bigl[\operatorname{cov}(\mathbf{f}_* \mid \mathcal{D}_i)\bigr]^{-1}$$

$$\mathbb{E}_q[\mathbf{f}_* \mid \mathcal{D}] = \operatorname{cov}_q(\mathbf{f}_* \mid \mathcal{D}) \sum_{i=1}^{p} \bigl[\operatorname{cov}(\mathbf{f}_* \mid \mathcal{D}_i)\bigr]^{-1} \mathbb{E}[\mathbf{f}_* \mid \mathcal{D}_i]$$

This is naturally parallelizable and amenable to incremental updates.

**Conjugate gradients** (Sec. 8.3.6): The linear system $(K + \sigma_n^2 I)\mathbf{v} = \mathbf{y}$ can be solved iteratively with CG in $O(kn^2)$ for $k$ iterations, warm-started from previous solutions when new data arrives.

**Derivative observations** (Sec. 9.4): When partial derivative observations are available, the GP framework naturally incorporates them by augmenting the joint covariance matrix:

$$\operatorname{cov}\!\left(f_i, \frac{\partial f_j}{\partial x_j^d}\right) = \frac{\partial k(\mathbf{x}_i, \mathbf{x}_j)}{\partial x_j^d}, \quad \operatorname{cov}\!\left(\frac{\partial f_i}{\partial x_i^d}, \frac{\partial f_j}{\partial x_j^e}\right) = \frac{\partial^2 k(\mathbf{x}_i, \mathbf{x}_j)}{\partial x_i^d \partial x_j^e}$$

This is useful for learning dynamics models where both state and derivative information are observed.

## Computational Complexity

| Operation | Complexity |
|---|---|
| Cholesky of $K + \sigma_n^2 I$ | $O(n^3 / 6)$ |
| Solving triangular systems ($\boldsymbol{\alpha}$) | $O(n^2 / 2)$ |
| Predictive mean per test point | $O(n)$ |
| Predictive variance per test point | $O(n^2)$ (or $O(n)$ after precomputation) |
| Marginal likelihood evaluation | Included in Cholesky |

The $O(n^3)$ inversion cost and $O(n^2)$ storage make exact GP infeasible for $n > 10{,}000$.

**Sparse approximations** (Chapter 8) reduce this using an active set of size $m \ll n$:

| Method | Training Cost | Prediction (mean) | Prediction (var) |
|---|---|---|---|
| Subset of Regressors (SR) | $O(m^2 n)$ | $O(m)$ | $O(m^2)$ |
| Projected Process (PP) | $O(m^2 n)$ | $O(m)$ | $O(m^2)$ |
| Subset of Datapoints (SD) | $O(m^3)$ | $O(m)$ | $O(m^2)$ |
| BCM ($p$ partitions) | $O(pm^3) = O(m^2 n)$ | $O(mn)$ per block | $O(mn)$ per block |

The PP approximation (Sec. 8.3.4) is preferred over SR because it gives a non-degenerate process with proper predictive variance that returns to the prior far from data:

$$\mathbb{V}_q[f(\mathbf{x}_*)] = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_m(\mathbf{x}_*)^\top K_{mm}^{-1} \mathbf{k}_m(\mathbf{x}_*) + \sigma_n^2 \mathbf{k}_m(\mathbf{x}_*)^\top (\sigma_n^2 K_{mm} + K_{mn} K_{nm})^{-1} \mathbf{k}_m(\mathbf{x}_*)$$

## Relevance to RoCBF-Net

**Phase 2: GP for modeling unmodeled dynamics $\Delta(\mathbf{x})$.** In the RoCBF-Net framework, the true system dynamics are $\dot{\mathbf{x}} = f_{\text{nom}}(\mathbf{x}) + \Delta(\mathbf{x})$ where $f_{\text{nom}}$ is a known nominal model and $\Delta(\mathbf{x})$ captures unmodeled effects. GP regression is the natural tool:

- **Posterior mean as $\hat{\Delta}(\mathbf{x})$:** The GP predictive mean $\bar{f}_*$ provides the best estimate of the unmodeled dynamics, directly usable in the CBF constraint as the nominal model correction.

- **Posterior variance as uncertainty bound $\varepsilon(\mathbf{x})$:** The predictive variance $\mathbb{V}[f_*]$ quantifies epistemic uncertainty. For a $k\sigma$ confidence region, $|\Delta(\mathbf{x}) - \hat{\Delta}(\mathbf{x})| \leq k\sqrt{\mathbb{V}[f_*]}$ with high probability. This bound feeds directly into the robust CBF condition.

- **Kernel selection for control systems:** The Matern $\nu = 5/2$ kernel is recommended over the SE kernel because:
  1. Physical dynamics are typically twice differentiable (smooth acceleration) but not infinitely smooth — Matern $\nu = 5/2$ exactly matches this regularity.
  2. The SE kernel's infinite smoothness can underestimate predictive variance, which is dangerous in safety-critical control.
  3. The half-integer Matern kernels have closed-form expressions, avoiding special function evaluation.

- **ARD for state-space relevance:** The ARD parameterization automatically identifies which state dimensions the unmodeled dynamics depend on, improving data efficiency.

- **Online adaptation:** As the system operates and new state-derivative observations become available, the GP must be updated. The Csato-Opper online sparse GP or sliding-window GP with PP approximation are practical choices, maintaining $O(m^2)$ per-step cost with a budgeted active set.

- **Noise model considerations (Ch. 9):** For control systems, noise may be input-dependent (heteroscedastic). Goldberg et al. (1998) place a GP prior on the log variance function, enabling the uncertainty bound $\varepsilon(\mathbf{x})$ to capture both epistemic and aleatoric uncertainty.

- **Derivative observations (Sec. 9.4):** When velocity or acceleration measurements are directly available, they can be incorporated as derivative observations, strengthening the dynamics model without requiring finite differences.

## Cross-References

- [[sarkka2013a]] — Kalman filtering and state-space GPs for temporal/online updating
- [[sarkka2013b]] — Spatiotemporal GP models and sequential inference
- [[das2024]] — RoCBF-Net paper using GP uncertainty in robust CBF control
- [[stein1999]] — Matern class justification; why SE is too smooth for physical processes
- [[csato2002]] — Online sparse GP with novelty-based active set management

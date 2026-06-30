---
title: GP-Bayesian — Infinite-Dimensional Bayesian Filtering and Smoothing
authors: [Simo Särkkä, Arno Solin, Jouni Hartikainen]
year: 2013
venue: IEEE Signal Processing Magazine
tags: [gaussian-process, bayesian-filtering, smoothing, state-space, spatio-temporal]
sources: [sarkka2013b]
updated: 2026-05-19
---

## One-Line Summary

Stationary spatio-temporal Gaussian process regression can be reformulated as infinite-dimensional state-space models, enabling Kalman filtering and smoothing at $O(T)$ time complexity instead of $O(T^3)$.

## Problem Setting

Standard Gaussian process regression over spatio-temporal data suffers from cubic computational complexity $O(N^3)$ in the number of observations. In the spatio-temporal setting with $M$ measurements per time step and $T$ total time steps, this becomes $O(M^3 T^3)$. Signal processing applications require processing long (potentially unbounded) time series, making linear complexity in $T$ essential. The paper asks: when and how can spatio-temporal GP regression be reformulated so that efficient sequential Bayesian filtering and smoothing replaces the costly batch GP solution?

## Key Contributions

1. **Systematic conversion procedure** from stationary covariance functions to (finite- or infinite-dimensional) state-space models via spectral factorization, generalizing the temporal procedure to the full spatio-temporal case.
2. **Infinite-dimensional Kalman filter and RTS smoother** for spatio-temporal GP regression, achieving $O(T)$ time complexity (linear in the number of time steps).
3. **Analytical worked examples** for Matern ($\nu = 3/2$) and squared exponential covariance functions in both 1D temporal and 2D spatio-temporal settings, including explicit SDE and operator matrices.
4. **Extensions** to non-linear/non-Gaussian measurement models and parameter estimation, inheriting the linear-time property from the state-space formulation.
5. **Two real-world applications**: precipitation interpolation over Colorado (55 410 observations) and fMRI brain imaging noise estimation (1 064 960 observations) --- both intractable under standard GP regression.

## Core Formulation

### GP to State-Space Conversion

The central insight is that a stationary Gaussian process $f(t) \sim \mathcal{GP}(0, k(t, t'))$ whose spectral density $S(\omega)$ is a rational function of $\omega^2$ can be represented as the solution of a linear time-invariant stochastic differential equation. The conversion procedure is:

1. Compute the spectral density $S(\omega)$ from the covariance function via Fourier transform.
2. If $S(\omega)$ is not rational in $\omega^2$, approximate it with a rational function (e.g., Taylor series or Pade approximants; needed for the squared exponential kernel).
3. Perform **spectral factorization**: find a stable minimum-phase transfer function $G(i\omega)$ and constant $q_c$ such that $S(\omega) = G(i\omega)\, q_c\, G(-i\omega)$.
4. Convert $G(i\omega)$ into an equivalent state-space model $\frac{d\mathbf{f}(t)}{dt} = \mathbf{A}\,\mathbf{f}(t) + \mathbf{L}\,w(t)$ using control-theoretic canonical forms.

The state vector $\mathbf{f}(t) = (f(t), df/dt, \ldots, d^{n-1}f/dt^{n-1})^\top$ augmented with time derivatives ensures the first component recovers the original GP.

### Infinite-Dimensional Spatio-Temporal Extension

For spatio-temporal processes $f(x,t)$ with stationary covariance $C(x,t)$:

1. Compute the space-time spectral density $S(\omega_x, \omega_t)$.
2. Fix $\omega_x$ and treat $\omega_t \mapsto S(\omega_x, \omega_t)$ as a parametric temporal spectral density, approximating it as rational in $\omega_t^2$.
3. Spectral factorize to obtain $G(i\omega_x, i\omega_t)$ and $\tilde{q}_c(\omega_x)$.
4. Convert to a spatial-Fourier-domain state-space model, then inverse-Fourier-transform the coefficients to obtain **pseudo-differential operators** $\mathcal{A}_j = \mathcal{F}_x^{-1}[a_j(i\omega_x)]$.

This yields the infinite-dimensional state-space model:

$$
\frac{\partial \mathbf{f}(x,t)}{\partial t} = \mathbf{A}\,\mathbf{f}(x,t) + \mathbf{L}\,w(x,t), \quad y_k = \mathbf{H}_k\,\mathbf{f}(x,t_k) + \varepsilon_k
$$

where $\mathbf{A}$ is a matrix of pseudo-differential operators and the driving noise $w(x,t)$ has spatial spectral density $\tilde{q}_c(\omega_x)$.

### Separable Covariance Functions

When the covariance function is separable, $C(x,t) = C_x(x)\,C_t(t)$, the transfer function $G(i\omega_t)$ is independent of $\omega_x$. The operator matrix $\mathbf{A}$ reduces to an ordinary matrix and spatial correlation enters solely through the noise covariance. No additional spatial approximations are needed; the infinite-dimensional filter/smoother is implemented by including all spatial measurement and test points in the state vector.

### Computational Efficiency $O(n)$

For $T$ time steps and state dimension $d$, Kalman filtering and RTS smoothing cost $O(T d^2)$ per step. Since $d$ is fixed by the order of the rational approximation (e.g., $d = 2$ for Matern $\nu = 3/2$), the overall complexity is $O(T)$ --- linear in the number of time steps. This replaces the $O(T^3)$ batch GP solution. Spatial discretization (eigenfunction expansion of the Laplacian) adds a constant factor controlled by the number of basis functions.

## Theoretical Results

- **Equivalence**: For stationary covariance functions with rational spectral densities, the state-space Kalman filter/smoother solution is *exactly* equivalent to the batch GP regression solution (Theorem implicit in Sec. III-A).
- **Matern family**: The Matern covariance with half-integer smoothness $\nu = p + 1/2$ admits an *exact* $p+1$ dimensional state-space representation with no approximation error.
- **Squared exponential**: The SE kernel's spectral density is non-rational (Gaussian in $\omega$). A truncated Taylor series approximation of order $n$ yields an $n$-dimensional state-space model. The approximation error manifests as bias at the origin of the covariance function; order $n = 6$ is typically sufficient.
- **Non-causal SPDEs**: A spatio-temporal covariance function may be a solution of an SPDE (e.g., Whittle's equation), yet direct conversion of that SPDE to a state-space model can yield an *unstable* system. Proper spectral factorization selecting upper-half-plane roots is essential for Markovian (stable) state-space representation.
- **Non-linear/non-Gaussian extensions**: The Kalman filter update can be replaced by non-linear filter updates (e.g., extended/unscented Kalman, sigma-point methods), maintaining the same sequential structure for classification or other non-Gaussian observation models.

## Algorithm / Implementation

1. **Offline --- Model construction**:
   - Choose a stationary covariance function $k(x, t; x', t')$.
   - Compute spectral density; approximate as rational in $\omega_t^2$ if needed.
   - Spectral factorize to obtain transfer function $G(i\omega_x, i\omega_t)$ and noise spectral density $\tilde{q}_c(\omega_x)$.
   - Convert to infinite-dimensional state-space form with pseudo-differential operator matrix $\mathbf{A}$.

2. **Offline --- Spatial discretization**:
   - Truncate the eigenbasis of the Laplace operator on the spatial domain (e.g., 384 eigenfunctions for the precipitation example).
   - This converts the infinite-dimensional system into a finite-dimensional one amenable to numerical computation.

3. **Online --- Filtering and smoothing**:
   - Run the Kalman filter forward through time steps $t_1, \ldots, t_T$.
   - Run the RTS smoother backward to compute the posterior $p(\mathbf{f}(x,t) \mid y_1, \ldots, y_T) = \mathcal{N}(\mathbf{f}(x,t) \mid \mathbf{m}^s(x,t), \mathbf{P}^s(x,t))$.
   - Extract the first state component at spatial points of interest.

4. **Parameter estimation**: Optimize marginal likelihood of the state-space model using standard methods (e.g., EM, gradient-based optimization), also at $O(T)$ cost.

## Limitations

- **Stationarity requirement**: The conversion procedure relies on the Fourier transform and thus applies only to *stationary* covariance functions. Non-stationary extensions require workarounds (embedding in non-stationary PDEs, coordinate warping).
- **Cubic spatial complexity**: While temporal complexity is $O(T)$, spatial complexity remains cubic in the number of spatial measurement points unless combined with sparse approximations or basis function expansions.
- **Rational approximation for SE kernel**: The squared exponential kernel requires an approximation whose error induces bias at short lags. The choice of approximation order $n$ trades off accuracy against state dimension.
- **Numerical pseudo-differential operators**: The infinite-dimensional filter/smoother cannot be implemented exactly; spatial discretization via eigenfunction expansion introduces approximation error controlled by the number of basis functions.
- **Non-separable covariance functions**: Require pseudo-differential operators in $\mathbf{A}$, complicating numerical implementation compared to the separable case where $\mathbf{A}$ is an ordinary matrix.

## Relevance to RoCBF-Net

This paper complements [[sarkka2013a]] (the AISTATS 2012 conference paper by Sarkka and Hartikainen) with a more systematic and tutorial treatment aimed at the signal processing community. Key additions beyond [[sarkka2013a]]:

- **Complete spectral factorization procedure**: The journal version provides the full algorithmic recipe for converting *any* stationary spatio-temporal covariance function into state-space form, including the spectral factorization step that was only briefly sketched in the conference paper.
- **Worked examples with explicit matrices**: Examples 1--4 give concrete SDE matrices, operator expressions, and transfer functions for Matern and SE kernels in both 1D and 2D, making the method directly implementable.
- **Non-causal SPDE discussion** (Section III-E): Clarifies why direct conversion of an SPDE like Whittle's can yield unstable state-space models, motivating the spectral factorization approach. This subtlety is absent from the conference paper.
- **Non-linear/non-Gaussian extension** (Section IV-B): Explicitly discusses how to replace the Kalman update with non-linear filter updates for classification or other non-Gaussian observation models.
- **Parameter estimation** (Section IV-C): Discusses how state-space model parameter estimation methods inherit the $O(T)$ complexity, with references to appropriate methods.
- **Real-world applications**: Two large-scale demonstrations (precipitation: 55K observations; fMRI: 1M observations) that go beyond the smaller-scale experiments in [[sarkka2013a]].

For RoCBF-Net, the state-space reformulation of GP priors is directly relevant wherever Gaussian process components appear in the control or estimation pipeline. The $O(T)$ complexity enables deployment over long horizons, and the explicit Matern state-space forms provide ready-made dynamic models for CBF-related uncertainty quantification. The non-linear extension pathway is important for handling the safety-constrained (non-Gaussian) observation models that arise in control barrier function settings.

## Cross-References

- [[rasmussen2006]] --- Standard reference for Gaussian process regression; provides the $O(N^3)$ baseline that this paper improves upon.
- [[sarkka2013a]] --- Conference precursor (AISTATS 2012) introducing infinite-dimensional Kalman filtering for spatio-temporal GP regression; this paper is the expanded journal treatment.

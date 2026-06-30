---
title: Robust CBF Methods
tags: [robust-cbf, uncertainty, parametric, estimator, gp]
sources: [cohen2022, das2024, choi2021]
updated: 2026-05-19
---

## Definition

Robust CBF methods extend standard CBFs to systems with uncertainty, ensuring safety guarantees hold despite model errors. Three main approaches exist in our reference papers.

## Approaches Overview

| Approach | Paper | Uncertainty Type | Method | Complexity |
|----------|-------|-----------------|--------|------------|
| HO-RaCBF | [[cohen2022]] | Parametric: $\dot{x}=f+Y\theta+gu$ | Robustness buffer $\|L_Y \psi\| \nu(t)$ | Low (concurrent learning) |
| Robust CBF+Estimator | [[das2024]] | Unmodeled: $\dot{x}=\hat{f}+\hat{g}u+\Delta$ | Estimator $\hat{\Delta}=\Lambda x - \xi$ + error bound | Low (observer) |
| CBVF (HJ-based) | [[choi2021]] | Bounded disturbance | HJ reachability value function | High (curse of dim) |

## Method 1: HO-RaCBF (Cohen & Belta 2022)

**System**: $\dot{x} = f(x) + Y(x)\theta + g(x)u$, unknown $\theta \in \Theta$

**Robustness buffer**: $\|L_Y \psi_{r-1}(x)\| \nu(t)$ where:
- $L_Y \psi_{r-1}$ is the Lie derivative of $\psi_{r-1}$ along the uncertainty direction $Y(x)$
- $\nu(t) = \|\hat{\theta}(t) - \theta\|$ decays exponentially via concurrent learning
- Requires **matching condition**: IRD = DRD (uncertainty enters through control channel)

**HO-RaCBF-QP constraint**:
$$L_f \psi_{r-1} + L_Y \psi_{r-1} \hat{\theta} + L_g \psi_{r-1} u \geq -\alpha_r(\psi_{r-1}) + \|L_Y \psi_{r-1}\| \nu(t)$$

**Pros**: Clean theoretical framework, asymptotic convergence of $\nu(t) \to 0$
**Cons**: Parametric uncertainty only, matching condition required, no unmodeled dynamics

## Method 2: Robust CBF with Estimator (Das & Burdick 2024)

**System**: $\dot{x} = \hat{f}(x) + \hat{g}(x)u + \Delta(x,u)$, unknown $\Delta$

**Uncertainty estimator**: $\hat{\Delta}(t) = \Lambda x - \xi(t)$, $\dot{\xi} = \Lambda(\hat{f} + \hat{g}u + \hat{\Delta})$

**Error bound**: $\|e(t)\| = \|\Delta(t) - \hat{\Delta}(t)\| \leq \delta(t)$ (exponentially converging)

**Robust CBF constraint**:
$$L_{\hat{f}} h + L_{\hat{g}} h \cdot u + \frac{\partial h}{\partial x} \hat{\Delta} - \left\|\frac{\partial h}{\partial x}\right\| \|e(t)\| \geq -\alpha(h)$$

**HOCBF extension (Corollary 1)**:
$$L_f^m h + L_{\hat{g}} L_f^{m-1} h \cdot u + \frac{\partial L_f^{m-1} h}{\partial x} \hat{\Delta} - \left\|\frac{\partial L_f^{m-1} h}{\partial x}\right\| \|e(t)\| + O(h) \geq -\alpha_m(\psi_{m-1})$$

**Pros**: Handles unmodeled dynamics, not restricted to parametric form, extends to HOCBF
**Cons**: Still requires matching condition for Method 1 (compensation), error bound depends on known Lipschitz constants, no learning component

## Method 3: CBVF (Choi et al. 2021)

**System**: $\dot{x} = f(x) + g(x)u + d(x,t)$, bounded disturbance $d$

**Approach**: Solve HJI variational inequality offline to compute CBVF, then use CBVF-QP online.

**CBVF definition**: $B_\gamma(x) = \sup_{u \in \mathcal{U}} \inf_{d \in \mathcal{D}} \sup_{t \geq 0} e^{\gamma t} h(\phi(t; x, u, d))$

**CBVF-QP**: $\min_u \|u - u_{\text{nom}}\|^2$ s.t. $\inf_d \dot{B}_\gamma + \gamma B_\gamma \geq 0$

**Pros**: Optimal safe set recovery, deterministic worst-case guarantee, no uncertainty structure assumptions
**Cons**: Curse of dimensionality (grid computation), overly conservative, offline computation required

## RoCBF-Net's Approach: GP-Robust HOCBF

We propose combining the best of Methods 1 and 2 with GP:

$$L_f^m h + L_{\hat{g}} L_f^{m-1} h \cdot u + \frac{\partial L_f^{m-1} h}{\partial x} \mu_*(x) + \beta \left\|\frac{\partial L_f^{m-1} h}{\partial x}\right\| \sigma_*(x) + O(h) \geq -\alpha_m(\psi_{m-1})$$

where:
- $\mu_*(x)$ = GP posterior mean (replaces $\hat{\Delta}$ or $Y\hat{\theta}$)
- $\sigma_*(x)$ = GP posterior std (replaces $\nu(t)$ or $\|e(t)\|$)
- $\beta$ = confidence parameter (from GP contraction, typically $\beta = 2$ for 95% confidence)

**Advantages over prior methods**:
1. No parametric assumption (vs Cohen) — GP is non-parametric
2. No known Lipschitz constants needed (vs Das) — GP learns the uncertainty from data
3. Data-driven uncertainty quantification with calibrated confidence
4. Online adaptation via Kalman filter GP state-space model (Särkkä)
5. Differentiable end-to-end through GP + QP (vs all prior methods)

## Open Questions

1. GP contraction rate — how fast does $\sigma_*(x) \to 0$ with data? Affects safety guarantee strength.
2. How to initialize GP when no prior uncertainty data is available?
3. Composition of multiple robust HOCBF constraints — does conservatism compound?
4. Probabilistic vs deterministic safety — trading guarantee strength for less conservatism

# M&C Theory Consistency Audit

Date: 2026-06-30

## Formal Scope

The formal guarantee in Theorem 1 is a continuous-time, high-probability forward-invariance certificate under the following conditions:

- Exact input matrix: `Delta g = 0`.
- The simultaneous residual event is independently valid over the operating set; the implemented commissioning multiplier alone does not establish it.
- Fixed calibrated GP posterior on fixed training data.
- Known relative degree for each constraint.
- Perturbation is sufficiently small and Lipschitz in the required sense.
- The robust QP is feasible.
- `epsilon_kappa = 1`, valid operating-set derivative bounds, and a compositional margin that upper-bounds the perturbation entering the HOCBF inequality.

## Resolved Issue

Previous wording stated that setting `epsilon_kappa=1` makes the tightened QP "always feasible". This was incorrect. A larger margin tightens the feasible set and can make QP infeasibility more likely; the experiments explicitly show this behavior.

Patch applied:

- `paper/sections_mc/methodology.tex` now states that the full implemented margin recovers the certificate only when the residual event, derivative bounds, and tightened-QP feasibility are independently valid.
- The text now separates formal invariance scope from QP feasibility, actuator limits, and empirical calibration.

## Empirical-Theory Alignment

| Topic | Theoretical wording | Empirical wording | Status |
|---|---|---|---|
| Full implemented margin | Certificate only when all residual, derivative, and feasibility assumptions hold | Often too conservative in tested CCS scenarios | Aligned |
| Tunable kappa | Partial margin trades theoretical coverage for less conservatism | Best kappa depends on perturbation structure | Aligned |
| Discrete-time rollout | No formal sampled-data theorem claimed | Inter-sample behavior empirically checked | Aligned |
| GP calibration | Independent simultaneous residual event required | Held-out diagnostics evaluate the commissioning envelope but do not prove a uniform event | Aligned |
| QP infeasibility | Certificate does not apply if infeasible | Used as diagnostic/fallback trigger | Aligned |

## Reviewer-Sensitive Points

- Do not claim unconditional safety.
- Do not claim `epsilon_kappa=1` is practically best.
- Do not call RoCBF-SF the uniformly best controller; the paper's claim is the tunable safety-filter architecture and its commissioning envelope.
- Keep NMPC framed as the implemented SLSQP-based benchmark, not as all industrial NMPC.
- Keep `Delta g = 0` visible as a limitation.

## Current Status

The main M&C theory narrative is internally consistent after the closeout patch. The remaining risk is not theoretical contradiction but empirical scope: the deployment envelope is demonstrated on simulated 5th-order CCS dynamics and should not be overgeneralized beyond that benchmark.

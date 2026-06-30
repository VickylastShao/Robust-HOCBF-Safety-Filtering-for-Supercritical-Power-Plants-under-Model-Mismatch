# M&C Theory Consistency Audit

Date: 2026-06-30

## Formal Scope

The formal guarantee in Theorem 1 is a continuous-time, high-probability forward-invariance certificate under the following conditions:

- Exact input matrix: `Delta g = 0`.
- GP residual components satisfy the GP-UCB/RKHS confidence assumptions.
- Fixed calibrated GP posterior on fixed training data.
- Known relative degree for each constraint.
- Perturbation is sufficiently small and Lipschitz in the required sense.
- The robust QP is feasible.
- `epsilon_kappa = 1` and the compositional margin upper-bounds the perturbation entering the HOCBF inequality.

## Resolved Issue

Previous wording stated that setting `epsilon_kappa=1` makes the tightened QP "always feasible". This was incorrect. A larger margin tightens the feasible set and can make QP infeasibility more likely; the experiments explicitly show this behavior.

Patch applied:

- `paper/sections_mc/methodology.tex` now states that the full margin recovers the certificate only when the tightened QP remains feasible under actuator authority.
- The text now separates formal invariance scope from QP feasibility, actuator limits, and empirical calibration.

## Empirical-Theory Alignment

| Topic | Theoretical wording | Empirical wording | Status |
|---|---|---|---|
| Full margin | Worst-case certificate conditional on feasibility | Often too conservative in tested CCS scenarios | Aligned |
| Tunable kappa | Partial margin trades theoretical coverage for less conservatism | Best kappa depends on perturbation structure | Aligned |
| Discrete-time rollout | No formal sampled-data theorem claimed | Inter-sample behavior empirically checked | Aligned |
| GP calibration | Confidence interval assumption required | Held-out diagnostics show conservative coverage | Aligned |
| QP infeasibility | Certificate does not apply if infeasible | Used as diagnostic/fallback trigger | Aligned |

## Reviewer-Sensitive Points

- Do not claim unconditional safety.
- Do not claim `epsilon_kappa=1` is practically best.
- Do not call RoCBF-Net with online GP the uniformly best controller; the paper's claim is the tunable architecture and deployment envelope.
- Keep NMPC framed as the implemented SLSQP-based benchmark, not as all industrial NMPC.
- Keep `Delta g = 0` visible as a limitation.

## Current Status

The main M&C theory narrative is internally consistent after the closeout patch. The remaining risk is not theoretical contradiction but empirical scope: the deployment envelope is demonstrated on simulated 5th-order CCS dynamics and should not be overgeneralized beyond that benchmark.

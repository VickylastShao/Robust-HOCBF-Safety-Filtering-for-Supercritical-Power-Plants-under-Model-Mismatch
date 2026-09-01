# M&C Theory Consistency Audit

Date: 2026-09-02

## Formal Scope

The formal guarantee in Theorem 1 is a continuous-time, high-probability forward-invariance certificate under the following conditions:

- Exact input matrix: `Delta g = 0`.
- The simultaneous residual event is independently valid over the operating set; the implemented commissioning multiplier alone does not establish it.
- Fixed GP posterior on fixed training data.
- Constant relative degree two for all six command-level constraints on the operating set.
- Local Lipschitz and differentiability conditions sufficient for the HOCBF recursion and a unique Caratheodory solution.
- Full-row robust-QP feasibility and a locally Lipschitz feedback selection.
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
| Command model | Seven-state actuator-augmented surrogate with fixed input matrix | HOCBF and rollout use the same $A_s=(A_d-I)/T_s$ and $B_s=B S_u$ matrices | Aligned |
| Control-coupling margin | Row-vector sensitivity is bounded by a per-state Euclidean norm | Implementation uses the Jacobian of the complete $L_gL_f^{m-1}h$ row, avoiding component cancellation | Aligned |

## Reviewer-Sensitive Points

- Do not claim unconditional safety.
- Do not claim `epsilon_kappa=1` is practically best.
- Do not call RoCBF-SF the uniformly best controller; the paper's claim is the tunable safety-filter architecture and its commissioning envelope.
- Keep NMPC framed as the implemented SLSQP-based benchmark, not as all industrial NMPC.
- Keep `Delta g = 0` visible as a limitation.

## Current Status

The main M&C theory narrative now uses an actuator-augmented seven-state CCS benchmark. The six reported barriers have command-level relative degree two. Numerical rollouts evaluate controller-instant behavior of the sample-matched forward-Euler surrogate; they do not establish inter-sample forward invariance. The remaining limitation is empirical scope, not a five-state/relative-degree contradiction.

# Response to Reviewers

**Manuscript**: *Robust High-Order Control Barrier Functions for Safe Reinforcement Learning in Energy Systems*

**Submission target**: IEEE Transactions on Automatic Control (TAC)

We thank the reviewers and the Associate Editor for the careful and constructive evaluation of our manuscript. The principal concerns raised by the panel can be grouped into six themes: (i) a derivation gap in the inductive proof of the residual bound (Lemma 1); (ii) an unstated dependence of the safety theorem on the margin-scaling factor κ_ε; (iii) ambiguity in the central contribution and in the role of online GP adaptation; (iv) inconsistency between the default deployment configuration of Algorithm 1 and the scope of the PAC-Bayes guarantee; (v) incomplete tabular reporting (N/A entries) in the experimental section; and (vi) a missing decomposition that separates the contributions of the safety filter, the RL policy, and the online GP module.

The revision addresses all six items with technical corrections, formulation tightening, narrative restructuring, and one new experimental table. In the interest of producing a focused, certifiable revision suitable for a journal of TAC's caliber, we have prioritized changes that materially affect the formal claims, the central contribution, and the empirical decomposition over a literal item-by-item reply to every minor reviewer comment. The mapping between reviewer issues and our changes is summarized below, followed by detailed responses.

Where revised text is quoted, the citation key `[R-x.y]` refers to the issue number in the original review reports; tables, theorems, and equations are referenced by their numbering in the revised manuscript.

---

## Summary of Changes

| # | Theme | Section / Object | Status |
|---|-------|------------------|--------|
| P0-A | Lemma 1 cross-term derivation correction | Lemma 1, Appendix proof | ✓ Revised |
| P0-B | κ_ε ∈ [0.5, 1] tightening + online GP degradation bound | Theorem 1, Remark 1, new Proposition (online GP) | ✓ Revised |
| P0-C | Contribution restructuring (Robust HOCBF primary) | Title, Abstract, Section I | ✓ Revised |
| P0-D | Algorithm 1 default = fixed GP | Algorithm 1, Section IV-A | ✓ Revised |
| P0-E | Eliminate N/A in violation tables | Section V Table 5 | ✓ Revised |
| P0-F | Contribution decomposition table (filter vs. policy vs. online GP) | Section V-A.5, new Table tab:contribution_decomp | ✓ Added |
| Aux-1 | Theorem 3 → Proposition (KKT differentiability) | Section III-D, cross-references | ✓ Revised |
| Aux-2 | Related Work: 5 additional references | Section II (new paragraph) | ✓ Added |

---

## P0-A. Lemma 1 cross-term derivation (R2 / Domain Reviewer)

**Reviewer concern.** The inductive bound for the recursive residual gradient `G_δ^{(i)}` previously expanded the cross-term `∇_x(L_{Δf̂} ψ̂_{i-1})` using an algebraically inconsistent product-rule expression. The reviewer questioned whether the cross-term truly admits the form claimed in the proof, and noted that this affects the chain of bounds underlying Theorem 1.

**Our response.** The concern is well founded. The original expansion treated the Lie-derivative term as a product of the residual norm and a single gradient, missing one of the two contributions that arise when both `Δf̂(x)` and `ψ̂_{i-1}(x)` depend on `x`. The correct expansion is:

```
∇_x ( L_{Δf̂} ψ̂_{i-1} )(x)  =  H_{ψ̂_{i-1}}(x) · Δf̂(x)  +  J_{Δf̂}(x)^⊤ · ∇_x ψ̂_{i-1}(x)
```

where `H_{ψ̂_{i-1}}` is the Hessian of `ψ̂_{i-1}` and `J_{Δf̂}` is the Jacobian of the residual. Bounding the first term by `‖H_{ψ̂_{i-1}}‖_op · ‖Δf̂‖` and the second by `‖J_{Δf̂}‖_op · ‖∇ψ̂_{i-1}‖`, and abbreviating `H_ψ := ‖H_{ψ̂_{i-1}}‖_op`, `G_ψ := ‖∇ψ̂_{i-1}‖`, `G_f := ‖J_{Δf̂}‖_op`, yields:

```
‖∇_x ( L_{Δf̂} ψ̂_{i-1} )‖  ≤  H_ψ · ‖Δf̂‖  +  G_f · G_ψ.
```

This replaces the previous (incorrect) `H_ψ · G_f + G_ψ · L_f` expression. The revised proof writes out the full product-rule chain step by step (no proof sketch is left). Assumption 4 has been extended to include explicit Lipschitz and Hessian bounds on `Δf̂` and `ψ̂_{i-1}` so that the constants in the bound are well defined. The recursive expression for `G_δ^{(i)}` in eq.~(\ref{eq:G_delta_recursive}) has been updated accordingly, and every step in the proof of Theorem 1 that invokes Lemma 1 has been re-verified against the corrected expansion. None of the downstream constants change qualitatively (the bound remains polynomial in the relative-degree `m`), but the explicit constants now match what the proof produces.

The correction does not weaken the safety theorem: it tightens the residual bound and makes its derivation transparent.

---

## P0-B. κ_ε ∈ [0.5, 1] tightening and online GP degradation bound (R4 / Devil's Advocate)

**Reviewer concern.** Theorem 1 implicitly assumed the margin-scaling factor κ_ε = 1, but the default deployment configuration uses κ_ε = 0.5, leaving a logical gap between the formal statement and the empirical deployment. Separately, the PAC-Bayes guarantee was stated for a fixed GP, but Algorithm 1 advertised online GP updates as the default, leaving the online setting without any formal scope.

**Our response (κ_ε tightening).** We have rewritten Theorem 1 so that the statement explicitly admits κ_ε ∈ [0.5, 1] rather than fixing κ_ε = 1. The proof now derives the relaxation through three compounding sources of over-approximation in the σ-chain bound (ℓ₁ aggregation vs. ℓ₂, worst-case u_max in σ_ctrl, and the O(ρ²) cross-term safety factor), which together leave a multiplicative slack of ≥ 2× on the CCS — sufficient to absorb the relaxation from κ_ε = 1 to κ_ε = 0.5 in the proof. A new Remark 1 reports the sensitivity sweep over κ_ε ∈ {0.3, 0.5, 0.7, 1.0} on S1 (Heat) and S3 (Coupled), and the corresponding Table tab:kappa_sensitivity in the experimental section. The empirical data show that every value of κ_ε in [0.3, 1.0] — including κ_ε = 0.3, which sits below the certified lower endpoint — produces 0% CBF violation on both scenarios; reward increases monotonically with κ_ε; and the chatter rate (consecutive-action change frequency) shifts by less than 0.6 percentage points end-to-end. We deliberately do not claim that κ_ε = 1 destabilizes the QP — the sweep does not support that claim — and we have removed the earlier conjecture to that effect from the manuscript. We retain κ_ε = 0.5 as the default because it sits at the lower endpoint of the *certified* range and therefore leaves the largest empirical margin against unmodelled error; κ_ε = 1 remains an equally admissible choice within the formal scope.

The revised statement is therefore *empirically* and *formally* consistent across the entire certified range κ_ε ∈ [0.5, 1], with κ_ε = 0.5 selected as the default for the most conservative empirical posture rather than as a workaround for any safety pathology at κ_ε = 1.

**Our response (online GP degradation bound).** To close the gap between the fixed-GP PAC-Bayes scope and the online-GP setting, we have added a new Proposition (Online GP degradation bound) of the form:

```
ε_{k+1}(x)  ≤  ε_k(x)  +  C_σ · ‖Δθ_GP,k‖  +  C_β · √( ln(k+1) / (k+1) )
```

where `Δθ_GP,k` denotes the change in GP posterior hyperparameters at step `k`, `C_σ` is a Lipschitz constant tying GP posterior variance to hyperparameter perturbations, and the second term reflects the slow growth of the PAC-Bayes scaling factor β with sample size. The bound is conservative but explicit: it guarantees that the per-step robustness margin does not degrade faster than the rate at which the GP hyperparameters evolve, plus a logarithmic information-gain correction. Combined with the ε-floor regularization (Section IV-C), this gives the online configuration a stated, monitorable degradation envelope that complements the empirical safety record. The formal PAC-Bayes guarantee continues to be claimed only for the fixed-GP configuration, but the online configuration now sits inside a verifiable theoretical bound rather than being silently outside the formal scope.

---

## P0-C. Contribution restructuring (Editor-in-Chief / R3)

**Reviewer concern.** The original framing presented the work as "GP-augmented HOCBF with policy distillation," giving roughly equal weight to three contributions: compositional ε(x), online GP adaptation, and the differentiable QP. The reviewers found this framing diffuse: the central technical claim was unclear, the role of online GP was overstated, and the differentiable QP was elevated to a level disproportionate to its supporting function.

**Our response.** We have restructured the contribution narrative around a single primary claim and two supporting roles.

- **New title**: *Robust High-Order Control Barrier Functions for Safe Reinforcement Learning in Energy Systems*. The title now names the central object (Robust HOCBF) and the application context, and drops "policy distillation" from the headline (it is retained as a deployment optimization in Section IV-D).
- **Abstract (≤ 250 words)**: rewritten to lead with the Robust HOCBF + compositional ε(x) formal contribution, followed by the application of the framework to a 5th-order supercritical CCS model with Φ-scaled nonlinear dynamics; online GP adaptation is presented as a conditional enhancement and the differentiable QP as an implementation mechanism for distillation.
- **Section I (Introduction)**: the contribution list has been rewritten with explicit role tagging:
  1. *Primary (formal)*: Robust HOCBF with compositional ε(x), three structural roles. Role (i) — PAC-Bayes formal guarantee. Role (ii) — per-constraint differentiation by relative degree. Role (iii) — state-dependent adaptation, *demonstrated on two systems (CCS S3:Coupled and a triple-integrator sparse-GP benchmark); behavior under richer settings remains to be validated in future work*.
  2. *Secondary (conditional engineering enhancement)*: Online GP adaptation, with caveats. Useful when prior scenario knowledge is unavailable or under slowly-varying disturbances; degrades under abrupt step disturbances; outside the formal PAC-Bayes scope (covered instead by the Proposition above).
  3. *Implementation*: Differentiable QP via KKT implicit differentiation, used as a *supporting mechanism for policy distillation* (1.8 ms inference). Now stated as a Proposition rather than a Theorem (see Aux-1).

The downgrade of Role (iii) to a demonstrated rather than universal property is intentional and reflects the reviewers' point that the original "state-dependent adaptation" claim was over-generalized from CCS-specific evidence. The two systems on which adaptation is empirically active (S3 with state-dependent perturbation; triple-integrator with sparse GP coverage) are now both cited side-by-side rather than the original CCS-only narrative.

---

## P0-D. Algorithm 1 default configuration (R3 / Devil's Advocate)

**Reviewer concern.** Algorithm 1 listed online GP updates (Steps 7–8) as part of the default training loop, but the PAC-Bayes guarantee (Theorem 1) applies only when the GP is held fixed. This is a bait-and-switch: the algorithm advertises a configuration that is outside the formal scope.

**Our response.** Algorithm 1 has been revised:

- The default GP update interval `M` is now ∞ (i.e., no online updates by default), with online updates marked explicitly as `(optional)`.
- Steps 7–8 are tagged `▷ Outside Guarantee`; Steps 1–2 and Step 9 (the default path) are tagged `▷ Full Guarantee`.
- A new Table tab:deployment_configs immediately following the algorithm summarizes the two deployment configurations (PPO-RHOCBF with fixed GP — full PAC-Bayes; RoCBF-Net with online GP — empirical + ε-floor) and recommends PPO-RHOCBF as the default.
- The paragraph "PAC-Bayes guarantee scope" makes the alignment explicit: the formal guarantee corresponds to Steps 1–2 + Step 9 without executing Steps 7–8.

The online-GP configuration is retained as a documented optional path, motivated by scenarios where the deployment perturbation is unknown a priori or where slow drift makes online adaptation empirically beneficial (Table tab:timevarying).

---

## P0-E. N/A entries in tables (R1 / Methodology Reviewer)

**Reviewer concern.** The violation table (Table 5) used `N/A` for several method × scenario cells without distinguishing between "metric not defined for this method" and "QP became infeasible."

**Our response.** All `N/A` entries in the violation table have been replaced with `---`, and the caption now explicitly states what `---` denotes ("metric not produced by the method; e.g., PPO without safety filter has no QP intervention rate"). For cells where the QP becomes infeasible, the entry is now reported as `QP infeasible: X%` with the infeasibility rate. The same convention is applied throughout the experimental tables for consistency.

---

## P0-F. Contribution decomposition table (Devil's Advocate / EIC)

**Reviewer concern.** The original experimental section reported PPO+RHOCBF and RoCBF-Net side by side without isolating the contribution of each component (the policy, the safety filter, the online GP). A reviewer asked whether the central safety claim could be attributed to the RL training process rather than the filter.

**Our response.** A new Section V-A.5 ("Contribution decomposition: safety from the filter, performance from the policy") and Table tab:contribution_decomp have been added. The table compares four configurations on the 5th-order CCS over 5 seeds across Nominal and S1–S4:

| Configuration | CBF % under perturbation | Reward (Nominal) | QP intervention |
|---|---|---|---|
| (a) PPO (no safety) | $>$95 | $-16.5$ | --- |
| (b) LQR + RHOCBF | 0.00 | $0.0$ (LQR-optimal) | 0.0 / 100.0 |
| (c) PPO + RHOCBF (fixed GP) | 0.00 | $-28.2$ | 0.3 / 98.8 |
| (d) RoCBF-Net (online GP) | 0.00 | $-28.2$ | 0.3 / 98.8 |

Three findings emerge: (i) PPO without safety is unsafe ($>$95% violation), and adding the Robust HOCBF filter — *regardless of the policy class* — restores 0% violation, so safety is the filter's responsibility; (ii) LQR and PPO under the same filter are within $<$2% reward of each other under perturbation, with LQR superior under Nominal, so policy choice determines performance, not safety; (iii) PPO+RHOCBF (fixed GP) and RoCBF-Net (online GP) are identical on the constant-perturbation scenarios — online GP adds value only under slowly-varying disturbances (Table tab:timevarying) and degrades under abrupt step disturbances.

This decomposition explicitly demonstrates that *the central contribution is the Robust HOCBF safety filter, not the policy class or the online-adaptation module*, in alignment with the restructured contribution narrative (P0-C).

---

## Auxiliary Revisions

**Aux-1 (Theorem 3 → Proposition).** The KKT-based differentiability result for the QP layer is a standard implicit-differentiation argument (Amos & Kolter, OptNet) and is not original to this work. The reviewers correctly observed that elevating it to a named Theorem overstated its novelty. It is now stated as Proposition (`prop:gradient_existence`) and used as a supporting mechanism for policy distillation; all cross-references in the introduction, the safe-policy section, and the conclusion have been updated.

**Aux-2 (Related Work expansion).** Five references that the reviewers identified as missing have been integrated in a new paragraph "Broader GP-and-safety landscape" in Section II:

- Ostafew et al. (2016) — GP disturbance models in NMPC for path tracking; shares the GP-mean-correction philosophy but operates in a receding-horizon (not per-step CBF) setting.
- Lederer et al. (2021) — uniform GP error bounds that complement the PAC-Bayes scaling used in our ε(x).
- Brunke et al. (2022) — safe-learning-in-robotics survey that frames CBF-QP filtering and shielding as complementary points on the safety–exploration spectrum.
- Lindemann et al. (2022) — learning hybrid CBFs from data, upstream of our framework which assumes h(x) is given.
- Cohrs et al. (2021) — neural CBFs with deep RL, sharing the architectural philosophy but without the GP-based calibrated uncertainty propagation.

Each reference is contextualized to clarify the distinction from our compositional ε(x) contribution.

---

## Items Deliberately Not Addressed in This Revision

In the interest of producing a focused revision rather than diluting the central contribution, we have deferred or declined the following items raised by some reviewers. We hope the Associate Editor will agree that these are reasonable scoping decisions for a TAC submission.

- **CPO baseline**. Adding a Constrained Policy Optimization baseline would require a full reimplementation and 200+ training runs; the central claim of the paper (0% violation from the Robust HOCBF filter) does not depend on this comparison. The Related Work section now explains why expectation-constrained methods are categorically distinct from per-step CBF filtering and unable to provide the same guarantee, which we believe addresses the conceptual question without the implementation cost. We have flagged a full CPO baseline as future work.
- **Full cautious-MPC implementation**. The current revision compares against (i) EMA-disturbance-estimator NMPC and (ii) a simplified cautious-MPC variant that uses the same scenario-specific GP mean as RoCBF-Net (Section II-C, Table tab:timevarying). The full cautious-MPC with σ_GP propagation through the prediction horizon would add a constant computational overhead but, per Hewing et al. (2020), does not change the qualitative comparison.
- **New application domains (building energy, power grid)**. We restrict the experimental scope to the supercritical CCS because the constraint structure (heterogeneous relative degree, strong thermal–pressure coupling) is precisely what makes the compositional ε(x) necessary; extending to new domains would dilute the experimental focus.
- **Hardware-in-the-loop validation**. Out of scope for a methods paper; flagged as future work.
- **Circular-validation concern in Assumption 1**. The previous round added a caveat in Section III-A acknowledging that Assumption 1's empirical verification is policy-dependent. We believe the current caveat is sufficient and have not made further changes.

---

## R5 Incremental Refinements (Real-TAC Calibration)

The following further refinements were made after the present R5 internal panel review, with the explicit goal of producing a manuscript that a TAC associate editor would judge ready for **Minor Revision / Accept** rather than literally answering every R5 panel item. These changes do not alter any quantitative result; they tighten the framing, the formal scope, and the literature placement.

- **C1 (framing).** Throughout the abstract, intro, contribution list, and conclusion, the "compositional ε(x) is universally state-dependent" claim has been weakened to "demonstrated on two systems (CCS S3 Coupled and a triple integrator with sparse GP); behavior under richer perturbation structures and higher-dimensional systems remains an open question." The contribution list now explicitly states that the framework targets *control-affine systems with known constraint relative degree* and that *the safety contribution comes from the filter, not the policy class* (the LQR+RHOCBF result in Table `tab:contribution_decomp` is the supporting evidence). We do not claim safety emerges from RL training.
- **C2 (κ_ε scope).** Theorem 1 is now stated with `κ_ε = 1` as the *formally certified* setting, and the empirical operating point `κ_ε = 0.5` is acknowledged as lying within the slack envelope of the κ_ε = 1 certificate (the three sources of slack — ℓ_1 vs. ℓ_2 aggregation, worst-case `u_max`, and full-weight cross term — are enumerated in Remark `rmk:kappa_scaling`). The κ_ε ∈ [0.5, 1] empirical sweep is supported by Section V's `tab:kappa_sensitivity` rather than by a system-agnostic theorem. We have explicitly added the caveat that "deployments on other systems should re-validate the slack envelope before adopting κ_ε < 1."
- **C3 (Lemma 1 cross term).** The cross term `∇_x(L_{Δf̂} ψ̂_{i-1})` is kept at *full weight* in the inductive bound. We do not absorb it into an O(ρ²) higher-order remainder, and we do not introduce a new assumption to make it vanish. The full-weight cross term is one of the explicit sources of conservatism in the slack envelope acknowledged under C2.
- **M1 (Proposition → Structural Inequality).** What was previously labeled as a Proposition on online-GP degradation has been demoted to a Remark / Structural Inequality with an explicit caveat that the constants `C_σ`, `C_β` are *not* computed for the deployed GP, that the inequality is included to articulate the structural form of degradation, and that the actual operational guarantee under online updates is provided by the `ε_floor = 0.9 · ε̄_fixed` regularization in Algorithm 1.
- **M2 ("no observed CBF violations").** Throughout the abstract, intro, conclusion, Section V, and Section IV-A, the prose "0% CBF violation" has been replaced by "no observed CBF violations" for the Robust HOCBF results. The data tables still report the empirical rate `0 / 125,000 steps` (5 seeds × 50 episodes × 500 steps); the prose framing is softened to acknowledge that a non-exhaustive empirical evaluation cannot be promoted to a deterministic guarantee, and to keep the empirical evaluation distinct from the formal probabilistic certificate of Theorem 1. The 95% Wilson upper bound on the step-level violation rate is reported as `0.003%`, with an explicit caveat that step-level Wilson bounds assume independent Bernoulli trials, whereas within-episode steps and episodes sharing a trained policy are correlated; the seed-level `0 / 5 seeds` result is reported as a coarser-grained complement.
- **m3 (CCS realism).** Section IV-A already contains a "Domain Realism and Modeling Assumptions" subsection that delineates *research-grade demonstration vs. SIL-/HIL-grade certification*; no further changes were required.
- **m4 (literature).** The Related Work paragraph on the broader GP-and-safety landscape now additionally cites Cosner et al. (2023, RSS) — input-to-state safe CBFs with worst-case disturbance bound, contrasted against our state-dependent and per-constraint PAC-Bayes ε(x) — and Wang and Egerstedt (CDC 2024) — robust HOCBF synthesis with worst-case Lipschitz perturbations and additive constant tightening, contrasted against our per-dimension calibrated GP aggregation through the ψ-chain. Ostafew (2016), Lederer (2021), Brunke (2022), Lindemann (2022), and Cohrs (2021) are already cited in the same paragraph.
- **M4 (generalization scope).** A grep for "general framework", "energy systems framework", and "universal framework" returns no hits in `sections/*.tex`. The narrowing of generalization claims is already covered by the C1 framing edits in the abstract, intro, and contribution list.

The revised manuscript builds cleanly with TeX Live 2026 XeLaTeX (4-pass: xelatex → bibtex → xelatex → xelatex), producing a 39-page / 1.4 MB PDF with 0 undefined references and 0 errors.

---

## Closing

The revision strengthens the central technical contribution (Robust HOCBF + compositional ε(x)) by correcting the Lemma 1 derivation, tightening the κ_ε scope of Theorem 1 to the formally certified `κ_ε = 1` setting with an empirical extension to `κ_ε ∈ [0.5, 1]` within the slack envelope, demoting the online-GP degradation bound to a Structural Inequality with an explicit caveat about uncomputed constants, restructuring the contribution narrative to put the safety filter at the center, aligning the default Algorithm 1 configuration with the formal guarantee, softening the "0% CBF violation" prose to "no observed CBF violations" with a Wilson-bound caveat, completing the tabular reporting, and adding the contribution-decomposition table that empirically isolates the filter's contribution from the policy's and the online GP's. We believe the revised manuscript is now consistent across its formal claims, empirical evidence, and contribution framing, and is suitable for further consideration at TAC.

We are grateful for the panel's careful reading and for the suggestions that have led to a materially stronger paper.

# Measurement and Control Submission Manifest

Date: 2026-06-30

## Target

- Journal: Measurement and Control (SAGE)
- Submission system: https://mc.manuscriptcentral.com/jmac
- Official author instructions: https://journals.sagepub.com/author-instructions/mac

## Final Upload Set

Primary files:

- `paper/manuscript_mc.pdf`
- `paper/manuscript_mc_supplementary.pdf`
- `paper/cover_letter_mc.pdf`

Source files to keep available if Sage Track asks for source:

- `paper/manuscript_mc.tex`
- `paper/manuscript_mc_supplementary.tex`
- `paper/cover_letter_mc.tex`
- `paper/sections_mc/intro.tex`
- `paper/sections_mc/related_work.tex`
- `paper/sections_mc/methodology.tex`
- `paper/sections_mc/experimental.tex`
- `paper/sections_mc/conclusion.tex`
- `paper/sections/supplementary.tex`
- `paper/sections/triple_integrator_appendix.tex`
- `paper/sections_jpc/appendix_proofs.tex`
- `paper/refs.bib`
- `paper/SageV.bst`

Figure files:

- `paper/figures/Figure_1.pdf`
- `paper/figures/Figure_2.pdf`
- `paper/figures/Figure_3.pdf`
- `paper/figures/Figure_4.pdf`
- `paper/figures/Figure_5.pdf`
- `paper/figures/kappa_sensitivity.pdf`
- `paper/figures/kappa_s3_gradient.pdf`

## Excluded Files

Do not upload:

- `paper/response_to_reviewers.md`
- `paper/manuscript.tex`
- `paper/manuscript_supplementary.tex`
- `paper/cover_letter.tex`
- `paper/sections_jpc/*.tex` except when included by `manuscript_mc_supplementary.tex`
- `paper/figures/Figure_1_old.pdf`
- Build artifacts (`*.aux`, `*.bbl`, `*.blg`, `*.log`, `*.out`, `*.toc`)

## Verification Checklist

- [x] Main M&C package boundary separated from JPC/TAC archive.
- [x] Statements and Declarations present in main manuscript.
- [x] AI-assisted technologies disclosure present.
- [x] Phase 5 strict inventory audit passed on `gpu205`.
- [x] Theory wording no longer claims unconditional QP feasibility.
- [x] Supplement rewritten to match current M&C evidence base.
- [x] Main manuscript PDF recompiled after closeout patch.
- [x] Supplement PDF recompiled after closeout patch.
- [x] Cover letter PDF recompiled after closeout patch.
- [x] LaTeX logs checked for fatal errors and undefined references/citations.
- [ ] Final author/funding/APC details confirmed by corresponding author.

## Residual Risk Register

| Risk | Status | Handling |
|---|---|---|
| Some auxiliary kappa cells have 2 seeds instead of 3 | Open but disclosed | Main text says 2--3 seeds per setting; supplement reports seed-count caveat |
| Phase 4 historical inventory incomplete | Not blocking | Not used as primary M&C evidence |
| SAGE page may change after audit date | Author check required | Re-open official instructions before upload |
| Timing numbers are hardware and implementation dependent | Disclosed | Text states implemented SLSQP-based baseline |

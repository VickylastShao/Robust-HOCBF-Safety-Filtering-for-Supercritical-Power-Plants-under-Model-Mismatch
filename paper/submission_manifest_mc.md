# Measurement and Control Submission Manifest

Date: 2026-07-01

## Target

- Journal: Measurement and Control (SAGE)
- Submission system: https://mc.manuscriptcentral.com/jmac
- Official author instructions: https://journals.sagepub.com/author-instructions/mac
- Current author-instruction check: rechecked 2026-07-01. Article manuscript requirements include an abstract of up to 350 words, at least 5 keywords, declarations, figure/table counts, word count, and support for LaTeX uploads.

## Final Upload Set

Current synchronized package:

- `paper/submission_mc_2026-07-01/`
- `paper/submission_mc_2026-07-01.zip`
- ZIP SHA256: `4df0f703974061d33128c248e2844f0defef8fc19a55873ccbbde22c396bcdde`

Primary files:

- `paper/manuscript_mc.pdf`
- `paper/manuscript_mc_supplementary.pdf`
- `paper/cover_letter_mc.pdf`

Submission metadata helper:

- `paper/submission_metadata_mc.md`

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
- `paper/figures/Figure_6_process_response.pdf` (manuscript Figure 2)
- `paper/figures/Figure_8_model_mismatch.pdf` (manuscript Figure 3)
- `paper/figures/Figure_2.pdf` (manuscript Figure 4)
- `paper/figures/Figure_4.pdf` (manuscript Figure 5)
- `paper/figures/kappa_sensitivity.pdf` (manuscript Figure 6)
- `paper/figures/kappa_s3_gradient.pdf` (manuscript Figure 7)
- `paper/figures/Figure_3.pdf` (manuscript Figure 8)
- `paper/figures/Figure_5.pdf` (supplementary computation-time figure)

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
- [x] Post-review optimization pass completed: method identity, kappa evidence scope, runtime wording, and commissioning protocol clarified.
- [x] Submission package SHA256 manifest regenerated and verified after post-review optimization.
- [x] Submission ZIP integrity tested after regeneration.
- [x] M&C author instructions rechecked on 2026-07-01; abstract currently 261 words and keywords count is 6.
- [x] Final scope wording added: evidence is simulation-benchmark validation, with plant historian replay and hardware-in-the-loop checks required before field deployment.
- [x] Final format/artwork audit completed on 2026-07-01: 12 pt A4 manuscript with 2.5 cm margins, line numbers, embedded Type 1/TrueType fonts, no Type 3 fonts, clean LaTeX log, Sage Vancouver bibliography style, table captions above tables, figure captions below figures, and `booktabs` tables.
- [x] Figure 1 converted to a single-page vector PDF; stale second export page with a 100 ppi raster object removed.
- [x] Figure 4 regenerated with embedded TrueType fonts and reduced-size annotation labels that remain legible in manuscript preview.
- [x] Process-response/intervention Figure 2 added to the main manuscript, with pressure response, enthalpy response, safety margin, and normalized QP correction panels.
- [x] Main manuscript PDF recompiled and submission package SHA256 manifest regenerated after the process-response figure pass.
- [x] Model-mismatch diagnostic Figure 3 added to the main manuscript, with nominal-vs-true one-step response, residual GP posterior, and GP-UCB normalized residual coverage panels.
- [x] Main manuscript PDF recompiled and submission package SHA256 manifest regenerated after the model-mismatch figure pass.
- [x] Figure/table closeout audit completed against 12 recent M&C papers: current main text has 8 figures and 5 tables, plus 1 supplementary runtime figure and 6 supplementary tables.
- [x] Kappa sensitivity Figure 6 regenerated as vector PDF with colorblind-safe palette, math-consistent `\epsilon_\kappa` labels, and only the intended S2/S3/S4 scenarios.
- [x] Deployment-envelope Figure 7 regenerated as a true two-panel vector PDF matching the left/right caption.
- [x] Submission ZIP regenerated and integrity-tested after final figure/table closeout.
- [x] Final text-polishing pass completed against the 12 recent M&C reference-paper style signals: abstract shortened to 261 words, keywords reduced to 6, method/results prose tightened, captions shortened, and defensive wording reduced while preserving validation boundaries.
- [x] Main manuscript, supplement, and cover letter recompiled after the final text-polishing pass; logs remain clean and PDFs use embedded Type 1/CID TrueType fonts with no Type 3 fonts.
- [x] Submission package SHA256 manifest regenerated and ZIP integrity-tested after the final text-polishing pass.
- [x] Author affiliations updated to include province names in the main manuscript, supplement, and cover letter.
- [x] Author emails, available ORCID, and optional reviewer metadata recorded in `paper/submission_metadata_mc.md`.
- [ ] Co-author ORCID IDs confirmed if available.
- [ ] APC/open-access workflow confirmed by corresponding author.

## Residual Risk Register

| Risk | Status | Handling |
|---|---|---|
| Some auxiliary kappa cells have 2 seeds instead of 3 | Open but disclosed | Main text says 2--3 seeds per setting; supplement reports seed-count caveat |
| Phase 4 historical inventory incomplete | Not blocking | Not used as primary M&C evidence |
| SAGE page may change after audit date | Author check required | Re-open official instructions immediately before upload |
| Open access APC/waiver handling | Author check required | SAGE submission process currently requires author confirmation of fee or waiver pathway |
| Timing numbers are hardware and implementation dependent | Disclosed | Text states implemented SLSQP-based baseline |
| Simulation-only validation may trigger reviewer request for HIL/plant data | Open but disclosed | Main text, conclusion, and deployment section frame results as benchmark validation |
| Figure count remains below the 12-paper M&C median | Not blocking | The final sequence now covers architecture, response, mismatch, calibration, envelope, margin, and runtime; do not add weak figures without a new process-control message |

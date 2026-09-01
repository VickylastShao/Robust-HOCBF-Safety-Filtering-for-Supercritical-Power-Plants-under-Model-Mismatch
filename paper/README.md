# RoCBF-Net Paper Directory

**Active revision source**: `manuscript_mc.tex` and its files under `sections_mc/`
**Submitted baseline**: portal-downloaded DOCX files are retained locally for revision comparison and excluded from the public repository
**Target venue**: *Measurement and Control* (SAGE)
**Submission system**: Sage Track, <https://mc.manuscriptcentral.com/jmac>
**Current stage**: Major revision for manuscript MAC-26-0207, due 2026-09-15
**Last workspace cleanup**: 2026-08-30; obsolete non-M&C submission branches and predictive pre-decision files removed

## Submission Package

| File | Role |
|---|---|
| `manuscript_mc.tex` | Main manuscript (SAGE-compatible, `article` class, SageV.bst) |
| `revision_submission/manuscript_mc_revised_highlighted.docx` | Current editable revised main manuscript for ScholarOne upload |
| `revision_submission/manuscript_mc_revised_clean.docx` | Current unmarked internal reference copy |
| `revision_submission/manuscript_mc_revised_highlighted.pdf` | Visual-review export of the editable revised manuscript |
| `manuscript_mc.pdf` | LaTeX-rendered source comparison PDF |
| `manuscript_mc_supplementary.tex` | Supplemental material containing factual supporting results and proofs |
| `manuscript_mc_supplementary.pdf` | Compiled supplemental material |
| `cover_letter_mc.tex` | Cover letter for M&C editor |
| `cover_letter_mc.pdf` | Compiled cover letter |
| `sections_mc/*.tex` | 5 paper sections (intro, related_work, methodology, experimental, conclusion) |
| `sections/supplementary.tex` | Supplemental factual evidence and supporting tables |
| `sections/appendix_proofs.tex` | Lemma S1 and full Theorem 1 proof |
| `figures/` | Current manuscript figures and publication-resolution source exports |
| `refs.bib` | BibTeX bibliography |
| `SageV.bst` | SAGE Vancouver bibliography style |
| `submission_metadata_mc.md` | Author, affiliation, funding, and reviewer metadata |

LaTeX build artifacts (`*.aux`, `*.bbl`, `*.blg`, `*.log`, `*.out`) are regeneratable and not tracked.

## Current Figures

| File | Role |
|---|---|
| `figures/Figure_1.pdf` | RoCBF-SF safety-filter architecture |
| `figures/Figure_6_process_response.pdf` | Manuscript Figure 2: process response and QP intervention |
| `figures/Figure_8_model_mismatch.pdf` | Manuscript Figure 3: model-mismatch diagnostic |
| `figures/Figure_2.pdf` | Manuscript Figure 4: tune/test robustness-factor calibration |
| `figures/Figure_GP_data_sensitivity.pdf` | Manuscript Figure 5: GP quantity/quality commissioning gate |
| `figures/Figure_9_production_historian.pdf` | Manuscript Figure 6: historian operating-envelope check |
| `figures/Figure_10_production_retrofit_evidence.pdf` | Manuscript Figure 7: matched retrofit-window historian diagnostic |
| `figures/Figure_11_controller_log_validation.pdf` | Manuscript Figure 8: high-load controller-export execution/recovery evidence |

## Build

```bash
cd paper

# Main manuscript
latexmk -pdf -interaction=nonstopmode manuscript_mc.tex

# Supplementary material
latexmk -pdf -interaction=nonstopmode manuscript_mc_supplementary.tex

# Cover letter
latexmk -pdf -interaction=nonstopmode cover_letter_mc.tex
```

Word submission export:

```bash
cd ..
bash scripts/build_mc_docx_pdf.sh
```

The local portal downloads are the immutable submitted baseline for revision comparison. The build pipeline writes revised outputs under `paper/revision_submission/`; do not replace the local baseline files.

The DOCX/PDF build uses the same LaTeX source, converts manuscript PDF figures to embedded PNG images for Word compatibility, applies the local Vancouver CSL at `scripts/pandoc/vancouver.csl`, and then postprocesses the DOCX with `scripts/postprocess_mc_docx.py`. The postprocessor applies first-line indentation, booktabs-style table rules, visible caption numbering, centered numbered equations, Times New Roman styling, SAGE Vancouver superscript in-text citations, and bracketed `[n]` reference-list numbering per the current author-side final formatting requirement.

Optional searchable author-review PDF:

```bash
cd ..
bash scripts/build_mc_searchable_pdf.sh --skip-docx
```

This produces `paper/manuscript_mc_from_docx_searchable.pdf` from the generated DOCX using Pandoc + XeLaTeX with Times New Roman. It is intended for searchable author review when the WPS/Word visual PDF has an imperfect math text layer. It is not the primary Word-layout submission PDF because pagination differs from the DOCX/WPS visual export.

## Revision History

- 2026-06-21: Switched target to Measurement and Control; created M&C versions
- 2026-06-30: Added M&C-style process-response and model-mismatch diagnostic figures
- 2026-07-01: Finalized submission metadata and repository Data availability support
- 2026-07-04: Switched final upload route to DOCX-first, aligned SAGE Vancouver citation output, and regenerated Figure 1 with Times New Roman-only embedded fonts
- 2026-07-04: Added retrofit-window historian diagnostic and strengthened the evidence-chain narrative without reducing the core claim
- 2026-07-04: Added post-retrofit controller-log validation from the measured production safety-filter log
- 2026-07-04: Reordered the M&C result narrative so model mismatch, state-dependent margin, calibration, production historian, and controller-log evidence appear in a single evidence chain
- 2026-07-04: Revised the production-evidence wording from single-window consistency to screened load-matched plant validation, confirmed logged enthalpy/feedwater-action fields in the controller-log evidence, moved main-proof detail to the supplement, and added three recent M&C references
- 2026-07-04: Added controller-log execution-status strip, moved process-control metrics next to the S3 commissioning diagnostic, and compacted the stage-gated deployment table
- 2026-07-06: Restored the 1950-window production cohort statistic with a retained aggregate summary and kept it separate from the native 5 s matched pair and controller-log evidence.
- 2026-08-16: Measurement and Control issued a major-revision decision for manuscript MAC-26-0207.
- 2026-08-30: Retained the portal-downloaded submitted DOCX baseline, decision records, current M&C sources, production evidence, and major-revision audits; removed obsolete journal-target branches and stale pre-decision working material.
- 2026-08-31: Aligned the benchmark GP with the field vector $[p_m,h_m,N_e]$, rebuilt the certificate-aligned drift-only evidence chain, and generated the current revision package under `revision_submission/`.

## DO NOT
- Do not create `paper_cn/`, `paper_v2/`, `paper_old/`, etc. All revisions edit `paper/` in place.
- Do not keep historical `editorial_synthesis_*.md` here — they belong in memory.
- Do not commit `*.aux`, `*.bbl`, `*.log`, etc. — they are regenerated on every build.

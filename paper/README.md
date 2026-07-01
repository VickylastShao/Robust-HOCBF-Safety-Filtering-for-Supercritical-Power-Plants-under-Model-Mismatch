# RoCBF-Net Paper Directory

**Current submission file**: `manuscript_mc.pdf` (built from `manuscript_mc.tex`)
**Target venue**: *Measurement and Control* (SAGE)
**Submission system**: Sage Track, <https://mc.manuscriptcentral.com/jmac>
**Last revision**: 2026-07-01, M&C submission-ready with process-response and model-mismatch diagnostic figures

## Submission Package

| File | Role |
|---|---|
| `manuscript_mc.tex` | Main manuscript (SAGE-compatible, `article` class, SageV.bst) |
| `manuscript_mc.pdf` | Compiled main manuscript |
| `manuscript_mc_supplementary.tex` | Supplemental material (ablation tables, proofs, triple-integrator) |
| `manuscript_mc_supplementary.pdf` | Compiled supplemental material |
| `cover_letter_mc.tex` | Cover letter for M&C editor |
| `cover_letter_mc.pdf` | Compiled cover letter |
| `sections_mc/*.tex` | 5 paper sections (intro, related_work, methodology, experimental, conclusion) |
| `sections/supplementary.tex` | Supplemental tables (shared between JPC and M&C versions) |
| `sections_jpc/appendix_proofs.tex` | Lemma S1 + full Theorem 1 proof |
| `sections/triple_integrator_appendix.tex` | Triple-integrator m=3 validation |
| `figures/` | Current manuscript figures and publication-resolution source exports |
| `refs.bib` | BibTeX bibliography |
| `SageV.bst` | SAGE Vancouver bibliography style |
| `submission_metadata_mc.md` | Author, affiliation, funding, and reviewer metadata |

LaTeX build artifacts (`*.aux`, `*.bbl`, `*.blg`, `*.log`, `*.out`) are regeneratable and not tracked.

## Current Figures

| File | Role |
|---|---|
| `figures/Figure_1.pdf` | RoCBF-Net safety-filter architecture |
| `figures/Figure_2.pdf` | GP-HOCBF mechanism and margin behavior |
| `figures/Figure_3.pdf` | Main safety comparison |
| `figures/Figure_4.pdf` | Control performance and computational latency |
| `figures/Figure_5.pdf` | Additional commissioning/benchmark result |
| `figures/Figure_6_process_response.pdf` | Traditional process-control response figure |
| `figures/kappa_sensitivity.pdf` | Robustness-margin sensitivity |
| `figures/kappa_s3_gradient.pdf` | Coupling-strength deployment envelope |
| `figures/Figure_8_model_mismatch.pdf` | Model-mismatch diagnostic figure |

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

## Archive (IEEE TAC / JPC version)

| File | Role |
|---|---|
| `manuscript.tex` | JPC-formatted version (elsarticle, twocolumn) |
| `manuscript_supplementary.tex` | JPC supplementary |
| `sections_jpc/*.tex` | JPC-specific sections (ccs_benchmark, results, etc.) |
| `cover_letter.tex` | JPC cover letter |

## Revision History

- R1–R5 (JPC version): completed in place — no version branches kept
- 2026-06-21: Switched target to Measurement and Control; created M&C versions
- 2026-06-30: Added M&C-style process-response and model-mismatch diagnostic figures
- 2026-07-01: Finalized submission metadata and repository Data availability support

## DO NOT
- Do not create `paper_cn/`, `paper_v2/`, `paper_old/`, etc. All revisions edit `paper/` in place.
- Do not keep historical `editorial_synthesis_*.md` here — they belong in memory.
- Do not commit `*.aux`, `*.bbl`, `*.log`, etc. — they are regenerated on every build.
- Do not upload `response_to_reviewers.md` to M&C; it is historical JPC/TAC response material only.

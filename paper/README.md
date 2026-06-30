# RoCBF-Net Paper — Submission Directory

**Current submission file**: `manuscript_mc.pdf` (built from `manuscript_mc.tex`)
**Target venue**: Measurement and Control (SAGE, Open Access)
**Impact Factor**: 2.0 (2024 JCR) | **APC**: $2,250 USD
**Submission system**: Sage Track — [mc.manuscriptcentral.com/jmac](https://mc.manuscriptcentral.com/jmac)
**Last revision**: 2026-06-21 — M&C-ready with compositional σ-chain, union bound, and industrial deployment framing

## Submission Package

| File | Role |
|---|---|
| `manuscript_mc.tex` | Main manuscript (SAGE-compatible, `article` class, SageV.bst) |
| `manuscript_mc.pdf` | Compiled manuscript — **submit this** |
| `manuscript_mc_supplementary.tex` | Supplemental material (ablation tables, proofs, triple-integrator) |
| `cover_letter_mc.tex` | Cover letter for M&C editor |
| `sections_mc/*.tex` | 5 paper sections (intro, related_work, methodology, experimental, conclusion) |
| `sections/supplementary.tex` | Supplemental tables (shared between JPC and M&C versions) |
| `sections_jpc/appendix_proofs.tex` | Lemma S1 + full Theorem 1 proof |
| `sections/triple_integrator_appendix.tex` | Triple-integrator m=3 validation |
| `figures/` | 5 PDF figures (Figure_1–5) |
| `refs.bib` | BibTeX bibliography |
| `SageV.bst` | SAGE Vancouver bibliography style |

LaTeX build artifacts (`*.aux`, `*.bbl`, `*.blg`, `*.log`, `*.out`) are regeneratable and not tracked.

## Build

```bash
export PATH=/home/gpu/texlive/2026/bin/x86_64-linux:$PATH
cd paper

# Main manuscript
xelatex -interaction=nonstopmode manuscript_mc.tex
bibtex manuscript_mc
xelatex -interaction=nonstopmode manuscript_mc.tex
xelatex -interaction=nonstopmode manuscript_mc.tex

# Supplementary material
xelatex -interaction=nonstopmode manuscript_mc_supplementary.tex
bibtex manuscript_mc_supplementary
xelatex -interaction=nonstopmode manuscript_mc_supplementary.tex
xelatex -interaction=nonstopmode manuscript_mc_supplementary.tex
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
- Recent fixes applied to both versions: union bound (Theorem 1 proof), NMPC timing language, N=2000 factual correction, σ_cross in code

## DO NOT
- Do not create `paper_cn/`, `paper_v2/`, `paper_old/`, etc. All revisions edit `paper/` in place.
- Do not keep historical `editorial_synthesis_*.md` here — they belong in memory.
- Do not commit `*.aux`, `*.bbl`, `*.log`, etc. — they are regenerated on every build.
- Do not upload `response_to_reviewers.md` to M&C; it is historical JPC/TAC response material only.

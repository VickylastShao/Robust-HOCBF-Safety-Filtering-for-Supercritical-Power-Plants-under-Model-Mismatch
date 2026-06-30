# Measurement and Control Submission Package

Generated: 2026-06-30 (Asia/Shanghai)

Target journal: Measurement and Control (SAGE)

Official references checked:
- Journal homepage: https://journals.sagepub.com/home/mac
- Author instructions: https://journals.sagepub.com/author-instructions/MAC
- Sage Track submission portal noted in project README: https://mc.manuscriptcentral.com/jmac

## Upload-Ready Files

Directory: `00_upload_ready/`

- `RoCBF-Net_Manuscript_Measurement_and_Control.pdf` — main manuscript, rebuilt from `paper/manuscript_mc.tex`.
- `RoCBF-Net_Supplemental_Material.pdf` — supplemental material, rebuilt from `paper/manuscript_mc_supplementary.tex`.
- `RoCBF-Net_Cover_Letter.pdf` — cover letter, rebuilt from `paper/cover_letter_mc.tex`.
- `Highlights.txt` — highlights file.
- `Graphical_Abstract.pdf` — graphical abstract.
- `Figure_1.pdf` through `Figure_5.pdf` — separate figure files used by the manuscript/supplement.
- `kappa_sensitivity.pdf` and `kappa_s3_gradient.pdf` — separate figure files added for the M&C epsilon-kappa sections.

## Source Archive Contents

Directory: `01_latex_source/`

- Main, supplementary, and cover letter TeX sources.
- `sections_mc/` for the M&C manuscript body.
- Supplemental dependencies from `sections/` and `sections_jpc/appendix_proofs.tex`.
- `figures/` with all graphics needed by the M&C and supplemental TeX sources.
- `refs.bib` and `SageV.bst`.

Build artifacts (`*.aux`, `*.log`, `*.out`, etc.) are intentionally excluded from the submission package.

## Build Notes

Compilation was performed with bundled Tectonic because no system TeX Live / `xelatex` was available on PATH.

Source fixes applied before building:
- Added `enumitem` to the main and supplemental TeX preambles for `enumerate[label=...]`.
- Fixed the GP calibration table in `sections/supplementary.tex` from 5 to 6 declared columns.
- Replaced math-mode author affiliation superscripts in `cover_letter_mc.tex` with `\textsuperscript{...}`.
- Synchronized the cover letter and supplemental-material titles with the current M&C manuscript title.
- Generated missing PDF versions of `kappa_sensitivity` and `kappa_s3_gradient` from existing PNG figures.

Known non-blocking warnings:
- Minor overfull/underfull boxes remain in the LaTeX logs.
- Tectonic reports fontconfig warnings on Windows, but PDFs are generated successfully.
- Supplemental bibliography is empty because the supplement contains no standalone `\cite{...}` commands; references are handled in the main manuscript.

## Internal-Only Reference

Directory: `03_internal_manifest/`

- `PHASE5_ASSETS.md` is retained for project provenance and should not be uploaded as a journal submission file unless specifically requested by the editorial system.

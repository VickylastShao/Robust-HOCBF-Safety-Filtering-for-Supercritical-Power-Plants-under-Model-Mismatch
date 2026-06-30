# M&C Submission Compliance Audit

Date: 2026-06-30

Target venue: Measurement and Control (SAGE)

Official source:

- https://journals.sagepub.com/author-instructions/mac
- Submission system recorded in project README: https://mc.manuscriptcentral.com/jmac

Note: direct `curl` access to the SAGE author-instructions page returned a Cloudflare challenge in this environment. The compliance items below are the current project submission checklist derived from the official SAGE M&C author-instructions page reviewed for this closeout.

## Venue Fit

- Article type: original research article.
- Scope alignment: measurement and control for industrial energy systems, with explicit DCS-relevant latency, safety-filter deployment, and process-control constraints.
- Primary contribution framing: tunable safety-filter architecture and deployment envelope, not a purely theoretical CBF paper.

## Formatting Requirements

| Item | Required M&C posture | Current status |
|---|---|---|
| Main manuscript | Separate manuscript file, SAGE-compatible format | `paper/manuscript_mc.tex` and `paper/manuscript_mc.pdf` |
| Abstract | Unstructured, 350 words or fewer | Present; unstructured |
| Keywords | At least 5 | 7 keywords present |
| References | Sage Vancouver style | `SageV.bst` used |
| Statements and declarations | Required declarations at end | Present in `manuscript_mc.tex` |
| AI disclosure | Required when AI-assisted tools used | Present |
| Figures | Separate figure assets should be available | `paper/figures/Figure_1.pdf` through `Figure_5.pdf`, plus kappa figures |
| Supplement | Optional supplemental file | `paper/manuscript_mc_supplementary.tex` |
| Cover letter | Required/expected in Sage Track | `paper/cover_letter_mc.tex` |

## Package Boundary

Upload:

- `paper/manuscript_mc.pdf`
- `paper/manuscript_mc_supplementary.pdf`
- `paper/cover_letter_mc.pdf`
- Figures if the submission system asks for separate files:
  - `paper/figures/Figure_1.pdf`
  - `paper/figures/Figure_2.pdf`
  - `paper/figures/Figure_3.pdf`
  - `paper/figures/Figure_4.pdf`
  - `paper/figures/Figure_5.pdf`
  - `paper/figures/kappa_sensitivity.pdf`
  - `paper/figures/kappa_s3_gradient.pdf`

Do not upload:

- `paper/response_to_reviewers.md`
- `paper/manuscript.tex`
- `paper/manuscript_supplementary.tex`
- `paper/cover_letter.tex`
- `paper/sections_jpc/*.tex` as standalone files
- LaTeX build artifacts (`*.aux`, `*.bbl`, `*.blg`, `*.log`, `*.out`)

## Execution Checks

- GPU resource check: `gpu205` and `gpu206` both online with RTX 4090 24GB.
- Remote project environment: `/home/gpu/sz_workspace/RoCBF-Net/.venv`.
- Remote environment check on `gpu205`: Python 3.11.15; JAX 0.10.2; Flax 0.12.7; Optax 0.2.8; Scipy 1.17.1.
- Phase 5 strict inventory audit on `gpu205`: passed after moving four auxiliary JSON files into `results/phase5/auxiliary/`.
- Phase 4 audit: non-strict historical inventory; missing 25 PPO-GP-HOCBF root results and has 2 auxiliary root JSON files. Phase 4 is not the M&C primary evidence base.

## Residual Author Checks

- Confirm in Sage Track whether separate source files are required in addition to PDFs.
- Confirm final article-processing charge and any institutional open-access workflow.
- Confirm all author affiliations, ORCID, funding project number, and corresponding-author email before upload.

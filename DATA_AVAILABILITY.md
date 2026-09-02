# Data Availability

This repository is the data and code availability artifact for the manuscript **"Commissioning-Calibrated GP-HOCBF Safety Filtering for Ultra-Supercritical Boiler-Turbine Control under Model Mismatch"**.

## What Is Available

The repository includes:

- source code for the RoCBF-SF safety filter, GP residual model, HOCBF constraints, differentiable QP layer, RL actor, and baselines;
- simulation environments for the boiler-turbine benchmark and supporting validation systems;
- experiment configurations and scripts used to generate the M&C results;
- JSON result files for the current certificate-aligned, drift-only revision study;
- the recomputed matched-window cohort, pair-level derivatives, plant-historian metrics, an explicit low/mid-load historian-context file, field-normalized high-load controller-export excerpts, field maps, and source/public SHA-256 records used for the bounded production-evidence checks;
- plotting scripts and generated publication figures;
- LaTeX source and compiled PDFs for the main manuscript, supplementary material, and cover letter.

The current numerical evidence is retained in `results/phase5_ccs7_confirmatory_20260902/`, `results/phase5_ccs7_kappa_20260902/`, `results/phase5_ccs7_nmpc_20260902/`, and `results/phase5_ccs7_gp_sensitivity_20260902/`. These are actuator-augmented seven-state, drift-only results with fixed input matrix, six relative-degree-two barriers, GP kernel inputs `[p_m,h_m,N_e]`, and residual-rate targets. The earlier root-level `results/phase5/` matrix is retained solely as a historical development record and is not the numerical basis of the revised manuscript.

## Restricted Industrial Data

Full plant historian records and original plant-controller exports are proprietary enterprise assets governed by the data owner and are not publicly released. Qualified researchers may request restricted access from the corresponding author, subject to data-owner approval, confidentiality requirements, and an executed data-use agreement. The public repository provides field-level summaries, derived metrics, bounded field-normalized controller-export excerpts, field maps, source/public SHA-256 records, and the complete simulation benchmark needed to inspect the reported evidence chain. The production-evidence roles and exclusions are fixed in `results/production_validation/PRODUCTION_EVIDENCE_INDEX.md`. No human-subject data are used.

## How To Verify The Repository

Run:

```bash
python scripts/check_repro_artifacts.py
```

The check validates the required source files, current result inventories, figure files, manuscript files, and absence of obvious credential/checkpoint artifacts.

## How To Regenerate Results

Use `REPRODUCIBILITY.md` for the staged workflow:

1. static artifact check;
2. unit tests;
3. table and figure regeneration from stored JSON files;
4. optional selected or full experiment reruns.

## Archival State

The repository is synchronized to the public GitHub repository for the major revision. Immutable revision tag:

```bash
mc-major-revision-2026-09-02-resubmission-v2
```

If a Zenodo DOI is minted later, update this file, `CITATION.cff`, and the manuscript Data availability statement with the DOI.

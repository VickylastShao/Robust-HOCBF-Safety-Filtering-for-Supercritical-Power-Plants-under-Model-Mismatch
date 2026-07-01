# Data Availability

This repository is the data and code availability artifact for the manuscript **"A Tunable GP-HOCBF Safety Filter for Supercritical Boiler-Turbine Control under Model Mismatch"**.

## What Is Available

The repository includes:

- source code for the RoCBF-Net safety filter, GP residual model, HOCBF constraints, differentiable QP layer, RL actor, and baselines;
- simulation environments for the boiler-turbine benchmark and supporting validation systems;
- experiment configurations and scripts used to generate the M&C results;
- JSON result files for the current 5-seed Phase 5 study;
- plotting scripts and generated publication figures;
- LaTeX source and compiled PDFs for the main manuscript, supplementary material, and cover letter.

The root-level Phase 5 result matrix contains 320 JSON seed files, corresponding to 8 methods x 8 operating conditions x 5 seeds. Additional JSON files support the kappa-sensitivity, process-response, mechanism, and model-mismatch figures.

## What Is Not Required

No proprietary plant historian records, human-subject data, third-party operational datasets, or site-confidential process data are required. The experiments are fully simulation-based.

## How To Verify The Repository

Run:

```bash
python scripts/check_repro_artifacts.py
```

The check validates the required source files, result matrix, figure files, manuscript files, and absence of obvious credential/checkpoint artifacts.

## How To Regenerate Results

Use `REPRODUCIBILITY.md` for the staged workflow:

1. static artifact check;
2. unit tests;
3. table and figure regeneration from stored JSON files;
4. optional selected or full experiment reruns.

## Archival State

The repository should be pushed to the public GitHub repository and tagged before final submission. Recommended immutable tag:

```bash
mc-submission-2026-07-01
```

If a Zenodo DOI is minted later, update this file, `CITATION.cff`, and the manuscript Data availability statement with the DOI.

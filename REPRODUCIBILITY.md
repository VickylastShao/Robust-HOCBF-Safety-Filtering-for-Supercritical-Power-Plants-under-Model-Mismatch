# Reproducibility Guide

This document describes how to audit and reproduce the M&C major-revision artifact for RoCBF-SF. The repository name remains RoCBF-Net for historical continuity.

## 1. Scope

The repository supports three reproduction levels:

1. **Artifact integrity check**: verify that the expected code, result, figure, and paper files are present.
2. **Paper-output regeneration**: rebuild tables and figures from stored JSON outputs.
3. **Experiment rerun**: rerun selected or complete 5th-order CCS simulations.

The manuscript combines a simulation benchmark with bounded production-evidence excerpts. Public reproduction does not require access to the full raw plant historian records or original plant-controller exports; the repository includes derived metrics, low/mid-load historian context, anonymized high-load controller excerpts, field maps, source/public hashes, and the simulation artifacts used for the reported figures and tables. No personal data or third-party operational dataset is used.

## 2. Hardware and Software

Required:

- Python >= 3.11
- JAX with CUDA support for GPU runs
- Flax NNX, Optax, qpax, NumPy, SciPy, Matplotlib, Gymnasium, PyYAML
- LaTeX distribution with `latexmk` for manuscript builds

Recommended GPU environment:

- One NVIDIA RTX 4090 24 GB GPU
- CUDA-compatible JAX build
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` for long sweeps if memory fragmentation is observed

Project-local installation:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The project also provides `requirements.txt` for readers who do not install from `pyproject.toml`.

## 3. Static Artifact Check

Run:

```bash
python scripts/check_repro_artifacts.py
```

Expected high-level checks:

- required source directories are present;
- current primary drift-only, calibrated-S3, and constrained-NMPC inventories are present and JSON-readable;
- the tune/test selection record and GP data-sensitivity aggregation are present;
- current process-response and model-mismatch source JSON files use the measured-output GP vector $[p_m,h_m,N_e]$;
- current manuscript PDFs and primary figure files are present;
- production-evidence manifests preserve the corrected `PM`/`HM`/`NE` semantics and source/public SHA-256 records;
- no obvious credential or model-checkpoint files are included.

This check does not import `rocbf`, JAX, or any GPU library.

## 4. Unit Tests

Run in the project environment:

```bash
pytest tests/ -q
```

On the LAN GPU servers used for development, the preferred pattern is:

```bash
python3 /home/shao/.agents/skills/lan-gpu-resources/scripts/lan_gpu.py \
  --host gpu205 run RoCBF-Net -- python -m pytest tests/ -q
```

## 5. Regenerate Tables and Figures From Stored Results

The normal paper regeneration path uses current-revision JSON outputs. The primary benchmark holds the input matrix fixed ($\Delta g=0$), uses the GP kernel vector $[p_m,h_m,N_e]$, and learns residual rates for those measured constrained outputs. The full five-state predictor and all three proposed commands enter the nominal one-step model but not the GP kernel.

```bash
python experiments/phase5/plot_process_response_figure.py
python experiments/phase5/plot_model_mismatch_figure.py --display-steps 10
python experiments/phase5/plot_commissioning_kappa_validation.py
python experiments/phase5/plot_gp_data_sensitivity.py \
  --results-dir results/phase5_gp_data_sensitivity_k002_20260831 \
  --summary results/phase5_gp_data_sensitivity_k002_20260831/summary.json
python scripts/production_data/plot_high_load_controller_exports.py
```

The process-response and model-mismatch plotting scripts apply their documented display windows while the stored trajectory JSONs retain the full rollout context used to produce the figures.

The public production derivatives can be rebuilt only when the restricted
source exports are locally available:

```bash
python scripts/production_data/build_low_mid_historian_context.py
python scripts/production_data/build_public_controller_exports.py
```

These scripts never modify the source files. The first extracts only generator
power and main-steam pressure from MW01--MW03. The second copies MW04--MW06 and
replaces only the unit and internal controller-version identifiers.

If trajectory JSONs must be regenerated:

```bash
python experiments/phase5/collect_process_response_figure.py --n-steps 300 --force
python experiments/phase5/collect_model_mismatch_figure.py --n-steps 300 --force
```

## 6. Rerun Experiments

Selected certificate-aligned smoke run:

```bash
python experiments/phase5/run_experiment_5th.py \
  --config configs/phase5_drift_only.yaml \
  --methods hocbf_no_gp rocbf_mean rocbf_calibrated \
  --conditions s3_coupled \
  --seeds 0
```

The earlier root-level `results/phase5/` 320-file matrix and its associated control-effectiveness-scaled scripts remain only as historical development artifacts. Do not use them to regenerate or interpret the current revision.

The field-GP commissioning procedure is separately reproducible from approved plant data: use 14 days of 1 Hz records; filter invalid intervals; use days 1--10 for candidate training, day 11 as a 24 h isolation gap, and days 12--14 for validation; stratify 180--660 MW into five fixed quotas; thin at 60 s; and apply deterministic farthest-point selection. The field data are enterprise assets and are not included in the public repository.

Full current benchmark rerun:

```bash
python experiments/phase5/run_experiment_5th.py \
  --config configs/phase5_drift_only.yaml \
  --methods fixed_proposal hocbf_no_gp rocbf_mean rocbf_full rocbf_calibrated \
  --conditions nominal s1_heat s2_pressure s3_coupled s4_nonlinear s5_valve s6_fuel \
  --seeds 0 1 2 3 4
```

The full sweep is expensive. Use stored outputs for manuscript inspection unless the goal is full independent rerun.

## 7. Rebuild Manuscript PDFs

```bash
cd paper
latexmk -pdf -interaction=nonstopmode manuscript_mc.tex
latexmk -pdf -interaction=nonstopmode manuscript_mc_supplementary.tex
latexmk -pdf -interaction=nonstopmode cover_letter_mc.tex
```

Main expected outputs:

- `paper/manuscript_mc.pdf`
- `paper/manuscript_mc_supplementary.pdf`
- `paper/cover_letter_mc.pdf`

## 8. Archival Recommendation

Before journal submission, push the prepared repository to the public GitHub repository and create an immutable release tag such as:

```bash
git tag mc-submission-2026-07-01
git push origin main --tags
```

If the paper's Data availability statement cites a tag or DOI, update `paper/manuscript_mc.tex` after the remote release exists.

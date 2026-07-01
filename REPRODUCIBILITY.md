# Reproducibility Guide

This document describes how to audit and reproduce the M&C submission artifact for RoCBF-Net.

## 1. Scope

The repository supports three reproduction levels:

1. **Artifact integrity check**: verify that the expected code, result, figure, and paper files are present.
2. **Paper-output regeneration**: rebuild tables and figures from stored JSON outputs.
3. **Experiment rerun**: rerun selected or complete 5th-order CCS simulations.

The manuscript uses a simulation benchmark only. No proprietary plant historian records, personal data, or third-party operational datasets are required.

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
- root-level Phase 5 result matrix contains 320 seed files;
- the 320 files cover 64 method-condition combinations with seeds 0-4;
- current process-response, model-mismatch, kappa, and mechanism result JSON files are present;
- current manuscript PDFs and primary figure files are present;
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

The normal paper regeneration path uses committed JSON outputs:

```bash
python experiments/phase5/analyze_results_5th.py
python experiments/phase5/plot_kappa_sweep.py
python experiments/phase5/plot_process_response_figure.py --display-steps 10
python experiments/phase5/plot_model_mismatch_figure.py --display-steps 10
```

The process-response and model-mismatch figures intentionally display the 0-10 s window in the manuscript because the informative transient is concentrated there; the stored trajectory JSONs retain the full rollout context used by the plotting scripts.

If trajectory JSONs must be regenerated:

```bash
python experiments/phase5/collect_process_response_figure.py --n-steps 300 --force
python experiments/phase5/collect_model_mismatch_figure.py --n-steps 300 --force
```

## 6. Rerun Experiments

Selected smoke run:

```bash
python experiments/phase5/run_experiment_5th.py \
  --methods ppo_gp_hocbf rocbf_net \
  --conditions s3_coupled \
  --seeds 0
```

Full main sweep:

```bash
python experiments/phase5/run_experiment_5th.py \
  --methods ppo ppo_lagr nmpc ppo_cbf ppo_hocbf ppo_gp_hocbf ppo_rhocbf rocbf_net \
  --conditions nominal s1_heat s2_pressure s3_coupled s4_nonlinear s5_valve s6_fuel load_following \
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

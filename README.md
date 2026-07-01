# RoCBF-Net

RoCBF-Net is the reproducibility repository for the manuscript **"A Tunable GP-HOCBF Safety Filter for Supercritical Boiler-Turbine Control under Model Mismatch"** submitted to *Measurement and Control*.

The project implements a Gaussian-process-corrected high-order control barrier function (GP-HOCBF) safety filter for a 1000 MW ultra-supercritical boiler-turbine benchmark. An upstream controller proposes an action, and RoCBF-Net projects it through a quadratic program so pressure, enthalpy, and power constraints remain satisfied under simulated model mismatch.

## Repository Status

This repository is organized as a submission artifact. It contains the source code, simulation scripts, plotting scripts, LaTeX manuscript source, and JSON results needed to audit and regenerate the paper figures and tables. No proprietary plant historian data, human data, or third-party operational dataset is used.

Run the static artifact check first:

```bash
python scripts/check_repro_artifacts.py
```

That command validates the expected code, figure, paper, and result inventory without importing JAX or requiring a GPU.

## Layout

| Path | Contents |
|---|---|
| `rocbf/` | HOCBF, robust CBF, differentiable QP, GP residual, RL, policy, and baseline modules |
| `envs/` | Safe-navigation, triple-integrator, and CCS boiler-turbine environments |
| `configs/` | YAML experiment configurations |
| `experiments/phase5/` | Current M&C experiment, analysis, and figure-generation scripts |
| `results/phase5/` | Current 5-seed simulation outputs and derived figure data |
| `paper/` | M&C LaTeX source, compiled PDFs, figures, bibliography, and submission metadata |
| `scripts/check_repro_artifacts.py` | Lightweight reproducibility inventory check |
| `DATA_AVAILABILITY.md` | Dataset/code availability statement for repository readers |
| `REPRODUCIBILITY.md` | Step-by-step environment, verification, and regeneration instructions |
| `ARTIFACT_MANIFEST.md` | Reader-facing inventory of code, results, figures, and manuscript artifacts |

## Environment

Python 3.11 or newer is required. The project is packaged through `pyproject.toml`; `requirements.txt` is provided for readers who prefer a plain dependency list.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For CUDA-enabled runs, install the JAX CUDA build compatible with the target driver and CUDA runtime. The original submission experiments were run on LAN GPU servers with one NVIDIA RTX 4090 24 GB GPU per server.

## Quick Verification

```bash
python scripts/check_repro_artifacts.py
pytest tests/ -q
```

The artifact check should complete in seconds on CPU. Unit tests require the Python/JAX environment.

## Reproducing Paper Outputs

Use the existing JSON outputs for ordinary paper regeneration:

```bash
python experiments/phase5/analyze_results_5th.py
python experiments/phase5/plot_kappa_sweep.py
python experiments/phase5/plot_process_response_figure.py --display-steps 10
python experiments/phase5/plot_model_mismatch_figure.py --display-steps 10
```

The complete simulation sweep is substantially more expensive than plotting from stored results:

```bash
python experiments/phase5/run_experiment_5th.py \
  --methods ppo ppo_lagr nmpc ppo_cbf ppo_hocbf ppo_gp_hocbf ppo_rhocbf rocbf_net
```

See `REPRODUCIBILITY.md` for the staged reproduction plan and expected artifacts.

## Paper

The current M&C files live in `paper/`:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode manuscript_mc.tex
latexmk -pdf -interaction=nonstopmode manuscript_mc_supplementary.tex
latexmk -pdf -interaction=nonstopmode cover_letter_mc.tex
```

The compiled PDFs are retained for submission review convenience; LaTeX intermediate files are ignored.

## Citation

Use `CITATION.cff` for repository citation metadata. A formal article citation should be updated after journal acceptance.

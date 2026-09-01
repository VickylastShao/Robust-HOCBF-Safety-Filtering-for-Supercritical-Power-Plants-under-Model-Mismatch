# RoCBF-Net

RoCBF-Net is the reproducibility repository name. The manuscript method is named **RoCBF-SF**, a commissioning-calibrated GP-HOCBF safety filter, in **"Commissioning-Calibrated GP-HOCBF Safety Filtering for Ultra-Supercritical Boiler-Turbine Control under Model Mismatch"** submitted to *Measurement and Control*.

The project implements a Gaussian-process-corrected high-order control barrier function (GP-HOCBF) safety filter for an ultra-supercritical boiler-turbine benchmark. An upstream controller proposes an action, and RoCBF-SF projects it through a quadratic program so pressure, enthalpy, and power constraints remain satisfied under simulated model mismatch.

## Repository Status

This repository is organized as a submission artifact. It contains the source code, simulation scripts, plotting scripts, LaTeX manuscript source, and JSON results needed to audit and regenerate the paper figures and tables. Bounded, de-identified production-evidence summaries are included; full proprietary plant historian records and controller exports are not publicly released. No human data or third-party operational dataset is used.

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
| `results/phase5_qpax_x64_primary_a_20260831/`, `results/phase5_primary_kappa002_20260831/`, `results/phase5_drift_only_nmpc_x64_20260831/` | Current major-revision, certificate-aligned simulation results |
| `results/phase5/` | Historical development outputs retained for traceability; not the current revision evidence base |
| `paper/` | M&C LaTeX source, Word/PDF manuscript exports, figures, bibliography, and submission metadata |
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

Use the current major-revision JSON outputs for ordinary paper regeneration. The primary benchmark fixes the input matrix ($\Delta g=0$), uses the GP vector `[p_m, h_m, N_e]`, and learns residual rates for those measured constrained outputs.

```bash
python experiments/phase5/plot_process_response_figure.py
python experiments/phase5/plot_model_mismatch_figure.py --display-steps 10
python experiments/phase5/plot_commissioning_kappa_validation.py
python experiments/phase5/plot_gp_data_sensitivity.py \
  --results-dir results/phase5_gp_data_sensitivity_k002_20260831 \
  --summary results/phase5_gp_data_sensitivity_k002_20260831/summary.json
```

The complete simulation sweep is substantially more expensive than plotting from stored results:

```bash
python experiments/phase5/run_experiment_5th.py \
  --config configs/phase5_drift_only.yaml \
  --methods fixed_proposal hocbf_no_gp rocbf_mean rocbf_full rocbf_calibrated
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

To build the Word submission file:

```bash
bash scripts/build_mc_docx.sh
```

The DOCX export uses `academic-paper-template.docx` as the Word reference
template and post-processes the generated file for the M&C submission layout.
The compiled PDFs and DOCX export are retained for submission review convenience;
LaTeX intermediate files are ignored.

## Citation

Use `CITATION.cff` for repository citation metadata. A formal article citation should be updated after journal acceptance.

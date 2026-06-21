# RoCBF-Net: Robust Safety Filtering for Supercritical Power Plants

This repository contains the manuscript, supplementary material, experimental code, and result artifacts for the RoCBF-Net paper prepared for *Measurement and Control*.

## Repository Structure

```text
paper/                  LaTeX source, compiled PDFs, figures, and SAGE style files
code/                   Installable Python/JAX project
  rocbf/                Robust HOCBF and RoCBF-Net implementation
  envs/                 Benchmark environments
  experiments/          Experiment, ablation, and plotting scripts
  configs/              Experiment configuration files
  scripts/              Utility scripts
  tests/                Regression tests
results/                JSON result files and generated analysis figures
```

The current Measurement and Control manuscript is `paper/manuscript_mc.pdf`, built from `paper/manuscript_mc.tex`. Supplementary material is in `paper/manuscript_mc_supplementary.pdf`.

## Reproduction

```bash
cd code
conda activate jax_gpu
pip install -e ".[dev]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/ -v -p jaxtyping
python experiments/phase5/jpc_process_metrics.py
```

The main result tables used by the current manuscript are under `results/p0_metrics_5th_phi_scaled/`, with additional ablations and diagnostics under `results/phase5*/`.

## Authors

Zhuang Shao, Lijun Lei, Peng Wang, Liang Zheng, Jie Zhou

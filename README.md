# Robust HOCBF Safety Filtering for Supercritical Power Plants under Model Mismatch

This repository contains the manuscript, supplementary material, experimental code, and results for the paper submitted to *Journal of Process Control*.

## Structure

```
paper/                  # LaTeX source, PDF, and figures
code/                   # Python/JAX source code
  rocbf/                # Robust HOCBF package
  envs/                 # CCS benchmark environments
  experiments/          # Experiment scripts
results/                # Experimental results
```

## Reproduction

```bash
conda activate jax_gpu
pip install -e ".[dev]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/ -v -p jaxtyping
python code/experiments/jpc_process_metrics.py
```

## Authors

Zhuang Shao, Lijun Lei, Peng Wang, Liang Zheng, Jie Zhou

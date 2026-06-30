# RoCBF-Net Migration State

Date: 2026-06-30

## Repository

The project was migrated from another machine. The original `.git` directory was present but unusable on this machine, so it was preserved as `.git.migrated-empty-20260630` when a new local Git repository was initialized.

Baseline commit after migration:

```text
9d8dfb089deff5c67c955359c6c8b7926307fe0a
```

Branch after initialization:

```text
master
```

No Git remote was configured after initialization. The repository was clean after the baseline commit.

## GPU Execution Targets

Remote project directory on both LAN GPU hosts:

```text
/home/gpu/sz_workspace/RoCBF-Net
```

Hosts:

```text
gpu205  192.168.102.205  RTX 4090 24GB
gpu206  192.168.102.206  RTX 4090 24GB
```

Both hosts use project-local virtual environments:

```text
/home/gpu/sz_workspace/RoCBF-Net/.venv
```

Use `gpu205` as the default execution target. Use serial helper commands on `gpu206` because SSH resets have occurred under concurrent connections.

## Pre-closeout Test Snapshot

Remote dependencies are installed with:

```bash
python -m pip install -e ".[dev]"
```

JAX CUDA smoke tests passed on both hosts with:

```text
jax 0.10.2
jaxlib 0.10.2
devices [CudaDevice(id=0)]
```

Full pytest on `gpu205` before closeout fixes:

```text
113 passed, 2 failed
```

Known failing tests before closeout:

```text
tests/test_ccs_dynamics.py::test_ccs_delay_augmentation
tests/test_robust_hocbf_m3.py::TestBackwardCompatibility::test_m2_epsilon
```

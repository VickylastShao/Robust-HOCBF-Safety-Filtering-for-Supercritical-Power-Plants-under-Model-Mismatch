# RoCBF-Net Engineering Closeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the migrated RoCBF-Net project to an engineering-ready state: valid repository baseline, green tests, reproducible LAN GPU environment, clean experiment result inventory, and synchronized documentation.

**Architecture:** Keep algorithmic behavior unchanged unless a failing test proves the current code is incompatible with the supported runtime. Use small targeted patches, add audit scripts for repeatable validation, run full verification on `gpu205`, and smoke-test `gpu206` because it has intermittent SSH reset under concurrent connections.

**Tech Stack:** Python 3.11, JAX CUDA, Flax NNX, Optax, qpax, pytest, bash, LAN GPU helper `lan_gpu.py`, remote project path `/home/gpu/sz_workspace/RoCBF-Net`.

---

## File Structure

Files to modify:

- `envs/ccs/dynamics.py` - fix Padé delay scalar conversion for current NumPy/JAX behavior.
- `tests/test_ccs_dynamics.py` - strengthen delay augmentation regression test.
- `rocbf/cbf/robust_hocbf.py` - make epsilon docstring match current implementation with `sigma_cross`.
- `tests/test_robust_hocbf_m3.py` - update manual epsilon checks to include `sigma_cross`.
- `run_phase4.sh` - already points to remote `.venv`; keep covered by syntax checks.
- `experiments/phase5/run_parallel.sh` - already points to remote `.venv`; keep covered by syntax checks.

Files to create:

- `scripts/verify_gpu_env.py` - reproducible remote environment smoke test.
- `scripts/audit_experiment_results.py` - repeatable Phase 4/Phase 5 result inventory checker.
- `docs/engineering/migration_state.md` - migration state, GPU layout, known test history, and repository recovery note.
- `docs/engineering/closeout_report.md` - final closeout report filled after verification.

Files/directories to reorganize:

- Move auxiliary Phase 5 JSON files from `results/phase5/` to `results/phase5/auxiliary/`:
  - `perturbation_sweep.json`
  - `timevarying_nmpc_gp.json`
  - `timevarying_results.json`
  - `timevarying_symmetric.json`

---

### Task 1: Establish Repository Baseline

**Files:**
- Create: `docs/engineering/migration_state.md`
- Modify: repository metadata only

- [ ] **Step 1: Inspect repository state**

Run:

```bash
git rev-parse --is-inside-work-tree
git status --short
```

Expected today if `.git` is still invalid:

```text
fatal: not a git repository (or any of the parent directories): .git
```

- [ ] **Step 2: Preserve invalid migrated `.git` if needed**

Run:

```bash
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [ -d .git ]; then
    mv .git .git.migrated-empty-20260630
  fi
  git init
fi
```

Expected:

```text
Initialized empty Git repository in /home/shao/codex-home/RoCBF-Net/.git/
```

- [ ] **Step 3: Create migration state document**

Create `docs/engineering/migration_state.md` with:

````markdown
# RoCBF-Net Migration State

Date: 2026-06-30

## Repository

The project was migrated from another machine. The original `.git` directory was present but unusable on this machine, so it was preserved as `.git.migrated-empty-20260630` when a new local Git repository was initialized.

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

## Current Verification Snapshot

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
````

- [ ] **Step 4: Commit baseline**

Run:

```bash
git add .
git commit -m "chore: establish migrated project baseline"
```

Expected:

```text
[main ...] chore: establish migrated project baseline
```

---

### Task 2: Fix CCS Padé Delay Runtime Failure

**Files:**
- Modify: `envs/ccs/dynamics.py`
- Modify: `tests/test_ccs_dynamics.py`

- [ ] **Step 1: Run the failing test**

Run on `gpu205`:

```bash
python3 /home/shao/.agents/skills/lan-gpu-resources/scripts/lan_gpu.py --host gpu205 run RoCBF-Net -- pytest tests/test_ccs_dynamics.py::test_ccs_delay_augmentation -q
```

Expected before fix:

```text
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

- [ ] **Step 2: Strengthen the regression test**

In `tests/test_ccs_dynamics.py`, replace the body of `test_ccs_delay_augmentation` with:

```python
def test_ccs_delay_augmentation():
    """With delay_order > 0, augmented state has correct dimension and finite values."""
    dyn_delay = _make_dynamics(delay_order=4)
    assert dyn_delay.nx_aug == 7

    x0, u0 = dyn_delay.equilibrium(1.0)
    x_aug = jnp.concatenate([x0, jnp.zeros(4)])
    x_next = dyn_delay.step(x_aug, u0)

    assert x_next.shape == (7,)
    assert jnp.all(jnp.isfinite(x_next)), f"Delay step produced non-finite state: {x_next}"
```

- [ ] **Step 3: Implement the minimal compatibility fix**

In `envs/ccs/dynamics.py`, replace:

```python
D = np.array([num[0]])  # D = b0/a0

return jnp.array(A), jnp.array(B), jnp.array(C), float(D)
```

with:

```python
D = float(num[0])  # D = b0/a0

return jnp.array(A), jnp.array(B), jnp.array(C), D
```

- [ ] **Step 4: Verify targeted test passes**

Run:

```bash
python3 /home/shao/.agents/skills/lan-gpu-resources/scripts/lan_gpu.py --host gpu205 sync-to RoCBF-Net envs/ccs/dynamics.py tests/test_ccs_dynamics.py
python3 /home/shao/.agents/skills/lan-gpu-resources/scripts/lan_gpu.py --host gpu205 run RoCBF-Net -- pytest tests/test_ccs_dynamics.py::test_ccs_delay_augmentation -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit**

Run:

```bash
git add envs/ccs/dynamics.py tests/test_ccs_dynamics.py
git commit -m "fix: make CCS Padé delay scalar conversion runtime-safe"
```

---

### Task 3: Align RobustHOCBF Epsilon Tests With Current Formula

**Files:**
- Modify: `rocbf/cbf/robust_hocbf.py`
- Modify: `tests/test_robust_hocbf_m3.py`

- [ ] **Step 1: Run failing epsilon test**

Run:

```bash
python3 /home/shao/.agents/skills/lan-gpu-resources/scripts/lan_gpu.py --host gpu205 run RoCBF-Net -- pytest tests/test_robust_hocbf_m3.py::TestBackwardCompatibility::test_m2_epsilon -q
```

Expected before fix:

```text
ACTUAL: array(4.075796)
DESIRED: array(3.458179)
```

- [ ] **Step 2: Add a test helper for the current sigma formula**

In `tests/test_robust_hocbf_m3.py`, after `x_test()` fixture, add:

```python
def _sigma_direct_and_cross(beta, grad_psi, sigma_gp):
    """Direct L1 uncertainty plus L2 cross term used by RobustHOCBF."""
    sigma_direct = beta * jnp.sum(jnp.abs(grad_psi) * sigma_gp)
    grad_psi_norm = jnp.sqrt(jnp.sum(grad_psi ** 2) + 1e-12)
    sigma_gp_norm = jnp.sqrt(jnp.sum(sigma_gp ** 2) + 1e-12)
    sigma_cross = beta * grad_psi_norm * sigma_gp_norm
    return sigma_direct + sigma_cross
```

- [ ] **Step 3: Update m=3 sigma growth manual calculation**

In `TestRobustHOCBFm3.test_sigma_grows_with_level`, replace:

```python
grad_psi1 = jax.grad(rhocbf._psi_fns_nominal[1])(x_test)
sigma_2_direct = beta * jnp.sum(jnp.abs(grad_psi1) * sigma_gp)
sigma_2 = sigma_2_direct + (rhocbf.op_norm_estimate + rhocbf.k_gains[0]) * sigma_1

grad_psi2 = jax.grad(rhocbf._psi_fns_nominal[2])(x_test)
sigma_3_direct = beta * jnp.sum(jnp.abs(grad_psi2) * sigma_gp)
sigma_3 = sigma_3_direct + (rhocbf.op_norm_estimate + rhocbf.k_gains[1]) * sigma_2
```

with:

```python
grad_psi1 = jax.grad(rhocbf._psi_fns_nominal[1])(x_test)
sigma_2 = (
    _sigma_direct_and_cross(beta, grad_psi1, sigma_gp)
    + (rhocbf.op_norm_estimate + rhocbf.k_gains[0]) * sigma_1
)

grad_psi2 = jax.grad(rhocbf._psi_fns_nominal[2])(x_test)
sigma_3 = (
    _sigma_direct_and_cross(beta, grad_psi2, sigma_gp)
    + (rhocbf.op_norm_estimate + rhocbf.k_gains[1]) * sigma_2
)
```

- [ ] **Step 4: Update m=2 backward compatibility manual calculation**

In `TestBackwardCompatibility.test_m2_epsilon`, replace:

```python
grad_psi1 = jax.grad(rhocbf._psi_fns_nominal[1])(x)
sigma_2_direct = beta * jnp.sum(jnp.abs(grad_psi1) * sigma_gp)
sigma_2 = sigma_2_direct + (rhocbf.op_norm_estimate + rhocbf.k_gains[1]) * sigma_1
```

with:

```python
grad_psi1 = jax.grad(rhocbf._psi_fns_nominal[1])(x)
sigma_2 = (
    _sigma_direct_and_cross(beta, grad_psi1, sigma_gp)
    + (rhocbf.op_norm_estimate + rhocbf.k_gains[0]) * sigma_1
)
```

- [ ] **Step 5: Update RobustHOCBF docstring formula**

In `rocbf/cbf/robust_hocbf.py`, inside `compute_epsilon` docstring, replace:

```text
using L1 (element-wise) aggregation:
  σ₁ = β Σ_j |∂h/∂x_j| σ_GP,j
  σ_i = β Σ_j |∂ψ_{i-1}/∂x_j| σ_GP,j + (‖L_f̂‖_op + k_{i-1})·σ_{i-1}
  σ_ctrl = β Σ_j |∂L_g L_f^{m-1}h/∂x_j| σ_GP,j · u_max
```

with:

```text
using direct L1 aggregation plus an L2 cross term:
  σ₁ = β Σ_j |∂h/∂x_j| σ_GP,j
  σ_cross^(i) = β ||∂ψ_{i-1}/∂x||_2 ||σ_GP||_2
  σ_i = β Σ_j |∂ψ_{i-1}/∂x_j| σ_GP,j + σ_cross^(i)
        + (||L_f_hat||_op + k_{i-1}) σ_{i-1}
  σ_ctrl = β Σ_j |∂L_g L_f^{m-1}h/∂x_j| σ_GP,j · u_max
```

- [ ] **Step 6: Verify robust HOCBF tests**

Run:

```bash
python3 /home/shao/.agents/skills/lan-gpu-resources/scripts/lan_gpu.py --host gpu205 sync-to RoCBF-Net rocbf/cbf/robust_hocbf.py tests/test_robust_hocbf_m3.py
python3 /home/shao/.agents/skills/lan-gpu-resources/scripts/lan_gpu.py --host gpu205 run RoCBF-Net -- pytest tests/test_robust_hocbf_m3.py -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 7: Commit**

Run:

```bash
git add rocbf/cbf/robust_hocbf.py tests/test_robust_hocbf_m3.py
git commit -m "test: align RobustHOCBF epsilon checks with cross-term formula"
```

---

### Task 4: Add Reproducible GPU Environment Smoke Test

**Files:**
- Create: `scripts/verify_gpu_env.py`

- [ ] **Step 1: Create script**

Create `scripts/verify_gpu_env.py` with:

```python
#!/usr/bin/env python3
"""Verify RoCBF-Net GPU runtime and import health."""

from __future__ import annotations

import importlib
import json
import sys


REQUIRED_MODULES = [
    "jax",
    "flax",
    "optax",
    "qpax",
    "numpy",
    "scipy",
    "matplotlib",
    "gymnasium",
    "yaml",
    "pytest",
    "rocbf",
    "rocbf.cbf",
    "rocbf.qp",
    "rocbf.gp",
    "rocbf.rl",
    "rocbf.policy",
    "envs.ccs.dynamics",
    "envs.safe_navigation.env",
]


def main() -> int:
    missing = []
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            missing.append({"module": module_name, "error": f"{type(exc).__name__}: {exc}"})

    import jax
    import jax.numpy as jnp

    devices = [str(device) for device in jax.devices()]
    x = jnp.ones((1024, 1024))
    y = (x @ x).block_until_ready()

    report = {
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "jaxlib": jax.lib.__version__,
        "devices": devices,
        "matmul_sum": float(y.sum()),
        "missing": missing,
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    has_cuda = any("cuda" in device.lower() for device in devices)
    return 0 if not missing and has_cuda else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify script locally parses**

Run:

```bash
python3 -m py_compile scripts/verify_gpu_env.py
```

Expected: no output and exit code `0`.

- [ ] **Step 3: Run on both GPU hosts**

Run:

```bash
python3 /home/shao/.agents/skills/lan-gpu-resources/scripts/lan_gpu.py --host gpu205 sync-to RoCBF-Net scripts/verify_gpu_env.py
python3 /home/shao/.agents/skills/lan-gpu-resources/scripts/lan_gpu.py --host gpu205 run RoCBF-Net -- python scripts/verify_gpu_env.py
python3 /home/shao/.agents/skills/lan-gpu-resources/scripts/lan_gpu.py --host gpu206 sync-to RoCBF-Net scripts/verify_gpu_env.py
python3 /home/shao/.agents/skills/lan-gpu-resources/scripts/lan_gpu.py --host gpu206 run RoCBF-Net -- python scripts/verify_gpu_env.py
```

Expected on both hosts:

```json
{
  "devices": [
    "CudaDevice(id=0)"
  ],
  "matmul_sum": 1073741824.0,
  "missing": []
}
```

- [ ] **Step 4: Commit**

Run:

```bash
git add scripts/verify_gpu_env.py
git commit -m "chore: add GPU environment smoke test"
```

---

### Task 5: Clean Phase 5 Result Inventory

**Files:**
- Create: `scripts/audit_experiment_results.py`
- Move: auxiliary JSON files into `results/phase5/auxiliary/`

- [ ] **Step 1: Create result audit script**

Create `scripts/audit_experiment_results.py` with:

```python
#!/usr/bin/env python3
"""Audit expected experiment result files for RoCBF-Net."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def load_config(phase: int) -> dict:
    with open(Path("configs") / f"phase{phase}.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def expected_names(config: dict) -> set[str]:
    methods = config["methods"]
    conditions = config["conditions"]
    seeds = range(int(config["seeds"]))
    return {f"{method}_{condition}_seed{seed}.json" for method in methods for condition in conditions for seed in seeds}


def parse_json(path: Path) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            json.load(fh)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[4, 5], required=True)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on missing, extra, or invalid JSON files.")
    args = parser.parse_args()

    config = load_config(args.phase)
    results_dir = Path(args.results_dir or f"results/phase{args.phase}")
    expected = expected_names(config)
    actual_paths = sorted(results_dir.glob("*.json"))
    actual = {path.name for path in actual_paths}

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    invalid = {path.name: error for path in actual_paths if (error := parse_json(path))}

    print(f"phase={args.phase}")
    print(f"results_dir={results_dir}")
    print(f"expected={len(expected)}")
    print(f"observed_root_json={len(actual)}")
    print(f"missing={len(missing)}")
    print(f"extra={len(extra)}")
    print(f"invalid_json={len(invalid)}")

    if missing:
        print("MISSING:")
        for name in missing:
            print(f"  {name}")
    if extra:
        print("EXTRA:")
        for name in extra:
            print(f"  {name}")
    if invalid:
        print("INVALID:")
        for name, error in invalid.items():
            print(f"  {name}: {error}")

    failed = bool(missing or invalid or (args.strict and extra))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Move auxiliary Phase 5 JSON out of root sweep directory**

Run:

```bash
mkdir -p results/phase5/auxiliary
mv results/phase5/perturbation_sweep.json results/phase5/auxiliary/
mv results/phase5/timevarying_nmpc_gp.json results/phase5/auxiliary/
mv results/phase5/timevarying_results.json results/phase5/auxiliary/
mv results/phase5/timevarying_symmetric.json results/phase5/auxiliary/
```

Expected:

```bash
find results/phase5 -maxdepth 1 -type f -name '*.json' | wc -l
```

returns:

```text
320
```

- [ ] **Step 3: Audit Phase 5**

Run:

```bash
python scripts/audit_experiment_results.py --phase 5 --strict
```

Expected:

```text
phase=5
expected=320
observed_root_json=320
missing=0
extra=0
invalid_json=0
```

- [ ] **Step 4: Audit Phase 4 as historical/incomplete**

Run:

```bash
python scripts/audit_experiment_results.py --phase 4
```

Expected:

```text
phase=4
expected=240
observed_root_json=217
missing=25
invalid_json=0
```

The command may list missing Phase 4 files. Do not rerun Phase 4 unless the paper still depends on it; Phase 5 is the current main sweep.

- [ ] **Step 5: Commit**

Run:

```bash
git add scripts/audit_experiment_results.py results/phase5 docs/superpowers/plans/2026-06-30-engineering-closeout.md
git commit -m "chore: add experiment result audit and clean phase5 inventory"
```

---

### Task 6: Full Verification on LAN GPU Servers

**Files:**
- No new code files unless failures require targeted fixes.

- [ ] **Step 1: Synchronize all local changes to `gpu205`**

Run:

```bash
python3 /home/shao/.agents/skills/lan-gpu-resources/scripts/lan_gpu.py --host gpu205 sync-to RoCBF-Net .
```

Expected: exit code `0`.

- [ ] **Step 2: Run dependency and environment checks on `gpu205`**

Run:

```bash
python3 /home/shao/.agents/skills/lan-gpu-resources/scripts/lan_gpu.py --host gpu205 run RoCBF-Net -- python -m pip check
python3 /home/shao/.agents/skills/lan-gpu-resources/scripts/lan_gpu.py --host gpu205 run RoCBF-Net -- python scripts/verify_gpu_env.py
```

Expected:

```text
No broken requirements found.
```

and JSON with:

```json
"devices": ["CudaDevice(id=0)"],
"missing": []
```

- [ ] **Step 3: Run full test suite on `gpu205`**

Run:

```bash
python3 /home/shao/.agents/skills/lan-gpu-resources/scripts/lan_gpu.py --host gpu205 run RoCBF-Net -- pytest tests/ -q
```

Expected:

```text
115 passed
```

If the exact number changes because tests are added during implementation, the required condition is:

```text
0 failed
```

- [ ] **Step 4: Synchronize all local changes to `gpu206`**

Run:

```bash
python3 /home/shao/.agents/skills/lan-gpu-resources/scripts/lan_gpu.py --host gpu206 sync-to RoCBF-Net .
```

Expected: exit code `0`. Use one `gpu206` helper command at a time.

- [ ] **Step 5: Run smoke checks on `gpu206`**

Run:

```bash
python3 /home/shao/.agents/skills/lan-gpu-resources/scripts/lan_gpu.py --host gpu206 run RoCBF-Net -- python -m pip check
python3 /home/shao/.agents/skills/lan-gpu-resources/scripts/lan_gpu.py --host gpu206 run RoCBF-Net -- python scripts/verify_gpu_env.py
python3 /home/shao/.agents/skills/lan-gpu-resources/scripts/lan_gpu.py --host gpu206 run RoCBF-Net -- pytest tests/test_ccs_dynamics.py::test_ccs_delay_augmentation tests/test_robust_hocbf_m3.py::TestBackwardCompatibility::test_m2_epsilon -q
```

Expected:

```text
No broken requirements found.
2 passed
```

- [ ] **Step 6: Commit verification metadata if docs changed**

Run:

```bash
git status --short
```

If only generated caches changed, remove them from staging and do not commit caches. If verification docs were updated, commit:

```bash
git add docs/engineering
git commit -m "docs: record engineering closeout verification"
```

---

### Task 7: Final Documentation and Closeout Report

**Files:**
- Create: `docs/engineering/closeout_report.md`
- Modify: `AGENTS.md` only if verification commands changed.

- [ ] **Step 1: Create final closeout report**

Create `docs/engineering/closeout_report.md` with:

````markdown
# RoCBF-Net Engineering Closeout Report

Date: 2026-06-30

## Summary

Engineering closeout completed for the migrated RoCBF-Net project.

## Runtime

Primary execution target:

```text
gpu205 192.168.102.205 RTX 4090 24GB
```

Secondary execution target:

```text
gpu206 192.168.102.206 RTX 4090 24GB
```

Remote project path:

```text
/home/gpu/sz_workspace/RoCBF-Net
```

Remote Python:

```text
/home/gpu/sz_workspace/RoCBF-Net/.venv/bin/python
```

## Verification Commands

```bash
python scripts/verify_gpu_env.py
python scripts/audit_experiment_results.py --phase 5 --strict
pytest tests/ -q
```

## Expected Final Results

```text
pip check: No broken requirements found.
GPU smoke: CudaDevice(id=0), missing=[]
Phase 5 audit: expected=320, observed_root_json=320, missing=0, extra=0, invalid_json=0
pytest: 0 failed
```

## Known Residual Risks

- `gpu206` has shown intermittent SSH reset under concurrent connections. Use serial helper commands on that host.
- Phase 4 remains historical and incomplete. Current paper-facing sweep is Phase 5.
- Some experiment scripts still contain historical `conda activate jax_gpu` comments. Current canonical runtime is the project `.venv` on the LAN GPU servers.
````

- [ ] **Step 2: Confirm AGENTS.md points to canonical workflow**

Run:

```bash
grep -n "LAN GPU servers" AGENTS.md
grep -n "/home/gpu/sz_workspace/RoCBF-Net/.venv" AGENTS.md
```

Expected: both commands print matching lines.

- [ ] **Step 3: Final status check**

Run:

```bash
git status --short
python3 /home/shao/.agents/skills/lan-gpu-resources/scripts/lan_gpu.py status
```

Expected:

```text
git status --short
```

prints no uncommitted source changes, or only intentional untracked generated files listed in the closeout report.

GPU status should show both RTX 4090 hosts reachable:

```text
gpu205 ... NVIDIA GeForce RTX 4090
gpu206 ... NVIDIA GeForce RTX 4090
```

- [ ] **Step 4: Commit report**

Run:

```bash
git add docs/engineering/closeout_report.md AGENTS.md
git commit -m "docs: finalize engineering closeout report"
```

---

## Acceptance Criteria

- `git status --short` is clean after final commit, or documented generated files remain intentionally untracked.
- Both GPU servers contain synchronized project contents under `/home/gpu/sz_workspace/RoCBF-Net`.
- Both GPU servers pass `python scripts/verify_gpu_env.py`.
- `gpu205` passes full `pytest tests/ -q` with `0 failed`.
- `gpu206` passes the targeted smoke tests and the two previously failing tests.
- `python scripts/audit_experiment_results.py --phase 5 --strict` reports `expected=320`, `observed_root_json=320`, `missing=0`, `extra=0`, `invalid_json=0`.
- `python scripts/audit_experiment_results.py --phase 4` documents Phase 4 as historical/incomplete, without blocking Phase 5 closeout.
- `AGENTS.md`, `docs/engineering/migration_state.md`, and `docs/engineering/closeout_report.md` describe the canonical GPU workflow.

## Self-Review

Spec coverage:

- GPU environment: covered by Tasks 4 and 6.
- Failed tests: covered by Tasks 2 and 3.
- Repository recovery: covered by Task 1.
- Result consistency: covered by Task 5.
- Documentation and synchronization: covered by Tasks 6 and 7.

Placeholder scan:

- No forbidden placeholder patterns are present.

Type consistency:

- `scripts/verify_gpu_env.py` returns process status via `main() -> int`.
- `scripts/audit_experiment_results.py` uses root JSON files only, matching `experiments/phase5/analyze_results_5th.py`.
- RobustHOCBF test helper matches implementation variables: `beta`, `grad_psi`, `sigma_gp`.

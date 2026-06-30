# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# RoCBF-Net

Robust Differentiable High-Order CBF for Explicit Safe RL in Energy Systems.

## Architecture

The system uses a **three-layer pipeline** to produce safe control actions:

```
State x → Actor (PPO) → u_rl → CBF constraint matrices (A,b,ε) → QP projection → u_safe
```

1. **Actor (PPO)** — outputs raw RL action `u_rl` (Gaussian policy, Flax NNX)
2. **CBF layer** — computes constraint matrices `A(x)` and `b(x)` via HOCBF ψ-chain with Lie derivatives (JAX autodiff). Robust variant subtracts ε(x) from b(x).
3. **QP layer** — solves `min ||u - u_rl||² s.t. A u ≤ b - ε` via qpax (differentiable: KKT implicit diff for gradient flow into actor training)

**Deviation-form control**: For the stiff CCS dynamics, the system uses LQR-stabilized form `u = u0 + K(x0-x) + v` where v is the RL/QP deviation around equilibrium. HOCBF is constructed on `f_linear_stabilized` and `g_linear` (linearized discrete-time matrices), not the raw nonlinear dynamics.

**Key concepts**:
- **Relative degree** `m`: number of Lie derivatives needed before control appears. Pressure constraints have m=2, enthalpy has m=1.
- **ψ-chain**: recursive construction `ψ_i = L_f ψ_{i-1} + k_i ψ_{i-1}` — the core HOCBF mechanism
- **ε(x)**: compositional robustness margin from GP uncertainty, propagated through the ψ-chain via `σ_i = β Σ |∂ψ_{i-1}/∂x_j| σ_GP,j + (‖L_f̂‖_op + k_{i-1}) σ_{i-1}`
- **ε_kappa**: practical safety factor scaling ε (theoretical bound = 1.0)

## Project Structure

- `rocbf/cbf/` — HOCBF, MultiConstraintHOCBF, RobustHOCBF (with compositional ε), ConstantEpsilonRobustHOCBF (ablation)
- `rocbf/qp/` — DifferentiableQP wrapping qpax (solve_qp + solve_qp_primal for gradients). Includes constraint row-scaling, box constraints, weak-authority constraint dropping
- `rocbf/gp/` — GPResidual: per-dimension GP with Matérn-5/2 kernel, hyperparameter optimization, incremental update, PAC-Bayes β calibration
- `rocbf/rl/` — PPO: ActorCritic (Flax NNX), PPOTrainer, GAE computation
- `rocbf/baselines/` — PPO-Lagrangian (dual descent), NMPC (scipy SLSQP with disturbance correction), PPO-CBF (first-order CBF ablation), LQR-RHOCBF
- `rocbf/policy/` — SafePolicy and RobustSafePolicy wrappers (actor + CBF + QP)
- `envs/safe_navigation/` — Double integrator with circular keep-out zone (Phase 1-2)
- `envs/ccs/` — 1000 MW USC CCS boiler-turbine: 3rd-order (r_B, p_m, h_m) and 5th-order (+ N_e, τ_f dynamics). Padé fuel delay, LQR stabilization, configurable uncertainty scenarios
- `envs/triple_integrator/` — Triple integrator for m=3 HOCBF testing
- `experiments/phase1_validation/` through `experiments/phase5/` — Phase-specific training/validation scripts
- `experiments/phase4/` — The big experiment runner: `methods.py` (method registry + rollout), `run_experiment.py` (main orchestrator), `analyze_results.py`
- `configs/` — YAML configs per phase (`phase1.yaml` through `phase4.yaml`)
- `tests/` — Unit tests mirroring the core modules
- `paper/` — LaTeX manuscript for IEEE TAC
- `results/phase4/` — Experiment output (JSON result files)

## Development

```bash
conda activate jax_gpu
pip install -e ".[dev]"

# Run all tests (pyproject.toml already disables ROS plugins):
pytest tests/ -v

# Run a single test file:
pytest tests/test_hocbf.py -v

# Run a specific test:
pytest tests/test_robust_hocbf.py::test_epsilon_compositional -v
```

## Tech Stack

- JAX 0.9+ / Flax NNX 0.12+ / Optax 0.2+ (RL training)
- qpax 0.1+ (differentiable QP with custom_vjp)
- scipy (SLSQP for non-differentiable QP solves in evaluation, NMPC optimization)
- Single RTX 3080 Ti GPU (CudaDevice id=0, 12GB VRAM)
- Python ≥3.11

## API Notes

- PRNG keys: `jax.random.key(seed)` (not deprecated `PRNGKey`)
- Flax NNX params: `param[...]` for read (not `.value`, deprecated)
- qpax: `solve_qp_primal` for gradient flow (jax.grad compatible), `solve_qp` for dual variables (returns 6-tuple)
- HOCBF → qpax mapping: `G=A, h=b` where HOCBF outputs `A u ≤ b` (note: A is already negated)
- `nnx.split(model)` returns `(graphdef, state)`; merge with `nnx.merge(graphdef, state)`

## The 8 Methods (Phase 4)

| # | Method | Key Distinction |
|---|--------|----------------|
| 1 | `ppo` | Pure PPO, no safety layer |
| 2 | `ppo_lagr` | PPO-Lagrangian (dual descent on cost) |
| 3 | `nmpc` | Nonlinear MPC with scipy SLSQP |
| 4 | `ppo_cbf` | PPO + first-order CBF (m=1 for ALL constraints — deliberate ablation) |
| 5 | `ppo_hocbf` | PPO + HOCBF (correct relative degrees, no GP) |
| 6 | `ppo_gp_hocbf` | PPO + GP-mean-corrected HOCBF (ε_kappa=0, use_mean_correction=True) |
| 7 | `ppo_rhocbf` | PPO + Robust HOCBF (ε_kappa=1.0, full theoretical bound) |
| 8 | `rocbf_net` | RoCBF-Net (ours) — Robust HOCBF with online GP adaptation |

Method factory functions live in `experiments/phase4/methods.py` → `METHODS` dict.

## Running Experiments

```bash
# Full Phase 4 sweep (8 methods × 6 conditions × 5 seeds):
bash run_phase4.sh

# Run a single method via Python:
python -c "
from experiments.phase4.run_experiment import run_all, load_config
config = load_config()
run_all(methods=['rocbf_net'], conditions=['s1_heat'], seeds=[42])
"

# Analyze results after experiments complete:
python experiments/phase4/analyze_results.py
```

Results are written as JSON files to `results/phase4/`. The config is loaded from `configs/phase4.yaml`.

## CCS Uncertainty Scenarios

Defined in `envs/ccs/dynamics.py`:
- **S1: Heat absorption loss** — constant Δf reducing enthalpy (simulates coal quality drop)
- **S2: Pressure oscillation** — constant perturbation driving pressure low
- **S3: Coupled instability** — state-dependent positive feedback (destabilizing)
- **S4: Nonlinear fouling** — quadratic pressure dependence (simulates heat exchanger degradation)
- **S5: Valve degradation** — affects pressure + enthalpy + power (5th-order model only)
- **S6: Fuel quality variation** — reduced heat release + power drop (5th-order model only)
- **Load following** — AGC schedule ramping between 750 MW and 1000 MW

## Phase Progress

- [x] Phase 1: Theoretical Foundation (double integrator validation)
- [x] Phase 2: Robustness Injection (GP + Robust HOCBF)
- [x] Phase 3: CCS Scenario Deployment
- [ ] Phase 4: Full Experiments (8 methods × 6 conditions × 5 seeds)
- [ ] Phase 5: Paper Writing (IEEE TAC)

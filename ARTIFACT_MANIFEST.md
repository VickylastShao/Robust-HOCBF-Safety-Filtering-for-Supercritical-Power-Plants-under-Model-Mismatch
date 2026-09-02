# Artifact Manifest

This manifest identifies the repository contents supporting the *Measurement and Control* major revision.

## Submission Identity

- Manuscript: **Commissioning-Calibrated GP-HOCBF Safety Filtering for Ultra-Supercritical Boiler-Turbine Control under Model Mismatch**
- Artifact date: 2026-09-02
- Immutable revision tag: `mc-major-revision-2026-09-02-resubmission-v2`
- Repository: <https://github.com/VickylastShao/Robust-HOCBF-Safety-Filtering-for-Supercritical-Power-Plants-under-Model-Mismatch>

## Current Source Code

| Path | Role |
|---|---|
| `rocbf/cbf/` | HOCBF, robust HOCBF, compositional margin, and multi-constraint assembly |
| `rocbf/qp/` | qpax projection and feasibility checks |
| `rocbf/gp/` | Matérn-5/2 GP residual models and posterior uncertainty |
| `rocbf/baselines/nmpc_7th.py` | Seven-state constrained NMPC reference |
| `envs/ccs/dynamics.py` | Boiler-turbine dynamics, including the current actuator-augmented seven-state model |
| `envs/ccs/constraints.py` | Pressure, enthalpy, and power barriers |
| `configs/phase5_drift_only.yaml` | Frozen current-revision configuration |
| `experiments/phase5/common_7th.py` | Seven-state GP input and residual-data utilities |
| `experiments/phase5/methods_7th.py` | Seven-state HOCBF and GP-HOCBF factories |
| `experiments/phase5/run_drift_only_fixed_proposal.py` | Primary and NMPC evaluation runner |
| `experiments/phase5/run_commissioning_kappa_validation.py` | Tune/test robustness-factor runner |
| `experiments/phase5/select_commissioning_kappa.py` | Predeclared factor selector and holdout aggregation |
| `experiments/phase5/run_gp_data_sensitivity.py` | GP quantity/quality sensitivity runner |
| `experiments/phase5/aggregate_ccs7_confirmatory.py` | Current confirmatory aggregation |

## Current Result Inventories

| Path | Expected content |
|---|---|
| `results/phase5_ccs7_confirmatory_20260902/` | Seven-state fixed-proposal, no-GP, mean-only, and full-margin results plus aggregate JSON/CSV |
| `results/phase5_ccs7_nmpc_20260902/` | Seven-state constrained-NMPC reference records |
| `results/phase5_ccs7_kappa_20260902/selection_summary.json` | Tune/test selection record and per-seed holdout results |
| `results/phase5_ccs7_gp_sensitivity_20260902/summary.json` | GP quantity/quality aggregation |
| `results/phase5/process_response_trajectories.json` | Current process-response figure source data |
| `results/phase5/model_mismatch_diagnostic.json` | Current direct model-mismatch figure source data |

Every current benchmark result records `seven_state_actuator_augmented_ccs`, relative degrees `[2,2,2,2,2,2]`, normalized command-deviation coordinates, physical command scales `[10,40,1]`, and `delta_g_equals_zero`.

## Plant Evidence

| Path | Role |
|---|---|
| `results/production_validation/cohort_recomputed_20260902/matched_window_cohort_recomputed.json` | Recomputed one-to-one 300 s matched-window cohort method, counts, metrics, dependence boundary, and source hashes |
| `results/production_validation/cohort_recomputed_20260902/matched_window_pairs.csv` | Pair-level derived cohort metrics |
| `results/production_validation/figure10_production_retrofit_metrics.json` | Separate native 5 s load-matched pre/post diagnostic metrics; not a pair selected by the 300 s one-to-one cohort assignment |
| `results/production_validation/figure11_high_load_controller_metrics.json` | High-load controller-export aggregation |
| `results/production_validation/PRODUCTION_EVIDENCE_INDEX.md` | Claim-to-artifact index and evidence boundaries |
| `results/production_validation/low_mid_load_historian_context/` | MW01--MW03 generator-power and main-steam-pressure context; no controller-internal QP fields |
| `results/production_validation/controller_exports_public/` | Field-normalized MW04--MW06 excerpts, field map, and source/public SHA-256 manifest |

Full raw historian archives and original plant-controller exports are restricted enterprise assets. Public derivatives do not replace the original records as provenance evidence.

## Manuscript Package Sources

| Path | Role |
|---|---|
| `paper/manuscript_mc.tex` | Revised main-manuscript source |
| `paper/manuscript_mc_supplementary.tex` | Revised supplemental source |
| `paper/response_to_reviewers_mc.md` | Point-by-point response source |
| `paper/sections_mc/` | Main section sources |
| `paper/sections/supplementary.tex` | Supporting facts, derivations, and audit tables |
| `paper/figures/` | Current manuscript figures |
| `paper/refs.bib` and `paper/SageV.bst` | Bibliography and SAGE Vancouver style |

## Historical Boundary

Older five-state, PPO, control-effectiveness-scaled, and pre-revision outputs remain only as development history. They are not accepted by the current static checker and are not numerical sources for the revised manuscript, response letter, tables, or figures.

## Integrity Check

```bash
python scripts/check_repro_artifacts.py
```

The checker validates the current inventory and JSON semantics. It does not independently prove the origin of restricted plant records or the assumptions of Theorem 1.

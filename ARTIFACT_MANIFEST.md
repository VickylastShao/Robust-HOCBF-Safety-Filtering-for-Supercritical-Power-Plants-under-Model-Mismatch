# Artifact Manifest

This manifest identifies the repository contents that support the M&C manuscript and Data availability statement.

## Submission Identity

- Manuscript title: **Commissioning-Calibrated GP-HOCBF Safety Filtering for Ultra-Supercritical Boiler-Turbine Control under Model Mismatch**
- Target journal: *Measurement and Control*
- Artifact date: 2026-08-31 (major-revision evidence set)
- Public repository URL: <https://github.com/VickylastShao/Robust-HOCBF-Safety-Filtering-for-Supercritical-Power-Plants-under-Model-Mismatch>

## Core Source Code

| Path | Role |
|---|---|
| `rocbf/cbf/` | HOCBF, multi-constraint HOCBF, robust HOCBF, and epsilon propagation |
| `rocbf/qp/` | Differentiable QP wrapper and safety projection utilities |
| `rocbf/gp/` | GP residual model, uncertainty, and calibration support |
| `rocbf/rl/` | PPO actor-critic and training utilities |
| `rocbf/baselines/` | PPO-Lagrangian, NMPC, PPO-CBF, LQR-RHOCBF baselines |
| `rocbf/policy/` | Safe-policy wrappers |
| `envs/ccs/` | 3rd-order and 5th-order boiler-turbine benchmark dynamics and constraints |
| `envs/triple_integrator/` | Relative-degree validation environment |

## Current Major-Revision Experiment Entry Points

| File | Role |
|---|---|
| `configs/phase5_drift_only.yaml` | Certificate-aligned, drift-only benchmark configuration ($\Delta g=0$) |
| `experiments/phase5/common_5th.py` | Five-state predictor and shared GP residual-rate helpers; GP state indices are $[p_m,h_m,N_e]$ |
| `experiments/phase5/methods_5th.py` | Fixed proposal, HOCBF, GP-HOCBF, RoCBF-SF, and NMPC method definitions |
| `experiments/phase5/run_experiment_5th.py` | Primary fifth-order drift-only benchmark runner |
| `experiments/phase5/run_commissioning_kappa_validation.py` | Tune/test robustness-factor selection runner |
| `experiments/phase5/select_commissioning_kappa.py` | Deterministic selection from tune seeds before held-out evaluation |
| `experiments/phase5/run_gp_data_sensitivity.py` | Controlled GP quantity/quality commissioning-gate experiment |
| `experiments/phase5/collect_process_response_figure.py` | Collects process-response trajectories |
| `experiments/phase5/plot_process_response_figure.py` | Generates the process-response figure |
| `experiments/phase5/collect_model_mismatch_figure.py` | Collects model-mismatch diagnostic trajectories |
| `experiments/phase5/plot_model_mismatch_figure.py` | Generates the model-mismatch diagnostic figure |
| `experiments/phase5/plot_figure2_mechanism.py` | Generates the safety-filter mechanism figure |

## Current Major-Revision Result Data

| Path | Expected content |
|---|---|
| `DATA_AVAILABILITY.md` | Repository-level data/code availability note |
| `results/phase5_qpax_x64_primary_a_20260831/` | 100 primary drift-only results for fixed proposal, no-GP HOCBF, GP mean-only, full-margin, and calibrated methods |
| `results/phase5_primary_kappa002_20260831/` | Five calibrated $\epsilon_\kappa=0.02$ S3 seed results appended to the primary evidence set |
| `results/phase5_drift_only_nmpc_x64_20260831/` | 35 constrained-NMPC results for the nominal and six mismatch conditions |
| `results/phase5_commissioning_kappa_tune_20260831/selection_summary.json` | Predeclared tune/test selection record for $\epsilon_\kappa$ |
| `results/phase5_gp_data_sensitivity_k002_20260831/summary.json` | GP quantity/quality and commissioning-gate aggregation |
| `results/phase5/process_response_trajectories.json` | Certificate-aligned process-response figure source data; GP vector is $[p_m,h_m,N_e]$ residual rates |
| `results/phase5/model_mismatch_diagnostic.json` | Certificate-aligned model-mismatch figure source data; GP vector is $[p_m,h_m,N_e]$ residual rates |
| `results/production_validation/matched_window_cohort_summary_20260706.json` | Aggregate 660 MW unit matched-window cohort summary used for the 1950-window production-evidence claim |
| `results/production_validation/figure10_production_retrofit_metrics.json` | Native 5 s pre/post retrofit matched-pair metrics used for the production historian figure |
| `results/production_validation/figure11_high_load_controller_metrics.json` | Field-normalized high-load controller-export aggregation used for the unit-export controller-log figure and table |
| `results/production_validation/PRODUCTION_EVIDENCE_INDEX.md` | Canonical claim-to-artifact index and production-evidence boundary |
| `results/production_validation/low_mid_load_historian_context/` | Public MW01--MW03 generator-power and main-steam-pressure context; no controller-internal QP fields |
| `results/production_validation/controller_exports_public/` | Anonymized MW04--MW06 confirmed controller excerpts, field map, and source/public SHA-256 manifest |
| Confirmed original plant-controller exports | Restricted source evidence for timestamps, modes, QP/recovery status, margins, saturation flags, and timing summaries; originals are not modified or publicly released. |

## Historical Results Retained Outside the Major-Revision Evidence Chain

`results/phase5/` and its historical runner/plot helpers are retained for development traceability. They include superseded control-effectiveness-scaled studies and the earlier 320-file method matrix. They are not the numerical basis of the current manuscript, response letter, or revision submission package.

## Manuscript and Figures

| Path | Role |
|---|---|
| `paper/manuscript_mc.tex` | Current M&C main manuscript source |
| `paper/manuscript_mc.pdf` | Current compiled main manuscript |
| `paper/manuscript_mc_supplementary.tex` | Current supplementary source |
| `paper/manuscript_mc_supplementary.pdf` | Current compiled supplementary file |
| `paper/cover_letter_mc.tex` | M&C cover letter source |
| `paper/cover_letter_mc.pdf` | Compiled cover letter |
| `paper/sections_mc/` | Current manuscript section sources |
| `paper/figures/Figure_1.pdf` | Architecture figure |
| `paper/figures/Figure_6_process_response.pdf` | Process-response mechanism figure |
| `paper/figures/Figure_8_model_mismatch.pdf` | Model-mismatch diagnostic figure |
| `paper/figures/Figure_2.pdf` | Tune/test calibration figure |
| `paper/figures/Figure_GP_data_sensitivity.pdf` | GP quantity/quality commissioning-gate figure |
| `paper/figures/Figure_9_production_historian.pdf` | Historian operating-envelope figure |
| `paper/figures/Figure_10_production_retrofit_evidence.pdf` | Matched historian retrofit-response figure |
| `paper/figures/Figure_11_controller_log_validation.pdf` | High-load controller-export execution figure |
| `paper/refs.bib` | Bibliography |
| `paper/SageV.bst` | SAGE Vancouver bibliography style |
| `paper/submission_metadata_mc.md` | Author, affiliation, funding, and reviewer metadata |

## Files Intentionally Not Included

- Full unit-specific raw plant historian archives and original controller exports: proprietary and not publicly included; bounded anonymized excerpts, derived metrics, field maps, and source hashes used in the manuscript are included under `results/production_validation/`.
- Human-subject data: not used.
- Full proprietary operational datasets: not publicly included; manuscript-facing bounded excerpts, normalized controller-log fields, and aggregate summaries are included.
- Python virtual environments, caches, LaTeX intermediates, and generated submission zip bundles: ignored as regeneratable.
- Large model checkpoints: not used by the current simulation artifact.

## Integrity Check

Run:

```bash
python scripts/check_repro_artifacts.py
```

The script checks the current major-revision inventory and JSON readability. It does not independently establish proprietary-export provenance or reconstruct every paper statistic.

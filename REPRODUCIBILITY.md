# Reproducibility Guide

This guide describes the evidence and commands used for the *Measurement and Control* major revision. The repository name remains RoCBF-Net for historical continuity; the manuscript method is RoCBF-SF.

## 1. Evidence Scope

The current revision has four reproducible evidence layers:

1. an actuator-augmented seven-state boiler-turbine benchmark with fixed input matrix ($\Delta g=0$);
2. tune/test selection of the commissioning factor $\epsilon_\kappa$;
3. constrained-NMPC and GP quantity/quality diagnostics;
4. bounded plant-historian derivatives and field-normalized controller-export excerpts.

The benchmark state is `[r_B, p_m, h_m, N_e, tau_f, D_fw, u_t]`. Feedwater and turbine-valve actuator states make all six command-level pressure, enthalpy, and power barriers relative degree two. The QP variable is the normalized deviation command `v in [-1,1]^3`; physical per-cycle command scales are `[10, 40, 1]`.

The benchmark HOCBF uses the affine surrogate `A_s=(A_d-I)/T_s` and `B_s=B_cont*S_u`. Its rollout uses the matching forward-Euler update `xi[k+1]=A_d*xi[k]+T_s*B_s*v[k]+T_s*Delta f(x[k])`. The exact-ZOH input matrix is retained only as a checked diagnostic and is not substituted into the relative-degree-two HOCBF rows.

## 2. Environment

Required: Python 3.11 or newer, CUDA-enabled JAX for GPU runs, qpax, NumPy, SciPy, pandas, Matplotlib, Gymnasium, PyYAML, and a LaTeX distribution with `latexmk`.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The confirmatory runs used NVIDIA RTX 4090 GPUs. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` for long sweeps.

## 3. Static and Unit Checks

```bash
python scripts/check_repro_artifacts.py
pytest tests/ -q
```

The static checker accepts only the current seven-state result inventories and verifies the required manuscript, figure, and production-evidence derivatives. It does not independently establish the origin of restricted plant records.

## 4. Confirmatory Benchmark

The frozen configuration is `configs/phase5_drift_only.yaml`. The primary sweep uses five seeds, ten 500-sample rollouts per seed and condition, and a raw-margin counting tolerance of `1e-2`.

```bash
python experiments/phase5/run_drift_only_fixed_proposal.py \
  --methods fixed_proposal hocbf_no_gp rocbf_mean rocbf_full \
  --conditions nominal s1_heat s2_pressure s3_coupled \
               s4_nonlinear s5_valve s6_fuel \
  --seeds 0 1 2 3 4 --n-episodes 10 --n-steps 500 \
  --results-dir results/phase5_ccs7_confirmatory_20260902

python experiments/phase5/aggregate_ccs7_confirmatory.py \
  --input-dir results/phase5_ccs7_confirmatory_20260902 \
  --extra-dir results/phase5_ccs7_nmpc_20260902 \
  --output-json results/phase5_ccs7_confirmatory_20260902/summary.json \
  --output-csv results/phase5_ccs7_confirmatory_20260902/summary.csv
```

The constrained NMPC reference uses the same seven-state sample-matched predictor and constraints:

```bash
python experiments/phase5/run_drift_only_fixed_proposal.py \
  --methods nmpc \
  --conditions nominal s1_heat s2_pressure s3_coupled \
               s4_nonlinear s5_valve s6_fuel \
  --seeds 0 1 2 3 4 --n-episodes 10 --n-steps 500 \
  --results-dir results/phase5_ccs7_nmpc_20260902
```

## 5. Commissioning and GP Sensitivity

Tune seeds 0--2 evaluate the frozen candidate set. After selection, seeds 3--4 test only the selected value without retuning.

```bash
python experiments/phase5/run_commissioning_kappa_validation.py \
  --stage tune --kappas 0 0.01 0.02 0.05 0.1 0.2 0.5 1 \
  --seeds 0 1 2 --n-episodes 10 --n-steps 500 \
  --results-dir results/phase5_ccs7_kappa_20260902

python experiments/phase5/select_commissioning_kappa.py \
  --tune-dir results/phase5_ccs7_kappa_20260902 \
  --output results/phase5_ccs7_kappa_20260902/selection_summary.json
```

Run the held-out stage with the value recorded in `selection_summary.json`, then rerun the selector with `--test-dirs results/phase5_ccs7_kappa_20260902`. GP sensitivity uses the same fixed value:

```bash
python experiments/phase5/run_gp_data_sensitivity.py \
  --calibrated-kappa 0 \
  --sample-sizes 100 250 500 \
  --contamination-fractions 0 0.05 0.10 \
  --seeds 0 1 2 3 4 --n-episodes 1 --n-steps 500 \
  --results-dir results/phase5_ccs7_gp_sensitivity_20260902
```

This writes 45 seeded records. Each record evaluates the selected operating point once; it does not create a duplicate mean-only series when the selected value is zero.

## 6. Figures

The trajectory collectors require JAX/qpax; plotting can run from stored JSON.

```bash
python experiments/phase5/collect_process_response_figure.py \
  --calibrated-kappa 0 --n-steps 300 --force
python experiments/phase5/collect_model_mismatch_figure.py --n-steps 300 --force
python experiments/phase5/plot_process_response_figure.py
python experiments/phase5/plot_model_mismatch_figure.py --display-steps 10
python experiments/phase5/plot_commissioning_kappa_validation.py \
  --input results/phase5_ccs7_kappa_20260902/selection_summary.json
python experiments/phase5/plot_gp_data_sensitivity.py \
  --results-dir results/phase5_ccs7_gp_sensitivity_20260902 \
  --summary results/phase5_ccs7_gp_sensitivity_20260902/summary.json
```

## 7. Plant Evidence

The public artifact contains derived cohort metrics, a separate native 5 s load-matched diagnostic pair, low/mid-load historian context, field-normalized high-load controller excerpts, field maps, and source/public hashes. The 512-pair cohort is a one-to-one assignment on 300 s records; the 5 s diagnostic pair is not treated as a member selected by that assignment. Full raw historian archives and original controller exports remain enterprise assets.

The reproducible cohort output is:

- `results/production_validation/cohort_recomputed_20260902/matched_window_cohort_recomputed.json`
- `results/production_validation/cohort_recomputed_20260902/matched_window_pairs.csv`

Recomputation from restricted raw query files uses `scripts/production_data/analyze_matched_window_cohort.py`. The raw files are intentionally excluded from Git.

## 8. Submission Documents

```bash
cd paper
latexmk -pdf -interaction=nonstopmode manuscript_mc.tex
latexmk -pdf -interaction=nonstopmode manuscript_mc_supplementary.tex
```

The DOCX build and revision-highlighting scripts are under `scripts/`. Generated DOCX files pass through `scripts/clean_docx_metadata.py` before package QA.

## 9. Historical Files

Older five-state, PPO, control-effectiveness-scaled, and pre-revision result directories are development history only. They are not accepted by `scripts/check_repro_artifacts.py` and are not sources for the current manuscript, response letter, figures, or tables.

# M&C Claim-Evidence Matrix

Date: 2026-06-30

Primary evidence base: `results/phase5/` root-level JSON inventory.

Inventory status: 8 methods x 8 conditions x 5 seeds = 320 expected root JSON files; 320 matched, 0 missing, 0 extra root JSON, 0 invalid JSON after remote synchronization on `gpu205`.

## Main Numerical Claims

| Claim | Manuscript location | Evidence | Status |
|---|---|---|---|
| Phase 5 fair comparison has 8 methods, 8 conditions, 5 seeds | Methods / Experimental Validation | `configs/phase5.yaml`; remote strict audit | Supported |
| PPO, PPO-Lagrangian, and PPO-CBF exceed 99% violation under static perturbations | Table 1 and text | `results/phase5/{ppo,ppo_lagr,ppo_cbf}_s*_seed*.json`; remote `analyze_results_5th.py` | Supported |
| HOCBF without GP is safe nominally but fails under model mismatch | Table 1 and text | `results/phase5/ppo_hocbf_*_seed*.json` | Supported |
| GP mean correction reduces HOCBF violations to <=0.18% on six of seven uncertainty scenarios | Abstract, Results, Conclusion | `ppo_gp_hocbf`: S1 0.12, S2 0.04, S4 0.04, S5 0.15, S6 0.18, load-following 0.00; S3 exception 39.65 | Supported |
| Improvement is >500x versus uncorrected HOCBF on the six non-S3 uncertainty scenarios | Abstract, Intro, Results, Conclusion | Ratios versus PPO-HOCBF: S1 855x, S2 2393x, S4 2476x, S5 656x, S6 545x; load-following denominator is 0 and is not used for the ratio | Supported, but interpret as static non-S3 uncertainty scenarios |
| Full margin `epsilon_kappa=1.0` is not empirically optimal | Abstract, Results, Conclusion | Main Phase 5 table plus kappa sweep: S2 and S4 collapse at high kappa; S3 best at 0.1 | Supported |
| S3 kappa=0 failure is 33.5% and kappa=0.1 restores 0% | Abstract, Results, Conclusion | `results/phase5/kappa_sweep/kappa0.0_s3_coupled_seed*.json` mean 33.47%; `kappa0.1` mean 0.00% | Supported |
| Additive S2/S4 are safe at kappa=0 | Results | `results/phase5/kappa_sweep/kappa0.0_s2_pressure_seed*.json`; `kappa0.0_s4_nonlinear_seed*.json` | Supported |
| Deployment envelope collapses to `{0}` at gamma >= 1.5 | Results, Conclusion | `results/phase5/kappa_sweep/kappa*_s3_midstrong_seed*.json`; `kappa*_s3_strong_seed*.json` | Supported |
| Pressure and power show no observed violations; violations are enthalpy-dominated | Results, Supplement | `per_constraint_type` logs in 318/320 Phase 5 root files; two S1 rerun files have aggregate-only logs | Supported with caveat now stated in supplement |
| Safety-filter latency is about 25 ms and NMPC is about 254 ms | Abstract, Results, Supplement | `paper/figures/Figure_5.pdf`; prior timing analysis; main text now avoids machine-specific GPU model | Supported as implemented-baseline comparison |
| Phase 4 generic mixed GP failure is 82.7--99.7% | Deployment Considerations | Historical Phase 4 auxiliary results; not primary M&C evidence | Supported as historical/auxiliary claim only |

## Claims Patched During Closeout

- Removed the statement that the full margin makes the tightened QP "always feasible".
- Changed kappa sensitivity sample count from "3 seeds each" to "2--3 seeds per setting" because three auxiliary cells currently have only two completed seeds.
- Removed bold formatting from poor RoCBF-Net table values.
- Changed "Four key findings" to "Five key findings".
- Rewrote the supplement to remove older JPC/TAC claims that contradicted the M&C result table.

## Remaining Evidence Risks

- The kappa sweep is auxiliary and has 2 completed seeds for S3/S4 at some high-kappa settings. The manuscript now states this explicitly.
- Timing numbers should be treated as implementation-latency measurements, not universal hardware benchmarks.
- Phase 4 is not complete and should not be used as a main evidence base.

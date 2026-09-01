# M&C Major-Revision Claim-Evidence Matrix

Date: 2026-08-31

## Evidence Boundary

The current simulation evidence uses a drift-only fifth-order benchmark with a fixed input matrix ($\Delta g=0$). Its GP input is $[p_m,h_m,N_e]^\top$ and its three targets are nominal-model residual rates. The full five-state predictor and proposed fuel, feedwater, and turbine-valve commands are not GP-kernel inputs. The finite benchmark aligns with the theorem's input-matrix assumption but does not independently prove the uniform residual event or derivative bounds. The earlier root-level `results/phase5/` matrix is retained for traceability but is not a numerical source for the revision.

Plant-controller exports directly support time stamps, operating modes, QP/recovery status, logged margins, saturation flags, and timing. GP lifecycle, DCS permission, Kubernetes lease, and fallback-state-machine details are deployment configuration specifications, not per-record controller-export fields.

## Main Claims

| Claim | Manuscript location | Current evidence | Status |
|---|---|---|---|
| The formal endpoint remains conditional on $\epsilon_\kappa=1$, an independently valid simultaneous residual event, valid derivative bounds, $\Delta g=0$, and full-row robust-QP feasibility. | Section 3.4; Conclusion | Theorem 1 and proof; `sections_mc/methodology.tex` | Formal conditional result; the commissioning multiplier alone is not proof |
| No-GP HOCBF records 10,756 violations in 15,000 mismatch samples; GP mean correction reduces this count to 36. | Abstract; Section 4.2; Conclusion | `results/phase5_qpax_x64_primary_a_20260831/` | Stored primary aggregation |
| The tune rule selects $\epsilon_\kappa=0.02$ before confirmation; the scenario-wise primary row uses 0.02 in S3 and 0 elsewhere and records 0/17,500 observed violations. | Sections 4.2--4.4; Supplemental Material | `results/phase5_commissioning_kappa_tune_20260831/selection_summary.json`; `results/phase5_primary_kappa002_20260831/` | Stored tune/test evidence |
| The fixed selected setting has 66/50,000 violations and 72/50,000 QP rejections on held-out seeds. | Sections 4.3--4.4; Supplemental Material | `selection_summary.json` | Stored held-out evidence; not a zero-probability claim |
| Full-margin operation can lose feasibility under actuator limits. | Section 4.2; Supplemental Material | Primary S3 endpoint records in `results/phase5_qpax_x64_primary_a_20260831/` | Stored mechanism diagnostic |
| Constrained NMPC is an effective reference with 35/17,500 violations and no solver failure in the implemented test. | Section 4.2; Supplemental Material | `results/phase5_drift_only_nmpc_x64_20260831/` | Stored reference result |
| Clean 100/250/500-point GPs pass the controlled commissioning gate; deliberately corrupted fits fail it. | Section 4.5; Supplemental Material | `results/phase5_gp_data_sensitivity_k002_20260831/summary.json` | Controlled synthetic fault-injection diagnostic |
| The matched historian cohort contains 1,950 loaded two-hour windows and reports 12.6%/23.4% cohort reductions. | Section 4.6 | `results/production_validation/matched_window_cohort_summary_20260706.json` | Observational historian evidence |
| The representative pre/post pair changes pressure-error standard deviation from 0.375 to 0.156 MPa under matched load movement. | Section 4.6; Figure 7 | `results/production_validation/figure10_production_retrofit_metrics.json` | Observational historian evidence |
| MW04--MW06 cover 4,320 exported records over 480.7--629.7 MW; controller-export timing remains below the 1,000 ms deadline. | Section 4.6; Figure 8; Table 4 | `results/production_validation/figure11_high_load_controller_metrics.json`; confirmed original plant-controller exports | Direct export evidence |
| The 43 MW04 rows establish only reduced-QP recovery after a guarded pressure-low-row removal. | Section 4.6; Supplemental Material | Confirmed original plant-controller exports; `figure11_high_load_controller_metrics.json` | Direct export evidence; excluded from full-row certificate |

## Excluded or Qualified Claims

- The current revision makes no full-load causal claim from the historian comparison. Routine outage maintenance occurred with strategy integration; the comparison is observational.
- A 2.0 MPa pressure-low recovery guard is an engineering gate derived from the observed MW04 direct-margin range. It is not a theorem-derived safety boundary.
- A controller export at 5 s spacing is not an internal 1 s control-cycle trace. The exported records establish reported status and timing summaries, not every control action.
- The current benchmark does not claim a formal certificate for actuator-gain uncertainty or sampled-data forward invariance.

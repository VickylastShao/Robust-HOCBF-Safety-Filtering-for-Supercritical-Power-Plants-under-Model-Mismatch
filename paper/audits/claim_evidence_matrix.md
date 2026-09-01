# M&C Major-Revision Claim-Evidence Matrix

Date: 2026-09-02

## Evidence Boundary

The confirmatory simulation uses a seven-state, actuator-augmented boiler--turbine benchmark with a fixed input matrix ($\Delta g=0$). The GP kernel input is $[p_m,h_m,N_e]^\top$; the nominal one-step predictor uses all seven states and all three proposed commands. The finite benchmark aligns with the theorem's input-matrix assumption but does not independently prove the uniform residual event, derivative bounds, inter-sample invariance, or robust-QP feasibility.

Historian records support operating context and observational response comparisons. Plant-controller exports directly support timestamps, operating modes, QP/recovery status, logged margins, saturation flags, and timing. GP lifecycle, DCS permission, Kubernetes lease, and fallback-state-machine details are deployment configuration specifications rather than per-record export fields.

## Main Claims

| Claim | Manuscript location | Current evidence | Status |
|---|---|---|---|
| The formal endpoint is conditional on $\epsilon_\kappa=1$, a valid simultaneous residual event and compositional derivative bound, $\Delta g=0$, and full-row robust-QP feasibility. | Section 3.4; Conclusion | Theorem 1, proof, and `paper/sections_mc/methodology.tex` | Formal conditional result; no calibrated setting is called certified |
| No-GP HOCBF records 65,134 violations in 150,000 mismatch samples. | Abstract; Section 4.2; Conclusion | `results/phase5_ccs7_confirmatory_20260902/summary.json` | Five seeds, ten 500-sample rollouts per condition |
| The predeclared S3 selector chooses $\epsilon_\kappa=0$; the fixed setting records 0/50,000 violations and 0/50,000 QP rejections on held-out seeds. | Sections 4.2--4.3; Supplemental Section S2 | `results/phase5_ccs7_kappa_20260902/selection_summary.json` | Finite-sample tune/test evidence, not a zero-probability claim |
| Tune-selected RoCBF-SF and NMPC each record 0/175,000 observed violations in the primary seven-condition sweep. | Abstract; Section 4.2; Conclusion | `results/phase5_ccs7_confirmatory_20260902/summary.json`; `results/phase5_ccs7_nmpc_20260902/` | Equal reported violation count; no universal superiority claim |
| The full implemented-margin endpoint rejects 149,800/150,000 mismatch-scenario QPs and reproduces the upstream mismatch violation count. | Section 4.2; Supplemental Section S1 | `results/phase5_ccs7_confirmatory_20260902/summary.json` | Feasibility stress test; theorem feasibility condition is not met |
| Clean 100/250/500-point GPs pass the controlled gate; 100/250-point fits degrade under bounded synthetic target corruption, while the 500-point fits retain zero observed violations in this test. | Section 4.3; Supplemental Section S3 | `results/phase5_ccs7_gp_sensitivity_20260902/summary.json` | Controlled fault injection, not measured plant bad-point rate |
| The recomputed historian cohort contains 512 one-to-one matched pairs; median pressure-residual standard deviation changes from 0.523 to 0.502 MPa and the post-day cluster-bootstrap interval crosses zero. | Section 4.5; Supplemental Section S5 | `results/production_validation/cohort_recomputed_20260902/matched_window_cohort_recomputed.json`; pair table | Operating-context consistency, not causal inference |
| A separate native-5 s matched pair changes pressure-error standard deviation from 0.375 to 0.156 MPa. | Section 4.5; matched-response figure | `results/production_validation/figure10_production_retrofit_metrics.json` | Observational response diagnostic; not a member of the 300 s cohort |
| MW04--MW06 contain 4320 exported records over 480.7--629.7 MW and remain below the 1000 ms task deadline. | Section 4.5; high-load figure and table | `results/production_validation/figure11_high_load_controller_metrics.json`; confirmed original plant-controller exports | Direct controller-export evidence |
| The 43 MW04 rows establish guarded reduced-QP recovery after pressure-low-row removal, not full-row feasibility. | Section 4.5; Supplemental Section S5 | Confirmed original plant-controller exports and derived metrics | Excluded from the full-row certificate claim |

## Excluded or Qualified Claims

- The revision makes no causal retrofit claim and no full-load response-improvement claim. Routine outage maintenance occurred with algorithm integration.
- The benchmark tune/test result $\epsilon_\kappa=0$ and field setting $\epsilon_\kappa=0.1$ are independent finite-sample commissioning outcomes.
- A 2.0 MPa pressure-low recovery guard is an engineering gate, not a theorem-derived safety boundary.
- A controller export at 5 s spacing is not an internal 1 s control-cycle trace.
- The current benchmark does not provide a formal certificate for actuator-gain uncertainty or sampled-data forward invariance.

# Matched-Window Production-Evidence Traceability Note

Date: 2026-09-01

The current manuscript uses four distinct production-evidence layers. Their
roles must not be merged.

1. `matched_window_cohort_summary_20260706.json` is the aggregate historian
   screening source for 1950 retained loaded windows and the period-median
   reductions of 12.6% and 23.4%.
2. `figure10_production_retrofit_metrics.json` is the canonical source for the
   native 5 s matched-pair metrics: 1440 records per period, pressure-error
   standard deviation 0.375434 to 0.155954 MPa, and absolute-error p95
   0.905646 to 0.508274 MPa.
3. `low_mid_load_historian_context/MW01_MW03_historian_context.csv` supplies
   only generator active power and main-steam pressure for the operating-
   envelope figure. It is not a source of controller-internal QP evidence.
4. `controller_exports_public/evidence_manifest.json` indexes the confirmed
   MW04--MW06 high-load controller excerpts and their original/public hashes.

The corrected controller mapping is `PM == DCS2_20HAG10CP101` (separator
pressure), `HM == DCS2_SEPARATOUT_ENTH`, and `NE == 20CQTP_MW`.
`DCS2_MAIN_PRESS` is independent main-steam pressure and is not `PM`.

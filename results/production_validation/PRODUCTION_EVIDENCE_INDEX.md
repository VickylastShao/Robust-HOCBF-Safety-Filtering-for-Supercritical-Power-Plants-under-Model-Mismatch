# Production Evidence Index

This index fixes the role, field semantics, and claim boundary of each public
production artifact used in the M&C major revision.

| Evidence ID | Public artifact | Supported claim | Explicitly not supported |
|---|---|---|---|
| PE-01 | `unit_660mw_historian_24h_summary.json` | 24 h loaded operating envelope at 300 s spacing | safety-filter constraint compliance |
| PE-02 | `matched_window_cohort_summary_20260706.json` | aggregate screening of 1950 loaded two-hour windows | randomized or isolated causal effect |
| PE-03 | `figure10_production_retrofit_metrics.json` | native 5 s matched pre/post pressure-response metrics | high-load response improvement; pre-retrofit QP status |
| PE-04 | `low_mid_load_historian_context/MW01_MW03_historian_context.csv` | low-to-mid-load generator power and main-steam-pressure context | controller mode, QP status, CBF margins, or controller state `PM` |
| PE-05 | `controller_exports_public/MW04_CONTROLLER_EXPORT_5S.csv` | high-load execution, guarded reduced-QP recovery, one fuel-saturation flag, margins, and timing | full-row certificate during recovered cycles |
| PE-06 | `controller_exports_public/MW05_CONTROLLER_EXPORT_5S.csv` | routine high-load full-QP execution, margins, and timing | pre/post causal performance comparison |
| PE-07 | `controller_exports_public/MW06_CONTROLLER_EXPORT_5S.csv` | routine high-load full-QP execution, margins, and timing | pre/post causal performance comparison |

## Source Integrity

- `low_mid_load_historian_context/evidence_manifest.json` records the private
  source basenames and SHA-256 values for PE-04.
- `controller_exports_public/evidence_manifest.json` records the private
  original and public derivative SHA-256 values for PE-05--PE-07.
- The only high-load anonymization transformations are
  `UNIT -> 660MW-USC-Unit` and
  `CTRL_VER -> RoCBF-SF-v2-public`; all other fields are copied verbatim.

## Semantic Invariant

`PM` is separator pressure, not main-steam pressure. The exact exported mapping
is `PM == DCS2_20HAG10CP101`. `DCS2_MAIN_PRESS` is an independent main-steam
pressure measurement. `PST_HI` is a main-steam-pressure replay upper bound.

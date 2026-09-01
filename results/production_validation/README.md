# Production Validation Evidence

This directory contains the bounded evidence used for the manuscript's
measured-operation claims for a 660 MW ultra-supercritical unit. Public
derivatives are separated from proprietary originals and from historical
development packages.

## Canonical Evidence Layers

1. **Matched-window cohort screening**
   - `cohort_recomputed_20260902/matched_window_cohort_recomputed.json`
   - `cohort_recomputed_20260902/matched_window_pairs.csv`
   - 512 one-to-one matched two-hour window pairs obtained from 14,675
     pre-retrofit and 517 post-retrofit loaded candidates.
   - windows advance every 30 minutes and overlap; cohort estimates and the
     post-day cluster bootstrap are descriptive rather than independent-sample
     causal inference.
2. **Native 5 s pre/post historian pair**
   - `figure10_production_retrofit_metrics.json`
   - supports the plotted pressure-response comparison and its exact counts.
3. **Low-to-mid-load historian context**
   - `low_mid_load_historian_context/MW01_MW03_historian_context.csv`
   - supplies generator active power and **main-steam pressure** for Figure 6.
   - these records do not contain or support controller-internal QP fields.
4. **Confirmed high-load plant-controller exports**
   - `controller_exports_public/MW04_CONTROLLER_EXPORT_5S.csv`
   - `controller_exports_public/MW05_CONTROLLER_EXPORT_5S.csv`
   - `controller_exports_public/MW06_CONTROLLER_EXPORT_5S.csv`
   - supports the reported operation mode, QP/recovery status, direct and
     logged margins, saturation flags, and execution timing.

`PRODUCTION_EVIDENCE_INDEX.md` is the claim-to-artifact index. The two JSON
manifests under the public evidence directories record source and derivative
SHA-256 values.

## Corrected Field Semantics

The current controller-export mapping is:

| Controller symbol | Exported plant field | Meaning |
|---|---|---|
| `PM` | `DCS2_20HAG10CP101` | separator pressure |
| `HM` | `DCS2_SEPARATOUT_ENTH` | separator-outlet specific enthalpy |
| `NE` | `20CQTP_MW` | generator active power |
| not `PM` | `DCS2_MAIN_PRESS` | independent main-steam pressure |
| replay bound | `PST_HI` | main-steam-pressure upper replay bound |

The public high-load excerpts preserve all timestamps, measured values,
commands, status fields, margins, and timing values. Only `UNIT` and `CTRL_VER`
are replaced by public identifiers. The field-level mapping is recorded in
`controller_exports_public/controller_export_field_map.csv`.

## Evidence Boundary

The proprietary originals are retained outside the public evidence boundary
and are excluded by `.gitignore`. Their basenames and SHA-256 values are
recorded in the public manifests. No controller-internal values are assigned to
the pre-retrofit PID period. The MW01--MW03 legacy mixed package is preserved in
a restricted archive and is not part of the current public controller-evidence
chain.

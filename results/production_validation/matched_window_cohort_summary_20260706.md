# Matched-Window Cohort Summary

Date: 2026-07-06

This file restores the aggregate cohort statistics used to contextualize the
high-resolution 660 MW ultra-supercritical unit matched-window figure. The cohort statistic is
separate from the native 5 s Figure 7 pair, the low/mid-load historian context,
and the three high-load plant-controller exports.

## Restored Manuscript-Facing Cohort Metrics

- Retained loaded two-hour pre/post-retrofit windows: 1950.
- Period-median reduction in pressure setpoint-residual standard deviation:
  12.6%.
- Period-median reduction in 95th-percentile absolute pressure residual:
  23.4%.
- Reduction formula: `(median_pre - median_post) / median_pre`.

## Scope Boundary

These numbers are aggregate historian screening statistics. They support the
claim that the plotted native 5 s pair is embedded in a broader load-matched
screening exercise. They are not randomized causal estimates and they are not
controller-log execution statistics.

The directly traceable native 5 s pair remains:

- `results/production_validation/figure10_production_retrofit_metrics.json`
- `results/production_validation/raw/unit_660mw_historical_pre_match_20251105_1100_5s.csv`
- `results/production_validation/raw/unit_660mw_post_match_20260625_1230_5s.csv`

The directly traceable low/mid-load historian context is:

- `results/production_validation/low_mid_load_historian_context/MW01_MW03_historian_context.csv`
- `results/production_validation/low_mid_load_historian_context/evidence_manifest.json`

The confirmed high-load plant-controller excerpts are indexed by:

- `results/production_validation/controller_exports_public/evidence_manifest.json`

## Calculation Provenance

The retained aggregate values follow the production-upgrade analyzer logic:
candidate windows are grouped by period and the pre/post period medians are
compared for `pressure_setpoint_residual_std` and
`pressure_setpoint_residual_abs_p95`. The wide-scan proprietary intermediate
files are not included in the current public package; the manuscript-facing
aggregate is fixed here so that the source boundary is explicit.

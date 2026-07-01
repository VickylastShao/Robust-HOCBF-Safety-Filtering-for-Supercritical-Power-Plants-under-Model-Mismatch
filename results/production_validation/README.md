# Production Validation Snapshot

This directory stores derived artifacts from a China Resources Xiantao Unit 2
plant-historian query used to ground the manuscript's measurement channels and
operating variability.

## Source Boundary

- Database: `yulin`
- Schema/table: `unit2.report_hourly`
- Query window: `2026-06-24T08:00` to `2026-07-02T02:00`
- Sampling: hourly DCS operating report snapshots
- Public raw data: no

The raw historian CSV is written under `results/production_validation/raw/`,
which is intentionally ignored by Git because it contains proprietary plant
operating records. The committed JSON summary contains only derived statistics
and point metadata needed to audit the manuscript figure.

## Reproduction

Start the local VMware/PostgreSQL relay, then run:

```bash
PGHOST=127.0.0.1 \
PGPORT=15432 \
PGUSER=postgres \
PGPASSWORD='<local password>' \
PGDATABASE=yulin \
python3 scripts/production_data/extract_xiantao_hourly_report.py \
  --schema unit2 \
  --start 2026-06-24 \
  --end 2026-07-02 \
  --start-ts 2026-06-24T08:00 \
  --end-ts 2026-07-02T02:00 \
  --raw-output results/production_validation/raw/xiantao_unit2_loaded_hourly_2026-06-24T08_2026-07-02T02.csv \
  --summary-output results/production_validation/xiantao_unit2_loaded_hourly_summary.json \
  --figure-output paper/figures/Figure_9_production_historian.pdf
```

The figure is also copied into the current M&C submission source folder before
building the manuscript package.

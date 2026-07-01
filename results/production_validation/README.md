# Production Validation Snapshot

This directory stores derived artifacts from China Resources Xiantao Unit 2
plant-historian queries used to ground the manuscript's measurement channels
and operating variability.

## Current Figure Source

- Interface: CRICP FastAPI historian adapter
- Endpoint: `/gethistrange_millisecondtimestamp`
- Unit: Xiantao Unit 2
- Query window: `2026-07-01T02:33:56+08:00` to `2026-07-02T02:28:56+08:00`
- Sampling: 300 s history samples
- Public raw data: no

The current Fig. 9 is generated from
`results/production_validation/xiantao_unit2_fastapi_24h_summary.json` and the
private raw files under `results/production_validation/raw/`.

## PostgreSQL Report Fallback

- Database: `yulin`
- Schema/table: `unit2.report_hourly`
- Query window: `2026-06-24T08:00` to `2026-07-02T02:00`
- Sampling: hourly DCS operating report snapshots
- Public raw data: no

The PostgreSQL hourly report can still be used as a low-frequency fallback, but
the manuscript figure now uses the FastAPI history snapshot above.

## Data Boundary

Raw historian CSV/JSON files are written under
`results/production_validation/raw/`, which is intentionally ignored by Git
because it contains proprietary plant operating records. The committed JSON
summaries contain only derived statistics and point metadata needed to audit the
manuscript figure.

## Reproduction

If the FastAPI base URL is directly reachable, run:

```bash
CRICP_BASE_URL='<site FastAPI base URL>' \
python3 scripts/production_data/extract_xiantao_fastapi_history.py \
  --start-sec 1782844436 \
  --end-sec 1782930836 \
  --interval-sec 300 \
  --raw-output results/production_validation/raw/xiantao_unit2_fastapi_24h_300s.csv \
  --summary-output results/production_validation/xiantao_unit2_fastapi_24h_summary.json \
  --figure-output paper/figures/Figure_9_production_historian.pdf
```

When WSL cannot directly reach the site network, fetch a bounded response via
the VMware guest and run the same script with `--input-json`:

```bash
python3 scripts/production_data/extract_xiantao_fastapi_history.py \
  --input-json results/production_validation/raw/xiantao_unit2_fastapi_24h_300s.json \
  --interval-sec 300 \
  --raw-output results/production_validation/raw/xiantao_unit2_fastapi_24h_300s.csv \
  --summary-output results/production_validation/xiantao_unit2_fastapi_24h_summary.json \
  --figure-output paper/figures/Figure_9_production_historian.pdf
```

For the PostgreSQL report fallback, start the local VMware/PostgreSQL relay,
then run:

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

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A UK Energy Performance Certificate (EPC) analytics pipeline processing ~29.2M rows using **DuckDB + dbt + Polars**. The pipeline uses Medallion Architecture: raw CSV ingestion → staging naturalization → mart normalization → analytics views.

## Common Commands

All dbt commands run from inside `ducklake_energy_uk/`:

```bash
# Install dbt package dependencies (run once or after packages.yml changes)
cd ducklake_energy_uk && dbt deps

# Run full pipeline
cd ducklake_energy_uk && dbt run

# Run a single model
cd ducklake_energy_uk && dbt run --select stg_epc__domestic
cd ducklake_energy_uk && dbt run --select dim_properties

# Run tests
cd ducklake_energy_uk && dbt test

# Run tests for a single model
cd ducklake_energy_uk && dbt test --select stg_epc__domestic

# Compile SQL without running (useful for debugging Jinja)
cd ducklake_energy_uk && dbt compile --select dim_properties

# Generate and serve docs
cd ducklake_energy_uk && dbt docs generate && dbt docs serve
```

Python scripts run from repo root with the venv active:

```bash
source dbt-env/bin/activate
python bulk_load_epc.py   # Ingest CSVs into DuckDB
python eda_uk_energy.py   # Generate Plotly EDA reports
```

## Project Structure

```
ducklake_energy_uk/          # dbt project
  models/
    staging/epc/             # Bronze→Silver: type casting, null filtering, standardization
    marts/energy/            # Silver→Gold: star schema (dims + facts)
    marts/analytics/         # Gold→Views: pre-joined analytical views
  macros/                    # Custom Jinja macros
  seeds/                     # Static reference data (CSV)
  tests/                     # Singular data tests
bulk_load_epc.py             # Raw CSV → DuckDB raw.epc_domestic
eda_uk_energy.py             # DuckDB → Polars → Plotly reports
dbt-env/                     # Python virtualenv (do not edit)
all-domestic-certificates/   # Raw EPC CSVs (git-ignored, ~50GB)
```

## Architecture

### Data Flow

1. **Ingestion**: `bulk_load_epc.py` bulk-loads all CSVs as `VARCHAR` into `raw.epc_domestic` in DuckDB (`dev.duckdb` / `prod.duckdb`). Loading as VARCHAR first avoids mid-load type failures on messy real-world data.

2. **Staging** (`stg_epc__domestic`): Casts types, standardizes fuel types and construction age bands via `CASE` expressions, filters invalid rows (`energy_rating != 'INVALID!'`, not-null UPRN/inspection date). All surrogate keys are created using `dbt_utils.generate_surrogate_key` (MD5 hash).

3. **Marts — Star Schema**:
   - `dim_properties`: One row per UPRN (latest inspection only) — deduplication via `ROW_NUMBER() OVER (PARTITION BY uprn ORDER BY inspection_at DESC)`
   - `dim_locations`: One row per postcode, surrogate key from postcode
   - `fct_certificates`: Thin fact table — foreign keys + metrics (CO2, costs, ratings) only
   - `fct_property_aggregations`: Pre-aggregated by county/property_type/tenure/age_band/year for fast BI queries

4. **Analytics Views** (`marts/analytics/`): Pre-joined views over the star schema for regional performance, construction age analysis, and efficiency gaps.

### DuckDB Profiles

Defined in `~/.dbt/profiles.yml`:
- `dev` target: `dev.duckdb` (single thread, default)
- `prod` target: `prod.duckdb` (4 threads)

Switch targets with `dbt run --target prod`.

### Key Patterns

- **Surrogate keys**: Always `{{ dbt_utils.generate_surrogate_key(['column']) }}` — never auto-increment integers. This keeps keys deterministic across load batches.
- **Latest-property deduplication**: `ROW_NUMBER() OVER (PARTITION BY uprn ORDER BY inspection_at DESC, lodgement_at DESC)` — used in `dim_properties` to get the most recent state per building.
- **Zero-copy analytics**: `conn.query("SELECT ...").pl()` returns a Polars DataFrame via Apache Arrow with no serialization overhead.
- **Materialization**: Core dimension/fact tables use `{{ config(materialized='table') }}`; analytical summaries use `view`.

### dbt Package

`dbt_utils` v1.1.1 (dbt-labs) is the only external dependency — used for `generate_surrogate_key` and generic tests.

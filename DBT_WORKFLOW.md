# dbt Masterclass: Scaling UK Energy Data (29.2M Rows)

**A Professional Engineering Journey from Raw Ingestion to a Normalized BI Layer.**

This document explains every concept used in this project in plain English — written for anyone, regardless of whether you have a data engineering background. It covers the "why" behind each decision, not just the "what."

---

## Part 1: Foundations — What Are We Actually Doing?

Before touching any dbt-specific concepts, it helps to understand the broader picture.

### What is a Database?

A database is a structured system for storing and retrieving data. When you have 29.2 million rows of data, you can't keep it in a spreadsheet or a list in memory — you need a system that can:
- Store data on disk efficiently
- Answer complex questions quickly ("What is the average CO2 for all E-rated flats in Yorkshire, grouped by decade?")
- Handle multiple users and prevent corrupted reads

**SQL (Structured Query Language)** is the standard language used to ask these questions. A query like:

```sql
SELECT county, AVG(co2_emissions_current_tonnes_per_year)
FROM fct_certificates
GROUP BY county
ORDER BY 2 DESC
```

...reads as: "For each county, calculate the average CO2, and show me the highest first."

### What is ELT vs ETL?

The traditional approach to data pipelines was **ETL**: Extract from source → Transform outside the database → Load into the database. Transformation happened in Python/Java code before the data ever touched the warehouse.

Modern pipelines use **ELT**: Extract → Load into the database first → Transform *inside* the database using SQL.

**Why the shift?** Modern databases (like DuckDB) are so fast at SQL that it is more efficient to load raw data first, then transform it with SQL inside the warehouse. You also get a permanent record of the raw data, which means you can re-run any transformation without going back to the source.

This project is pure ELT:
1. `bulk_load_epc.py` **Extracts** CSV files and **Loads** them into DuckDB as-is.
2. dbt **Transforms** the raw data inside DuckDB using SQL models.

### What is dbt?

dbt (Data Build Tool) is the "T" in ELT. It does *only* transformations — it does not move data in or out of your database. What it adds on top of raw SQL:

| Raw SQL (without dbt) | dbt |
|---|---|
| Scripts run in whatever order you remember | Automatic dependency ordering via a DAG |
| No testing — you find out about bad data when dashboards break | `dbt test` validates data quality before it reaches analysts |
| SQL files scattered in folders | Organized layers: staging → marts → analytics |
| No documentation | Auto-generated lineage graphs and column-level docs |
| Duplicated logic copy-pasted everywhere | Reusable Jinja macros |

Think of dbt as bringing **software engineering practices** (version control, testing, modularity, documentation) to SQL.

---

## Part 2: The Medallion Architecture

The **Medallion Architecture** (also called Bronze/Silver/Gold) is an industry-standard pattern for organizing data warehouse layers. Each layer has a specific responsibility and quality contract.

```
Raw CSVs
   ↓
[Bronze] raw.epc_domestic        ← Exact copy of source. Never modified.
   ↓
[Silver] stg_epc__domestic       ← Cleaned, typed, standardized. Trustworthy.
   ↓
[Gold]   dim_*, fct_*            ← Business-structured. Fast to query.
   ↓
[Analytics] v_*                  ← Pre-joined views for BI tools and EDA.
```

### Why not transform in one step?

Each layer serves a distinct purpose. If you go straight from raw to mart:
- Debugging is hard: you can't tell if a problem came from bad source data or bad business logic.
- You lose the audit trail: you can never see what the original messy data looked like.
- Reprocessing is risky: if you need to change a transformation, you might destroy data you can't get back.

With layered architecture, you can re-run any layer independently without touching the others.

---

## Part 3: Core dbt Concepts

### Models

A **model** is a single `.sql` file that defines one table or view. The filename becomes the table name. dbt compiles and executes these files against your database.

```
models/
  staging/epc/
    stg_epc__domestic.sql       → creates table: stg_epc__domestic
  marts/energy/
    dim_properties.sql          → creates table: dim_properties
    fct_certificates.sql        → creates table: fct_certificates
```

**Naming conventions** used in this project:
- `stg_` prefix = staging model (Silver layer)
- `dim_` prefix = dimension table (Gold layer)
- `fct_` prefix = fact table (Gold layer)
- `v_` prefix = analytical view (reporting layer)
- Double underscores (`stg_epc__domestic`) = source name separator (`epc` is the source, `domestic` is the dataset)

---

### Sources

A **source** is how dbt refers to raw data that dbt itself did not create — in our case, the table loaded by `bulk_load_epc.py`.

Defined in `models/staging/epc/src_epc.yml`:

```yaml
sources:
  - name: epc_raw
    schema: raw
    tables:
      - name: epc_domestic       # → raw.epc_domestic in DuckDB
        columns:
          - name: LMK_KEY
            tests:
              - unique
              - not_null
```

In SQL models, you reference a source with `{{ source('epc_raw', 'epc_domestic') }}` rather than a raw table name. This allows dbt to:
- Track lineage all the way back to the raw source
- Run freshness checks (alert if the source hasn't been updated)
- Abstract the schema name so it can change without updating every model

---

### The `ref()` Function — How Models Depend on Each Other

The most important concept in dbt. Instead of writing `FROM stg_epc__domestic` in a mart model, you write `FROM {{ ref('stg_epc__domestic') }}`.

This tells dbt two things:
1. **Dependency**: "This model needs `stg_epc__domestic` to exist first."
2. **Environment awareness**: In dev, `ref()` resolves to the dev database; in prod, to the prod database.

dbt reads all `ref()` calls across all models and builds a **DAG** (Directed Acyclic Graph) — a dependency map that determines the correct execution order. You never have to manually order your scripts again.

```
raw.epc_domestic (source)
    ↓
stg_epc__domestic
    ↓              ↓              ↓              ↓
dim_properties  dim_locations  fct_certificates  fct_property_energy_performance
    ↓              ↓              ↓
         fct_property_aggregations
                    ↓              ↓              ↓
          v_regional_...   v_construction_...  v_property_efficiency_gaps
```

---

### Materialization Strategies

**Materialization** controls how dbt persists a model. Set at model level with `{{ config(materialized='table') }}` or globally in `dbt_project.yml`.

#### `view`
The SQL is saved as a *query definition*. Every time you query it, the database re-runs the SQL. Nothing is stored.

```sql
{{ config(materialized='view') }}
-- v_regional_energy_performance.sql
-- This view always reflects the latest fct_certificates data
-- but costs a query execution every time it's read
```

**Use when**: The query is cheap, or you always want the freshest data (analytics views).

#### `table`
Runs the SQL once and saves the result as a real table. Subsequent queries read the pre-built result — no recomputation.

```sql
{{ config(materialized='table') }}
-- dim_properties.sql
-- 29.2M rows deduplicated to ~15M unique properties
-- Built once with dbt run, then read instantly by any query
```

**Use when**: The model is expensive to compute and is queried frequently (all dims and facts here).

#### `incremental`
On first run: builds the full table. On subsequent runs: only processes rows newer than the last run. This is the strategy for very large tables in production where a full refresh would take hours.

**Not used in this project yet**, but it's the next step for production-grade pipelines.

---

### Seeds

Seeds are static CSV files in the `seeds/` directory that dbt loads into the database as-is. Useful for small reference tables that change rarely (country codes, tax brackets, mapping tables).

**Rule of thumb**: Seeds are appropriate for < 1MB of data. Larger static datasets should be loaded by the ingestion script instead.

---

### Macros (Jinja Templating)

dbt models are not plain SQL — they're **Jinja templates** that compile to SQL. Jinja is a templating language (same one used in Django/Flask web frameworks) that lets you embed logic inside SQL.

**Why Jinja?**
- Reuse logic across models without copy-pasting
- Adapt queries to different databases (DuckDB syntax vs Snowflake syntax)
- Generate SQL dynamically from loops, conditions, and variables

Example of the most-used macro in this project:

```sql
-- Without macro (fragile, duplicated everywhere):
md5(cast(coalesce(postcode, '') as varchar)) as location_id

-- With dbt_utils macro (clean, consistent, cross-database compatible):
{{ dbt_utils.generate_surrogate_key(['postcode']) }} as location_id
```

The macro handles null-safety, type casting, and database dialect differences automatically.

---

### Tests (Data Quality as Code)

dbt tests are assertions that run against your data. If any assertion fails, `dbt test` exits with an error and reports exactly which rows violated the rule.

#### Generic Tests (defined in `.yml` schema files)

```yaml
# stg_epc__domestic.yml
columns:
  - name: certificate_key
    tests:
      - unique          # No two rows can have the same certificate_key
      - not_null        # certificate_key must never be NULL

  - name: energy_rating_current
    tests:
      - accepted_values:
          values: ['A', 'B', 'C', 'D', 'E', 'F', 'G']
          # Any other value (like 'INVALID!') causes the test to fail
```

These tests protect downstream models. If a bad energy rating slips through staging, every mart and analytics view that depends on it becomes unreliable.

#### Singular Tests (custom SQL in `tests/`)

For complex assertions that can't be expressed in YAML:

```sql
-- tests/assert_no_future_inspections.sql
-- Fails if any certificate has an inspection date in the future
select *
from {{ ref('stg_epc__domestic') }}
where inspection_at > current_date
```

Any test that returns *zero rows* passes. Any test that returns rows fails.

#### dbt_utils Generic Tests

The `dbt_utils` package adds extra generic tests beyond the built-ins:
- `dbt_utils.unique_combination_of_columns` — composite unique key checks
- `dbt_utils.not_null_proportion` — "at least 90% of rows must have this value"
- `dbt_utils.recency` — "this table should have been updated within the last day"

---

### Documentation

dbt auto-generates documentation from:
1. `description:` fields in `.yml` files
2. Column-level comments
3. The dependency graph inferred from `ref()` calls

```bash
dbt docs generate   # Compiles docs to target/catalog.json
dbt docs serve      # Opens browser at localhost:8080
```

The documentation portal includes a **visual DAG** (lineage graph) showing exactly which models feed which — invaluable for debugging and onboarding.

---

## Part 4: This Project's Data Models in Detail

### Bronze Layer: `raw.epc_domestic`

Created by `bulk_load_epc.py` using DuckDB's `read_csv_auto()`. All 100+ columns are loaded as `VARCHAR` (text), even numbers and dates.

**Why load everything as text?** Real-world CSV data is messy. A column called `construction_age_band` might contain values like `"England and Wales: before 1900"`, `"1990"`, `" "` (blank), or `"invalid_format"` — none of which can be safely auto-cast to a date or integer. Loading as VARCHAR first means the ingestion never fails due to type errors. Type safety is enforced in the Silver layer where we have full control.

---

### Silver Layer: `stg_epc__domestic`

The naturalization model. Takes raw text and produces trustworthy, typed data.

**Key transformations:**

**1. Type Casting**
```sql
cast(inspection_date as date)                          as inspection_at,
cast(co2_emissions_current as decimal(10,2))           as co2_emissions_current_tonnes_per_year,
cast(total_floor_area as decimal(10,2))                as total_floor_area_sqm,
cast(number_habitable_rooms as integer)                as count_habitable_rooms,
```

**2. Rating Validation — Null out anything not A-G**
```sql
case
    when current_energy_rating in ('A','B','C','D','E','F','G') then current_energy_rating
    else null
end as energy_rating_current
```

This is more defensive than filtering the row entirely — it lets us keep rows where the rating was garbled but all other data is valid.

**3. Construction Age Band Standardization**

The raw data contains 30+ different text formats for the same decade:
- `"England and Wales: before 1900"`
- `"Scotland: before 1919"`
- `"1900"` (a raw year as text)

These all collapse to 8 clean bands:

```sql
case
    when construction_age_band like '%before 1900%' then 'Pre-1900'
    when construction_age_band like '%1900-1929%'   then '1900-1929'
    when construction_age_band like '%1930-1949%'   then '1930-1949'
    -- ... and so on
    when construction_age_band ~ '^[0-9]{4}$' then
        case when cast(construction_age_band as integer) < 1900 then 'Pre-1900'
             ...
        end
    else 'Unknown'
end as construction_age_band
```

The `~` operator is a regex match in DuckDB — `'^[0-9]{4}$'` means "exactly 4 digits, nothing else" — used to catch raw year values like `"1985"`.

**4. Fuel Type Standardization**

Same approach — 20+ raw values collapse to 7 clean categories:

```sql
case
    when lower(main_fuel) like '%gas%'        then 'Mains Gas'
    when lower(main_fuel) like '%electricity%' then 'Electricity'
    when lower(main_fuel) like '%oil%'         then 'Heating Oil'
    when lower(main_fuel) like '%lpg%'         then 'LPG'
    when lower(main_fuel) like '%coal%'
      or lower(main_fuel) like '%anthracite%'  then 'Solid Fuel'
    when lower(main_fuel) like '%wood%'
      or lower(main_fuel) like '%biomass%'     then 'Biomass'
    else 'Other/Unknown'
end as main_fuel
```

`lower()` is used first to make matching case-insensitive — raw data often mixes `"Mains Gas"`, `"mains gas"`, `"MAINS GAS"`.

**5. Row-level Filtering**

```sql
where current_energy_rating is not null
  and current_energy_rating != 'INVALID!'
  and uprn is not null
  and inspection_date is not null
```

Only rows that have a valid UPRN, a clean rating, and an inspection date are passed downstream.

---

### Gold Layer: The Star Schema

#### `dim_properties` — One Row Per Building

**The deduplication problem**: A UPRN (Unique Property Reference Number) identifies a specific building. That same building might have been inspected in 2010, 2015, and 2022. We only want its *current* state in the dimension table.

**Solution — Window Functions:**

```sql
with ranked_properties as (
    select
        uprn, property_type, built_form, construction_age_band,
        tenure, total_floor_area_sqm,
        row_number() over (
            partition by uprn                        -- "for each unique building..."
            order by inspection_at desc,             -- "...rank inspections newest-first"
                     lodgement_at desc               -- "tie-break by lodgement date"
        ) as property_rank
    from stg_epc__domestic
    where uprn is not null
),

latest_characteristics as (
    select
        {{ dbt_utils.generate_surrogate_key(['uprn']) }} as property_id,
        uprn, property_type, built_form, construction_age_band,
        tenure, total_floor_area_sqm, count_habitable_rooms,
        count_heated_rooms, count_extensions
    from ranked_properties
    where property_rank = 1      -- Keep only the most recent inspection per building
)
```

**Understanding `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)`:**

`PARTITION BY uprn` divides all rows into groups — one group per UPRN. Within each group, `ORDER BY inspection_at DESC` sorts newest first. `ROW_NUMBER()` then assigns 1, 2, 3... to each row *within its group*. Filtering to `property_rank = 1` keeps exactly one row per building — the most recent.

This is done entirely in SQL — no Python, no external tools. The database engine processes all 29.2M rows in one pass.

#### `dim_locations` — One Row Per Postcode

```sql
select distinct postcode, post_town, local_authority, constituency, county
from stg_epc__domestic
where postcode is not null
```

`DISTINCT` keeps only unique combinations. A surrogate key is generated from postcode:
```sql
{{ dbt_utils.generate_surrogate_key(['postcode']) }} as location_id
```

This creates a compact geographic registry: ~1.7M unique postcodes out of 29.2M rows.

#### `fct_certificates` — One Row Per Inspection Event

The core fact table. Each row is one EPC assessment. It contains **only** metrics and foreign keys — no descriptive attributes (those are in the dimension tables).

```sql
select
    {{ dbt_utils.generate_surrogate_key(['certificate_id']) }} as certificate_key,
    certificate_id,
    {{ dbt_utils.generate_surrogate_key(['uprn']) }}      as property_id,   -- FK to dim_properties
    {{ dbt_utils.generate_surrogate_key(['postcode']) }}   as location_id,   -- FK to dim_locations

    -- All the metrics (the "what happened")
    energy_rating_current, energy_rating_potential,
    energy_efficiency_current, energy_efficiency_potential,
    co2_emissions_current_tonnes_per_year,
    co2_emissions_potential_tonnes_per_year,
    lighting_cost_current, heating_cost_current, hot_water_cost_current,

    inspection_at, lodgement_at
from stg_epc__domestic
where certificate_id is not null
```

**Why separate facts from dimensions?** Joining a 29M-row fact table with a 1.7M-row location table is far faster than querying a single 29M-row table with 100 columns. The database only needs to read the specific columns requested, not the entire wide row.

#### `fct_property_energy_performance` — Efficiency Gain Analysis

A supplementary fact table calculated from staging data directly. Adds a derived metric: `total_estimated_annual_cost_current` (sum of heating + hot water + lighting costs) and `efficiency_gain_potential` (how much the property *could* improve with upgrades).

```sql
select
    certificate_id, uprn, address1, postcode,
    energy_rating_current, energy_rating_potential,
    energy_efficiency_current, energy_efficiency_potential,
    (energy_efficiency_potential - energy_efficiency_current) as efficiency_gain_potential,
    (heating_cost_current + hot_water_cost_current + lighting_cost_current) as total_estimated_annual_cost_current
from stg_epc__domestic
```

#### `fct_property_aggregations` — Pre-rolled BI Layer

For dashboards and EDA, querying 29M raw fact rows every time is wasteful. This model pre-aggregates results by every dimension combination useful for reporting:

```sql
select
    county, property_type, tenure, construction_age_band,
    date_trunc('year', inspection_at) as inspection_year,
    count(*)                                         as certificate_count,
    round(avg(energy_efficiency_current), 2)         as avg_energy_efficiency,
    round(sum(co2_emissions_current_tonnes_per_year), 2) as total_co2_emissions
from fct_certificates
left join dim_properties on ...
left join dim_locations  on ...
group by 1, 2, 3, 4, 5
```

`date_trunc('year', inspection_at)` truncates a date to just its year (e.g., `2022-09-14` → `2022-01-01`), enabling year-over-year groupings.

The result is a compact table (~100K rows instead of 29M) that powers the EDA script and interactive Plotly dashboards with sub-millisecond query times.

---

### Analytics Layer: Views

These views pre-join the star schema tables for specific analytical questions. They are `materialized='view'` — they always reflect the latest data in the underlying tables but have no storage cost.

| View | Joins | Answers |
|---|---|---|
| `v_regional_energy_performance` | `fct_certificates` + `dim_locations` | Average efficiency and CO₂ by county, town, local authority |
| `v_construction_age_analysis` | `fct_certificates` + `dim_properties` | Average efficiency and consumption by decade and property type |
| `v_property_efficiency_gaps` | `fct_certificates` + `dim_properties` | Properties where `efficiency_gap > 30` — highest upgrade potential |
| `v_retrofit_priority` | `fct_certificates` + `dim_properties` + `dim_locations` | Composite retrofit priority score 0–100 per property type × age band × local authority |

#### `v_retrofit_priority` — The Composite Score

This model answers "which types of properties, in which areas, are the most worth retrofitting?" using a weighted composite of three components:

| Component | Weight | Formula | Intuition |
|---|---|---|---|
| Current inefficiency | 35% | `(100 - energy_efficiency_current) / max_inefficiency` | Worse-rated properties score higher |
| Efficiency gap | 40% | `(potential - current) / max_gap` | How many SAP points can actually be gained? |
| CO₂ saving potential | 25% | `(co2_current - co2_potential) / max_co2_saving` | Climate impact of the retrofit |

All three components are normalized 0-1 against the global maximum (across all 29M certificates), then combined:

```
retrofit_priority_score = (0.35 × inefficiency_norm + 0.40 × gap_norm + 0.25 × co2_norm) × 100
```

A score of **100** = the most impactful retrofit candidate in the dataset. A score of **0** = already at maximum efficiency potential.

The model aggregates to `property_type × construction_age_band × county × local_authority`, so it can power both the heatmap matrix and the geographic treemap in the EDA suite.

The efficiency gap view is particularly useful for policy modeling:
```sql
select
    certificate_id, uprn, property_type,
    energy_efficiency_current,
    energy_efficiency_potential,
    (energy_efficiency_potential - energy_efficiency_current)      as efficiency_gap,
    (co2_current - co2_potential)                                  as co2_reduction_potential_tonnes
from fct_certificates f
join dim_properties p on f.property_id = p.property_id
where efficiency_gap > 30   -- Only high-impact properties
order by efficiency_gap desc
```

---

## Part 4b: The EDA Dashboard (`eda_uk_energy.py`)

`eda_uk_energy.py` reads from the star schema and produces a single interactive HTML file (`reports/dashboard.html`). It is organized as 12 chart builders, each returning a `go.Figure`, which are assembled into one self-contained page with Plotly.js loaded once from CDN.

### Why a single HTML file?

Each chart is embedded as a `<div>` snippet via `fig.to_html(full_html=False, include_plotlyjs=False)`. This means you can open the dashboard in any browser — no server, no network required — and share it as a single attachment.

### Design choices for large datasets

With 29.2M rows, naive scatter plots produce an unreadable ink-blot. The dashboard uses:

| Problem | Solution |
|---|---|
| 100k+ overlapping points | Box plots (CO₂ by property type) and 2D density contours (efficiency vs CO₂) |
| Large GeoJSON (18 MB) embedded in HTML | Ramer-Douglas-Peucker simplification reduces to 2.6 MB (86% reduction) before embedding |
| Choropleth ID-matching failures in Plotly 6 | Districts drawn as individual filled `go.Scattergeo` polygons with explicit coordinates |
| Column names from raw CSVs are UPPERCASE | `_pl()` helper lowercases all DuckDB result columns before handing to Polars |

### Dashboard sections

| Section | Data source | Key finding |
|---|---|---|
| 1. Rating Distribution | `fct_certificates` | 58%+ rated D or below — majority miss the 2035 Band C target |
| 2. County Efficiency | `v_regional_energy_performance` | 20-point SAP gap between worst rural and best urban counties |
| 3. Efficiency by Decade | `fct_certificates` + `dim_properties` | Pre-1900 homes: SAP 53 → SAP 75 potential (+22 pts, largest gap of any era) |
| 4. CO₂ by Property Type | `fct_certificates` + `dim_properties` | Detached houses emit 5× more CO₂ than a typical flat |
| 5. Efficiency vs CO₂ Density | `fct_certificates` + `dim_properties` | D/E band (60–70 SAP / 2–3 t CO₂) is the densest cluster in the dataset |
| 6. Retrofit Priority Matrix | `v_retrofit_priority` | Pre-1900 detached and semi-detached homes score highest priority |
| 7. Local Authority Treemap | `v_retrofit_priority` | Box size = CO₂ saving potential, colour = avg SAP — largest red boxes = biggest wins |
| 8. Postcode Area Heatmap | `fct_certificates` + `dim_locations` | Rural areas (TR, PL, EX) skew E–G; urban (E, N, SW) skew C–D |
| 9. Geographic EPC Map | `v_retrofit_priority` + martinjc GeoJSON | 362 UK local authority districts coloured by avg SAP score |
| 10. Fuel Type Impact | `stg_epc__domestic` | 31-point SAP gap between mains gas (66.3) and solid fuel (35.1) |
| 11. EPC Score Trend 2008–2024 | `fct_certificates` | +7.2 SAP points in 16 years — improvement rate needs to triple for 2035 target |
| 12. Annual Energy Cost | `fct_certificates` + `dim_properties` | Detached houses spend £1,094/yr; flats spend £622/yr — a £472 structural gap |

### The Retrofit Priority Score formula

`v_retrofit_priority` calculates a composite score per `property_type × construction_age_band × county × local_authority`:

```
retrofit_priority_score =
    0.35 × (current_inefficiency / global_max_inefficiency)    -- how bad is it today?
  + 0.40 × (efficiency_gap / global_max_gap)                   -- how much can it improve?
  + 0.25 × (co2_saving / global_max_co2_saving)                -- what is the climate impact?
  × 100
```

All three components are normalized against the global maximum across all 29.2M certificates, ensuring the score is comparable across any property type or location.

---

## Part 5: Running the Pipeline

### Full Execution

```bash
# Activate the Python environment
source dbt-env/bin/activate

# Step 1: Load raw data (run once, or when new CSVs are received)
python bulk_load_epc.py

# Step 2: Build the transformation layer
cd ducklake_energy_uk
dbt deps          # Download packages (dbt_utils)
dbt run           # Execute all models in DAG order
dbt test          # Run all data quality tests

# Step 3: Generate reports
cd ..
python eda_uk_energy.py
```

### Targeting a Single Model

```bash
# Run one model only (useful during development)
dbt run --select stg_epc__domestic
dbt run --select dim_properties

# Run a model and everything downstream of it
dbt run --select stg_epc__domestic+

# Run a model and everything upstream of it
dbt run --select +fct_certificates

# Run tests for one model only
dbt test --select stg_epc__domestic
```

### Dev vs Prod Environments

dbt profiles (`~/.dbt/profiles.yml`) define two targets:

```yaml
ducklake_energy_uk:
  outputs:
    dev:
      type: duckdb
      path: dev.duckdb    # Used during development (default)
      threads: 1
    prod:
      type: duckdb
      path: prod.duckdb   # Used for full production runs
      threads: 4          # 4 parallel threads for faster builds
  target: dev
```

Switch targets with `dbt run --target prod`. This separates your exploratory dev work from production data.

### Debugging Tips

```bash
# Compile SQL without executing (see the final SQL dbt generates)
dbt compile --select dim_properties
# Output goes to target/compiled/ducklake_energy_uk/models/...

# View dbt logs
cat logs/dbt.log

# Check what models exist and their status
dbt ls

# Wipe compiled outputs and start fresh
dbt clean
```

---

## Part 6: Key Concepts Summary

| Concept | What it is | Where used in this project |
|---|---|---|
| **ELT** | Load raw first, transform in-database | `bulk_load_epc.py` loads raw, dbt transforms |
| **Medallion Architecture** | Bronze/Silver/Gold layers | raw → stg → marts → analytics |
| **Materialization** | How dbt stores a model (view/table) | tables for dims/facts, views for analytics |
| **DAG** | Dependency graph of all models | Built automatically from `ref()` calls |
| **`ref()`** | How models declare dependencies on each other | Used in every mart and analytics model |
| **`source()`** | How dbt refers to externally-loaded raw tables | Points to `raw.epc_domestic` |
| **Window Function** | SQL for ranking/aggregating within groups without collapsing rows | `ROW_NUMBER() OVER (PARTITION BY uprn...)` in `dim_properties` |
| **Surrogate Key** | System-generated deterministic ID via MD5 hash | Every `_id` column in dims and facts |
| **Natural Key** | Real-world ID from the source (UPRN, postcode) | Retained alongside surrogate keys |
| **Normalization** | Breaking a flat table into related dimension + fact tables | Raw 100-column flat → 5-table star schema |
| **Star Schema** | Central fact table surrounded by dimension tables | `fct_certificates` + `dim_properties` + `dim_locations` |
| **Jinja** | Templating language embedded in dbt SQL | `{{ ref() }}`, `{{ source() }}`, `{{ config() }}`, `{{ dbt_utils.* }}` |
| **dbt_utils** | Package of extra macros and generic tests | `generate_surrogate_key`, `accepted_range` |
| **Generic Tests** | YAML-defined data assertions | `unique`, `not_null`, `accepted_values` in `.yml` schema files |
| **Singular Tests** | Custom SQL assertions in `tests/` | Any complex validation not expressible in YAML |
| **Apache Arrow** | Zero-copy in-memory columnar format shared between DuckDB and Polars | `.pl()` in `eda_uk_energy.py` |
| **Columnar Storage** | Data stored column-by-column (fast for aggregations) | DuckDB's internal storage format |
| **Vectorized Execution** | Operations applied to batches of values using CPU SIMD | DuckDB's query engine |

---

## EPC Domain Terms

Quick reference for energy and property terms used across this project's models and dashboards. See `README.md → Key Terminology` for the full plain-English glossary.

| Term | Meaning | Where it appears |
|---|---|---|
| **SAP Score** | Standard Assessment Procedure — energy efficiency score 1–100 | `energy_efficiency_current`, `energy_efficiency_potential` columns |
| **EPC Band (A–G)** | Letter grade derived from SAP score; Band C (SAP 69+) is the 2035 government target | `energy_rating_current`, `energy_rating_potential` |
| **UPRN** | Unique Property Reference Number — permanent government building ID | Primary natural key; deduplication key in `dim_properties` |
| **LAD** | Local Authority District — one of 362 administrative areas in England & Wales | `local_authority` column; geography level in `v_regional_energy_performance` |
| **Efficiency Gap** | `energy_efficiency_potential − energy_efficiency_current` — SAP points recoverable via retrofits | Computed in `fct_property_energy_performance`; filters in `v_property_efficiency_gaps` |
| **CO₂ (tonnes/yr)** | Estimated annual carbon dioxide emissions for the property | `co2_emissions_current_tonnes_per_year`, `co2_emissions_potential_tonnes_per_year` |
| **Retrofit Priority Score** | Composite 0–100 score = 35% inefficiency + 40% efficiency gap + 25% CO₂ saving, all normalised | `v_retrofit_priority` |
| **Median / Q1 / Q3 / IQR** | Statistical summary of distributions used in box-plot charts (see README Glossary for definitions) | Dashboard sections 4, 5 |

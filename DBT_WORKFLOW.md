# dbt Workflow Guide: Naturalization & Normalization

This guide provides a step-by-step breakdown of the terminal commands and transformations used to scale the UK Energy EPC dataset from raw ingestion to a high-performance Star Schema.

---

## 🛠️ Step 0: Initializing the Project
We started by creating the dbt environment and initializing the project.

```bash
# Verify dbt is installed
dbt --version

# Initialize the project (we named ours ducklake_energy_uk)
dbt init ducklake_energy_uk
cd ducklake_energy_uk
```

---

## 🔗 Step 1: Connecting to DuckDB
dbt requires a `profiles.yml` (usually in `~/.dbt/`) to know where your database lives. Our configuration uses the local `dev.duckdb` file.

```yaml
# profiles.yml snippet
ducklake_energy_uk:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: dev.duckdb
      schema: main
```

**Verification**:
```bash
dbt debug  # Ensure the connection to the .duckdb file is successful
```

---

## 🧼 Step 2: Naturalization (The Staging Phase)
"Naturalization" is the process of taking the messy, raw `VARCHAR` data from the bulk load and casting it into its natural types (Date, Decimal, Integer).

1. **Install Dependencies**: We used `dbt-utils` for advanced SQL logic.
   ```bash
   dbt deps
   ```

2. **Run Staging**: 
   ```bash
   dbt run --select staging
   ```
   *Terminal Output: OK created table main_staging.stg_epc__domestic...*

---

## 🏗️ Step 3: Normalization (The Marts Phase)
"Normalization" moves the data from a single flat table into a **Star Schema** to reduce redundancy and improve join performance.

1. **The Dimension Layer**: Creating `dim_properties` (latest state) and `dim_locations` (geographic unique keys).
2. **The Fact Layer**: Creating `fct_certificates` (the 29.2M row record of calculations).

**Execute Normalization**:
```bash
dbt run --select marts.energy
```
*Terminal Output: OK created table main_marts.fct_certificates (29,214,082 rows)...*

---

## 📈 Step 4: Analytical Views
Building summarized views for fast interactive reporting and the Polars EDA suite.

```bash
dbt run --select marts.analytics
```

---

## 📜 Step 5: Documentation & Lineage
dbt automatically tracks the relationship between your models. You can view the visual flowchart of the 29.2M row journey.

```bash
# Generate the documentation site
dbt docs generate

# Launch the locally hosted documentation portal
dbt docs serve
```

---

## 🎯 Summary of Achievement
- **Bulk Load**: 29.2M rows ingested as `VARCHAR` (100% success).
- **Naturalized**: Correct types applied via staging models.
- **Normalized**: Star Schema built for high-performance sub-second joins.
- **Analyzed**: Advanced EDA generated in < 60s using Polars + DuckDB.

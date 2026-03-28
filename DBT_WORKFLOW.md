# dbt Masterclass: Scaling UK Energy Data (29.2M Rows)

🚀 **A Professional Engineering Journey from Raw Ingestion to a Normalized BI Layer.**

This documentation acts as a learning tool for anyone mastering **dbt (Data Build Tool)** and **Modern Data Warehousing**. It explains the core concepts, the industry-standard architecture, and the exact steps we took to handle 29.2 million UK Energy records.

---

## 🏛️ Core dbt Concepts: The Building Blocks

In the IT industry, dbt is used to manage the **"T" (Transformation)** in ELT (Extract, Load, Transform). Here is how we structured this project:

### 1. Models
The heart of any dbt project. A **Model** is a single `.sql` file that defines a table or view.
- **Project Context**: We have models for Staging, Marts, and Analytics.

### 2. Staging (The "Naturalization" Layer)
Staging models take raw data and prepare it for downstream use. This is where we perform **"Naturalization"**:
- **Casting**: Converting strings into Dates, Decimals, and Integers.
- **Renaming**: Applying a consistent naming convention (camelCase, snake_case).
- **Cleansing**: Removing "INVALID!" ratings or handling nulls.
- **Standardization**: Consolidating 30+ Construction Age Bands and Fuel Types into 8-10 clean categories.

### 3. Marts (The "Normalization" / Gold Layer)
Marts are where the business logic lives. We follow the **Star Schema** architecture:
- **Dimensions (`dim_`)**: Descriptive tables like `dim_properties` (building characteristics) or `dim_locations` (geographic unique keys).
- **Facts (`fct_`)**: Quantitative tables like `fct_certificates` (detailed assessments) or `fct_property_aggregations` (summarized results).

### 4. Seeds
Static CSV files (e.g., country codes, tax rates) that you want to version control and load into your database.
- **Industry Practice**: Use seeds only for small, slowly-changing lookup data (< 1MB).

### 5. Macros
Reusable pieces of code (written in Jinja) that help keep your SQL **DRY (Don't Repeat Yourself)**.
- **Project Context**: We use the `dbt_utils.generate_surrogate_key` macro for reliable hashing.

### 6. Analyses
SQL files that are compiled but not materialized as tables. Perfect for one-off exploratory questions that don't need to be in the final warehouse.

### 7. Tests (Data Quality as Code)
Automatic validations that run against your data. If a test fails, the pipeline fails.
- **Industry Practice**: We implement `not_null`, `unique`, and `accepted_values` (A-G ratings) to ensure 29.2M rows are trusted.

---

## 📈 The Journey: How We Built It

We followed the **Medallion Architecture** (Bronze → Silver → Gold) to handle the 50GB EPC dataset.

### Step 1: Raw Ingestion (Bronze Layer)
We used a high-performance Python/DuckDB script (`bulk_load_epc.py`) to move 29.2 million records into a `raw.epc_domestic` table in ~10 seconds.
- **Strategy**: Load everything as `VARCHAR` first to avoid truncation or casting errors during ingestion.

### Step 2: Staging & Standardization (Silver Layer)
We moved from "Messy" to "Clean".
- **Did we do this?** Yes. We implemented **`stg_epc__domestic.sql`**.
- **The How**: We used large `CASE` statements to standardize construction age bands and fuel types.
  ```sql
  -- Standardization Example
  case when lower(main_fuel) like '%gas%' then 'Mains Gas' ... end as main_fuel
  ```

### Step 3: Normalization & Aggregation (Gold Layer)
We move from "Flat" to "Structured".
- **Normalization**: Breaking the flat table into `dim_properties` and `fct_certificates`.
- **Aggregation**: Running the **`fct_property_aggregations.sql`** model to roll up 29.2M rows by:
    - **Property Type** (House, Flat, Maisonette)
    - **Tenure** (Owner-occupied, Rental, Social)
    - **Region** (County)
    - **Year** (Inspection Date)

---

## 🏗️ The Star Schema: Why This Matters
In the IT industry, the **Star Schema** is the gold standard for BI and Analytics.
- **Fact Tables**: Contain the "What happened" (The 29.2M inspections).
- **Dimension Tables**: Contain the "To whom/where" (The building specs).
- **Result**: Joining a 29M row Fact table with a 5k row Location table is significantly faster than querying one massive 100-column table.

---

## 🏁 How to Execute the Masterclass

### 1. Install Dependencies
```bash
dbt deps  # Installs dbt-utils for surrogate keys
```

### 2. The Build Process
```bash
dbt run   # Standard Building (Full-Refresh if needed)
dbt test  # Data Quality check (A-G, Not-Null)
```

### 3. Documentation Portal
```bash
dbt docs generate
dbt docs serve    # View the visual data lineage
```

---

## 🏆 Final Summary of Step 1 Achievement
- [x] **Ingested**: 29.2M rows in < 15 seconds.
- [x] **Naturalized**: Correct types (Date, Decimal) applied.
- [x] **Standardized**: Age Bands and Fuel Types cleaned for 100% data consistency.
- [x] **Normalized**: High-performance Star Schema built (Dim/Fct).
- [x] **Aggregated**: Business-ready mart for regional reporting.
- [x] **Tested**: `accepted_values` (A-G) and `not_null` constraints enforced.

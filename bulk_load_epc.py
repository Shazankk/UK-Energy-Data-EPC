import duckdb
import os

# Configuration
DB_PATH = 'ducklake_energy_uk/dev.duckdb'
BULK_DATA_PATH = './all-domestic-certificates/domestic-*/certificates.csv'
RAW_TABLE = 'raw.epc_domestic'

def bulk_load():
    """Perform high-performance bulk load from local CSVs into DuckDB."""
    print(f"Connecting to {DB_PATH}...")
    con = duckdb.connect(DB_PATH)
    
    print("Setting up raw schema...")
    con.execute("CREATE SCHEMA IF NOT EXISTS raw;")
    
    # Industry standard: Land the data as VARCHAR to avoid conversion errors during ingestion.
    # We use union_by_name=True in case columns vary slightly across regions.
    load_sql = f"""
    CREATE OR REPLACE TABLE {RAW_TABLE} AS 
    SELECT * FROM read_csv_auto(
        '{BULK_DATA_PATH}', 
        all_varchar=True, 
        union_by_name=True,
        filename=True
    );
    """
    
    print(f"Loading data from {BULK_DATA_PATH}...")
    print("This may take a minute for millions of rows...")
    
    try:
        con.execute(load_sql)
        
        # Verify the load
        count = con.execute(f"SELECT count(*) FROM {RAW_TABLE}").fetchone()[0]
        print(f"Successfully loaded {count:,} rows into {RAW_TABLE}.")
        
    except Exception as e:
        print(f"Error during bulk load: {e}")
    finally:
        con.close()

if __name__ == "__main__":
    bulk_load()

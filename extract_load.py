import urllib.request
from urllib.parse import urlencode
import os
import duckdb
from dotenv import load_dotenv
import io
import pandas as pd

# Load environment variables
load_dotenv()

# Configuration
AUTH_TOKEN = os.getenv('AUTHENTICATION_TOKEN')
DB_PATH = 'ducklake_energy_uk/dev.duckdb'
QUERY_SIZE = 5000
BASE_URL = 'https://epc.opendatacommunities.org/api/v1/domestic/search'
RAW_TABLE = 'raw.epc_domestic'

# Set up headers
headers = {
    'Accept': 'text/csv',
    'Authorization': f'Basic {AUTH_TOKEN}'
}

def setup_db(con):
    """Ensure the raw schema and table exist."""
    con.execute("CREATE SCHEMA IF NOT EXISTS raw;")
    # We don't pre-create the table because headers may vary slightly or we want duckdb to infer the types initially
    # but we'll use read_csv_auto

def fetch_and_load():
    con = duckdb.connect(DB_PATH)
    setup_db(con)
    
    first_request = True
    search_after = None
    query_params = {'size': QUERY_SIZE}
    
    print(f"Starting extraction to {DB_PATH}...")
    
    iteration = 0
    while search_after is not None or first_request:
        if not first_request:
            query_params["search-after"] = search_after
            
        encoded_params = urlencode(query_params)
        full_url = f"{BASE_URL}?{encoded_params}"
        
        try:
            req = urllib.request.Request(full_url, headers=headers)
            with urllib.request.urlopen(req) as response:
                body = response.read().decode('utf-8')
                search_after = response.getheader('X-Next-Search-After')
                
                if not body.strip():
                    break
                
                # Load all columns as strings to prevent schema inference errors during raw load
                df = pd.read_csv(io.StringIO(body), dtype=str)
                
                if first_request:
                    # Create the table on first chunk
                    con.execute(f"CREATE OR REPLACE TABLE {RAW_TABLE} AS SELECT * FROM df")
                    print(f"Created table {RAW_TABLE}")
                else:
                    # Append on subsequent chunks
                    con.execute(f"INSERT INTO {RAW_TABLE} SELECT * FROM df")
                
                iteration += 1
                print(f"Interation {iteration}: Loaded {len(df)} rows. Total progress: {iteration * QUERY_SIZE} (approx)")
                
                first_request = False
                
        except Exception as e:
            print(f"Error during extraction at iteration {iteration}: {e}")
            break
            
    con.close()
    print("Extraction complete.")

if __name__ == "__main__":
    fetch_and_load()

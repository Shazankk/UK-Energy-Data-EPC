import duckdb

db_path = "ducklake_energy_uk/dev.duckdb"
con = duckdb.connect(db_path)

print("--- Construction Age Bands ---")
res = con.execute("SELECT construction_age_band, COUNT(*) as count FROM main.stg_epc__domestic GROUP BY 1 ORDER BY 2 DESC LIMIT 30;").fetchall()
for row in res:
    print(f"{row[0]}: {row[1]}")

print("\n--- Fuel Types ---")
res = con.execute("SELECT main_fuel, COUNT(*) as count FROM main.stg_epc__domestic GROUP BY 1 ORDER BY 2 DESC LIMIT 30;").fetchall()
for row in res:
    print(f"{row[0]}: {row[1]}")

print("\n--- Current Energy Ratings ---")
res = con.execute("SELECT energy_rating_current, COUNT(*) FROM main.stg_epc__domestic GROUP BY 1 ORDER BY 1;").fetchall()
for row in res:
    print(f"{row[0]}: {row[1]}")

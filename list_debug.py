import duckdb

db_path = "ducklake_energy_uk/dev.duckdb"
con = duckdb.connect(db_path)

print("--- Schemas ---")
res = con.execute("SELECT schema_name FROM information_schema.schemata;").fetchall()
for row in res:
    print(row[0])

print("\n--- Tables ---")
res = con.execute("SELECT table_schema, table_name FROM information_schema.tables;").fetchall()
for row in res:
    print(f"{row[0]}.{row[1]}")

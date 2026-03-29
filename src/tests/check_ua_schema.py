import duckdb
import pathlib

ua_files = sorted(pathlib.Path('downloads/urban_atlas_parquets').glob('*.parquet'))
for f in ua_files:
    schema = duckdb.sql(f"SELECT * FROM read_parquet('{f}') LIMIT 1").description
    count = duckdb.sql(f"SELECT COUNT(*) FROM read_parquet('{f}')").fetchone()[0]
    print(f"{f.name}: {len(schema)} columns, {count} rows")
    for col in schema:
        print(f"  - {col[0]}: {col[1]}")
    print()

#!/usr/bin/env python
"""Verify prepared data was created correctly."""

import duckdb
from pathlib import Path

db_path = Path('prepared_stream_data/stream_index.duckdb')
conn = duckdb.connect(str(db_path), read_only=True)

# List tables
try:
    tables = conn.execute('SELECT table_name FROM information_schema.tables WHERE table_type = "BASE TABLE"').fetchall()
    print('DuckDB Tables:')
    for table in tables:
        count = conn.execute(f'SELECT COUNT(*) FROM {table[0]}').fetchone()[0]
        print(f'  ✓ {table[0]}: {count:,} rows')
except Exception as e:
    print(f'Error listing tables: {e}')

# Show Urban Atlas by year
try:
    ua_sample = conn.execute('SELECT year, COUNT(*) as count FROM urban_atlas GROUP BY year ORDER BY year').fetchall()
    print('\nUrban Atlas data by year:')
    for year, count in ua_sample:
        print(f'  {year}: {count:,} features')
except Exception as e:
    print(f'Urban Atlas query failed: {e}')

conn.close()
print("\n✓ Data preparation verification complete!")

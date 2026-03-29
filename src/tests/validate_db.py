import duckdb

conn = duckdb.connect('prepared_stream_data/stream_index.duckdb')

print('Tables in database:')
tables = [t[0] for t in conn.sql('SELECT table_name FROM information_schema.tables WHERE table_type = "BASE TABLE"').fetchall()]
print(tables)
print()

print('Trees statistics:')
trees_stats = conn.sql('SELECT COUNT(*) as total FROM trees').fetchall()
print(f"  Total records: {trees_stats[0][0]}")

treescoords = conn.sql('SELECT COUNT(DISTINCT latitude) as lat_values, COUNT(DISTINCT longitude) as lon_values FROM trees').fetchall()
print(f"  Distinct lat values: {treescoords[0][0]}, Distinct lon values: {treescoords[0][1]}")

print()
print('Urban Atlas statistics:')
ua_stats = conn.sql('SELECT COUNT(*) as total, COUNT(DISTINCT year) as years FROM urban_atlas').fetchall()
print(f"  Total records: {ua_stats[0][0]}, Years: {ua_stats[0][1]}")

ua_years = conn.sql('SELECT DISTINCT year FROM urban_atlas ORDER BY year').fetchall()
print(f"  Years in database: {[y[0] for y in ua_years]}")

print()
print('LST metadata statistics:')
lst_stats = conn.sql('SELECT COUNT(*) as total FROM information_schema.tables WHERE table_name = "lst_metadata"').fetchall()
print(f"  LST metadata table exists: {lst_stats[0][0] > 0}")

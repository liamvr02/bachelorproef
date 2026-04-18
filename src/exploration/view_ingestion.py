import duckdb
import sqlite3
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# =========================
# CONFIG
# =========================
DATA_DIR = Path(__file__).parent.parent / "prepared_stream_data"
OUTPUT_JSON = Path(__file__).parent / "db_overview.json"

CHUNK_SIZE = 100_000
MAX_UNIQUE_TRACK = 10_000   # stop tracking uniques after this
MAX_UNIQUE_SAVE = 100      # save at most this many unique values

CONTINUOUS_TYPES = {"FLOAT", "DOUBLE", "REAL", "DECIMAL"}

# =========================
# HELPERS
# =========================

def is_continuous(dtype: str) -> bool:
    if dtype is None:
        return False
    return any(t in dtype.upper() for t in CONTINUOUS_TYPES)


def get_duckdb_tables(conn):
    return [row[0] for row in conn.execute("SHOW TABLES").fetchall()]


def get_sqlite_tables(conn):
    return [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
    ]


def get_duckdb_columns(conn, table):
    res = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    return [(r[1], r[2]) for r in res]  # (name, type)


def get_sqlite_columns(conn, table):
    res = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    return [(r[1], r[2]) for r in res]


def count_rows(conn, table, is_duck=True):
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    except Exception:
        return None


def stream_table(conn, table, columns, total_rows):
    offset = 0

    with tqdm(
        total=total_rows,
        desc=f"Scanning {table}",
        unit="rows",
        leave=False,
    ) as pbar:

        while True:
            query = f"""
                SELECT {', '.join(columns)}
                FROM {table}
                LIMIT {CHUNK_SIZE} OFFSET {offset}
            """
            rows = conn.execute(query).fetchall()

            if not rows:
                break

            yield rows

            offset += len(rows)
            pbar.update(len(rows))


# =========================
# CORE ANALYSIS
# =========================

def analyze_table(conn, table, columns, total_rows):
    col_names = [c[0] for c in columns]

    stats = {
        col: {
            "dtype": dtype,
            "non_null_count": 0,
            "unique_values": set(),
            "unique_count_estimate": 0,
            "tracking_uniques": not is_continuous(dtype),
        }
        for col, dtype in columns
    }

    for chunk in stream_table(conn, table, col_names, total_rows):
        for row in chunk:
            for col, value in zip(col_names, row):
                col_stat = stats[col]

                if value is not None:
                    col_stat["non_null_count"] += 1

                    if col_stat["tracking_uniques"]:
                        uniques = col_stat["unique_values"]
                        if len(uniques) < MAX_UNIQUE_TRACK:
                            uniques.add(value)
                        else:
                            col_stat["tracking_uniques"] = False

    # finalize stats
    for col, col_stat in stats.items():
        uniques = col_stat["unique_values"]

        col_stat["unique_count_estimate"] = len(uniques)

        # only save a sample
        col_stat["unique_sample"] = list(uniques)[:MAX_UNIQUE_SAVE]

        del col_stat["unique_values"]

    return stats


def analyze_duckdb(path):
    tqdm.write(f"\nOpening DuckDB: {path}")
    conn = duckdb.connect(str(path), read_only=True)

    db_result = {}

    tables = get_duckdb_tables(conn)

    for table in tqdm(tables, desc=f"{path.name} tables"):
        try:
            columns = get_duckdb_columns(conn, table)
            total_rows = count_rows(conn, table)

            tqdm.write(f"Table: {table} ({total_rows} rows)")

            col_stats = analyze_table(conn, table, columns, total_rows)

            db_result[table] = {
                "row_count": total_rows,
                "columns": col_stats,
            }

        except Exception as e:
            tqdm.write(f"Error in table {table}: {e}")

    conn.close()
    return db_result


def analyze_sqlite(path):
    tqdm.write(f"\nOpening SQLite: {path}")
    conn = sqlite3.connect(str(path))

    db_result = {}

    tables = get_sqlite_tables(conn)

    for table in tqdm(tables, desc=f"{path.name} tables"):
        try:
            columns = get_sqlite_columns(conn, table)
            total_rows = count_rows(conn, table, is_duck=False)

            tqdm.write(f"Table: {table} ({total_rows} rows)")

            col_stats = analyze_table(conn, table, columns, total_rows)

            db_result[table] = {
                "row_count": total_rows,
                "columns": col_stats,
            }

        except Exception as e:
            tqdm.write(f"Error in table {table}: {e}")

    conn.close()
    return db_result


# =========================
# MAIN
# =========================

def main():
    all_results = {}

    db_files = list(DATA_DIR.glob("*.duckdb")) + list(DATA_DIR.glob("*.db"))

    for db_path in tqdm(db_files, desc="Databases"):
        try:
            if db_path.suffix == ".duckdb":
                result = analyze_duckdb(db_path)
            else:
                result = analyze_sqlite(db_path)

            all_results[db_path.name] = result

        except Exception as e:
            tqdm.write(f"Failed to process {db_path}: {e}")

    # save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    # =========================
    # SHORT OVERVIEW
    # =========================
    tqdm.write("\n=== OVERVIEW ===")

    for db_name, db in all_results.items():
        tqdm.write(f"\n{db_name}: {len(db)} tables")

        for table, info in db.items():
            tqdm.write(
                f"  - {table}: {info['row_count']} rows, {len(info['columns'])} columns"
            )


if __name__ == "__main__":
    main()
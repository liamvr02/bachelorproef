"""
ingest/catalog.py
=================
Write (or update) catalog.duckdb with dataset metadata and LST histograms.

Tables
------
dataset_metadata      one row per logical dataset
partition_statistics  one row per (dataset, partition_key, tile_id)
                      histogram_json stores fixed-bin temperature counts
histogram_config      shared bin edges read once by the streaming layer

All aggregation runs inside DuckDB via ATTACH — no temperature values
or large result sets cross into Python.  The 1.2 M-row partition_statistics
table is written with a single bulk INSERT INTO ... SELECT.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from config import (
    DATASET_REGISTRY,
    HIST_EDGES,
    LAT_EDGES,
    LON_EDGES,
    _compute_timestamp_edges,
)
from db import open_duckdb

log = logging.getLogger("ingest.catalog")


def write_catalog(output: Path, processed: list[str]) -> None:
    """Write catalog.duckdb with dataset metadata and LST partition histograms."""
    db_path = output / "catalog.duckdb"
    
    # Delete stale DB to avoid page accumulation across re-runs
    db_path.unlink(missing_ok=True)
    
    conn    = open_duckdb(db_path)

    conn.execute(f"SET temp_directory = '{db_path.parent.as_posix()}'")
    conn.execute("SET memory_limit = '8GB'")
    conn.execute("SET threads = 2")
    conn.execute("SET preserve_insertion_order = false")

    conn.execute("""
        CREATE TABLE dataset_metadata (
            dataset_id              VARCHAR PRIMARY KEY,
            description             VARCHAR,
            feature_columns         VARCHAR[],
            lookup_method           VARCHAR,
            temporal_behavior       VARCHAR,
            partition_column        VARCHAR,
            source_resolution_m     DOUBLE,
            is_driving              BOOLEAN,
            value_column            VARCHAR,
            db_file                 VARCHAR,
            db_table                VARCHAR,
            store                   VARCHAR,
            registered_at           VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE partition_statistics (
            dataset_id                       VARCHAR,
            partition_key                    VARCHAR,
            tile_id                          VARCHAR,
            row_count                        BIGINT,
            value_min                        DOUBLE,
            value_max                        DOUBLE,
            value_mean                       DOUBLE,
            histogram_counts                 BIGINT[],
            timestamp_histogram_counts       BIGINT[],
            longitude_histogram_counts       BIGINT[],
            latitude_histogram_counts        BIGINT[],
            PRIMARY KEY (dataset_id, partition_key, tile_id)
        )
    """)
    conn.execute("""
        CREATE TABLE histogram_config (
            dataset_id      VARCHAR PRIMARY KEY,
            bin_edges       VARCHAR[],
            n_bins          INTEGER
        )
    """)
    conn.execute(
        "INSERT INTO histogram_config VALUES (?, ?, ?)",
        ["lst", [str(v) for v in HIST_EDGES], len(HIST_EDGES) - 1],
    )

    now = datetime.now(timezone.utc).isoformat()
    for dataset_id in tqdm(processed, desc="Registering datasets", unit="dataset"):
        spec = DATASET_REGISTRY[dataset_id]
        
        # Handle new value_columns (list) vs old value_column (single string)
        # For backward compatibility, if value_columns exists, convert to JSON string
        value_col = spec.get("value_columns")
        if isinstance(value_col, list):
            value_col = json.dumps(value_col)
        else:
            value_col = spec.get("value_column")
        
        conn.execute(
            "INSERT OR REPLACE INTO dataset_metadata VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                dataset_id, spec["description"], spec["feature_columns"],
                spec["lookup_method"], spec["temporal_behavior"],
                spec.get("partition_column"), spec.get("source_resolution_m"),
                spec["is_driving"], value_col,
                spec["db_file"], spec["table"], spec["store"], now,
            ],
        )

    # --- LST partition histograms ---
    # Everything runs inside DuckDB — no Python loops, no fetchall, no executemany.
    if "lst" in processed:
        lst_db = output / "lst.duckdb"
        if lst_db.exists():
            _write_lst_histograms(conn, lst_db)

    conn.close()
    tqdm.write(f"Catalog written to {db_path}")



def _write_lst_histograms(conn, lst_db: Path) -> None:
    """
    Compute and insert LST partition histograms, chunked by partition_key.

    Strategy
    --------
    The per-tile loop in the previous version issued one query per
    (partition_key, tile_id) pair — O(N_tiles) round-trips, each scanning
    the full table with a filter.  A single bulk query over all partitions
    at once hit OOM due to list_transform(range(0, 3600), ...) materialising
    huge lon/lat arrays for every tile simultaneously.

    This version chunks by partition_key (one month of data per iteration).
    Each chunk processes all tiles for that month in one query, keeping the
    working set small, while cutting round-trips from ~N_tiles to ~N_months.
    Lon/lat histograms use repeat()-based string building instead of list
    materialisation, eliminating the OOM trigger entirely.

    Progress: one tqdm tick per partition_key (≈ per month), so at most a
    few minutes between updates regardless of tile count.
    """
    # Compute bin parameters for all dimensions
    n_bins_temp     = len(HIST_EDGES) - 1
    bin_width_temp  = HIST_EDGES[1] - HIST_EDGES[0]
    edge_min_temp   = HIST_EDGES[0]

    timestamp_edges = _compute_timestamp_edges()
    n_bins_ts       = len(timestamp_edges) - 1

    n_bins_lon    = len(LON_EDGES) - 1
    bin_width_lon = LON_EDGES[1] - LON_EDGES[0]
    edge_min_lon  = LON_EDGES[0]

    n_bins_lat    = len(LAT_EDGES) - 1
    bin_width_lat = LAT_EDGES[1] - LAT_EDGES[0]
    edge_min_lat  = LAT_EDGES[0]

    conn.execute("DELETE FROM histogram_config")
    conn.execute("INSERT INTO histogram_config VALUES (?, ?, ?)",
                 ["temperature", [str(v) for v in HIST_EDGES], n_bins_temp])
    conn.execute("INSERT INTO histogram_config VALUES (?, ?, ?)",
                 ["timestamp", timestamp_edges, n_bins_ts])
    conn.execute("INSERT INTO histogram_config VALUES (?, ?, ?)",
                 ["longitude", [str(v) for v in LON_EDGES], n_bins_lon])
    conn.execute("INSERT INTO histogram_config VALUES (?, ?, ?)",
                 ["latitude", [str(v) for v in LAT_EDGES], n_bins_lat])

    lst_db_posix = lst_db.as_posix()
    tqdm.write("Computing LST partition histograms (temperature, timestamp, coordinates) ...")
    conn.execute(f"ATTACH '{lst_db_posix}' AS lst_db (READ_ONLY)")

    partition_keys = [
        r[0] for r in conn.execute(
            "SELECT DISTINCT partition_key FROM lst_db.lst ORDER BY partition_key"
        ).fetchall()
    ]
    tqdm.write(f"Processing {len(partition_keys)} partition_key(s) ...")

    for part_key in tqdm(partition_keys, desc="LST histograms", unit="month"):
        conn.execute(f"""
            INSERT INTO partition_statistics
            WITH
            part_stats AS (
                SELECT
                    partition_key,
                    tile_id,
                    COUNT(*)         AS row_count,
                    MIN(COALESCE(aster_lst, modis_lst, ndvi)) AS t_min,
                    MAX(COALESCE(aster_lst, modis_lst, ndvi)) AS t_max,
                    AVG(COALESCE(aster_lst, modis_lst, ndvi)) AS t_mean,
                    MIN(longitude)   AS lon_min,
                    MAX(longitude)   AS lon_max,
                    MIN(latitude)    AS lat_min,
                    MAX(latitude)    AS lat_max
                FROM lst_db.lst
                WHERE partition_key = '{part_key}'
                GROUP BY partition_key, tile_id
            ),
            temp_bins AS (
                SELECT
                    partition_key,
                    tile_id,
                    GREATEST(0, LEAST({n_bins_temp - 1},
                        FLOOR((COALESCE(aster_lst, modis_lst, ndvi) - ({edge_min_temp})) / {bin_width_temp})::INTEGER
                    )) AS bin_idx,
                    COUNT(*) AS bin_count
                FROM lst_db.lst
                WHERE partition_key = '{part_key}' AND COALESCE(aster_lst, modis_lst, ndvi) IS NOT NULL
                GROUP BY partition_key, tile_id, bin_idx
            ),
            ts_bins AS (
                SELECT
                    partition_key,
                    tile_id,
                    GREATEST(0, LEAST({n_bins_ts - 1},
                        CAST(
                            ((CAST(substring(timestamp, 1, 4) AS INTEGER) - 2000) * 4 +
                             ((CAST(substring(timestamp, 6, 2) AS INTEGER) - 1) / 3))
                            AS INTEGER
                        )
                    )) AS bin_idx,
                    COUNT(*) AS bin_count
                FROM lst_db.lst
                WHERE partition_key = '{part_key}'
                GROUP BY partition_key, tile_id, bin_idx
            ),
            coord_stats AS (
                SELECT
                    partition_key,
                    tile_id,
                    GREATEST(0, LEAST({n_bins_lon - 1},
                        FLOOR((lon_min - ({edge_min_lon})) / {bin_width_lon})::INTEGER
                    )) AS lon_bin_start,
                    GREATEST(0, LEAST({n_bins_lon - 1},
                        FLOOR((lon_max - ({edge_min_lon})) / {bin_width_lon})::INTEGER
                    )) AS lon_bin_end,
                    GREATEST(0, LEAST({n_bins_lat - 1},
                        FLOOR((lat_min - ({edge_min_lat})) / {bin_width_lat})::INTEGER
                    )) AS lat_bin_start,
                    GREATEST(0, LEAST({n_bins_lat - 1},
                        FLOOR((lat_max - ({edge_min_lat})) / {bin_width_lat})::INTEGER
                    )) AS lat_bin_end
                FROM part_stats
            ),
            temp_counts_agg AS (
                SELECT
                    ps.partition_key,
                    ps.tile_id,
                    array_agg(COALESCE(tb.bin_count, 0) ORDER BY r.idx) AS counts_array
                FROM (SELECT DISTINCT partition_key, tile_id FROM part_stats) ps
                CROSS JOIN (SELECT * FROM range(0, {n_bins_temp})) r(idx)
                LEFT JOIN temp_bins tb 
                    ON ps.partition_key = tb.partition_key 
                    AND ps.tile_id = tb.tile_id 
                    AND r.idx = tb.bin_idx
                GROUP BY ps.partition_key, ps.tile_id
            ),
            ts_counts_agg AS (
                SELECT
                    ps.partition_key,
                    ps.tile_id,
                    array_agg(COALESCE(tsb.bin_count, 0) ORDER BY r.idx) AS counts_array
                FROM (SELECT DISTINCT partition_key, tile_id FROM part_stats) ps
                CROSS JOIN (SELECT * FROM range(0, {n_bins_ts})) r(idx)
                LEFT JOIN ts_bins tsb 
                    ON ps.partition_key = tsb.partition_key 
                    AND ps.tile_id = tsb.tile_id 
                    AND r.idx = tsb.bin_idx
                GROUP BY ps.partition_key, ps.tile_id
            )
            SELECT
                'lst'                        AS dataset_id,
                ps.partition_key,
                ps.tile_id,
                ps.row_count,
                ps.t_min                     AS value_min,
                ps.t_max                     AS value_max,
                ps.t_mean                    AS value_mean,
                tca.counts_array             AS histogram_counts,
                tsca.counts_array            AS timestamp_histogram_counts,
                CAST(string_split(                              -- lon/lat as BIGINT[] built via repeat()
                    rtrim(concat(
                        CASE WHEN cs.lon_bin_start > 0 THEN repeat('0,', cs.lon_bin_start) ELSE '' END,
                        repeat('1,', cs.lon_bin_end - cs.lon_bin_start + 1),
                        CASE WHEN cs.lon_bin_end < {n_bins_lon - 1} 
                             THEN repeat('0,', {n_bins_lon - 1} - cs.lon_bin_end) 
                             ELSE '' END
                    ), ','), ',') AS BIGINT[]) AS longitude_histogram_counts,
                CAST(string_split(
                    rtrim(concat(
                        CASE WHEN cs.lat_bin_start > 0 THEN repeat('0,', cs.lat_bin_start) ELSE '' END,
                        repeat('1,', cs.lat_bin_end - cs.lat_bin_start + 1),
                        CASE WHEN cs.lat_bin_end < {n_bins_lat - 1}
                             THEN repeat('0,', {n_bins_lat - 1} - cs.lat_bin_end)
                             ELSE '' END
                    ), ','), ',') AS BIGINT[]) AS latitude_histogram_counts
            FROM part_stats ps
            JOIN temp_counts_agg tca ON ps.partition_key = tca.partition_key AND ps.tile_id = tca.tile_id
            JOIN ts_counts_agg tsca ON ps.partition_key = tsca.partition_key AND ps.tile_id = tsca.tile_id
            JOIN coord_stats cs ON ps.partition_key = cs.partition_key AND ps.tile_id = cs.tile_id
        """)
    
    # Single CHECKPOINT after all partitions (outside the loop)
    conn.execute("CHECKPOINT")

    conn.execute("DETACH lst_db")
    tqdm.write("Histogram catalog complete.")
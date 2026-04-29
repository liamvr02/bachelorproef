"""
ingest/catalog.py
=================
Write (or update) catalog.duckdb with dataset metadata and LST histograms.

Tables
------
dataset_metadata      one row per logical dataset
partition_statistics  one row per (dataset, partition_key, tile_id)
                      one BIGINT[] histogram column per entry in DIMENSION_CATALOG
histogram_config      one row per dimension — bin edges consumed by the stream layer

Extensibility
-------------
Adding a new dimension requires only:
  1. A new entry in config.DIMENSION_CATALOG (col name, edges, sql_alias).
  2. A corresponding SQL extraction expression in _DIMENSION_SQL below.
  3. A new BIGINT[] column in partition_statistics (added automatically by
     _build_partition_statistics_ddl()).

The histogram computation loop and catalog registration are fully data-driven;
no other code in this file needs touching.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from config import (
    DATASET_REGISTRY,
    DIMENSION_CATALOG,
    HIST_EDGES,
    _compute_timestamp_edges,
    get_dimension_edges,
)
from db import open_duckdb

log = logging.getLogger("ingest.catalog")


# ============================================================
# Dimension SQL extraction expressions
# ============================================================
# Maps dimension name → SQL expression that computes a 0-based integer bin
# index from columns available in the lst table.
#
# Conventions:
#   • Every expression must produce an integer in [0, n_bins-1].
#   • GREATEST/LEAST clamps out-of-range values.
#   • Expressions may reference any column in lst_db.lst.
#   • The special token {n_bins} is substituted at query-build time.
#
# To add a new dimension: add an entry here and a matching entry in
# config.DIMENSION_CATALOG.  The rest is automatic.
_DIMENSION_SQL: Dict[str, str] = {
    # Temperature: 2°C bins, floor division
    "temperature": (
        "GREATEST(0, LEAST({n_bins} - 1, "
        "  FLOOR((COALESCE(aster_lst, modis_lst, ndvi) - {edge_min}) / {bin_width})::INTEGER"
        "))"
    ),

    # Quarterly timestamp labels (string mapping via arithmetic on year/month)
    "timestamp": (
        "GREATEST(0, LEAST({n_bins} - 1, "
        "  CAST("
        "    ((CAST(substring(timestamp, 1, 4) AS INTEGER) - 2000) * 4 +"
        "    ((CAST(substring(timestamp, 6, 2) AS INTEGER) - 1) / 3))"
        "  AS INTEGER)"
        "))"
    ),

    # Year: bin = year - first_edge  (one bin per calendar year)
    "year": (
        "GREATEST(0, LEAST({n_bins} - 1, "
        "  CAST(year - {edge_min} AS INTEGER)"
        "))"
    ),

    # Month-of-year: bin = month - 1  (January = bin 0)
    "month_of_year": (
        "GREATEST(0, LEAST({n_bins} - 1, "
        "  CAST(month_of_year - 1 AS INTEGER)"
        "))"
    ),

    # Day-of-month: bin = day - 1  (1st = bin 0)
    "day_of_month": (
        "GREATEST(0, LEAST({n_bins} - 1, "
        "  CAST(day_of_month - 1 AS INTEGER)"
        "))"
    ),

    # Day-of-year: variable-width bins defined by DOY_BREAKPOINTS.
    # We use the CASE expression to map each day into the right breakpoint bin.
    # Edges are [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 367],
    # so bin k covers days [edges[k], edges[k+1]).
    # A simple floor search: find the largest edge_index s.t. edges[i] <= doy.
    # DuckDB supports array literals so we use list_position-based logic via
    # a generated CASE block built at query time (see _doy_case_expr()).
    "day_of_year": "__doy_case__",   # sentinel replaced by _doy_case_expr()

    # Hour-of-day: fractional hour → floor to integer bin (0–23)
    "hour_of_day": (
        "GREATEST(0, LEAST({n_bins} - 1, "
        "  FLOOR(hour_of_day)::INTEGER"
        "))"
    ),

    # Longitude and latitude use per-tile min/max from part_stats (not per-row),
    # so they are handled separately via the repeat()-string method.
    "longitude": "__coord__",
    "latitude":  "__coord__",
}


def _doy_case_expr(edges: list[float], n_bins: int) -> str:
    """
    Build a CASE WHEN … END expression that maps day_of_year to a bin index.

    Each bin k covers [edges[k], edges[k+1]).  Because the edges are irregular
    (monthly breakpoints) we cannot use simple floor division.
    """
    lines = ["GREATEST(0, LEAST(%d,\n  CASE" % (n_bins - 1)]
    for k in range(n_bins - 1, -1, -1):
        lines.append(f"    WHEN day_of_year >= {int(edges[k])} THEN {k}")
    lines.append("    ELSE 0\n  END\n))")
    return "\n".join(lines)


def _build_partition_statistics_ddl() -> str:
    """Return the CREATE TABLE statement with one column per DIMENSION_CATALOG entry."""
    coord_dims = {"longitude", "latitude"}
    lines = [
        "CREATE TABLE partition_statistics (",
        "    dataset_id    VARCHAR,",
        "    partition_key VARCHAR,",
        "    tile_id       VARCHAR,",
        "    row_count     BIGINT,",
        "    value_min     DOUBLE,",
        "    value_max     DOUBLE,",
        "    value_mean    DOUBLE,",
    ]
    for dim, meta in DIMENSION_CATALOG.items():
        lines.append(f"    {meta['col']}  BIGINT[],")
    # Primary key closes the statement
    lines.append("    PRIMARY KEY (dataset_id, partition_key, tile_id)")
    lines.append(")")
    return "\n".join(lines)


def write_catalog(output: Path, processed: list[str]) -> None:
    """Write catalog.duckdb with dataset metadata and LST partition histograms."""
    db_path = output / "catalog.duckdb"
    db_path.unlink(missing_ok=True)

    conn = open_duckdb(db_path)
    conn.execute(f"SET temp_directory = '{db_path.parent.as_posix()}'")
    conn.execute("SET memory_limit = '8GB'")
    conn.execute("SET threads = 2")
    conn.execute("SET preserve_insertion_order = false")

    conn.execute("""
        CREATE TABLE dataset_metadata (
            dataset_id          VARCHAR PRIMARY KEY,
            description         VARCHAR,
            feature_columns     VARCHAR[],
            lookup_method       VARCHAR,
            temporal_behavior   VARCHAR,
            partition_column    VARCHAR,
            source_resolution_m DOUBLE,
            is_driving          BOOLEAN,
            value_column        VARCHAR,
            db_file             VARCHAR,
            db_table            VARCHAR,
            store               VARCHAR,
            registered_at       VARCHAR
        )
    """)

    conn.execute(_build_partition_statistics_ddl())

    conn.execute("""
        CREATE TABLE histogram_config (
            dataset_id VARCHAR PRIMARY KEY,
            bin_edges  VARCHAR[],
            n_bins     INTEGER,
            numeric    BOOLEAN
        )
    """)

    # Seed initial histogram_config row (overwritten by _write_lst_histograms)
    conn.execute(
        "INSERT INTO histogram_config VALUES (?, ?, ?, ?)",
        ["lst", [str(v) for v in HIST_EDGES], len(HIST_EDGES) - 1, True],
    )

    now = datetime.now(timezone.utc).isoformat()
    for dataset_id in tqdm(processed, desc="Registering datasets", unit="dataset"):
        spec = DATASET_REGISTRY[dataset_id]
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

    if "lst" in processed:
        lst_db = output / "lst.duckdb"
        if lst_db.exists():
            _write_lst_histograms(conn, lst_db)

    conn.close()
    tqdm.write(f"Catalog written to {db_path}")


def _write_lst_histograms(conn, lst_db: Path) -> None:
    """
    Compute and insert LST partition histograms for every dimension in DIMENSION_CATALOG.

    Structure
    ---------
    One query per partition_key (≈ one month of data).  Each query computes
    all dimension histograms in a single table scan via CTEs:

      For "normal" numeric dimensions (temperature, year, month_of_year,
      day_of_month, hour_of_day): use the SQL expression in _DIMENSION_SQL to
      compute a bin index per row, then GROUP BY to count, then aggregate into
      a dense array via array_agg + LEFT JOIN against range(0, n_bins).

      For "day_of_year": same pattern but with a CASE WHEN bin index expression
      built by _doy_case_expr().

      For "timestamp" (string-keyed quarterly): existing arithmetic expression.

      For "longitude" / "latitude": uses the per-tile min/max from part_stats and
      the repeat()-string trick to avoid materialising huge lon/lat arrays.

    Registering histogram_config
    ----------------------------
    Every dimension gets one row in histogram_config so that the streaming
    layer can recover bin edges without hard-coding them.
    """
    # ---- Register all dimensions in histogram_config -------------------
    conn.execute("DELETE FROM histogram_config")
    for dim, meta in DIMENSION_CATALOG.items():
        edges = get_dimension_edges(dim)
        is_numeric = meta["numeric"]
        conn.execute(
            "INSERT INTO histogram_config VALUES (?, ?, ?, ?)",
            [
                dim,
                [str(e) for e in edges],
                len(edges) - 1,
                is_numeric,
            ],
        )

    # ---- Resolve bin parameters for every dimension --------------------
    # dim_params[dim] = dict of substitution values for _DIMENSION_SQL
    dim_params: Dict[str, dict] = {}
    for dim, meta in DIMENSION_CATALOG.items():
        if _DIMENSION_SQL.get(dim) in ("__coord__", "__doy_case__"):
            edges = get_dimension_edges(dim)
            dim_params[dim] = {"n_bins": len(edges) - 1, "edges": edges}
            continue
        edges = get_dimension_edges(dim)
        n_bins = len(edges) - 1
        params = {"n_bins": n_bins, "edges": edges}
        if meta["numeric"] and dim not in ("timestamp", "day_of_year"):
            params["edge_min"]  = edges[0]
            params["bin_width"] = edges[1] - edges[0]
        dim_params[dim] = params

    # Coordinate dimensions share a special repeat()-based path
    coord_meta = {
        dim: {
            "n_bins":     len(get_dimension_edges(dim)) - 1,
            "edge_min":   get_dimension_edges(dim)[0],
            "bin_width":  get_dimension_edges(dim)[1] - get_dimension_edges(dim)[0],
            "col":        DIMENSION_CATALOG[dim]["col"],
        }
        for dim in ("longitude", "latitude")
    }

    lst_db_posix = lst_db.as_posix()
    tqdm.write("Computing LST partition histograms for all dimensions ...")
    conn.execute(f"ATTACH '{lst_db_posix}' AS lst_db (READ_ONLY)")

    partition_keys = [
        r[0] for r in conn.execute(
            "SELECT DISTINCT partition_key FROM lst_db.lst ORDER BY partition_key"
        ).fetchall()
    ]
    tqdm.write(f"Processing {len(partition_keys)} partition_key(s) ...")

    for part_key in tqdm(partition_keys, desc="LST histograms", unit="month"):
        _insert_partition(conn, part_key, dim_params, coord_meta)

    conn.execute("CHECKPOINT")
    conn.execute("DETACH lst_db")
    tqdm.write("Histogram catalog complete.")


def _bin_cte_sql(dim: str, params: dict) -> tuple[str, str]:
    """
    Return (cte_body_sql, agg_cte_sql) for one non-coordinate dimension.

    cte_body_sql  — the CTE that produces (partition_key, tile_id, bin_idx, bin_count)
    agg_cte_sql   — the CTE that aggregates to a dense BIGINT[] array
    """
    n_bins = params["n_bins"]
    cte_name     = f"{dim}_bins"
    agg_cte_name = f"{dim}_counts_agg"

    sql_tmpl = _DIMENSION_SQL[dim]

    # Resolve the bin index expression
    if sql_tmpl == "__doy_case__":
        bin_expr = _doy_case_expr(params["edges"], n_bins)
    else:
        # Substitute numeric format strings
        substitutions = {k: v for k, v in params.items() if k != "edges"}
        bin_expr = sql_tmpl.format(**substitutions)

    # Extra WHERE for temperature (skip NULLs)
    extra_where = ""
    if dim == "temperature":
        extra_where = "AND COALESCE(aster_lst, modis_lst, ndvi) IS NOT NULL"

    cte_body = f"""
        {cte_name} AS (
            SELECT
                partition_key,
                tile_id,
                {bin_expr} AS bin_idx,
                COUNT(*) AS bin_count
            FROM lst_db.lst
            WHERE partition_key = '{{part_key}}'
              {extra_where}
            GROUP BY partition_key, tile_id, bin_idx
        )"""

    agg_cte = f"""
        {agg_cte_name} AS (
            SELECT
                ps.partition_key,
                ps.tile_id,
                array_agg(COALESCE(b.bin_count, 0) ORDER BY r.idx) AS counts_array
            FROM (SELECT DISTINCT partition_key, tile_id FROM part_stats) ps
            CROSS JOIN (SELECT * FROM range(0, {n_bins})) r(idx)
            LEFT JOIN {cte_name} b
                ON  ps.partition_key = b.partition_key
                AND ps.tile_id       = b.tile_id
                AND r.idx            = b.bin_idx
            GROUP BY ps.partition_key, ps.tile_id
        )"""

    return cte_body, agg_cte, agg_cte_name


def _coord_select_expr(dim: str, meta: dict, stat_col_min: str, stat_col_max: str) -> str:
    """
    Build the repeat()-based BIGINT[] expression for a coordinate dimension.

    Instead of computing per-row bins (which requires materialising a huge
    list), we use the per-tile min/max from part_stats and mark every bin
    between bin(min) and bin(max) as occupied.  This is memory-safe but
    treats each tile's extent as uniform — acceptable for partition scoring.
    """
    n_bins    = meta["n_bins"]
    edge_min  = meta["edge_min"]
    bin_width = meta["bin_width"]

    start_expr = (
        f"GREATEST(0, LEAST({n_bins - 1}, "
        f"FLOOR(({stat_col_min} - ({edge_min})) / {bin_width})::INTEGER))"
    )
    end_expr = (
        f"GREATEST(0, LEAST({n_bins - 1}, "
        f"FLOOR(({stat_col_max} - ({edge_min})) / {bin_width})::INTEGER))"
    )

    return (
        f"CAST(string_split(\n"
        f"    rtrim(concat(\n"
        f"        CASE WHEN {start_expr} > 0\n"
        f"             THEN repeat('0,', {start_expr}) ELSE '' END,\n"
        f"        repeat('1,', {end_expr} - {start_expr} + 1),\n"
        f"        CASE WHEN {end_expr} < {n_bins - 1}\n"
        f"             THEN repeat('0,', {n_bins - 1} - {end_expr})\n"
        f"             ELSE '' END\n"
        f"    ), ','), ',') AS BIGINT[])"
    )


def _insert_partition(conn, part_key: str, dim_params: dict, coord_meta: dict) -> None:
    """
    Run one bulk INSERT INTO partition_statistics for a single partition_key.

    Builds all CTEs and the final SELECT dynamically from DIMENSION_CATALOG
    so adding a new dimension requires no changes here.
    """
    # Collect dimension metadata, skipping coordinate dimensions (handled separately)
    normal_dims = [
        dim for dim in DIMENSION_CATALOG
        if _DIMENSION_SQL.get(dim) not in ("__coord__",)
    ]

    # ---- Build CTEs ----------------------------------------------------
    all_cte_bodies = []
    all_agg_ctes   = []
    agg_cte_names  = {}   # dim → agg CTE name

    for dim in normal_dims:
        params = dim_params[dim]
        body, agg, agg_name = _bin_cte_sql(dim, params)
        all_cte_bodies.append(body.format(part_key=part_key))
        all_agg_ctes.append(agg)
        agg_cte_names[dim] = agg_name

    # part_stats always comes first (provides per-tile row counts + coord ranges)
    part_stats_cte = f"""
        part_stats AS (
            SELECT
                partition_key,
                tile_id,
                COUNT(*) AS row_count,
                MIN(COALESCE(aster_lst, modis_lst, ndvi)) AS t_min,
                MAX(COALESCE(aster_lst, modis_lst, ndvi)) AS t_max,
                AVG(COALESCE(aster_lst, modis_lst, ndvi)) AS t_mean,
                MIN(longitude) AS lon_min,
                MAX(longitude) AS lon_max,
                MIN(latitude)  AS lat_min,
                MAX(latitude)  AS lat_max
            FROM lst_db.lst
            WHERE partition_key = '{part_key}'
            GROUP BY partition_key, tile_id
        )"""

    all_ctes = (
        ["WITH", part_stats_cte, ","]
        + [",\n".join(all_cte_bodies), ","]
        + [",\n".join(all_agg_ctes)]
    )

    # ---- Build SELECT columns ------------------------------------------
    select_parts = [
        "'lst'               AS dataset_id",
        "ps.partition_key",
        "ps.tile_id",
        "ps.row_count",
        "ps.t_min           AS value_min",
        "ps.t_max           AS value_max",
        "ps.t_mean          AS value_mean",
    ]

    join_parts = []

    for dim, meta in DIMENSION_CATALOG.items():
        col = meta["col"]
        if _DIMENSION_SQL.get(dim) == "__coord__":
            # Coordinate dimensions — inline repeat() expression
            if dim == "longitude":
                expr = _coord_select_expr(dim, coord_meta[dim], "cs.lon_min", "cs.lon_max")
            else:
                expr = _coord_select_expr(dim, coord_meta[dim], "cs.lat_min", "cs.lat_max")
            select_parts.append(f"{expr} AS {col}")
        else:
            agg_name = agg_cte_names[dim]
            select_parts.append(f"{agg_name}.counts_array AS {col}")
            join_parts.append(
                f"JOIN {agg_name} ON ps.partition_key = {agg_name}.partition_key"
                f" AND ps.tile_id = {agg_name}.tile_id"
            )

    # coord_stats CTE provides lon/lat min/max for the repeat() expressions
    coord_stats_cte = f"""
        coord_stats AS (
            SELECT partition_key, tile_id,
                   lon_min, lon_max, lat_min, lat_max
            FROM part_stats
        )"""
    # Append coord_stats to the CTE block and alias it as cs in joins
    all_ctes.append(",\n" + coord_stats_cte)
    join_parts.append(
        "JOIN coord_stats cs ON ps.partition_key = cs.partition_key"
        " AND ps.tile_id = cs.tile_id"
    )

    select_clause = ",\n        ".join(select_parts)
    join_clause   = "\n        ".join(join_parts)
    cte_block     = "\n".join(all_ctes)

    sql = f"""
        INSERT INTO partition_statistics
        {cte_block}
        SELECT
            {select_clause}
        FROM part_stats ps
        {join_clause}
    """
    conn.execute(sql)
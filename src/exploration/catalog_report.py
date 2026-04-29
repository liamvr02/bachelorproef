"""
catalog_report.py
=================
Generate an interactive HTML report from catalog.duckdb.

The report contains:
  - Dataset metadata table
  - Total histograms (temperature, timestamp, lon, lat) aggregated across
    all partition_keys and tiles. Temperature = COALESCE(aster_lst, modis_lst) - first non-null LST value
    across ASTER and MODIS. NDVI is a vegetation index (unitless, −1 to +1)
    used as an emissivity correction input, not a temperature fallback.
  - Per-partition_key histograms: one set of 4 histograms per month,
    switchable via a dropdown
  - Per-tile histograms: one set of 4 histograms per tile_id,
    switchable via a dropdown (within the selected partition_key)

Note
----
LST table unifies ASTER and MODIS LST sources into a single table.
ASTER (90 m, 16-day) and MODIS (1 km, daily) both measure land surface
temperature in °C and are interchangeable in COALESCE. NDVI is a separate
dimensionless vegetation index (−1 to +1) derived from NIR/red reflectance;
it is stored alongside LST as an emissivity correction input and must NOT
be used as a temperature fallback.

Performance notes
-----------------
All histogram aggregation is pushed into DuckDB via array_aggregate(..., 'sum')
GROUP BY queries. Python never iterates over raw tile rows - it only receives
one pre-summed row per partition_key and one per tile_id. This reduces the
Python-side work from O(N_tiles * N_bins) to O(N_groups * N_bins).

Usage
-----
    python catalog_report.py
    python catalog_report.py --catalog /path/to/catalog.duckdb --output report.html
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

log = logging.getLogger("catalog_report")


# ============================================================
# DB helpers
# ============================================================

def _open(path: Path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(path), read_only=True)
    temp_dir = Path(path).parent / ".duckdb_tmp"
    temp_dir.mkdir(exist_ok=True)
    conn.execute(f"SET temp_directory = '{temp_dir.as_posix()}'")
    conn.execute("SET threads = 4")
    conn.execute("SET memory_limit = '512MB'")
    return conn


def _fetch(conn, sql: str):
    return conn.execute(sql).fetchall()


def _fetchdf(conn, sql: str):
    return conn.execute(sql).df()


# ============================================================
# DuckDB-side aggregation helpers
# ============================================================

# Column names in partition_statistics that hold histogram BIGINT[] arrays.
_HIST_DB_COLS = [
    "histogram_counts",
    "timestamp_histogram_counts",
    "longitude_histogram_counts",
    "latitude_histogram_counts",
]

# NOTE: array_aggregate(col, 'sum') sums ALL elements to a single scalar -
# it does NOT do an element-wise (per-bin) sum across rows.
# The correct approach is: unnest each array with its bin index, sum per bin,
# then re-collect into a list. The helper below generates that CTE pattern
# for an arbitrary group-by key (or no key for the global total).

def _elementwise_sum_query(group_col: str | None = None) -> str:
    """
    Return a SQL query that produces one row per group (or one global row)
    with element-wise summed histogram arrays.

    group_col: column name to GROUP BY (e.g. 'partition_key', 'tile_id'),
               or None for a global aggregate.

    Result columns: [group_col,] hist0, hist1, hist2, hist3
    where each histN is a BIGINT[] of per-bin counts summed across all rows
    in the group.
    """
    select_group  = f"{group_col}," if group_col else ""
    groupby_outer = f"GROUP BY {group_col}" if group_col else ""
    groupby_inner = f", {group_col}" if group_col else ""
    orderby       = f"ORDER BY {group_col}" if group_col else ""

    # Build one CTE per histogram column.
    ctes = []
    selects = []
    for i, col in enumerate(_HIST_DB_COLS):
        alias = f"h{i}"
        ctes.append(f"""
    {alias} AS (
        SELECT
            pos,
            sum(val) AS val_sum
            {f', {group_col}' if group_col else ''}
        FROM (
            SELECT
                generate_subscripts({col}, 1) AS pos,
                unnest({col}) AS val
                {f', {group_col}' if group_col else ''}
            FROM partition_statistics
            WHERE dataset_id = 'lst'
        )
        GROUP BY pos{groupby_inner}
    ),
    {alias}_agg AS (
        SELECT
            {f'{group_col},' if group_col else ''}
            list(val_sum ORDER BY pos) AS {col}
        FROM {alias}
        {groupby_outer}
    )""")
        selects.append(f"{alias}_agg.{col}")

    # Join all per-column aggregates on group key (or cross-join for global).
    join_col = group_col or None
    if join_col:
        joins = "\n    ".join(
            f"JOIN h{i}_agg USING ({join_col})"
            for i in range(1, len(_HIST_DB_COLS))
        )
        from_clause = f"h0_agg\n    {joins}"
    else:
        from_clause = ", ".join(f"h{i}_agg" for i in range(len(_HIST_DB_COLS)))

    cte_block = "WITH" + ",".join(ctes)

    return f"""
{cte_block}
SELECT
    {select_group}
    {", ".join(selects)}
FROM {from_clause}
{orderby}
"""


def _fetch_total_histograms(conn) -> tuple:
    """One row: globally element-wise summed histogram arrays."""
    rows = _fetch(conn, _elementwise_sum_query(group_col=None))
    return rows[0] if rows else (None, None, None, None)


def _fetch_by_partition(conn) -> list[tuple[str, tuple]]:
    """One element-wise summed row per partition_key, sorted."""
    rows = _fetch(conn, _elementwise_sum_query(group_col="partition_key"))
    # rows: (partition_key, hist0, hist1, hist2, hist3)
    return [(r[0], r[1:]) for r in rows]



# ============================================================
# Histogram aggregation (NumPy, operates on already-summed rows)
# ============================================================

HIST_COLS = {
    #  key          col_offset  label                      color
    "temperature": (0, "Temperature (°C)",        "Royalblue"),
    "timestamp":   (1, "Timestamp (season bins)", "Darkorange"),
    "longitude":   (2, "Longitude (°)",            "Seagreen"),
    "latitude":    (3, "Latitude (°)",             "Crimson"),
}

SUBPLOT_TITLES = [v[1] for v in HIST_COLS.values()]


def _counts_from_agg_row(agg_row: tuple, col_offset: int) -> list[int]:
    """
    Extract counts list from a pre-aggregated row returned by _fetch_by_*.
    agg_row is (hist0, hist1, hist2, hist3) - the non-key tail of the DB row.
    Uses NumPy for the (rare) case where multiple raw rows were summed in Python;
    here it's mostly a passthrough since DuckDB already summed everything.
    """
    arr = agg_row[col_offset]
    if arr is None:
        return []
    return list(arr)


def _is_numeric_edges(edges: list) -> bool:
    """Return True if every edge can be interpreted as a float."""
    try:
        [float(e) for e in edges]
        return True
    except (ValueError, TypeError):
        return False


def _mid_points(edges: list[float]) -> list[float]:
    e = np.asarray(edges, dtype=np.float64)
    return ((e[:-1] + e[1:]) / 2).tolist()


# ============================================================
# Plotly helpers
# ============================================================

def _bar_trace(edges, counts, color, name="", visible=True):
    if not edges or not counts:
        return go.Bar(x=[], y=[], name=name, visible=visible, marker_color=color)

    if _is_numeric_edges(edges):
        # Numeric bins: place bars at bin midpoints with explicit widths.
        xs    = _mid_points(edges)
        width = np.diff(np.asarray(edges, dtype=np.float64)).tolist()
        return go.Bar(
            x=xs, y=counts, width=width,
            name=name, visible=visible,
            marker_color=color, marker_line_width=0,
            hovertemplate="%{x:.3f}: %{y:,}<extra></extra>",
        )
    else:
        # Categorical bins (e.g. '2000-Q1'): use the left edge label as the
        # category name; Plotly will space bars evenly.
        labels = edges[:-1]   # N bins from N+1 edges, drop the trailing sentinel
        return go.Bar(
            x=labels, y=counts,
            name=name, visible=visible,
            marker_color=color, marker_line_width=0,
            hovertemplate="%{x}: %{y:,}<extra></extra>",
        )


def _apply_common_layout(fig, title, height, margin_t):
    fig.update_layout(
        title_text=f"{title}<br><sub>Temperature = COALESCE(aster_lst, modis_lst)</sub>",
        title_font_size=14,
        showlegend=False,
        height=height,
        margin=dict(t=margin_t, b=40, l=60, r=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")


def _make_total_figure(title: str, agg_row: tuple, edges_map: dict) -> go.Figure:
    """4-panel figure for a single pre-aggregated row (global totals)."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=SUBPLOT_TITLES,
        horizontal_spacing=0.10,
        vertical_spacing=0.16,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for (key, (col_offset, label, color)), (r, c) in zip(HIST_COLS.items(), positions):
        counts = _counts_from_agg_row(agg_row, col_offset)
        edges  = edges_map.get(key, [])
        fig.add_trace(_bar_trace(edges, counts, color, name=label), row=r, col=c)
    _apply_common_layout(fig, title, height=650, margin_t=100)
    return fig


def _make_dropdown_figure(
    title: str,
    groups: list[tuple[str, tuple]],   # (label, agg_row)
    edges_map: dict,
) -> go.Figure:
    """
    4-panel figure with a dropdown to switch between groups.
    groups: list of (label, agg_row) where agg_row is a pre-summed tuple
            (hist0, hist1, hist2, hist3) - no raw rows, no Python aggregation.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=SUBPLOT_TITLES,
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
    )
    positions    = [(1, 1), (1, 2), (2, 1), (2, 2)]
    n_hist       = 4
    n_groups     = len(groups)
    total_traces = n_groups * n_hist

    for g_idx, (label, agg_row) in enumerate(groups):
        visible = (g_idx == 0)
        for (key, (col_offset, hist_label, color)), (r, c) in zip(HIST_COLS.items(), positions):
            counts = _counts_from_agg_row(agg_row, col_offset)
            edges  = edges_map.get(key, [])
            fig.add_trace(
                _bar_trace(edges, counts, color, name=hist_label, visible=visible),
                row=r, col=c,
            )

    buttons = []
    for g_idx, (label, _) in enumerate(groups):
        vis = [False] * total_traces
        for t in range(n_hist):
            vis[g_idx * n_hist + t] = True
        buttons.append(dict(
            label=label,
            method="update",
            args=[
                {"visible": vis},
                {"title": {"text": f"{title} - {label}"}},
            ],
        ))

    first_label = groups[0][0] if groups else ""
    fig.update_layout(
        title_text=f"{title} - {first_label}<br><sub>Temperature = COALESCE(aster_lst, modis_lst)  |  NDVI = vegetation index (emissivity input, not LST)</sub>",
        title_font_size=13,
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.0, xanchor="left",
            y=1.22, yanchor="top",
            bgcolor="#f0f0f0",
            bordercolor="#cccccc",
        )],
        showlegend=False,
        height=700,
        margin=dict(t=140, b=40, l=60, r=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    return fig


# ============================================================
# Metadata table
# ============================================================

def _make_metadata_table(conn: duckdb.DuckDBPyConnection) -> go.Figure:
    df = _fetchdf(conn, """
        SELECT dataset_id, description, lookup_method, temporal_behavior,
               source_resolution_m, is_driving, store, db_file
        FROM dataset_metadata
        ORDER BY dataset_id
    """)
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No dataset_metadata rows found.",
                           showarrow=False, font_size=14)
        return fig

    df = df.copy()
    df["description"] = df["description"].astype(str)

    fig = go.Figure(data=[go.Table(
        columnwidth=[100, 280, 140, 120, 100, 80, 90, 130],
        header=dict(
            values=["<b>dataset_id</b>", "<b>description</b>", "<b>lookup_method</b>",
                    "<b>temporal_behavior</b>", "<b>resolution_m</b>", "<b>driving</b>",
                    "<b>store</b>", "<b>db_file</b>"],
            fill_color="#2c3e50",
            font=dict(color="white", size=12),
            align="left",
            height=32,
        ),
        cells=dict(
            values=[df[c].astype(str).tolist() for c in df.columns],
            fill_color=[["#f9f9f9", "#ffffff"] * (len(df) // 2 + 1)],
            align="left",
            font_size=11,
            height=26,
        ),
    )])
    fig.update_layout(
        title_text="Dataset Metadata<br><sub>LST: ASTER (90 m, 16-day) and MODIS (1 km, daily) are interchangeable LST sources. NDVI is a dimensionless vegetation index used as emissivity input.</sub>",
        title_font_size=16,
        margin=dict(t=80, b=20, l=20, r=20),
        height=max(200, 80 + len(df) * 30),
    )
    return fig


# ============================================================
# Summary stats table
# ============================================================

def _make_stats_table(conn: duckdb.DuckDBPyConnection) -> go.Figure:
    import pandas as pd

    df = _fetchdf(conn, """
        SELECT
            partition_key,
            COUNT(DISTINCT tile_id)      AS tiles,
            SUM(row_count)               AS total_rows,
            ROUND(MIN(value_min),  2)    AS t_min,
            ROUND(MAX(value_max),  2)    AS t_max,
            ROUND(AVG(value_mean), 2)    AS t_mean_avg
        FROM partition_statistics
        WHERE dataset_id = 'lst'
        GROUP BY partition_key
        ORDER BY partition_key
    """)
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No partition_statistics rows found.",
                           showarrow=False, font_size=14)
        return fig

    totals = {
        "partition_key": "TOTAL",
        "tiles":         df["tiles"].sum(),
        "total_rows":    df["total_rows"].sum(),
        "t_min":         df["t_min"].min(),
        "t_max":         df["t_max"].max(),
        "t_mean_avg":    round(float(df["t_mean_avg"].mean()), 2),
    }
    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

    row_colors = [
        "#dce8f7" if i == len(df) - 1 else ("#f0f0f0" if i % 2 == 0 else "#ffffff")
        for i in range(len(df))
    ]
    fig = go.Figure(data=[go.Table(
        columnwidth=[110, 70, 120, 80, 80, 100],
        header=dict(
            values=["<b>partition_key</b>", "<b>tiles</b>", "<b>total_rows</b>",
                    "<b>t_min (°C)*</b>", "<b>t_max (°C)*</b>", "<b>t_mean (°C)*</b>"],
            fill_color="#2c3e50",
            font=dict(color="white", size=12),
            align="left",
            height=32,
        ),
        cells=dict(
            values=[df[c].astype(str).tolist() for c in df.columns],
            fill_color=[row_colors],
            align="left",
            font_size=11,
            height=26,
        ),
    )])
    fig.update_layout(
        title_text=(
            "Partition Summary (LST)<br>"
            "<sub>* Temperature = COALESCE(aster_lst, modis_lst) - first non-null LST across ASTER/MODIS. "
            "NDVI is a vegetation index (emissivity input) and is not included in this fallback.</sub>"
        ),
        title_font_size=14,
        margin=dict(t=100, b=20, l=20, r=20),
        height=max(200, 120 + len(df) * 28),
    )
    return fig


# ============================================================
# HTML assembly
# ============================================================

_SECTION_STYLE = (
    "font-family: Arial, sans-serif; font-size: 22px; font-weight: bold; "
    "color: #2c3e50; margin: 32px 0 8px 0; padding-left: 8px; "
    "border-left: 5px solid #2980b9;"
)

_PAGE_STYLE = """
<style>
  body { background: #f4f6f9; margin: 0; padding: 0; }
  .report-header {
    background: #2c3e50; color: white;
    padding: 28px 40px 20px 40px;
    font-family: Arial, sans-serif;
  }
  .report-header h1 { margin: 0 0 6px 0; font-size: 28px; }
  .report-header p  { margin: 0; font-size: 14px; opacity: 0.8; }
  .section-label { """ + _SECTION_STYLE + """ }
  .plot-wrapper  { background: white; border-radius: 6px;
                   box-shadow: 0 1px 4px rgba(0,0,0,0.08);
                   margin: 0 24px 24px 24px; padding: 4px; }
</style>
"""


def _fig_div(fig: go.Figure) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=None,
        config={"displayModeBar": True, "responsive": True},
    )


# ============================================================
# Main report builder
# ============================================================

def build_report(catalog_path: Path, output_path: Path) -> None:
    log.info("Opening %s ...", catalog_path)
    conn = _open(catalog_path)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── metadata & stats tables ──────────────────────────────────────────
    log.info("Building metadata table ...")
    fig_meta  = _make_metadata_table(conn)
    log.info("Building partition stats table ...")
    fig_stats = _make_stats_table(conn)

    # ── fetch edges once ─────────────────────────────────────────────────
    log.info("Fetching histogram edges ...")
    edges_config = _fetchdf(conn, """
        SELECT dataset_id, bin_edges FROM histogram_config
        WHERE dataset_id IN ('temperature', 'timestamp', 'longitude', 'latitude')
    """)
    edges_map = {
        row["dataset_id"]: list(row["bin_edges"])
        for _, row in edges_config.iterrows()
    }

    # ── total histograms: single aggregated row from DuckDB ──────────────
    log.info("Fetching globally aggregated histograms ...")
    total_agg = _fetch_total_histograms(conn)
    fig_total = _make_total_figure(
        "Total LST Histograms (ASTER + MODIS)",
        total_agg, edges_map,
    )

    # ── per-partition_key: one pre-summed row per partition from DuckDB ──
    log.info("Fetching per-partition_key aggregated histograms ...")
    part_groups = _fetch_by_partition(conn)   # [(pk, agg_row), ...]
    log.info("  %d partition(s)", len(part_groups))
    fig_by_partition = _make_dropdown_figure(
        "LST Histograms by Month (ASTER + MODIS)",
        part_groups, edges_map,
    )


    conn.close()

    # ── assemble HTML ────────────────────────────────────────────────────
    log.info("Writing HTML to %s ...", output_path)
    sections = [
        ("Dataset Metadata",            fig_meta),
        ("Partition Summary",           fig_stats),
        ("Total Histograms (All Data)", fig_total),
        ("Histograms by Month",         fig_by_partition),
    ]

    body_parts = []
    for heading, fig in sections:
        body_parts.append(f'<div class="section-label">{heading}</div>')
        body_parts.append(f'<div class="plot-wrapper">{_fig_div(fig)}</div>')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Catalog Report</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  {_PAGE_STYLE}
</head>
<body>
  <div class="report-header">
    <h1>Catalog Report</h1>
    <p>Generated: {generated} &nbsp;|&nbsp; Source: {catalog_path}</p>
  </div>
  {''.join(body_parts)}
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    log.info("Report written to %s", output_path)


# ============================================================
# CLI
# ============================================================

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    here  = Path(__file__).resolve().parent
    _SRC  = here.parent
    _REPO = _SRC.parent

    default_catalog = _SRC / "prepared_stream_data" / "catalog.duckdb"
    default_output  = here / "catalog_report.html"

    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML report from catalog.duckdb.",
    )
    parser.add_argument(
        "--catalog", type=Path, default=default_catalog,
        help=f"Path to catalog.duckdb  (default: {default_catalog})",
    )
    parser.add_argument(
        "--output", type=Path, default=default_output,
        help=f"Output HTML path  (default: {default_output})",
    )
    args = parser.parse_args()

    if not args.catalog.exists():
        log.error("Catalog not found: %s", args.catalog)
        raise SystemExit(1)

    build_report(args.catalog, args.output)
    print(f"\nReport ready: {args.output}")


if __name__ == "__main__":
    main()
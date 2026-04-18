"""
ingest/ingest_trees.py
======================
Ingestor: Ghent tree inventory CSV  →  trees.duckdb
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from db import open_duckdb

log = logging.getLogger("ingest.trees")

# Candidate column names (first entry is the confirmed real name; extras are fallbacks)
_TREE_LON_COLS     = ["longitude", "lon", "x", "lng"]
_TREE_LAT_COLS     = ["latitude",  "lat", "y"]
_TREE_SPECIES_COLS = ["sortiment", "species", "soort", "boomsoort"]
_TREE_HEIGHT_COLS  = ["hoogte", "height", "height_m", "kroonhoogte"]
_TREE_YEAR_COLS    = ["aanlegjaar", "planting_year", "plantjaar", "jaar"]
_TREE_DIAM_COLS    = ["stamomtrek", "diameter", "trunk_diameter_cm", "omtrek"]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    return None


def ingest_trees(downloads: Path, output: Path) -> int:
    """
    Load all trees CSV files into trees.duckdb — no SpatiaLite, no GEOS.

    Source columns (confirmed by survey_sources.py):
        geo_point_2d            "lat, lon" string → split to latitude / longitude
        sortiment               species/variety name (~80% full)
        hoogte                  height category text, e.g. '6-9 m.' (~7% full)
        aanlegjaar              planting year integer (~27% full)
        stamomtrek              trunk circumference in cm (~67% full)

    Schema (trees table in trees.duckdb):
        longitude               DOUBLE
        latitude                DOUBLE
        species                 VARCHAR   ← sortiment
        height_m                VARCHAR   ← hoogte (raw text range)
        planting_year           INTEGER   ← aanlegjaar
        trunk_circumference_cm  DOUBLE    ← stamomtrek (cm)
    """
    csv_candidates = list(downloads.rglob("ghent_trees_*.csv"))
    csv_candidates = list(dict.fromkeys(csv_candidates))
    if not csv_candidates:
        log.warning("No trees CSV files found under %s — skipping", downloads)
        return 0

    csv_files = [max(csv_candidates, key=lambda p: p.name)]
    log.info("Trees: using latest file: %s  (%d candidate(s) found)",
             csv_files[0].name, len(csv_candidates))

    db_path = output / "trees.duckdb"
    conn    = open_duckdb(db_path)
    conn.execute("DROP TABLE IF EXISTS trees")
    conn.execute("""
        CREATE TABLE trees (
            longitude              DOUBLE,
            latitude               DOUBLE,
            species                VARCHAR,
            height_m               VARCHAR,
            planting_year          INTEGER,
            trunk_circumference_cm DOUBLE
        )
    """)

    total = 0
    for csv_path in tqdm(csv_files, desc="Trees CSVs", unit="file"):
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(4096)
        delimiter = ";" if sample.count(";") > sample.count(",") else ","

        df = pd.read_csv(csv_path, delimiter=delimiter, low_memory=False)
        df = df.drop(columns=["Geometry"], errors="ignore")

        if "geo_point_2d" in df.columns:
            coords = df["geo_point_2d"].str.split(r",\s*", expand=True)
            df["latitude"]  = pd.to_numeric(coords[0], errors="coerce")
            df["longitude"] = pd.to_numeric(coords[1], errors="coerce")

        lon_col = _find_col(df, _TREE_LON_COLS)
        lat_col = _find_col(df, _TREE_LAT_COLS)
        if lon_col is None or lat_col is None:
            log.warning("Cannot find lon/lat columns in %s — skipping", csv_path.name)
            continue

        df = df.dropna(subset=[lon_col, lat_col])
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df = df.dropna(subset=[lon_col, lat_col])

        species_col = _find_col(df, _TREE_SPECIES_COLS)
        height_col  = _find_col(df, _TREE_HEIGHT_COLS)
        year_col    = _find_col(df, _TREE_YEAR_COLS)
        circ_col    = _find_col(df, _TREE_DIAM_COLS)

        log.info("Trees %s: lon=%s lat=%s species=%s height=%s year=%s circ=%s",
                 csv_path.name, lon_col, lat_col,
                 species_col, height_col, year_col, circ_col)

        out_rows = []
        for row in df.itertuples(index=False):
            lon = float(getattr(row, lon_col))
            lat = float(getattr(row, lat_col))
            if not (lon == lon and lat == lat):  # NaN check
                continue

            species_val = None
            if species_col:
                v = getattr(row, species_col, None)
                if v is not None and str(v) not in ("nan", "None", ""):
                    species_val = str(v)

            height_val = None
            if height_col:
                v = getattr(row, height_col, None)
                if v is not None and str(v) not in ("nan", "None", ""):
                    height_val = str(v)

            year_val = None
            if year_col:
                v = getattr(row, year_col, None)
                try:
                    iv = int(float(v))
                    if iv > 0:
                        year_val = iv
                except (TypeError, ValueError):
                    pass

            circ_val = None
            if circ_col:
                v = getattr(row, circ_col, None)
                try:
                    fv = float(v)
                    if fv == fv:  # not NaN
                        circ_val = fv
                except (TypeError, ValueError):
                    pass

            out_rows.append({
                "longitude":              lon,
                "latitude":               lat,
                "species":                species_val,
                "height_m":               height_val,
                "planting_year":          year_val,
                "trunk_circumference_cm": circ_val,
            })

        out = pd.DataFrame(out_rows)
        if out.empty:
            log.warning("Trees %s: no valid rows", csv_path.name)
            continue
        conn.append("trees", out)
        total += len(out)
        log.info("Trees %s: %d rows", csv_path.name, len(out))

    conn.execute("CHECKPOINT")
    conn.close()
    return total

"""
ingest/ingest_trees.py
======================
Ingestor: Ghent tree inventory CSV  ->  trees.duckdb
"""

from __future__ import annotations

import logging
import re
import unicodedata
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
_TREE_PHASE_COLS   = ["beheerfase", "management_phase", "phase", "fase"]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    return None


# ---------------------------------------------------------------------------
# Genus extraction from `sortiment`
# ---------------------------------------------------------------------------
# Plant family names in Latin nomenclature end in `-aceae` (e.g. "Ulmaceae",
# "Cornaceae").  When a record begins with a family name the actual genus is
# the next token, e.g. "Ulmaceae zelkova 'Serrata'" → "Zelkova".
_FAMILY_SUFFIX = re.compile(r"aceae$", re.IGNORECASE)
# "Te Bepalen" is the Dutch placeholder for "to be determined" — no real genus.
_NON_GENUS_FIRST_TOKENS = {"te"}


def _extract_genus(sortiment: Optional[str]) -> Optional[str]:
    """
    Best-effort genus extraction from a binomial-nomenclature *sortiment* value.

    Heuristic
    ---------
    1. Strip and tokenise on whitespace (NBSP normalised to space first).
    2. If the first token is a placeholder ("Te" from "Te Bepalen"), return None.
    3. If the first token ends in ``-aceae`` (Latin family suffix), use the
       second token instead — this rescues "Ulmaceae zelkova" → "Zelkova".
    4. Strip stray punctuation (``,'"()×``) and Title-case the result.

    Typos are *not* corrected ("Qeurcus", "Cercius"), so they form their own
    small genus groups; that is acceptable for downstream feature use.

    Returns ``None`` for null / empty / placeholder input.
    """
    if sortiment is None:
        return None
    s = unicodedata.normalize("NFKC", str(sortiment)).replace("\xa0", " ").strip()
    if not s:
        return None
    tokens = s.split()
    if not tokens:
        return None
    head = tokens[0].strip("'\"().,;:")
    if head.lower() in _NON_GENUS_FIRST_TOKENS:
        return None
    if _FAMILY_SUFFIX.search(head) and len(tokens) >= 2:
        head = tokens[1].strip("'\"().,×x")
    if not head:
        return None
    return head[:1].upper() + head[1:].lower()


def ingest_trees(downloads: Path, output: Path) -> int:
    """
    Load all trees CSV files into trees.duckdb - no SpatiaLite, no GEOS.

    Source columns (confirmed by survey_sources.py):
        geo_point_2d            "lat, lon" string -> split to latitude / longitude
        sortiment               species/variety name (~80% full)
        hoogte                  height category text, e.g. '6-9 m.' (~7% full)
        aanlegjaar              planting year integer (~27% full)
        stamomtrek              trunk circumference in cm (~67% full)
        beheerfase              management phase text (Jeugd / Volwassen / Veteranen)

    Schema (trees table in trees.duckdb) — original Dutch column names retained:
        longitude    DOUBLE
        latitude     DOUBLE
        sortiment    VARCHAR   (binomial nomenclature, ~80% full)
        hoogte       VARCHAR   (raw height-range text, ~7% full)
        aanlegjaar   INTEGER   (planting year, ~27% full)
        stamomtrek   DOUBLE    (trunk circumference cm, ~67% full)
        beheerfase   VARCHAR   (Jeugdfase / Volwassenfase / Veteranenfase / NULL)
        genus        VARCHAR   (derived from sortiment via _extract_genus, ~80% full)
    """
    csv_candidates = list(downloads.rglob("ghent_trees_*.csv"))
    csv_candidates = list(dict.fromkeys(csv_candidates))
    if not csv_candidates:
        log.warning("No trees CSV files found under %s - skipping", downloads)
        return 0

    csv_files = [max(csv_candidates, key=lambda p: p.name)]
    log.info("Trees: using latest file: %s  (%d candidate(s) found)",
             csv_files[0].name, len(csv_candidates))

    db_path = output / "trees.duckdb"
    conn    = open_duckdb(db_path)
    conn.execute("DROP TABLE IF EXISTS trees")
    conn.execute("""
        CREATE TABLE trees (
            longitude    DOUBLE,
            latitude     DOUBLE,
            sortiment    VARCHAR,
            hoogte       VARCHAR,
            aanlegjaar   INTEGER,
            stamomtrek   DOUBLE,
            beheerfase   VARCHAR,
            genus        VARCHAR
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
            log.warning("Cannot find lon/lat columns in %s - skipping", csv_path.name)
            continue

        df = df.dropna(subset=[lon_col, lat_col])
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df = df.dropna(subset=[lon_col, lat_col])

        sortiment_col = _find_col(df, _TREE_SPECIES_COLS)
        hoogte_col    = _find_col(df, _TREE_HEIGHT_COLS)
        year_col      = _find_col(df, _TREE_YEAR_COLS)
        circ_col      = _find_col(df, _TREE_DIAM_COLS)
        phase_col     = _find_col(df, _TREE_PHASE_COLS)

        log.info("Trees %s: lon=%s lat=%s sortiment=%s hoogte=%s aanlegjaar=%s "
                 "stamomtrek=%s beheerfase=%s",
                 csv_path.name, lon_col, lat_col,
                 sortiment_col, hoogte_col, year_col, circ_col, phase_col)

        out_rows = []
        for row in df.itertuples(index=False):
            lon = float(getattr(row, lon_col))
            lat = float(getattr(row, lat_col))
            if not (lon == lon and lat == lat):  # NaN check
                continue

            sortiment_val = None
            if sortiment_col:
                v = getattr(row, sortiment_col, None)
                if v is not None and str(v) not in ("nan", "None", ""):
                    sortiment_val = str(v)

            hoogte_val = None
            if hoogte_col:
                v = getattr(row, hoogte_col, None)
                if v is not None and str(v) not in ("nan", "None", ""):
                    hoogte_val = str(v)

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

            phase_val = None
            if phase_col:
                v = getattr(row, phase_col, None)
                if v is not None and str(v) not in ("nan", "None", ""):
                    phase_val = str(v)

            out_rows.append({
                "longitude":  lon,
                "latitude":   lat,
                "sortiment":  sortiment_val,
                "hoogte":     hoogte_val,
                "aanlegjaar": year_val,
                "stamomtrek": circ_val,
                "beheerfase": phase_val,
                "genus":      _extract_genus(sortiment_val),
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

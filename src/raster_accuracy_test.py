"""
raster_accuracy_test.py — Verify cached raster layers against Shapely truth.

For every (grid_key, layer_key) in stream_cache/rasters/raster_cache.duckdb:
  1.  Decode the cached fraction grid (n_lon × n_lat, float32).
  2.  Sample N random non-zero cells and N random zero cells at grid centres.
  3.  For each sample, compute the ground-truth fraction directly from the
      source SpatiaLite polygons via _ua_make_circle + _ua_compute_fraction
      (the same Shapely path the live batch streaming uses).
  4.  Assert |raster_value − truth| ≤ tolerance.

Per-layer and overall pass/fail counts and error percentiles are printed.

Usage:
    python raster_accuracy_test.py [--samples N] [--tol 0.01]
                                   [--zero-tol 0.005] [--layers wis:Rijweg,12100:2018]
                                   [--seed 42]
"""

from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np

from stream.poly_raster import (
    _ua_compute_fraction,
    _ua_make_circle,
    _ua_fetch_candidates,
    _wis_fetch_candidates,
)


SRC          = Path(__file__).resolve().parent
SPATIAL_DB   = SRC / "prepared_stream_data" / "spatial.db"
CACHE_DB     = SRC / "stream_cache" / "rasters" / "raster_cache.duckdb"


# ---------------------------------------------------------------------------
# SpatiaLite open
# ---------------------------------------------------------------------------
def open_spatialite() -> sqlite3.Connection:
    if not SPATIAL_DB.exists():
        raise FileNotFoundError(f"Missing {SPATIAL_DB}")
    db = sqlite3.connect(str(SPATIAL_DB))
    db.enable_load_extension(True)
    for lib in ("mod_spatialite", "mod_spatialite.so",
                "mod_spatialite.dylib",
                "/usr/lib/x86_64-linux-gnu/mod_spatialite.so"):
        try:
            db.load_extension(lib)
            break
        except sqlite3.OperationalError:
            continue
    else:
        raise RuntimeError("Could not load mod_spatialite")
    return db


# ---------------------------------------------------------------------------
# Layer-key parsing
# ---------------------------------------------------------------------------
def parse_layer_key(layer_key: str) -> Tuple[str, dict]:
    """
    Resolve a cached layer_key to (kind, params).

    kind == 'ua'  → params = {'luc_code': str, 'ua_year': int}
    kind == 'wis' → params = {'attr_val': str}      (attr_col resolved later)
    """
    if layer_key.startswith("fft:wis:"):
        return "wis", {"attr_val": layer_key[len("fft:wis:"):]}
    if layer_key.startswith("wis:"):
        return "wis", {"attr_val": layer_key[len("wis:"):]}
    # UA: "{luc_code}:{ua_year}"
    parts = layer_key.rsplit(":", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return "ua", {"luc_code": parts[0], "ua_year": int(parts[1])}
    raise ValueError(f"unrecognised layer key: {layer_key!r}")


def resolve_wis_attr_col(db: sqlite3.Connection, attr_val: str) -> Optional[str]:
    """Return 'bestemming' or 'materiaalsoort' depending on which column holds
    rows with the given value.  Returns None if neither matches."""
    for col in ("bestemming", "materiaalsoort"):
        n = db.execute(
            f"SELECT 1 FROM wis WHERE {col} = ? LIMIT 1", (attr_val,)
        ).fetchone()
        if n is not None:
            return col
    return None


# ---------------------------------------------------------------------------
# Truth computation
# ---------------------------------------------------------------------------
def truth_fraction(
    db: sqlite3.Connection,
    kind: str,
    params: dict,
    lon: float,
    lat: float,
    radius_m: float,
) -> float:
    """Compute the ground-truth covered-area fraction at (lon, lat)."""
    if kind == "ua":
        blobs = _ua_fetch_candidates(
            db, lon, lat, radius_m,
            params["luc_code"], params["ua_year"],
        )
    elif kind == "wis":
        blobs = _wis_fetch_candidates(
            db, lon, lat, radius_m,
            params["attr_col"], params["attr_val"],
        )
    else:
        raise ValueError(kind)
    if not blobs:
        return 0.0
    circle = _ua_make_circle(lon, lat, radius_m)
    return _ua_compute_fraction(blobs, circle)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def pick_indices(
    arr: np.ndarray, n: int, kind: str, rng: random.Random,
) -> List[Tuple[int, int]]:
    """Return up to n (ix, iy) pairs from arr matching `kind` ('zero' or 'nonzero')."""
    if kind == "nonzero":
        mask = (arr > 0) & np.isfinite(arr)
    elif kind == "zero":
        mask = (arr == 0.0)
    else:
        raise ValueError(kind)
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return []
    if len(coords) <= n:
        return [tuple(c) for c in coords]
    idx = rng.sample(range(len(coords)), n)
    return [tuple(coords[i]) for i in idx]


# ---------------------------------------------------------------------------
# Per-layer test
# ---------------------------------------------------------------------------
def test_layer(
    sdb: sqlite3.Connection,
    grid: dict,
    layer_key: str,
    arr: np.ndarray,
    radius_m: float,
    n_samples: int,
    tol: float,
    zero_tol: float,
    rng: random.Random,
) -> dict:
    """Run accuracy checks on one layer.  Returns a stats dict."""
    kind, params = parse_layer_key(layer_key)
    if kind == "wis":
        attr_col = resolve_wis_attr_col(sdb, params["attr_val"])
        if attr_col is None:
            return {
                "layer": layer_key, "skipped": "no rows for attr_val",
                "nz_total": 0, "z_total": 0, "nz_pass": 0, "z_pass": 0,
                "nz_err": [], "z_err": [],
            }
        params["attr_col"] = attr_col

    nz_idx = pick_indices(arr, n_samples, "nonzero", rng)
    z_idx  = pick_indices(arr, n_samples, "zero",    rng)

    nz_err: List[float] = []
    nz_pass = 0
    nz_fail_examples: List[str] = []
    for ix, iy in nz_idx:
        lon = grid["lon0"] + ix * grid["step_lon"]
        lat = grid["lat0"] + iy * grid["step_lat"]
        cached = float(arr[ix, iy])
        true_v = truth_fraction(sdb, kind, params, lon, lat, radius_m)
        err = abs(cached - true_v)
        nz_err.append(err)
        if err <= tol:
            nz_pass += 1
        elif len(nz_fail_examples) < 3:
            nz_fail_examples.append(
                f"({lon:.5f},{lat:.5f}) cached={cached:.4f} truth={true_v:.4f} Δ={err:.4f}"
            )

    z_err: List[float] = []
    z_pass = 0
    z_fail_examples: List[str] = []
    for ix, iy in z_idx:
        lon = grid["lon0"] + ix * grid["step_lon"]
        lat = grid["lat0"] + iy * grid["step_lat"]
        cached = float(arr[ix, iy])
        true_v = truth_fraction(sdb, kind, params, lon, lat, radius_m)
        err = abs(cached - true_v)
        z_err.append(err)
        if true_v <= zero_tol and cached <= zero_tol:
            z_pass += 1
        elif len(z_fail_examples) < 3:
            z_fail_examples.append(
                f"({lon:.5f},{lat:.5f}) cached={cached:.4f} truth={true_v:.4f}"
            )

    return {
        "layer": layer_key,
        "kind":  kind,
        "radius_m": radius_m,
        "nz_total": len(nz_idx), "nz_pass": nz_pass, "nz_err": nz_err,
        "nz_fail_examples": nz_fail_examples,
        "z_total":  len(z_idx),  "z_pass":  z_pass,  "z_err":  z_err,
        "z_fail_examples": z_fail_examples,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def fmt_pct(passed: int, total: int) -> str:
    if total == 0:
        return "  n/a "
    return f"{passed}/{total} ({100.0 * passed / total:5.1f}%)"


def percentiles(errs: List[float]) -> str:
    if not errs:
        return "—"
    a = np.asarray(errs)
    return f"max={a.max():.4f} p95={np.percentile(a, 95):.4f} mean={a.mean():.4f}"


def print_layer_row(s: dict) -> None:
    if "skipped" in s:
        print(f"  {s['layer']:<40s} SKIPPED ({s['skipped']})")
        return
    print(
        f"  {s['layer']:<40s} r={s['radius_m']:>5.0f}m  "
        f"nonzero {fmt_pct(s['nz_pass'], s['nz_total']):>14s}  "
        f"zero {fmt_pct(s['z_pass'], s['z_total']):>14s}  "
        f"err[{percentiles(s['nz_err'])}]"
    )
    for ex in s.get("nz_fail_examples", []):
        print(f"      ✗ nonzero  {ex}")
    for ex in s.get("z_fail_examples", []):
        print(f"      ✗ zero     {ex}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples",  type=int,   default=5,
                    help="non-zero AND zero samples per layer (default 5+5)")
    ap.add_argument("--tol",      type=float, default=0.01,
                    help="absolute fraction tolerance for non-zero cells (default 0.01)")
    ap.add_argument("--zero-tol", type=float, default=0.005,
                    help="absolute fraction tolerance for zero cells (default 0.005)")
    ap.add_argument("--layers",   type=str,   default=None,
                    help="comma-separated substrings; only matching layer keys are tested")
    ap.add_argument("--seed",     type=int,   default=42)
    args = ap.parse_args()

    if not CACHE_DB.exists():
        print(f"ERROR: no raster cache at {CACHE_DB}", file=sys.stderr)
        return 2

    rng  = random.Random(args.seed)
    cdb  = duckdb.connect(str(CACHE_DB), read_only=True)
    sdb  = open_spatialite()

    grids = {
        row[0]: {
            "resolution_m": row[1], "lon0": row[2], "lat0": row[3],
            "step_lon": row[4], "step_lat": row[5],
            "n_lon": int(row[6]), "n_lat": int(row[7]),
        }
        for row in cdb.execute(
            "SELECT grid_key, resolution_m, lon0, lat0, step_lon, step_lat, n_lon, n_lat "
            "FROM raster_grid"
        ).fetchall()
    }

    layer_rows = cdb.execute(
        "SELECT grid_key, layer_key, radius_m, n_nonzero, array_blob "
        "FROM raster_layer ORDER BY layer_key"
    ).fetchall()

    if args.layers:
        wanted = [s.strip() for s in args.layers.split(",") if s.strip()]
        layer_rows = [r for r in layer_rows
                      if any(w in r[1] for w in wanted)]

    print(f"\n{'='*100}")
    print(f"  Raster accuracy test")
    print(f"  cache:        {CACHE_DB}")
    print(f"  spatial:      {SPATIAL_DB}")
    print(f"  grids:        {len(grids)}     layers tested: {len(layer_rows)}")
    print(f"  samples/layer:{args.samples} non-zero + {args.samples} zero")
    print(f"  tolerance:    nz≤{args.tol}  z≤{args.zero_tol}")
    print(f"{'='*100}\n")

    overall = {"nz_total": 0, "nz_pass": 0, "z_total": 0, "z_pass": 0,
               "nz_err": [], "z_err": []}
    layer_stats: List[dict] = []

    for grid_key, layer_key, radius_m, n_nonzero, blob in layer_rows:
        grid = grids.get(grid_key)
        if grid is None:
            print(f"  {layer_key:<40s} SKIPPED (no grid row)")
            continue
        arr = np.frombuffer(bytes(blob), dtype=np.float32).reshape(
            grid["n_lon"], grid["n_lat"]).copy()

        s = test_layer(sdb, grid, layer_key, arr, float(radius_m),
                       args.samples, args.tol, args.zero_tol, rng)
        print_layer_row(s)
        layer_stats.append(s)
        if "skipped" in s:
            continue
        overall["nz_total"] += s["nz_total"]
        overall["nz_pass"]  += s["nz_pass"]
        overall["z_total"]  += s["z_total"]
        overall["z_pass"]   += s["z_pass"]
        overall["nz_err"]   += s["nz_err"]
        overall["z_err"]    += s["z_err"]

    print(f"\n{'-'*100}")
    print(f"  Overall non-zero: {fmt_pct(overall['nz_pass'], overall['nz_total'])}  "
          f"err[{percentiles(overall['nz_err'])}]")
    print(f"  Overall zero:     {fmt_pct(overall['z_pass'],  overall['z_total'])}  "
          f"err[{percentiles(overall['z_err'])}]")
    print(f"{'-'*100}\n")

    cdb.close()
    sdb.close()

    nz_ok = overall["nz_pass"] == overall["nz_total"]
    z_ok  = overall["z_pass"]  == overall["z_total"]
    return 0 if (nz_ok and z_ok) else 1


if __name__ == "__main__":
    sys.exit(main())

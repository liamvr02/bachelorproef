# -*- coding: utf-8 -*-
"""
validate_radius_fix.py
======================
Sanity-check for the UA / WIS radius-duplication and double-prefix bug fixes.

Builds (or reloads) a small parquet that exercises every feature factory at
two radii (50 m and 100 m), then asserts correctness properties.

Checks
------
  1. No double-prefix columns (ua_ua_*, wis_wis_*)
  2. All expected feature columns exist
  3. UA classification fractions differ across radii (the main bug)
  4. WIS fractions differ across radii (same bug, different factory)
  5. Tree-count and planted-by counts are non-decreasing with radius
  6. All *_frac columns are bounded in [0, 1]

Usage
-----
    uv run python validate_radius_fix.py             # load cached parquet + assert
    uv run python validate_radius_fix.py --rebuild   # rebuild from stream, then assert
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import io
import numpy as np
import pandas as pd

# Force UTF-8 stdout on Windows so Unicode symbols print correctly.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from stream.classification_groups import UA, WIS_BESTEMMING, WIS_MATERIAAL
from stream.features import (
    FeatureRegistry,
    aggregate_in_radius,
    trees_count_planted_by,
    urban_atlas_classifications_fractions,
    wis_fraction,
)
from stream.stream import StreamConfig

# -- config --------------------------------------------------------------------

PARQUET        = SRC / "test_data" / "radius_fix_test.parquet"
_RADII         = (50, 100)
_PARTITION_KEY = "2018-07"   # one warm-month partition — quick but non-trivial
_MAX_ROWS      = 2_000

# One representative attribute value from each WIS dimension
_WIS_BESTEMMING_VAL = "Rijweg"          # most common road surface class
_WIS_MATERIAAL_VAL  = "Gebakken klinkers"

# -- registry ------------------------------------------------------------------

def build_registry() -> FeatureRegistry:
    """Minimal registry: every factory type at both radii."""
    reg = FeatureRegistry()
    for r in _RADII:
        # aggregate_in_radius — DHM elevation mean
        reg.add(aggregate_in_radius(
            "dhm", radius_m=r, columns=["elevation"],
            agg="avg", temporal="last_previous",
        ))
        # aggregate_in_radius — trees total count
        reg.add(aggregate_in_radius(
            "trees", radius_m=r, columns=[], agg="count",
            attr_filter=None, temporal="none",
        ))
        # trees_count_planted_by — DiD treatment dose
        reg.add(trees_count_planted_by(
            "trees", radius_m=r, columns=[], agg="count",
            attr_filter=None, temporal="none",
        ))
        # urban_atlas_classifications_fractions — all UA classes
        reg.add(urban_atlas_classifications_fractions(
            classification_map=UA, radius_m=r,
        ))
        # wis_fraction — road surface bestemming
        reg.add(wis_fraction(
            attr_col="bestemming", attr_val=_WIS_BESTEMMING_VAL, radius_m=r,
        ))
        # wis_fraction — road surface materiaal
        reg.add(wis_fraction(
            attr_col="materiaalsoort", attr_val=_WIS_MATERIAAL_VAL, radius_m=r,
        ))
    return reg


# -- rebuild ------------------------------------------------------------------

def rebuild() -> pd.DataFrame:
    print(f"Building stream  partition={_PARTITION_KEY}  max_rows={_MAX_ROWS}")
    reg = build_registry()
    cfg = StreamConfig(partition_keys=[_PARTITION_KEY])
    chunks = [batch for batch in cfg.stream(reg, max_rows=_MAX_ROWS)]
    if not chunks:
        raise RuntimeError("Stream returned no rows — check partition key and data path")
    df = pd.concat(chunks, ignore_index=True)
    PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET, index=False)
    print(f"Saved {len(df):,} rows  {len(df.columns)} cols  →  {PARQUET.name}\n")
    return df


# -- assertion helpers --------------------------------------------------------

_PASS = "\033[32mPASS\033[0m"
_FAIL = "\033[31mFAIL\033[0m"
_failures: list[str] = []


def ok(label: str, condition: bool, detail: str = "") -> bool:
    tag = _PASS if condition else _FAIL
    line = f"  [{tag}] {label}"
    if not condition and detail:
        line += f"\n         {detail}"
    print(line)
    if not condition:
        _failures.append(label)
    return condition


# -- assertion groups ----------------------------------------------------------

def check_no_double_prefix(cols: set[str]) -> None:
    print("-- 1. No double-prefix columns --------------------------------------")
    bad_ua  = sorted(c for c in cols if c.startswith("ua_ua_"))
    bad_wis = sorted(c for c in cols if c.startswith("wis_wis_"))
    ok("No  ua_ua_*  columns", len(bad_ua)  == 0, f"found: {bad_ua[:4]}"  if bad_ua  else "")
    ok("No wis_wis_* columns", len(bad_wis) == 0, f"found: {bad_wis[:4]}" if bad_wis else "")


def check_expected_columns(cols: set[str]) -> None:
    print("\n-- 2. Expected columns present --------------------------------------")
    r_s, r_l = _RADII

    for cls in UA:
        for r in _RADII:
            ok(f"  ua_{cls}_{r}m_frac", f"ua_{cls}_{r}m_frac" in cols)

    safe_b = _WIS_BESTEMMING_VAL.replace(" ", "_").replace("/", "_")
    safe_m = _WIS_MATERIAAL_VAL.replace(" ", "_").replace("/", "_")
    for r in _RADII:
        ok(f"  wis_{safe_b}_{r}m_frac",  f"wis_{safe_b}_{r}m_frac"  in cols)
        ok(f"  wis_{safe_m}_{r}m_frac",  f"wis_{safe_m}_{r}m_frac"  in cols)

    for r in _RADII:
        tree_total   = [c for c in cols if f"trees_count{r}m"    in c and c.endswith("count")]
        tree_planted = [c for c in cols if f"trees_plantedby_{r}" in c and c.endswith("count")]
        ok(f"  trees total count at {r}m",    len(tree_total)   > 0, f"candidates: {tree_total}")
        ok(f"  trees planted-by at {r}m",     len(tree_planted) > 0, f"candidates: {tree_planted}")
        dhm = [c for c in cols if f"dhm_avg{r}m" in c]
        ok(f"  dhm avg at {r}m",              len(dhm) > 0,         f"candidates: {dhm}")


def check_fracs_differ_across_radii(df: pd.DataFrame, cols: set[str]) -> None:
    print("\n-- 3. UA fractions differ across radii (radius-duplication check) --")
    r_s, r_l = _RADII
    for cls in UA:
        c_s = f"ua_{cls}_{r_s}m_frac"
        c_l = f"ua_{cls}_{r_l}m_frac"
        if c_s not in cols or c_l not in cols:
            ok(f"  ua_{cls}: columns exist for comparison", False)
            continue
        max_diff = (df[c_l] - df[c_s]).abs().max()
        mean_s   = df[c_s].mean()
        mean_l   = df[c_l].mean()
        ok(
            f"  ua_{cls}: {r_l}m ≠ {r_s}m  (max|diff|={max_diff:.4f}  "
            f"mean {r_s}m={mean_s:.3f}  {r_l}m={mean_l:.3f})",
            max_diff > 1e-4,
            f"all rows identical — radius not applied to raster lookup",
        )

    print("\n-- 4. WIS fractions differ across radii -----------------------------")
    for safe_val in (
        _WIS_BESTEMMING_VAL.replace(" ", "_").replace("/", "_"),
        _WIS_MATERIAAL_VAL.replace(" ", "_").replace("/", "_"),
    ):
        c_s = f"wis_{safe_val}_{r_s}m_frac"
        c_l = f"wis_{safe_val}_{r_l}m_frac"
        if c_s not in cols or c_l not in cols:
            ok(f"  wis_{safe_val}: columns exist for comparison", False)
            continue
        max_diff = (df[c_l] - df[c_s]).abs().max()
        mean_s   = df[c_s].mean()
        mean_l   = df[c_l].mean()
        ok(
            f"  wis_{safe_val}: {r_l}m ≠ {r_s}m  (max|diff|={max_diff:.4f}  "
            f"mean {r_s}m={mean_s:.3f}  {r_l}m={mean_l:.3f})",
            max_diff > 1e-4,
            f"all rows identical — radius not applied to raster lookup",
        )


def check_counts_nondecreasing(df: pd.DataFrame, cols: set[str]) -> None:
    print("\n-- 5. Tree counts non-decreasing with radius ------------------------")
    r_s, r_l = _RADII
    for prefix_tag in ("trees_count", "trees_plantedby_"):
        c_s = next((c for c in cols if f"{prefix_tag}{r_s}m" in c and c.endswith("count")), None)
        c_l = next((c for c in cols if f"{prefix_tag}{r_l}m" in c and c.endswith("count")), None)
        if c_s is None or c_l is None:
            ok(f"  {prefix_tag}*: count cols found", False,
               f"{r_s}m → {c_s!r}   {r_l}m → {c_l!r}")
            continue
        m_s = df[c_s].mean()
        m_l = df[c_l].mean()
        ok(
            f"  {prefix_tag}*: mean({r_l}m)={m_l:.2f} >= mean({r_s}m)={m_s:.2f}",
            m_l >= m_s - 0.01,
        )


def check_fraction_bounds(df: pd.DataFrame, cols: set[str]) -> None:
    print("\n-- 6. All *_frac columns in [0, 1] ----------------------------------")
    frac_cols = [c for c in cols if c.endswith("_frac")]
    if not frac_cols:
        ok("fraction columns found", False)
        return
    out_of_range = []
    for c in frac_cols:
        s = df[c].dropna()
        if len(s) and (s.min() < -1e-6 or s.max() > 1.0 + 1e-6):
            out_of_range.append(f"{c}: [{s.min():.4f}, {s.max():.4f}]")
    ok(
        f"All {len(frac_cols)} *_frac columns in [0, 1]",
        len(out_of_range) == 0,
        "\n         ".join(out_of_range[:5]) if out_of_range else "",
    )


# -- main ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true",
                    help="Rebuild parquet from stream even if cached file exists")
    ap.add_argument("--parquet", metavar="PATH",
                    help="Load this parquet instead of the default path "
                         "(useful for testing existing datasets without rebuilding)")
    args = ap.parse_args()

    target = Path(args.parquet) if args.parquet else PARQUET

    if args.rebuild or not target.exists():
        if args.parquet:
            ap.error("--parquet and --rebuild cannot be combined; "
                     "--parquet loads an existing file")
        df = rebuild()
        target = PARQUET
    else:
        print(f"Loading {target}")
        df = pd.read_parquet(target)
        print(f"Loaded {len(df):,} rows  {len(df.columns)} cols\n")

    cols = set(df.columns)
    print(f"Columns: {len(cols)}   Rows: {len(df):,}\n")

    check_no_double_prefix(cols)
    check_expected_columns(cols)
    check_fracs_differ_across_radii(df, cols)
    check_counts_nondecreasing(df, cols)
    check_fraction_bounds(df, cols)

    print(f"\n{'=' * 60}")
    if not _failures:
        print("  All checks passed.")
    else:
        print(f"  {len(_failures)} check(s) FAILED:")
        for f in _failures:
            print(f"    • {f}")
    print("=" * 60)
    sys.exit(1 if _failures else 0)


if __name__ == "__main__":
    main()

"""
lst_did_test.py — Staggered DiD on streamed LST × tree-planting panel.

Estimates the causal effect of trees planted within a chosen radius on
LST, using two-way fixed effects (tile + year-month) with cluster-robust
standard errors.  Identification comes from within-tile variation across
LST scenes spanning years.

Stream features
---------------
- trees_count_planted_by(radius)          → treatment dose (DiD)
- urban_atlas_classifications_fractions   → time-varying controls

Static covariates (DHM, WIS) are absorbed by the tile FE and intentionally
omitted.

Run
---
    python lst_did_test.py
    python lst_did_test.py --rows 5_000_000 --radius 50
    python lst_did_test.py --radius 50 70 100   # one DiD per radius

Radius rationale
----------------
LST has a native ~100x100 m pixel.  The default radii are chosen relative to
that pixel rather than to typical street-tree spacing:

  - 50 m  : circle inscribed in the 100 m pixel (≈79% of pixel area).
            Captures dose strictly below the LST measurement footprint.
  - 70 m  : circle that escribes the 100 m square (corners just inside).
            Covers the full pixel plus a thin halo into neighbours.
  - 100 m : superset capturing the LST pixel and its first-ring neighbours
            (≈9 LST pixels in the 200 m bounding box).

Anything below ~50 m is sub-pixel relative to the LST resolution and can
only weaken the treatment dose without adding new identifying variation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

from ml import LSTDiD
from stream.classification_groups import UA
from stream.features import (
    FeatureRegistry,
    trees_count_planted_by,
    urban_atlas_classifications_fractions,
)
from stream.logging_config import configure_logging
from stream_configs.presets import all_rows

_SRC     = Path(__file__).parent
_REPORTS = _SRC / "reports"
_REPORTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Registry — one treatment dose feature per radius + time-varying controls
# ---------------------------------------------------------------------------
def build_registry(radii: List[int]) -> FeatureRegistry:
    reg = FeatureRegistry()
    for r in radii:
        reg.add(trees_count_planted_by(radius_m=r))
    # Urban Atlas classification fractions at 100 m — time-varying controls.
    # UA has 4 surveys (2006/2012/2018/2021), looked up via last_previous.
    reg.add(urban_atlas_classifications_fractions(
        classification_map=UA, radius_m=100,
    ))
    return reg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows",   type=int,   default=2_000_000,
                    help="rows to stream (default 2M)")
    ap.add_argument("--batch",  type=int,   default=100_000)
    ap.add_argument("--radius", type=int,   nargs="+", default=[50, 70, 100],
                    metavar="R",
                    help="treatment radius/radii in metres "
                         "(default 50 70 100, sized to the 100x100m LST pixel: "
                         "50m inscribes, 70m escribes, 100m covers neighbours). "
                         "One DiD fit per radius.")
    ap.add_argument("--max-panel", type=int, default=5_000_000,
                    help="cap on panel rows after eligibility filter "
                         "(tile-stratified subsample if exceeded)")
    ap.add_argument("--event-window", type=int, nargs=2, default=[-5, 10],
                    metavar=("LO", "HI"),
                    help="event-time window in years (default -5 10)")
    ap.add_argument("--seed",   type=int,   default=0)
    args = ap.parse_args()

    configure_logging(level="INFO")

    radii: List[int] = list(args.radius)
    reg = build_registry(radii)
    # DiD needs the natural temporal density per tile — no uniform resampling,
    # which would shred the per-tile panel and weaken FE identification.
    cfg = all_rows(batch_size=args.batch)

    print(f"\n[lst_did_test] DiD radii: {radii}")
    print(f"[lst_did_test] streaming up to {args.rows:,} rows in {args.batch:,}-row batches")
    print(f"[lst_did_test] event window: k ∈ {tuple(args.event_window)} years")
    print(f"[lst_did_test] panel cap: {args.max_panel:,} (tile-stratified)\n")

    overall_t0 = time.perf_counter()
    for r in radii:
        treatment_col = f"trees_plantedby_{r}m_count"
        # UA classifications factory emits double-prefixed columns
        # (existing behaviour: "ua_ua_<cls>_<R>m_frac"); match that here.
        control_cols  = [f"ua_ua_{cls}_100m_frac" for cls in UA.keys()]

        did = LSTDiD(
            outcome_col      = "temperature",
            treatment_col    = treatment_col,
            tile_col         = "tile_id",
            time_col         = None,                 # derive _scene_ym from timestamp
            control_cols     = control_cols,
            event_window     = tuple(args.event_window),
            cluster_on       = "tile_id",
            max_panel_rows   = args.max_panel,
            min_obs_per_tile = 3,
            seed             = args.seed,
        )

        try:
            did.fit(
                source     = cfg,
                registry   = reg,
                batch_size = args.batch,
                max_rows   = args.rows,
                verbose    = True,
            )
        except Exception as exc:
            print(f"\n[lst_did_test] DiD r={r}m FAILED: {exc}", file=sys.stderr)
            continue

        text = did.report()
        print("\n" + text)

        out_html = _REPORTS / f"did_r{r}m.html"
        did.report(out_path=out_html)
        print(f"\n[lst_did_test] HTML report: {out_html}")

    print(f"\n[lst_did_test] all radii done in {time.perf_counter() - overall_t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

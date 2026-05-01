"""
lst_models_test.py — quick model smoke test on a small streamed slice.

Streams a configurable number of rows directly through ``train_all`` —
no full-DataFrame materialisation, so peak RAM stays at ~one batch even
when ``--rows`` is very large.

Run
---
    python lst_models_test.py
    python lst_models_test.py --rows 500_000 --batch 50_000
    python lst_models_test.py --rows -1                # full stream
    python lst_models_test.py --csv sample_stream_output.csv  # legacy CSV path
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Generator, List, Optional

import pandas as pd
from tqdm.auto import tqdm

from ml import train_all, cyclical
from stream.classification_groups import UA, WIS_BESTEMMING
from stream.features import (
    FeatureRegistry,
    aggregate_in_radius,
    urban_atlas_classifications_fractions,
    wis_fraction,
)
from stream.logging_config import configure_logging
from stream_configs.presets import all_rows

_SRC      = Path(__file__).parent
_REPORTS  = _SRC / "reports"
_REPORTS.mkdir(exist_ok=True)


def build_registry() -> FeatureRegistry:
    """Compact registry — same shape as lst_sanity_test for fast smoke runs."""
    reg = FeatureRegistry()

    for r in (50, 100):
        for agg in ("avg", "max"):
            reg.add(aggregate_in_radius(
                "dhm", radius_m=r, columns=["elevation"],
                agg=agg, temporal="last_previous",
            ))

    for r in (50, 100):
        reg.add(aggregate_in_radius(
            "trees", radius_m=r, columns=[], agg="count", temporal="none",
        ))

    reg.add(urban_atlas_classifications_fractions(
        classification_map=UA, radius_m=100,
    ))

    for leaves in WIS_BESTEMMING.values():
        for val in leaves:
            reg.add(wis_fraction(attr_col="bestemming", attr_val=val, radius_m=50))

    return reg


def _df_chunks(df: pd.DataFrame, batch_size: int) -> Generator[pd.DataFrame, None, None]:
    """Yield consecutive slices of *df* without materialising index copies."""
    for start in range(0, len(df), batch_size):
        yield df.iloc[start:start + batch_size]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows",  type=int, default=200_000,
                    help="rows to stream (default 200k; -1 = full stream)")
    ap.add_argument("--batch", type=int, default=50_000)
    ap.add_argument("--csv",   type=str, default=None,
                    help="optional path to a pre-streamed CSV; if given, the "
                         "stream pipeline is skipped and rows come from the file")
    args = ap.parse_args()

    configure_logging(level="INFO")

    transforms = [
        cyclical("hour_of_day",   24),
        cyclical("day_of_year",   365),
        cyclical("month_of_year", 12),
    ]

    rows = None if args.rows == -1 else args.rows

    # ---- Eval split: grab a small final slice of the source ----
    # For the streaming path we hold back the *last* batch worth of rows as
    # an in-distribution eval set.  For the CSV path we keep the legacy 95/5 split.
    if args.csv is not None:
        source = pd.read_csv(args.csv)
        if rows is not None:
            source = source.head(rows)
        split = int(len(source) * 0.95)
        tr_df = source.iloc[:split]
        ev_df = source.iloc[split:]
        print(f"[csv] {args.csv} → train {len(tr_df):,}  eval {len(ev_df):,}")

        results = train_all(
            tr_df,
            transforms = transforms,
            batch_size = args.batch,
            max_rows   = len(tr_df),
        )
        eval_for_report = ev_df
    else:
        # Streaming source — train_all handles fit_stream end-to-end.
        # Hold back ~5% of the cap (or one batch, whichever is larger) for eval.
        eval_cap_rows = max(args.batch, int((rows or 200_000) * 0.05))
        cfg = all_rows(batch_size=args.batch)

        # Stream a small dedicated eval set first (separate config so it
        # doesn't consume the training stream).
        eval_cfg = all_rows(batch_size=args.batch)
        reg      = build_registry()

        eval_batches: List[pd.DataFrame] = []
        bar = tqdm(total=eval_cap_rows, desc="stream:eval", unit="row")
        for df in eval_cfg.stream(reg, batch_size=args.batch, max_rows=eval_cap_rows):
            eval_batches.append(df)
            bar.update(len(df))
        bar.close()
        eval_for_report = (
            pd.concat(eval_batches, ignore_index=True)
            if eval_batches else pd.DataFrame()
        )
        print(f"[stream] eval slice: {len(eval_for_report):,} rows")

        train_max = (rows - len(eval_for_report)) if rows is not None else None
        results = train_all(
            source     = cfg,
            registry   = reg,
            transforms = transforms,
            batch_size = args.batch,
            max_rows   = train_max,
        )

    # ---- Reports ----
    for name, model in results.items():
        model.report("stdout",
                     eval_df=eval_for_report if not eval_for_report.empty else None)
        model.report(
            "html",
            path    = _REPORTS / f"{name}.html",
            eval_df = eval_for_report if not eval_for_report.empty else None,
            n_shap  = 1000,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

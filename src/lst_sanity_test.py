"""
lst_sanity_test.py — exercise every ml.sanity check on a small representative stream.

Trains the fast linear-family models (huber, linear, ridge, elastic_net, sgd,
nystroem_sgd) on a slice of the streaming pipeline with sanity checks
enabled.  Each model run exercises the four sanity hooks at least once:

    1. check_scaler_alignment        — once at start of fit_stream
    2. check_input_batch             — every batch in fit_stream
    3. check_post_step               — every batch in fit_stream (linear models only)
    4. check_predictions             — once per model in evaluate()

A summary at the end groups every captured sanity record by model.

Memory contract
---------------
``train_all`` consumes the StreamConfig directly via ``fit_stream`` — no
``pd.concat`` over the whole stream — so this script can be pointed at the
full Ghent stream with ``--rows -1`` without OOM.  A small eval slice is
materialised separately to drive ``check_predictions``.

Run:
    python lst_sanity_test.py [--rows N | -1] [--strict] [--models a,b,c]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm.auto import tqdm

from ml import StreamingScaler, sanity, train_all, cyclical
from stream.classification_groups import UA, WIS_BESTEMMING
from stream.features import (
    FeatureRegistry,
    aggregate_in_radius,
    urban_atlas_classifications_fractions,
    wis_fraction,
)
from stream.logging_config import configure_logging
from stream_configs.presets import all_rows


_FAST_MODELS = ["huber", "linear", "ridge", "elastic_net", "sgd", "nystroem_sgd"]


# ---------------------------------------------------------------------------
# Sanity log capture
# ---------------------------------------------------------------------------
class _SanityCapture(logging.Handler):
    """Append every sanity record to a flat list, grouped on read."""
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("lst_models.sanity"):
            self.records.append(record)

    def grouped(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = defaultdict(list)
        for r in self.records:
            out[r.levelname].append(r.getMessage())
        return out


# ---------------------------------------------------------------------------
# Tiny but representative registry
# ---------------------------------------------------------------------------
def build_registry() -> FeatureRegistry:
    reg = FeatureRegistry()

    # DHM aggregates — covers numeric scalar features.
    for r in (50, 100):
        for agg in ("avg", "max"):
            reg.add(aggregate_in_radius(
                "dhm", radius_m=r, columns=["elevation"],
                agg=agg, temporal="last_previous",
            ))

    # Tree count — covers integer counts.
    for r in (50, 100):
        reg.add(aggregate_in_radius(
            "trees", radius_m=r, columns=[], agg="count", temporal="none",
        ))

    # UA classifications — temporal fraction features.
    reg.add(urban_atlas_classifications_fractions(
        classification_map=UA, radius_m=100,
    ))

    # WIS bestemming — static fraction features (one radius is enough).
    for leaves in WIS_BESTEMMING.values():
        for val in leaves:
            reg.add(wis_fraction(attr_col="bestemming", attr_val=val, radius_m=50))

    return reg


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_summary(capture: _SanityCapture) -> int:
    grouped = capture.grouped()
    n_err   = len(grouped.get("ERROR",   []))
    n_warn  = len(grouped.get("WARNING", []))
    n_info  = len(grouped.get("INFO",    []))

    bar = "─" * 78
    print(f"\n{bar}")
    print(f"  Sanity check summary")
    print(f"  records: {n_err} error, {n_warn} warning, {n_info} info")
    print(f"{bar}")

    for level in ("ERROR", "WARNING", "INFO"):
        msgs = grouped.get(level, [])
        if not msgs:
            continue
        print(f"\n  {level} ({len(msgs)})")
        for m in msgs:
            print(f"    {m}")

    print(f"\n{bar}\n")
    return 0 if n_err == 0 else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows",   type=int, default=200_000,
                    help="rows to stream (default 200k; -1 = full stream)")
    ap.add_argument("--batch",  type=int, default=50_000)
    ap.add_argument("--strict", action="store_true",
                    help="raise SanityCheckFailure on any error (otherwise log+continue)")
    ap.add_argument("--models", type=str,
                    default=",".join(_FAST_MODELS),
                    help=f"comma-separated subset of {_FAST_MODELS}")
    args = ap.parse_args()

    configure_logging(level="INFO")

    capture = _SanityCapture()
    logging.getLogger("lst_models.sanity").addHandler(capture)
    sanity.enable(strict=args.strict)

    rows = None if args.rows == -1 else args.rows
    reg  = build_registry()

    # ---- Eval slice: stream a small dedicated chunk to drive check_predictions ----
    eval_cap = max(args.batch, int((rows or 200_000) * 0.10))
    eval_cfg = all_rows(batch_size=args.batch)
    eval_batches: List[pd.DataFrame] = []
    bar = tqdm(total=eval_cap, desc="stream:eval", unit="row")
    for df in eval_cfg.stream(reg, batch_size=args.batch, max_rows=eval_cap):
        eval_batches.append(df)
        bar.update(len(df))
    bar.close()
    if not eval_batches:
        print("no rows streamed for eval — aborting", file=sys.stderr)
        return 2
    ev = pd.concat(eval_batches, ignore_index=True)
    print(ev.describe().T)
    print(f"\neval: {len(ev):,} rows | features in df: {ev.shape[1]}\n")

    # ---- Train: pass StreamConfig directly to train_all (no full materialisation) ----
    transforms = [
        cyclical("hour_of_day",   24),
        cyclical("day_of_year",   365),
        cyclical("month_of_year", 12),
    ]
    chosen = [m.strip() for m in args.models.split(",") if m.strip()]

    train_max = (rows - len(ev)) if rows is not None else None
    train_cfg = all_rows(batch_size=args.batch)

    t0 = time.perf_counter()
    results = train_all(
        source     = train_cfg,
        registry   = reg,
        transforms = transforms,
        batch_size = args.batch,
        max_rows   = train_max,
        models     = chosen,
    )
    t_train = time.perf_counter() - t0
    print(f"\n[train_all] {len(results)}/{len(chosen)} models in {t_train:.1f}s")

    # Trigger the predictions check (separate from training-time eval inside fit_stream)
    print("\n[evaluate] running held-out evaluation to exercise check_predictions")
    for name, model in results.items():
        try:
            metrics = model.evaluate(ev, batch_size=args.batch)
            print(f"  {name:<14s}  RMSE={metrics.get('rmse', float('nan')):.3f}  "
                  f"R²={metrics.get('r2', float('nan')):.3f}")
        except sanity.SanityCheckFailure as exc:
            print(f"  {name:<14s}  STRICT FAIL — {exc}")
        except Exception as exc:
            print(f"  {name:<14s}  evaluate raised: {exc}")

    return print_summary(capture)


if __name__ == "__main__":
    sys.exit(main())

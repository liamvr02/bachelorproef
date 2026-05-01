"""
lst_grid_search.py — hyperparameter grid search across LST models.

Streams a representative slice once with a uniform year/month/hour
distribution, **caches it to parquet** so memory stays bounded by batch
size regardless of how many rows were requested, then trains every
(model × param-combo) on the same on-disk train/eval split so cross-model
comparison is fair.

Memory contract
---------------
Earlier versions materialised the entire streamed sample into a single
pandas DataFrame via ``pd.concat([...])`` and then trained every combo on
it.  Past ~1.5 M rows this hit hard OOM.  The current implementation
writes one parquet shard per stream and reads it back via
``pyarrow.ParquetFile.iter_batches``, so peak RAM is ~``batch_size`` rows.

Cache layout
------------
    src/stream_cache/grid_search/
        train.parquet   (uniform-distribution training rows, outliers excluded)
        eval.parquet    (concatenated outlier-month rows)
        manifest.json   (registry shape + row counts; used to validate cache)

Pass ``--rebuild-cache`` to force a re-stream.

Per-model grids
---------------
Two presets:

  --quick (default)
      Compact, hand-picked grids — typically 3-12 combos per model.
      Designed to finish in roughly 10-20 minutes per million training rows.

  --full
      Use each model's ``default_hyperparameter_grid`` property.

Run
---
    python lst_grid_search.py
    python lst_grid_search.py --rows 5_000_000
    python lst_grid_search.py --only linear ridge xgboost
    python lst_grid_search.py --only random_forest extra_trees --rebuild-cache
    python lst_grid_search.py --rows -1                # full stream
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ml import StreamingScaler, cyclical
from ml.base import NEVER_FEATURES
from ml.registry import ModelRegistry
from ml.transforms import remove
from stream.classification_groups import UA, WIS_BESTEMMING, WIS_MATERIAAL
from stream.features import (
    FeatureRegistry,
    aggregate_in_radius,
    urban_atlas_classifications_fractions,
    wis_fraction,
)
from stream.logging_config import configure_logging
from stream.stream import StreamConfig
from stream_configs.outliers import outlier_keys
from stream_configs.presets import representative


_SRC      = Path(__file__).parent
_REPORTS  = _SRC / "reports"
_CACHE    = _SRC / "stream_cache" / "grid_search"
_REPORTS.mkdir(exist_ok=True)
_CACHE.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Grid presets — every key below MUST exist as a constructor kwarg of the
# corresponding LST<Model> wrapper.  Don't add untested params here; the grid
# search just spawns failed combos for any unknown kwarg.
# ---------------------------------------------------------------------------
_QUICK_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    # ── Linear family ──
    # All five SGD-based models hit the same ~5.22 RMSE ceiling regardless
    # of regularisation — the features just aren't linearly separable.  Keep
    # them as cheap baselines (one combo each).
    "linear":      {"alpha": [1e-4],                                "penalty": ["l2"]},
    "ridge":       {"alpha": [1e-4]},
    "elastic_net": {"alpha": [1e-4], "l1_ratio": [0.5]},
    "huber":       {"alpha": [1e-4], "epsilon":  [1.75]},
    "sgd":         {"alpha": [1e-4], "learning_rate": ["invscaling"], "eta0": [0.01]},

    # ── Nystroem-SGD ──
    # n_components and gamma are the only knobs that move the needle.
    "nystroem_sgd": {
        "n_components": [300, 500],
        "alpha":        [1e-4],
        "gamma":        [0.05, 0.1],
    },

    # ── XGBoost ──
    # All keys exposed on LSTXGBoost — no silent failures.
    # Compact: 12 combos.  Sweeps depth × LR × L2 × min_child × split-gain.
    "xgboost": {
        "n_estimators_per_batch": [50],
        "max_depth":              [8, 12],
        "learning_rate":          [0.05, 0.1],
        "subsample":              [0.8],
        "colsample_bytree":       [0.8],
        "reg_lambda":             [1.0, 5.0],
        "min_child_weight":       [1, 5],
        "gamma":                  [0.0, 0.1],
    },

    # ── HistGradientBoosting ──
    # Reservoir-fit (single sklearn fit per combo); ~5-10 min each on 1M rows.
    # All keys exposed on LSTHistGradientBoosting.
    "hist_gb": {
        "max_iter":            [600, 1000],
        "learning_rate":       [0.05, 0.1],
        "max_depth":           [None, 10],
        "min_samples_leaf":    [10, 20],
        "l2_regularization":   [0.0, 1.0],
        "early_stopping":      [True],
        "validation_fraction": [0.1],
    },

    # ── Random Forest ──
    "random_forest": {
        "n_estimators":     [400],
        "min_samples_leaf": [1, 3],
        "max_depth":        [None],
    },

    # ── Extra Trees ──
    "extra_trees": {
        "n_estimators":     [200],
        "min_samples_leaf": [1],
        "max_depth":        [None],
    },
}

# Default model set — every registered model.  Override with --only.
_DEFAULT_MODELS = [
    "linear", "ridge", "elastic_net", "huber", "sgd", "nystroem_sgd",
    "xgboost", "hist_gb", "random_forest", "extra_trees",
]


# ---------------------------------------------------------------------------
# Feature registry — same shape as lst_full_test, slightly trimmed
# ---------------------------------------------------------------------------
def build_registry() -> FeatureRegistry:
    reg = FeatureRegistry()

    for r in (50, 100):
        for agg in ("avg", "max", "min"):
            reg.add(aggregate_in_radius(
                "dhm", radius_m=r, columns=["elevation"],
                agg=agg, temporal="last_previous",
            ))

    for r in (50, 100):
        for beheerfase in ("Jeugdfase", "Volwassenfase", "Veteranenfase"):
            reg.add(aggregate_in_radius(
                "trees", radius_m=r, columns=[], agg="count",
                attr_filter={"beheerfase": beheerfase}, temporal="none",
            ))

    for r in (50, 100):
        reg.add(urban_atlas_classifications_fractions(
            classification_map=UA, radius_m=r,
        ))

    for r in (50, 100):
        for leaves in WIS_BESTEMMING.values():
            for val in leaves:
                reg.add(wis_fraction(
                    attr_col="bestemming", attr_val=val, radius_m=r,
                ))
        for leaves in WIS_MATERIAAL.values():
            for val in leaves:
                reg.add(wis_fraction(
                    attr_col="materiaalsoort", attr_val=val, radius_m=r,
                ))

    return reg


# ---------------------------------------------------------------------------
# Stream → parquet (single pass, bounded memory)
# ---------------------------------------------------------------------------
def _registry_fingerprint(reg: FeatureRegistry) -> str:
    """
    Stable identifier for a registry's shape.

    Uses descriptor names, which already encode dataset/agg/radius/attr_filter
    suffixes — sufficient to detect any meaningful registry change.
    """
    names = sorted(d.name for d in reg._descriptors)
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()[:16]


def _stream_to_parquet(
    cfg:        StreamConfig,
    reg:        FeatureRegistry,
    max_rows:   Optional[int],
    batch_size: int,
    path:       Path,
    label:      str,
) -> int:
    """
    Stream rows from *cfg* and append to a parquet file at *path*.

    Uses pyarrow.parquet.ParquetWriter for incremental append — memory stays
    bounded by ``batch_size`` rows.  Returns the total number of rows written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if path.exists():
        path.unlink()

    writer: Optional[pq.ParquetWriter] = None
    n_total = 0
    bar = tqdm(total=max_rows, desc=f"stream:{label}", unit="row",
               dynamic_ncols=True)
    try:
        for df in cfg.stream(reg, batch_size=batch_size, max_rows=max_rows):
            if df.empty:
                continue
            tbl = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            n_total += len(df)
            bar.update(len(df))
    finally:
        if writer is not None:
            writer.close()
        bar.close()
    return n_total


def _pq_iter(path: Path, batch_size: int) -> Generator[pd.DataFrame, None, None]:
    """Yield parquet record batches as pandas DataFrames, ~``batch_size`` rows each."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for rb in pf.iter_batches(batch_size=batch_size):
        yield rb.to_pandas()


def _pq_n_rows(path: Path) -> int:
    """Cheap row count without reading the data."""
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).metadata.num_rows


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------
def _manifest_path() -> Path:
    return _CACHE / "manifest.json"


def _read_manifest() -> Optional[Dict[str, Any]]:
    p = _manifest_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _write_manifest(d: Dict[str, Any]) -> None:
    _manifest_path().write_text(json.dumps(d, indent=2, default=str))


def _ensure_cache(
    reg:           FeatureRegistry,
    rows:          Optional[int],
    outlier_rows:  Optional[int],
    batch_size:    int,
    rebuild:       bool,
) -> Tuple[Path, Path]:
    """
    Return (train_pq, eval_pq), streaming if cache is missing/invalid.

    Cache validity = manifest exists, registry fingerprint matches, and the
    requested row counts are <= the cached row counts (so a smaller --rows
    re-uses a larger existing cache).
    """
    train_pq = _CACHE / "train.parquet"
    eval_pq  = _CACHE / "eval.parquet"

    fp        = _registry_fingerprint(reg)
    manifest  = _read_manifest()
    have_pq   = train_pq.exists() and eval_pq.exists()
    cache_ok  = (
        not rebuild
        and have_pq
        and manifest is not None
        and manifest.get("registry_fp") == fp
        and (rows         is None or manifest.get("train_rows", 0) >= rows)
        and (outlier_rows is None or manifest.get("eval_rows",  0) >= outlier_rows)
    )

    if cache_ok:
        print(f"[cache] reusing {train_pq.name} ({manifest['train_rows']:,} rows) "
              f"and {eval_pq.name} ({manifest['eval_rows']:,} rows)")
        return train_pq, eval_pq

    if rebuild and have_pq:
        print("[cache] --rebuild-cache: discarding existing parquet shards")
    elif manifest is None or manifest.get("registry_fp") != fp:
        print("[cache] registry fingerprint changed or missing — restreaming")
    else:
        print("[cache] insufficient cached rows for requested run — restreaming")

    excluded_keys, outlier_keys_by_label = outlier_keys()

    # ---- Training stream (uniform distribution, outliers excluded) ----
    cfg = representative(excluded_keys=excluded_keys, batch_size=batch_size)
    n_train = _stream_to_parquet(cfg, reg, rows, batch_size, train_pq, "train")

    # ---- Eval stream: concatenate every outlier split into one parquet ----
    if eval_pq.exists():
        eval_pq.unlink()
    n_eval = 0
    import pyarrow as pa
    import pyarrow.parquet as pq
    writer: Optional[pq.ParquetWriter] = None
    try:
        for label, keys in outlier_keys_by_label.items():
            ocfg = StreamConfig(partition_keys=keys, batch_size=batch_size)
            bar = tqdm(total=outlier_rows, desc=f"stream:{label}", unit="row",
                       dynamic_ncols=True, leave=False)
            for df in ocfg.stream(reg, batch_size=batch_size, max_rows=outlier_rows):
                if df.empty:
                    continue
                # Tag rows with their outlier label so we can stratify later
                df = df.assign(_outlier_label=label)
                tbl = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(eval_pq, tbl.schema, compression="zstd")
                writer.write_table(tbl)
                n_eval += len(df)
                bar.update(len(df))
            bar.close()
    finally:
        if writer is not None:
            writer.close()

    _write_manifest({
        "registry_fp":       fp,
        "train_rows":        n_train,
        "eval_rows":         n_eval,
        "outlier_labels":    list(outlier_keys_by_label.keys()),
        "outlier_per_label": outlier_rows,
        "batch_size":        batch_size,
    })
    print(f"[cache] wrote {train_pq} ({n_train:,} rows), "
          f"{eval_pq} ({n_eval:,} rows)")
    return train_pq, eval_pq


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
@dataclass
class _Result:
    model:    str
    params:   Dict[str, Any]
    rmse:     float
    mae:      float
    r2:       float
    fit_s:    float
    eval_s:   float
    n_train:  int
    n_eval:   int
    error:    Optional[str] = None


def _expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not grid:
        return [{}]
    keys, values = zip(*grid.items())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _fit_one(
    model_name: str,
    params:     Dict[str, Any],
    train_pq:   Path,
    eval_pq:    Path,
    n_train:    int,
    n_eval:     int,
    transforms: List[Any],
    scaler:     StreamingScaler,
    batch_size: int,
) -> _Result:
    klass = ModelRegistry.get(model_name)
    try:
        model = klass(**params)
        model.set_transforms(transforms)
        model.set_scaler(scaler)

        t0 = time.perf_counter()
        model.fit_stream(
            _pq_iter(train_pq, batch_size),
            max_rows=n_train, verbose=False,
        )
        fit_s  = time.perf_counter() - t0

        t0 = time.perf_counter()
        m  = model.evaluate(
            _pq_iter(eval_pq, batch_size),
            batch_size=batch_size,
        )
        eval_s = time.perf_counter() - t0

        return _Result(
            model   = model_name,
            params  = params,
            rmse    = float(m.get("rmse",  float("nan"))),
            mae     = float(m.get("mae",   float("nan"))),
            r2      = float(m.get("r2",    float("nan"))),
            fit_s   = fit_s,
            eval_s  = eval_s,
            n_train = n_train,
            n_eval  = n_eval,
        )
    except Exception as exc:
        return _Result(
            model = model_name, params = params,
            rmse  = float("inf"), mae = float("inf"), r2 = float("-inf"),
            fit_s = 0.0, eval_s = 0.0, n_train = n_train, n_eval = n_eval,
            error = f"{type(exc).__name__}: {exc}",
        )


def grid_search_one_model(
    model_name: str,
    grid:       Dict[str, List[Any]],
    train_pq:   Path,
    eval_pq:    Path,
    n_train:    int,
    n_eval:     int,
    transforms: List[Any],
    scaler:     StreamingScaler,
    batch_size: int,
) -> List[_Result]:
    combos = _expand_grid(grid)
    results: List[_Result] = []
    bar = tqdm(combos, desc=f"  {model_name:<14s}", unit="combo", leave=False)
    for params in bar:
        res = _fit_one(model_name, params, train_pq, eval_pq,
                       n_train, n_eval, transforms, scaler, batch_size)
        results.append(res)
        if res.error:
            bar.set_postfix(err=res.error[:30], refresh=False)
        else:
            bar.set_postfix(rmse=f"{res.rmse:.3f}", r2=f"{res.r2:.3f}",
                            refresh=False)
    bar.close()
    results.sort(key=lambda r: (r.rmse, -r.r2))
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt_params(p: Dict[str, Any]) -> str:
    if not p:
        return "(defaults)"
    return ", ".join(f"{k}={v}" for k, v in p.items())


def print_per_model(model_name: str, results: List[_Result], top_n: int) -> None:
    print(f"\n{'─' * 78}")
    print(f"  {model_name}  —  top {min(top_n, len(results))} of {len(results)} combos")
    print(f"{'─' * 78}")
    print(f"  {'#':>2}  {'RMSE':>7}  {'MAE':>7}  {'R²':>7}  {'fit s':>6}  params")
    for i, r in enumerate(results[:top_n], 1):
        if r.error:
            print(f"  {i:>2}  {'FAIL':>7}  {'':>7}  {'':>7}  {'':>6}  "
                  f"{_fmt_params(r.params)}  ← {r.error}")
        else:
            print(f"  {i:>2}  {r.rmse:>7.4f}  {r.mae:>7.4f}  {r.r2:>7.4f}  "
                  f"{r.fit_s:>6.1f}  {_fmt_params(r.params)}")


def print_leaderboard(best_per_model: Dict[str, _Result]) -> None:
    bar = "═" * 78
    print(f"\n{bar}")
    print("  Cross-model leaderboard (best params per model, ranked by RMSE)")
    print(bar)
    print(f"  {'Model':<14}  {'RMSE':>7}  {'MAE':>7}  {'R²':>7}  best params")
    print("  " + "─" * 76)
    ranked = sorted(best_per_model.values(), key=lambda r: (r.rmse, -r.r2))
    for r in ranked:
        if r.error:
            print(f"  {r.model:<14}  {'FAIL':>7}  {'':>7}  {'':>7}  "
                  f"← {r.error}")
        else:
            print(f"  {r.model:<14}  {r.rmse:>7.4f}  {r.mae:>7.4f}  "
                  f"{r.r2:>7.4f}  {_fmt_params(r.params)}")
    print(bar)


def write_json_report(all_results: Dict[str, List[_Result]], path: Path) -> None:
    payload = {
        m: [
            {
                "params":  r.params,
                "rmse":    r.rmse,
                "mae":     r.mae,
                "r2":      r.r2,
                "fit_s":   r.fit_s,
                "eval_s":  r.eval_s,
                "n_train": r.n_train,
                "n_eval":  r.n_eval,
                "error":   r.error,
            }
            for r in results
        ]
        for m, results in all_results.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  full results written to {path}")


# ---------------------------------------------------------------------------
# Scaler fitting (parquet-streamed)
# ---------------------------------------------------------------------------
def _fit_scaler_from_parquet(
    train_pq:   Path,
    n_train:    int,
    transforms: List[Any],
    auto_hints: Dict[str, str],
    batch_size: int,
) -> StreamingScaler:
    scaler = StreamingScaler(
        default_scaler = "standard",
        column_scalers = auto_hints,
        transforms     = transforms,
    )
    bar = tqdm(total=n_train, desc="[scaler] fitting", unit="row",
               leave=False, dynamic_ncols=True)
    for chunk in _pq_iter(train_pq, batch_size):
        # Apply transforms (same as model pipeline) before partial_fit
        chunk_t = chunk.copy()
        for fn in transforms:
            try:
                extra = fn(chunk_t)
                if isinstance(extra, pd.DataFrame):
                    for c in extra.columns:
                        chunk_t[c] = extra[c].values
            except Exception:
                pass
        cols = [c for c in chunk_t.columns
                if c not in NEVER_FEATURES
                and pd.api.types.is_numeric_dtype(chunk_t[c])]
        scaler.partial_fit(chunk_t, cols)
        bar.update(len(chunk))
    bar.close()
    return scaler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows",  type=int, default=1_500_000,
                    help="rows to stream for training (default 1.5M; "
                         "pass -1 for the entire stream)")
    ap.add_argument("--batch", type=int, default=100_000)
    ap.add_argument("--outlier-rows", type=int, default=200_000,
                    help="max rows per outlier split; all are concatenated "
                         "into a single eval parquet (default 200 000; -1 = unlimited)")
    ap.add_argument("--only", nargs="+", default=None, metavar="MODEL",
                    help="subset of models to run (space-separated); "
                         "default = every registered model")
    ap.add_argument("--full", action="store_true",
                    help="use each model's default_hyperparameter_grid "
                         "(many more combos)")
    ap.add_argument("--top", type=int, default=5,
                    help="top-N combos to print per model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="ignore any existing train/eval parquet cache and re-stream")
    ap.add_argument("--report",
                    type=str, default=str(_REPORTS / "grid_search_results.json"),
                    help="JSON output path for full results")
    args = ap.parse_args()

    configure_logging(level="INFO")
    # Quiet the per-model "resolved feature columns" info spam — one line per
    # combo would otherwise drown out tqdm.
    logging.getLogger("lst_models").setLevel(logging.WARNING)

    rows         = None if args.rows         == -1 else args.rows
    outlier_rows = None if args.outlier_rows == -1 else args.outlier_rows

    # ---- Resolve which models to run ----
    available = ModelRegistry.available()
    chosen = args.only if args.only else [m for m in _DEFAULT_MODELS
                                          if m in available]
    unknown = [m for m in chosen if m not in available]
    if unknown:
        print(f"unknown models: {unknown}\nregistered: {available}",
              file=sys.stderr)
        return 2

    # ---- Build registry + ensure parquet cache ----
    reg = build_registry()

    excluded_keys, outlier_keys_by_label = outlier_keys()
    print(f"Outlier months excluded from training: {len(excluded_keys)}")
    print(f"Outlier eval splits: {len(outlier_keys_by_label)}")

    train_pq, eval_pq = _ensure_cache(
        reg          = reg,
        rows         = rows,
        outlier_rows = outlier_rows,
        batch_size   = args.batch,
        rebuild      = args.rebuild_cache,
    )

    # If the cache holds more rows than this run wants, cap row counts so the
    # streaming progress bars are accurate.
    n_train = min(_pq_n_rows(train_pq), rows) if rows is not None else _pq_n_rows(train_pq)
    n_eval  = _pq_n_rows(eval_pq)
    print(f"\ntrain {n_train:,} rows  |  eval (outliers) {n_eval:,} rows\n")

    # ---- Build a shared StreamingScaler once ----
    transforms = [
        cyclical("hour_of_day",   24),
        cyclical("day_of_year",   365),
        remove("month_of_year"),
        remove("day_of_month"),
    ]
    auto_hints = reg.column_scaler_hints()

    print(f"  fitting StreamingScaler  ({len(auto_hints)} auto-minmax columns)")
    scaler = _fit_scaler_from_parquet(
        train_pq, n_train, transforms, auto_hints, args.batch,
    )
    print(f"  scaler ready: {scaler}")

    # ---- Iterate every (model, combo) ----
    all_results:    Dict[str, List[_Result]] = {}
    best_per_model: Dict[str, _Result]       = {}

    print(f"\n{'═' * 78}")
    print(f"  Grid search: {len(chosen)} models, "
          f"preset={'full' if args.full else 'quick'}")
    print(f"{'═' * 78}")

    t_total = time.perf_counter()
    for model_name in chosen:
        klass = ModelRegistry.get(model_name)
        if args.full:
            grid = klass(random_state=args.seed).default_hyperparameter_grid
        else:
            grid = _QUICK_GRIDS.get(model_name, {})

        n_combos = len(_expand_grid(grid))
        print(f"\n  ▸ {model_name}  ({n_combos} combos)")
        results = grid_search_one_model(
            model_name = model_name,
            grid       = grid,
            train_pq   = train_pq,
            eval_pq    = eval_pq,
            n_train    = n_train,
            n_eval     = n_eval,
            transforms = transforms,
            scaler     = scaler,
            batch_size = args.batch,
        )
        all_results[model_name] = results
        # Best successful result for this model — pick first non-error
        best = next((r for r in results if r.error is None), results[0])
        best_per_model[model_name] = best

        print_per_model(model_name, results, args.top)

    # ---- Final reports ----
    print_leaderboard(best_per_model)
    print(f"\n  total wall time: {time.perf_counter() - t_total:.1f}s")

    write_json_report(all_results, Path(args.report))
    return 0


if __name__ == "__main__":
    sys.exit(main())

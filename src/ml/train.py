"""
ml/train.py
===========
train_all — fit every registered model on the same data source.

Design
------
Models split into two categories based on whether they can truly stream:

  STREAMING models  (XGBoost, HGB, all SGD variants, NystroemSGD):
      Consume a generator directly via fit_stream().  Data flows through
      one batch at a time; nothing is held in memory beyond the current batch
      and the model's own state.

  MEMORY-BOUNDED models  (RandomForest, ExtraTrees):
      Cannot meaningfully accumulate knowledge across separate fit() calls.
      These receive the stream via their own overridden fit_stream() which
      performs reservoir sampling internally, sizing the reservoir to
      available RAM automatically.

Shared StreamingScaler
----------------------
Before any model is trained, a StreamingScaler is fitted on the full stream
(or loaded from cache if the configuration fingerprint matches a previous run).
Every model then receives the same fitted scaler via model.set_scaler().

The fingerprint includes:
  - sorted feature columns
  - per-column scaler type assignments
  - transform function names
  - stream distribution dimensions (if a StreamConfig is provided)

This ensures that if you change transforms or the distribution target, the
cache is invalidated and the scaler is rebuilt.

Usage
-----
    from ml.train import train_all
    from ml.transforms import cyclical, log1p_transform
    from ml.scaler import StreamingScaler
    from stream import StreamConfig
    from features import FeatureRegistry, nearest
    from config import get_dimension_edges

    cfg = StreamConfig()
    cfg.set_distribution({
        "year":        ({}, get_dimension_edges("year")),
        "month_of_year": ({}, get_dimension_edges("month_of_year")),
        "hour_of_day":   ({}, get_dimension_edges("hour_of_day")),
    })

    reg = FeatureRegistry()
    reg.add(nearest("dhm1", ["elevation"], temporal="last_previous"))

    transforms = [
        cyclical("month_of_year", 12),
        cyclical("hour_of_day",   24),
        log1p_transform(["dhm1_elevation"]),
    ]

    models = train_all(
        source        = cfg,
        registry      = reg,
        transforms    = transforms,
        max_rows      = 2_000_000,
        models        = ["xgboost", "hist_gb", "random_forest"],
        cache_dir     = "prepared_stream_data/scaler_cache",
    )
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ml.base import LSTModel, NEVER_FEATURES
from ml.registry import ModelRegistry
from ml.scaler import StreamingScaler

log = logging.getLogger("lst_models.train")

# Models that consume a stream batch-by-batch via fit_stream()
# (any model not in this set uses its own fit_stream override,
#  e.g. forest.py's reservoir strategy)
_STREAMING_MODEL_NAMES = frozenset({
    "linear", "ridge", "elastic_net", "huber", "sgd", "nystroem_sgd",
    "xgboost", "hist_gb",
    # Class name aliases
    "LSTLinearRegression", "LSTRidgeRegression", "LSTElasticNet",
    "LSTHuberRegression", "LSTSGDRegressor", "LSTNystroemSGD",
    "LSTXGBoost", "LSTHistGradientBoosting",
})


def train_all(
    source:         Union[Any, pd.DataFrame],
    registry:       Optional[Any]          = None,
    transforms:     Optional[List[Callable]] = None,
    feature_cols:   Optional[List[str]]    = None,
    default_scaler: str                    = "standard",
    column_scalers: Optional[Dict[str, str]] = None,
    batch_size:     int                    = 100_000,
    max_rows:       Optional[int]          = 2_000_000,
    models:         Optional[List[str]]    = None,
    cache_dir:      Optional[Union[str, Path]] = None,
    verbose:        bool                   = True,
    model_kwargs:   Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, LSTModel]:
    """
    Fit every requested model on the same data source.

    Steps
    -----
    1. Build / load a StreamingScaler from the full stream (cached by config
       fingerprint).
    2. For each model:
         - Streaming models  → call fit_stream() on a fresh stream generator
         - Memory-bounded    → call fit_stream() which does reservoir sampling
    3. Return {name: fitted_model}.

    Parameters
    ----------
    source : StreamConfig or pd.DataFrame
        Data source.  StreamConfig drives the streaming; DataFrame is used
        directly (scaler and models both see the same in-memory data).
    registry : FeatureRegistry, optional
        Forwarded to source.stream().
    transforms : list of callables, optional
        Feature transform pipeline applied before scaling and modelling.
        All models share the same transforms.
    feature_cols : list[str], optional
        Explicit feature columns.  None = auto-detect.
    default_scaler : str
        Default per-column scaler: "standard" | "minmax" | "maxabs" | "none".
    column_scalers : dict {col: scaler_name}, optional
        Per-column scaler overrides.
    batch_size : int
        Rows per streaming batch.
    max_rows : int, optional
        Maximum rows to stream.  Keep ≤ 5 000 000 to avoid OOM.
    models : list[str], optional
        Model names from ModelRegistry.  None = all registered models.
    cache_dir : str or Path, optional
        Directory for caching the fitted StreamingScaler.  Highly recommended
        to avoid re-scanning the full dataset on every run.
    verbose : bool
        Print progress.
    model_kwargs : dict {model_name: {kwarg: value}}, optional
        Per-model constructor overrides, e.g.:
            model_kwargs={"xgboost": {"n_estimators_per_batch": 50}}

    Returns
    -------
    dict mapping model name → fitted LSTModel instance.
    """
    t_total    = time.perf_counter()
    transforms = transforms or []
    mkwargs    = model_kwargs or {}

    # ---- 1. Build the StreamingScaler ----------------------------------
    if verbose:
        print("\n" + "=" * 70)
        print("  train_all: building StreamingScaler")
        print("=" * 70)

    extra_ctx = _stream_context(source)

    # Merge auto-detected scaler hints (e.g. fraction columns → minmax) with
    # any explicit column_scalers the caller passed.  Explicit wins.
    auto_hints: Dict[str, str] = {}
    if registry is not None and hasattr(registry, "column_scaler_hints"):
        try:
            auto_hints = registry.column_scaler_hints() or {}
        except Exception as exc:
            log.warning("registry.column_scaler_hints() failed: %s", exc)
    merged_scalers = {**auto_hints, **(column_scalers or {})}
    if verbose and auto_hints:
        n_auto = len(auto_hints)
        n_user = len((column_scalers or {}))
        print(f"  auto scaler hints: {n_auto} columns "
              f"({n_user} explicit overrides)")

    scaler = StreamingScaler(
        default_scaler = default_scaler,
        column_scalers = merged_scalers,
        transforms     = transforms,
        cache_dir      = cache_dir,
    )

    if isinstance(source, pd.DataFrame):
        # Fit scaler in chunks to avoid index-copy OOM on large frames
        df_full = source.head(max_rows) if max_rows else source
        df_t    = _apply_transforms(df_full.iloc[:min(100_000, len(df_full))], transforms)
        cols    = feature_cols or _auto_cols(df_t)
        # Partial-fit scaler across all chunks — tqdm bar only
        starts = range(0, len(df_full), batch_size)
        for start in tqdm(
            starts,
            desc="[StreamingScaler] fitting",
            unit="batch",
            total=len(starts),
            disable=not verbose,
            leave=False,
            dynamic_ncols=True,
        ):
            chunk   = df_full.iloc[start:start + batch_size]
            chunk_t = _apply_transforms(chunk, transforms)
            scaler.partial_fit(chunk_t, cols)
    else:
        scaler.fit_stream(
            source        = source,
            registry      = registry,
            transforms    = transforms,
            feature_cols  = feature_cols,
            batch_size    = batch_size,
            max_rows      = max_rows,
            extra_context = extra_ctx,
            verbose       = verbose,
        )

    if verbose:
        print(f"  StreamingScaler ready: {scaler}")

    # ---- 2. Train models -----------------------------------------------
    names   = models or ModelRegistry.available()
    results = {}

    for name in names:
        if verbose:
            print(f"\n{'─' * 70}")
            print(f"  Training: {name}")
            print("─" * 70)

        try:
            klass = ModelRegistry.get(name)
            kwargs = mkwargs.get(name, {})
            if feature_cols:
                kwargs.setdefault("feature_cols", feature_cols)
            model = klass(**kwargs)
            model.set_transforms(transforms)
            model.set_scaler(scaler)

            if isinstance(source, pd.DataFrame):
                # Always stream in chunks — never dump the full DataFrame into
                # fit_batch, which causes OOM on large frames (index copy, RF tree build).
                df_train = source.head(max_rows) if max_rows else source
                def _df_gen(df=df_train, bs=batch_size):
                    for start in range(0, len(df), bs):
                        yield df.iloc[start:start + bs]
                model.fit_stream(_df_gen(), verbose=verbose)
            else:
                model.fit_stream(
                    source     = source,
                    registry   = registry,
                    batch_size = batch_size,
                    max_rows   = max_rows,
                    verbose    = verbose,
                )

            results[name] = model

        except Exception as exc:
            log.error("train_all: %s failed — %s", name, exc, exc_info=True)
            if verbose:
                print(f"  ✗ {name}: {exc}")

    # ---- 3. Summary table ----------------------------------------------
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  train_all complete — {len(results)}/{len(names)} models trained "
              f"({time.perf_counter() - t_total:.1f}s)")
        print("=" * 70)
        _print_comparison(results)

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_transforms(df: pd.DataFrame, transforms: List[Callable]) -> pd.DataFrame:
    """Apply a list of transforms to a DataFrame (mirrors base.py logic)."""
    result = df.copy()
    for fn in transforms:
        try:
            extra = fn(result)
            if isinstance(extra, pd.DataFrame):
                for col in extra.columns:
                    result[col] = extra[col].values
        except Exception as exc:
            log.warning("transform %s failed: %s", getattr(fn, "__name__", fn), exc)
    return result


def _auto_cols(df: pd.DataFrame) -> List[str]:
    """Auto-detect numeric non-metadata feature columns."""
    return [
        c for c in df.columns
        if c not in NEVER_FEATURES
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def _stream_context(source: Any) -> Optional[Dict]:
    """
    Extract a serialisable context dict from a StreamConfig for fingerprinting.

    The context captures the distribution dimensions and partition_keys so
    that the scaler fingerprint changes when the stream target changes.
    """
    if not hasattr(source, "_distribution"):
        return None
    ctx: Dict = {}
    if source._distribution is not None:
        ctx["dimensions"] = sorted(source._distribution.dimensions.keys())
    if source._partitions:
        ctx["partition_keys"] = sorted(source._partitions)
    return ctx or None


def _print_comparison(models: Dict[str, LSTModel]) -> None:
    """Print a compact comparison table of all fitted models.

    Uses stream-aggregate metrics (rmse/mae/r² over the full stream) rather
    than last-batch values, which were unstable on low-variance batches.
    """
    print()
    print(f"  {'Model':<30}  {'RMSE':>7}  {'MAE':>7}  {'R²':>7}  {'Rows':>10}  {'Batches':>8}")
    print("  " + "─" * 68)
    for name, model in sorted(models.items()):
        s    = model.summary()
        h    = model._training_history
        rmse = f"{s['last_rmse']:.4f}" if s["last_rmse"] is not None else "n/a"
        mae  = f"{s['last_mae']:.4f}"  if s.get("last_mae") is not None else "n/a"
        r2   = f"{s['last_r2']:.4f}"   if s["last_r2"]   is not None else "n/a"
        n    = f"{s['rows_trained']:>10,}"
        b    = f"{len(h):>8}"
        print(f"  {name:<30}  {rmse:>7}  {mae:>7}  {r2:>7}  {n}  {b}")
    print()
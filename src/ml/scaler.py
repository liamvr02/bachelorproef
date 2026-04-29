"""
ml/scaler.py
============
StreamingScaler — a per-column scaler fitted incrementally on a full stream.

Design rationale
----------------
Previous versions used a StandardScaler embedded inside each model and fitted
on training batches only.  This had two problems:

  1. Different models got different scalers (minor data-leakage risk from
     sequential batch exposure).
  2. The scaler saw only training data, not the whole distribution.

StreamingScaler is trained separately, before any model, on the complete
stream (or as many rows as memory allows).  It is then injected into every
model via model.set_scaler().

The fitted scaler is cached to disk (joblib) keyed by a fingerprint of:
  - The feature columns (sorted)
  - The per-column scaler types
  - The transform pipeline (function names)
  - The StreamConfig parameters (partition_keys, distribution dimensions)

If the fingerprint matches a cached file, the scaler is loaded instantly
rather than re-fitted.  If any configuration differs (e.g. a new transform
was added), the cache is invalidated and the scaler is rebuilt.

Supported per-column scalers
-----------------------------
All scalers used here implement partial_fit() for streaming support:

  "standard"   : StandardScaler  — zero mean, unit variance
  "minmax"     : MinMaxScaler    — [0, 1] range
  "maxabs"     : MaxAbsScaler    — [-1, 1] range (preserves sparsity)
  "none"       : identity        — no scaling

RobustScaler does NOT support partial_fit and is therefore excluded.
If you need robust scaling, use clip_outliers() in the transform pipeline
before applying standard scaling.

Usage
-----
    from ml.scaler import StreamingScaler
    from ml.transforms import cyclical, log1p_transform

    transforms = [cyclical("month_of_year", 12), log1p_transform(["dhm1_elevation"])]
    scaler = StreamingScaler(
        default_scaler  = "standard",
        column_scalers  = {"ua_vegetation_100m_frac": "minmax"},
        transforms      = transforms,
        cache_dir       = Path("prepared_stream_data/scaler_cache"),
    )
    # Fit on the full stream (or use train_all which does this automatically)
    scaler.fit_stream(cfg, registry=reg, max_rows=5_000_000)
    scaler.save()   # writes to cache_dir / <fingerprint>.pkl

    # Inject into models
    for model in models.values():
        model.set_scaler(scaler)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from tqdm.auto import tqdm

log = logging.getLogger("lst_models.scaler")

# Map short name → sklearn class (all support partial_fit)
_SCALER_CLASSES = {
    "standard": StandardScaler,
    "minmax":   MinMaxScaler,
    "maxabs":   MaxAbsScaler,
    "none":     None,           # identity
}


class StreamingScaler:
    """
    Per-column feature scaler trained incrementally on a stream.

    Parameters
    ----------
    default_scaler : str
        Scaler applied to every column not listed in column_scalers.
        One of "standard", "minmax", "maxabs", "none".
    column_scalers : dict {col: scaler_name}, optional
        Per-column overrides.  Columns not listed get default_scaler.
    transforms : list of callables, optional
        The same transform list applied in the model pipeline.
        Used ONLY for fingerprinting (to detect config changes).
    cache_dir : Path, optional
        Directory where fitted scalers are cached.  When None, caching is
        disabled and the scaler is always built from scratch.
    """

    def __init__(
        self,
        default_scaler:  str                        = "standard",
        column_scalers:  Optional[Dict[str, str]]   = None,
        transforms:      Optional[List[Callable]]   = None,
        cache_dir:       Optional[Union[str, Path]] = None,
    ) -> None:
        if default_scaler not in _SCALER_CLASSES:
            raise ValueError(f"Unknown scaler '{default_scaler}'. "
                             f"Valid: {list(_SCALER_CLASSES)}")
        self.default_scaler = default_scaler
        self.column_scalers = column_scalers or {}
        self.transforms     = transforms or []
        self.cache_dir      = Path(cache_dir) if cache_dir else None

        # Fitted state
        self._scalers:    Dict[str, Any]  = {}   # col → fitted scaler instance
        self._is_fitted:  bool            = False
        self._feature_cols: List[str]     = []   # cols seen during fit

    # ------------------------------------------------------------------
    # Fingerprint / cache
    # ------------------------------------------------------------------

    def fingerprint(
        self,
        feature_cols: Optional[List[str]] = None,
        extra_context: Optional[Dict] = None,
    ) -> str:
        """
        Compute a 16-char SHA-256 fingerprint of the scaler configuration.

        The fingerprint encodes:
          - sorted feature columns
          - per-column scaler type assignments
          - transform function names (order-sensitive)
          - any extra context (e.g. StreamConfig distribution dimensions)

        Two scalers with identical fingerprints will produce identical
        transformations on the same data and may safely share a cache file.
        """
        spec = {
            "feature_cols":     sorted(feature_cols or self._feature_cols),
            "default_scaler":   self.default_scaler,
            "column_scalers":   dict(sorted(self.column_scalers.items())),
            "transform_names":  [
                getattr(fn, "__name__", str(fn)) for fn in self.transforms
            ],
            "extra":            extra_context or {},
        }
        return hashlib.sha256(
            json.dumps(spec, sort_keys=True).encode()
        ).hexdigest()[:16]

    def _cache_path(self, fingerprint: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"scaler_{fingerprint}.pkl"

    def try_load_cache(
        self,
        feature_cols:   Optional[List[str]] = None,
        extra_context:  Optional[Dict]       = None,
    ) -> bool:
        """
        Try to load a matching cached scaler.

        Returns True and populates self._scalers / self._feature_cols if a
        cache hit is found; returns False otherwise.
        """
        fp   = self.fingerprint(feature_cols, extra_context)
        path = self._cache_path(fp)
        if path is None or not path.exists():
            return False
        try:
            state = joblib.load(path)
            self._scalers      = state["scalers"]
            self._feature_cols = state["feature_cols"]
            self._is_fitted    = True
            log.info("StreamingScaler: cache HIT  fp=%s  path=%s", fp, path)
            return True
        except Exception as exc:
            log.warning("StreamingScaler: cache load failed (%s) — rebuilding", exc)
            return False

    def save(
        self,
        feature_cols:  Optional[List[str]] = None,
        extra_context: Optional[Dict]       = None,
    ) -> Optional[Path]:
        """Save fitted scaler to cache.  Returns the cache path or None."""
        fp   = self.fingerprint(feature_cols, extra_context)
        path = self._cache_path(fp)
        if path is None:
            return None
        state = {"scalers": self._scalers, "feature_cols": self._feature_cols}
        joblib.dump(state, path)
        log.info("StreamingScaler: saved  fp=%s  path=%s", fp, path)
        return path

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _get_or_create_scaler(self, col: str) -> Optional[Any]:
        """Return (creating if necessary) the sklearn scaler for a column."""
        if col not in self._scalers:
            scaler_name = self.column_scalers.get(col, self.default_scaler)
            klass = _SCALER_CLASSES[scaler_name]
            self._scalers[col] = klass() if klass is not None else None
        return self._scalers[col]

    def partial_fit(self, df: pd.DataFrame, cols: List[str]) -> None:
        """Update scaler statistics from one batch DataFrame."""
        for c in cols:
            scaler = self._get_or_create_scaler(c)
            if scaler is None:
                continue   # "none" scaler — skip
            if c not in df.columns:
                continue
            vals = df[[c]].to_numpy(dtype=np.float64)
            # Replace NaN with 0 for fitting (median imputation not yet applied here)
            vals = np.where(np.isfinite(vals), vals, 0.0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scaler.partial_fit(vals)
        if not self._feature_cols:
            self._feature_cols = [c for c in cols]
        self._is_fitted = True

    def fit_stream(
        self,
        source:         Any,
        registry:       Optional[Any] = None,
        transforms:     Optional[List[Callable]] = None,
        feature_cols:   Optional[List[str]] = None,
        batch_size:     int  = 100_000,
        max_rows:       Optional[int] = None,
        extra_context:  Optional[Dict] = None,
        verbose:        bool = True,
    ) -> "StreamingScaler":
        """
        Fit the scaler on a full stream, with cache check/write.

        Parameters
        ----------
        source : StreamConfig or generator of DataFrames
        registry : FeatureRegistry (forwarded to StreamConfig.stream)
        transforms : transforms to apply before fitting (same as model pipeline)
        feature_cols : explicit column list; None = auto-detect from first batch
        batch_size, max_rows : streaming parameters
        extra_context : additional metadata for the fingerprint (e.g. partition info)
        verbose : print progress
        """
        # Use provided transforms or fall back to those stored in __init__
        active_transforms = transforms or self.transforms

        # Cache check (skip if feature_cols unknown — fingerprint needs them)
        if feature_cols is not None:
            if self.try_load_cache(feature_cols, extra_context):
                if verbose:
                    print("[StreamingScaler] cache HIT — skipping fit")
                return self

        # Resolve generator — wrap DataFrame in a one-item list so the loop
        # below is uniform regardless of source type.
        if isinstance(source, pd.DataFrame):
            gen = (source.iloc[i:i+batch_size]
                   for i in range(0, len(source), batch_size))
        elif hasattr(source, "stream"):
            gen = source.stream(registry=registry, batch_size=batch_size, max_rows=max_rows)
        else:
            gen = source

        t0        = time.perf_counter()
        total     = 0
        resolved  = False

        total_batches = None
        if max_rows is not None and batch_size:
            total_batches = (max_rows + batch_size - 1) // batch_size

        bar = tqdm(
            gen,
            desc="[StreamingScaler] fitting",
            unit="batch",
            total=total_batches,
            disable=not verbose,
            leave=False,
            dynamic_ncols=True,
        )

        for batch_df in bar:
            # Apply transforms (same as model pipeline)
            if active_transforms:
                for fn in active_transforms:
                    try:
                        extra = fn(batch_df)
                        if isinstance(extra, pd.DataFrame):
                            for col in extra.columns:
                                batch_df[col] = extra[col].values
                    except Exception as exc:
                        log.warning("StreamingScaler transform %s failed: %s", fn, exc)

            # Resolve feature columns on first batch
            if not resolved:
                if feature_cols is not None:
                    cols = [c for c in feature_cols if c in batch_df.columns]
                else:
                    from ml.base import NEVER_FEATURES
                    cols = [
                        c for c in batch_df.columns
                        if c not in NEVER_FEATURES
                        and pd.api.types.is_numeric_dtype(batch_df[c])
                    ]
                self._feature_cols = cols
                resolved = True

                # Second cache check with resolved columns
                if self.try_load_cache(cols, extra_context):
                    bar.close()
                    if verbose:
                        print("[StreamingScaler] cache HIT after col resolution — skipping fit")
                    return self

            self.partial_fit(batch_df, cols)
            total += len(batch_df)
            bar.set_postfix(rows=f"{total:,}", refresh=False)

        bar.close()
        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"  [StreamingScaler] {total:,} rows | "
                  f"{len(self._feature_cols)} cols | "
                  f"scaler='{self.default_scaler}' | {elapsed:.1f}s")

        # Write cache
        saved = self.save(self._feature_cols, extra_context)
        if verbose and saved:
            print(f"  [StreamingScaler] cached to {saved}")

        return self

    # ------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------

    def transform_array(self, X: np.ndarray, cols: List[str]) -> np.ndarray:
        """
        Apply per-column scaling to a (n_samples, n_features) float32 array.

        Parameters
        ----------
        X    : preprocessed feature array, columns aligned to cols
        cols : column names corresponding to X[:, i]

        Returns
        -------
        Scaled array, same shape and dtype as X.
        """
        if not self._is_fitted:
            return X   # graceful pass-through if not yet fitted
        out = X.copy().astype(np.float64)
        for i, c in enumerate(cols):
            scaler = self._scalers.get(c)
            if scaler is None:
                continue
            # Skip columns not yet seen during fitting (no mean_ attribute yet)
            if not hasattr(scaler, "mean_") and not hasattr(scaler, "data_min_") \
                    and not hasattr(scaler, "max_abs_"):
                continue
            col_vec = out[:, i].reshape(-1, 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    out[:, i] = scaler.transform(col_vec).ravel()
                except Exception:
                    pass   # unfitted column — leave unscaled
        return out.astype(np.float32)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply per-column scaling to a DataFrame, returning a new DataFrame.

        Only columns present in self._feature_cols are scaled; others are
        passed through unchanged.
        """
        df2  = df.copy()
        cols = self._feature_cols
        X    = df2[cols].to_numpy(dtype=np.float32)
        Xt   = self.transform_array(X, cols)
        for i, c in enumerate(cols):
            df2[c] = Xt[:, i]
        return df2

    def inverse_transform_array(self, X: np.ndarray, cols: List[str]) -> np.ndarray:
        """Reverse the scaling (useful for interpreting predictions in original units)."""
        if not self._is_fitted:
            return X
        out = X.copy().astype(np.float64)
        for i, c in enumerate(cols):
            scaler = self._scalers.get(c)
            if scaler is None or not hasattr(scaler, "inverse_transform"):
                continue
            col_vec = out[:, i].reshape(-1, 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[:, i] = scaler.inverse_transform(col_vec).ravel()
        return out.astype(np.float32)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def scaler_for(self, col: str) -> str:
        """Return the scaler type name for a column."""
        scaler = self._scalers.get(col)
        if scaler is None:
            return "none"
        return type(scaler).__name__

    def summary(self) -> Dict[str, Any]:
        """Return a dict summarising fitted statistics."""
        info = {
            "is_fitted":      self._is_fitted,
            "n_features":     len(self._feature_cols),
            "default_scaler": self.default_scaler,
            "columns":        {},
        }
        for c in self._feature_cols:
            scaler = self._scalers.get(c)
            if scaler is None:
                info["columns"][c] = {"type": "none"}
            elif isinstance(scaler, StandardScaler) and hasattr(scaler, "mean_"):
                info["columns"][c] = {
                    "type": "standard",
                    "mean": float(scaler.mean_[0]),
                    "std":  float(np.sqrt(scaler.var_[0])),
                }
            elif isinstance(scaler, MinMaxScaler) and hasattr(scaler, "data_min_"):
                info["columns"][c] = {
                    "type": "minmax",
                    "min":  float(scaler.data_min_[0]),
                    "max":  float(scaler.data_max_[0]),
                }
            elif isinstance(scaler, MaxAbsScaler) and hasattr(scaler, "max_abs_"):
                info["columns"][c] = {
                    "type": "maxabs",
                    "max_abs": float(scaler.max_abs_[0]),
                }
        return info

    def __repr__(self) -> str:
        return (f"StreamingScaler(default={self.default_scaler}, "
                f"fitted={self._is_fitted}, "
                f"n_features={len(self._feature_cols)})")
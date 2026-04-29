"""
ml/models/forest.py
===================
LSTRandomForest and LSTExtraTrees — tree ensemble models for LST regression.

Memory-bounded streaming strategy
----------------------------------
Random Forest and Extra Trees cannot update existing trees with new data —
warm_start only adds new trees trained on the *current* fit() call's data.
The ideal solution is to train on a single maximally representative sample
that fits in available RAM.

This module implements **reservoir sampling** over the stream: as batches
arrive, each row is included in (or swapped into) a fixed-size reservoir
with the correct uniform probability so that, at the end of the stream, the
reservoir is a statistically representative i.i.d. sample of the full dataset
regardless of which partitions were streamed or in what order.

Reservoir size is determined automatically from available memory:
    safe_rows = int(avail_bytes * memory_fraction / bytes_per_row)
where bytes_per_row = n_features * 4 (float32) + 4 (target, float32).

After the stream is exhausted, fit() is called once on the complete reservoir.
This produces a standard fully-fitted RandomForest with correct ensemble
behaviour, unlike the earlier per-batch warm_start approach.

Explainability
--------------
Gini-impurity-based feature_importances_ (exact, deterministic).
SHAP via shap.TreeExplainer (exact for trees, fast).

Models
------
LSTRandomForest  — RandomForestRegressor
LSTExtraTrees    — ExtraTreesRegressor (faster, higher variance)
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import shap as _shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

from tqdm.auto import tqdm

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from ml.base import LSTModel, BatchRecord

import logging
import math
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

log = logging.getLogger("lst_models.forest")


def _safe_reservoir_rows(n_features: int, memory_fraction: float = 0.6) -> int:
    """
    Compute the number of rows that safely fit in available RAM.

    Uses psutil if available; falls back to a conservative 2 GB estimate.

    Parameters
    ----------
    n_features       : number of feature columns (determines row size)
    memory_fraction  : fraction of available RAM to use (default 0.6)
    """
    if _HAS_PSUTIL:
        avail_bytes = _psutil.virtual_memory().available
    else:
        avail_bytes = 2 * 1024 ** 3   # 2 GB fallback

    bytes_per_row = (n_features + 1) * 4   # float32 features + float32 target
    safe_rows     = int(avail_bytes * memory_fraction / bytes_per_row)
    # Clamp to a sensible range
    return max(50_000, min(safe_rows, 5_000_000))


class _Reservoir:
    """
    Vectorised reservoir sampler for streaming DataFrames.

    Maintains a fixed-size in-memory reservoir of (X, y) arrays such that
    after consuming any stream, each row in the reservoir is an independent
    uniform sample from all rows seen so far.

    Algorithm: Vitter's Algorithm R (classic reservoir sampling).
    """

    def __init__(self, capacity: int, rng_seed: int = 42) -> None:
        self.capacity  = capacity
        self.rng       = np.random.default_rng(rng_seed)
        self._X: Optional[np.ndarray] = None   # (capacity, n_features)
        self._y: Optional[np.ndarray] = None   # (capacity,)
        self._n_seen   = 0    # total rows offered
        self._n_stored = 0    # rows currently in reservoir

    def offer(self, X: np.ndarray, y: np.ndarray) -> None:
        """Offer a batch to the reservoir; updates in-place."""
        n = len(X)
        if n == 0:
            return

        if self._X is None:
            # Allocate reservoir on first non-empty batch
            self._X = np.empty((self.capacity, X.shape[1]), dtype=np.float32)
            self._y = np.empty(self.capacity, dtype=np.float32)

        for i in range(n):
            self._n_seen += 1
            if self._n_stored < self.capacity:
                # Still filling up — accept unconditionally
                self._X[self._n_stored] = X[i]
                self._y[self._n_stored] = y[i]
                self._n_stored += 1
            else:
                # Reservoir full — swap with probability capacity / n_seen
                j = int(self.rng.integers(0, self._n_seen))
                if j < self.capacity:
                    self._X[j] = X[i]
                    self._y[j] = y[i]

    @property
    def is_ready(self) -> bool:
        return self._n_stored > 0

    def get(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the current (X, y) reservoir arrays (n_stored, n_features)."""
        if not self.is_ready:
            raise RuntimeError("Reservoir is empty")
        return self._X[:self._n_stored], self._y[:self._n_stored]


def _reservoir_fit_stream(
    model:            "LSTModel",
    source:           Any,
    registry:         Optional[Any] = None,
    batch_size:       int            = 100_000,
    max_rows:         Optional[int]  = None,
    shap_sample_size: int            = 500,
    verbose:          bool           = True,
    memory_fraction:  Optional[float] = None,
    post_fit_hook:    Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
) -> "LSTModel":
    """
    Shared reservoir-then-fit training loop.

    Used by LSTRandomForest, LSTExtraTrees, and LSTHistGradientBoosting.
    Phase 1 streams the source into a fixed-size reservoir (sized to RAM);
    Phase 2 calls model._model.fit() once on the reservoir; Phase 3 computes
    metrics and stores them in training_history + _aggregate_metrics.

    Progress is shown with a single tqdm bar. No per-batch lines are printed;
    a single headline is emitted at the end.

    post_fit_hook(Xr, yr) lets subclasses (e.g. HGB) compute additional state
    after the main fit, such as permutation importance.
    """
    if hasattr(source, "stream"):
        generator = source.stream(
            registry=registry, batch_size=batch_size, max_rows=max_rows,
        )
    else:
        generator = source

    total_batches = None
    if max_rows is not None and batch_size:
        total_batches = (max_rows + batch_size - 1) // batch_size

    session_t0     = time.perf_counter()
    batch_idx      = 0
    shap_rows_kept = 0
    n_features_set = False
    mem_frac       = memory_fraction if memory_fraction is not None \
                     else getattr(model, "_memory_fraction", 0.6)

    bar = tqdm(
        generator,
        desc=f"[{model.model_name}] reservoir",
        unit="batch",
        total=total_batches,
        disable=not verbose,
        leave=False,
        dynamic_ncols=True,
    )

    for batch_df in bar:
        X, y = model._preprocess(batch_df, fit=True)
        if len(X) == 0:
            continue

        # Size the reservoir on the first non-empty batch
        if not n_features_set:
            capacity = _safe_reservoir_rows(X.shape[1], mem_frac)
            model._reservoir = _Reservoir(capacity, rng_seed=model.random_state)
            n_features_set   = True

        model._reservoir.offer(X, y)
        model._cumulative_rows += len(X)

        if shap_rows_kept < shap_sample_size:
            need  = shap_sample_size - shap_rows_kept
            chunk = X[:need]
            model._shap_background = (
                chunk if model._shap_background is None
                else np.vstack([model._shap_background, chunk])
            )
            shap_rows_kept += len(chunk)

        if n_features_set:
            bar.set_postfix(
                rows=f"{model._cumulative_rows:,}",
                reservoir=f"{model._reservoir._n_stored:,}/{model._reservoir.capacity:,}",
                refresh=False,
            )
        batch_idx += 1

    bar.close()

    if not (model._reservoir and model._reservoir.is_ready):
        log.warning("%s: no data collected — not fitted", model.model_name)
        return model

    Xr, yr = model._reservoir.get()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model._model.fit(Xr, yr)
    model._is_fitted = True

    if post_fit_hook is not None:
        try:
            post_fit_hook(Xr, yr)
        except Exception as exc:
            log.warning("%s: post_fit_hook failed: %s", model.model_name, exc)

    # Metrics computed on the reservoir (no held-out split by design — the
    # reservoir *is* the training set; held-out evaluation is the caller's job).
    y_hat = model._model.predict(Xr)
    mse   = float(mean_squared_error(yr, y_hat))
    mae   = float(mean_absolute_error(yr, y_hat))
    r2    = float(r2_score(yr, y_hat))
    elapsed = time.perf_counter() - session_t0

    model._training_history.append(BatchRecord(
        batch_idx       = 0,
        n_rows          = len(Xr),
        mse             = mse,
        rmse            = math.sqrt(mse),
        mae             = mae,
        r2              = r2,
        elapsed_s       = elapsed,
        cumulative_rows = model._cumulative_rows,
    ))
    model._aggregate_metrics = {
        "n_rows": float(len(Xr)),
        "rmse":   math.sqrt(mse),
        "mae":    mae,
        "r2":     r2,
        "mse":    mse,
    }

    if verbose:
        print(
            f"  [{model.model_name}] {model._cumulative_rows:,} stream rows | "
            f"reservoir {len(Xr):,} | "
            f"RMSE {math.sqrt(mse):.3f} | MAE {mae:.3f} | "
            f"R² {r2:.3f} | {elapsed:.1f}s"
        )

    return model


class LSTRandomForest(LSTModel):
    """
    Random Forest regressor trained on a memory-safe reservoir sample.

    The model consumes the full stream via reservoir sampling (uniform i.i.d.
    sample of all rows seen) and then trains once on the collected sample.
    This is the correct approach for sklearn RandomForest, which cannot
    meaningfully accumulate knowledge across separate fit() calls.

    Parameters
    ----------
    n_estimators     : number of trees (default 200)
    max_depth        : maximum tree depth (None = unlimited)
    min_samples_leaf : minimum samples per leaf
    max_features     : feature fraction per split ("sqrt", float, or int)
    memory_fraction  : fraction of available RAM for the reservoir (default 0.6)
    n_jobs           : parallelism (-1 = all cores)
    """

    NEEDS_SCALING = False  # trees are invariant to monotone rescaling

    def __init__(
        self,
        n_estimators:     int            = 200,
        max_depth:        Optional[int]  = None,
        min_samples_leaf: int            = 5,
        max_features:     Any            = "sqrt",
        memory_fraction:  float          = 0.6,
        n_jobs:           int            = -1,
        **kwargs,
    ) -> None:
        self._n_estimators     = n_estimators
        self._max_depth        = max_depth
        self._min_samples_leaf = min_samples_leaf
        self._max_features     = max_features
        self._memory_fraction  = memory_fraction
        self._n_jobs           = n_jobs
        self._reservoir: Optional[_Reservoir] = None
        super().__init__(**kwargs)

    @property
    def model_name(self) -> str:
        return "Random Forest"

    # Trees are invariant to monotone rescaling — skip the StreamingScaler.
    NEEDS_SCALING: bool = False

    @property
    def default_hyperparameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "n_estimators":     [100, 200, 300],
            "max_depth":        [10, 20, None],
            "min_samples_leaf": [2, 5, 10],
            "max_features":     ["sqrt", 0.5, 0.8],
        }

    def _init_model(self) -> None:
        self._model = RandomForestRegressor(
            n_estimators     = getattr(self, "_n_estimators",     200),
            max_depth        = getattr(self, "_max_depth",        None),
            min_samples_leaf = getattr(self, "_min_samples_leaf", 5),
            max_features     = getattr(self, "_max_features",     "sqrt"),
            n_jobs           = getattr(self, "_n_jobs",           -1),
            random_state     = getattr(self, "random_state",      42),
        )
        self._reservoir = None

    def _partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # During streaming, only accumulate into the reservoir.
        # The actual sklearn fit() happens once in fit_stream() after the
        # stream is exhausted (see the overridden fit_stream below).
        if self._reservoir is not None:
            self._reservoir.offer(X, y)

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def _feature_importance_raw(self) -> np.ndarray:
        return self._model.feature_importances_

    def _shap_values(self, X: np.ndarray) -> np.ndarray:
        if _HAS_SHAP:
            # shap_values() returns ndarray for RF; .shap_values() is the
            # old API that always returns a plain array (not Explanation object)
            exp = _shap.TreeExplainer(self._model)
            return np.array(exp.shap_values(X))
        # Fallback: importance-weighted deviation from mean
        fi        = self._feature_importance_raw()
        mean_pred = self._predict_raw(X).mean()
        return (X - X.mean(axis=0)) * fi[np.newaxis, :] * (mean_pred / (fi.sum() + 1e-9))

    def _apply_params(self, params: Dict[str, Any]) -> None:
        for k, v in params.items():
            setattr(self._model, k, v)

    def _save_impl(self, path: Path) -> None:
        pass   # everything serialised by base.save()

    def _load_impl(self, path: Path) -> None:
        pass

    def fit_batch(self, df: pd.DataFrame, eval: bool = True) -> "LSTRandomForest":
        """
        Train on a single in-memory DataFrame.

        RF cannot meaningfully use partial_fit; instead we run a single
        sklearn fit() on the preprocessed data directly.  The base class
        fit_batch() calls _partial_fit() → reservoir offer → no fit(), so we
        override here to skip the reservoir entirely when data is already in
        memory.
        """
        t0   = time.perf_counter()
        X, y = self._preprocess(df, fit=True)
        if len(X) == 0:
            log.warning("fit_batch: no valid rows after preprocessing")
            return self

        if self._shap_background is None:
            self._shap_background = X[:min(500, len(X))]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X, y)

        self._is_fitted        = True
        self._cumulative_rows += len(X)

        if eval:
            y_hat = self._model.predict(X)
            mse   = float(mean_squared_error(y, y_hat))
            mae   = float(mean_absolute_error(y, y_hat))
            r2    = float(r2_score(y, y_hat))
            self._training_history.append(BatchRecord(
                batch_idx       = len(self._training_history),
                n_rows          = len(X),
                mse             = mse,
                rmse            = math.sqrt(mse),
                mae             = mae,
                r2              = r2,
                elapsed_s       = time.perf_counter() - t0,
                cumulative_rows = self._cumulative_rows,
            ))
        return self

    # ------------------------------------------------------------------
    # Override fit_stream to implement reservoir-then-fit strategy
    # ------------------------------------------------------------------

    def fit_stream(
        self,
        source:           Union[Any, Generator],
        registry:         Optional[Any] = None,
        batch_size:       int            = 10_000,
        max_rows:         Optional[int]  = None,
        eval_every:       int            = 1,
        shap_sample_size: int            = 500,
        verbose:          bool           = True,
    ) -> "LSTRandomForest":
        """
        Consume the stream via reservoir sampling, then train once on the sample.

        Phase 1 — Stream pass: each batch is preprocessed and offered to the
        reservoir.  The reservoir size is capped to available RAM.

        Phase 2 — Fit: sklearn RandomForest is trained once on the full
        reservoir using all CPUs.

        Phase 3 — Evaluate: metrics are computed on a fresh pass over the
        reservoir and recorded in training_history.
        """
        # Delegate to the shared reservoir-then-fit helper — LSTHistGradientBoosting
        # uses the same strategy, so the implementation lives at module level.
        return _reservoir_fit_stream(
            self, source,
            registry         = registry,
            batch_size       = batch_size,
            max_rows         = max_rows,
            shap_sample_size = shap_sample_size,
            verbose          = verbose,
        )

    @property
    def n_trees(self) -> int:
        """Number of trees in the fitted forest."""
        return self._model.n_estimators


class LSTExtraTrees(LSTRandomForest):
    """
    Extremely Randomised Trees — like Random Forest but with random split
    thresholds instead of optimal splits per tree.

    Faster to train than RandomForest, lower bias but higher variance.
    Ensemble averaging cancels much of the variance, making it competitive
    in accuracy while being substantially faster.

    Uses the same reservoir-sampling stream strategy as LSTRandomForest.
    """

    NEEDS_SCALING = False

    @property
    def model_name(self) -> str:
        return "Extra Trees"

    def _init_model(self) -> None:
        self._model = ExtraTreesRegressor(
            n_estimators     = getattr(self, "_n_estimators",     200),
            max_depth        = getattr(self, "_max_depth",        None),
            min_samples_leaf = getattr(self, "_min_samples_leaf", 5),
            max_features     = getattr(self, "_max_features",     "sqrt"),
            n_jobs           = getattr(self, "_n_jobs",           -1),
            random_state     = getattr(self, "random_state",      42),
        )
        self._reservoir = None

    def _shap_values(self, X: np.ndarray) -> np.ndarray:
        if _HAS_SHAP:
            exp = _shap.TreeExplainer(self._model)
            return np.array(exp.shap_values(X))
        fi        = self._feature_importance_raw()
        mean_pred = self._predict_raw(X).mean()
        return (X - X.mean(axis=0)) * fi[np.newaxis, :] * (mean_pred / (fi.sum() + 1e-9))
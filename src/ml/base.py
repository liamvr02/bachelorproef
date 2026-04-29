"""
ml/base.py
==========
Abstract base class for all LST temperature regression models.

Every concrete model inherits all public interface — streaming training,
preprocessing, evaluation, explainability, reporting, hyperparameter search,
and persistence — and only needs to implement eight narrow methods:

    _init_model()
    _partial_fit(X, y)
    _predict_raw(X)
    _feature_importance_raw()
    _shap_values(X)
    _save_impl(path)          # currently a no-op; base save() serialises all
    _load_impl(path)          # currently a no-op; base load() deserialises all
    model_name (property)
    default_hyperparameter_grid (property)

Preprocessing pipeline
----------------------
Applied to every batch before the model sees data:

  1. Transform functions (optional)  — vectorised DataFrame → DataFrame functions
     registered via set_transforms().  Applied first, in order.
  2. NaN imputation                  — missing values filled with per-column
     running medians (EMA updated every batch).
  3. StreamingScaler (optional)      — applied after transforms, if provided.
     The scaler is trained separately on the full stream and injected via
     set_scaler().  The model does NOT fit its own scaler.

The old `scale_features` / `StandardScaler` per-model approach is removed.
Scaling is now always the caller's responsibility via StreamingScaler.
"""

from __future__ import annotations

import abc
import copy
import datetime
import io
import logging
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

try:
    import shap as _shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

from ml import sanity as _sanity

log = logging.getLogger("lst_models")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns always present in a streamed batch that must NEVER be used as
# model features — they are identifiers, metadata, or the target itself.
NEVER_FEATURES: frozenset = frozenset({
    "longitude", "latitude",
    "aster_lst", "modis_lst", "ndvi",
    "temperature",
    "image_id", "timestamp", "partition_key", "tile_id",
})

TARGET_COL = "temperature"


# ---------------------------------------------------------------------------
# BatchRecord — one entry in the training history
# ---------------------------------------------------------------------------

@dataclass
class BatchRecord:
    """Metrics captured after each training batch."""
    batch_idx:       int
    n_rows:          int
    mse:             float
    rmse:            float
    mae:             float
    r2:              float
    elapsed_s:       float
    cumulative_rows: int


# ---------------------------------------------------------------------------
# LSTModel — abstract base class
# ---------------------------------------------------------------------------

class LSTModel(abc.ABC):
    """
    Abstract base class for streaming-compatible LST temperature regression models.

    Parameters
    ----------
    feature_cols : list[str] or None
        Columns to use as features.  None = auto-detect from first batch
        (all numeric non-metadata columns not in NEVER_FEATURES, after
        transforms are applied).
    target_col : str
        Regression target column (default: "temperature").
    random_state : int
        Forwarded to sklearn / XGBoost constructors.
    """

    def __init__(
        self,
        feature_cols:  Optional[List[str]] = None,
        target_col:    str = TARGET_COL,
        random_state:  int = 42,
    ) -> None:
        self.feature_cols   = feature_cols
        self.target_col     = target_col
        self.random_state   = random_state

        self._model:                  Any                 = None
        self._scaler:                 Optional[Any]       = None  # injected StreamingScaler
        self._transforms:             List[Callable]      = []    # batch → batch transforms
        self._col_medians:            Dict[str, float]    = {}    # for NaN imputation
        self._is_fitted:              bool                = False
        self._training_history:       List[BatchRecord]   = []
        self._cumulative_rows:        int                 = 0
        self._resolved_feature_cols:  List[str]           = []
        self._shap_background:        Optional[np.ndarray] = None  # small sample for SHAP
        # Aggregate metrics over the full stream (computed at end of fit_stream).
        # Preferred over per-batch history for reporting — per-batch R² is
        # unstable on low-variance batches.
        self._aggregate_metrics:      Dict[str, float]    = {}

        self._init_model()

    # ------------------------------------------------------------------
    # Abstract interface — subclasses implement these
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _init_model(self) -> None:
        """Initialise self._model with default or user-supplied hyperparameters."""

    @abc.abstractmethod
    def _partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update the model with one preprocessed batch."""

    @abc.abstractmethod
    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Return raw predictions for preprocessed X (1-D float array)."""

    @abc.abstractmethod
    def _feature_importance_raw(self) -> np.ndarray:
        """
        Return importances aligned to self._resolved_feature_cols.
        Values need not sum to 1.
        """

    @abc.abstractmethod
    def _shap_values(self, X: np.ndarray) -> np.ndarray:
        """Return SHAP values, shape (n_samples, n_features)."""

    @abc.abstractmethod
    def _save_impl(self, path: Path) -> None:
        """Persist model-specific state (base handles the common state)."""

    @abc.abstractmethod
    def _load_impl(self, path: Path) -> None:
        """Restore model-specific state."""

    def _get_extra_state(self) -> Dict[str, Any]:
        """
        Return a dict of subclass-specific attributes to include in saves.

        Override in subclasses that carry state beyond self._model, e.g.:
            LSTNystroemSGD  — _nystroem, _nys_fitted
            LSTXGBoost      — _booster
            LSTHistGB       — _perm_imp, _current_max_iter
            LSTRandomForest — _current_n_estimators
        """
        return {}

    def _set_extra_state(self, state: Dict[str, Any]) -> None:
        """Restore subclass-specific attributes from a dict returned by _get_extra_state."""
        for k, v in state.items():
            setattr(self, k, v)

    # Tree-based subclasses override this to False.
    # When False, the injected StreamingScaler is stored but never applied —
    # tree splits are invariant to monotone rescaling so scaling wastes time
    # and makes coefficients uninterpretable.
    NEEDS_SCALING: bool = True

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Short human-readable model name."""

    @property
    @abc.abstractmethod
    def default_hyperparameter_grid(self) -> Dict[str, List[Any]]:
        """Default parameter grid for hyperparameter_search()."""

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_transforms(self, transforms: List[Callable]) -> "LSTModel":
        """
        Register a list of vectorised transform functions.

        Each function receives a DataFrame and returns a DataFrame.
        Transforms are applied in order, before imputation and scaling.

        Typical usage (see transforms.py for factories):

            from ml.transforms import cyclical, log1p_transform
            model.set_transforms([
                cyclical("month_of_year", period=12),
                cyclical("hour_of_day",   period=24),
                log1p_transform(["dhm1_elevation"]),
            ])
        """
        self._transforms = list(transforms)
        # Reset resolved columns so they are re-derived after transforms
        self._resolved_feature_cols = []
        return self

    def set_scaler(self, scaler: Any) -> "LSTModel":
        """
        Inject a pre-trained StreamingScaler.

        The scaler is applied to each batch after transforms but before the
        model.  It must implement transform(df) → np.ndarray.
        """
        self._scaler = scaler
        return self

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _apply_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all registered transforms to df, returning an augmented DataFrame.

        Handles both plain callables and Transform objects (from transforms.py):
          - Transform.should_skip(model_name) → skip this transform for this model
          - Transform.drop_inputs             → drop input columns immediately
                                                 after the transform runs (covers
                                                 both raw and intermediate inputs)

        Global drop-raw-inputs rule
        ---------------------------
        After the full pipeline runs, every column that was (a) present in the
        original DataFrame and (b) listed as an input to *any* applied
        transform is removed.  This means: registering a transform on column A
        means raw A no longer reaches the model — only the transform outputs
        do.  Use ``remove(cols)`` from ml.transforms to drop a column without
        producing a transformed replacement.
        """
        if not self._transforms:
            return df
        result      = df.copy()
        start_cols  = set(df.columns)
        inputs_used: set = set()
        for fn in self._transforms:
            # Respect skip_models on Transform objects
            if hasattr(fn, "should_skip") and fn.should_skip(self.model_name):
                continue
            try:
                extra = fn(result)
            except Exception as exc:
                log.warning("transform %s failed: %s",
                            getattr(fn, "__name__", fn), exc)
                continue
            inputs_used.update(getattr(fn, "input_cols", []))
            if isinstance(extra, pd.DataFrame):
                if not extra.empty:
                    for col in extra.columns:
                        result[col] = extra[col].values
                # Per-transform drop_inputs: drops the transform's input
                # columns immediately, regardless of whether they were raw
                # or produced upstream.  Honoured even when the transform
                # produced no outputs (see remove()).
                if getattr(fn, "drop_inputs", False):
                    extra_cols = set(extra.columns)
                    to_drop = [c for c in getattr(fn, "input_cols", [])
                               if c in result.columns and c not in extra_cols]
                    result.drop(columns=to_drop, inplace=True, errors="ignore")

        # Global rule: every raw column consumed by any applied transform
        # is removed once the pipeline finishes.  Transform outputs (which
        # were not in start_cols) are unaffected.
        raw_to_drop = [c for c in start_cols
                       if c in inputs_used and c in result.columns]
        if raw_to_drop:
            result.drop(columns=raw_to_drop, inplace=True, errors="ignore")
        return result

    def _resolve_cols(self, df: pd.DataFrame) -> List[str]:
        """Determine feature columns from df (called once after transforms)."""
        if not self._resolved_feature_cols:
            if self.feature_cols is not None:
                cols = [c for c in self.feature_cols if c in df.columns]
                missing = [c for c in self.feature_cols if c not in df.columns]
                if missing:
                    log.warning("%s: requested feature columns not found: %s",
                                self.model_name, missing)
            else:
                cols = [
                    c for c in df.columns
                    if c not in NEVER_FEATURES
                    and pd.api.types.is_numeric_dtype(df[c])
                ]
            if not cols:
                raise ValueError(
                    "No valid feature columns found after transforms.  "
                    "Supply feature_cols explicitly or check your transform pipeline."
                )
            self._resolved_feature_cols = cols
            log.info("%s: resolved %d feature columns: %s",
                     self.model_name, len(cols), cols)
        return self._resolved_feature_cols

    def _impute(self, df: pd.DataFrame, cols: List[str], update: bool) -> pd.DataFrame:
        """Fill NaN values using running column medians (EMA)."""
        for c in cols:
            col_vals = df[c].dropna()
            if not col_vals.empty:
                new_med = float(col_vals.median())
                if update:
                    old = self._col_medians.get(c, new_med)
                    self._col_medians[c] = 0.9 * old + 0.1 * new_med
            if df[c].isna().any():
                fill = self._col_medians.get(c, 0.0)
                df = df.copy()
                df[c] = df[c].fillna(fill)
        return df

    def _preprocess(
        self, df: pd.DataFrame, fit: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline: transforms → resolve cols → impute → scale.
        Returns (X, y) as float32 arrays.  Rows with NaN target are dropped.

        Uses dropna() instead of boolean loc+reset_index to avoid allocating
        a full int64 positional index array on large DataFrames (which causes
        OOM at ~15 M rows × 8 bytes = 121 MiB just for the index).
        """
        # Drop rows with no target — dropna is faster and allocates no index copy
        df = df.dropna(subset=[self.target_col])
        if df.empty:
            n = len(self._resolved_feature_cols) or 1
            return np.empty((0, n), dtype=np.float32), np.empty(0, dtype=np.float32)

        # 1. Transforms
        df = self._apply_transforms(df)

        # 2. Resolve feature columns (first call only)
        cols = self._resolve_cols(df)

        # 3. NaN imputation
        df = self._impute(df, cols, update=fit)

        X = df[cols].to_numpy(dtype=np.float32)
        y = df[self.target_col].to_numpy(dtype=np.float32)

        # 4. Scale via injected StreamingScaler — only for models that benefit
        if self._scaler is not None and self.NEEDS_SCALING:
            X = self._scaler.transform_array(X, cols)

        return X, y

    def _preprocess_predict(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess for inference (no target, no median updates)."""
        df   = df.copy()
        df   = self._apply_transforms(df)
        cols = self._resolved_feature_cols
        for c in cols:
            if c not in df.columns:
                df[c] = self._col_medians.get(c, 0.0)
        df = self._impute(df, cols, update=False)
        X  = df[cols].to_numpy(dtype=np.float32)
        if self._scaler is not None and self.NEEDS_SCALING:
            X = self._scaler.transform_array(X, cols)
        return X

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit_stream(
        self,
        source:        Union[Any, Generator],
        registry:      Optional[Any]  = None,
        batch_size:    int             = 100_000,
        max_rows:      Optional[int]   = None,
        eval_every:    int             = 1,
        shap_sample_size: int          = 500,
        verbose:       bool            = True,
    ) -> "LSTModel":
        """
        Train by consuming batches from a stream.

        Progress is shown via tqdm (one bar for the full stream). No per-batch
        lines are printed — per-batch R² is unstable on low-variance batches
        and was previously drowning out useful signal. Aggregate RMSE / MAE /
        R² over the entire stream is computed via running sums (Welford-style
        for SST) and printed as a single headline when training completes.

        BatchRecord entries are still appended to self._training_history (used
        for the HTML report's training-history chart) but no longer printed.
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

        # Running aggregates for end-of-stream metrics
        agg_n       = 0
        agg_sse     = 0.0     # Σ (y − ŷ)²
        agg_abs_err = 0.0     # Σ |y − ŷ|
        agg_y_sum   = 0.0     # Σ y
        agg_y2_sum  = 0.0     # Σ y²

        bar = tqdm(
            generator,
            desc=f"[{self.model_name}] training",
            unit="batch",
            total=total_batches,
            disable=not verbose,
            leave=False,
            dynamic_ncols=True,
        )

        scaler_checked = False

        for batch_df in bar:
            t0 = time.perf_counter()
            X, y = self._preprocess(batch_df, fit=True)
            if len(X) == 0:
                continue

            # ---- sanity: scaler alignment (once, on first non-empty batch)
            if not scaler_checked:
                _sanity.check_scaler_alignment(
                    self._scaler, self._resolved_feature_cols,
                    model_name=self.model_name,
                )
                scaler_checked = True

            # ---- sanity: per-batch input ranges + finiteness
            _sanity.check_input_batch(
                X, y, self._resolved_feature_cols, batch_idx,
                model_name=self.model_name,
            )

            # Accumulate SHAP background from early batches
            if shap_rows_kept < shap_sample_size:
                need  = shap_sample_size - shap_rows_kept
                chunk = X[:need]
                self._shap_background = (
                    chunk if self._shap_background is None
                    else np.vstack([self._shap_background, chunk])
                )
                shap_rows_kept += len(chunk)

            self._partial_fit(X, y)
            self._is_fitted        = True
            self._cumulative_rows += len(X)

            # ---- sanity: linear-model weight explosion (cheap, no-op for trees)
            _sanity.check_post_step(
                self, batch_idx, model_name=self.model_name,
            )

            if batch_idx % eval_every == 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y_hat = self._predict_raw(X)
                y_d   = y.astype(np.float64, copy=False)
                yh_d  = np.asarray(y_hat, dtype=np.float64)
                diff  = yh_d - y_d
                sse   = float(np.dot(diff, diff))
                abs_e = float(np.abs(diff).sum())
                agg_n       += len(y_d)
                agg_sse     += sse
                agg_abs_err += abs_e
                agg_y_sum   += float(y_d.sum())
                agg_y2_sum  += float(np.dot(y_d, y_d))

                batch_mse = sse / len(y_d)
                batch_mae = abs_e / len(y_d)
                batch_r2  = (
                    float(r2_score(y_d, yh_d)) if len(y_d) > 1 else float("nan")
                )
                self._training_history.append(BatchRecord(
                    batch_idx       = batch_idx,
                    n_rows          = len(X),
                    mse             = batch_mse,
                    rmse            = math.sqrt(batch_mse),
                    mae             = batch_mae,
                    r2              = batch_r2,
                    elapsed_s       = time.perf_counter() - t0,
                    cumulative_rows = self._cumulative_rows,
                ))

                if agg_n > 0:
                    running_rmse = math.sqrt(agg_sse / agg_n)
                    running_mae  = agg_abs_err / agg_n
                    bar.set_postfix(
                        rows=f"{self._cumulative_rows:,}",
                        rmse=f"{running_rmse:.3f}",
                        mae=f"{running_mae:.3f}",
                        refresh=False,
                    )
            batch_idx += 1

        bar.close()

        total = time.perf_counter() - session_t0
        if agg_n > 0:
            y_mean = agg_y_sum / agg_n
            sst    = agg_y2_sum - agg_n * y_mean * y_mean
            agg_rmse = math.sqrt(agg_sse / agg_n)
            agg_mae  = agg_abs_err / agg_n
            agg_r2   = 1.0 - (agg_sse / sst) if sst > 0 else float("nan")
            self._aggregate_metrics = {
                "n_rows": float(agg_n),
                "rmse":   agg_rmse,
                "mae":    agg_mae,
                "r2":     agg_r2,
                "mse":    agg_sse / agg_n,
            }
            if verbose:
                print(
                    f"  [{self.model_name}] {self._cumulative_rows:,} rows | "
                    f"{batch_idx} batches | "
                    f"RMSE {agg_rmse:.3f} | MAE {agg_mae:.3f} | "
                    f"R² {agg_r2:.3f} | {total:.1f}s"
                )
        elif verbose:
            print(f"  [{self.model_name}] training complete — "
                  f"no batches processed ({total:.1f}s)")
        return self

    def fit_batch(self, df: pd.DataFrame, eval: bool = True) -> "LSTModel":
        """Train on a single in-memory DataFrame."""
        t0   = time.perf_counter()
        X, y = self._preprocess(df, fit=True)
        if len(X) == 0:
            log.warning("fit_batch: no valid rows after preprocessing")
            return self

        self._partial_fit(X, y)
        self._is_fitted        = True
        self._cumulative_rows += len(X)

        if self._shap_background is None:
            self._shap_background = X[:min(500, len(X))]

        if eval:
            y_hat = self._predict_raw(X)
            mse   = float(mean_squared_error(y, y_hat))
            mae   = float(mean_absolute_error(y, y_hat))
            r2    = float(r2_score(y, y_hat)) if len(y) > 1 else float("nan")
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
    # Inference and evaluation
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict temperature for each row of df."""
        self._check_fitted()
        y_pred = self._predict_raw(self._preprocess_predict(df)).astype(np.float32)
        _sanity.check_predictions(y_pred, label="predict",
                                  model_name=self.model_name)
        return y_pred

    def evaluate(
        self,
        source:     Union[pd.DataFrame, Any, Generator],
        registry:   Optional[Any] = None,
        batch_size: int = 50_000,
        max_rows:   Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute metrics over a dataset.

        DataFrames are processed in chunks of batch_size rows to avoid
        allocating a full positional index copy on large frames (OOM at ~15 M rows).

        Returns dict: n_rows, mse, rmse, mae, r2, mape, bias, max_error.
        """
        self._check_fitted()

        def _gen():
            if isinstance(source, pd.DataFrame):
                df = source.head(max_rows) if max_rows else source
                for start in range(0, len(df), batch_size):
                    yield df.iloc[start:start + batch_size]
            elif hasattr(source, "stream"):
                yield from source.stream(
                    registry=registry, batch_size=batch_size, max_rows=max_rows
                )
            else:
                yield from source

        ys, yhats = [], []
        for batch_df in _gen():
            X, y = self._preprocess(batch_df, fit=False)
            if len(X) == 0:
                continue
            ys.append(y)
            yhats.append(self._predict_raw(X))

        if not ys:
            return {}

        yt = np.concatenate(ys).astype(np.float64)
        yh = np.concatenate(yhats).astype(np.float64)
        _sanity.check_predictions(yh, yt, label="evaluate",
                                  model_name=self.model_name)
        n  = len(yt)
        mse = float(mean_squared_error(yt, yh))
        mae = float(mean_absolute_error(yt, yh))
        r2  = float(r2_score(yt, yh)) if n > 1 else float("nan")
        return {
            "n_rows":    n,
            "mse":       mse,
            "rmse":      math.sqrt(mse),
            "mae":       mae,
            "r2":        r2,
            "mape":      float(np.mean(np.abs((yt - yh) / np.where(np.abs(yt) > 1e-6, yt, 1.0))) * 100),
            "bias":      float(np.mean(yh - yt)),
            "max_error": float(np.max(np.abs(yt - yh))),
        }

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def feature_importance(self) -> Dict[str, float]:
        """
        Return {column: importance} normalised to sum to 1.

        For linear models the raw importance is |coefficient|.  When a
        StreamingScaler with StandardScaler columns is active, coefficients are
        in scaled space; multiplying by the per-column std converts them back
        to original-unit magnitudes so features are comparable across different
        value ranges.  Tree models use Gini/permutation importance which is
        already scale-free.
        """
        self._check_fitted()
        raw  = self._feature_importance_raw().copy().astype(np.float64)
        cols = self._resolved_feature_cols

        # Unscale linear coefficients into original-unit magnitudes
        if (self.NEEDS_SCALING
                and self._scaler is not None
                and self._scaler._is_fitted):
            for i, c in enumerate(cols):
                scaler_obj = self._scaler._scalers.get(c)
                if scaler_obj is not None and hasattr(scaler_obj, "var_"):
                    # StandardScaler: coef in scaled space → multiply by σ
                    raw[i] *= float(np.sqrt(scaler_obj.var_[0]) + 1e-12)
                elif scaler_obj is not None and hasattr(scaler_obj, "data_range_"):
                    # MinMaxScaler: coef in [0,1] space → multiply by range
                    raw[i] *= float(scaler_obj.data_range_[0] + 1e-12)

        raw   = np.abs(raw)   # importance is unsigned magnitude
        total = float(raw.sum()) or 1.0
        return {c: float(v / total) for c, v in zip(cols, raw)}

    def shap_sample(
        self,
        df: Optional[pd.DataFrame] = None,
        n:  int = 200,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Compute SHAP values for up to n rows.

        Returns (explanation_df, raw_shap_array).
        """
        self._check_fitted()
        if df is not None:
            X = self._preprocess_predict(df.head(n))
        elif self._shap_background is not None:
            X = self._shap_background[:n]
        else:
            raise RuntimeError("No data for SHAP.  Pass a DataFrame or train first.")

        sv   = self._shap_values(X)
        cols = self._resolved_feature_cols
        result = pd.DataFrame(X, columns=[f"feat_{c}" for c in cols])
        result["prediction"] = self._predict_raw(X)
        for i, c in enumerate(cols):
            result[f"shap_{c}"] = sv[:, i]
        return result, sv

    # ------------------------------------------------------------------
    # Hyperparameter search
    # ------------------------------------------------------------------

    def hyperparameter_search(
        self,
        param_grid:  Optional[Dict[str, List[Any]]] = None,
        source:      Union[Any, Generator, pd.DataFrame, None] = None,
        registry:    Optional[Any] = None,
        sample_rows: int  = 100_000,
        cv_batches:  int  = 3,
        verbose:     bool = True,
    ) -> Dict[str, Any]:
        """
        Grid search via a stream sample.

        Each candidate trains on half the sample and evaluates on the other
        half, repeated cv_batches times.  Best params are applied to self.
        """
        if source is None:
            raise ValueError("source is required for hyperparameter_search()")

        grid = list(ParameterGrid(param_grid or self.default_hyperparameter_grid))
        if verbose:
            print(f"\n[{self.model_name}] hyperparameter search: "
                  f"{len(grid)} combinations × {cv_batches} folds")

        sample_df = self._collect_sample(source, registry, sample_rows)
        if sample_df.empty:
            raise RuntimeError("hyperparameter_search: no valid rows in sample")

        split   = len(sample_df) // 2
        results = []

        for params in grid:
            rmses = []
            for fold in range(cv_batches):
                shuf  = sample_df.sample(frac=1, random_state=fold).reset_index(drop=True)
                train = shuf.iloc[:split]
                evald = shuf.iloc[split:]
                cand  = self._clone_with_params(params)
                cand.fit_batch(train, eval=False)
                if not cand._is_fitted:
                    continue
                m = cand.evaluate(evald)
                rmses.append(m.get("rmse", float("inf")))

            mean_r = float(np.mean(rmses))  if rmses else float("inf")
            std_r  = float(np.std(rmses))   if rmses else float("inf")
            results.append({"params": params, "mean_rmse": mean_r, "std_rmse": std_r})
            if verbose:
                print(f"  {params}  RMSE={mean_r:.4f} ± {std_r:.4f}")

        results.sort(key=lambda r: r["mean_rmse"])
        best = results[0]
        if verbose:
            print(f"\n  ✓ best: {best['params']}  RMSE={best['mean_rmse']:.4f}")
        self._apply_params(best["params"])
        return {"best_params": best["params"], "best_rmse": best["mean_rmse"], "all_results": results}

    def _collect_sample(self, source, registry, n_rows) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            return source.head(n_rows)
        if hasattr(source, "stream"):
            gen = source.stream(registry=registry, batch_size=10_000, max_rows=n_rows)
        else:
            gen = source
        chunks, total = [], 0
        for b in gen:
            chunks.append(b)
            total += len(b)
            if total >= n_rows:
                break
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    def _clone_with_params(self, params: Dict[str, Any]) -> "LSTModel":
        clone = self.__class__(
            feature_cols = self.feature_cols,
            target_col   = self.target_col,
            random_state = self.random_state,
        )
        clone._apply_params(params)
        clone._col_medians           = dict(self._col_medians)
        clone._resolved_feature_cols = list(self._resolved_feature_cols)
        clone._transforms            = list(self._transforms)
        if self._scaler is not None:
            clone._scaler = self._scaler   # shared read-only scaler is fine
        return clone

    def _apply_params(self, params: Dict[str, Any]) -> None:
        """Set params on self._model.  Override for custom handling."""
        for k, v in params.items():
            if hasattr(self._model, k):
                setattr(self._model, k, v)
            else:
                log.warning("%s: model has no attr '%s'", self.model_name, k)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the full wrapper to disk.

        Uses cloudpickle so that transform closures (from factories such as
        cyclical(), log1p_transform(), etc.) survive serialisation.  The
        StreamingScaler stored in self._scaler may also hold closures; stdlib
        pickle / joblib cannot handle these, but cloudpickle can.

        The file format is a raw cloudpickle byte stream.  Load it back with
        LSTModel.load() — do NOT use joblib.load() directly.
        """
        import cloudpickle

        path = Path(path)
        if path.suffix not in (".pkl", ".joblib"):
            path = path.with_suffix(".pkl")
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "class":                 self.__class__.__name__,
            "feature_cols":          self.feature_cols,
            "target_col":            self.target_col,
            "random_state":          self.random_state,
            "resolved_feature_cols": self._resolved_feature_cols,
            "col_medians":           self._col_medians,
            "scaler":                self._scaler,
            "transforms":            self._transforms,
            "training_history":      self._training_history,
            "aggregate_metrics":     self._aggregate_metrics,
            "cumulative_rows":       self._cumulative_rows,
            "is_fitted":             self._is_fitted,
            "shap_background":       self._shap_background,
            "model":                 self._model,
            # Subclass-specific extra state (e.g. Nystroem transformer,
            # reservoir object, booster, accumulated importance arrays).
            # Each subclass that needs this overrides _get_extra_state().
            "extra_state":           self._get_extra_state(),
        }
        path.write_bytes(cloudpickle.dumps(state))
        log.info("%s: saved to %s", self.model_name, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LSTModel":
        """
        Load a saved model from disk.

        Works for any subclass; the class name embedded in the file is used
        to reconstruct the correct type via ModelRegistry.
        """
        import cloudpickle
        from ml.registry import ModelRegistry

        path  = Path(path)
        state = cloudpickle.loads(path.read_bytes())
        klass = ModelRegistry.get(state["class"])
        obj   = klass(
            feature_cols = state["feature_cols"],
            target_col   = state["target_col"],
            random_state = state["random_state"],
        )
        obj._resolved_feature_cols = state["resolved_feature_cols"]
        obj._col_medians           = state["col_medians"]
        obj._scaler                = state.get("scaler")
        obj._transforms            = state.get("transforms", [])
        obj._training_history      = state["training_history"]
        obj._aggregate_metrics     = state.get("aggregate_metrics", {}) or {}
        obj._cumulative_rows       = state["cumulative_rows"]
        obj._is_fitted             = state["is_fitted"]
        obj._shap_background       = state.get("shap_background")
        obj._model                 = state["model"]
        obj._set_extra_state(state.get("extra_state", {}))
        log.info("%s: loaded from %s", obj.model_name, path)
        return obj

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(
        self,
        format:  str = "stdout",
        path:    Optional[Union[str, Path]] = None,
        eval_df: Optional[pd.DataFrame] = None,
        n_shap:  int = 200,
    ) -> Optional[str]:
        """
        Generate a model report.

        Parameters
        ----------
        format : "stdout" | "html"
        path : write HTML to file when provided
        eval_df : optional held-out data for separate evaluation metrics
        n_shap : rows to explain with SHAP in the HTML report

        Exceptions raised during HTML generation (e.g. SHAP OOM, matplotlib
        errors) are caught per-section so a failure in one section never
        prevents the rest of the report or the caller's loop from continuing.
        """
        self._check_fitted()
        metrics_train = self._last_training_metrics()

        metrics_eval = None
        if eval_df is not None:
            try:
                metrics_eval = self.evaluate(eval_df)
            except Exception as exc:
                log.warning("%s: evaluate() failed in report: %s", self.model_name, exc)

        try:
            importance = self.feature_importance()
        except Exception as exc:
            log.warning("%s: feature_importance() failed in report: %s", self.model_name, exc)
            importance = {}

        if format == "stdout":
            self._report_stdout(metrics_train, metrics_eval, importance)
            return None
        elif format == "html":
            try:
                html = self._report_html(metrics_train, metrics_eval, importance, n_shap)
            except Exception as exc:
                log.error("%s: _report_html() failed: %s", self.model_name, exc, exc_info=True)
                html = (f"<html><body><h1>{self.model_name} — Report Error</h1>"
                        f"<pre>{exc}</pre></body></html>")
            if path is not None:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(html, encoding="utf-8")
                log.info("Report written to %s", path)
                return None
            return html
        else:
            raise ValueError(f"Unknown format '{format}'. Use 'stdout' or 'html'.")

    def _last_training_metrics(self) -> Dict[str, float]:
        # Prefer stream-wide aggregate (set by fit_stream) over the last batch,
        # because per-batch R² collapses on low-variance batches.
        if self._aggregate_metrics:
            m = dict(self._aggregate_metrics)
            m["n_rows"] = self._cumulative_rows
            return m
        if not self._training_history:
            return {}
        last = self._training_history[-1]
        return {"rmse": last.rmse, "mae": last.mae, "r2": last.r2,
                "mse": last.mse, "n_rows": self._cumulative_rows}

    def _report_stdout(self, m_train, m_eval, importance) -> None:
        w   = 70
        sep = "─" * w
        S   = lambda t: (print(f"\n{sep}"), print(f"  {t}"), print(sep))

        print(f"\n{'═'*w}")
        print(f"  {self.model_name} — LST Temperature Regression Report")
        print(f"  Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"{'═'*w}")
        S("Model")
        print(f"  Class            : {self.__class__.__name__}")
        print(f"  Target           : {self.target_col}")
        print(f"  Features         : {len(self._resolved_feature_cols)}")
        print(f"  Rows trained on  : {self._cumulative_rows:,}")
        print(f"  Training batches : {len(self._training_history)}")
        if self._transforms:
            names = [getattr(fn, "__name__", str(fn)) for fn in self._transforms]
            print(f"  Transforms       : {', '.join(names)}")

        def _table(m, title):
            if not m:
                return
            S(title)
            for k, v in m.items():
                print(f"  {k:<20s}: {v:,.0f}" if k == "n_rows" else f"  {k:<20s}: {v:.4f}")

        _table(m_train, "Training Metrics (stream aggregate)")
        _table(m_eval,  "Evaluation Metrics (held-out)")

        S("Feature Importance (top 20)")
        top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
        mx  = top[0][1] if top else 1.0
        for name, imp in top:
            bar = "█" * int(30 * imp / mx)
            print(f"  {name:<40s} {imp:>7.4f}  {bar}")

        if self._training_history:
            S("Training History")
            hist = self._training_history
            step = max(1, len(hist) // 20)
            for rec in hist[::step]:
                r2_clamped = max(0.0, min(rec.r2, 0.999))
                bar = "░" * max(1, int(30 * (1.0 - r2_clamped)))
                r2_str = f"{rec.r2:>6.3f}" if -1e6 < rec.r2 < 1e6 else f"{rec.r2:>10.2e}"
                print(f"  batch {rec.batch_idx:>4d} | rows {rec.cumulative_rows:>8,} | "
                      f"RMSE {rec.rmse:>6.3f} | R² {r2_str}  {bar}")
        print(f"\n{'═'*w}\n")

    def _report_html(self, m_train, m_eval, importance, n_shap) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def _svg(fig) -> str:
            buf = io.BytesIO()
            fig.savefig(buf, format="svg", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return buf.read().decode("utf-8")

        # Feature importance chart
        top   = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
        names = [k for k, _ in top]
        vals  = [v for _, v in top]
        fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.4)))
        bars = ax.barh(names[::-1], vals[::-1], color="#4C72B0")
        ax.set_xlabel("Normalised importance")
        ax.set_title("Feature Importance")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        ax.margins(x=0.15)
        fi_svg = _svg(fig)

        # Training history chart
        hist_svg = ""
        if len(self._training_history) >= 2:
            hist  = self._training_history
            rows  = [r.cumulative_rows for r in hist]
            rmses = [r.rmse for r in hist]
            r2s   = [r.r2   for r in hist]
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
            a1.plot(rows, rmses, color="#DD4444", linewidth=1.5)
            a1.set(xlabel="Cumulative rows", ylabel="RMSE (°C)", title="RMSE over training")
            a1.grid(alpha=0.3)
            a2.plot(rows, r2s, color="#44AA44", linewidth=1.5)
            a2.set(xlabel="Cumulative rows", ylabel="R²", title="R² over training")
            a2.grid(alpha=0.3)
            hist_svg = _svg(fig)

        # SHAP chart — using shap_values() directly on background sample
        shap_svg = ""
        if _HAS_SHAP and self._shap_background is not None:
            try:
                X_bg = self._shap_background[:n_shap]
                sv   = self._shap_values(X_bg)

                # Normalise whatever _shap_values returns to a plain 2-D array.
                # Some backends return list-of-arrays (multi-output), 3-D arrays
                # (e.g. older SHAP versions with interaction values), or Explanation
                # objects.
                if isinstance(sv, list):
                    sv = sv[0]
                sv = np.asarray(sv)
                if sv.ndim == 3:
                    sv = sv.mean(axis=2)

                cols = self._resolved_feature_cols
                mean_abs = np.abs(sv).mean(axis=0)
                idx  = np.argsort(mean_abs)
                fig, ax = plt.subplots(figsize=(9, max(4, len(cols) * 0.38)))
                ax.barh([cols[i] for i in idx], [mean_abs[i] for i in idx], color="#9467BD")
                ax.set(xlabel="Mean |SHAP| (°C)", title="SHAP Feature Impact")
                shap_svg = _svg(fig)
            except Exception as exc:
                log.warning("SHAP chart failed for %s: %s", self.model_name, exc)

        def _tbl(m, title):
            if not m:
                return ""
            rows = "".join(
                f"<tr><td>{k}</td><td>{f'{v:,.0f}' if k == 'n_rows' else f'{v:.4f}'}</td></tr>"
                for k, v in m.items()
            )
            return (f"<h3>{title}</h3><table class='metrics'>"
                    f"<tr><th>Metric</th><th>Value</th></tr>{rows}</table>")

        fi_rows = "".join(
            f"<tr><td>{n}</td><td>{v:.4f}</td></tr>" for n, v in top
        )
        hist_rows = ""
        if self._training_history:
            step = max(1, len(self._training_history) // 50)
            for r in self._training_history[::step]:
                hist_rows += (f"<tr><td>{r.batch_idx}</td><td>{r.cumulative_rows:,}</td>"
                              f"<td>{r.rmse:.4f}</td><td>{r.mae:.4f}</td>"
                              f"<td>{r.r2:.4f}</td><td>{r.elapsed_s:.2f}</td></tr>")

        transforms_info = ""
        if self._transforms:
            names = [getattr(fn, "__name__", str(fn)) for fn in self._transforms]
            transforms_info = f"<p><strong>Transforms:</strong> {', '.join(names)}</p>"

        return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>{self.model_name} Report</title>
<style>
  body{{font-family:system-ui,sans-serif;margin:2em;color:#222}}
  h1{{border-bottom:2px solid #4C72B0;padding-bottom:.3em}}
  h2{{color:#4C72B0;margin-top:2em}} h3{{color:#555}}
  table.metrics{{border-collapse:collapse;margin:1em 0}}
  table.metrics td,table.metrics th{{border:1px solid #ccc;padding:.4em .8em}}
  table.metrics tr:nth-child(even){{background:#f5f5f5}}
  table.hist td,table.hist th{{border:1px solid #ddd;padding:.3em .6em;font-size:.85em}}
  table.hist tr:nth-child(even){{background:#fafafa}}
  .meta{{color:#666;font-size:.9em}} .chart{{margin:1.5em 0}}
  code{{background:#f0f0f0;padding:.1em .3em;border-radius:3px}}
</style></head><body>
<h1>{self.model_name} — LST Temperature Regression Report</h1>
<p class="meta">
  Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}<br>
  Class: <code>{self.__class__.__name__}</code> |
  Target: <code>{self.target_col}</code> |
  Features: {len(self._resolved_feature_cols)} |
  Rows: {self._cumulative_rows:,} |
  Batches: {len(self._training_history)}
</p>
{transforms_info}
<h2>Metrics</h2>
{_tbl(m_train, "Training (stream aggregate)")}{_tbl(m_eval or {}, "Evaluation (held-out)")}
<h2>Feature Importance</h2>
<div class="chart">{fi_svg}</div>
<table class="metrics"><tr><th>Feature</th><th>Importance</th></tr>{fi_rows}</table>
<h2>Training History</h2>
{"<div class='chart'>" + hist_svg + "</div>" if hist_svg else "<p><em>Single-batch training — no history curve.</em></p>"}
<table class="hist"><tr><th>Batch</th><th>Rows</th><th>RMSE</th><th>MAE</th><th>R²</th><th>s</th></tr>
{hist_rows}</table>
{"<h2>SHAP Impact</h2><div class='chart'>" + shap_svg + "</div>" if shap_svg else ""}
<h2>Feature List</h2><code>{", ".join(self._resolved_feature_cols)}</code>
</body></html>"""

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.model_name} has not been fitted.  "
                "Call fit_stream() or fit_batch() first."
            )

    def summary(self) -> Dict[str, Any]:
        # Prefer aggregate over last-batch for the headline numbers.
        last = self._training_history[-1] if self._training_history else None
        if self._aggregate_metrics:
            rmse_val = self._aggregate_metrics.get("rmse")
            r2_val   = self._aggregate_metrics.get("r2")
            mae_val  = self._aggregate_metrics.get("mae")
        else:
            rmse_val = last.rmse if last else None
            r2_val   = last.r2   if last else None
            mae_val  = last.mae  if last else None
        return {
            "model":        self.model_name,
            "class":        self.__class__.__name__,
            "fitted":       self._is_fitted,
            "rows_trained": self._cumulative_rows,
            "n_features":   len(self._resolved_feature_cols),
            "last_rmse":    rmse_val,
            "last_r2":      r2_val,
            "last_mae":     mae_val,
        }

    def __repr__(self) -> str:
        s    = self.summary()
        rmse = f"{s['last_rmse']:.3f}" if s["last_rmse"] is not None else "n/a"
        return (f"{s['class']}(fitted={s['fitted']}, "
                f"rows={s['rows_trained']:,}, features={s['n_features']}, rmse={rmse})")
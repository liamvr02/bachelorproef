"""
ml/models/boosting.py
=====================
Gradient boosting models for LST temperature regression.

LSTXGBoost
----------
Uses XGBoost's incremental booster continuation: each call to fit() with
xgb_model=<previous_booster> adds n_estimators_per_batch new trees to the
existing ensemble without modifying earlier trees.  The booster accumulates
knowledge monotonically across all batches — this is genuine streaming.

Explainability: gain-based importance + SHAP via TreeExplainer (exact, fast).

LSTHistGradientBoosting
-----------------------
sklearn's HistGradientBoostingRegressor does NOT support true streaming:
warm_start across disjoint batches fits the new trees on the residuals of
whatever batch happens to be passed in, producing an incoherent patchwork
ensemble whose predictions diverge over time (we measured RMSE ~200 with
R² in the thousands-negative range on the previous implementation).

The correct strategy — matching LSTRandomForest / LSTExtraTrees — is
reservoir sampling: stream the full source, keep a uniform i.i.d. sample
sized to available RAM, then fit HGB once on the reservoir.  Permutation
importance is computed once on the reservoir as well.

Explainability: permutation importance (single pass on reservoir) +
SHAP via TreeExplainer.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import shap as _shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from ml.base import LSTModel
from ml.models.forest import _reservoir_fit_stream

import logging
log = logging.getLogger("lst_models.boosting")


# ---------------------------------------------------------------------------
# LSTXGBoost
# ---------------------------------------------------------------------------

class LSTXGBoost(LSTModel):
    """
    XGBoost regressor — true incremental streaming via booster continuation.

    Each training batch adds n_estimators_per_batch new trees to the
    cumulative booster without modifying previously fitted trees.  This is
    XGBoost's built-in incremental learning mode (xgb_model parameter).

    Explainability
    --------------
    Gain-based feature importance (get_score(importance_type="gain")).
    SHAP via shap.TreeExplainer — exact attribution, efficient for trees.

    Hyperparameters
    ---------------
    n_estimators_per_batch : trees added per batch (default 20)
    max_depth              : tree depth (default 6)
    learning_rate          : boosting shrinkage (default 0.1)
    subsample              : row sample per tree (default 0.8)
    colsample_bytree       : feature sample per tree (default 0.8)
    reg_alpha              : L1 regularisation
    reg_lambda             : L2 regularisation
    min_child_weight       : minimum sum of instance Hessian per leaf (default 1.0)
    gamma                  : minimum loss reduction for a split (default 0.0)
    """

    NEEDS_SCALING = False  # gradient boosted trees are scale-invariant

    def __init__(
        self,
        n_estimators_per_batch: int   = 20,
        max_depth:              int   = 6,
        learning_rate:          float = 0.1,
        subsample:              float = 0.8,
        colsample_bytree:       float = 0.8,
        reg_alpha:              float = 0.0,
        reg_lambda:             float = 1.0,
        min_child_weight:       float = 1.0,
        gamma:                  float = 0.0,
        **kwargs,
    ) -> None:
        if not _HAS_XGB:
            raise ImportError("xgboost is required.  pip install xgboost")
        self._n_per_batch     = n_estimators_per_batch
        self._max_depth       = max_depth
        self._lr              = learning_rate
        self._subsample       = subsample
        self._colsample       = colsample_bytree
        self._reg_alpha       = reg_alpha
        self._reg_lambda      = reg_lambda
        self._min_child_weight = min_child_weight
        self._gamma           = gamma
        self._booster         = None   # accumulated XGBoost Booster
        super().__init__(**kwargs)

    @property
    def model_name(self) -> str:
        return "XGBoost"

    # Gradient boosting on trees is split-threshold invariant — scaling wastes
    # compute and makes gain-based importance harder to interpret.
    NEEDS_SCALING: bool = False

    @property
    def default_hyperparameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "max_depth":     [4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample":     [0.7, 0.8, 1.0],
            "reg_lambda":    [0.5, 1.0, 5.0],
        }

    def _init_model(self) -> None:
        self._model = xgb.XGBRegressor(
            n_estimators     = getattr(self, "_n_per_batch",       20),
            max_depth        = getattr(self, "_max_depth",         6),
            learning_rate    = getattr(self, "_lr",                0.1),
            subsample        = getattr(self, "_subsample",         0.8),
            colsample_bytree = getattr(self, "_colsample",         0.8),
            reg_alpha        = getattr(self, "_reg_alpha",         0.0),
            reg_lambda       = getattr(self, "_reg_lambda",        1.0),
            min_child_weight = getattr(self, "_min_child_weight",  1.0),
            gamma            = getattr(self, "_gamma",             0.0),
            tree_method      = "hist",
            random_state     = getattr(self, "random_state",       42),
            verbosity        = 0,
            n_jobs           = -1,
        )
        self._booster = None

    def _partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Add n_estimators_per_batch trees to the accumulated booster."""
        fit_kwargs: Dict[str, Any] = {"verbose": False}
        if self._booster is not None:
            fit_kwargs["xgb_model"] = self._booster
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X, y, **fit_kwargs)
        # Persist the booster — do NOT set feature_names because XGBoost
        # then enforces named input on every subsequent predict/fit call,
        # which conflicts with our plain ndarray pipeline.
        self._booster = self._model.get_booster()

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X, validate_features=False)

    def _feature_importance_raw(self) -> np.ndarray:
        booster = self._model.get_booster()
        score   = booster.get_score(importance_type="gain")
        cols    = self._resolved_feature_cols
        fi      = np.zeros(len(cols), dtype=np.float64)

        for fname, val in score.items():
            # Without feature_names set, XGBoost uses "f0", "f1", ... indices
            if fname.startswith("f"):
                try:
                    idx = int(fname[1:])
                    if idx < len(cols):
                        fi[idx] = val
                except ValueError:
                    pass
            elif fname in cols:
                fi[cols.index(fname)] = val
        return fi

    def _shap_values(self, X: np.ndarray) -> np.ndarray:
        if _HAS_SHAP:
            # Use old-API .shap_values() which always returns plain ndarray
            exp = _shap.TreeExplainer(self._model)
            return np.array(exp.shap_values(X))
        # Native XGBoost pred_contribs fallback
        try:
            dmat     = xgb.DMatrix(X)
            contribs = self._booster.predict(dmat, pred_contribs=True)
            return contribs[:, :-1]   # drop bias column
        except Exception:
            fi = self._feature_importance_raw()
            return (X - X.mean(axis=0)) * fi[np.newaxis, :]

    def _apply_params(self, params: Dict[str, Any]) -> None:
        param_map = {
            "n_estimators_per_batch": "_n_per_batch",
            "max_depth":              "_max_depth",
            "learning_rate":          "_lr",
            "subsample":              "_subsample",
            "colsample_bytree":       "_colsample",
            "reg_lambda":             "_reg_lambda",
            "reg_alpha":              "_reg_alpha",
            "min_child_weight":       "_min_child_weight",
            "gamma":                  "_gamma",
        }
        for k, v in params.items():
            if k in param_map:
                setattr(self, param_map[k], v)
            # Map streaming-wrapper alias to sklearn-XGB attr name
            sk_key = "n_estimators" if k == "n_estimators_per_batch" else k
            if hasattr(self._model, sk_key):
                setattr(self._model, sk_key, v)
        # Reset booster so params take effect on the next batch
        self._booster = None

    def _save_impl(self, path: Path) -> None:
        pass

    def _load_impl(self, path: Path) -> None:
        pass

    def _get_extra_state(self) -> dict:
        """Preserve the accumulated booster so incremental training can continue."""
        return {"_booster": self._booster}

    @property
    def total_trees(self) -> int:
        """Total trees in the accumulated booster."""
        if self._booster is None:
            return 0
        return self._booster.num_boosted_rounds()


# ---------------------------------------------------------------------------
# LSTHistGradientBoosting
# ---------------------------------------------------------------------------

class LSTHistGradientBoosting(LSTModel):
    """
    Histogram-based Gradient Boosting regressor, trained on a reservoir sample.

    HGB cannot stream correctly (see module docstring).  We stream the full
    source into a uniform reservoir sample sized to available RAM and then
    call fit() once with the full `max_iter`.  Permutation importance is
    computed in a single pass on the reservoir after fitting.

    Natively handles NaN values, so the preprocessing pipeline's NaN
    imputation is harmless but unnecessary.

    Hyperparameters
    ---------------
    max_iter            : total boosting iterations (default 300)
    max_depth           : tree depth (None = unlimited)
    learning_rate       : shrinkage (default 0.1)
    l2_regularization   : L2 penalty on leaf values
    min_samples_leaf    : minimum samples per leaf (default 20)
    max_bins            : histogram bins per feature (≤255, default 255)
    early_stopping      : "auto" | True | False — sklearn auto picks based on n_samples
    validation_fraction : held-out fraction for early stopping (default 0.1)
    memory_fraction     : fraction of available RAM for the reservoir (default 0.6)
    """

    NEEDS_SCALING = False  # histogram splits are scale-invariant

    def __init__(
        self,
        max_iter:            int                       = 300,
        max_depth:           Optional[int]             = None,
        learning_rate:       float                     = 0.1,
        l2_regularization:   float                     = 0.0,
        min_samples_leaf:    int                       = 20,
        max_bins:            int                       = 255,
        early_stopping:      Any                       = "auto",
        validation_fraction: float                     = 0.1,
        memory_fraction:     float                     = 0.6,
        **kwargs,
    ) -> None:
        self._max_iter            = max_iter
        self._max_depth           = max_depth
        self._lr                  = learning_rate
        self._l2                  = l2_regularization
        self._min_leaf            = min_samples_leaf
        self._max_bins            = max_bins
        self._early_stopping      = early_stopping
        self._validation_fraction = validation_fraction
        self._memory_fraction     = memory_fraction
        self._perm_imp:  Optional[np.ndarray] = None
        self._reservoir: Optional[Any]        = None   # set by _reservoir_fit_stream
        super().__init__(**kwargs)

    @property
    def model_name(self) -> str:
        return "HistGradientBoosting"

    @property
    def default_hyperparameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "max_depth":         [3, 5, 10, None],
            "learning_rate":     [0.05, 0.1, 0.2],
            "l2_regularization": [0.0, 0.1, 1.0],
            "min_samples_leaf":  [10, 20, 50],
        }

    def _init_model(self) -> None:
        # Clamp max_bins to sklearn's 255 hard limit; warn if grid sweeps higher.
        max_bins = int(getattr(self, "_max_bins", 255))
        if max_bins > 255:
            log.warning("HGB: max_bins=%d exceeds sklearn limit of 255 — clamping",
                        max_bins)
            max_bins = 255
        self._model = HistGradientBoostingRegressor(
            max_iter            = getattr(self, "_max_iter",            300),
            max_depth           = getattr(self, "_max_depth",           None),
            learning_rate       = getattr(self, "_lr",                  0.1),
            l2_regularization   = getattr(self, "_l2",                  0.0),
            min_samples_leaf    = getattr(self, "_min_leaf",            20),
            max_bins            = max_bins,
            early_stopping      = getattr(self, "_early_stopping",      "auto"),
            validation_fraction = getattr(self, "_validation_fraction", 0.1),
            random_state        = getattr(self, "random_state",         42),
        )
        self._perm_imp  = None
        self._reservoir = None

    def _partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """No-op: reservoir fit happens once in fit_stream after streaming."""
        if self._reservoir is not None:
            self._reservoir.offer(X, y)

    def fit_stream(
        self,
        source,
        registry         = None,
        batch_size:       int  = 100_000,
        max_rows:         Optional[int] = None,
        eval_every:       int  = 1,
        shap_sample_size: int  = 500,
        verbose:          bool = True,
    ) -> "LSTHistGradientBoosting":
        """Reservoir-then-fit training, with permutation importance computed once."""
        return _reservoir_fit_stream(
            self, source,
            registry         = registry,
            batch_size       = batch_size,
            max_rows         = max_rows,
            shap_sample_size = shap_sample_size,
            verbose          = verbose,
            memory_fraction  = self._memory_fraction,
            post_fit_hook    = self._compute_perm_importance,
        )

    def fit_batch(self, df, eval: bool = True) -> "LSTHistGradientBoosting":
        """Train on a single in-memory DataFrame (single HGB fit, no reservoir)."""
        import time
        from sklearn.metrics import mean_absolute_error, r2_score
        from ml.base import BatchRecord

        t0 = time.perf_counter()
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
        self._compute_perm_importance(X, y)

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
            self._aggregate_metrics = {
                "n_rows": float(len(X)),
                "rmse":   math.sqrt(mse),
                "mae":    mae,
                "r2":     r2,
                "mse":    mse,
            }
        return self

    def _compute_perm_importance(self, X: np.ndarray, y: np.ndarray) -> None:
        """Single-pass permutation importance over up to 200 000 reservoir rows.

        Subsampled for speed — 200 k rows × n_features permutations is already
        the dominant cost after fitting; using the full reservoir would double
        the end-of-training runtime with negligible gain.
        """
        n_eval = min(200_000, len(X))
        rng    = np.random.default_rng(seed=self.random_state)
        idx    = rng.choice(len(X), size=n_eval, replace=False) if len(X) > n_eval \
                 else np.arange(len(X))
        X_ev   = X[idx]
        y_ev   = y[idx]

        baseline = float(mean_squared_error(y_ev, self._model.predict(X_ev)))
        n_feats  = X_ev.shape[1]
        imp      = np.zeros(n_feats, dtype=np.float64)

        for j in range(n_feats):
            X_perm       = X_ev.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            imp[j]       = float(mean_squared_error(y_ev, self._model.predict(X_perm))) - baseline

        self._perm_imp = np.maximum(imp, 0.0)

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def _feature_importance_raw(self) -> np.ndarray:
        if self._perm_imp is not None:
            return self._perm_imp
        n = len(self._resolved_feature_cols) if self._resolved_feature_cols else 1
        return np.ones(n, dtype=np.float64)

    def _shap_values(self, X: np.ndarray) -> np.ndarray:
        if _HAS_SHAP:
            try:
                exp = _shap.TreeExplainer(self._model)
                return np.array(exp.shap_values(X))
            except Exception:
                pass
            try:
                bg  = X[:50]
                exp = _shap.Explainer(self._model.predict, _shap.maskers.Independent(bg))
                sv  = exp(X, silent=True)
                return np.array(sv.values)
            except Exception as exc:
                log.warning("HGB SHAP failed: %s", exc)

        fi = self._feature_importance_raw()
        return (X - X.mean(axis=0)) * (fi / (fi.sum() + 1e-9))[np.newaxis, :]

    def _apply_params(self, params: Dict[str, Any]) -> None:
        param_map = {
            "max_iter":            "_max_iter",
            "max_depth":           "_max_depth",
            "learning_rate":       "_lr",
            "l2_regularization":   "_l2",
            "min_samples_leaf":    "_min_leaf",
            "max_bins":            "_max_bins",
            "early_stopping":      "_early_stopping",
            "validation_fraction": "_validation_fraction",
        }
        for k, v in params.items():
            if k in param_map:
                setattr(self, param_map[k], v)
            if hasattr(self._model, k):
                setattr(self._model, k, v)

    def _save_impl(self, path: Path) -> None:
        pass

    def _load_impl(self, path: Path) -> None:
        pass

    def _get_extra_state(self) -> dict:
        return {"_perm_imp": self._perm_imp}

    @property
    def total_iterations(self) -> int:
        """Total boosting iterations in the fitted model."""
        return int(getattr(self._model, "n_iter_", self._max_iter))
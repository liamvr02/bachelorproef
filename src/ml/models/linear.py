"""
ml/models/linear.py
===================
SGD-based linear regression models for LST temperature prediction.

All models use sklearn's SGDRegressor with partial_fit for true online
learning — each batch updates the model without storing historical data.

Linear models perform poorly when the relationship between features and
temperature is non-linear.  To improve them:
  1. Use cyclical encoding for periodic features (month, hour).
  2. Apply log1p_transform to skewed features (elevation, tree density).
  3. Use polynomial_features for quadratic interactions.
  4. Use LSTNystroemSGD (below) for an implicit kernel expansion.

These transforms are applied via model.set_transforms([...]) before training.

Explainability
--------------
All models expose signed coefficients (via .coefficients()) and normalised
absolute magnitudes as feature importance.  SHAP uses LinearExplainer for
exact attribution.

Models
------
LSTLinearRegression  — standard L2-penalised SGD (squared error loss)
LSTRidgeRegression   — explicit Ridge alias (same as linear, different default α)
LSTElasticNet        — L1 + L2 via ElasticNet penalty
LSTHuberRegression   — Huber loss for outlier-robustness
LSTSGDRegressor      — fully configurable SGDRegressor wrapper
LSTNystroemSGD       — RBF kernel approx (Nystroem) + SGDRegressor
                       for non-linear predictions with linear complexity
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import shap as _shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDRegressor

from ml.base import LSTModel


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _shap_linear(model, X: np.ndarray, background: Optional[np.ndarray]) -> np.ndarray:
    """Compute SHAP values for any SGDRegressor."""
    if _HAS_SHAP:
        bg = background if background is not None else X
        explainer = _shap.LinearExplainer(model, bg)
        return np.array(explainer.shap_values(X))
    # Fallback: coefficient × (feature - background mean)
    coef = model.coef_
    mean = (background.mean(axis=0) if background is not None
            else np.zeros(len(coef), dtype=np.float32))
    return (X - mean) * coef[np.newaxis, :]


# ---------------------------------------------------------------------------
# LSTLinearRegression
# ---------------------------------------------------------------------------

class LSTLinearRegression(LSTModel):
    """
    Online linear regression via SGDRegressor with L2 penalty.

    True streaming model — partial_fit is called per batch with no
    accumulated data.  Converges quickly but captures only linear
    relationships; pair with cyclical and polynomial transforms for
    better performance.

    Hyperparameters
    ---------------
    alpha   : regularisation strength (default 1e-4)
    loss    : "squared_error" (L2) or "huber" (robust)
    penalty : "l2" | "l1" | "elasticnet"
    """

    def __init__(
        self,
        alpha:   float = 1e-4,
        loss:    str   = "squared_error",
        penalty: str   = "l2",
        **kwargs,
    ) -> None:
        self._alpha   = alpha
        self._loss    = loss
        self._penalty = penalty
        super().__init__(**kwargs)

    @property
    def model_name(self) -> str:
        return "Linear Regression (SGD)"

    @property
    def default_hyperparameter_grid(self) -> Dict[str, List[Any]]:
        return {"alpha": [1e-5, 1e-4, 1e-3, 1e-2], "penalty": ["l2", "l1", "elasticnet"]}

    def _init_model(self) -> None:
        self._model = SGDRegressor(
            loss         = getattr(self, "_loss",    "squared_error"),
            penalty      = getattr(self, "_penalty", "l2"),
            alpha        = getattr(self, "_alpha",   1e-4),
            max_iter     = 1,
            tol          = None,
            warm_start   = True,
            random_state = getattr(self, "random_state", 42),
        )

    def _partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.partial_fit(X, y)

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def _feature_importance_raw(self) -> np.ndarray:
        return np.abs(self._model.coef_)

    def _shap_values(self, X: np.ndarray) -> np.ndarray:
        return _shap_linear(self._model, X, self._shap_background)

    def _save_impl(self, path: Path) -> None:
        pass

    def _load_impl(self, path: Path) -> None:
        pass

    def coefficients(self) -> Dict[str, float]:
        """Return the raw signed coefficients per feature."""
        self._check_fitted()
        return dict(zip(self._resolved_feature_cols, self._model.coef_.tolist()))


# ---------------------------------------------------------------------------
# LSTRidgeRegression
# ---------------------------------------------------------------------------

class LSTRidgeRegression(LSTLinearRegression):
    """Ridge regression (L2) — explicit alias with a larger default alpha."""

    def __init__(self, alpha: float = 1e-3, **kwargs) -> None:
        super().__init__(alpha=alpha, penalty="l2", **kwargs)

    @property
    def model_name(self) -> str:
        return "Ridge Regression (SGD)"

    @property
    def default_hyperparameter_grid(self) -> Dict[str, List[Any]]:
        return {"alpha": [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]}


# ---------------------------------------------------------------------------
# LSTElasticNet
# ---------------------------------------------------------------------------

class LSTElasticNet(LSTLinearRegression):
    """
    Elastic Net (L1 + L2) online regression.

    Encourages feature sparsity (L1) while remaining well-conditioned (L2).
    Useful when many engineered features are redundant.

    Hyperparameters
    ---------------
    alpha     : overall regularisation strength
    l1_ratio  : fraction of penalty that is L1 (0=Ridge, 1=Lasso)
    """

    def __init__(self, alpha: float = 1e-3, l1_ratio: float = 0.15, **kwargs) -> None:
        super().__init__(alpha=alpha, penalty="elasticnet", **kwargs)
        self._l1_ratio = l1_ratio

    @property
    def model_name(self) -> str:
        return "Elastic Net (SGD)"

    @property
    def default_hyperparameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "alpha":    [1e-4, 1e-3, 1e-2, 0.1],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }

    def _init_model(self) -> None:
        self._model = SGDRegressor(
            loss         = "squared_error",
            penalty      = "elasticnet",
            alpha        = getattr(self, "_alpha",    1e-3),
            l1_ratio     = getattr(self, "_l1_ratio", 0.15),
            max_iter     = 1,
            tol          = None,
            warm_start   = True,
            random_state = getattr(self, "random_state", 42),
        )


# ---------------------------------------------------------------------------
# LSTHuberRegression
# ---------------------------------------------------------------------------

class LSTHuberRegression(LSTLinearRegression):
    """
    Huber loss SGD regression.

    Robust to outliers (LST can contain cloud-contaminated extremes).
    The epsilon parameter controls the transition from L2 (quadratic) to
    L1 (linear) loss — larger ε = more tolerance for outliers.

    Hyperparameters
    ---------------
    alpha   : regularisation strength
    epsilon : outlier tolerance (Huber transition point, default 1.35)
    """

    def __init__(self, alpha: float = 1e-4, epsilon: float = 1.35, **kwargs) -> None:
        self._epsilon = epsilon
        super().__init__(alpha=alpha, loss="huber", penalty="l2", **kwargs)

    @property
    def model_name(self) -> str:
        return "Huber Regression (SGD)"

    @property
    def default_hyperparameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "alpha":   [1e-5, 1e-4, 1e-3],
            "epsilon": [1.1, 1.35, 1.5, 2.0],
        }

    def _init_model(self) -> None:
        self._model = SGDRegressor(
            loss         = "huber",
            penalty      = "l2",
            alpha        = getattr(self, "_alpha",   1e-4),
            epsilon      = getattr(self, "_epsilon", 1.35),
            max_iter     = 1,
            tol          = None,
            warm_start   = True,
            random_state = getattr(self, "random_state", 42),
        )


# ---------------------------------------------------------------------------
# LSTSGDRegressor — fully configurable SGD wrapper
# ---------------------------------------------------------------------------

class LSTSGDRegressor(LSTLinearRegression):
    """
    Fully configurable SGDRegressor wrapper.

    Exposes the full sklearn SGDRegressor parameter set.  Use this when
    you want to experiment with exotic loss/penalty combinations or
    specific learning rate schedules.

    Hyperparameters
    ---------------
    loss            : "squared_error" | "huber" | "epsilon_insensitive" | ...
    penalty         : "l2" | "l1" | "elasticnet"
    alpha           : regularisation strength
    l1_ratio        : ElasticNet mixing parameter
    epsilon         : Huber / epsilon-insensitive threshold
    learning_rate   : "optimal" | "invscaling" | "constant" | "adaptive"
    eta0            : initial learning rate (for "constant" / "invscaling")
    power_t         : exponent for "invscaling" schedule
    """

    def __init__(
        self,
        loss:          str   = "squared_error",
        penalty:       str   = "l2",
        alpha:         float = 1e-4,
        l1_ratio:      float = 0.15,
        epsilon:       float = 0.1,
        learning_rate: str   = "invscaling",
        eta0:          float = 0.01,
        power_t:       float = 0.25,
        **kwargs,
    ) -> None:
        self._sgd_loss          = loss
        self._sgd_penalty       = penalty
        self._sgd_alpha         = alpha
        self._sgd_l1_ratio      = l1_ratio
        self._sgd_epsilon       = epsilon
        self._sgd_learning_rate = learning_rate
        self._sgd_eta0          = eta0
        self._sgd_power_t       = power_t
        # Don't pass loss/penalty to parent — it overrides them
        super(LSTLinearRegression, self).__init__(**kwargs)
        self._init_model()

    @property
    def model_name(self) -> str:
        return f"SGD({self._sgd_loss}/{self._sgd_penalty})"

    @property
    def default_hyperparameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "alpha":         [1e-5, 1e-4, 1e-3, 1e-2],
            "learning_rate": ["optimal", "invscaling", "adaptive"],
            "eta0":          [0.001, 0.01, 0.1],
        }

    def _init_model(self) -> None:
        self._model = SGDRegressor(
            loss          = getattr(self, "_sgd_loss",          "squared_error"),
            penalty       = getattr(self, "_sgd_penalty",       "l2"),
            alpha         = getattr(self, "_sgd_alpha",         1e-4),
            l1_ratio      = getattr(self, "_sgd_l1_ratio",      0.15),
            epsilon       = getattr(self, "_sgd_epsilon",       0.1),
            learning_rate = getattr(self, "_sgd_learning_rate", "invscaling"),
            eta0          = getattr(self, "_sgd_eta0",          0.01),
            power_t       = getattr(self, "_sgd_power_t",       0.25),
            max_iter      = 1,
            tol           = None,
            warm_start    = True,
            random_state  = getattr(self, "random_state", 42),
        )

    def _apply_params(self, params: Dict[str, Any]) -> None:
        for k, v in params.items():
            attr = f"_sgd_{k}"
            if hasattr(self, attr):
                setattr(self, attr, v)
            if hasattr(self._model, k):
                setattr(self._model, k, v)


# ---------------------------------------------------------------------------
# LSTNystroemSGD — non-linear via kernel approximation
# ---------------------------------------------------------------------------

class LSTNystroemSGD(LSTModel):
    """
    Non-linear regression: Nystroem RBF kernel approximation + SGDRegressor.

    Approximates a kernelised support vector regressor at O(N * n_components)
    cost instead of O(N²).  The Nystroem transformer maps input features into
    a higher-dimensional space where linear SGD can capture non-linear
    temperature–feature relationships without explicit polynomial expansion.

    The Nystroem transformer is fitted once on the first batch (a sample of
    n_components training points is drawn) and held fixed thereafter.  The
    SGDRegressor continues to update every batch via partial_fit.

    Explainability: uses LinearExplainer on the Nystroem-transformed space.
    Importance values represent feature contributions in the kernel space;
    they are less directly interpretable than tree importances but still
    valid for relative feature comparison.

    Hyperparameters
    ---------------
    n_components  : number of Nystroem basis functions (default 300)
    gamma         : RBF kernel bandwidth (default "scale" = 1/n_features)
    alpha         : SGD L2 regularisation
    """

    def __init__(
        self,
        n_components: int   = 300,
        gamma:        Any   = "scale",
        alpha:        float = 1e-3,
        **kwargs,
    ) -> None:
        self._n_components = n_components
        self._gamma        = gamma
        self._sgd_alpha    = alpha
        self._nystroem:    Optional[Nystroem] = None
        self._nys_fitted:  bool = False
        super().__init__(**kwargs)

    @property
    def model_name(self) -> str:
        return f"Nystroem-SGD(k={self._n_components})"

    @property
    def default_hyperparameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "n_components": [100, 200, 300, 500],
            "alpha":        [1e-4, 1e-3, 1e-2],
        }

    def _init_model(self) -> None:
        gamma = getattr(self, "_gamma", "scale")
        n     = getattr(self, "_n_components", 300)
        self._model = SGDRegressor(
            alpha        = getattr(self, "_sgd_alpha", 1e-3),
            max_iter     = 1,
            tol          = None,
            warm_start   = True,
            random_state = getattr(self, "random_state", 42),
        )
        self._nystroem  = Nystroem(
            kernel       = "rbf",
            gamma        = None if gamma == "scale" else gamma,
            n_components = n,
            random_state = getattr(self, "random_state", 42),
        )
        self._nys_fitted = False

    def _transform_nystroem(self, X: np.ndarray) -> np.ndarray:
        """Fit (first call only) and transform via Nystroem."""
        if not self._nys_fitted:
            # Set gamma = 1/n_features if "scale"
            if getattr(self, "_gamma", "scale") == "scale":
                self._nystroem.gamma = 1.0 / X.shape[1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._nystroem.fit(X)
            self._nys_fitted = True
        return self._nystroem.transform(X)

    def _partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xt = self._transform_nystroem(X)
        self._model.partial_fit(Xt, y)

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        if not self._nys_fitted:
            return np.zeros(len(X), dtype=np.float32)
        return self._model.predict(self._nystroem.transform(X))

    def _feature_importance_raw(self) -> np.ndarray:
        # Map kernel-space coefficients back to input space via Nystroem components
        coef = np.abs(self._model.coef_)                   # shape: (n_components,)
        comp = self._nystroem.components_                  # shape: (n_components, n_features)
        # Weighted sum of absolute component values
        importance = (np.abs(comp) * coef[:, np.newaxis]).mean(axis=0)
        return importance

    def _shap_values(self, X: np.ndarray) -> np.ndarray:
        if not self._nys_fitted:
            return np.zeros((len(X), len(self._resolved_feature_cols)))
        Xt = self._nystroem.transform(X)
        if _HAS_SHAP:
            bg  = (self._nystroem.transform(self._shap_background[:50])
                   if self._shap_background is not None
                   else Xt[:50])
            exp = _shap.LinearExplainer(self._model, bg)
            sv_kernel = np.array(exp.shap_values(Xt))   # shape: (n, n_components)
        else:
            coef      = self._model.coef_
            mean_bg   = Xt.mean(axis=0)
            sv_kernel = (Xt - mean_bg) * coef[np.newaxis, :]

        # Project kernel SHAP back to input space via Nystroem components
        comp = self._nystroem.components_   # (n_components, n_features)
        sv   = sv_kernel @ comp             # (n_samples, n_features)
        return sv

    def _apply_params(self, params: Dict[str, Any]) -> None:
        for k, v in params.items():
            if k == "n_components":
                self._n_components = v
                self._nystroem.n_components = v
            elif k == "alpha":
                self._sgd_alpha = v
                self._model.alpha = v
            elif hasattr(self._model, k):
                setattr(self._model, k, v)

    def fit_batch(self, df: pd.DataFrame, eval: bool = True, chunk_size: int = 50_000) -> "LSTNystroemSGD":
        """
        Train on a DataFrame by splitting into chunks.

        Nystroem.transform() materialises an (n_rows × n_components) float32
        matrix; at n_rows=15M and n_components=300 this is ~17 GiB.  Chunking
        limits peak memory to chunk_size × n_components × 4 bytes
        (≈ 57 MB at chunk_size=50k, n_components=300).
        """
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start:start + chunk_size]
            super().fit_batch(chunk, eval=False)

        # Record a single history entry evaluated on the full df (subsampled)
        if eval and self._is_fitted:
            import math as _math
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            sample = df.sample(min(10_000, len(df)), random_state=42)
            X, y = self._preprocess(sample, fit=False)
            if len(X) > 0:
                y_hat = self._predict_raw(X)
                mse   = float(mean_squared_error(y, y_hat))
                from ml.base import BatchRecord
                import time
                self._training_history.append(BatchRecord(
                    batch_idx       = len(self._training_history),
                    n_rows          = self._cumulative_rows,
                    mse             = mse,
                    rmse            = _math.sqrt(mse),
                    mae             = float(mean_absolute_error(y, y_hat)),
                    r2              = float(r2_score(y, y_hat)),
                    elapsed_s       = 0.0,
                    cumulative_rows = self._cumulative_rows,
                ))
        return self

    def _save_impl(self, path: Path) -> None:
        pass

    def _load_impl(self, path: Path) -> None:
        pass

    def _get_extra_state(self) -> dict:
        """Preserve the fitted Nystroem transformer and its fitted flag."""
        return {"_nystroem": self._nystroem, "_nys_fitted": self._nys_fitted}
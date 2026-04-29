"""
ml/  —  LST temperature regression model library
=================================================
Public API — everything a caller needs is importable from this package.

Quick start
-----------
    from ml import train_all, LSTXGBoost, LSTRandomForest
    from ml import StreamingScaler, ModelRegistry
    from ml.transforms import cyclical, log1p_transform, polynomial_features
"""

from ml import sanity
from ml.base import LSTModel, BatchRecord, NEVER_FEATURES, TARGET_COL
from ml.did import LSTDiD, DiDResult
from ml.registry import ModelRegistry
from ml.scaler import StreamingScaler
from ml.train import train_all
from ml.transforms import (
    Transform, func_transform,
    cyclical, log1p_transform, sqrt_transform, polynomial_features,
    interaction_terms, ratio, delta, difference, clip_outliers, rolling_zscore,
)
from ml.models.linear import (
    LSTLinearRegression, LSTRidgeRegression, LSTElasticNet,
    LSTHuberRegression, LSTSGDRegressor, LSTNystroemSGD,
)
from ml.models.forest import LSTRandomForest, LSTExtraTrees
from ml.models.boosting import LSTHistGradientBoosting

try:
    from ml.models.boosting import LSTXGBoost
except ImportError:
    pass
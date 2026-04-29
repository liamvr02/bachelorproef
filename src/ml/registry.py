"""
ml/registry.py
==============
ModelRegistry — central lookup table for all LSTModel subclasses.

Registering a custom model
---------------------------
    from ml.registry import ModelRegistry
    from ml.base import LSTModel

    class MyModel(LSTModel):
        # ... implement abstract methods ...

    ModelRegistry.register("my_model", MyModel)
    model = ModelRegistry.create("my_model", feature_cols=["elev", "month"])

The registry is pre-populated with all built-in models.  Both short names
(e.g. "xgboost") and class names (e.g. "LSTXGBoost") are registered so
that LSTModel.load() can reconstruct any saved model regardless of which
alias was used when saving.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Type

from ml.base import LSTModel

log = logging.getLogger("lst_models.registry")


class ModelRegistry:
    """Central registry mapping short names to LSTModel subclasses."""

    # Populated at module load time via _bootstrap() below
    _registry: Dict[str, Type[LSTModel]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[LSTModel]) -> None:
        """
        Register a model class under a short name.

        Both the short name and the class's __name__ are registered so
        LSTModel.load() can always find the class by class name.
        """
        if not (isinstance(model_class, type) and issubclass(model_class, LSTModel)):
            raise TypeError(f"{model_class} must be a subclass of LSTModel")
        cls._registry[name]                 = model_class
        cls._registry[model_class.__name__] = model_class
        log.debug("ModelRegistry: '%s' → %s", name, model_class.__name__)

    @classmethod
    def get(cls, name: str) -> Type[LSTModel]:
        """Look up a model class by short name or class name."""
        klass = cls._registry.get(name)
        if klass is None:
            raise KeyError(
                f"Unknown model '{name}'.  "
                f"Registered names: {cls.available()}"
            )
        return klass

    @classmethod
    def create(cls, name: str, **kwargs) -> LSTModel:
        """Instantiate a model by registry name, forwarding kwargs."""
        return cls.get(name)(**kwargs)

    @classmethod
    def available(cls) -> List[str]:
        """Return all registered short names (excludes class-name aliases)."""
        return sorted(
            k for k in cls._registry
            if not k.startswith("LST")
        )

    @classmethod
    def all_classes(cls) -> Dict[str, Type[LSTModel]]:
        """Return {short_name: class} for all short-name registrations."""
        return {
            k: v for k, v in cls._registry.items()
            if not k.startswith("LST")
        }


def _bootstrap() -> None:
    """Populate the registry with all built-in models."""
    from ml.models.linear import (
        LSTLinearRegression,
        LSTRidgeRegression,
        LSTElasticNet,
        LSTHuberRegression,
        LSTSGDRegressor,
        LSTNystroemSGD,
    )
    from ml.models.forest import LSTRandomForest, LSTExtraTrees
    from ml.models.boosting import LSTHistGradientBoosting

    _builtin = [
        ("linear",          LSTLinearRegression),
        ("ridge",           LSTRidgeRegression),
        ("elastic_net",     LSTElasticNet),
        ("huber",           LSTHuberRegression),
        ("sgd",             LSTSGDRegressor),
        ("nystroem_sgd",    LSTNystroemSGD),
        ("random_forest",   LSTRandomForest),
        ("extra_trees",     LSTExtraTrees),
        ("hist_gb",         LSTHistGradientBoosting),
    ]

    # XGBoost is optional
    try:
        from ml.models.boosting import LSTXGBoost
        _builtin.append(("xgboost", LSTXGBoost))
    except ImportError:
        pass

    for name, klass in _builtin:
        ModelRegistry.register(name, klass)

    log.debug("ModelRegistry bootstrapped: %s", ModelRegistry.available())


_bootstrap()
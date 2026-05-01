"""
stream_configs — reusable StreamConfig factories for lst_*.py scripts.

Modules
-------
outliers     : OUTLIER_RANGES constant, expand_months helper, outlier_keys(),
               outlier_configs() — one StreamConfig per heat/cold event label.
presets      : all_rows(), representative() — natural and uniform-distribution
               stream configs.
point_filter : point_filter() — restrict a stream to a specific location and/or
               time slice without touching the core StreamConfig.
"""

from stream_configs.outliers import (
    OUTLIER_RANGES,
    expand_months,
    outlier_keys,
    outlier_configs,
)
from stream_configs.presets import all_rows, representative
from stream_configs.point_filter import PointFilterStream, point_filter

__all__ = [
    "OUTLIER_RANGES",
    "expand_months",
    "outlier_keys",
    "outlier_configs",
    "all_rows",
    "representative",
    "PointFilterStream",
    "point_filter",
]

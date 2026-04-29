"""
distribution.py  -  /src/stream/distribution.py
================================================
Target distribution objects for weighted partition sampling.

Design
------
A DistributionTarget holds one DimensionTarget per requested dimension.
The streaming layer scores each partition by calling partition_weight()
with the histogram count arrays retrieved from catalog.duckdb.

Extensibility
-------------
Any dimension name registered in config.DIMENSION_CATALOG can be targeted.
The DistributionTarget constructor validates names against the catalog so
typos are caught early.

Uniform shorthand
-----------------
Passing an empty dict {} as a dimension's target is the canonical way to
request a flat (uniform) distribution over all bins of that dimension.
This is the correct way to express "I want even coverage across all years"
or "I want even coverage across all hours of the day".

Usage examples
--------------
# Even distribution across year, month, and time-of-day:
cfg.set_distribution({
    "year":          ({}, year_edges),
    "month_of_year": ({}, month_edges),
    "hour_of_day":   ({}, hour_edges),
})

# Skewed temperature, uniform time-of-day:
cfg.set_distribution({
    "temperature": ({20: 0.3, 25: 0.5, 30: 0.2}, temp_edges),
    "hour_of_day": ({},                            hour_edges),
})

# Backwards-compatible simple format (temperature only, flat dict):
cfg.set_distribution({15: 0.2, 20: 0.3, 25: 0.5})
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union


class DimensionTarget:
    """
    Target distribution for a single scoreable dimension.

    Parameters
    ----------
    dimension : str
        Name of the dimension.  Must match an entry in config.DIMENSION_CATALOG.
    target : dict
        Mapping of {bin_edge: desired_proportion}.
        An empty dict {} requests a uniform (flat) distribution over all bins.
        Non-empty dicts are normalised to sum to 1.0.
    bin_edges : list
        Ordered bin boundary values (length = n_bins + 1).
        For numeric dimensions these are floats; for 'timestamp' they are
        quarterly label strings ("YYYY-Q{1..4}").
    """

    def __init__(self, dimension: str, target: Dict, bin_edges: List) -> None:
        self.dimension = dimension
        self.bin_edges = bin_edges
        n_bins = len(bin_edges) - 1

        if not target:
            # Uniform: every bin gets equal weight
            uniform = 1.0 / n_bins if n_bins > 0 else 0.0
            self.target: Dict[Any, float] = {bin_edges[i]: uniform for i in range(n_bins)}
        else:
            total = sum(target.values())
            self.target = {k: v / total for k, v in target.items()}

    def _find_bin(self, edge_value) -> int:
        """Return the 0-based bin index whose lower edge is closest to edge_value."""
        n_bins = len(self.bin_edges) - 1
        if isinstance(edge_value, str):
            # String dimensions (timestamp): exact match with fallback to index 0
            for i, e in enumerate(self.bin_edges[:-1]):
                if e == edge_value:
                    return i
            return 0
        # Numeric: closest lower edge
        return min(range(n_bins), key=lambda i: abs(float(self.bin_edges[i]) - float(edge_value)))

    def partition_weight(self, hist_counts: List[int]) -> float:
        """
        Score this dimension's contribution to the target distribution.

        The score is sum(min(actual_proportion, target_proportion)) over all
        bins — a standard overlap metric that equals 1.0 when the partition's
        distribution exactly matches the target, and 0.0 when there is no overlap.

        Parameters
        ----------
        hist_counts : list[int]
            Dense count array of length n_bins for this dimension.

        Returns
        -------
        float in [0, 1].  Returns 0.0 when the partition has no rows.
        """
        total = sum(hist_counts)
        if total == 0:
            return 0.0
        score = 0.0
        for edge_value, desired in self.target.items():
            bin_idx     = self._find_bin(edge_value)
            actual_prop = hist_counts[bin_idx] / total
            score      += min(actual_prop, desired)
        return score


class DistributionTarget:
    """
    Multi-dimensional distribution target for weighted partition sampling.

    The partition weight is the product of per-dimension scores.  Partitions
    with weight 0 are skipped entirely.  Partitions are then ordered by
    weight descending so the most valuable data is streamed first, enabling
    early stopping (max_rows) to yield a well-distributed sample.

    Parameters
    ----------
    targets : dict — two accepted formats:

        Multi-dimension (recommended):
            {
                "year":          ({},                          year_edges),
                "month_of_year": ({},                          month_edges),
                "hour_of_day":   ({},                          hour_edges),
                "temperature":   ({20: 0.3, 25: 0.4, 30: 0.3}, temp_edges),
            }
        Each value is a (target_dict, bin_edges) tuple.
        An empty target_dict {} means uniform over all bins.

        Simple backwards-compatible (temperature only):
            {15: 0.2, 20: 0.3, 25: 0.5}
        Bin edges are injected later via set_temperature_bins().

    valid_dimensions : set[str], optional
        When provided, dimension names are validated against this set.
        Pass config.DIMENSION_CATALOG.keys() from the calling layer.
    """

    def __init__(
        self,
        targets: Dict,
        valid_dimensions: Optional[set] = None,
    ) -> None:
        self.dimensions: Dict[str, DimensionTarget] = {}
        self._valid_dimensions = valid_dimensions

        # Detect backwards-compatible flat dict (all numeric keys)
        is_simple = bool(targets) and all(
            isinstance(k, (int, float)) for k in targets.keys()
        )

        if is_simple:
            # Temperature-only shorthand; bin edges injected later
            self._pending_simple_target: Optional[Dict] = targets
        else:
            self._pending_simple_target = None
            for dimension, value in targets.items():
                if not (isinstance(value, (list, tuple)) and len(value) == 2):
                    raise ValueError(
                        f"Target for dimension '{dimension}' must be "
                        f"(target_dict, bin_edges); got {type(value).__name__}."
                    )
                if valid_dimensions is not None and dimension not in valid_dimensions:
                    raise KeyError(
                        f"Unknown dimension '{dimension}'. "
                        f"Valid dimensions: {sorted(valid_dimensions)}"
                    )
                target_dict, bin_edges = value
                self.dimensions[dimension] = DimensionTarget(dimension, target_dict, bin_edges)

    def set_temperature_bins(self, bin_edges: List[float]) -> "DistributionTarget":
        """Inject bin edges for the backwards-compatible simple temperature target."""
        if self._pending_simple_target is not None:
            self.dimensions["temperature"] = DimensionTarget(
                "temperature", self._pending_simple_target, bin_edges
            )
            self._pending_simple_target = None
        return self

    def partition_weight(self, counts_by_dim: Dict[str, List[int]]) -> float:
        """
        Score a partition by the product of all dimension weights.

        Parameters
        ----------
        counts_by_dim : dict mapping dimension name → dense count list.
            Only dimensions present in self.dimensions are consulted.
            A missing array for a requested dimension returns 0.0.

        Returns
        -------
        float in [0, 1].  Returns 1.0 when no dimensions are configured.
        """
        if not self.dimensions:
            return 1.0

        weight = 1.0
        for dimension, dim_target in self.dimensions.items():
            counts = counts_by_dim.get(dimension)
            if counts is None:
                return 0.0
            weight *= dim_target.partition_weight(counts)
        return weight
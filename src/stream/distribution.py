"""
distribution.py  -  /src/stream/distribution.py
================================================
Multi-dimensional distribution targeting for weighted partition sampling.

Supports targeting distributions for:
  - temperature: LST temperature values in degrees C
  - timestamp: ISO 8601 dates (YYYY-MM-DD format)
  - longitude: WGS-84 longitudinal coordinate in degrees
  - latitude: WGS-84 latitudinal coordinate in degrees
"""

from __future__ import annotations

from typing import Dict, List, Optional


class DimensionTarget:
    """
    Target distribution for a single dimension (e.g., temperature, timestamp, location).

    Parameters
    ----------
    dimension : str
        Name of the dimension: "temperature", "timestamp", "longitude", or "latitude"
    target : dict mapping bin edge to desired proportion
        Proportions are normalised to sum to 1.
        Example:  {15: 0.20, 20: 0.30, 25: 0.30, 30: 0.15, 35: 0.05}
    bin_edges : list of bin lower edges (+ upper edge of last bin)
        E.g., [-10, -8, -6, ..., 58, 60] for 2°C bins from -10 to 60°C
    """

    def __init__(
        self,
        dimension: str,
        target: Dict[float, float],
        bin_edges: List[float],
    ):
        self.dimension = dimension
        self.bin_edges = bin_edges
        total = sum(target.values())
        self.target = {float(k): v / total for k, v in target.items()}

    def partition_weight(self, hist_counts: List[int]) -> float:
        """
        Score this dimension's contribution to the target distribution.

        Parameters
        ----------
        hist_counts : list of bin counts from the histogram

        Returns
        -------
        float in [0, 1]. Weight 0 means partition doesn't match target at all.
        """
        total = sum(hist_counts)
        if total == 0:
            return 0.0
        score = 0.0
        for bin_edge, desired_prop in self.target.items():
            # Find the bin index for this edge
            bin_idx = min(
                range(len(self.bin_edges) - 1),
                key=lambda i: abs(self.bin_edges[i] - bin_edge),
            )
            actual_prop = hist_counts[bin_idx] / total
            score += min(actual_prop, desired_prop)
        return score


class DistributionTarget:
    """
    Multi-dimensional distribution target for weighted partition sampling.

    Combines multiple dimension targets (temperature, timestamp, coordinates).
    Partition weight is the product of weights across all dimensions.

    Parameters
    ----------
    targets : dict mapping dimension name to (target_dict, bin_edges)
        Example:
            {
                "temperature": ({15: 0.2, 20: 0.3, 25: 0.3, 30: 0.2}, [-10, -8, ..., 60]),
                "timestamp": ({"2010-01": 0.3, "2015-01": 0.4, "2020-01": 0.3},
                              ["2000-01", "2005-01", ..., "2025-01"]),
            }

    Backwards compatibility
    -----------------------
    Accepts a simple dict of edge->proportion for temperature-only targeting.
    If called with a flat dict, assumes temperature-only mode:
        DistributionTarget({15: 0.2, 20: 0.3, 25: 0.5})
    """

    def __init__(self, targets: Dict):
        self.dimensions: Dict[str, DimensionTarget] = {}

        # Detect simple (temperature-only) vs multi-dimension format
        is_simple_format = (
            targets and
            all(isinstance(k, (int, float)) for k in targets.keys())
        )

        if is_simple_format:
            # Backwards compatibility: flat dict → temperature-only
            # Caller must pass default bin edges via set_temperature_bins
            self._pending_simple_target = targets
            self._pending_temperature_bins: Optional[List[float]] = None
        else:
            # Multi-dimension format: dict of {dimension: (target_dict, bin_edges)}
            for dimension, (target_dict, bin_edges) in targets.items():
                self.dimensions[dimension] = DimensionTarget(
                    dimension, target_dict, bin_edges
                )

    def set_temperature_bins(self, bin_edges: List[float]) -> "DistributionTarget":
        """Set temperature bins for backwards-compatible simple format."""
        if hasattr(self, "_pending_simple_target"):
            self.dimensions["temperature"] = DimensionTarget(
                "temperature", self._pending_simple_target, bin_edges
            )
            del self._pending_simple_target
        return self

    def partition_weight(
        self,
        temperature_counts: Optional[List[int]] = None,
        timestamp_counts: Optional[List[int]] = None,
        longitude_counts: Optional[List[int]] = None,
        latitude_counts: Optional[List[int]] = None,
    ) -> float:
        """
        Score a partition by its contribution to all target distributions.

        Parameters
        ----------
        temperature_counts : histogram counts for temperature dimension
        timestamp_counts : histogram counts for timestamp dimension
        longitude_counts : histogram counts for longitude dimension
        latitude_counts : histogram counts for latitude dimension

        Returns
        -------
        float in [0, 1]. Product of individual dimension weights.
        Partitions with weight 0 are skipped.
        """
        if not self.dimensions:
            return 1.0  # No targets defined

        weights = []
        for dimension, target in self.dimensions.items():
            if dimension == "temperature":
                if temperature_counts is None:
                    return 0.0
                weights.append(target.partition_weight(temperature_counts))
            elif dimension == "timestamp":
                if timestamp_counts is None:
                    return 0.0
                weights.append(target.partition_weight(timestamp_counts))
            elif dimension == "longitude":
                if longitude_counts is None:
                    return 0.0
                weights.append(target.partition_weight(longitude_counts))
            elif dimension == "latitude":
                if latitude_counts is None:
                    return 0.0
                weights.append(target.partition_weight(latitude_counts))

        # Product of all dimension weights
        weight = 1.0
        for w in weights:
            weight *= w
        return weight
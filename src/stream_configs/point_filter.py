"""
stream_configs.point_filter — StreamConfig wrapper that restricts to a specific
location and/or time slice via post-batch filtering.

Usage
-----
    from stream_configs import point_filter

    # All rows near a specific pixel, all years
    cfg = point_filter(lon=3.72, lat=51.05, tol_deg=0.001)

    # One calendar month across all years
    cfg = point_filter(month=7)

    # One day at a specific location
    cfg = point_filter(lon=3.72, lat=51.05, year=2018, month=7, day=15)

    # Specific hour window (±0.5 h)
    cfg = point_filter(lon=3.72, lat=51.05, hour=10.5)

    for df in cfg.stream(registry, max_rows=50_000):
        print(df.head())

Notes
-----
- year+month pre-filter to a single partition_key, avoiding a full catalog scan.
- year-only or month-only partially pre-filters partition_keys before post-filtering.
- lon/lat/day/hour are always applied as post-batch predicates.
- tol_deg applies to both lon and lat (default 0.001 ° ≈ 80–100 m at Ghent's latitude).
- hour tolerance is fixed at ±0.5 fractional UTC hours.
"""

from __future__ import annotations

from typing import Callable, Generator, List, Optional

import pandas as pd

from stream.features import FeatureRegistry
from stream.stream import StreamConfig


class PointFilterStream:
    """
    Thin wrapper around StreamConfig that applies a per-batch filter predicate.

    The interface mirrors StreamConfig.stream() so it can be passed anywhere a
    StreamConfig is accepted as a streaming source.
    """

    def __init__(self, cfg: StreamConfig, predicate: Callable[[pd.DataFrame], pd.Series]) -> None:
        self._cfg = cfg
        self._predicate = predicate

    def stream(
        self,
        registry: Optional[FeatureRegistry] = None,
        batch_size: Optional[int] = None,
        max_rows: Optional[int] = None,
    ) -> Generator[pd.DataFrame, None, None]:
        # max_rows caps OUTPUT rows (post-predicate), not scan rows.
        # The underlying scan runs unbounded so that sparse filters (spatial
        # points, specific hours) cover the full temporal range rather than
        # stopping after the first few partitions.
        n_output = 0
        for df in self._cfg.stream(registry, batch_size=batch_size, max_rows=max_rows):
            filtered = df.loc[self._predicate(df)]
            if filtered.empty:
                continue
            if max_rows is not None:
                remaining = max_rows - n_output
                if remaining <= 0:
                    return
                if len(filtered) > remaining:
                    filtered = filtered.iloc[:remaining]
            yield filtered
            n_output += len(filtered)
            if max_rows is not None and n_output >= max_rows:
                return

    def set_distribution(self, dist: dict) -> None:
        self._cfg.set_distribution(dist)


def point_filter(
    lon: Optional[float] = None,
    lat: Optional[float] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    day: Optional[int] = None,
    hour: Optional[float] = None,
    batch_size: int = 100_000,
    tol_deg: float = 0.001,
) -> PointFilterStream:
    """
    Return a stream restricted to rows matching the given spatial/temporal criteria.

    Parameters
    ----------
    lon, lat   : WGS-84 target coordinates; rows within *tol_deg* of each are kept.
    year       : calendar year (exact match).
    month      : month of year 1-12 (exact match on month_of_year column).
    day        : day of month 1-31 (exact match on day_of_month column).
    hour       : fractional UTC hour; rows within ±0.5 h are kept.
    batch_size : rows per batch passed to the underlying StreamConfig.
    tol_deg    : coordinate tolerance in degrees (default 0.001 ° ≈ 80-100 m).
    """
    partition_keys: Optional[List[str]] = None

    if year is not None and month is not None:
        partition_keys = [f"{year:04d}-{month:02d}"]
    elif year is not None:
        partition_keys = [f"{year:04d}-{m:02d}" for m in range(1, 13)]
    elif month is not None:
        partition_keys = [f"{y:04d}-{month:02d}" for y in range(2000, 2026)]

    cfg = StreamConfig(partition_keys=partition_keys, batch_size=batch_size)

    def predicate(df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        if lon is not None and "longitude" in df.columns:
            mask &= (df["longitude"] - lon).abs() <= tol_deg
        if lat is not None and "latitude" in df.columns:
            mask &= (df["latitude"] - lat).abs() <= tol_deg
        if year is not None and "year" in df.columns:
            mask &= df["year"] == year
        if month is not None and "month_of_year" in df.columns:
            mask &= df["month_of_year"] == month
        if day is not None and "day_of_month" in df.columns:
            mask &= df["day_of_month"] == day
        if hour is not None and "hour_of_day" in df.columns:
            mask &= (df["hour_of_day"] - hour).abs() <= 0.5
        return mask

    return PointFilterStream(cfg, predicate)

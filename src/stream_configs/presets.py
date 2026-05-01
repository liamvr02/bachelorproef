"""
stream_configs.presets — common StreamConfig presets for lst_*.py scripts.

all_rows()       All partitions, natural distribution (no reweighting).
representative() Uniform year/month/hour distribution; outlier months optionally
                 excluded.  Pass max_rows to .stream() to cap the row count.
"""

from __future__ import annotations

from typing import List, Optional

from stream.config import get_dimension_edges
from stream.stream import StreamConfig

_UNIFORM_DISTRIBUTION = {
    "year":          ({}, get_dimension_edges("year")),
    "month_of_year": ({}, get_dimension_edges("month_of_year")),
    "hour_of_day":   ({}, get_dimension_edges("hour_of_day")),
}


def all_rows(batch_size: int = 100_000) -> StreamConfig:
    """All partitions, natural temporal density, no distribution reweighting."""
    return StreamConfig(batch_size=batch_size)


def representative(
    excluded_keys: Optional[List[str]] = None,
    batch_size: int = 100_000,
) -> StreamConfig:
    """
    Uniform year/month/hour distribution stream.

    Loads the catalog to resolve all available partition_keys, removes any
    keys listed in *excluded_keys* (typically outlier months), then applies
    uniform distribution weighting across year, month_of_year, and hour_of_day.

    Pass max_rows to the returned config's .stream() call to cap row count:

        cfg = representative(excluded_keys=excluded)
        for df in cfg.stream(registry, max_rows=1_500_000):
            ...
    """
    probe = StreamConfig(batch_size=batch_size)
    probe._load_catalog()
    exclude = set(excluded_keys or [])
    train_keys = [k for k in probe._partition_keys if k not in exclude]
    cfg = StreamConfig(partition_keys=train_keys, batch_size=batch_size)
    cfg.set_distribution(_UNIFORM_DISTRIBUTION)
    return cfg

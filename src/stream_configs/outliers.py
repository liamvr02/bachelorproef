"""
stream_configs.outliers — canonical outlier date ranges and StreamConfig factories.

The outlier periods are meteorologically anomalous Belgian summers/winters that
are held out of training and used as out-of-distribution evaluation splits.
They are defined once here and imported by every lst_*.py script.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Tuple

from stream.stream import StreamConfig


OUTLIER_RANGES: List[Tuple[date, date, str]] = [
    (date(2006, 6, 1),  date(2006, 9, 1),  "heat_2006_summer"),
    (date(2007, 4, 1),  date(2007, 5, 1),  "warm_2007_apr"),
    (date(2010, 1, 1),  date(2010, 3, 1),  "cold_2010_winter"),
    (date(2012, 2, 1),  date(2012, 3, 1),  "cold_2012_feb"),
    (date(2013, 7, 1),  date(2013, 9, 1),  "heat_2013_summer"),
    (date(2015, 7, 1),  date(2015, 9, 1),  "heat_2015_summer"),
    (date(2018, 5, 1),  date(2018, 9, 1),  "drought_2018"),
    (date(2019, 6, 1),  date(2019, 8, 1),  "heat_2019_jun_jul"),
    (date(2020, 8, 1),  date(2020, 9, 1),  "heat_2020_aug"),
]


def expand_months(start: date, end: date) -> List[str]:
    """Return 'YYYY-MM' keys from *start* (inclusive) to *end* (exclusive)."""
    out: List[str] = []
    y, m = start.year, start.month
    while (y, m) < (end.year, end.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            m, y = 1, y + 1
    return out


def outlier_keys() -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Return (excluded_keys, keys_by_label).

    excluded_keys   : sorted list of every partition_key that belongs to an
                      outlier period — pass to representative() as excluded_keys.
    keys_by_label   : mapping of label → list of partition_keys for that period.
    """
    by_label: Dict[str, List[str]] = {
        label: expand_months(start, end)
        for start, end, label in OUTLIER_RANGES
    }
    excluded = sorted({k for ks in by_label.values() for k in ks})
    return excluded, by_label


def outlier_configs(batch_size: int = 100_000) -> Dict[str, StreamConfig]:
    """
    Return a StreamConfig per outlier label, restricted to that label's months.

    The natural temporal density is preserved (no distribution reweighting) so
    per-period evaluation reflects the actual data density in each event.
    """
    _, by_label = outlier_keys()
    return {
        label: StreamConfig(partition_keys=keys, batch_size=batch_size)
        for label, keys in by_label.items()
    }

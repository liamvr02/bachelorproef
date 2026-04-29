"""
stream_example.py - Stream 5 M rows with even temporal distribution.

Demonstrates how to use the generalised distribution targeting system to
achieve an even spread of samples across:
  • calendar year  (2000-2025)
  • month of year  (January-December)
  • hour of day    (00:00-23:59 UTC, expressed as a fractional float)

Passing an empty dict {} as the target for any dimension is the uniform
shorthand: the system assigns equal desired probability to every bin.

The DistributionTarget scores each monthly partition by the product of its
overlap with each requested distribution.  Partitions are then ordered by
score descending so the most temporally diverse data is streamed first.
Early stopping via max_rows lets you collect a well-distributed sample
without reading the entire dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stream.config import get_dimension_edges
from stream.features import FeatureRegistry, aggregate_in_radius, nearest, urban_atlas_classifications_fractions
from stream.classification_groups import UA
from stream.stream import StreamConfig


# ============================================================
# Feature registry
# ============================================================

def build_registry() -> FeatureRegistry:
    """
    Build a feature registry with a representative set of spatial features.

    The time-component columns (year, month_of_year, day_of_month,
    day_of_year, hour_of_day) are always present in every output batch
    without needing explicit registration - they come directly from the
    lst table and appear in FeatureRow and the yielded DataFrames.
    """
    reg = FeatureRegistry()

    # Nearest elevation from the 2007 digital height model
    reg.add(nearest("dhm", columns=["elevation"], temporal="last_previous"))

    # Tree count within 50 m
    reg.add(aggregate_in_radius("trees", radius_m=50, columns=[], agg="count", temporal="none"))

    # Urban Atlas land-use fractions at 100 m and 30 m radii
    reg.add(urban_atlas_classifications_fractions(classification_map=UA, radius_m=100))
    reg.add(urban_atlas_classifications_fractions(classification_map=UA, radius_m=30))

    return reg


# ============================================================
# Stream helpers
# ============================================================

def make_even_temporal_config(
    max_rows: int = 5_000_000,
    also_balance_temperature: bool = False,
) -> StreamConfig:
    """
    Build a StreamConfig that targets an even distribution over time.

    Dimensions targeted
    -------------------
    year          - uniform across all 26 calendar years (2000-2025)
    month_of_year - uniform across all 12 calendar months
    hour_of_day   - uniform across all 24 UTC hours (fractional)

    Optionally also balance temperature (biases away from very common
    warm summer values toward the full -10 °C to 60 °C range).

    Parameters
    ----------
    max_rows : int
        Upper bound on rows to collect.  Passed to stream() so early stopping
        happens before feature computation rather than after.
    also_balance_temperature : bool
        When True, adds a mildly non-uniform temperature target that gives
        extra weight to cooler and hotter extremes.
    """
    cfg = StreamConfig()

    distribution: dict = {
        # {} = uniform over all bins of this dimension
        "year":          ({}, get_dimension_edges("year")),
        "month_of_year": ({}, get_dimension_edges("month_of_year")),
        "hour_of_day":   ({}, get_dimension_edges("hour_of_day")),
    }

    if also_balance_temperature:
        # Slightly up-weight the tails relative to a purely uniform target.
        # The proportions are normalised to sum 1 inside DimensionTarget.
        distribution["temperature"] = (
            {
                -5: 0.05,
                 5: 0.08,
                10: 0.10,
                15: 0.12,
                20: 0.15,
                25: 0.15,
                30: 0.12,
                35: 0.10,
                40: 0.08,
                45: 0.05,
            },
            get_dimension_edges("temperature"),
        )

    cfg.set_distribution(distribution)
    return cfg


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Stream 5 M LST rows with even temporal coverage and save to CSV.

    Steps
    -----
    1.  Build a StreamConfig that weights partitions for even year / month /
        hour_of_day coverage.
    2.  Build a FeatureRegistry with spatial features.
    3.  Stream up to max_rows rows; partitions ordered by temporal-coverage
        weight so the richest months come first.
    4.  Save the resulting DataFrame to CSV.
    5.  Print a temporal balance report.
    """
    max_rows = 20_000_000

    print("=" * 70)
    print(f"Streaming {max_rows:,} rows with even temporal distribution")
    print("  Dimensions: year, month_of_year, hour_of_day  (uniform)")
    print("=" * 70)

    cfg = make_even_temporal_config(max_rows=max_rows, also_balance_temperature=False)
    reg = build_registry()

    batches = []
    for batch_df in cfg.stream(reg, batch_size=10_000, max_rows=max_rows):
        batches.append(batch_df)

    if not batches:
        print("No rows collected.")
        return

    df = pd.concat(batches, ignore_index=True)
    print(f"\nCollected {len(df):,} rows, {df.shape[1]} columns")

    # ---- Temporal balance report ----------------------------------------
    print("\n── Year distribution (top 10) ──────────────────────────────────")
    print(
        df["year"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\n── Month-of-year distribution ───────────────────────────────────")
    month_names = {
        1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
        7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec",
    }
    month_counts = df["month_of_year"].value_counts().sort_index()
    for m, cnt in month_counts.items():
        bar = "█" * int(30 * cnt / month_counts.max())
        print(f"  {month_names.get(m, m):>3}  {cnt:>8,}  {bar}")

    print("\n── Hour-of-day distribution (binned to nearest hour) ────────────")
    hour_counts = df["hour_of_day"].apply(lambda h: int(h)).value_counts().sort_index()
    for h, cnt in hour_counts.items():
        bar = "█" * int(30 * cnt / hour_counts.max())
        print(f"  {h:>2}h  {cnt:>8,}  {bar}")

    # ---- Save -------------------------------------------------------------
    output_path = Path("sample_stream_output.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(df):,} rows -> {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
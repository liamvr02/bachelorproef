"""
stream_configs.registry — canonical FeatureRegistry for lst_*.py scripts.

The registry defined here is the most complete spatial feature set used across
all ML experiments:

  DHM elevation   — avg / max / min at radii 50, 70, 100 m (last_previous)
  Trees (total)   — count at radii 50, 70, 100 m
  Trees (by phase)— count per beheerfase (Jeugdfase / Volwassenfase /
                    Veteranenfase) at radii 50, 70, 100 m
  Urban Atlas     — classification fractions at radii 50, 70, 100 m
  WIS bestemming  — fraction per value at radii 50, 70, 100 m
  WIS materiaal   — fraction per value at radii 50, 70, 100 m

Note on the None beheerfase
---------------------------
Passing ``attr_filter={"beheerfase": None}`` would generate the SQL fragment
``AND beheerfase = 'None'`` — a match on the *literal string* "None", which
returns zero rows in every real dataset.  The total-count feature (all trees,
no filter) is produced instead by passing ``attr_filter=None``, which omits
the WHERE fragment entirely.
"""

from __future__ import annotations

from stream.classification_groups import UA, WIS_BESTEMMING, WIS_MATERIAAL
from stream.features import (
    FeatureRegistry,
    aggregate_in_radius,
    urban_atlas_classifications_fractions,
    wis_fraction,
    trees_count_planted_by,
)

_RADII = (50, 70, 100)
_BEHEERFASEN = ("Jeugdfase", "Volwassenfase", "Veteranenfase")


def build_registry() -> FeatureRegistry:
    """
    Return the full feature registry used by all ML experiments.

    Tree counts are split into a total (no filter) plus one count per
    beheerfase growth phase, giving the model visibility into the
    maturity distribution of nearby trees at each radius.
    """
    reg = FeatureRegistry()

    # ── DHM ──────────────────────────────────────────────────────────────────
    for r in _RADII:
        for agg in ("avg", "max", "min"):
            reg.add(aggregate_in_radius(
                "dhm", radius_m=r, columns=["elevation"],
                agg=agg, temporal="last_previous",
            ))

    # ── Trees — total count (no beheerfase filter) ────────────────────────
    for r in _RADII:
        reg.add(trees_count_planted_by(
            "trees", radius_m=r, columns=[], agg="count",
            attr_filter=None, temporal="none",
        ))

    # ── Trees — per growth phase ───────────────────────────────────────────
    # attr_filter=None for the total count (above) omits the WHERE fragment.
    # Each named phase uses an equality predicate on the beheerfase column.
    for r in _RADII:
        for phase in _BEHEERFASEN:
            reg.add(trees_count_planted_by(
                "trees", radius_m=r, columns=[], agg="count",
                attr_filter={"beheerfase": phase}, temporal="none",
            ))

    # ── Urban Atlas ───────────────────────────────────────────────────────
    for r in _RADII:
        reg.add(urban_atlas_classifications_fractions(
            classification_map=UA, radius_m=r,
        ))

    # ── WIS ───────────────────────────────────────────────────────────────
    for r in _RADII:
        for leaves in WIS_BESTEMMING.values():
            for val in leaves:
                reg.add(wis_fraction(attr_col="bestemming", attr_val=val, radius_m=r))
        for leaves in WIS_MATERIAAL.values():
            for val in leaves:
                reg.add(wis_fraction(attr_col="materiaalsoort", attr_val=val, radius_m=r))

    return reg

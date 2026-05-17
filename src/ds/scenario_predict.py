"""
scenario_predict.py
===================

Quantitative scenario prediction: "if location X has Y% green coverage,
what is the expected LST cooling?"

Fits a joint OLS on [Z_temporal | Z_morph | X_greening] for each dataset,
then evaluates user-defined greening scenarios relative to an arbitrary
morphology baseline (default: dataset mean morphology).

Output
------
  src/ds_reports/scenario_predict.json   per-dataset scenario table
  src/ds_reports/scenario_predict.csv    flat CSV for easy reading

Methodology
-----------
Uses the same sufficient-statistics accumulator as partial_corr.py, but
fits ALL greening features jointly in a single model (not one at a time).
This gives correct multi-feature scenario predictions without double-counting
correlated features.

The joint OLS coefficient for a greening feature x_j is:
    beta_j = coefficient in  T ~ [Z | X_green_joint]

Scenario prediction (holding Z_morph at dataset mean):
    Delta_T = sum_j  beta_j * (scenario_j - baseline_j)

where baseline_j is the dataset mean of each greening feature (i.e., "average
location in this dataset").

Usage
-----
    # All datasets, default scenarios
    uv run python ds/scenario_predict.py

    # One dataset
    uv run python ds/scenario_predict.py --dataset full_representative

    # Custom scenarios (JSON string)
    uv run python ds/scenario_predict.py --scenarios '[
        {"name": "30pct_veg", "ua_ua_vegetation_100m_frac": 0.30},
        {"name": "30pct_veg_10pct_water", "ua_ua_vegetation_100m_frac": 0.30,
         "ua_ua_water_wetlands_100m_frac": 0.10}
    ]'
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_DS_DIR      = Path(__file__).parent
_SRC         = _DS_DIR.parent
_REPORTS_DIR = _SRC / "ds_reports"

sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_DS_DIR))

# ---------------------------------------------------------------------------
# Default scenario grid (policy-relevant increments)
# All values are absolute fractions (0–1) or counts, not deltas.
# ---------------------------------------------------------------------------
_DEFAULT_SCENARIOS: List[Dict] = [
    # Single-feature: vegetation fraction only
    {"name": "veg_10pct",  "ua_ua_vegetation_100m_frac": 0.10},
    {"name": "veg_20pct",  "ua_ua_vegetation_100m_frac": 0.20},
    {"name": "veg_30pct",  "ua_ua_vegetation_100m_frac": 0.30},
    {"name": "veg_50pct",  "ua_ua_vegetation_100m_frac": 0.50},
    # Single-feature: water/wetlands
    {"name": "water_5pct", "ua_ua_water_wetlands_100m_frac": 0.05},
    {"name": "water_10pct","ua_ua_water_wetlands_100m_frac": 0.10},
    {"name": "water_20pct","ua_ua_water_wetlands_100m_frac": 0.20},
    # Combined: vegetation + water (user's example)
    {"name": "veg30_water10",
     "ua_ua_vegetation_100m_frac":    0.30,
     "ua_ua_water_wetlands_100m_frac":0.10},
    {"name": "veg20_water5",
     "ua_ua_vegetation_100m_frac":    0.20,
     "ua_ua_water_wetlands_100m_frac":0.05},
    # Trees only (with endogeneity caveat noted in output)
    {"name": "trees_10",  "trees_plantedby_100m_count": 10.0},
    {"name": "trees_25",  "trees_plantedby_100m_count": 25.0},
    {"name": "trees_50",  "trees_plantedby_100m_count": 50.0},
]

# Features used as greening regressors in the joint model.
# UA features: use only 100m radius (50m/70m are identical in current parquets).
# Trees: use total count at 100m only (mature sub-count is highly collinear).
# Structural NDVI excluded: positive sign in representative dataset and known
# co-measurement artefact; reported separately with caveat.
_JOINT_GREENING_FEATURES = [
    "ua_ua_vegetation_100m_frac",
    "ua_ua_water_wetlands_100m_frac",
    "trees_plantedby_100m_count",
]

# Morphology confounders (same as partial_corr.py)
_MORPH_COLS = [
    "dhm_avg50m_elevation_avg", "dhm_max50m_elevation_max", "dhm_min50m_elevation_min",
    "dhm_avg70m_elevation_avg", "dhm_max70m_elevation_max", "dhm_min70m_elevation_min",
    "dhm_avg100m_elevation_avg","dhm_max100m_elevation_max","dhm_min100m_elevation_min",
    "ua_ua_dense_built_up_50m_frac",    "ua_ua_mixed_urban_50m_frac",
    "ua_ua_transport_infrastructure_50m_frac", "ua_ua_bare_sparse_50m_frac",
    "ua_ua_dense_built_up_70m_frac",    "ua_ua_mixed_urban_70m_frac",
    "ua_ua_transport_infrastructure_70m_frac", "ua_ua_bare_sparse_70m_frac",
    "ua_ua_dense_built_up_100m_frac",   "ua_ua_mixed_urban_100m_frac",
    "ua_ua_transport_infrastructure_100m_frac","ua_ua_bare_sparse_100m_frac",
]


# ---------------------------------------------------------------------------
# Sufficient-statistics accumulator (copied from partial_corr.py but multivariate)
# ---------------------------------------------------------------------------
class _JointAccumulator:
    """
    Accumulates XtX and Xty for a joint OLS  y ~ X  in a single streaming pass.

    X is the full design matrix [Z_temporal | Z_morph_std | X_greening_std],
    y is the temperature column.
    """
    def __init__(self, n_features: int) -> None:
        self.n   = 0
        self.XtX = np.zeros((n_features, n_features), dtype=np.float64)
        self.Xty = np.zeros(n_features,               dtype=np.float64)

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        mask = np.isfinite(y)
        if not mask.any():
            return
        Xm, ym = X[mask], y[mask]
        self.XtX += Xm.T @ Xm
        self.Xty += Xm.T @ ym
        self.n   += int(mask.sum())

    def solve(self) -> Optional[np.ndarray]:
        """
        Return OLS coefficients via minimum-norm least-squares.

        Uses lstsq on the normal equations so rank-deficient designs
        (e.g. identical UA fraction columns at multiple radii) are handled
        gracefully rather than raising LinAlgError.  rcond is set to 1e-10
        to silence the FutureWarning on newer NumPy.
        """
        try:
            beta, _, _, _ = np.linalg.lstsq(self.XtX, self.Xty, rcond=1e-10)
            return beta
        except np.linalg.LinAlgError:
            return None


def _temporal_block(df) -> np.ndarray:
    """Return temporal confounder design block [1, year_norm, sin/cos month, sin/cos hour]."""
    n = len(df)
    X = np.ones((n, 6), dtype=np.float64)
    if "year" in df.columns:
        X[:, 1] = (df["year"].to_numpy(dtype=np.float64) - 2012.5) / 6.25
    if "month_of_year" in df.columns:
        m = df["month_of_year"].to_numpy(dtype=np.float64) * (2 * np.pi / 12)
        X[:, 2], X[:, 3] = np.sin(m), np.cos(m)
    if "hour_of_day" in df.columns:
        h = df["hour_of_day"].to_numpy(dtype=np.float64) * (2 * np.pi / 24)
        X[:, 4], X[:, 5] = np.sin(h), np.cos(h)
    return X


def _morph_block(df, morph_cols: List[str],
                 morph_mean: Optional[np.ndarray],
                 morph_std:  Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return standardised morphology block and (running) mean/std.

    On first pass (morph_mean/std = None), compute from this batch.
    """
    present = [c for c in morph_cols if c in df.columns]
    M = np.zeros((len(df), len(morph_cols)), dtype=np.float64)
    for i, c in enumerate(morph_cols):
        if c in df.columns:
            M[:, i] = df[c].to_numpy(dtype=np.float64)

    if morph_mean is None:
        morph_mean = np.nanmean(M, axis=0)
        morph_std  = np.nanstd(M,  axis=0)
        morph_std[morph_std < 1e-12] = 1.0

    M = (M - morph_mean) / morph_std
    np.nan_to_num(M, nan=0.0, copy=False)
    return M, morph_mean, morph_std


# ---------------------------------------------------------------------------
# Per-dataset joint OLS fit
# ---------------------------------------------------------------------------
def _fit_dataset(
    dataset:         str,
    batch_size:      int,
    max_rows:        Optional[int],
) -> Optional[Dict]:
    """
    Fit joint OLS  T ~ [Z_temporal | Z_morph_std | X_green_std]  on a parquet dataset.

    Returns a dict with:
      - coefficients: {feature_name: beta}
      - feature_means, feature_stds (greening only, in natural units)
      - dataset stats
      - n_rows
    """
    from utils import load_parquet_sample, dataset_stats, run_timestamp, version_meta

    data_dir = _SRC / "test_data"
    parquet   = data_dir / f"{dataset}.parquet"
    if not parquet.exists():
        return {"error": f"parquet not found: {parquet}"}

    df_full = load_parquet_sample(parquet, max_rows=max_rows, seed=42)

    # Resolve temperature column
    temp_col = next(
        (c for c in ("temperature", "aster_lst", "modis_lst") if c in df_full.columns),
        None,
    )
    if temp_col is None:
        return {"error": "no temperature column found"}

    # Greening features present in this dataset
    green_cols = [c for c in _JOINT_GREENING_FEATURES if c in df_full.columns]
    if not green_cols:
        return {"error": "no greening features present"}

    n_temporal = 6
    n_morph    = len(_MORPH_COLS)
    n_green    = len(green_cols)
    n_total    = n_temporal + n_morph + n_green

    # Compute greening means and stds from full dataset
    green_means = {}
    green_stds  = {}
    for c in green_cols:
        v = df_full[c].to_numpy(dtype=np.float64)
        green_means[c] = float(np.nanmean(v))
        s = float(np.nanstd(v))
        green_stds[c]  = s if s > 1e-12 else 1.0

    # Morphology normalisation from full dataset
    morph_mean, morph_std = None, None
    M_full, morph_mean, morph_std = _morph_block(df_full, _MORPH_COLS, None, None)

    acc = _JointAccumulator(n_total)

    # Single batch (dataset already loaded)
    X_temporal = _temporal_block(df_full)
    M_green    = np.zeros((len(df_full), n_green), dtype=np.float64)
    for i, c in enumerate(green_cols):
        v = df_full[c].to_numpy(dtype=np.float64)
        np.nan_to_num(v, nan=green_means[c], copy=False)
        M_green[:, i] = (v - green_means[c]) / green_stds[c]

    X = np.concatenate([X_temporal, M_full, M_green], axis=1)
    y = df_full[temp_col].to_numpy(dtype=np.float64)
    acc.update(X, y)

    beta = acc.solve()
    if beta is None:
        return {"error": "singular design matrix"}

    # Extract greening coefficients (last n_green entries)
    # These are in standardised units; convert to natural units for scenarios
    green_beta_std = beta[n_temporal + n_morph:]
    green_beta_nat = {
        c: float(green_beta_std[i]) / green_stds[c]
        for i, c in enumerate(green_cols)
    }

    return {
        "dataset":        dataset,
        "n_rows":         int(acc.n),
        "green_cols":     green_cols,
        "green_beta_nat": green_beta_nat,   # K per natural unit
        "green_means":    {c: round(green_means[c], 6) for c in green_cols},
        "green_stds":     {c: round(green_stds[c],  6) for c in green_cols},
        "dataset_stats":  dataset_stats(df_full),
        "run_timestamp":  run_timestamp(),
        "versions":       version_meta(),
    }


# ---------------------------------------------------------------------------
# Scenario evaluation
# ---------------------------------------------------------------------------
def _evaluate_scenarios(
    fit:       Dict,
    scenarios: List[Dict],
) -> List[Dict]:
    """
    For each scenario, compute DeltaT = sum_j  beta_j * (scenario_j - mean_j).

    Scenarios specify absolute feature values; the baseline is dataset mean.
    Features not mentioned in a scenario are held at their dataset mean (Delta=0).
    """
    green_beta = fit["green_beta_nat"]
    green_means = fit["green_means"]
    results = []

    for sc in scenarios:
        name    = sc.get("name", "unnamed")
        delta_T = 0.0
        terms   = {}
        for feat, beta in green_beta.items():
            scenario_val  = sc.get(feat, green_means.get(feat, 0.0))
            baseline_val  = green_means.get(feat, 0.0)
            delta_feat    = scenario_val - baseline_val
            contribution  = beta * delta_feat
            delta_T      += contribution
            terms[feat]   = {
                "beta_K_per_unit": round(beta, 6),
                "baseline":        round(baseline_val, 6),
                "scenario_val":    round(scenario_val, 6),
                "delta":           round(delta_feat, 6),
                "contribution_K":  round(contribution, 4),
            }
        results.append({
            "scenario":        name,
            "delta_T_K":       round(delta_T, 4),
            "delta_T_K_airT_est": round(delta_T * 0.5, 4),  # rough 0.5x LST→air conversion
            "terms":           terms,
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(
    datasets:   Optional[List[str]] = None,
    scenarios:  Optional[List[Dict]] = None,
    max_rows:   Optional[int]        = None,
    batch_size: int                  = 50_000,
    out_dir:    Path                 = _REPORTS_DIR,
) -> Dict:
    from utils import list_datasets

    if scenarios is None:
        scenarios = _DEFAULT_SCENARIOS

    if datasets is None:
        datasets = [name for name, _ in list_datasets()]

    all_results = {}
    for ds in datasets:
        print(f"  fitting {ds} ...", end=" ", flush=True)
        t0  = time.perf_counter()
        fit = _fit_dataset(ds, batch_size=batch_size, max_rows=max_rows)
        if "error" in fit:
            print(f"SKIP ({fit['error']})")
            all_results[ds] = fit
            continue
        sc_results = _evaluate_scenarios(fit, scenarios)
        elapsed = time.perf_counter() - t0
        print(f"{elapsed:.1f}s  ({fit['n_rows']:,} rows, {len(fit['green_cols'])} features)")
        all_results[ds] = {
            "_fit":     fit,
            "scenarios": sc_results,
        }

    return all_results


def _write_csv(results: Dict, path: Path) -> None:
    rows = []
    for ds, v in results.items():
        if "error" in v:
            continue
        for sc in v.get("scenarios", []):
            rows.append({
                "dataset":            ds,
                "scenario":           sc["scenario"],
                "delta_T_K_lst":      sc["delta_T_K"],
                "delta_T_K_air_est":  sc["delta_T_K_airT_est"],
            })
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def _print_policy_table(results: Dict) -> None:
    datasets_of_interest = [
        "full_representative",
        "outlier_heat_2019_jun_jul",
        "outlier_heat_2006_summer",
        "outlier_heat_2020_aug",
        "outlier_drought_2018",
        "outlier_cold_2012_feb",
    ]
    scenarios_of_interest = [
        "veg_10pct", "veg_20pct", "veg_30pct",
        "water_10pct",
        "veg30_water10",
        "trees_25",
    ]

    print(f"\n{'Scenario':<18}", end="")
    for ds in datasets_of_interest:
        short = ds.replace("full_representative", "full_rep") \
                  .replace("outlier_heat_", "heat_") \
                  .replace("outlier_drought_", "drt_") \
                  .replace("outlier_cold_", "cold_") \
                  .replace("_jun_jul", "19")
        print(f"  {short:<10}", end="")
    print()
    print("-" * (18 + 12 * len(datasets_of_interest)))

    for sc_name in scenarios_of_interest:
        print(f"{sc_name:<18}", end="")
        for ds in datasets_of_interest:
            if ds not in results or "scenarios" not in results[ds]:
                print(f"  {'—':<10}", end="")
                continue
            row = next((r for r in results[ds]["scenarios"] if r["scenario"] == sc_name), None)
            if row:
                print(f"  {row['delta_T_K']:>+7.2f} K  ", end="")
            else:
                print(f"  {'—':<10}", end="")
        print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset",    type=str,   default=None)
    ap.add_argument("--max-rows",   type=int,   default=None)
    ap.add_argument("--batch-size", type=int,   default=50_000)
    ap.add_argument("--scenarios",  type=str,   default=None,
                    help="JSON array of scenario dicts")
    ap.add_argument("--out-dir",    type=Path,  default=_REPORTS_DIR)
    args = ap.parse_args()

    scenarios = None
    if args.scenarios:
        try:
            scenarios = json.loads(args.scenarios)
        except json.JSONDecodeError as e:
            print(f"invalid --scenarios JSON: {e}", file=sys.stderr)
            return 2

    datasets = [args.dataset] if args.dataset else None

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("\nFitting joint OLS for scenario prediction ...")
    t_total = time.perf_counter()
    results = run(
        datasets=datasets,
        scenarios=scenarios,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        out_dir=args.out_dir,
    )

    json_path = args.out_dir / "scenario_predict.json"
    csv_path  = args.out_dir / "scenario_predict.csv"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    _write_csv(results, csv_path)

    _print_policy_table(results)

    elapsed = time.perf_counter() - t_total
    print(f"\nWrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"Total: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

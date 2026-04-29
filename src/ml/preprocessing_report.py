"""
ml/preprocessing_report.py
==========================
Inspect the preprocessing pipeline stage-by-stage and emit an HTML report.

Stages reported (one row per numeric feature column):

    1. raw           — input DataFrame as-loaded
    2. transformed   — after applying the transform pipeline (ml.transforms)
    3. imputed       — after NaN imputation using per-column medians
    4. scaled        — after StreamingScaler.transform_array

Per-column summary statistics (count, n_nan, mean, median, std, min, max,
q25, q75) and a matplotlib histogram are captured at every stage so drift,
outliers, or broken transforms jump out visually.

Usage
-----
    from ml.preprocessing_report import build_preprocessing_report
    from ml.transforms import cyclical
    from ml.scaler import StreamingScaler
    import pandas as pd

    df = pd.read_csv("sample_stream_output.csv")
    transforms = [cyclical("hour_of_day", 24),
                  cyclical("month_of_year", 12)]

    build_preprocessing_report(
        df,
        transforms = transforms,
        scaler     = None,           # None = fit a fresh StandardScaler sample
        out_path   = "reports/preprocessing.html",
        sample_n   = 100_000,
    )

Run directly (uses lst_models_test.py's sample CSV):
    python -m ml.preprocessing_report
"""

from __future__ import annotations

import datetime
import io
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ml.base import NEVER_FEATURES
from ml.scaler import StreamingScaler

log = logging.getLogger("lst_models.preprocessing_report")

# ---------------------------------------------------------------------------
# Stage names
# ---------------------------------------------------------------------------

_STAGES = ("raw", "transformed", "imputed", "scaled")
_STAGE_COLORS = {
    "raw":         "#6C757D",
    "transformed": "#4C72B0",
    "imputed":     "#55A868",
    "scaled":      "#C44E52",
}


# ---------------------------------------------------------------------------
# Pipeline reproduction (mirrors ml/base.py, without fitting a model)
# ---------------------------------------------------------------------------

def _apply_transforms(df: pd.DataFrame, transforms: Sequence[Callable]) -> pd.DataFrame:
    """Apply a transform list the way ml.base.LSTModel._apply_transforms does."""
    if not transforms:
        return df.copy()
    result      = df.copy()
    start_cols  = set(df.columns)
    inputs_used: set = set()
    for fn in transforms:
        try:
            extra = fn(result)
        except Exception as exc:
            log.warning("transform %s failed: %s", getattr(fn, "__name__", fn), exc)
            continue
        inputs_used.update(getattr(fn, "input_cols", []))
        if isinstance(extra, pd.DataFrame):
            if not extra.empty:
                for col in extra.columns:
                    result[col] = extra[col].values
            if getattr(fn, "drop_inputs", False):
                extra_cols = set(extra.columns)
                to_drop = [c for c in getattr(fn, "input_cols", [])
                           if c in result.columns and c not in extra_cols]
                result.drop(columns=to_drop, inplace=True, errors="ignore")

    # Global rule: any *raw* column consumed by an applied transform is
    # removed after the pipeline finishes.  See ml.base.LSTModel._apply_transforms.
    raw_to_drop = [c for c in start_cols
                   if c in inputs_used and c in result.columns]
    if raw_to_drop:
        result.drop(columns=raw_to_drop, inplace=True, errors="ignore")
    return result


def _auto_feature_cols(df: pd.DataFrame) -> List[str]:
    return [
        c for c in df.columns
        if c not in NEVER_FEATURES
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def _impute(df: pd.DataFrame, cols: Sequence[str]) -> tuple[pd.DataFrame, Dict[str, float]]:
    """Fill NaN via column medians; return (filled_df, medians_used)."""
    out = df.copy()
    medians: Dict[str, float] = {}
    for c in cols:
        if c not in out.columns:
            continue
        vals = out[c].dropna()
        med  = float(vals.median()) if not vals.empty else 0.0
        medians[c] = med
        if out[c].isna().any():
            out[c] = out[c].fillna(med)
    return out, medians


# ---------------------------------------------------------------------------
# Statistics + histograms
# ---------------------------------------------------------------------------

def _stats(series: pd.Series) -> Dict[str, float]:
    arr = pd.to_numeric(series, errors="coerce")
    clean = arr.dropna()
    if clean.empty:
        return {"count": len(arr), "n_nan": int(arr.isna().sum()),
                "mean": float("nan"), "median": float("nan"),
                "std":  float("nan"), "min":    float("nan"),
                "max":  float("nan"), "q25":    float("nan"),
                "q75":  float("nan")}
    q25, q50, q75 = np.quantile(clean, [0.25, 0.50, 0.75])
    return {
        "count":  int(len(arr)),
        "n_nan":  int(arr.isna().sum()),
        "mean":   float(clean.mean()),
        "median": float(q50),
        "std":    float(clean.std()),
        "min":    float(clean.min()),
        "max":    float(clean.max()),
        "q25":    float(q25),
        "q75":    float(q75),
    }


def _histogram_svg(
    series: pd.Series,
    title:  str,
    color:  str,
    bins:   int = 50,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(3.6, 2.0))
    if arr.size == 0:
        ax.text(0.5, 0.5, "all NaN", ha="center", va="center",
                transform=ax.transAxes, color="#999", fontsize=9)
    else:
        ax.hist(arr, bins=bins, color=color, edgecolor="white", linewidth=0.4)
        ax.axvline(float(np.mean(arr)),   color="black", linestyle="-",  linewidth=1)
        ax.axvline(float(np.median(arr)), color="black", linestyle="--", linewidth=1)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue().decode("utf-8")


# ---------------------------------------------------------------------------
# Scaler handling
# ---------------------------------------------------------------------------

def _build_or_use_scaler(
    imputed_df:   pd.DataFrame,
    cols:         Sequence[str],
    scaler:       Optional[StreamingScaler],
    default_scaler: str,
) -> StreamingScaler:
    """Return a fitted scaler — either the supplied one, or a fresh one."""
    if scaler is not None and getattr(scaler, "_is_fitted", False):
        return scaler
    s = scaler or StreamingScaler(default_scaler=default_scaler)
    s.partial_fit(imputed_df, list(cols))
    return s


def _scaled_frame(
    imputed_df: pd.DataFrame,
    cols:       Sequence[str],
    scaler:     StreamingScaler,
) -> pd.DataFrame:
    X  = imputed_df[list(cols)].to_numpy(dtype=np.float32)
    Xs = scaler.transform_array(X, list(cols))
    out = imputed_df.copy()
    for i, c in enumerate(cols):
        out[c] = Xs[:, i]
    return out


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_preprocessing_report(
    df:             pd.DataFrame,
    transforms:     Optional[Sequence[Callable]] = None,
    scaler:         Optional[StreamingScaler]    = None,
    out_path:       Union[str, Path]             = "reports/preprocessing.html",
    sample_n:       Optional[int]                = 100_000,
    feature_cols:   Optional[Sequence[str]]      = None,
    default_scaler: str                          = "standard",
    bins:           int                          = 50,
    random_state:   int                          = 42,
) -> Path:
    """Generate an HTML report of the preprocessing pipeline.

    Parameters
    ----------
    df : DataFrame
        Raw input. A random sample of ``sample_n`` rows is used to keep the
        report fast and the SVGs small; set ``sample_n=None`` to use all rows.
    transforms : list of Transform callables, optional
        Same list you'd pass to model.set_transforms().
    scaler : StreamingScaler, optional
        A fitted StreamingScaler. If None (or unfitted), a fresh
        ``StandardScaler`` is fitted on the imputed sample just for this report
        so the scaled-stage panel still shows something meaningful.
    sample_n : int, optional
        Subsample size (default 100 000). None = use full df.
    feature_cols : list, optional
        Restrict inspection to these columns. Default: auto-detect numeric
        non-metadata columns on the transformed frame.
    default_scaler : str
        Scaler type when no fitted scaler is provided.
    bins : int
        Histogram bins per panel.

    Returns
    -------
    Path written.
    """
    transforms = list(transforms or [])
    out_path   = Path(out_path)

    # Sample once, up front, so every stage sees the same rows
    if sample_n is not None and len(df) > sample_n:
        df_raw = df.sample(n=sample_n, random_state=random_state).reset_index(drop=True)
    else:
        df_raw = df.reset_index(drop=True)

    print(f"[preprocessing_report] sample: {len(df_raw):,} rows × {df_raw.shape[1]} cols")

    # Stage 2: transforms
    df_tr = _apply_transforms(df_raw, transforms)

    # Resolve feature columns against the transformed frame
    if feature_cols is None:
        cols = _auto_feature_cols(df_tr)
    else:
        cols = [c for c in feature_cols if c in df_tr.columns]
    if not cols:
        raise ValueError("No numeric feature columns found after transforms.")

    # Stage 3: imputation (only on the resolved feature columns)
    df_im, medians = _impute(df_tr, cols)

    # Stage 4: scaling
    active_scaler = _build_or_use_scaler(df_im, cols, scaler, default_scaler)
    df_sc         = _scaled_frame(df_im, cols, active_scaler)

    stage_frames: Dict[str, pd.DataFrame] = {
        "raw":         df_raw,
        "transformed": df_tr,
        "imputed":     df_im,
        "scaled":      df_sc,
    }

    # Build per-column panels
    column_sections: List[str] = []
    for col in tqdm(cols, desc="[preprocessing_report] columns", unit="col"):
        panels: List[str] = []
        stats_rows: List[str] = []
        for stage in _STAGES:
            frame = stage_frames[stage]
            if col not in frame.columns:
                panels.append(
                    f"<div class='panel'><div class='panel-title'>{stage}</div>"
                    f"<div class='missing'>column absent at this stage</div></div>"
                )
                stats_rows.append(f"<tr><td>{stage}</td>"
                                  f"<td colspan='9' class='missing'>–</td></tr>")
                continue

            s     = _stats(frame[col])
            svg   = _histogram_svg(frame[col],
                                   title=f"{stage} (μ={s['mean']:.3g}, σ={s['std']:.3g})",
                                   color=_STAGE_COLORS[stage], bins=bins)
            panels.append(
                f"<div class='panel'>"
                f"  <div class='panel-title'>{stage}</div>"
                f"  <div class='chart'>{svg}</div>"
                f"</div>"
            )
            stats_rows.append(
                f"<tr><td>{stage}</td>"
                f"<td>{s['count']:,}</td><td>{s['n_nan']:,}</td>"
                f"<td>{s['mean']:.4g}</td><td>{s['median']:.4g}</td>"
                f"<td>{s['std']:.4g}</td><td>{s['min']:.4g}</td>"
                f"<td>{s['max']:.4g}</td><td>{s['q25']:.4g}</td>"
                f"<td>{s['q75']:.4g}</td></tr>"
            )

        column_sections.append(f"""
<section class="col-section">
  <h3>{col}</h3>
  <div class="panel-row">{''.join(panels)}</div>
  <table class="stats">
    <thead><tr><th>stage</th><th>count</th><th>n_nan</th><th>mean</th>
      <th>median</th><th>std</th><th>min</th><th>max</th>
      <th>q25</th><th>q75</th></tr></thead>
    <tbody>{''.join(stats_rows)}</tbody>
  </table>
</section>
""")

    # Pipeline summary (transforms + scaler types)
    transform_names = [getattr(fn, "__name__", str(fn)) for fn in transforms] or ["(none)"]
    scaler_summary  = ", ".join(
        f"{c}:{active_scaler.scaler_for(c)}" for c in cols[:6]
    )
    if len(cols) > 6:
        scaler_summary += f", … (+{len(cols) - 6} more)"

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Preprocessing Pipeline Report</title>
<style>
  body{{font-family:system-ui,sans-serif;margin:1.5em;color:#222}}
  h1{{border-bottom:2px solid #4C72B0;padding-bottom:.3em}}
  h3{{color:#333;margin-top:1.2em;border-bottom:1px solid #eee;padding-bottom:.2em}}
  .meta{{color:#666;font-size:.9em;margin-bottom:1.5em}}
  .col-section{{margin-bottom:2em;page-break-inside:avoid}}
  .panel-row{{display:flex;flex-wrap:wrap;gap:.4em;margin-bottom:.6em}}
  .panel{{flex:1 1 24%;min-width:280px;border:1px solid #ddd;border-radius:4px;
         padding:.4em;background:#fafafa}}
  .panel-title{{font-size:.8em;color:#555;margin-bottom:.2em;text-transform:uppercase;
               letter-spacing:.05em}}
  .chart svg{{max-width:100%;height:auto}}
  .missing{{color:#aaa;font-style:italic;font-size:.85em}}
  table.stats{{border-collapse:collapse;font-size:.82em;width:100%}}
  table.stats td,table.stats th{{border:1px solid #ccc;padding:.25em .5em;text-align:right}}
  table.stats th{{background:#eef3f8;text-align:center}}
  table.stats td:first-child{{text-align:left;font-weight:600;background:#f5f5f5}}
  code{{background:#f0f0f0;padding:.1em .3em;border-radius:3px}}
</style></head><body>
<h1>Preprocessing Pipeline Report</h1>
<p class="meta">
  Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}<br>
  Sample: {len(df_raw):,} rows (from {len(df):,} total)<br>
  Columns inspected: {len(cols)}<br>
  Transforms: <code>{', '.join(transform_names)}</code><br>
  Scaler: <code>{scaler_summary}</code>
  {'<span style="color:#C44E52"> (fitted for this report only — pass a pre-fitted one for accuracy)</span>'
    if scaler is None or not getattr(scaler, '_is_fitted', False) else ''}
</p>
{''.join(column_sections)}
</body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"[preprocessing_report] wrote {out_path} ({len(html):,} bytes)")
    return out_path


# ---------------------------------------------------------------------------
# CLI entry point — mirrors lst_models_test.py's sample source
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from ml.transforms import cyclical

    parser = argparse.ArgumentParser(description="Preprocessing pipeline HTML report")
    parser.add_argument("--csv",      default=None,
                        help="Input CSV (default: <src>/sample_stream_output.csv)")
    parser.add_argument("--out",      default=None,
                        help="Output HTML path (default: <src>/reports/preprocessing.html)")
    parser.add_argument("--sample-n", type=int, default=100_000)
    parser.add_argument("--bins",     type=int, default=50)
    args = parser.parse_args()

    src      = Path(__file__).resolve().parent.parent
    csv_path = Path(args.csv) if args.csv else src / "sample_stream_output.csv"
    out_path = Path(args.out) if args.out else src / "reports" / "preprocessing.html"

    print(f"[preprocessing_report] loading {csv_path} ...")
    df_in = pd.read_csv(csv_path, nrows=args.sample_n * 5 if args.sample_n else None)

    transforms = [
        cyclical("hour_of_day",   24),
        cyclical("day_of_year",   365),
        cyclical("month_of_year", 12),
    ]

    build_preprocessing_report(
        df_in,
        transforms = transforms,
        scaler     = None,
        out_path   = out_path,
        sample_n   = args.sample_n,
        bins       = args.bins,
    )

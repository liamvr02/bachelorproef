"""
render_stability_report.py
==========================

Cross-dataset stability analysis of feature correlations from ds_reports/cross_corr.json.

A correlation is considered "stable" if its magnitude is consistent across many
independent data slices.  This script aggregates the per-pair metrics from every
dataset, computes mean / std / CV per pair, and renders a single HTML report at
ds_reports/stability.html.

Methodology
-----------
Datasets are split into two reliability tiers:

  FULL datasets  (>=10 000 rows) : full_representative, outlier_*, single_year_*
                                   Provide reliable estimates; used for primary
                                   stability statistics.

  SPATIAL datasets (<10 000 rows): point_*
                                   Estimates are noisy due to small n; treated as
                                   a secondary "local agreement" check, not included
                                   in stability scores.

Per pair, over the FULL datasets:
  mean_abs_r   = mean of |pearson_r| across datasets
  std_abs_r    = std  of |pearson_r| across datasets
  cv           = std_abs_r / (mean_abs_r + 1e-6)   (coefficient of variation)
  sign_pct     = fraction of datasets where pearson_r has the majority sign
  stability    = max(0, 1 - cv)  -- 1 = perfectly stable, 0 = completely erratic

Same metrics are computed for spearman_r and MI.

Output
------
  ds_reports/stability.html   visual HTML report
  ds_test_stability.json      machine-readable aggregated stats

Usage
-----
    python src/ds/render_stability_report.py
    python src/ds/render_stability_report.py --results src/ds_reports/cross_corr.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_DS_DIR            = Path(__file__).parent
_SRC               = _DS_DIR.parent
_RESULTS           = _SRC / "ds_reports" / "cross_corr.json"
_OUT_DIR           = _SRC / "ds_reports"
_STABILITY_JSON    = _SRC / "ds_test_stability.json"

_FULL_ROW_THRESH   = 10_000
_SOURCE_ORDER      = ["LST", "DHM", "Trees", "UA", "WIS"]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _split_datasets(results: dict) -> Tuple[List[str], List[str]]:
    full, spatial = [], []
    for k, v in results.items():
        if k == "_meta":
            continue
        (full if v.get("n_rows_total", 0) >= _FULL_ROW_THRESH else spatial).append(k)
    return sorted(full), sorted(spatial)


def aggregate(results: dict) -> Tuple[List[dict], dict]:
    """
    Return (pair_stats, meta) where pair_stats is a list of dicts,
    one per unique (feature_a, feature_b) pair.
    """
    full_names, spatial_names = _split_datasets(results)

    full_vals:    Dict[Tuple, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    spatial_vals: Dict[Tuple, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    datasets_seen: Dict[Tuple, Dict[str, List[str]]] = defaultdict(lambda: {"full": [], "spatial": []})

    for ds_name, bucket, acc in [
        (n, "full",    full_vals)    for n in full_names
    ] + [
        (n, "spatial", spatial_vals) for n in spatial_names
    ]:
        for pair in results[ds_name].get("pairs", []):
            key = (pair["feature_a"], pair["source_a"],
                   pair["feature_b"], pair["source_b"])
            for metric in ("pearson_r", "spearman_r", "mi"):
                v = pair.get(metric)
                if v is not None:
                    acc[key][metric].append(v)
            datasets_seen[key][bucket].append(ds_name)

    def _stats(vals: List[float], abs_mode: bool) -> dict:
        if not vals:
            return {}
        vs = [abs(v) for v in vals] if abs_mode else vals
        n  = len(vs)
        mu = sum(vs) / n
        var = sum((x - mu) ** 2 for x in vs) / n if n > 1 else 0.0
        sd = math.sqrt(var)
        cv = sd / (mu + 1e-6)
        return {
            "mean": round(mu, 6),
            "std":  round(sd, 6),
            "cv":   round(cv, 4),
            "min":  round(min(vs), 6),
            "max":  round(max(vs), 6),
            "n":    n,
        }

    def _sign_pct(vals: List[float]) -> Optional[float]:
        if not vals:
            return None
        pos = sum(1 for v in vals if v > 0)
        return round(max(pos, len(vals) - pos) / len(vals), 3)

    pair_stats = []
    all_keys = sorted(
        set(full_vals.keys()) | set(spatial_vals.keys()),
        key=lambda k: (k[1], k[3], k[0], k[2]),
    )

    for key in all_keys:
        fa, sa, fb, sb = key
        fv = full_vals[key]
        sv = spatial_vals[key]

        pr_full = fv.get("pearson_r",  [])
        sr_full = fv.get("spearman_r", [])
        mi_full = fv.get("mi",         [])
        pr_spat = sv.get("pearson_r",  [])

        ps_full = _stats(pr_full, abs_mode=True)
        ss_full = _stats(sr_full, abs_mode=True)
        ms_full = _stats(mi_full, abs_mode=False)

        stability = round(max(0.0, 1.0 - ps_full.get("cv", 1.0)), 4) if ps_full else None
        sign_pct  = _sign_pct(pr_full)

        full_sign = 1 if sum(pr_full) > 0 else -1 if pr_full else 0
        spatial_agree = None
        if pr_spat and full_sign != 0:
            spat_same = sum(1 for v in pr_spat if (v > 0) == (full_sign > 0))
            spatial_agree = round(spat_same / len(pr_spat), 3)

        pair_stats.append({
            "feature_a":      fa,
            "source_a":       sa,
            "feature_b":      fb,
            "source_b":       sb,
            "n_full":         len(datasets_seen[key]["full"]),
            "n_spatial":      len(datasets_seen[key]["spatial"]),
            "pearson":        ps_full,
            "spearman":       ss_full,
            "mi":             ms_full,
            "stability":      stability,
            "sign_pct":       sign_pct,
            "spatial_agree":  spatial_agree,
            "full_datasets":  datasets_seen[key]["full"],
        })

    meta = {
        "full_datasets":    full_names,
        "spatial_datasets": spatial_names,
        "n_pairs":          len(pair_stats),
        "full_row_thresh":  _FULL_ROW_THRESH,
    }
    return pair_stats, meta


# ---------------------------------------------------------------------------
# Colour / format helpers
# ---------------------------------------------------------------------------

def _lerp(lo, hi, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(lo, hi))

def _hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

_WHITE  = (255, 255, 255)
_ORANGE = (255, 160,  60)
_RED    = (180,  30,  10)
_GREEN  = ( 27, 140,  40)
_YELLOW = (255, 220,  50)
_BLUE   = ( 21, 101, 192)

def _r_col(v):
    v = min(1.0, abs(v))
    return _hex(_lerp(_lerp(_WHITE, _ORANGE, v / 0.5), _lerp(_ORANGE, _RED, (v - 0.5) / 0.5), 1 if v > 0.5 else 0))

def _cv_col(cv):
    t = min(1.0, cv / 1.5)
    if t < 0.5:
        return _hex(_lerp(_GREEN, _YELLOW, t / 0.5))
    return _hex(_lerp(_YELLOW, _RED, (t - 0.5) / 0.5))

def _mi_col(v, max_v):
    t = v / max_v if max_v > 0 else 0
    return _hex(_lerp(_WHITE, _BLUE, t))

def _stab_col(s):
    return _hex(_lerp(_RED, _GREEN, s))

def _fmt(v, d=3):
    return f"{v:.{d}f}" if v is not None else "—"

def _pm(s: dict, key="mean", sd_key="std") -> str:
    mu = s.get(key)
    sd = s.get(sd_key)
    if mu is None:
        return "—"
    if sd is None:
        return f"{mu:.3f}"
    return f"{mu:.3f}<small> &plusmn;{sd:.3f}</small>"


# ---------------------------------------------------------------------------
# Heatmap helpers
# ---------------------------------------------------------------------------

def _heatmap_data(pair_stats, get_val):
    acc = defaultdict(list)
    for p in pair_stats:
        v = get_val(p)
        if v is None:
            continue
        key = tuple(sorted([p["source_a"], p["source_b"]]))
        acc[key].append(v)
    return {k: sum(v) / len(v) for k, v in acc.items()}

def _render_heatmap(sources, data, col_fn, fmt=".3f") -> str:
    header = "".join(f"<th>{s}</th>" for s in sources)
    rows = []
    for sa in sources:
        cells = [f"<td class='hm-label'>{sa}</td>"]
        for sb in sources:
            key = tuple(sorted([sa, sb]))
            if sa == sb:
                cells.append("<td class='hm-na'>—</td>")
            elif key in data:
                v   = data[key]
                bg  = col_fn(v)
                txt = format(v, fmt)
                cells.append(f"<td class='hm-cell' style='background:{bg}'>{txt}</td>")
            else:
                cells.append("<td class='hm-na'>n/a</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (f"<table class='heatmap'>"
            f"<thead><tr><th></th>{header}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


# ---------------------------------------------------------------------------
# Table rows
# ---------------------------------------------------------------------------

def _pair_rows(pair_stats: List[dict], max_mi: float) -> str:
    rows = []
    for p in pair_stats:
        fa, sa = p["feature_a"], p["source_a"]
        fb, sb = p["feature_b"], p["source_b"]

        pr = p.get("pearson",  {})
        sr = p.get("spearman", {})
        mi = p.get("mi",       {})
        stab     = p.get("stability")
        sign_pct = p.get("sign_pct")
        spat     = p.get("spatial_agree")
        n_full   = p.get("n_full", 0)

        pr_mu = pr.get("mean", 0) or 0
        pr_cv = pr.get("cv",   1) or 1
        mi_mu = mi.get("mean", 0) or 0

        stab_bg  = _stab_col(stab) if stab is not None else "#eee"
        cv_bg    = _cv_col(pr_cv)
        r_bg     = _r_col(pr_mu)
        mi_bg    = _mi_col(mi_mu, max_mi)

        sign_html = ""
        if sign_pct is not None:
            sign_html = f"{sign_pct*100:.0f}%"

        spat_html = "—"
        if spat is not None:
            icon = "&#10003;" if spat >= 0.5 else "&#10007;"
            spat_html = f"{icon} {spat*100:.0f}%"

        def r_pair(s):
            mu = s.get("mean"); sd = s.get("std"); cv = s.get("cv")
            if mu is None:
                return "<td>—</td><td>—</td>"
            r_c = _r_col(mu)
            cv_c = _cv_col(cv or 0)
            return (
                f"<td style='background:{r_c}' data-v='{mu:.6f}'>"
                f"<strong>{mu:.3f}</strong>"
                f"<span style='color:#666;font-size:11px'> &plusmn;{sd:.3f}</span></td>"
                f"<td style='background:{cv_c}' data-v='{cv:.4f}'>{cv:.3f}</td>"
            )

        rows.append(
            f"<tr>"
            f"<td class='feat'>{fa}</td>"
            f"<td><span class='src-badge src-{sa}'>{sa}</span></td>"
            f"<td class='feat'>{fb}</td>"
            f"<td><span class='src-badge src-{sb}'>{sb}</span></td>"
            f"<td>{n_full}</td>"
            + r_pair(pr) + r_pair(sr)
            + f"<td style='background:{mi_bg}' data-v='{mi_mu:.6f}'>"
            f"<strong>{mi_mu:.3f}</strong>"
            f"<span style='color:#666;font-size:11px'> &plusmn;{mi.get('std',0):.3f}</span></td>"
            f"<td>{sign_html}</td>"
            f"<td data-v='{stab if stab is not None else -1:.4f}' "
            f"style='background:{stab_bg};font-weight:700'>"
            f"{'—' if stab is None else f'{stab:.3f}'}</td>"
            f"<td>{spat_html}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# CSS + JS
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font: 14px/1.5 'Segoe UI', Arial, sans-serif; color: #222;
       background: #f4f6f8; padding: 24px; }
a { color: #1565c0; text-decoration: none; }
a:hover { text-decoration: underline; }
h1 { font-size: 1.5rem; margin-bottom: 8px; }
h2 { font-size: 1.1rem; margin: 24px 0 8px; color: #444;
     border-bottom: 1px solid #ccc; padding-bottom: 4px; }
.meta-box { background: #fff; border: 1px solid #dde; border-radius: 4px;
            padding: 12px 16px; margin-bottom: 12px; display: flex;
            flex-wrap: wrap; gap: 24px; font-size: 13px; }
.meta-box span { color: #555; }
.meta-box strong { color: #111; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 8px;
          font-size: 12px; color: #555; }
.legend-swatch { display: inline-block; width: 40px; height: 14px;
                 border-radius: 2px; vertical-align: middle; margin-right: 4px; }
.heatmaps { display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 8px; }
.heatmap-block h3 { font-size: .85rem; font-weight: 600; margin-bottom: 4px; color: #555; }
table.heatmap { border-collapse: collapse; font-size: 12px; }
table.heatmap th, table.heatmap td { border: 1px solid #ccc; padding: 4px 8px;
                                      text-align: center; min-width: 54px; }
table.heatmap th { background: #e8ecf0; font-weight: 600; }
.hm-label { background: #e8ecf0; font-weight: 600; text-align: left; }
.hm-na { background: #f0f0f0; color: #aaa; }
.controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
             margin-bottom: 8px; }
.controls input, .controls select {
    border: 1px solid #bbb; border-radius: 3px; padding: 4px 8px;
    font-size: 13px; background: #fff; }
.controls input[type=text] { width: 200px; }
.controls input[type=range] { width: 120px; }
.controls label { font-size: 13px; color: #555; }
.count-info { font-size: 12px; color: #777; margin-left: auto; }
.table-wrap { overflow-x: auto; max-height: 72vh; overflow-y: auto;
              border: 1px solid #dde; border-radius: 4px; background: #fff; }
table.pairs { border-collapse: collapse; width: 100%; font-size: 12px; }
table.pairs thead tr { position: sticky; top: 0; z-index: 2; }
table.pairs th { background: #2c3e50; color: #fff; padding: 6px 10px;
                  text-align: left; white-space: nowrap; cursor: pointer;
                  user-select: none; }
table.pairs th:hover { background: #3d5166; }
table.pairs th.sorted-asc::after  { content: ' \\25b2'; font-size: 10px; }
table.pairs th.sorted-desc::after { content: ' \\25bc'; font-size: 10px; }
table.pairs td { padding: 4px 10px; border-bottom: 1px solid #eee; white-space: nowrap; }
table.pairs tr:hover td { background: #f0f4fa; }
table.pairs tr.hidden { display: none; }
.feat { font-family: monospace; font-size: 11px; }
.src-badge { display: inline-block; padding: 1px 6px; border-radius: 10px;
              font-size: 11px; font-weight: 600; }
.src-LST   { background: #e3f2fd; color: #0d47a1; }
.src-DHM   { background: #e8f5e9; color: #1b5e20; }
.src-Trees { background: #f3e5f5; color: #4a148c; }
.src-UA    { background: #fff3e0; color: #e65100; }
.src-WIS   { background: #fce4ec; color: #880e4f; }
.nav-bar   { margin-bottom: 16px; font-size: 13px; }
small { font-size: 11px; }
"""

_JS = """
(function() {
  var sortCol = 7, sortAsc = false;  // default: stability desc

  function cellVal(cell) {
    return cell.dataset.v !== undefined ? cell.dataset.v : cell.textContent.trim();
  }
  function doSort(col) {
    var table  = document.getElementById('pairs-table');
    var tbody  = table.querySelector('tbody');
    var ths    = table.querySelectorAll('thead th');
    if (sortCol === col) { sortAsc = !sortAsc; }
    else { sortCol = col; sortAsc = col < 4; }
    ths.forEach(function(th, i) {
      th.className = i === col ? (sortAsc ? 'sorted-asc' : 'sorted-desc') : '';
    });
    var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
    rows.sort(function(a, b) {
      var av = cellVal(a.cells[col]), bv = cellVal(b.cells[col]);
      var an = parseFloat(av), bn = parseFloat(bv);
      var cmp = (!isNaN(an) && !isNaN(bn)) ? (an - bn) : av.localeCompare(bv);
      return sortAsc ? cmp : -cmp;
    });
    rows.forEach(function(r) { tbody.appendChild(r); });
    updateCount();
  }
  function applyFilter() {
    var text   = document.getElementById('filter').value.toLowerCase();
    var srcA   = document.getElementById('src-a').value;
    var srcB   = document.getElementById('src-b').value;
    var minR   = parseFloat(document.getElementById('min-r').value) || 0;
    var maxCV  = parseFloat(document.getElementById('max-cv').value);
    if (isNaN(maxCV)) maxCV = 999;
    document.getElementById('min-r-val').textContent  = minR.toFixed(2);
    document.getElementById('max-cv-val').textContent = maxCV.toFixed(2);
    var rows = document.querySelectorAll('#pairs-table tbody tr');
    rows.forEach(function(r) {
      var fa = r.cells[0].textContent.toLowerCase();
      var sa = r.cells[1].textContent.trim();
      var fb = r.cells[2].textContent.toLowerCase();
      var sb = r.cells[3].textContent.trim();
      var rv = parseFloat(r.cells[5].dataset.v || 0);
      var cv = parseFloat(r.cells[6].dataset.v || 999);
      var show = (
        (fa.includes(text) || fb.includes(text)) &&
        (srcA === '' || sa === srcA) &&
        (srcB === '' || sb === srcB) &&
        rv >= minR && cv <= maxCV
      );
      r.classList.toggle('hidden', !show);
    });
    updateCount();
  }
  function updateCount() {
    var all = document.querySelectorAll('#pairs-table tbody tr').length;
    var vis = document.querySelectorAll('#pairs-table tbody tr:not(.hidden)').length;
    document.getElementById('row-count').textContent =
      'Showing ' + vis + ' of ' + all + ' pairs';
  }
  window.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('#pairs-table thead th').forEach(function(th, i) {
      th.addEventListener('click', function() { doSort(i); });
    });
    ['filter','src-a','src-b','min-r','max-cv'].forEach(function(id) {
      document.getElementById(id).addEventListener('input', applyFilter);
      document.getElementById(id).addEventListener('change', applyFilter);
    });
    doSort(9);
    updateCount();
  });
})();
"""

# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------

def _source_opts(sources):
    return "<option value=''>All</option>" + "".join(
        f"<option>{s}</option>" for s in sources
    )

def render(pair_stats: List[dict], agg_meta: dict, results_meta: dict,
           out_path: Path) -> None:

    sources = sorted(
        {p["source_a"] for p in pair_stats} | {p["source_b"] for p in pair_stats},
        key=lambda s: _SOURCE_ORDER.index(s) if s in _SOURCE_ORDER else 99,
    )

    max_mi = max((p["mi"].get("mean") or 0) for p in pair_stats) or 1.0

    hm_r    = _heatmap_data(pair_stats, lambda p: p["pearson"].get("mean"))
    hm_cv   = _heatmap_data(pair_stats, lambda p: p["pearson"].get("cv"))
    hm_stab = _heatmap_data(pair_stats, lambda p: p.get("stability"))

    heatmaps_html = (
        "<div class='heatmaps'>"
        "<div class='heatmap-block'><h3>Mean |Pearson r| (strength)</h3>"
        + _render_heatmap(sources, hm_r, _r_col) +
        "</div>"
        "<div class='heatmap-block'><h3>Mean CV of |Pearson r| (lower = more stable)</h3>"
        + _render_heatmap(sources, hm_cv, _cv_col) +
        "</div>"
        "<div class='heatmap-block'><h3>Mean stability score (higher = more stable)</h3>"
        + _render_heatmap(sources, hm_stab, _stab_col) +
        "</div></div>"
    )

    legend_html = (
        "<div class='legend'>"
        "<div><span class='legend-swatch' "
        "style='background:linear-gradient(to right,#fff,#ffa03c,#b41e0a)'></span>"
        " |r|: low &rarr; high</div>"
        "<div><span class='legend-swatch' "
        "style='background:linear-gradient(to right,#1b8c28,#ffdc32,#b41e0a)'></span>"
        " CV: stable &rarr; unstable</div>"
        "<div><span class='legend-swatch' "
        "style='background:linear-gradient(to right,#b41e0a,#1b8c28)'></span>"
        " Stability: low &rarr; high</div>"
        "</div>"
    )

    n_strong_stable = sum(
        1 for p in pair_stats
        if (p["pearson"].get("mean") or 0) >= 0.2
        and (p.get("stability") or 0) >= 0.7
    )
    meta_html = (
        "<div class='meta-box'>"
        f"<div><span>Full datasets </span><strong>{len(agg_meta['full_datasets'])}</strong></div>"
        f"<div><span>Spatial datasets </span><strong>{len(agg_meta['spatial_datasets'])}</strong></div>"
        f"<div><span>Unique pairs </span><strong>{len(pair_stats):,}</strong></div>"
        f"<div><span>Strong + stable (|r|>=0.2, stab>=0.7) </span>"
        f"<strong>{n_strong_stable:,}</strong></div>"
        f"<div><span>Full datasets </span>"
        + " ".join(f"<span class='tag'>{d}</span>" for d in agg_meta["full_datasets"])
        + "</div>"
        "</div>"
    )

    src_opts = _source_opts(sources)
    controls_html = (
        "<div class='controls'>"
        "<label>Filter: <input id='filter' type='text' placeholder='feature name...'></label>"
        f"<label>Src A: <select id='src-a'>{src_opts}</select></label>"
        f"<label>Src B: <select id='src-b'>{src_opts}</select></label>"
        "<label>Min |r|: <input id='min-r' type='range' min='0' max='1' step='0.05' value='0'>"
        " <span id='min-r-val'>0.00</span></label>"
        "<label>Max CV: <input id='max-cv' type='range' min='0' max='3' step='0.1' value='3'>"
        " <span id='max-cv-val'>3.00</span></label>"
        "<span class='count-info' id='row-count'></span>"
        "</div>"
    )

    table_html = (
        "<div class='table-wrap'>"
        "<table class='pairs' id='pairs-table'>"
        "<thead><tr>"
        "<th>Feature A</th><th>Src A</th>"
        "<th>Feature B</th><th>Src B</th>"
        "<th>N</th>"
        "<th>Mean |Pearson r|</th><th>CV</th>"
        "<th>Mean |Spearman r|</th><th>CV</th>"
        "<th>Mean MI</th>"
        "<th>Sign%</th>"
        "<th>Stability</th>"
        "<th>Spatial agree</th>"
        "</tr></thead>"
        f"<tbody>{_pair_rows(pair_stats, max_mi)}</tbody>"
        "</table></div>"
    )

    body = (
        "<div class='nav-bar'><a href='index.html'>&larr; Index</a></div>"
        "<h1>Cross-dataset stability of feature correlations</h1>"
        f"{meta_html}"
        "<h2>Source-pair overview</h2>"
        f"{legend_html}"
        f"{heatmaps_html}"
        "<h2>All pairs <small style='font-weight:normal;color:#777'>"
        "(default sort: stability desc; sliders filter min |r| and max CV)</small></h2>"
        f"{controls_html}"
        f"{table_html}"
    )

    html = (
        "<!DOCTYPE html><html lang='en'><head>"
        "<meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>DS Stability Report</title>"
        f"<style>{_CSS}</style>"
        f"<script>{_JS}</script>"
        f"</head><body>{body}</body></html>"
    )
    out_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results",  type=Path, default=_RESULTS)
    parser.add_argument("--out-dir",  type=Path, default=_OUT_DIR)
    parser.add_argument("--out-json", type=Path, default=_STABILITY_JSON)
    args = parser.parse_args()

    if not args.results.exists():
        print(f"results not found: {args.results}", file=sys.stderr)
        return 1

    results = json.loads(args.results.read_text(encoding="utf-8"))
    results_meta = results.get("_meta", {})

    print("aggregating cross-dataset pair statistics ...")
    pair_stats, agg_meta = aggregate(results)
    print(f"  {len(pair_stats):,} unique pairs over "
          f"{len(agg_meta['full_datasets'])} full + "
          f"{len(agg_meta['spatial_datasets'])} spatial datasets")

    args.out_json.write_text(
        json.dumps({"meta": agg_meta, "pairs": pair_stats}, indent=2),
        encoding="utf-8",
    )
    print(f"  wrote {args.out_json}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_html = args.out_dir / "stability.html"
    render(pair_stats, agg_meta, results_meta, out_html)
    print(f"  wrote {out_html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

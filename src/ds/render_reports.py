"""
render_reports.py
=================
Unified HTML report generator for all ds analysis outputs.

Reads (whichever exist):
  ds_reports/simple_corr.json
  ds_reports/cross_corr.json
  ds_reports/partial_corr.json
  ds_reports/morans_i.json
  ds_reports/spatial_cv.json

Generates (all self-contained, no external deps):
  ds_reports/index.html           -- navigation hub
  ds_reports/simple_corr.html     -- Pearson r, Spearman rho, slope per greening feature
  ds_reports/{dataset}.html       -- per-dataset cross-corr heatmaps + pairs table
  ds_reports/partial_corr.html    -- partial correlations for all datasets
  ds_reports/morans_i.html        -- Moran's I results for all datasets
  ds_reports/spatial_cv.html      -- spatial CV results for all datasets
  ds_reports/stability.html       -- cross-dataset stability for all four analyses

Usage
-----
    python src/ds/render_reports.py
    python src/ds/render_reports.py --results-dir src/ds_reports --out src/ds_reports
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_DS_DIR      = Path(__file__).parent
_SRC         = _DS_DIR.parent
_RESULTS_DIR = _SRC / "ds_reports"
_OUT_DIR     = _SRC / "ds_reports"

_SOURCE_ORDER  = ["LST", "DHM", "Trees", "UA", "WIS"]
_FULL_ROW_THRESH = 10_000


# ── Colour helpers ─────────────────────────────────────────────────────────────

def _lerp(lo: Tuple, hi: Tuple, t: float) -> Tuple:
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(lo, hi))

def _hex(rgb: Tuple) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)

_WHITE  = (255, 255, 255)
_ORANGE = (255, 160,  60)
_RED    = (180,  30,  10)
_GREEN  = ( 27, 140,  40)
_YELLOW = (255, 220,  50)
_BLUE   = ( 21, 101, 192)

def _r_col(v: float) -> str:
    v = min(1.0, abs(float(v)))
    if v <= 0.5:
        return _hex(_lerp(_WHITE, _ORANGE, v / 0.5))
    return _hex(_lerp(_ORANGE, _RED, (v - 0.5) / 0.5))

def _mi_col(v: float, max_v: float) -> str:
    t = v / max_v if max_v > 0 else 0.0
    return _hex(_lerp(_WHITE, _BLUE, t))

def _cv_col(cv: float) -> str:
    t = min(1.0, cv / 1.5)
    if t < 0.5:
        return _hex(_lerp(_GREEN, _YELLOW, t / 0.5))
    return _hex(_lerp(_YELLOW, _RED, (t - 0.5) / 0.5))

def _stab_col(s: float) -> str:
    return _hex(_lerp(_RED, _GREEN, s))

def _moran_col(I: float) -> str:
    """Blue for positive I (clustered), white for 0, orange for negative."""
    if I >= 0:
        return _hex(_lerp(_WHITE, _BLUE, min(1.0, I / 0.5)))
    return _hex(_lerp(_WHITE, _ORANGE, min(1.0, abs(I) / 0.3)))

def _r2_col(r2: float) -> str:
    r2c = max(0.0, min(1.0, float(r2)))
    return _hex(_lerp(_ORANGE, _GREEN, r2c))

def _sig(p: Optional[float]) -> str:
    if p is None:    return ""
    if p < 0.001:    return "***"
    if p < 0.01:     return "**"
    if p < 0.05:     return "*"
    return ""

def _fmt_r(v: Optional[float]) -> str:
    return f"{v:+.3f}" if v is not None else "—"

def _fmt_p(v: Optional[float]) -> str:
    if v is None:    return "—"
    if v < 1e-10:    return "&lt;1e-10"
    if v < 0.001:    return f"{v:.2e}"
    return f"{v:.4f}"

def _fmt(v: Optional[float], d: int = 3) -> str:
    return f"{v:.{d}f}" if v is not None else "—"


# ── Shared CSS + JS ────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font: 14px/1.5 'Segoe UI', Arial, sans-serif; color: #222;
       background: #f4f6f8; padding: 24px; }
a { color: #1565c0; text-decoration: none; }
a:hover { text-decoration: underline; }
h1 { font-size: 1.5rem; margin-bottom: 8px; }
h2 { font-size: 1.1rem; margin: 24px 0 8px; color: #444;
     border-bottom: 1px solid #ccc; padding-bottom: 4px; }
h3 { font-size: .9rem; margin: 16px 0 6px; color: #555; }
.nav-bar { margin-bottom: 16px; font-size: 13px; }
.meta-box { background: #fff; border: 1px solid #dde; border-radius: 4px;
            padding: 12px 16px; margin-bottom: 12px; display: flex;
            flex-wrap: wrap; gap: 24px; font-size: 13px; }
.meta-box span { color: #555; }
.meta-box strong { color: #111; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 8px;
          font-size: 12px; color: #555; }
.legend-swatch { display: inline-block; width: 40px; height: 14px;
                 border-radius: 2px; vertical-align: middle; margin-right: 4px; }

/* Cards on index */
.card-grid { display: flex; flex-wrap: wrap; gap: 16px; margin: 16px 0; }
.card { background: #fff; border: 1px solid #dde; border-radius: 6px;
        padding: 16px 20px; min-width: 200px; flex: 1 1 200px;
        box-shadow: 0 1px 3px rgba(0,0,0,.06); }
.card h3 { font-size: 1rem; margin-bottom: 6px; color: #1565c0; }
.card p  { font-size: 12px; color: #666; margin-top: 4px; }

/* Heatmaps */
.heatmaps { display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 8px; }
.heatmap-block h3 { font-size: .85rem; font-weight: 600; margin-bottom: 4px; color: #555; }
table.heatmap { border-collapse: collapse; font-size: 12px; }
table.heatmap th, table.heatmap td { border: 1px solid #ccc; padding: 4px 8px;
                                      text-align: center; min-width: 52px; }
table.heatmap th { background: #e8ecf0; font-weight: 600; }
.hm-label { background: #e8ecf0; font-weight: 600; text-align: left !important; }
.hm-na    { background: #f0f0f0; color: #aaa; }
.hm-cell  { cursor: default; font-size: 11px; }

/* Controls */
.controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
             margin-bottom: 8px; }
.controls input, .controls select {
    border: 1px solid #bbb; border-radius: 3px; padding: 4px 8px;
    font-size: 13px; background: #fff; }
.controls input[type=text] { width: 200px; }
.controls input[type=range] { width: 110px; }
.controls label { font-size: 13px; color: #555; }
.count-info { font-size: 12px; color: #777; margin-left: auto; }

/* Generic sortable table */
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

/* Index overview table */
table.idx { border-collapse: collapse; background: #fff;
             border: 1px solid #dde; border-radius: 4px; width: 100%; }
table.idx th { background: #2c3e50; color: #fff; padding: 6px 12px; text-align: left; }
table.idx td { padding: 6px 12px; border-bottom: 1px solid #eee; font-size: 13px; }
table.idx tr:hover td { background: #f0f4fa; }

.feat { font-family: monospace; font-size: 11px; }
.src-badge { display: inline-block; padding: 1px 6px; border-radius: 10px;
              font-size: 11px; font-weight: 600; }
.src-LST   { background: #e3f2fd; color: #0d47a1; }
.src-DHM   { background: #e8f5e9; color: #1b5e20; }
.src-Trees { background: #f3e5f5; color: #4a148c; }
.src-UA    { background: #fff3e0; color: #e65100; }
.src-WIS   { background: #fce4ec; color: #880e4f; }
.src-other { background: #eceff1; color: #37474f; }
.r-cell    { font-weight: 600; }
.sig       { font-size: 10px; color: #c62828; margin-left: 1px; }
.tag { display: inline-block; padding: 1px 5px; border-radius: 3px;
        font-size: 11px; background: #e8ecf0; color: #444; margin: 1px; }
small { font-size: 11px; }

/* Section tabs on stability page */
.tab-bar { display: flex; gap: 4px; margin-bottom: 0; border-bottom: 2px solid #ccc; }
.tab-btn { padding: 6px 16px; background: #e8ecf0; border: 1px solid #ccc;
           border-bottom: none; border-radius: 4px 4px 0 0; cursor: pointer;
           font-size: 13px; color: #444; }
.tab-btn.active { background: #fff; color: #1565c0; font-weight: 600;
                   border-bottom: 2px solid #fff; margin-bottom: -2px; }
.tab-panel { display: none; background: #fff; border: 1px solid #ccc;
             border-top: none; padding: 16px; border-radius: 0 0 4px 4px; }
.tab-panel.active { display: block; }
"""

_SORT_JS = """
function makeSortable(tableId) {
  var sortCol = -1, sortAsc = true;
  var table = document.getElementById(tableId);
  if (!table) return;
  var ths = table.querySelectorAll('thead th');
  ths.forEach(function(th, i) {
    th.addEventListener('click', function() {
      if (sortCol === i) { sortAsc = !sortAsc; }
      else { sortCol = i; sortAsc = true; }
      ths.forEach(function(t, j) {
        t.className = j === i ? (sortAsc ? 'sorted-asc' : 'sorted-desc') : '';
      });
      var tbody = table.querySelector('tbody');
      var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
      rows.sort(function(a, b) {
        var av = a.cells[i].dataset.v !== undefined
                  ? a.cells[i].dataset.v : a.cells[i].textContent.trim();
        var bv = b.cells[i].dataset.v !== undefined
                  ? b.cells[i].dataset.v : b.cells[i].textContent.trim();
        var an = parseFloat(av), bn = parseFloat(bv);
        var cmp = (!isNaN(an) && !isNaN(bn)) ? (an - bn) : av.localeCompare(bv);
        return sortAsc ? cmp : -cmp;
      });
      rows.forEach(function(r) { tbody.appendChild(r); });
      _updateCount(tableId);
    });
  });
}

function _updateCount(tableId) {
  var countId = tableId + '-count';
  var el = document.getElementById(countId);
  if (!el) return;
  var all = document.querySelectorAll('#' + tableId + ' tbody tr').length;
  var vis = document.querySelectorAll('#' + tableId + ' tbody tr:not(.hidden)').length;
  el.textContent = 'Showing ' + vis + ' of ' + all;
}

function makeFilterable(tableId, filterId, srcAId, srcBId) {
  function apply() {
    var text = document.getElementById(filterId) ?
               document.getElementById(filterId).value.toLowerCase() : '';
    var srcA = srcAId && document.getElementById(srcAId) ?
               document.getElementById(srcAId).value : '';
    var srcB = srcBId && document.getElementById(srcBId) ?
               document.getElementById(srcBId).value : '';
    var rows = document.querySelectorAll('#' + tableId + ' tbody tr');
    rows.forEach(function(r) {
      var fa = r.cells[0] ? r.cells[0].textContent.toLowerCase() : '';
      var sa = r.cells[1] ? r.cells[1].textContent.trim() : '';
      var fb = r.cells[2] ? r.cells[2].textContent.toLowerCase() : '';
      var sb = r.cells[3] ? r.cells[3].textContent.trim() : '';
      var show = (
        (!text || fa.includes(text) || fb.includes(text)) &&
        (!srcA || sa === srcA) &&
        (!srcB || sb === srcB)
      );
      r.classList.toggle('hidden', !show);
    });
    _updateCount(tableId);
  }
  [filterId, srcAId, srcBId].forEach(function(id) {
    if (!id) return;
    var el = document.getElementById(id);
    if (el) { el.addEventListener('input', apply); el.addEventListener('change', apply); }
  });
  _updateCount(tableId);
}

function makeStabFilterable(tableId, minRId, maxCVId, srcAId, srcBId, filterId) {
  function apply() {
    var text   = filterId && document.getElementById(filterId) ?
                 document.getElementById(filterId).value.toLowerCase() : '';
    var srcA   = srcAId && document.getElementById(srcAId) ?
                 document.getElementById(srcAId).value : '';
    var srcB   = srcBId && document.getElementById(srcBId) ?
                 document.getElementById(srcBId).value : '';
    var minR   = minRId ? parseFloat(document.getElementById(minRId).value) || 0 : 0;
    var maxCV  = maxCVId ? parseFloat(document.getElementById(maxCVId).value) : 999;
    if (isNaN(maxCV)) maxCV = 999;
    if (minRId)  document.getElementById(minRId  + '-val').textContent = minR.toFixed(2);
    if (maxCVId) document.getElementById(maxCVId + '-val').textContent = maxCV.toFixed(2);
    var rows = document.querySelectorAll('#' + tableId + ' tbody tr');
    rows.forEach(function(r) {
      var fa = r.cells[0] ? r.cells[0].textContent.toLowerCase() : '';
      var sa = r.cells[1] ? r.cells[1].textContent.trim() : '';
      var fb = r.cells[2] ? r.cells[2].textContent.toLowerCase() : '';
      var sb = r.cells[3] ? r.cells[3].textContent.trim() : '';
      var rv = parseFloat((r.cells[5] || {dataset:{}}).dataset.v || 0);
      var cv = parseFloat((r.cells[6] || {dataset:{v:999}}).dataset.v || 999);
      var show = (
        (!text || fa.includes(text) || fb.includes(text)) &&
        (!srcA || sa === srcA) &&
        (!srcB || sb === srcB) &&
        rv >= minR && cv <= maxCV
      );
      r.classList.toggle('hidden', !show);
    });
    _updateCount(tableId);
  }
  [filterId, srcAId, srcBId, minRId, maxCVId].forEach(function(id) {
    if (!id) return;
    var el = document.getElementById(id);
    if (el) { el.addEventListener('input', apply); el.addEventListener('change', apply); }
  });
  _updateCount(tableId);
}

function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(function(btn) {
    btn.addEventListener('click', function() {
      var group = btn.dataset.group;
      document.querySelectorAll('.tab-btn[data-group="' + group + '"]').forEach(
        function(b) { b.classList.remove('active'); });
      document.querySelectorAll('.tab-panel[data-group="' + group + '"]').forEach(
        function(p) { p.classList.remove('active'); });
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab).classList.add('active');
    });
  });
}

window.addEventListener('DOMContentLoaded', function() {
  initTabs();
});
"""


# ── Page shell ──────────────────────────────────────────────────────────────────

def _shell(title: str, body: str, extra_js: str = "") -> str:
    return (
        "<!DOCTYPE html><html lang='en'><head>"
        "<meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{title}</title>"
        f"<style>{_CSS}</style>"
        f"<script>{_SORT_JS}{extra_js}</script>"
        f"</head><body>{body}</body></html>"
    )


# ── Heatmap helpers ────────────────────────────────────────────────────────────

def _build_heatmap_data(
    pairs: List[dict],
    metric: str,
    sources: List[str],
) -> Dict[Tuple, float]:
    acc: Dict[Tuple, List[float]] = defaultdict(list)
    for p in pairs:
        v = p.get(metric)
        if v is None:
            continue
        key = tuple(sorted([p["source_a"], p["source_b"]]))
        acc[key].append(abs(v) if "r" in metric else float(v))
    return {k: sum(vs) / len(vs) for k, vs in acc.items()}


def _heatmap_table(
    sources: List[str],
    data: Dict[Tuple, float],
    colour_fn: Callable[[float], str],
    decimals: int = 3,
    counts: Optional[Dict[Tuple, int]] = None,
) -> str:
    header = "".join(f"<th>{s}</th>" for s in sources)
    rows_html = []
    for sa in sources:
        cells = [f"<td class='hm-label'>{sa}</td>"]
        for sb in sources:
            key = tuple(sorted([sa, sb]))
            if sa == sb:
                cells.append("<td class='hm-na'>—</td>")
            elif key in data:
                v   = data[key]
                bg  = colour_fn(v)
                n   = counts[key] if counts and key in counts else ""
                txt = f"{v:.{decimals}f}"
                tip = f"mean={txt} n={n}"
                cells.append(
                    f"<td class='hm-cell' style='background:{bg}' title='{tip}'>"
                    f"{txt}</td>"
                )
            else:
                cells.append("<td class='hm-na'>n/a</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<table class='heatmap'><thead><tr><th></th>{header}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
    )


def _source_opts(sources: List[str]) -> str:
    return "<option value=''>All</option>" + "".join(
        f"<option>{s}</option>" for s in sources
    )


# ── Cross-corr per-dataset page (from render_ds_reports.py) ───────────────────

def _cc_pairs_rows(pairs: List[dict], max_mi: float) -> str:
    rows = []
    for p in pairs:
        fa, sa = p["feature_a"], p["source_a"]
        fb, sb = p["feature_b"], p["source_b"]
        pr, pp = p.get("pearson_r"),  p.get("pearson_p")
        sr, sp = p.get("spearman_r"), p.get("spearman_p")
        mi     = p.get("mi", 0.0) or 0.0
        nv     = p.get("n_valid", "")

        def rcell(v, pv):
            if v is None:
                return "<td>—</td><td>—</td>"
            bg = _r_col(abs(v))
            return (
                f"<td class='r-cell' style='background:{bg}' data-v='{v:.6f}'>"
                f"{_fmt_r(v)}<span class='sig'>{_sig(pv)}</span></td>"
                f"<td data-v='{pv if pv is not None else 1:.6f}'>{_fmt_p(pv)}</td>"
            )

        mi_bg = _mi_col(mi, max_mi)
        rows.append(
            f"<tr>"
            f"<td class='feat'>{fa}</td>"
            f"<td><span class='src-badge src-{sa}'>{sa}</span></td>"
            f"<td class='feat'>{fb}</td>"
            f"<td><span class='src-badge src-{sb}'>{sb}</span></td>"
            f"<td data-v='{nv}'>{nv:,}</td>"
            + rcell(pr, pp) + rcell(sr, sp)
            + f"<td class='r-cell' style='background:{mi_bg}' data-v='{mi:.6f}'>"
              f"{mi:.4f}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def render_cc_dataset_page(name: str, data: dict, out_dir: Path) -> None:
    pairs = data.get("pairs", [])
    if not pairs:
        note  = data.get("note", "no pairs computed")
        html  = _shell(
            f"Cross-corr: {name}",
            f"<div class='nav-bar'><a href='index.html'>&larr; Index</a></div>"
            f"<h1>{name}</h1><p style='color:#888;margin-top:16px'>{note}</p>",
        )
        (out_dir / f"{name}.html").write_text(html, encoding="utf-8")
        return

    sources = sorted(
        {p["source_a"] for p in pairs} | {p["source_b"] for p in pairs},
        key=lambda s: _SOURCE_ORDER.index(s) if s in _SOURCE_ORDER else 99,
    )
    max_mi   = max((p.get("mi") or 0) for p in pairs) or 1.0

    hm_p  = _build_heatmap_data(pairs, "pearson_r",  sources)
    hm_s  = _build_heatmap_data(pairs, "spearman_r", sources)
    hm_mi = _build_heatmap_data(pairs, "mi",         sources)
    mi_max = max(hm_mi.values()) if hm_mi else 1.0

    heatmaps = (
        "<div class='heatmaps'>"
        "<div class='heatmap-block'><h3>Mean |Pearson r|</h3>"
        + _heatmap_table(sources, hm_p,  _r_col) +
        "</div><div class='heatmap-block'><h3>Mean |Spearman r|</h3>"
        + _heatmap_table(sources, hm_s,  _r_col) +
        "</div><div class='heatmap-block'><h3>Mean MI</h3>"
        + _heatmap_table(sources, hm_mi, lambda v: _mi_col(v, mi_max)) +
        "</div></div>"
    )

    meta = (
        "<div class='meta-box'>"
        f"<div><span>Rows total </span><strong>{data.get('n_rows_total','?'):,}</strong></div>"
        f"<div><span>Rows analysed </span><strong>{data.get('n_rows_analysed','?'):,}</strong></div>"
        f"<div><span>Pairs </span><strong>{len(pairs):,}</strong></div>"
        f"<div><span>Same-source skipped </span><strong>{data.get('skipped_same_source',0):,}</strong></div>"
        "</div>"
    )

    opts = _source_opts(sources)
    controls = (
        "<div class='controls'>"
        "<label>Filter: <input id='cc-filter' type='text' placeholder='feature name...'></label>"
        f"<label>Source A: <select id='cc-srcA'>{opts}</select></label>"
        f"<label>Source B: <select id='cc-srcB'>{opts}</select></label>"
        "<span class='count-info' id='cc-table-count'></span>"
        "</div>"
    )

    table = (
        "<div class='table-wrap'>"
        "<table class='pairs' id='cc-table'><thead><tr>"
        "<th>Feature A</th><th>Src A</th><th>Feature B</th><th>Src B</th>"
        "<th>N valid</th><th>Pearson r</th><th>p</th>"
        "<th>Spearman r</th><th>p</th><th>MI</th>"
        "</tr></thead>"
        f"<tbody>{_cc_pairs_rows(pairs, max_mi)}</tbody>"
        "</table></div>"
    )

    body = (
        "<div class='nav-bar'><a href='index.html'>&larr; Index</a></div>"
        f"<h1>Cross-correlation: {name}</h1>"
        + meta + "<h2>Source-pair summary</h2>" + heatmaps
        + "<h2>All pairs <small style='font-weight:normal;color:#777'>"
          "(click headers to sort)</small></h2>"
        + controls + table
    )

    extra_js = (
        "window.addEventListener('DOMContentLoaded', function(){"
        "  makeSortable('cc-table');"
        "  makeFilterable('cc-table','cc-filter','cc-srcA','cc-srcB');"
        "});"
    )
    (out_dir / f"{name}.html").write_text(
        _shell(f"Cross-corr: {name}", body, extra_js), encoding="utf-8"
    )


# ── Simple-corr page ───────────────────────────────────────────────────────────

def render_simple_corr_page(results: dict, out_dir: Path) -> None:
    """
    Renders simple_corr.html: one row per greening feature, one column-pair
    (Pearson r | slope K/unit) per dataset, plus a trailing stability column.
    Mirrors the partial_corr layout so the two pages are easy to compare.
    """
    datasets = [(k, v) for k, v in results.items() if k != "_meta"]

    feat_lookup: dict = {}   # feat -> {dataset_name -> row}
    all_features: dict = {}  # feat -> {"source": ...}
    for name, ds_data in datasets:
        for row in ds_data.get("results", []):
            f = row["feature"]
            all_features.setdefault(f, {})["source"] = row.get("source", "other")
            feat_lookup.setdefault(f, {})[name] = row

    ds_names = [n for n, _ in datasets]

    # Split datasets into full (≥ threshold rows) and sparse.
    # Stability is computed only on full datasets; sparse datasets are shown
    # in the per-dataset columns but excluded from the stability summary.
    full_ds_names   = []
    sparse_ds_names = []
    for name, ds_data in datasets:
        n_rows = ds_data.get("_meta", {}).get("dataset_stats", {}).get("n_rows") or 0
        (full_ds_names if n_rows >= _FULL_ROW_THRESH else sparse_ds_names).append(name)

    # Pre-compute per-feature stability stats across full datasets only.
    # Variance on |r| (abs_mode) so sign flips don't inflate spread; sign
    # consistency tracked separately. NaN pearson_r values (constant-input
    # features) are excluded via math.isfinite.
    def _feat_stats(feat: str) -> dict:
        vals = [
            feat_lookup.get(feat, {}).get(n, {}).get("pearson_r")
            for n in full_ds_names
        ]
        vals = [v for v in vals if v is not None and math.isfinite(v)]
        if not vals:
            return {}
        n         = len(vals)
        mu_signed = sum(vals) / n
        abs_vals  = [abs(v) for v in vals]
        mu_abs    = sum(abs_vals) / n
        var       = sum((v - mu_abs) ** 2 for v in abs_vals) / max(n - 1, 1)
        sd        = math.sqrt(var)
        cv        = sd / (mu_abs + 1e-6)
        stab      = max(0.0, 1.0 - cv)
        pos       = sum(1 for v in vals if v > 0)
        sign_pct  = max(pos, n - pos) / n
        return {
            "mean_r": mu_signed, "mean_abs_r": mu_abs,
            "std_r": sd, "cv": cv, "stability": stab,
            "sign_pct": sign_pct, "n": n,
        }

    feat_stats = {f: _feat_stats(f) for f in all_features}

    # Sort by descending mean |pearson_r| across full datasets
    sorted_features = sorted(
        all_features.keys(),
        key=lambda f: feat_stats[f].get("mean_abs_r", 0),
        reverse=True,
    )

    stab_note_ds = (
        f"Stability computed from {len(full_ds_names)} full dataset(s) "
        f"(&ge;{_FULL_ROW_THRESH:,} rows): {', '.join(full_ds_names)}."
        + (
            f" Excluded from stability ({len(sparse_ds_names)} sparse): "
            f"{', '.join(sparse_ds_names)}."
            if sparse_ds_names else ""
        )
    )

    # Two sub-columns per dataset + trailing stability group
    thead = (
        "<tr><th rowspan='2'>Feature</th><th rowspan='2'>Src</th>"
        + "".join(f"<th colspan='2'>{n}</th>" for n in ds_names)
        + f"<th colspan='3'>Stability ({len(full_ds_names)} full datasets)</th>"
        + "</tr>"
        "<tr>"
        + "".join("<th>Pearson r</th><th>slope K/unit</th>" for _ in ds_names)
        + "<th>mean r &plusmn; std</th><th>sign%</th><th>stability</th>"
        + "</tr>"
    )

    def _slope_col(v: Optional[float]) -> str:
        if v is None:
            return "#f0f0f0"
        t = min(1.0, abs(v) / 2.0)
        if v < 0:
            return _hex(_lerp(_WHITE, _GREEN, t))
        return _hex(_lerp(_WHITE, _ORANGE, t))

    tbody_rows = []
    for feat in sorted_features:
        src   = all_features[feat].get("source", "other")
        cells = [
            f"<td class='feat'>{feat}</td>"
            f"<td><span class='src-badge src-{src}'>{src}</span></td>"
        ]
        for name in ds_names:
            row = feat_lookup.get(feat, {}).get(name)
            if row is None:
                cells.append("<td>—</td><td>—</td>")
                continue
            pr = row.get("pearson_r")
            pp = row.get("pearson_p")
            sl = row.get("slope_k_per_unit")

            r_cell = (
                f"<td style='background:{_r_col(abs(pr))}' data-v='{pr:.6f}'>"
                f"{_fmt_r(pr)}<span class='sig'>{_sig(pp)}</span></td>"
                if pr is not None and math.isfinite(pr) else "<td>—</td>"
            )
            sl_cell = (
                f"<td style='background:{_slope_col(sl)}' data-v='{sl:.6f}'>"
                f"{sl:+.3f}</td>"
                if sl is not None else "<td>—</td>"
            )
            cells.append(r_cell + sl_cell)

        # Stability columns
        s = feat_stats.get(feat, {})
        if s:
            mu   = s["mean_r"];    sd   = s["std_r"]
            stab = s["stability"]; sp   = s["sign_pct"]
            stab_bg = _stab_col(stab)
            cells.append(
                f"<td style='background:{_r_col(s['mean_abs_r'])}' "
                f"data-v='{mu:.6f}'>"
                f"<strong>{mu:+.3f}</strong>"
                f"<span style='color:#666;font-size:11px'> &plusmn;{sd:.3f}</span></td>"
                f"<td>{sp*100:.0f}%</td>"
                f"<td style='background:{stab_bg};font-weight:700' "
                f"data-v='{stab:.4f}'>{stab:.3f}</td>"
            )
        else:
            cells.append("<td>—</td><td>—</td><td>—</td>")

        tbody_rows.append("<tr>" + "".join(cells) + "</tr>")

    table = (
        "<div class='table-wrap'>"
        "<table class='pairs' id='sc-table'><thead>"
        + thead
        + f"</thead><tbody>{''.join(tbody_rows)}</tbody></table></div>"
    )

    meta_parts = []
    for name, ds_data in datasets:
        m  = ds_data.get("_meta", {})
        ds = m.get("dataset_stats", {})
        n  = ds.get("n_rows") or m.get("n_rows", "?")
        meta_parts.append(
            f"<div><span>{name}: </span>"
            f"<strong>{n:,} rows</strong></div>"
        )
    meta = "<div class='meta-box'>" + "".join(meta_parts) + "</div>"

    legend = (
        "<div class='legend'>"
        "<div><span class='legend-swatch' "
        "style='background:linear-gradient(to right,#fff,#ffa03c,#b41e0a)'></span>"
        " |Pearson r|: low &rarr; high</div>"
        "<div><span class='legend-swatch' "
        "style='background:linear-gradient(to right,#1b8c28,#fff,#ffa03c)'></span>"
        " slope: cooling (negative) &larr; 0 &rarr; warming (positive)</div>"
        "<div><span class='legend-swatch' "
        "style='background:linear-gradient(to right,#b41e0a,#1b8c28)'></span>"
        " stability: low &rarr; high (stability = 1 &minus; CV)</div>"
        "</div>"
    )

    note = (
        "<p style='font-size:12px;color:#666;margin:6px 0 4px'>"
        "<strong>slope K/unit</strong> = Pearson r &times; (&sigma;<sub>temp</sub> / "
        "&sigma;<sub>feature</sub>): expected Kelvin change per 1-unit increase in the feature. "
        "For UA fraction features (range 0&ndash;1) multiply by 0.1 to get K per 10 pp. "
        "<strong>Stability</strong> = 1 &minus; CV of |r| across full datasets; "
        "high stability means the magnitude is consistent across data slices. "
        "No confounders removed &mdash; compare with "
        "<a href='partial_corr.html'>partial_corr.html</a> "
        "for the confounder-adjusted view.</p>"
        f"<p style='font-size:11px;color:#888;margin:0 0 12px'>{stab_note_ds}</p>"
    )

    controls = (
        "<div class='controls'>"
        "<label>Filter: <input id='sc-filter' type='text' placeholder='feature name...'></label>"
        "<span class='count-info' id='sc-table-count'></span>"
        "</div>"
    )

    body = (
        "<div class='nav-bar'><a href='index.html'>&larr; Index</a></div>"
        "<h1>Simple correlations with temperature</h1>"
        "<p style='font-size:13px;color:#555;margin:4px 0 8px'>"
        "Pearson r, Spearman &rho;, and OLS slope (K/unit) for each greening feature "
        "vs temperature.  No confounder removal &mdash; raw associations.</p>"
        + meta + note + legend
        + "<h2>Pearson r and slope per feature per dataset "
          "<small style='font-weight:normal;color:#777'>(click headers to sort)</small></h2>"
        + controls + table
    )

    extra_js = (
        "window.addEventListener('DOMContentLoaded', function(){"
        "  makeSortable('sc-table');"
        "  (function(){"
        "    document.getElementById('sc-filter').addEventListener('input', function(){"
        "      var text = this.value.toLowerCase();"
        "      document.querySelectorAll('#sc-table tbody tr').forEach(function(r){"
        "        r.classList.toggle('hidden', text && "
        "          !r.cells[0].textContent.toLowerCase().includes(text));"
        "      });"
        "      _updateCount('sc-table');"
        "    });"
        "    _updateCount('sc-table');"
        "  })();"
        "});"
    )

    (out_dir / "simple_corr.html").write_text(
        _shell("Simple correlations", body, extra_js), encoding="utf-8"
    )


# ── Partial-corr page ──────────────────────────────────────────────────────────

def render_partial_corr_page(results: dict, out_dir: Path) -> None:
    datasets = [(k, v) for k, v in results.items() if k != "_meta"]

    # Detect greening_only mode from first dataset meta
    greening_only = any(
        ds_data.get("_meta", {}).get("mode") == "greening_only"
        for _, ds_data in datasets
    )

    # Build feature index
    feat_lookup: dict[str, dict[str, dict]] = {}
    all_features: dict[str, dict] = {}
    for name, ds_data in datasets:
        for row in ds_data.get("results", []):
            f = row["feature"]
            all_features.setdefault(f, {})["source"] = row.get("source", "other")
            feat_lookup.setdefault(f, {})[name] = row

    sorted_features = sorted(
        all_features.keys(),
        key=lambda f: (all_features[f].get("source", "other"), f),
    )
    ds_names = [n for n, _ in datasets]

    # Columns: feature, source, then per dataset: r_adj (or r), CF
    def _ds_header(n):
        if greening_only:
            return f"<th colspan='2'>{n}</th>"
        return f"<th>{n}</th>"

    def _ds_subheader(n):
        if greening_only:
            return "<th>adj r</th><th>CF</th>"
        return "<th>r</th>"

    thead = (
        "<tr><th rowspan='2'>Feature</th><th rowspan='2'>Src</th>"
        + "".join(_ds_header(n) for n in ds_names) + "</tr>"
        "<tr>" + "".join(_ds_subheader(n) for n in ds_names) + "</tr>"
    ) if greening_only else (
        "<tr><th>Feature</th><th>Src</th>"
        + "".join(f"<th>{n}</th>" for n in ds_names) + "</tr>"
    )

    tbody_rows = []
    for feat in sorted_features:
        src = all_features[feat].get("source", "other")
        r_key = "partial_r_adj" if greening_only else "partial_r"
        vals = [feat_lookup.get(feat, {}).get(n, {}).get(r_key) for n in ds_names]
        mean_abs = sum(abs(v) for v in vals if v is not None)
        n_v      = sum(1 for v in vals if v is not None)
        mean_abs_v = mean_abs / n_v if n_v else 0.0

        cells = [
            f"<td class='feat' data-v='{feat}'>{feat}</td>"
            f"<td><span class='src-badge src-{src}'>{src}</span></td>"
        ]
        for name in ds_names:
            row = feat_lookup.get(feat, {}).get(name)
            if row is None:
                cells.append("<td>—</td>" + ("<td>—</td>" if greening_only else ""))
                continue
            v  = row.get(r_key)
            pv = row.get("partial_p_adj" if greening_only else "partial_p")
            if v is None:
                cells.append("<td>—</td>" + ("<td>—</td>" if greening_only else ""))
            else:
                bg = _r_col(abs(v))
                cell = (
                    f"<td style='background:{bg}' data-v='{v:.6f}'>"
                    f"{_fmt_r(v)}<span class='sig'>{_sig(pv)}</span></td>"
                )
                if greening_only:
                    cf  = row.get("confounding_fraction")
                    cf_bg = _cv_col(abs(cf)) if cf is not None else "#eee"
                    cf_txt = f"{cf*100:+.0f}%" if cf is not None else "—"
                    cell += (
                        f"<td style='background:{cf_bg}' "
                        f"title='confounding fraction' data-v='{cf or 0:.4f}'>"
                        f"{cf_txt}</td>"
                    )
                cells.append(cell)
        tbody_rows.append(
            f"<tr data-mean='{mean_abs_v:.6f}'>{''.join(cells)}</tr>"
        )

    table = (
        "<div class='table-wrap'>"
        "<table class='pairs' id='pc-table'><thead>"
        + thead +
        f"</thead><tbody>{''.join(tbody_rows)}</tbody></table></div>"
    )

    # Sample morphology confounders from first dataset
    first_meta   = datasets[0][1].get("_meta", {}) if datasets else {}
    morph_list   = first_meta.get("morphology_confounders", [])
    morph_note   = (
        f"<p style='font-size:12px;color:#555;margin-bottom:8px'>"
        f"<strong>Morphology confounders added to Z:</strong> "
        f"{', '.join(morph_list[:8]) or '—'}"
        f"{'…' if len(morph_list) > 8 else ''}</p>"
    ) if greening_only else ""

    meta_parts = []
    for name, ds_data in datasets:
        m = ds_data.get("_meta", {})
        meta_parts.append(
            f"<div><span>{name}: </span>"
            f"<strong>{m.get('total_rows_streamed', '?'):,} rows</strong></div>"
        )
    meta = "<div class='meta-box'>" + "".join(meta_parts) + "</div>"

    legend = (
        "<div class='legend'>"
        "<div><span class='legend-swatch' "
        "style='background:linear-gradient(to right,#fff,#ffa03c,#b41e0a)'></span>"
        " |adj partial r|: low &rarr; high</div>"
        + (
            "<div><span class='legend-swatch' "
            "style='background:linear-gradient(to right,#1b8c28,#ffdc32,#b41e0a)'></span>"
            " CF: 0% (no confounding) &rarr; high (morphology explains effect)</div>"
            if greening_only else ""
        )
        + "</div>"
    )

    mode_label = (
        "Greening features only — adjusted for morphology (DHM + built UA)"
        if greening_only else "All features — temporal confounders only"
    )

    controls = (
        "<div class='controls'>"
        "<label>Filter: <input id='pc-filter' type='text' placeholder='feature name...'></label>"
        "<span class='count-info' id='pc-table-count'></span>"
        "</div>"
    )

    body = (
        "<div class='nav-bar'><a href='index.html'>&larr; Index</a></div>"
        "<h1>Partial correlations with temperature (FWL)</h1>"
        f"<p style='font-size:13px;color:#555;margin:4px 0 12px'>{mode_label}</p>"
        + meta + morph_note + legend
        + "<h2>Adjusted partial r per greening feature per dataset "
          "<small style='font-weight:normal;color:#777'>"
          "(click headers to sort; CF = confounding fraction)</small></h2>"
        + controls + table
    )

    extra_js = (
        "window.addEventListener('DOMContentLoaded', function(){"
        "  makeSortable('pc-table');"
        "  (function(){"
        "    var text = '';"
        "    document.getElementById('pc-filter').addEventListener('input', function(){"
        "      text = this.value.toLowerCase();"
        "      document.querySelectorAll('#pc-table tbody tr').forEach(function(r){"
        "        r.classList.toggle('hidden', text && "
        "          !r.cells[0].textContent.toLowerCase().includes(text));"
        "      });"
        "      _updateCount('pc-table');"
        "    });"
        "    _updateCount('pc-table');"
        "  })();"
        "});"
    )

    (out_dir / "partial_corr.html").write_text(
        _shell("Partial correlations", body, extra_js), encoding="utf-8"
    )


# ── Moran's I page ─────────────────────────────────────────────────────────────

def render_morans_i_page(results: dict, out_dir: Path) -> None:
    datasets = [(k, v) for k, v in results.items() if k != "_meta"]

    def _model_row(name: str, model_label: str, m: dict, delta_i=None) -> str:
        if "error" in m:
            return (
                f"<tr><td>{name}</td><td>{model_label}</td>"
                f"<td colspan='7' style='color:#888'>{m.get('error','')}</td></tr>"
            )
        I    = m.get("morans_I")
        E_I  = m.get("morans_E_I")
        p    = m.get("morans_p")
        r2   = m.get("global_r2")
        nt   = m.get("n_tiles", "?")
        nobs = m.get("n_obs",  "?")
        interp = m.get("interpretation", "")
        I_bg  = _moran_col(I or 0.0)
        r2_bg = _r2_col(r2 or 0.0)
        i_icon = "&#9989;" if "spatially random" in interp else "&#9888;"
        # ΔI cell: shown only on the full-model row
        if delta_i is not None:
            di_bg  = _stab_col(min(1.0, max(0.0, delta_i * 5)))
            di_txt = f"{delta_i:+.4f}"
            di_cell = (
                f"<td data-v='{delta_i:.6f}' style='background:{di_bg};font-weight:700' "
                f"title='I(morphology) − I(full): greening reduces spatial clustering by this amount'>"
                f"{di_txt}</td>"
            )
        else:
            di_cell = "<td style='color:#ccc'>—</td>"
        nobs_fmt = f"{nobs:,}" if isinstance(nobs, int) else str(nobs)
        return (
            f"<tr>"
            f"<td>{name}</td><td>{model_label}</td>"
            f"<td data-v='{nobs if isinstance(nobs,int) else 0}'>{nobs_fmt}</td>"
            f"<td>{nt}</td>"
            f"<td data-v='{I or 0:.6f}' style='background:{I_bg};font-weight:700'>"
            f"{_fmt(I)}</td>"
            f"<td data-v='{E_I or 0:.6f}'>{_fmt(E_I)}</td>"
            f"<td data-v='{p or 1:.6f}'>{_fmt_p(p)}</td>"
            f"<td data-v='{r2 or 0:.6f}' style='background:{r2_bg}'>{_fmt(r2)}</td>"
            + di_cell +
            f"<td><small>{i_icon} {interp[:70]}</small></td>"
            f"</tr>"
        )

    rows = []
    for name, ds_data in datasets:
        delta_i = ds_data.get("delta_i_greening")
        for model_key, label in [
            ("null_model",       "null"),
            ("morphology_model", "morphology"),
            ("full_model",       "full"),
        ]:
            m = ds_data.get(model_key, {})
            if not m:
                continue
            di = delta_i if model_key == "full_model" else None
            rows.append(_model_row(name, label, m, delta_i=di))

    table = (
        "<div class='table-wrap'>"
        "<table class='pairs' id='mi-table'><thead><tr>"
        "<th>Dataset</th><th>Model</th><th>N obs</th><th>N tiles</th>"
        "<th>Moran's I</th><th>E[I]</th><th>p-value</th>"
        "<th>Global R²</th><th title='I(morphology)−I(full): unique greening contribution'>&#916;I greening</th>"
        "<th>Interpretation</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )

    legend = (
        "<div class='legend'>"
        "<div><span class='legend-swatch' "
        "style='background:linear-gradient(to right,#fff,#1565c0)'></span>"
        " Moran's I: 0 &rarr; clustered (positive)</div>"
        "<div><span class='legend-swatch' "
        "style='background:linear-gradient(to right,#ffa03c,#1b8c28)'></span>"
        " R²: low &rarr; high</div>"
        "</div>"
    )

    body = (
        "<div class='nav-bar'><a href='index.html'>&larr; Index</a></div>"
        "<h1>Moran's I — spatial autocorrelation of OLS residuals</h1>"
        + legend
        + "<p style='margin:8px 0 16px;font-size:13px;color:#555'>"
          "Positive Moran's I means residuals are spatially clustered — "
          "features do not fully explain spatial temperature variation.</p>"
        + "<h2>Results per dataset "
          "<small style='font-weight:normal;color:#777'>(click headers to sort)</small></h2>"
        + table
    )

    extra_js = (
        "window.addEventListener('DOMContentLoaded', function(){"
        "  makeSortable('mi-table');"
        "});"
    )

    (out_dir / "morans_i.html").write_text(
        _shell("Moran's I", body, extra_js), encoding="utf-8"
    )


# ── Spatial-CV page ────────────────────────────────────────────────────────────

def render_spatial_cv_page(results: dict, out_dir: Path) -> None:
    datasets = [(k, v) for k, v in results.items() if k != "_meta"]

    summary_rows = []
    for name, ds_data in datasets:
        s    = ds_data.get("summary", {})
        rand = ds_data.get("random_fold_cv", {})
        meta = ds_data.get("_meta", {})
        if not s:
            summary_rows.append(
                f"<tr><td>{name}</td>"
                f"<td colspan='9' style='color:#888'>no summary</td></tr>"
            )
            continue
        r2   = s.get("r2_mean")
        r2bg = _r2_col(r2 or 0.0)
        rr2  = rand.get("r2_mean")
        rr2bg = _r2_col(rr2 or 0.0)
        summary_rows.append(
            f"<tr>"
            f"<td>{name}</td>"
            f"<td>{s.get('n_blocks','?')}</td>"
            f"<td data-v='{r2 or 0:.6f}' style='background:{r2bg};font-weight:700'>"
            f"{_fmt(r2)}</td>"
            f"<td>{_fmt(s.get('r2_std'))}</td>"
            f"<td>{_fmt(s.get('r2_q10'))}</td>"
            f"<td>{_fmt(s.get('r2_q90'))}</td>"
            f"<td>{_fmt(s.get('mae_mean'))}</td>"
            f"<td>{_fmt(s.get('rmse_mean'))}</td>"
            f"<td data-v='{rr2 or 0:.6f}' style='background:{rr2bg}'>{_fmt(rr2)}</td>"
            f"<td>{meta.get('block_col','?')}</td>"
            f"</tr>"
        )

    summary_table = (
        "<div class='table-wrap'>"
        "<table class='pairs' id='scv-table'><thead><tr>"
        "<th>Dataset</th><th>N blocks</th>"
        "<th>LOO R² mean</th><th>R² std</th><th>R² Q10</th><th>R² Q90</th>"
        "<th>MAE mean</th><th>RMSE mean</th>"
        "<th>Random-fold R²</th><th>Block col</th>"
        "</tr></thead>"
        f"<tbody>{''.join(summary_rows)}</tbody></table></div>"
    )

    legend = (
        "<div class='legend'>"
        "<div><span class='legend-swatch' "
        "style='background:linear-gradient(to right,#ffa03c,#1b8c28)'></span>"
        " R²: low &rarr; high</div>"
        "</div>"
    )

    body = (
        "<div class='nav-bar'><a href='index.html'>&larr; Index</a></div>"
        "<h1>Spatial leave-one-block-out cross-validation</h1>"
        + legend
        + "<p style='margin:8px 0 16px;font-size:13px;color:#555'>"
          "LOO-block-CV: each spatial block is held out in turn; "
          "the model is re-fitted on the remaining data using sufficient-statistics subtraction. "
          "High LOO R² means the feature set generalises spatially.</p>"
        + "<h2>Summary per dataset "
          "<small style='font-weight:normal;color:#777'>(click headers to sort)</small></h2>"
        + summary_table
    )

    extra_js = (
        "window.addEventListener('DOMContentLoaded', function(){"
        "  makeSortable('scv-table');"
        "});"
    )

    (out_dir / "spatial_cv.html").write_text(
        _shell("Spatial CV", body, extra_js), encoding="utf-8"
    )


# ── Stability aggregation ──────────────────────────────────────────────────────

def _basic_stats(vals: List[float], abs_mode: bool = True) -> dict:
    if not vals:
        return {}
    vs = [abs(v) for v in vals] if abs_mode else vals
    n  = len(vs)
    mu = sum(vs) / n
    var = sum((x - mu) ** 2 for x in vs) / max(n - 1, 1)
    sd = math.sqrt(var)
    cv = sd / (mu + 1e-6)
    return {
        "mean": round(mu, 6), "std": round(sd, 6), "cv": round(cv, 4),
        "min": round(min(vs), 6), "max": round(max(vs), 6), "n": n,
    }


def _sign_pct(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    pos = sum(1 for v in vals if v > 0)
    return round(max(pos, len(vals) - pos) / len(vals), 3)


def _split_datasets_by_rows(results: dict, row_thresh: int = _FULL_ROW_THRESH):
    full, small = [], []
    for k, v in results.items():
        if k == "_meta":
            continue
        rows = v.get("n_rows_total") or v.get("_meta", {}).get("total_rows_streamed", 0) or 0
        (full if rows >= row_thresh else small).append(k)
    return sorted(full), sorted(small)


def aggregate_cross_corr(cc_results: dict):
    full_names, spatial_names = _split_datasets_by_rows(cc_results)

    full_acc:    Dict[Tuple, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    spatial_acc: Dict[Tuple, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    seen:        Dict[Tuple, Dict[str, List]] = defaultdict(lambda: {"full": [], "spatial": []})

    for ds, bucket, acc in (
        [(n, "full",    full_acc)    for n in full_names] +
        [(n, "spatial", spatial_acc) for n in spatial_names]
    ):
        for pair in cc_results[ds].get("pairs", []):
            key = (pair["feature_a"], pair["source_a"], pair["feature_b"], pair["source_b"])
            for m in ("pearson_r", "spearman_r", "mi"):
                v = pair.get(m)
                if v is not None:
                    acc[key][m].append(v)
            seen[key][bucket].append(ds)

    pair_stats = []
    for key in sorted(
        set(full_acc.keys()) | set(spatial_acc.keys()),
        key=lambda k: (k[1], k[3], k[0], k[2]),
    ):
        fa, sa, fb, sb = key
        fv = full_acc[key]
        sv = spatial_acc[key]
        pr_full = fv.get("pearson_r",  [])
        sr_full = fv.get("spearman_r", [])
        mi_full = fv.get("mi",         [])
        pr_spat = sv.get("pearson_r",  [])

        ps = _basic_stats(pr_full)
        ss = _basic_stats(sr_full)
        ms = _basic_stats(mi_full, abs_mode=False)
        stab     = round(max(0.0, 1.0 - ps.get("cv", 1.0)), 4) if ps else None
        sp       = _sign_pct(pr_full)
        full_sign = 1 if sum(pr_full) > 0 else (-1 if pr_full else 0)
        spat_agree = None
        if pr_spat and full_sign:
            same = sum(1 for v in pr_spat if (v > 0) == (full_sign > 0))
            spat_agree = round(same / len(pr_spat), 3)

        pair_stats.append({
            "feature_a": fa, "source_a": sa, "feature_b": fb, "source_b": sb,
            "n_full": len(seen[key]["full"]), "n_spatial": len(seen[key]["spatial"]),
            "pearson": ps, "spearman": ss, "mi": ms,
            "stability": stab, "sign_pct": sp, "spatial_agree": spat_agree,
        })

    return pair_stats, {"full_datasets": full_names, "spatial_datasets": spatial_names}


def aggregate_partial_corr(pc_results: dict) -> List[dict]:
    full_names, _ = _split_datasets_by_rows(pc_results)
    feat_acc: Dict[str, Dict] = defaultdict(lambda: {
        "vals": [], "vals_adj": [], "cfs": [],
        "source": "other", "datasets": [],
    })

    for name in full_names:
        for row in pc_results[name].get("results", []):
            f = row["feature"]
            # Prefer adjusted partial r; fall back to unadjusted for legacy outputs
            v     = row.get("partial_r_adj") or row.get("partial_r")
            v_raw = row.get("partial_r")
            cf    = row.get("confounding_fraction")
            if v is not None:
                feat_acc[f]["vals"].append(v)
                feat_acc[f]["source"]   = row.get("source", "other")
                feat_acc[f]["datasets"].append(name)
            if v_raw is not None:
                feat_acc[f]["vals_adj"].append(v_raw)
            if cf is not None:
                feat_acc[f]["cfs"].append(cf)

    out = []
    for feat, acc in sorted(feat_acc.items()):
        vals = acc["vals"]   # adjusted (or raw if legacy)
        ps   = _basic_stats(vals)
        sp   = _sign_pct(vals)
        cfs  = acc["cfs"]
        mean_cf = round(sum(cfs) / len(cfs), 4) if cfs else None
        out.append({
            "feature":            feat,
            "source":             acc["source"],
            "n_full":             len(acc["datasets"]),
            "stats":              ps,
            "sign_pct":           sp,
            "mean_cf":            mean_cf,
            "stability":          round(max(0.0, 1.0 - ps.get("cv", 1.0)), 4) if ps else None,
        })
    out.sort(key=lambda r: r.get("stability") or 0, reverse=True)
    return out


def aggregate_morans_i(mi_results: dict) -> List[dict]:
    full_names, _ = _split_datasets_by_rows(mi_results)
    model_acc: Dict[str, List[float]] = {"null": [], "full": []}

    rows = []
    for name in full_names:
        ds = mi_results[name]
        for label in ("null", "full"):
            m = ds.get(f"{label}_model", {})
            I = m.get("morans_I")
            if I is not None:
                model_acc[label].append(I)

    out = []
    for label in ("null", "full"):
        vals = model_acc[label]
        ps   = _basic_stats(vals, abs_mode=False)
        out.append({
            "model":    label,
            "n_full":   len(vals),
            "stats":    ps,
            "stability": round(max(0.0, 1.0 - ps.get("cv", 1.0)), 4) if ps else None,
        })
    return out


def aggregate_spatial_cv(scv_results: dict) -> dict:
    full_names, _ = _split_datasets_by_rows(scv_results)
    r2s = []
    for name in full_names:
        s = scv_results[name].get("summary", {})
        r2 = s.get("r2_mean")
        if r2 is not None:
            r2s.append(r2)
    return {"stats": _basic_stats(r2s, abs_mode=False), "n_full": len(r2s), "datasets": full_names}


# ── Stability page ─────────────────────────────────────────────────────────────

def _cc_stability_rows(pair_stats: List[dict]) -> Tuple[str, float]:
    max_mi = max((p["mi"].get("mean") or 0) for p in pair_stats) or 1.0
    rows = []
    for p in pair_stats:
        fa, sa = p["feature_a"], p["source_a"]
        fb, sb = p["feature_b"], p["source_b"]
        pr = p.get("pearson",  {}); sr = p.get("spearman", {}); mi = p.get("mi", {})
        stab     = p.get("stability")
        sign_pct = p.get("sign_pct")
        spat     = p.get("spatial_agree")
        n_full   = p.get("n_full", 0)

        pr_mu = pr.get("mean", 0) or 0
        pr_cv = pr.get("cv",   1) or 1
        mi_mu = mi.get("mean", 0) or 0

        def rpair(s):
            mu = s.get("mean"); sd = s.get("std"); cv = s.get("cv")
            if mu is None:
                return "<td>—</td><td>—</td>"
            return (
                f"<td style='background:{_r_col(mu)}' data-v='{mu:.6f}'>"
                f"<strong>{mu:.3f}</strong>"
                f"<span style='color:#666;font-size:11px'> &plusmn;{sd:.3f}</span></td>"
                f"<td style='background:{_cv_col(cv or 0)}' data-v='{cv:.4f}'>{cv:.3f}</td>"
            )

        stab_bg  = _stab_col(stab) if stab is not None else "#eee"
        spat_html = "—"
        if spat is not None:
            icon = "&#10003;" if spat >= 0.5 else "&#10007;"
            spat_html = f"{icon} {spat*100:.0f}%"

        rows.append(
            f"<tr>"
            f"<td class='feat'>{fa}</td><td><span class='src-badge src-{sa}'>{sa}</span></td>"
            f"<td class='feat'>{fb}</td><td><span class='src-badge src-{sb}'>{sb}</span></td>"
            f"<td>{n_full}</td>"
            + rpair(pr) + rpair(sr)
            + f"<td style='background:{_mi_col(mi_mu, max_mi)}' data-v='{mi_mu:.6f}'>"
              f"<strong>{mi_mu:.3f}</strong>"
              f"<span style='color:#666;font-size:11px'> &plusmn;{mi.get('std',0):.3f}</span></td>"
            f"<td>{sign_pct*100:.0f}%</td>"
            f"<td data-v='{stab if stab is not None else -1:.4f}' "
            f"style='background:{stab_bg};font-weight:700'>"
            f"{'—' if stab is None else f'{stab:.3f}'}</td>"
            f"<td>{spat_html}</td>"
            f"</tr>"
        )
    return "\n".join(rows), max_mi


def _pc_stability_rows(pc_agg: List[dict]) -> str:
    rows = []
    for p in pc_agg:
        feat    = p["feature"]
        src     = p.get("source", "other")
        s       = p.get("stats", {})
        stab    = p.get("stability")
        sp      = p.get("sign_pct")
        mean_cf = p.get("mean_cf")
        n_f     = p.get("n_full", 0)
        mu      = s.get("mean")
        sd      = s.get("std")
        cv      = s.get("cv")
        stab_bg = _stab_col(stab) if stab is not None else "#eee"
        cf_bg   = _cv_col(abs(mean_cf)) if mean_cf is not None else "#eee"
        rows.append(
            f"<tr>"
            f"<td class='feat'>{feat}</td>"
            f"<td><span class='src-badge src-{src}'>{src}</span></td>"
            f"<td>{n_f}</td>"
            f"<td data-v='{mu or 0:.6f}' style='background:{_r_col(mu or 0)}'>"
            f"<strong>{_fmt(mu)}</strong></td>"
            f"<td data-v='{sd or 0:.6f}'>{_fmt(sd)}</td>"
            f"<td data-v='{cv or 1:.4f}' style='background:{_cv_col(cv or 1)}'>{_fmt(cv)}</td>"
            f"<td>{f'{sp*100:.0f}%' if sp is not None else '—'}</td>"
            f"<td data-v='{mean_cf or 0:.4f}' style='background:{cf_bg}'>"
            f"{'—' if mean_cf is None else f'{mean_cf*100:+.0f}%'}</td>"
            f"<td data-v='{stab if stab is not None else -1:.4f}' "
            f"style='background:{stab_bg};font-weight:700'>"
            f"{'—' if stab is None else f'{stab:.3f}'}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def render_stability_page(
    cc_results:  Optional[dict],
    pc_results:  Optional[dict],
    mi_results:  Optional[dict],
    scv_results: Optional[dict],
    out_dir:     Path,
) -> None:
    sources = _SOURCE_ORDER

    # ── Cross-corr section ──
    cc_section = "<p style='color:#888'>cross_corr.json not found.</p>"
    if cc_results:
        pair_stats, agg_meta = aggregate_cross_corr(cc_results)
        cc_rows, max_mi = _cc_stability_rows(pair_stats)

        opts = _source_opts(sources)
        controls = (
            "<div class='controls'>"
            "<label>Filter: <input id='cc-stab-f' type='text' placeholder='feature...'></label>"
            f"<label>Src A: <select id='cc-stab-a'>{opts}</select></label>"
            f"<label>Src B: <select id='cc-stab-b'>{opts}</select></label>"
            "<label>Min |r|: <input id='cc-stab-minr' type='range' min='0' max='1' step='0.05' value='0'>"
            " <span id='cc-stab-minr-val'>0.00</span></label>"
            "<label>Max CV: <input id='cc-stab-maxcv' type='range' min='0' max='3' step='0.1' value='3'>"
            " <span id='cc-stab-maxcv-val'>3.00</span></label>"
            "<span class='count-info' id='cc-stab-table-count'></span>"
            "</div>"
        )
        hm_r    = _build_heatmap_data(
            [{"source_a": p["source_a"], "source_b": p["source_b"],
              "pearson_r": p["pearson"].get("mean") or 0} for p in pair_stats],
            "pearson_r", sources
        )
        hm_stab = {}
        for p in pair_stats:
            key = tuple(sorted([p["source_a"], p["source_b"]]))
            if p.get("stability") is not None:
                hm_stab.setdefault(key, []).append(p["stability"])
        hm_stab = {k: sum(v) / len(v) for k, v in hm_stab.items()}

        heatmaps = (
            "<div class='heatmaps'>"
            "<div class='heatmap-block'><h3>Mean |Pearson r|</h3>"
            + _heatmap_table(sources, hm_r, _r_col) +
            "</div>"
            "<div class='heatmap-block'><h3>Mean stability score</h3>"
            + _heatmap_table(sources, hm_stab, _stab_col) +
            "</div></div>"
        )

        n_strong = sum(
            1 for p in pair_stats
            if (p["pearson"].get("mean") or 0) >= 0.2
            and (p.get("stability") or 0) >= 0.7
        )
        meta = (
            "<div class='meta-box'>"
            f"<div><span>Full datasets </span><strong>{len(agg_meta['full_datasets'])}</strong></div>"
            f"<div><span>Unique pairs </span><strong>{len(pair_stats):,}</strong></div>"
            f"<div><span>Strong+stable (|r|≥0.2, stab≥0.7) </span><strong>{n_strong:,}</strong></div>"
            "</div>"
        )

        table = (
            "<div class='table-wrap'>"
            "<table class='pairs' id='cc-stab-table'><thead><tr>"
            "<th>Feature A</th><th>Src A</th><th>Feature B</th><th>Src B</th><th>N</th>"
            "<th>Mean |Pearson r|</th><th>CV</th>"
            "<th>Mean |Spearman r|</th><th>CV</th>"
            "<th>Mean MI</th><th>Sign%</th><th>Stability</th><th>Spatial agree</th>"
            "</tr></thead>"
            f"<tbody>{cc_rows}</tbody></table></div>"
        )
        cc_section = meta + heatmaps + controls + table

    # ── Partial-corr stability section ──
    pc_section = "<p style='color:#888'>partial_corr.json not found.</p>"
    if pc_results:
        pc_agg = aggregate_partial_corr(pc_results)
        pc_rows = _pc_stability_rows(pc_agg)
        pc_section = (
            "<div class='table-wrap'>"
            "<table class='pairs' id='pc-stab-table'><thead><tr>"
            "<th>Feature</th><th>Source</th><th>N datasets</th>"
            "<th>Mean |partial r|</th><th>Std</th><th>CV</th>"
            "<th>Sign%</th><th>Stability</th>"
            "</tr></thead>"
            f"<tbody>{pc_rows}</tbody></table></div>"
        )

    # ── Moran's I stability section ──
    mi_section = "<p style='color:#888'>morans_i.json not found.</p>"
    if mi_results:
        mi_agg = aggregate_morans_i(mi_results)
        mi_rows = []
        for m in mi_agg:
            s    = m.get("stats", {})
            mu   = s.get("mean")
            stab = m.get("stability")
            stab_bg = _stab_col(stab) if stab is not None else "#eee"
            mi_rows.append(
                f"<tr>"
                f"<td>{m['model']}</td><td>{m['n_full']}</td>"
                f"<td data-v='{mu or 0:.6f}'>{_fmt(mu)}</td>"
                f"<td>{_fmt(s.get('std'))}</td>"
                f"<td>{_fmt(s.get('min'))} – {_fmt(s.get('max'))}</td>"
                f"<td data-v='{stab if stab is not None else -1:.4f}' "
                f"style='background:{stab_bg};font-weight:700'>"
                f"{'—' if stab is None else f'{stab:.3f}'}</td>"
                f"</tr>"
            )
        mi_section = (
            "<div class='table-wrap'>"
            "<table class='pairs' id='mi-stab-table'><thead><tr>"
            "<th>Model</th><th>N datasets</th>"
            "<th>Mean Moran's I</th><th>Std</th><th>Range</th><th>Stability</th>"
            "</tr></thead>"
            f"<tbody>{''.join(mi_rows)}</tbody></table></div>"
            "<p style='font-size:12px;color:#666;margin-top:8px'>"
            "Stability of Moran's I across full datasets. "
            "A stable positive I means spatial autocorrelation consistently remains "
            "in residuals; stable near-zero means features consistently capture "
            "spatial structure.</p>"
        )

    # ── Spatial-CV stability section ──
    scv_section = "<p style='color:#888'>spatial_cv.json not found.</p>"
    if scv_results:
        scv_agg = aggregate_spatial_cv(scv_results)
        s     = scv_agg.get("stats", {})
        mu    = s.get("mean")
        stab  = scv_agg.get("stats", {}).get("cv")
        stab_score = round(max(0.0, 1.0 - (stab or 1.0)), 4) if stab is not None else None
        stab_bg = _stab_col(stab_score) if stab_score is not None else "#eee"
        scv_section = (
            "<div class='meta-box'>"
            f"<div><span>Full datasets </span><strong>{scv_agg['n_full']}</strong></div>"
            f"<div><span>Mean LOO R² </span><strong>{_fmt(mu)}</strong></div>"
            f"<div><span>Std LOO R² </span><strong>{_fmt(s.get('std'))}</strong></div>"
            f"<div><span>Range </span><strong>{_fmt(s.get('min'))} – {_fmt(s.get('max'))}</strong></div>"
            f"<div><span>CV </span><strong>{_fmt(s.get('cv'))}</strong></div>"
            f"<div><span>Stability score </span>"
            f"<strong style='background:{stab_bg};padding:2px 6px;border-radius:3px'>"
            f"{'—' if stab_score is None else f'{stab_score:.3f}'}</strong></div>"
            "</div>"
            "<p style='font-size:12px;color:#666;margin-top:8px'>"
            "Cross-dataset consistency of spatial LOO-CV R². "
            "High stability means the feature set's predictive power is "
            "robust across different data slices.</p>"
        )

    # ── Legend ──
    legend = (
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

    # ── Tab layout ──
    tabs_html = (
        "<div class='tab-bar'>"
        "<button class='tab-btn active' data-group='stab' data-tab='stab-cc'>"
        "Cross-corr pairs</button>"
        "<button class='tab-btn' data-group='stab' data-tab='stab-pc'>"
        "Partial correlations</button>"
        "<button class='tab-btn' data-group='stab' data-tab='stab-mi'>"
        "Moran's I</button>"
        "<button class='tab-btn' data-group='stab' data-tab='stab-scv'>"
        "Spatial CV</button>"
        "</div>"
        "<div id='stab-cc'  class='tab-panel active' data-group='stab'>" + cc_section  + "</div>"
        "<div id='stab-pc'  class='tab-panel'        data-group='stab'>" + pc_section  + "</div>"
        "<div id='stab-mi'  class='tab-panel'        data-group='stab'>" + mi_section  + "</div>"
        "<div id='stab-scv' class='tab-panel'        data-group='stab'>" + scv_section + "</div>"
    )

    body = (
        "<div class='nav-bar'><a href='index.html'>&larr; Index</a></div>"
        "<h1>Cross-dataset stability</h1>"
        + legend
        + "<p style='margin:8px 0 16px;font-size:13px;color:#555'>"
          "Stability = 1 &minus; CV, where CV = std / mean across full datasets "
          "(&ge;10 000 rows).  High stability means the metric is consistent "
          "across different data slices.</p>"
        + tabs_html
    )

    extra_js = (
        "window.addEventListener('DOMContentLoaded', function(){"
        "  makeSortable('cc-stab-table');"
        "  makeStabFilterable('cc-stab-table','cc-stab-minr','cc-stab-maxcv',"
        "    'cc-stab-a','cc-stab-b','cc-stab-f');"
        "  makeSortable('pc-stab-table');"
        "  makeSortable('mi-stab-table');"
        "});"
    )

    (out_dir / "stability.html").write_text(
        _shell("Stability report", body, extra_js), encoding="utf-8"
    )


# ── Guide ─────────────────────────────────────────────────────────────────────

def _guide_html() -> str:
    """Return a self-contained glossary / methodology guide for index.html."""

    _GUIDE_CSS = """
.guide { margin-top: 32px; }
.guide h2 { font-size: 1.1rem; margin: 24px 0 8px; color: #444;
            border-bottom: 1px solid #ccc; padding-bottom: 4px; }
.guide-section { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 4px; }
.guide-card {
    background: #fff; border: 1px solid #dde; border-radius: 5px;
    padding: 12px 16px; flex: 1 1 300px; max-width: 480px;
    font-size: 13px; line-height: 1.55;
}
.guide-card .gterm {
    font-weight: 700; color: #1565c0; font-size: 13px; display: block;
    margin-bottom: 3px;
}
.guide-card .gsub {
    font-size: 11px; color: #888; font-style: italic; margin-left: 6px;
}
.guide-card p { color: #444; margin: 0; }
.guide-card .gformula {
    font-family: monospace; font-size: 11px; background: #f4f6f8;
    border-radius: 3px; padding: 2px 6px; display: inline-block;
    margin: 3px 0; color: #333;
}
.guide-group-label {
    width: 100%; font-size: 11px; font-weight: 700; color: #888;
    text-transform: uppercase; letter-spacing: .05em;
    margin: 8px 0 2px;
}
"""

    def card(term, sub, body, formula=None):
        f = f"<span class='gformula'>{formula}</span><br>" if formula else ""
        return (
            f"<div class='guide-card'>"
            f"<span class='gterm'>{term} <span class='gsub'>{sub}</span></span>"
            f"{f}<p>{body}</p>"
            f"</div>"
        )

    def group(label):
        return f"<div class='guide-group-label'>{label}</div>"

    sections = []

    # ── Simple correlations ──
    sections.append("<h2>Simple correlations &mdash; <code>simple_corr.html</code></h2>")
    sections.append("<div class='guide-section'>")
    sections.append(group("Raw feature–temperature associations: the simplest interpretable numbers"))
    sections.append(card(
        "Pearson r", "linear correlation",
        "Measures the strength and direction of the <em>linear</em> relationship between a "
        "greening feature and temperature. Ranges from &minus;1 (perfect negative linear) "
        "to +1 (perfect positive linear). "
        "A negative r means: locations with more greenery tend to be cooler. "
        "No confounders removed — compare with the adjusted partial r on the partial_corr page.",
        "r = cov(x, y) / (σ_x · σ_y)",
    ))
    sections.append(card(
        "Spearman ρ", "rank-based correlation",
        "Pearson r applied to the ranks of x and y. Captures monotone (not just linear) "
        "relationships and is robust to outliers. "
        "If Pearson r ≈ Spearman ρ, the relationship is roughly linear and the slope "
        "interpretation is clean. A large gap signals a non-linear or outlier-driven association.",
        "ρ = 1 − 6·Σd²/[n(n²−1)]",
    ))
    sections.append(card(
        "p-value", "two-sided test of r = 0",
        "Probability of observing |r| this large (or larger) if the true correlation were zero. "
        "Stars: *** p&lt;0.001 &nbsp; ** p&lt;0.01 &nbsp; * p&lt;0.05. "
        "With large N (millions of rows) even tiny, practically irrelevant r values become "
        "significant — always interpret together with the magnitude of r.",
    ))
    sections.append(card(
        "Slope K/unit", "OLS slope in natural units",
        "Expected change in temperature (Kelvin) per 1-unit increase in the feature. "
        "Derived from Pearson r as: slope = r &times; (&sigma;<sub>temp</sub> / &sigma;<sub>feature</sub>). "
        "This is equivalent to the coefficient in a simple OLS regression of temperature on the feature. "
        "<strong>Planning interpretation:</strong> for UA fraction features (range 0–1), "
        "multiply by 0.1 to get K per 10 percentage-point increase in green fraction. "
        "For tree counts, the slope gives K per additional tree in the buffer radius. "
        "No confounders removed; compare with effect_slope in partial_corr for the "
        "confounder-adjusted version.",
        "slope = r · σ_y / σ_x",
    ))
    sections.append(card(
        "Per-std effect", "slope in standardised units",
        "Expected change in temperature (Kelvin) per 1-standard-deviation increase in the "
        "feature: per_std = r &times; &sigma;<sub>temp</sub>. "
        "Useful for comparing the relative importance of features on different scales "
        "(e.g., NDVI 0–1 vs. tree count 0–200). "
        "Stored in the JSON as <code>slope_k_per_std</code>.",
        "per_std = r · σ_y",
    ))
    sections.append(card(
        "MI", "mutual information (k-NN)",
        "Non-negative measure of statistical dependence; captures non-linear relationships "
        "that r misses. Estimated on a subsample (default 50 000 rows) via the k-nearest-"
        "neighbour method (sklearn). "
        "Zero only if x and y are statistically independent. "
        "Not directly comparable across datasets of different sizes. "
        "Stored per feature in the JSON; visible in the cross_corr pairs table.",
        "MI(x,y) = H(x) + H(y) − H(x,y)",
    ))
    sections.append("</div>")

    # ── Cross-correlations ──
    sections.append("<h2>Cross-correlations &mdash; <code>cross_corr.html</code></h2>")
    sections.append("<div class='guide-section'>")
    sections.append(group("Bivariate association between feature pairs across sources"))
    sections.append(card(
        "Pearson r", "linear correlation",
        "Measures the strength and direction of the linear relationship between two features. "
        "Ranges from &minus;1 (perfect negative) to +1 (perfect positive). "
        "Sensitive to outliers; assumes normally distributed variables. "
        "<strong>Only cross-source pairs are shown</strong> — same-source pairs are skipped.",
        "r = cov(x,y) / (σ_x · σ_y)",
    ))
    sections.append(card(
        "p-value", "two-sided test of r = 0",
        "Probability of observing |r| this large (or larger) if the true correlation were zero. "
        "Stars: *** p&lt;0.001 &nbsp; ** p&lt;0.01 &nbsp; * p&lt;0.05. "
        "With large N even tiny, practically irrelevant r values become significant — "
        "interpret together with the magnitude of r.",
    ))
    sections.append(card(
        "Spearman r", "rank-based correlation",
        "Pearson r applied to the ranks of x and y instead of raw values. "
        "Robust to outliers and monotone non-linear relationships. "
        "Preferred when distributions are skewed or the relationship is non-linear but monotone.",
        "r_s = 1 − 6·Σd²/[n(n²−1)]",
    ))
    sections.append(card(
        "MI", "mutual information",
        "Measures how much knowing x reduces uncertainty about y (and vice versa). "
        "Non-negative; zero only if x and y are statistically independent. "
        "Unlike r, captures non-linear dependencies. "
        "Estimated via k-nearest-neighbour method (scikit-learn). "
        "Not directly comparable across datasets of different sizes.",
        "MI(x,y) = H(x) + H(y) − H(x,y)",
    ))
    sections.append(card(
        "N valid", "pair sample size",
        "Number of rows where both features are non-NaN and used in the correlation computation. "
        "May differ per pair because different sources have different spatial/temporal coverage.",
    ))
    sections.append("</div>")

    # ── Partial correlations ──
    sections.append("<h2>Partial correlations &mdash; <code>partial_corr.html</code></h2>")
    sections.append("<div class='guide-section'>")
    sections.append(group("Feature–temperature association after removing shared temporal variance (FWL theorem)"))
    sections.append(card(
        "Partial r", "Frisch–Waugh–Lovell",
        "Correlation between a feature and temperature after both have been orthogonalised "
        "with respect to the temporal confounders (intercept, normalised year, "
        "sin/cos of month-of-year, sin/cos of hour-of-day). "
        "Removes seasonal and diurnal bias from the raw correlation, isolating the "
        "<em>net</em> feature–LST relationship.",
        "r_partial = corr(M_Z x, M_Z y)",
    ))
    sections.append(card(
        "M_Z", "residual maker matrix",
        "Projects out the confounder subspace: M_Z = I − Z(Z′Z)⁻¹Z′. "
        "Applied via sufficient-statistics accumulation in a single streaming pass — "
        "never materialised explicitly. "
        "Equivalent to running separate OLS regressions of x and y on Z and correlating the residuals.",
    ))
    sections.append(card(
        "p-value (partial)", "t-test on partial r",
        "Tests H₀: partial r = 0 via a t-statistic with df = n &minus; q &minus; 1, "
        "where q is the number of confounders. "
        "Stars same as cross-corr. With large n, very small partial r values are significant — "
        "focus on the magnitude.",
        "t = r·√(df/(1−r²))",
    ))
    sections.append(card(
        "Confounders Z", "temporal controls (q = 6)",
        "Six columns: (1) intercept, (2) (year−2012.5)/6.25, "
        "(3–4) sin/cos(2π·month/12), (5–6) sin/cos(2π·hour/24). "
        "These absorb annual trends, seasonality, and the diurnal cycle common to all "
        "remote-sensing LST products.",
    ))
    sections.append("</div>")

    # ── Moran's I ──
    sections.append("<h2>Moran&apos;s I &mdash; <code>morans_i.html</code></h2>")
    sections.append("<div class='guide-section'>")
    sections.append(group("Spatial autocorrelation of OLS tile-residuals: do unexplained values cluster?"))
    sections.append(card(
        "Moran's I", "global spatial autocorrelation statistic",
        "Measures whether the OLS residuals of neighbouring H3 tiles (ring-1 binary "
        "weights, row-standardised) are more similar than expected under spatial randomness. "
        "Positive I: residuals cluster (similar values near each other). "
        "Negative I: residuals checker-board. Near E[I]: spatially random. "
        "The <strong>full model</strong> includes all registry features; "
        "the <strong>null model</strong> uses only temporal confounders.",
        "I = (n/S₀) · z′Wz / z′z",
    ))
    sections.append(card(
        "E[I]", "expected value under H₀",
        "Expected Moran's I if residuals were spatially random: E[I] = &minus;1/(n&minus;1). "
        "For large n this is close to 0. "
        "If the observed I is close to E[I], the feature set has absorbed the spatial structure.",
    ))
    sections.append(card(
        "p-value (Moran's)", "permutation test",
        "Fraction of permuted I values (default 999 permutations) whose |I_perm| "
        "exceeds |I_obs|, plus 1 in numerator and denominator (Phipson &amp; Smyth 2010). "
        "Small p: the spatial clustering is unlikely under randomness.",
        "p = (#{|I_perm| ≥ |I_obs|} + 1) / (n_perm + 1)",
    ))
    sections.append(card(
        "Global R²", "OLS in-sample fit on tile means",
        "Coefficient of determination of the OLS model on the tile-level mean residuals. "
        "Not the primary result here — it contextualises how well the model fits "
        "before the spatial check is applied. "
        "Higher R² with lower Moran's I means the features both fit well <em>and</em> "
        "remove spatial autocorrelation.",
        "R² = 1 − SS_res/SS_tot",
    ))
    sections.append(card(
        "Tile level", "H3 resolution 9 (~30 m)",
        "Each observation is assigned to an H3 hexagonal cell at resolution 9 "
        "(edge length ≈ 174 m, area ≈ 0.1 km²). "
        "Residuals are averaged within each tile before the spatial weight matrix is built. "
        "Only tiles with ≥ min_tile_n rows (default 5) are included.",
    ))
    sections.append("</div>")

    # ── Morphology-adjusted greening analysis ──
    sections.append(
        "<h2>Morphology-adjusted greening &mdash; "
        "<code>partial_corr.html</code> &amp; <code>morans_i.html</code></h2>"
    )
    sections.append("<div class='guide-section'>")
    sections.append(group(
        "Three scalars that together support a scientifically defensible greening conclusion"
    ))
    sections.append(card(
        "Greening features (X)", "what counts as greening",
        "Four types, each independent of the LST measurement: "
        "(1) <strong>Structural NDVI</strong> (<em>ndvi_struct_mean/std</em>) — "
        "per-tile temporal mean of NDVI across all observations in the dataset slice; "
        "a proxy for persistent canopy cover, unlike instantaneous NDVI which is co-measured with LST. "
        "(2) <strong>Trees</strong> (<em>trees_plantedby_*</em>) — city tree register counts at 50/70/100 m. "
        "(3) <strong>UA vegetation</strong> (<em>ua_ua_vegetation_*</em>) — Urban Atlas green fraction. "
        "(4) <strong>UA water &amp; wetlands</strong> (<em>ua_ua_water_wetlands_*</em>). "
        "Instantaneous NDVI is excluded: same satellite acquisition as LST = co-measurement, not predictor.",
    ))
    sections.append(card(
        "Morphology confounders", "what is controlled for",
        "Digital height model statistics (DHM avg/max/min at 50/70/100 m buffers) and "
        "built-area urban-atlas fractions (dense built-up, mixed urban, transport, bare/sparse) "
        "at the same buffer radii. "
        "These proxy for building height, urban density, and impervious surface fraction — "
        "the main non-greening determinants of local LST that spatially co-locate with "
        "vegetation. Columns are standardised to mean=0 std=1 per batch; NaN filled with 0 "
        "(= column mean, a conservative treatment that slightly underestimates confounding).",
    ))
    sections.append(card(
        "Adjusted partial r", "scalar 1 — strength of the greening–LST effect",
        "Partial r of a greening feature with temperature after removing the shared variance "
        "of <em>both</em> temporal confounders and morphology confounders. "
        "Negative adjusted r means greening consistently cools LST even among pixels with "
        "similar building height and urban density. "
        "This is the primary effect-size estimate. "
        "Derived via FWL theorem using a dual-accumulator streaming pass. "
        "<strong>Regime dependence</strong>: structural NDVI shows r_adj ≈ −0.23 in the "
        "representative balanced dataset but r_adj ≈ −0.63 to −0.71 during heat events — "
        "the greening–cooling relationship is substantially stronger under thermal stress.",
        "r_adj = corr(M_{Z+morph} · x, M_{Z+morph} · y)",
    ))
    sections.append(card(
        "Confounding fraction (CF)", "scalar 2 — how much is morphological co-location",
        "The fraction of the raw (temporal-only adjusted) partial r that disappears when "
        "morphology confounders are also controlled. "
        "<span class='gformula'>CF = (r_base &minus; r_adj) / r_base</span>. "
        "CF = 0: greening is not co-located with morphology — the effect is entirely its own. "
        "CF = 1: the raw association vanishes once building height and density are controlled — "
        "it was morphological confounding throughout. "
        "CF &gt; 0.5 suggests the majority of the raw correlation is due to co-location of "
        "green space with low-rise, low-density areas rather than a direct cooling effect.",
        "CF = (r_base − r_adj) / r_base",
    ))
    sections.append(card(
        "Effect slope", "OLS coefficient in natural units",
        "The OLS slope of (M<sub>Z</sub>&thinsp;·&thinsp;y) regressed on (M<sub>Z</sub>&thinsp;·&thinsp;x): "
        "the expected change in temperature (Kelvin) per 1-unit increase in the feature, "
        "after partialling out confounders. "
        "Multiply by <em>feature_std</em> to get the per-1-SD cooling in Kelvin — "
        "the planning-relevant effect size. "
        "For structural NDVI (range 0–1), slope ≈ 0.4–1.0 K per NDVI unit during heat events; "
        "a park with 0.1 higher mean NDVI is 0.04–0.10 K cooler than a comparable built-up area. "
        "For tree counts, slope is typically −0.01 to −0.05 K per tree-in-buffer.",
        "slope = x&prime;M_Z y / x&prime;M_Z x",
    ))
    sections.append(card(
        "&Delta;I greening", "scalar 3 — spatial evidence for a direct effect",
        "Defined as I(morphology model) &minus; I(full model). "
        "A <strong>positive</strong> &Delta;I means greening features reduce tile-level spatial "
        "autocorrelation beyond what morphology alone achieves — neighbourhood-scale spatial "
        "evidence for a direct cooling effect. "
        "A <strong>negative</strong> &Delta;I means greening increases residual clustering: "
        "the model uses spatially clustered greening features to fit local variation but leaves "
        "inter-cluster residuals more structured. "
        "<strong>Regime dependence</strong>: across 9 full datasets, &Delta;I_greening is "
        "positive in 7 (all heat/drought/warm/midday outliers: +0.04 to +0.25) and negative "
        "in 2 (full_representative: −0.10, single_year_2017: −0.04). "
        "This suggests greening has a detectable neighbourhood-scale spatial cooling signal "
        "specifically under thermal stress, but not across the full balanced temporal distribution.",
        "&Delta;I = I(morphology model) &minus; I(full model)",
    ))
    sections.append(card(
        "Traceability fields", "run_timestamp · versions · dataset_stats · skipped_features",
        "Every JSON output includes: "
        "<strong>run_timestamp</strong> (UTC ISO-8601 of the analysis run), "
        "<strong>versions</strong> (Python / NumPy / Pandas), "
        "<strong>dataset_stats</strong> (n_rows, n_tiles, year_range, month_range, "
        "temperature mean/std/min/max, LST source breakdown, NDVI coverage fraction), "
        "<strong>greening_by_type</strong> (count of features per physical category), "
        "<strong>skipped_features</strong> (features excluded due to insufficient valid data, "
        "e.g. <em>trees_plantedby_*_Veteranenfase</em>), "
        "and per-result <strong>feature_mean / feature_std</strong> (for unit conversion). "
        "These fields exist to make any result reproducible and self-describing — "
        "paste the JSON block into any future analysis to reconstruct exact conditions.",
    ))
    sections.append(card(
        "Three-model comparison", "null → morphology → full",
        "All three Moran&apos;s I values (null, morphology, full) are computed in a single "
        "streaming pass. "
        "<strong>Null model</strong>: temperature ~ temporal confounders Z only. "
        "<strong>Morphology model</strong>: temperature ~ morphology features + Z. "
        "<strong>Full model</strong>: temperature ~ all features (greening + morphology) + Z. "
        "The null→morphology drop shows how much spatial structure is explained by built form. "
        "The morphology→full change (i.e. &Delta;I) isolates greening&apos;s unique spatial "
        "contribution above and beyond morphology.",
    ))
    sections.append("</div>")

    # ── Spatial CV ──
    sections.append("<h2>Spatial CV &mdash; <code>spatial_cv.html</code></h2>")
    sections.append("<div class='guide-section'>")
    sections.append(group("How well do features generalise to unseen spatial locations?"))
    sections.append(card(
        "LOO-block-CV", "leave-one-block-out cross-validation",
        "For each spatial block b, the model is re-fitted on all <em>other</em> blocks "
        "using sufficient-statistics subtraction (XtX_train = XtX &minus; XtX_b) — "
        "no second streaming pass is required. "
        "The held-out block provides the test set. "
        "Spatial LOO-CV is a stricter test than random CV: it measures whether "
        "learned patterns transfer across space, revealing overfitting to local patterns.",
    ))
    sections.append(card(
        "R² (LOO)", "coefficient of determination on held-out block",
        "Proportion of temperature variance explained by the model fitted without that block. "
        "R² = 1 means perfect prediction; R² = 0 means no better than the held-out mean; "
        "R² &lt; 0 means worse than the mean (extrapolation failure). "
        "The <strong>mean, std, Q10, Q90</strong> summarise the distribution across all blocks.",
        "R² = 1 − Σ(y−ŷ)² / Σ(y−ȳ)²",
    ))
    sections.append(card(
        "MAE", "mean absolute error",
        "Average absolute difference between predicted and observed temperature (°C or K). "
        "Robust to large individual errors. "
        "Easier to interpret than RMSE: 'on average the model is off by X units'.",
        "MAE = mean(|y − ŷ|)",
    ))
    sections.append(card(
        "RMSE", "root mean squared error",
        "Square root of the mean squared residual. "
        "Penalises large errors more than MAE does. "
        "In the same units as temperature. "
        "RMSE ≥ MAE always; a large gap signals heavy-tailed residual distribution.",
        "RMSE = √(mean((y−ŷ)²))",
    ))
    sections.append(card(
        "Random-fold R²", "random K-fold CV baseline",
        "K-fold CV (default K=10) on the same stored test rows but with folds assigned "
        "randomly rather than spatially. "
        "Comparison baseline: if LOO R² &lt;&lt; Random-fold R², the model overfits spatially "
        "(it memorises local structure but fails to extrapolate). "
        "If both are similar, spatial generalisation is good.",
    ))
    sections.append(card(
        "Block column", "spatial partitioning strategy",
        "How rows are grouped into spatial blocks for LOO-CV. "
        "<strong>h3r8</strong> (default): H3 resolution 8 hexagons (~860 m). "
        "<strong>h3r7</strong>: H3 resolution 7 (~2.3 km, fewer, larger blocks). "
        "<strong>rect1km / rect2km</strong>: rectangular Lambert grid cells. "
        "Coarser blocks create a harder generalisation test.",
    ))
    sections.append(card(
        "N blocks", "number of evaluated blocks",
        "Blocks with fewer than min_block_n training rows (default 50) are excluded. "
        "More blocks = finer spatial resolution + longer runtime.",
    ))
    sections.append("</div>")

    # ── Stability ──
    sections.append("<h2>Stability &mdash; <code>stability.html</code></h2>")
    sections.append("<div class='guide-section'>")
    sections.append(group("Cross-dataset consistency: how robust is each metric across data slices?"))
    sections.append(card(
        "Full vs Spatial datasets", "reliability tier",
        "<strong>Full datasets</strong> (≥ 10 000 rows): full_representative, outlier_*, "
        "single_year_*. Provide reliable estimates; used as primary stability inputs. "
        "<strong>Spatial datasets</strong> (&lt; 10 000 rows): point_*, single_image. "
        "Small-n, noisy estimates; shown separately as a local-agreement check "
        "but not counted in the stability score.",
    ))
    sections.append(card(
        "Stability score", "1 − CV",
        "Main summary statistic. CV = std / mean of the metric across full datasets. "
        "Stability = max(0, 1 − CV). "
        "Ranges 0–1: 1 = identical value across all datasets (perfectly stable); "
        "0 = erratic, CV ≥ 1 (std ≥ mean). "
        "Colour scale: red (unstable) → green (stable).",
        "stability = max(0, 1 − σ/(μ + ε))",
    ))
    sections.append(card(
        "CV", "coefficient of variation",
        "Normalised spread of a metric across datasets: CV = std / mean. "
        "Unitless; allows comparing stability of metrics on different scales. "
        "CV &lt; 0.2 = very stable; CV 0.2–0.5 = moderately stable; "
        "CV &gt; 1 = highly variable. "
        "Colour scale: green (low CV) → yellow → red (high CV).",
    ))
    sections.append(card(
        "Sign%", "sign consistency",
        "For correlation-type metrics (Pearson r, Spearman r, partial r): "
        "fraction of full datasets where the metric has the majority sign. "
        "100% = always positive (or always negative). "
        "A metric with stable magnitude but inconsistent sign is unreliable.",
    ))
    sections.append(card(
        "Spatial agree", "local vs global sign agreement",
        "Applies to cross-corr pairs only. "
        "Fraction of spatial (small-n) datasets where the sign matches the majority sign "
        "from the full datasets. "
        "High spatial agreement means the relationship holds even in small local patches.",
    ))
    sections.append(card(
        "Mean |Pearson r| / Mean |Spearman r|", "average magnitude across datasets",
        "Mean of |r| across full datasets — ignores sign. "
        "Paired with its std and CV to show both strength and stability. "
        "A pair can be strong on average but unstable (high CV), or weak but rock-solid.",
    ))
    sections.append(card(
        "Mean MI", "average mutual information across datasets",
        "Mean of MI (non-negative, sign-less) across full datasets. "
        "Captures non-linear dependencies that r misses. "
        "Stable MI with near-zero r suggests a non-linear, consistently present relationship.",
    ))
    sections.append("</div>")

    return (
        f"<style>{_GUIDE_CSS}</style>"
        f"<div class='guide'>"
        f"<h2 style='margin-top:32px'>Metric guide</h2>"
        f"<p style='font-size:13px;color:#555;margin-bottom:8px'>"
        f"Explanations of every value reported across the four analyses.</p>"
        + "".join(sections)
        + "</div>"
    )


# ── Index page ─────────────────────────────────────────────────────────────────

def render_index(
    sc_results:  Optional[dict],
    cc_results:  Optional[dict],
    pc_results:  Optional[dict],
    mi_results:  Optional[dict],
    scv_results: Optional[dict],
    out_dir:     Path,
) -> None:
    cc_datasets = [(k, v) for k, v in cc_results.items() if k != "_meta"] if cc_results else []

    cards_html = (
        "<div class='card-grid'>"
        "<div class='card'><h3><a href='stability.html'>&#128202; Stability</a></h3>"
        "<p>Cross-dataset consistency for all four analyses.</p></div>"
        + ("<div class='card'><h3><a href='simple_corr.html'>&#128200; Simple correlations</a></h3>"
           "<p>Pearson r, Spearman &rho;, and slope K/unit per greening feature vs temperature.</p></div>"
           if sc_results else "")
        + ("<div class='card'><h3>&#128200; Cross-correlations</h3>"
           f"<p>{len(cc_datasets)} dataset(s) — feature pair heatmaps.</p>"
           "<p style='margin-top:6px'>"
           + " ".join(f"<a href='{n}.html'>{n}</a>" for n, _ in cc_datasets[:6])
           + ("&hellip;" if len(cc_datasets) > 6 else "")
           + "</p></div>" if cc_results else "")
        + ("<div class='card'><h3><a href='partial_corr.html'>&#128202; Partial correlations</a></h3>"
           "<p>Feature partial r with temperature (FWL), per dataset.</p></div>"
           if pc_results else "")
        + ("<div class='card'><h3><a href='morans_i.html'>&#127758; Moran&apos;s I</a></h3>"
           "<p>Spatial autocorrelation of OLS residuals — null vs full model.</p></div>"
           if mi_results else "")
        + ("<div class='card'><h3><a href='spatial_cv.html'>&#127975; Spatial CV</a></h3>"
           "<p>Leave-one-block-out cross-validation R², MAE, RMSE.</p></div>"
           if scv_results else "")
        + "</div>"
    )

    if cc_datasets:
        cc_rows = []
        for name, data in cc_datasets:
            pairs = data.get("pairs", [])
            top3  = sorted(pairs, key=lambda p: abs(p.get("pearson_r") or 0), reverse=True)[:3]
            top_s = " ".join(
                f"<span class='tag'>{p['source_a']}&times;{p['source_b']} "
                f"r={_fmt_r(p.get('pearson_r'))}</span>"
                for p in top3
            )
            cc_rows.append(
                f"<tr><td><a href='{name}.html'>{name}</a></td>"
                f"<td>{data.get('n_rows_total','?'):,}</td>"
                f"<td>{len(pairs):,}</td>"
                f"<td>{top_s}</td></tr>"
            )
        cc_table = (
            "<h2>Cross-correlation datasets</h2>"
            "<div class='table-wrap'>"
            "<table class='idx'><thead><tr>"
            "<th>Dataset</th><th>Rows</th><th>Pairs</th><th>Top Pearson pairs</th>"
            "</tr></thead><tbody>"
            + "".join(cc_rows) +
            "</tbody></table></div>"
        )
    else:
        cc_table = ""

    guide_html = _guide_html()

    body = (
        "<h1>Data-science analysis reports</h1>"
        "<p style='color:#666;font-size:13px;margin:6px 0 16px'>"
        "Generated from <code>ds_reports/*.json</code>.</p>"
        + cards_html + cc_table + guide_html
    )

    (out_dir / "index.html").write_text(_shell("DS Reports", body), encoding="utf-8")


# ── Driver ─────────────────────────────────────────────────────────────────────

def _load(path: Path) -> Optional[dict]:
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [warn] failed to parse {path.name}: {e}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--results-dir", type=Path, default=_RESULTS_DIR,
                    help="directory containing *.json analysis outputs")
    ap.add_argument("--out",         type=Path, default=_OUT_DIR,
                    help="output directory for HTML files")
    args = ap.parse_args()

    r = args.results_dir
    sc_results  = _load(r / "simple_corr.json")
    cc_results  = _load(r / "cross_corr.json")
    pc_results  = _load(r / "partial_corr.json")
    mi_results  = _load(r / "morans_i.json")
    scv_results = _load(r / "spatial_cv.json")

    args.out.mkdir(parents=True, exist_ok=True)

    # Simple-corr page
    if sc_results:
        render_simple_corr_page(sc_results, args.out)
        print("  wrote simple_corr.html")

    # Per-dataset cross-corr pages
    if cc_results:
        datasets = [(k, v) for k, v in cc_results.items() if k != "_meta"]
        print(f"rendering {len(datasets)} cross-corr dataset page(s) ...")
        for name, data in datasets:
            render_cc_dataset_page(name, data, args.out)
            print(f"  wrote {name}.html")

    # Analysis pages
    if pc_results:
        render_partial_corr_page(pc_results, args.out)
        print("  wrote partial_corr.html")

    if mi_results:
        render_morans_i_page(mi_results, args.out)
        print("  wrote morans_i.html")

    if scv_results:
        render_spatial_cv_page(scv_results, args.out)
        print("  wrote spatial_cv.html")

    # Stability page (combines all four)
    render_stability_page(cc_results, pc_results, mi_results, scv_results, args.out)
    print("  wrote stability.html")

    # Index
    render_index(sc_results, cc_results, pc_results, mi_results, scv_results, args.out)
    print("  wrote index.html")

    print(f"\ndone -> {args.out / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

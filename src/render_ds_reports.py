"""
render_ds_reports.py
====================

Render ds_test_results.json into self-contained HTML reports under ds_reports/.

    ds_reports/
        index.html           overview of all datasets + cross-dataset summary
        <dataset>.html       per-dataset report: heatmaps + sortable pair table

Usage
-----
    python src/render_ds_reports.py
    python src/render_ds_reports.py --results src/ds_test_results.json --out src/ds_reports
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SRC     = Path(__file__).parent
_RESULTS = _SRC / "ds_test_results.json"
_OUT_DIR = _SRC / "ds_reports"

_SOURCE_ORDER = ["LST", "DHM", "Trees", "UA", "WIS"]


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _lerp_colour(lo: Tuple[int,int,int], hi: Tuple[int,int,int], t: float) -> str:
    t = max(0.0, min(1.0, t))
    r = int(lo[0] + (hi[0] - lo[0]) * t)
    g = int(lo[1] + (hi[1] - lo[1]) * t)
    b = int(lo[2] + (hi[2] - lo[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"

_WHITE   = (255, 255, 255)
_ORANGE  = (255, 160,  60)
_RED     = (180,  30,  10)
_BLUE_HI = ( 21, 101, 192)

def _r_colour(abs_r: float) -> str:
    """White -> orange -> red scale for |correlation|."""
    if abs_r <= 0.5:
        return _lerp_colour(_WHITE, _ORANGE, abs_r / 0.5)
    return _lerp_colour(_ORANGE, _RED, (abs_r - 0.5) / 0.5)

def _mi_colour(mi: float, max_mi: float) -> str:
    """White -> blue scale for MI."""
    t = mi / max_mi if max_mi > 0 else 0.0
    return _lerp_colour(_WHITE, _BLUE_HI, t)

def _sig(p: Optional[float]) -> str:
    if p is None:  return ""
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return ""

def _fmt_r(v: Optional[float]) -> str:
    return f"{v:+.3f}" if v is not None else "—"

def _fmt_p(v: Optional[float]) -> str:
    if v is None:   return "—"
    if v < 1e-10:   return "<1e-10"
    if v < 0.001:   return f"{v:.2e}"
    return f"{v:.4f}"

def _fmt_mi(v: Optional[float]) -> str:
    return f"{v:.4f}" if v is not None else "—"


# ---------------------------------------------------------------------------
# Heatmap (mean |metric| per source pair)
# ---------------------------------------------------------------------------

def _build_heatmap_data(pairs: List[dict], metric: str) -> Dict[Tuple[str,str], List[float]]:
    acc: Dict[Tuple[str,str], List[float]] = defaultdict(list)
    for p in pairs:
        val = p.get(metric)
        if val is None:
            continue
        key = tuple(sorted([p["source_a"], p["source_b"]]))
        acc[key].append(abs(val) if "r" in metric else val)
    return acc

def _render_heatmap_table(
    sources: List[str],
    data: Dict[Tuple[str,str], List[float]],
    colour_fn,
) -> str:
    means: Dict[Tuple[str,str], float] = {}
    for key, vals in data.items():
        means[key] = sum(vals) / len(vals) if vals else 0.0

    header = "".join(f"<th>{s}</th>" for s in sources)
    rows_html = []
    for sa in sources:
        cells = [f"<td class='hm-label'>{sa}</td>"]
        for sb in sources:
            key = tuple(sorted([sa, sb]))
            if sa == sb:
                cells.append("<td class='hm-na'>—</td>")
            elif key in means:
                v   = means[key]
                bg  = colour_fn(v)
                tip = f"mean={v:.3f} (n={len(data[key])})"
                cells.append(
                    f"<td class='hm-cell' style='background:{bg}' title='{tip}'>"
                    f"{v:.3f}</td>"
                )
            else:
                cells.append("<td class='hm-na'>n/a</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    return (
        f"<table class='heatmap'>"
        f"<thead><tr><th></th>{header}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        f"</table>"
    )


# ---------------------------------------------------------------------------
# CSS + JS (embedded, no external deps)
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font: 14px/1.5 'Segoe UI', Arial, sans-serif; color: #222;
       background: #f4f6f8; padding: 24px; }
a { color: #1565c0; text-decoration: none; }
a:hover { text-decoration: underline; }
h1 { font-size: 1.5rem; margin-bottom: 8px; }
h2 { font-size: 1.1rem; margin: 24px 0 8px; color: #444; border-bottom: 1px solid #ccc;
     padding-bottom: 4px; }
.meta-box { background: #fff; border: 1px solid #dde; border-radius: 4px;
            padding: 12px 16px; margin-bottom: 12px; display: flex; flex-wrap: wrap;
            gap: 24px; font-size: 13px; }
.meta-box span { color: #555; }
.meta-box strong { color: #111; }

/* Heatmaps */
.heatmaps { display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 8px; }
.heatmap-block h3 { font-size: .85rem; font-weight: 600; margin-bottom: 4px;
                    color: #555; }
table.heatmap { border-collapse: collapse; font-size: 12px; }
table.heatmap th, table.heatmap td { border: 1px solid #ccc; padding: 4px 8px;
                                      text-align: center; min-width: 52px; }
table.heatmap th { background: #e8ecf0; font-weight: 600; }
.hm-label { background: #e8ecf0; font-weight: 600; text-align: left; }
.hm-na { background: #f0f0f0; color: #aaa; }
.hm-cell { cursor: default; font-size: 11px; }

/* Controls */
.controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
             margin-bottom: 8px; }
.controls input, .controls select {
    border: 1px solid #bbb; border-radius: 3px; padding: 4px 8px;
    font-size: 13px; background: #fff; }
.controls input { width: 220px; }
.controls label { font-size: 13px; color: #555; }
.count-info { font-size: 12px; color: #777; margin-left: auto; }

/* Pairs table */
.table-wrap { overflow-x: auto; max-height: 70vh; overflow-y: auto;
              border: 1px solid #dde; border-radius: 4px; background: #fff; }
table.pairs { border-collapse: collapse; width: 100%; font-size: 12px; }
table.pairs thead tr { position: sticky; top: 0; z-index: 2; }
table.pairs th { background: #2c3e50; color: #fff; padding: 6px 10px;
                  text-align: left; white-space: nowrap; cursor: pointer;
                  user-select: none; }
table.pairs th:hover { background: #3d5166; }
table.pairs th.sorted-asc::after  { content: ' \\25b2'; font-size: 10px; }
table.pairs th.sorted-desc::after { content: ' \\25bc'; font-size: 10px; }
table.pairs td { padding: 4px 10px; border-bottom: 1px solid #eee;
                  white-space: nowrap; }
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
.r-cell    { font-weight: 600; }
.sig       { font-size: 10px; color: #c62828; margin-left: 1px; }
.nav-bar   { margin-bottom: 16px; font-size: 13px; }

/* Index table */
table.idx { border-collapse: collapse; background: #fff;
             border: 1px solid #dde; border-radius: 4px; width: 100%; }
table.idx th { background: #2c3e50; color: #fff; padding: 6px 12px;
                text-align: left; }
table.idx td { padding: 6px 12px; border-bottom: 1px solid #eee; font-size: 13px; }
table.idx tr:hover td { background: #f0f4fa; }
.tag { display: inline-block; padding: 1px 5px; border-radius: 3px;
        font-size: 11px; background: #e8ecf0; color: #444; margin: 1px; }
"""

_SORT_JS = """
(function() {
  var sortCol = -1, sortAsc = true;

  function cellVal(cell) {
    return cell.dataset.v !== undefined ? cell.dataset.v : cell.textContent.trim();
  }

  function sortTable(col) {
    var table = document.getElementById('pairs-table');
    var thead = table.querySelector('thead');
    var tbody = table.querySelector('tbody');
    var ths = thead.querySelectorAll('th');
    if (sortCol === col) { sortAsc = !sortAsc; }
    else { sortCol = col; sortAsc = true; }

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
    var text  = document.getElementById('filter').value.toLowerCase();
    var srcA  = document.getElementById('src-a').value;
    var srcB  = document.getElementById('src-b').value;
    var rows  = document.querySelectorAll('#pairs-table tbody tr');
    rows.forEach(function(r) {
      var fa = r.cells[0].textContent.toLowerCase();
      var sa = r.cells[1].textContent.trim();
      var fb = r.cells[2].textContent.toLowerCase();
      var sb = r.cells[3].textContent.trim();
      var show = (
        (fa.includes(text) || fb.includes(text)) &&
        (srcA === '' || sa === srcA) &&
        (srcB === '' || sb === srcB)
      );
      r.classList.toggle('hidden', !show);
    });
    updateCount();
  }

  function updateCount() {
    var rows = document.querySelectorAll('#pairs-table tbody tr');
    var vis  = document.querySelectorAll('#pairs-table tbody tr:not(.hidden)');
    document.getElementById('row-count').textContent =
      'Showing ' + vis.length + ' of ' + rows.length + ' pairs';
  }

  window.addEventListener('DOMContentLoaded', function() {
    var ths = document.querySelectorAll('#pairs-table thead th');
    ths.forEach(function(th, i) {
      th.addEventListener('click', function() { sortTable(i); });
    });
    document.getElementById('filter').addEventListener('input', applyFilter);
    document.getElementById('src-a').addEventListener('change', applyFilter);
    document.getElementById('src-b').addEventListener('change', applyFilter);
    updateCount();
  });
})();
"""


# ---------------------------------------------------------------------------
# Per-dataset page
# ---------------------------------------------------------------------------

def _pairs_table_rows(pairs: List[dict], max_mi: float) -> str:
    rows = []
    for p in pairs:
        fa, sa = p["feature_a"], p["source_a"]
        fb, sb = p["feature_b"], p["source_b"]
        pr  = p.get("pearson_r")
        pp  = p.get("pearson_p")
        sr  = p.get("spearman_r")
        sp  = p.get("spearman_p")
        mi  = p.get("mi", 0.0) or 0.0
        nv  = p.get("n_valid", "")

        abs_r_max = max(abs(pr) if pr is not None else 0,
                        abs(sr) if sr is not None else 0)
        r_bg  = _r_colour(abs_r_max)
        mi_bg = _mi_colour(mi, max_mi)

        def r_cell(v, pv):
            if v is None:
                return "<td>—</td><td>—</td>"
            bg = _r_colour(abs(v))
            return (
                f"<td class='r-cell' style='background:{bg}' data-v='{v:.6f}'>"
                f"{_fmt_r(v)}<span class='sig'>{_sig(pv)}</span></td>"
                f"<td data-v='{pv if pv is not None else 1:.6f}'>{_fmt_p(pv)}</td>"
            )

        rows.append(
            f"<tr>"
            f"<td class='feat'>{fa}</td>"
            f"<td><span class='src-badge src-{sa}'>{sa}</span></td>"
            f"<td class='feat'>{fb}</td>"
            f"<td><span class='src-badge src-{sb}'>{sb}</span></td>"
            f"<td data-v='{nv}'>{nv:,}</td>"
            + r_cell(pr, pp)
            + r_cell(sr, sp)
            + f"<td class='r-cell' style='background:{mi_bg}' data-v='{mi:.6f}'>{_fmt_mi(mi)}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def _source_options(sources: List[str]) -> str:
    opts = "<option value=''>All</option>"
    opts += "".join(f"<option>{s}</option>" for s in sources)
    return opts


def render_dataset_page(name: str, data: dict, out_dir: Path) -> None:
    pairs = data.get("pairs", [])
    if not pairs:
        # Write a stub page so the index can still link to it.
        note = data.get("note", "no pairs computed")
        html = _page_shell(
            f"DS Report: {name}",
            f'<div class="nav-bar"><a href="index.html">&larr; Index</a></div>'
            f'<h1>{name}</h1><p style="color:#888;margin-top:16px">{note}</p>',
        )
        (out_dir / f"{name}.html").write_text(html, encoding="utf-8")
        return

    sources_present = sorted(
        {p["source_a"] for p in pairs} | {p["source_b"] for p in pairs},
        key=lambda s: _SOURCE_ORDER.index(s) if s in _SOURCE_ORDER else 99,
    )

    max_mi = max((p.get("mi") or 0) for p in pairs) or 1.0

    pearson_data  = _build_heatmap_data(pairs, "pearson_r")
    spearman_data = _build_heatmap_data(pairs, "spearman_r")
    mi_data       = _build_heatmap_data(pairs, "mi")
    mi_max_dict   = {k: max(v) for k, v in mi_data.items()} if mi_data else {}
    mi_global_max = max(mi_max_dict.values()) if mi_max_dict else 1.0

    heatmaps_html = (
        f"<div class='heatmaps'>"
        f"<div class='heatmap-block'><h3>Mean |Pearson r|</h3>"
        f"{_render_heatmap_table(sources_present, pearson_data, _r_colour)}</div>"
        f"<div class='heatmap-block'><h3>Mean |Spearman r|</h3>"
        f"{_render_heatmap_table(sources_present, spearman_data, _r_colour)}</div>"
        f"<div class='heatmap-block'><h3>Mean MI</h3>"
        f"{_render_heatmap_table(sources_present, mi_data, lambda v: _mi_colour(v, mi_global_max))}</div>"
        f"</div>"
    )

    n_rows_total    = data.get("n_rows_total", "?")
    n_rows_analysed = data.get("n_rows_analysed", "?")
    groups          = data.get("feature_groups", {})
    skipped         = data.get("skipped_same_source", 0)

    groups_tags = "".join(f"<span class='tag'>{k}: {v}</span>" for k, v in groups.items())
    meta_html = (
        f"<div class='meta-box'>"
        f"<div><span>Rows total </span><strong>{n_rows_total:,}</strong></div>"
        f"<div><span>Rows analysed </span><strong>{n_rows_analysed:,}</strong></div>"
        f"<div><span>Cross-source pairs </span><strong>{len(pairs):,}</strong></div>"
        f"<div><span>Same-source skipped </span><strong>{skipped:,}</strong></div>"
        f"<div><span>Feature groups </span>{groups_tags}</div>"
        f"</div>"
    )

    src_opts = _source_options(sources_present)
    controls_html = (
        f"<div class='controls'>"
        f"<label>Filter: <input id='filter' type='text' placeholder='feature name...'></label>"
        f"<label>Source A: <select id='src-a'>{src_opts}</select></label>"
        f"<label>Source B: <select id='src-b'>{src_opts}</select></label>"
        f"<span class='count-info' id='row-count'></span>"
        f"</div>"
    )

    table_html = (
        f"<div class='table-wrap'>"
        f"<table class='pairs' id='pairs-table'>"
        f"<thead><tr>"
        f"<th>Feature A</th><th>Source A</th>"
        f"<th>Feature B</th><th>Source B</th>"
        f"<th>N valid</th>"
        f"<th>Pearson r</th><th>p</th>"
        f"<th>Spearman r</th><th>p</th>"
        f"<th>MI</th>"
        f"</tr></thead>"
        f"<tbody>{_pairs_table_rows(pairs, max_mi)}</tbody>"
        f"</table></div>"
    )

    body = (
        f'<div class="nav-bar"><a href="index.html">&larr; Index</a></div>'
        f"<h1>Dataset: {name}</h1>"
        f"{meta_html}"
        f"<h2>Source-pair summary</h2>"
        f"{heatmaps_html}"
        f"<h2>All pairs <small style='font-weight:normal;color:#777'>"
        f"(click column headers to sort)</small></h2>"
        f"{controls_html}"
        f"{table_html}"
    )

    html = _page_shell(f"DS Report: {name}", body, include_sort_js=True)
    (out_dir / f"{name}.html").write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------

def _dataset_top_pairs(pairs: List[dict], n: int = 3) -> str:
    if not pairs:
        return "—"
    ranked = sorted(pairs, key=lambda p: abs(p.get("pearson_r") or 0), reverse=True)
    items = []
    for p in ranked[:n]:
        r   = p.get("pearson_r")
        tip = p["feature_a"] + " x " + p["feature_b"]
        sa  = p["source_a"]
        sb  = p["source_b"]
        items.append(
            f"<span class='tag' title='{tip}'>"
            f"{sa}&times;{sb} r={_fmt_r(r)}</span>"
        )
    return " ".join(items)


def render_index(results: dict, out_dir: Path) -> None:
    meta = results.get("_meta", {})
    datasets = [(k, v) for k, v in results.items() if k != "_meta"]

    rows_html = []
    for name, data in datasets:
        pairs     = data.get("pairs", [])
        n_rows    = data.get("n_rows_total", "?")
        n_pairs   = len(pairs)
        groups    = data.get("feature_groups", {})
        note      = data.get("note", "")
        group_str = " ".join(f"<span class='tag'>{k}:{v}</span>" for k, v in groups.items())
        top       = _dataset_top_pairs(pairs)
        page_link = f"<a href='{name}.html'>{name}</a>"
        rows_html.append(
            f"<tr>"
            f"<td>{page_link}</td>"
            f"<td>{n_rows:,}</td>"
            f"<td>{n_pairs:,}</td>"
            f"<td>{group_str}</td>"
            f"<td>{top}</td>"
            f"</tr>"
        )

    table_html = (
        f"<table class='idx'>"
        f"<thead><tr>"
        f"<th>Dataset</th><th>Rows</th><th>Pairs</th>"
        f"<th>Feature groups</th><th>Top Pearson pairs</th>"
        f"</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        f"</table>"
    )

    settings_html = (
        f"<div class='meta-box'>"
        f"<div><span>Max analysis rows </span><strong>{meta.get('max_analysis_rows', '?')}</strong></div>"
        f"<div><span>RNG seed </span><strong>{meta.get('seed', '?')}</strong></div>"
        f"<div><span>Metrics </span><strong>{', '.join(meta.get('metrics', []))}</strong></div>"
        f"<div><span>Source groups </span><strong>{', '.join(meta.get('source_groups', []))}</strong></div>"
        f"</div>"
    )

    stability_link = (
        "<p style='margin-bottom:12px'>"
        "<a href='stability.html'>&rarr; Cross-dataset stability report</a>"
        " &mdash; see which correlations are consistent across all datasets"
        "</p>"
    )

    body = (
        f"<h1>Data-science feature relation tests</h1>"
        f"{stability_link}"
        f"<h2>Run settings</h2>{settings_html}"
        f"<h2>Datasets ({len(datasets)})</h2>{table_html}"
    )
    html = _page_shell("DS Reports — Index", body)
    (out_dir / "index.html").write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Page shell
# ---------------------------------------------------------------------------

def _page_shell(title: str, body: str, include_sort_js: bool = False) -> str:
    js_block = f"<script>{_SORT_JS}</script>" if include_sort_js else ""
    return (
        f"<!DOCTYPE html><html lang='en'><head>"
        f"<meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{title}</title>"
        f"<style>{_CSS}</style>"
        f"{js_block}"
        f"</head><body>{body}</body></html>"
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=_RESULTS)
    parser.add_argument("--out",     type=Path, default=_OUT_DIR)
    args = parser.parse_args()

    if not args.results.exists():
        print(f"results file not found: {args.results}", file=sys.stderr)
        return 1

    results = json.loads(args.results.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    datasets = [(k, v) for k, v in results.items() if k != "_meta"]
    print(f"rendering {len(datasets)} dataset page(s) + index ...")

    for name, data in datasets:
        render_dataset_page(name, data, args.out)
        print(f"  wrote {name}.html  ({len(data.get('pairs', [])):,} pairs)")

    render_index(results, args.out)
    print(f"  wrote index.html")
    print(f"\ndone -> {args.out / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

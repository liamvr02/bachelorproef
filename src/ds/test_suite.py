"""
test_suite.py
=============

Orchestrator: runs all data-science analyses in sequence.

Analyses
--------
  cross_corr    cross_corr.py   Cross-source feature correlations (Pearson, Spearman, MI)
  partial_corr  partial_corr.py Partial correlations with temperature (FWL theorem)
  morans_i      morans_i.py     Moran's I spatial autocorrelation on OLS residuals
  spatial_cv    spatial_cv.py   Spatial leave-one-block-out cross-validation

Default behaviour
-----------------
All analyses run over every pre-built parquet in ``src/test_data/``, one at a
time.  Results are nested by dataset name in each output file.

Flags
-----
  --full-stream   Stream all source data instead of using pre-built parquets
                  (streaming analyses run once; cross_corr still uses parquets).
  --dataset NAME  Use one specific pre-built dataset for all analyses.
  --build-missing Build any missing test datasets before running.

Results for each analysis are written to ``src/ds_reports/<analysis>.json``.
A suite summary is written to ``src/ds_reports/suite_summary.json``.

Usage
-----
    # Default: all pre-built parquets
    python src/ds/test_suite.py

    # One dataset, build it first if needed
    python src/ds/test_suite.py --dataset full_representative --build-missing

    # Full-stream mode (slow, scans all data)
    python src/ds/test_suite.py --full-stream

    # Subset of analyses
    python src/ds/test_suite.py --only cross_corr,morans_i
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, List, Optional, Tuple

_DS_DIR      = Path(__file__).parent
_SRC         = _DS_DIR.parent
_REPORTS_DIR = _SRC / "ds_reports"

sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_DS_DIR))

_ALL_ANALYSES = ["simple_corr", "cross_corr", "partial_corr", "morans_i", "spatial_cv"]


def _list_datasets() -> List[Tuple[str, Path]]:
    from utils import list_datasets
    return list_datasets()


def _run_for_datasets(
    run_fn:    Callable,
    args,
    extra_kw:  dict,
    *,
    no_stream: bool = False,
) -> object:
    """
    Call *run_fn* according to the active mode and return the result.

    - ``--full-stream`` (and not no_stream): call once with ``dataset=None``
    - ``--dataset NAME``:  call once with ``dataset=NAME``
    - default:             call once per parquet, return ``{name: result, ...}``

    *no_stream* disables full-stream mode for analyses that have no streaming
    equivalent (e.g. cross_corr).
    """
    if args.full_stream and not no_stream:
        return run_fn(**extra_kw, dataset=None)

    if args.dataset:
        return {args.dataset: run_fn(**extra_kw, dataset=args.dataset)}

    datasets = _list_datasets()
    if not datasets:
        raise RuntimeError(
            f"no parquet files in {_SRC / 'test_data'} -- "
            "run build_test_datasets.py or pass --build-missing"
        )
    combined = {}
    for name, _ in datasets:
        print(f"  --- {name} ---")
        combined[name] = run_fn(**extra_kw, dataset=name)
    return combined


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--dataset", type=str, default=None,
        help="use one specific pre-built dataset for all analyses",
    )
    ap.add_argument(
        "--full-stream", action="store_true",
        help="stream all source data instead of pre-built parquets "
             "(cross_corr still uses parquets; mutually exclusive with --dataset)",
    )
    ap.add_argument(
        "--build-missing", action="store_true",
        help="build missing test datasets before running",
    )
    ap.add_argument(
        "--only", type=str, default="",
        help=f"comma-separated subset of analyses "
             f"(default: all; choices: {', '.join(_ALL_ANALYSES)})",
    )
    # Shared pass-through args
    ap.add_argument("--max-rows",          type=int,   default=None,
                    help="row cap / sample size for all analyses (default: all)")
    ap.add_argument("--batch-size",        type=int,   default=50_000)
    ap.add_argument("--seed",              type=int,   default=42)
    # morans_i overrides
    ap.add_argument("--n-perm",            type=int,   default=999)
    ap.add_argument("--min-tile-n",        type=int,   default=5)
    # spatial_cv overrides
    ap.add_argument("--block-col",         type=str,   default="h3r8",
                    choices=["h3r8", "h3r7", "rect1km", "rect2km"])
    ap.add_argument("--ridge",             type=float, default=1e-4)
    ap.add_argument("--out-dir",           type=Path,  default=_REPORTS_DIR)
    args = ap.parse_args()

    if args.full_stream and args.dataset:
        ap.error("--full-stream and --dataset are mutually exclusive")

    only = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else set(_ALL_ANALYSES)
    unknown = only - set(_ALL_ANALYSES)
    if unknown:
        print(f"unknown analyses: {sorted(unknown)} -- choices: {_ALL_ANALYSES}",
              file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.build_missing:
        from build_test_datasets import build as _build
        names = {args.dataset} if args.dataset else None
        _build(names=names, rebuild=False)

    import platform
    from datetime import datetime, timezone as _tz
    _ts = datetime.now(_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    summary: dict = {
        "_meta": {
            "run_timestamp": _ts,
            "dataset":       args.dataset,
            "full_stream":   args.full_stream,
            "analyses_run":  sorted(only),
            "python":        platform.python_version(),
            "platform":      platform.system(),
            "args": {
                "max_rows":   args.max_rows,
                "batch_size": args.batch_size,
                "seed":       args.seed,
                "n_perm":     args.n_perm,
                "min_tile_n": args.min_tile_n,
                "block_col":  args.block_col,
                "ridge":      args.ridge,
            },
        },
    }
    t_suite = time.perf_counter()

    # ── simple_corr ───────────────────────────────────────────────────────────
    if "simple_corr" in only:
        _section("simple_corr  (simple_corr.py)")
        from simple_corr import run as _sc_run
        t0 = time.perf_counter()
        out_path = args.out_dir / "simple_corr.json"
        try:
            result = _run_for_datasets(
                _sc_run, args,
                extra_kw={
                    "max_rows":  args.max_rows,
                    "mi_sample": 50_000,
                    "seed":      args.seed,
                },
                no_stream=True,
            )
            out_path.write_text(json.dumps(result, indent=2, default=str),
                                encoding="utf-8")
            elapsed = time.perf_counter() - t0
            summary["simple_corr"] = {
                "status": "ok", "out": str(out_path), "elapsed_s": round(elapsed, 1),
            }
            print(f"\n  wrote {out_path}  ({elapsed:.1f}s)")
        except Exception:
            elapsed = time.perf_counter() - t0
            summary["simple_corr"] = {
                "status": "error", "error": traceback.format_exc(),
                "elapsed_s": round(elapsed, 1),
            }
            traceback.print_exc()

    # ── cross_corr ────────────────────────────────────────────────────────────
    if "cross_corr" in only:
        _section("cross_corr  (cross_corr.py)")
        from cross_corr import run as _cc_run
        t0 = time.perf_counter()
        out_path = args.out_dir / "cross_corr.json"
        try:
            # cross_corr has no streaming equivalent: always uses parquets.
            result = _run_for_datasets(
                _cc_run, args,
                extra_kw={"max_rows": args.max_rows, "rng_seed": args.seed},
                no_stream=True,
            )
            if isinstance(result, dict) and "_meta" not in result:
                output = {
                    "_meta": {
                        "max_rows":      args.max_rows,
                        "seed":          args.seed,
                        "metrics":       ["pearson_r", "pearson_p",
                                         "spearman_r", "spearman_p", "mi"],
                        "source_groups": ["DHM", "Trees", "UA", "WIS", "LST"],
                    },
                    **result,
                }
            else:
                output = result
            out_path.write_text(json.dumps(output, indent=2, default=str),
                                encoding="utf-8")
            elapsed = time.perf_counter() - t0
            summary["cross_corr"] = {
                "status": "ok", "out": str(out_path), "elapsed_s": round(elapsed, 1),
            }
            print(f"\n  wrote {out_path}  ({elapsed:.1f}s)")
        except Exception:
            elapsed = time.perf_counter() - t0
            summary["cross_corr"] = {
                "status": "error", "error": traceback.format_exc(),
                "elapsed_s": round(elapsed, 1),
            }
            traceback.print_exc()

    # ── partial_corr ──────────────────────────────────────────────────────────
    if "partial_corr" in only:
        _section("partial_corr  (partial_corr.py)")
        from partial_corr import run as _pc_run
        t0 = time.perf_counter()
        out_path = args.out_dir / "partial_corr.json"
        try:
            output = _run_for_datasets(
                _pc_run, args,
                extra_kw={
                    "max_rows":        args.max_rows,
                    "batch_size":      args.batch_size,
                    "include_spatial": False,
                    "greening_only":   True,
                },
            )
            out_path.write_text(json.dumps(output, indent=2, default=str),
                                encoding="utf-8")
            elapsed = time.perf_counter() - t0
            # Collect per-dataset row counts from nested meta for traceability
            ds_rows = {
                ds: v.get("_meta", {}).get("total_rows_streamed")
                for ds, v in output.items() if isinstance(v, dict)
            }
            summary["partial_corr"] = {
                "status": "ok", "out": str(out_path),
                "elapsed_s": round(elapsed, 1),
                "datasets_rows": ds_rows,
            }
            print(f"wrote {out_path}")
        except Exception:
            elapsed = time.perf_counter() - t0
            summary["partial_corr"] = {
                "status": "error", "error": traceback.format_exc(),
                "elapsed_s": round(elapsed, 1),
            }
            traceback.print_exc()

    # ── morans_i ──────────────────────────────────────────────────────────────
    if "morans_i" in only:
        _section("morans_i  (morans_i.py)")
        from morans_i import run as _mi_run
        t0 = time.perf_counter()
        out_path = args.out_dir / "morans_i.json"
        try:
            output = _run_for_datasets(
                _mi_run, args,
                extra_kw={
                    "max_rows":   args.max_rows,
                    "batch_size": args.batch_size,
                    "n_perm":     args.n_perm,
                    "min_tile_n": args.min_tile_n,
                },
            )
            out_path.write_text(json.dumps(output, indent=2, default=str),
                                encoding="utf-8")
            elapsed = time.perf_counter() - t0
            ds_di = {
                ds: {
                    "delta_i_greening": v.get("delta_i_greening"),
                    "delta_i_morph":    v.get("delta_i_morph"),
                    "n_tiles":          v.get("full_model", {}).get("n_tiles"),
                }
                for ds, v in output.items() if isinstance(v, dict)
            }
            summary["morans_i"] = {
                "status": "ok", "out": str(out_path),
                "elapsed_s": round(elapsed, 1),
                "datasets_summary": ds_di,
            }
            print(f"wrote {out_path}")
        except Exception:
            elapsed = time.perf_counter() - t0
            summary["morans_i"] = {
                "status": "error", "error": traceback.format_exc(),
                "elapsed_s": round(elapsed, 1),
            }
            traceback.print_exc()

    # ── spatial_cv ────────────────────────────────────────────────────────────
    if "spatial_cv" in only:
        _section("spatial_cv  (spatial_cv.py)")
        from spatial_cv import run as _scv_run
        t0 = time.perf_counter()
        out_path = args.out_dir / "spatial_cv.json"
        try:
            output = _run_for_datasets(
                _scv_run, args,
                extra_kw={
                    "max_rows":       args.max_rows,
                    "batch_size":     args.batch_size,
                    "block_col":      args.block_col,
                    "max_test_rows":  300,
                    "ridge":          args.ridge,
                    "min_block_n":    50,
                    "n_random_folds": 10,
                },
            )
            out_path.write_text(json.dumps(output, indent=2, default=str),
                                encoding="utf-8")
            elapsed = time.perf_counter() - t0
            summary["spatial_cv"] = {
                "status": "ok", "out": str(out_path), "elapsed_s": round(elapsed, 1),
            }
            print(f"wrote {out_path}")
        except Exception:
            elapsed = time.perf_counter() - t0
            summary["spatial_cv"] = {
                "status": "error", "error": traceback.format_exc(),
                "elapsed_s": round(elapsed, 1),
            }
            traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────────
    total = time.perf_counter() - t_suite
    summary["_meta"]["total_elapsed_s"] = round(total, 1)

    summary_path = args.out_dir / "suite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str),
                            encoding="utf-8")

    _section("Suite complete")
    statuses = {k: v.get("status", "?") for k, v in summary.items() if k != "_meta"}
    for analysis, status in statuses.items():
        mark = "OK" if status == "ok" else "!!"
        print(f"  {mark}  {analysis:<15} {status}")
    print(f"\n  total: {total / 60:.1f} min")
    print(f"  summary: {summary_path}")

    errors = [k for k, v in statuses.items() if v != "ok"]
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

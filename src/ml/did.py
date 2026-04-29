"""
ml/did.py
=========
LSTDiD — staggered difference-in-differences estimator on streamed LST panel data.

Not a predictive model.  Estimates the *causal* effect of a treatment dose
(typically a planting-aware tree count within radius R) on LST, using two-way
fixed effects with cluster-robust standard errors.  Identification comes from
within-tile variation over time: each H3 tile acts as its own control.

Workflow
--------
    did = LSTDiD(
        treatment_col   = "trees_plantedby_50m_count",
        control_cols    = ["ua_vegetation_100m_frac", "ua_artificial_100m_frac"],
        event_window    = (-5, 10),
    )
    did.fit(source=cfg, registry=reg, max_rows=2_000_000)
    print(did.report())

Standalone API — does **not** inherit from LSTModel.  Most LSTModel semantics
(predict, evaluate→{rmse,r²}, scaler, transforms, SHAP, _training_history)
do not apply to a panel-FE causal estimator and would be misleading no-ops.

Dependencies
------------
Requires ``linearmodels`` for PanelOLS.  Optional: ``matplotlib`` for the
event-study plot in report(out_path=...).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

log = logging.getLogger("lst_models.did")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class DiDResult:
    """Outcome of a fitted LSTDiD."""
    att:               float
    att_se:            float
    att_ci_lo:         float
    att_ci_hi:         float
    att_p:             float
    n_obs:             int
    n_tiles:           int
    n_periods:         int
    cluster_on:        str
    treatment_col:     str
    outcome_col:       str
    control_cols:      List[str]
    # Event-study: dict {event_time_int -> (coef, se, ci_lo, ci_hi, p)}
    event_coefs:       Dict[int, Tuple[float, float, float, float, float]] = field(default_factory=dict)
    parallel_trends_p: Optional[float] = None
    # Convergence / fit diagnostics
    rsquared_within:   Optional[float] = None
    fit_seconds:       float = 0.0
    notes:             List[str]       = field(default_factory=list)


# ---------------------------------------------------------------------------
# LSTDiD
# ---------------------------------------------------------------------------
class LSTDiD:
    """
    Staggered DiD estimator with two-way fixed effects.

    Parameters
    ----------
    outcome_col : str
        Dependent variable column in the streamed batch (default
        ``"temperature"`` — the resolved LST value).
    treatment_col : str
        Treatment-dose column.  Typically the output of
        ``trees_count_planted_by(radius_m=R)`` from ``stream/features.py``,
        e.g. ``"trees_plantedby_50m_count"``.
    tile_col : str
        Per-tile identifier in the streamed batch (default ``"tile_id"`` —
        H3 res 9, ~150 m hexagons).  Acts as the cross-sectional unit.
    time_col : str | None
        Per-time identifier.  If ``None``, derived from the batch's
        ``timestamp`` column as ``YYYY-MM`` (year-month buckets).
    control_cols : list[str], optional
        Time-varying controls (e.g. UA fractions per radius).  Static
        covariates (DHM, WIS) are absorbed by the tile FE and need not
        be listed.
    event_window : tuple[int, int]
        Event-time bounds in years for the event-study spec.  Default
        ``(-5, 10)`` — 5 years pre, 10 years post.  k=-1 is the reference.
    cluster_on : str | None
        Column for cluster-robust SEs.  Defaults to *tile_col*.
    max_panel_rows : int
        Hard cap on panel size after eligibility filtering.  If exceeded,
        a **tile-stratified** subsample is taken (whole tiles, never random
        rows — random row sampling would shred per-tile counts and bias FE).
    min_obs_per_tile : int
        Tiles with fewer observations than this are dropped (singleton
        absorption; FE regression cannot identify them anyway).
    seed : int
        RNG seed for the tile-stratified subsample.
    """

    def __init__(
        self,
        outcome_col:      str = "temperature",
        treatment_col:    str = "trees_plantedby_50m_count",
        tile_col:         str = "tile_id",
        time_col:         Optional[str] = None,
        control_cols:     Optional[List[str]] = None,
        event_window:     Tuple[int, int] = (-5, 10),
        cluster_on:       Optional[str] = None,
        max_panel_rows:   int = 5_000_000,
        min_obs_per_tile: int = 3,
        seed:             int = 0,
    ) -> None:
        if event_window[0] >= event_window[1]:
            raise ValueError(f"event_window must be (lo, hi) with lo<hi, got {event_window!r}")
        self.outcome_col      = outcome_col
        self.treatment_col    = treatment_col
        self.tile_col         = tile_col
        self.time_col         = time_col
        self.control_cols     = list(control_cols or [])
        self.event_window     = event_window
        self.cluster_on       = cluster_on or tile_col
        self.max_panel_rows   = max_panel_rows
        self.min_obs_per_tile = min_obs_per_tile
        self.seed             = seed

        self._panel:    Optional[pd.DataFrame] = None
        self._result:   Optional[DiDResult]    = None
        self._fit_time: float                   = 0.0

    # ---- 1. fit -------------------------------------------------------
    def fit(
        self,
        source: Any,
        registry: Optional[Any] = None,
        batch_size: int = 100_000,
        max_rows: Optional[int] = None,
        verbose: bool = True,
    ) -> "LSTDiD":
        """
        Stream the data source, accumulate the panel, filter to eligible
        tiles, and fit the two-way FE regression.

        Returns ``self`` so calls can be chained.
        """
        t0 = time.perf_counter()

        # Accept either a StreamConfig (with .stream(reg, batch_size, max_rows))
        # or an in-memory DataFrame.
        if isinstance(source, pd.DataFrame):
            df_full = source.head(max_rows) if max_rows else source
            iterator = self._iter_dataframe(df_full, batch_size, verbose)
        else:
            iterator = self._iter_stream(source, registry, batch_size, max_rows, verbose)

        self._panel = self._collect_panel(iterator, verbose=verbose)
        if self._panel is None or self._panel.empty:
            raise RuntimeError("LSTDiD.fit: no rows collected from source")

        self._panel = self._filter_eligible(self._panel, verbose=verbose)
        self._panel = self._cap_panel_size(self._panel, verbose=verbose)

        self._result = self._fit_regressions(self._panel, verbose=verbose)
        self._fit_time = time.perf_counter() - t0
        if self._result is not None:
            self._result.fit_seconds = self._fit_time
        return self

    # ---- 1a. iterators ------------------------------------------------
    def _iter_dataframe(self, df, batch_size, verbose):
        n = len(df)
        bar = tqdm(total=n, desc="[DiD] streaming", unit="row",
                   disable=not verbose, dynamic_ncols=True)
        for start in range(0, n, batch_size):
            chunk = df.iloc[start:start + batch_size]
            yield chunk
            bar.update(len(chunk))
        bar.close()

    def _iter_stream(self, source, registry, batch_size, max_rows, verbose):
        bar = tqdm(total=max_rows, desc="[DiD] streaming", unit="row",
                   disable=(not verbose) or (max_rows is None),
                   dynamic_ncols=True)
        for chunk in source.stream(registry, batch_size=batch_size, max_rows=max_rows):
            yield chunk
            bar.update(len(chunk))
        bar.close()

    # ---- 1b. collect panel --------------------------------------------
    def _collect_panel(self, iterator, verbose: bool) -> pd.DataFrame:
        """
        Stream the panel into memory with **trimmed dtypes** so the row size
        is dominated by the model rather than pandas defaults.

        Per row: float32 outcome + float32 treatment + categorical tile_id
        (codes uint32 + small dictionary) + float32 controls + int32 _scene_ym.
        On the Ghent stream this is ~24 bytes per (outcome, treatment, time)
        triple plus 4 bytes per control — about 5× smaller than the
        float64+object default.

        PanelOLS is a single-batch fit; this pipeline cannot avoid holding the
        full panel in memory.  ``max_panel_rows`` is the user-facing safety
        valve — set ``--max-panel`` low enough to fit your machine and pass a
        large ``--rows``; the tile-stratified subsample cap fires after the
        eligibility filter.  For genuinely unbounded streams (``--rows -1``)
        the user should also set ``--max-panel`` to a memory-safe ceiling.
        """
        keep = [self.outcome_col, self.treatment_col, self.tile_col,
                *self.control_cols]
        # Need a time key — derive from "timestamp" if time_col is None.
        derive_time = self.time_col is None
        parts: List[pd.DataFrame] = []
        n_total = 0
        for chunk in iterator:
            cols_present = [c for c in keep if c in chunk.columns]
            missing = set(keep) - set(cols_present)
            if missing:
                raise KeyError(f"LSTDiD: required columns missing from stream: {sorted(missing)}")

            sub = chunk.loc[:, cols_present].copy()
            # Cast numeric panel columns to float32 (8 → 4 bytes/value)
            for c in (self.outcome_col, self.treatment_col, *self.control_cols):
                if c in sub.columns:
                    sub[c] = pd.to_numeric(sub[c], errors="coerce").astype("float32")
            if derive_time:
                if "timestamp" not in chunk.columns:
                    raise KeyError("LSTDiD: time_col=None requires 'timestamp' in stream")
                ts = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=True)
                # int32 holds year*12+month for any reasonable epoch
                sub["_scene_ym"] = (
                    ts.dt.year.astype("Int32") * 12 + ts.dt.month.astype("Int32")
                ).astype("Int32")
            else:
                if self.time_col not in chunk.columns:
                    raise KeyError(f"LSTDiD: time_col {self.time_col!r} not in stream")
                sub["_scene_ym"] = chunk[self.time_col].astype("Int32")
            parts.append(sub)
            n_total += len(sub)

        if not parts:
            return pd.DataFrame()
        panel = pd.concat(parts, ignore_index=True)
        panel = panel.dropna(subset=[self.outcome_col, self.treatment_col,
                                     self.tile_col, "_scene_ym"])
        # tile_id as categorical: groupby/isin operate on int32 codes, dramatic
        # memory savings when the same H3 cell appears thousands of times.
        if self.tile_col in panel.columns:
            panel[self.tile_col] = panel[self.tile_col].astype("category")
        if verbose:
            log.info("DiD panel: %d rows after dropna (was %d)", len(panel), n_total)
        return panel

    # ---- 1c. eligibility filter ---------------------------------------
    def _filter_eligible(self, panel: pd.DataFrame, verbose: bool) -> pd.DataFrame:
        """
        Drop tiles where treatment was never > 0 across the panel — they
        contribute no within-tile variation to the treatment coefficient.
        Then drop tiles below min_obs_per_tile (singleton absorption).
        """
        n_in = len(panel)

        ever_treated = panel.groupby(self.tile_col)[self.treatment_col].max()
        keep_tiles = ever_treated[ever_treated > 0].index
        panel = panel[panel[self.tile_col].isin(keep_tiles)]

        counts = panel.groupby(self.tile_col).size()
        keep_tiles2 = counts[counts >= self.min_obs_per_tile].index
        panel = panel[panel[self.tile_col].isin(keep_tiles2)]

        if verbose:
            log.info(
                "DiD eligibility: %d tiles ever-treated, %d tiles with >=%d obs, "
                "%d rows kept (was %d)",
                len(keep_tiles), len(keep_tiles2),
                self.min_obs_per_tile, len(panel), n_in,
            )
        return panel.reset_index(drop=True)

    # ---- 1d. tile-stratified subsample --------------------------------
    def _cap_panel_size(self, panel: pd.DataFrame, verbose: bool) -> pd.DataFrame:
        if len(panel) <= self.max_panel_rows:
            return panel
        rng = np.random.default_rng(self.seed)
        tiles = panel[self.tile_col].drop_duplicates().to_numpy()
        rng.shuffle(tiles)

        sizes = panel.groupby(self.tile_col).size().to_dict()
        kept: List[Any] = []
        running = 0
        for t in tiles:
            running += sizes[t]
            kept.append(t)
            if running >= self.max_panel_rows:
                break
        kept_set = set(kept)
        out = panel[panel[self.tile_col].isin(kept_set)].reset_index(drop=True)
        if verbose:
            log.info(
                "DiD subsample: tile-stratified down to %d tiles / %d rows "
                "(cap=%d, full=%d)",
                len(kept_set), len(out), self.max_panel_rows, len(panel),
            )
        return out

    # ---- 1e. regression -----------------------------------------------
    def _fit_regressions(self, panel: pd.DataFrame, verbose: bool) -> DiDResult:
        try:
            from linearmodels.panel import PanelOLS
        except ImportError as exc:
            raise ImportError(
                "LSTDiD requires the 'linearmodels' package.  "
                "Install with: pip install linearmodels"
            ) from exc

        # Build a (tile, time) MultiIndex panel.
        df = panel.copy()
        df = df.set_index([self.tile_col, "_scene_ym"]).sort_index()

        notes: List[str] = []

        # ATT regression: outcome ~ treatment + controls + entity + time FE.
        exog_cols = [self.treatment_col, *self.control_cols]
        y    = df[self.outcome_col].astype("float64")
        exog = df[exog_cols].astype("float64")

        mod = PanelOLS(
            dependent      = y,
            exog           = exog,
            entity_effects = True,
            time_effects   = True,
            drop_absorbed  = True,
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        att   = float(res.params[self.treatment_col])
        att_se = float(res.std_errors[self.treatment_col])
        att_ci = res.conf_int().loc[self.treatment_col]
        att_p  = float(res.pvalues[self.treatment_col])
        rsq    = float(res.rsquared_within) if hasattr(res, "rsquared_within") else None

        # Event-study spec: outcome ~ Σ 1{k}·event_time_dummies + controls + FEs.
        # First-treatment year per tile (smallest scene year where treatment > 0).
        event_coefs: Dict[int, Tuple[float, float, float, float, float]] = {}
        parallel_p: Optional[float] = None
        try:
            event_coefs, parallel_p = self._fit_event_study(panel, verbose=verbose)
        except Exception as exc:
            notes.append(f"event-study failed: {exc}")
            log.warning("LSTDiD event-study failed: %s", exc, exc_info=True)

        return DiDResult(
            att              = att,
            att_se           = att_se,
            att_ci_lo        = float(att_ci.iloc[0]),
            att_ci_hi        = float(att_ci.iloc[1]),
            att_p            = att_p,
            n_obs            = int(res.nobs),
            n_tiles          = int(df.index.get_level_values(0).nunique()),
            n_periods        = int(df.index.get_level_values(1).nunique()),
            cluster_on       = self.cluster_on,
            treatment_col    = self.treatment_col,
            outcome_col      = self.outcome_col,
            control_cols     = list(self.control_cols),
            event_coefs      = event_coefs,
            parallel_trends_p= parallel_p,
            rsquared_within  = rsq,
            notes            = notes,
        )

    # ---- 1f. event study ----------------------------------------------
    def _fit_event_study(
        self,
        panel: pd.DataFrame,
        verbose: bool,
    ) -> Tuple[Dict[int, Tuple[float, float, float, float, float]], Optional[float]]:
        from linearmodels.panel import PanelOLS

        # Treatment-onset year per tile = scene_year of first scene where
        # treatment > 0.  Tiles never above 0 are excluded by _filter_eligible.
        df = panel.copy()
        df["_scene_year"] = df["_scene_ym"] // 12

        treated_mask = df[self.treatment_col] > 0
        first_year = (
            df[treated_mask]
            .groupby(self.tile_col)["_scene_year"]
            .min()
            .rename("_treat_year")
        )
        df = df.merge(first_year, left_on=self.tile_col, right_index=True, how="left")
        # Drop tiles without a finite onset (shouldn't happen post-filter,
        # but guard anyway).
        df = df.dropna(subset=["_treat_year"])
        df["_treat_year"] = df["_treat_year"].astype("int64")
        df["_event_time"] = df["_scene_year"].astype("int64") - df["_treat_year"]

        lo, hi = self.event_window
        # Bin extremes into the endpoint dummies so observations outside the
        # window still contribute to FE estimation.
        df["_k_bin"] = df["_event_time"].clip(lower=lo, upper=hi)

        # k = -1 is the reference period — drop that dummy.
        ref = -1
        ks  = [k for k in range(lo, hi + 1) if k != ref]

        # Build the dummy matrix.
        dummies = pd.DataFrame(
            {f"k_{k}": (df["_k_bin"] == k).astype("float64").values for k in ks},
            index=df.index,
        )
        controls = df[self.control_cols].astype("float64") if self.control_cols else None
        exog = pd.concat([dummies, controls], axis=1) if controls is not None else dummies

        df_idx = df.set_index([self.tile_col, "_scene_ym"]).sort_index()
        exog_idx = exog
        exog_idx.index = df_idx.index
        y = df_idx[self.outcome_col].astype("float64")

        mod = PanelOLS(
            dependent      = y,
            exog           = exog_idx,
            entity_effects = True,
            time_effects   = True,
            drop_absorbed  = True,
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True)

        out: Dict[int, Tuple[float, float, float, float, float]] = {}
        ci = res.conf_int()
        for k in ks:
            name = f"k_{k}"
            if name not in res.params.index:
                continue
            out[k] = (
                float(res.params[name]),
                float(res.std_errors[name]),
                float(ci.loc[name].iloc[0]),
                float(ci.loc[name].iloc[1]),
                float(res.pvalues[name]),
            )

        # Parallel-trends test: joint Wald that all pre-period coefs (k < -1)
        # are zero.  k = -1 is the reference and not in the regression.
        pre_keys = [f"k_{k}" for k in ks if k < ref]
        parallel_p: Optional[float] = None
        if pre_keys:
            try:
                hypothesis = " = ".join(pre_keys + ["0"])
                wald = res.wald_test(formula=hypothesis)
                parallel_p = float(wald.pval)
            except Exception as exc:
                log.debug("parallel-trends Wald test failed: %s", exc)

        return out, parallel_p

    # ---- 2. summary ---------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        if self._result is None:
            return {"fitted": False}
        r = self._result
        return {
            "fitted":            True,
            "att":               r.att,
            "att_se":            r.att_se,
            "att_ci":            (r.att_ci_lo, r.att_ci_hi),
            "att_p":             r.att_p,
            "n_obs":             r.n_obs,
            "n_tiles":           r.n_tiles,
            "n_periods":         r.n_periods,
            "treatment_col":     r.treatment_col,
            "outcome_col":       r.outcome_col,
            "control_cols":      r.control_cols,
            "cluster_on":        r.cluster_on,
            "rsquared_within":   r.rsquared_within,
            "parallel_trends_p": r.parallel_trends_p,
            "event_coefs":       r.event_coefs,
            "fit_seconds":       r.fit_seconds,
            "notes":             r.notes,
        }

    # ---- 3. report ----------------------------------------------------
    def report(self, out_path: Optional[Union[str, Path]] = None) -> str:
        """
        Render a human-readable report.

        If *out_path* points to an .html file and matplotlib is available,
        also writes an embedded event-study plot.  The text report is always
        returned as the function's value and is suitable for ``print()``.
        """
        if self._result is None:
            return "(LSTDiD not yet fitted)"
        r = self._result

        bar = "─" * 78
        lines: List[str] = []
        lines.append(bar)
        lines.append(f"  LSTDiD — staggered difference-in-differences")
        lines.append(bar)
        lines.append(f"  outcome      : {r.outcome_col}")
        lines.append(f"  treatment    : {r.treatment_col}")
        lines.append(f"  controls     : {', '.join(r.control_cols) if r.control_cols else '(none)'}")
        lines.append(f"  cluster on   : {r.cluster_on}")
        lines.append(f"  panel        : {r.n_obs:,} obs  ×  "
                     f"{r.n_tiles:,} tiles  ×  {r.n_periods} periods")
        if r.rsquared_within is not None:
            lines.append(f"  R² within    : {r.rsquared_within:.4f}")
        lines.append(f"  fit time     : {r.fit_seconds:.1f}s")
        lines.append("")
        lines.append(f"  ATT (per +1 unit of treatment dose):")
        lines.append(f"    estimate   : {r.att:+.4f}")
        lines.append(f"    std. error : {r.att_se:.4f}")
        lines.append(f"    95% CI     : [{r.att_ci_lo:+.4f}, {r.att_ci_hi:+.4f}]")
        lines.append(f"    p-value    : {r.att_p:.4g}")
        if r.parallel_trends_p is not None:
            verdict = ("OK" if r.parallel_trends_p > 0.10 else
                       "WARN — pre-trends present" if r.parallel_trends_p > 0.01 else
                       "FAIL — strong pre-trends")
            lines.append("")
            lines.append(f"  parallel-trends Wald p = {r.parallel_trends_p:.3g}  →  {verdict}")
        lines.append("")
        if r.event_coefs:
            lines.append("  Event-study coefficients (k = years since planting; k=-1 reference):")
            lines.append(f"    {'k':>4}  {'coef':>10}  {'se':>10}  {'95% CI':>22}  {'p':>10}")
            lines.append(f"    {'-'*4}  {'-'*10}  {'-'*10}  {'-'*22}  {'-'*10}")
            for k in sorted(r.event_coefs.keys()):
                coef, se, lo, hi, p = r.event_coefs[k]
                ci_str = f"[{lo:+.3f}, {hi:+.3f}]"
                lines.append(f"    {k:>4d}  {coef:>+10.4f}  {se:>10.4f}  "
                             f"{ci_str:>22}  {p:>10.3g}")
        if r.notes:
            lines.append("")
            lines.append("  notes:")
            for n in r.notes:
                lines.append(f"    - {n}")
        lines.append(bar)
        text = "\n".join(lines)

        if out_path is not None:
            self._write_html(Path(out_path), text)
        return text

    # ---- 3a. optional html ---------------------------------------------
    def _write_html(self, path: Path, text: str) -> None:
        """Write a minimal HTML report with an event-study plot if possible."""
        path.parent.mkdir(parents=True, exist_ok=True)
        plot_html = ""
        try:
            import io
            import base64
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            r = self._result
            if r is not None and r.event_coefs:
                ks    = sorted(r.event_coefs.keys())
                coefs = [r.event_coefs[k][0] for k in ks]
                los   = [r.event_coefs[k][2] for k in ks]
                his   = [r.event_coefs[k][3] for k in ks]
                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.errorbar(ks, coefs,
                            yerr=[np.array(coefs) - np.array(los),
                                  np.array(his) - np.array(coefs)],
                            fmt="o", capsize=3)
                ax.axhline(0, linewidth=1, color="grey")
                ax.axvline(-0.5, linewidth=1, linestyle="--", color="grey")
                ax.set_xlabel("years since planting (k=-1 reference)")
                ax.set_ylabel(f"Δ {r.outcome_col}")
                ax.set_title("Event-study coefficients (95% CI)")
                fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=120)
                plt.close(fig)
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                plot_html = f'<img src="data:image/png;base64,{b64}" />'
        except Exception as exc:
            log.warning("LSTDiD: event-study plot skipped — %s", exc)

        path.write_text(
            "<html><head><meta charset='utf-8'>"
            "<title>LSTDiD report</title></head><body>"
            f"<pre>{text}</pre>{plot_html}"
            "</body></html>",
            encoding="utf-8",
        )

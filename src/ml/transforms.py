"""
ml/transforms.py
================
Vectorised feature transform factories for the LST preprocessing pipeline.

All factories return a ``Transform`` object — a callable that also carries
metadata (input columns, output columns, drop-inputs flag, skip-models set).
The ``Transform`` is callable as ``fn(df) → df``, adding output columns to df
(and optionally dropping input columns).

Column naming
-------------
Output columns are named ``{input}{symbol}`` where ``symbol`` is a short
operator string chosen per transform:
    func_transform(cols, np.sqrt)       → ``{col}√``
    func_transform(cols, np.log1p)      → ``{col}㏒``   (or supply symbol="㏒")
    cyclical("month", 12)               → ``month_sin``, ``month_cos``
    interaction_terms([("a","b")])      → ``a×b``
    delta("a", "b")                     → ``a−b``
    ratio("a", "b")                     → ``a÷b``
    polynomial_features(cols, degree=2) → sklearn names (``a^2``, ``a b``, …)

Chaining: because output columns are merged into the running DataFrame,
a later transform can reference outputs of an earlier one:
    [
        func_transform(["elevation"], np.log1p, symbol="㏒"),
        func_transform(["elevation㏒"], np.sqrt, symbol="√"),
    ]

Skip-models: pass ``skip_models={"Random Forest", "Extra Trees"}`` to prevent
a transform being applied to specific model types (matched against model_name).
Tree models are scale- and monotone-transform-invariant, so scaling transforms
like rolling_zscore should be skipped for them.

drop_inputs: set ``drop_inputs=True`` so that source columns are dropped
immediately after the transform is applied.  This is the right knob when you
want to drop an *intermediate* column produced by an earlier transform.

Global drop-raw-inputs rule
---------------------------
Independent of ``drop_inputs``, the pipeline (see ml.base.LSTModel
._apply_transforms) drops every *raw* column that any registered transform
consumed, after the full transform list has run.  Net effect: as soon as you
register at least one transform on column ``A``, raw ``A`` no longer reaches
the model — only the transform outputs do.  Other columns pass through
unchanged.  To drop a column without computing a replacement, use
``remove(cols)``.

Universal scalar factory
------------------------
Most single-column scalar transforms are one-liners via ``func_transform``:

    func_transform(["elevation"], np.sqrt)          # √elevation
    func_transform(["elevation"], np.log1p)         # log(1+elevation)
    func_transform(["elevation"], lambda x: x**2)   # elevation²

The dedicated ``log1p_transform``, ``sqrt_transform`` etc. remain as
convenience wrappers for readability, but internally call ``func_transform``.
"""

from __future__ import annotations

import math
from typing import Callable, Collection, FrozenSet, Optional, Sequence, Set, Union

import numpy as np
import pandas as pd

__all__ = [
    "Transform",
    "func_transform",
    "cyclical",
    "interaction_terms",
    "delta",
    "ratio",
    "polynomial_features",
    "clip_outliers",
    "rolling_zscore",
    "remove",
    # Convenience wrappers
    "log1p_transform",
    "sqrt_transform",
    "difference",       # alias for delta
]


# ---------------------------------------------------------------------------
# Transform wrapper
# ---------------------------------------------------------------------------

class Transform:
    """
    A callable transform with attached metadata.

    Parameters
    ----------
    fn           : core function (df → DataFrame of new columns)
    input_cols   : source column names (may be empty for multi-col transforms)
    output_cols  : output column names produced (may be determined lazily)
    symbol       : short string used in output naming and __name__
    drop_inputs  : when True, source columns are removed after the transform
    skip_models  : set of model_name strings that should skip this transform
    name         : human-readable label (shown in reports)
    """

    def __init__(
        self,
        fn:          Callable,
        input_cols:  Sequence[str],
        symbol:      str,
        drop_inputs: bool = False,
        skip_models: Optional[Collection[str]] = None,
        name:        Optional[str] = None,
    ) -> None:
        self._fn         = fn
        self.input_cols  = list(input_cols)
        self.symbol      = symbol
        self.drop_inputs = drop_inputs
        self.skip_models: FrozenSet[str] = frozenset(skip_models or [])
        self.__name__    = name or f"Transform({symbol})"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transform: returns a DataFrame of new columns only."""
        return self._fn(df)

    def should_skip(self, model_name: str) -> bool:
        """Return True if this transform should be skipped for the given model."""
        return model_name in self.skip_models

    def __repr__(self) -> str:
        base = self.__name__
        extras = []
        if self.drop_inputs:
            extras.append("drop_inputs")
        if self.skip_models:
            extras.append(f"skip={set(self.skip_models)}")
        return f"Transform({base}{', ' + ', '.join(extras) if extras else ''})"


# ---------------------------------------------------------------------------
# Universal scalar factory
# ---------------------------------------------------------------------------

def func_transform(
    cols:        Sequence[str],
    fn:          Callable[[np.ndarray], np.ndarray],
    symbol:      Optional[str] = None,
    drop_inputs: bool = False,
    skip_models: Optional[Collection[str]] = None,
) -> Transform:
    """
    Apply any vectorised numpy-compatible scalar function to one or more columns.

    Output columns are named ``{col}{symbol}``.  When ``symbol`` is None,
    the function's ``__name__`` is used (e.g. ``"sqrt"`` for ``np.sqrt``).

    Parameters
    ----------
    cols        : column names to transform
    fn          : function applied element-wise, e.g. np.sqrt, np.log1p,
                  np.exp, lambda x: x**2
    symbol      : short operator label appended to column names.
                  Defaults to fn.__name__ wrapped in braces: ``{sqrt}``.
    drop_inputs : if True, remove source columns after transform
    skip_models : model names (model_name strings) that should skip this transform

    Examples
    --------
    func_transform(["elevation"], np.sqrt)
        → column "elevation√" (pass symbol="√" explicitly or accept default)

    func_transform(["elevation"], np.log1p, symbol="㏒")
        → column "elevation㏒"

    func_transform(["count"], lambda x: x**2, symbol="²")
        → column "count²"
    """
    if symbol is None:
        fn_name = getattr(fn, "__name__", "fn")
        symbol  = f"{{{fn_name}}}"

    def _fn(df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in cols:
            if c in df.columns:
                v = df[c].to_numpy(dtype=np.float64)
                out[f"{c}{symbol}"] = fn(v)
        return pd.DataFrame(out, index=df.index)

    col_str  = ", ".join(cols)
    t = Transform(
        fn          = _fn,
        input_cols  = list(cols),
        symbol      = symbol,
        drop_inputs = drop_inputs,
        skip_models = skip_models,
        name        = f"func({col_str}, {symbol})",
    )
    return t


# ---------------------------------------------------------------------------
# Cyclical encoding
# ---------------------------------------------------------------------------

def cyclical(
    col:         str,
    period:      float,
    drop_inputs: bool = False,
    skip_models: Optional[Collection[str]] = None,
) -> Transform:
    """
    Encode a periodic column as (sin, cos) pair preserving circular topology.

    Output columns: ``{col}_sin``, ``{col}_cos``.

    Parameters
    ----------
    col         : source column (e.g. "month_of_year", "hour_of_day")
    period      : full cycle length (12 for months, 24 for hours, 365 for DOY)
    drop_inputs : remove source column after encoding
    skip_models : model names to skip

    Example
    -------
    cyclical("month_of_year", 12)  →  month_of_year_sin, month_of_year_cos
    """
    def _fn(df: pd.DataFrame) -> pd.DataFrame:
        if col not in df.columns:
            return pd.DataFrame(index=df.index)
        v = df[col].to_numpy(dtype=np.float64)
        return pd.DataFrame({
            f"{col}_sin": np.sin(2 * np.pi * v / period),
            f"{col}_cos": np.cos(2 * np.pi * v / period),
        }, index=df.index)

    return Transform(
        fn          = _fn,
        input_cols  = [col],
        symbol      = "~cyc",
        drop_inputs = drop_inputs,
        skip_models = skip_models,
        name        = f"cyclical({col}, p={period})",
    )


# ---------------------------------------------------------------------------
# Multi-column transforms
# ---------------------------------------------------------------------------

def interaction_terms(
    col_pairs:   Sequence[tuple],
    drop_inputs: bool = False,
    skip_models: Optional[Collection[str]] = None,
) -> Transform:
    """
    Multiplicative interaction between column pairs.

    Output columns: ``{col_a}×{col_b}``.

    Parameters
    ----------
    col_pairs   : list of (col_a, col_b) tuples
    drop_inputs : remove both source columns after computing interaction
    skip_models : model names to skip

    Example
    -------
    interaction_terms([("hour_of_day", "month_of_year")])
        → column "hour_of_day×month_of_year"
    """
    all_inputs = [c for pair in col_pairs for c in pair]

    def _fn(df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for a, b in col_pairs:
            if a in df.columns and b in df.columns:
                out[f"{a}×{b}"] = (
                    df[a].to_numpy(dtype=np.float64) *
                    df[b].to_numpy(dtype=np.float64)
                )
        return pd.DataFrame(out, index=df.index)

    pairs_str = ", ".join(f"{a}×{b}" for a, b in col_pairs)
    return Transform(
        fn          = _fn,
        input_cols  = all_inputs,
        symbol      = "×",
        drop_inputs = drop_inputs,
        skip_models = skip_models,
        name        = f"interaction({pairs_str})",
    )


def delta(
    col_a:       str,
    col_b:       str,
    drop_inputs: bool = False,
    skip_models: Optional[Collection[str]] = None,
) -> Transform:
    """
    Signed difference col_a − col_b.

    Output column: ``{col_a}−{col_b}``.

    Useful for lapse-rate corrections, anomaly features, etc.
    """
    def _fn(df: pd.DataFrame) -> pd.DataFrame:
        if col_a not in df.columns or col_b not in df.columns:
            return pd.DataFrame(index=df.index)
        return pd.DataFrame(
            {f"{col_a}−{col_b}":
             df[col_a].to_numpy(dtype=np.float64) -
             df[col_b].to_numpy(dtype=np.float64)},
            index=df.index,
        )

    return Transform(
        fn          = _fn,
        input_cols  = [col_a, col_b],
        symbol      = "−",
        drop_inputs = drop_inputs,
        skip_models = skip_models,
        name        = f"delta({col_a}−{col_b})",
    )


# Backwards-compatible alias
def difference(col_a: str, col_b: str, **kw) -> Transform:
    """Alias for delta()."""
    return delta(col_a, col_b, **kw)


def ratio(
    numerator:   str,
    denominator: str,
    epsilon:     float = 1e-6,
    drop_inputs: bool = False,
    skip_models: Optional[Collection[str]] = None,
) -> Transform:
    """
    Compute numerator ÷ (denominator + epsilon).

    Output column: ``{numerator}÷{denominator}``.
    """
    def _fn(df: pd.DataFrame) -> pd.DataFrame:
        if numerator not in df.columns or denominator not in df.columns:
            return pd.DataFrame(index=df.index)
        num = df[numerator].to_numpy(dtype=np.float64)
        den = df[denominator].to_numpy(dtype=np.float64)
        return pd.DataFrame(
            {f"{numerator}÷{denominator}": num / (den + epsilon)},
            index=df.index,
        )

    return Transform(
        fn          = _fn,
        input_cols  = [numerator, denominator],
        symbol      = "÷",
        drop_inputs = drop_inputs,
        skip_models = skip_models,
        name        = f"ratio({numerator}÷{denominator})",
    )


def polynomial_features(
    cols:             Sequence[str],
    degree:           int = 2,
    interaction_only: bool = False,
    include_bias:     bool = False,
    drop_inputs:      bool = False,
    skip_models:      Optional[Collection[str]] = None,
) -> Transform:
    """
    Expand cols to polynomial / interaction features via sklearn.

    The sklearn PolynomialFeatures transformer is fitted lazily on the first
    batch and reused thereafter.

    Output column names follow sklearn convention: ``a^2``, ``a b``, etc.

    Note: degree ≥ 3 on many columns produces large feature sets.
    Keep len(cols) ≤ 6 when degree ≥ 3.
    """
    from sklearn.preprocessing import PolynomialFeatures as _PF

    _pf     = _PF(degree=degree, interaction_only=interaction_only,
                  include_bias=include_bias)
    _fitted = [False]

    def _fn(df: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in cols if c in df.columns]
        if not present:
            return pd.DataFrame(index=df.index)
        X = df[present].to_numpy(dtype=np.float64)
        if not _fitted[0]:
            _pf.fit(X)
            _fitted[0] = True
        Xt    = _pf.transform(X)
        names = _pf.get_feature_names_out(present)
        return pd.DataFrame(Xt, columns=names, index=df.index)

    return Transform(
        fn          = _fn,
        input_cols  = list(cols),
        symbol      = f"^{degree}",
        drop_inputs = drop_inputs,
        skip_models = skip_models,
        name        = f"poly(d={degree}, {', '.join(cols)})",
    )


# ---------------------------------------------------------------------------
# Column removal
# ---------------------------------------------------------------------------

def remove(
    cols:        Union[str, Sequence[str]],
    skip_models: Optional[Collection[str]] = None,
) -> Transform:
    """
    Drop one or more columns without producing a transformed replacement.

    Useful for testing — register ``remove("foo")`` to confirm a model still
    works without column ``foo``, or to strip an intermediate column produced
    by an earlier transform.

    Behaviour
    ---------
    The transform produces no output columns.  Its inputs are dropped both
    immediately (via ``drop_inputs=True``, which handles intermediate columns
    not present in the original DataFrame) and, redundantly, by the global
    drop-raw-inputs rule applied at the end of the pipeline.
    """
    if isinstance(cols, str):
        cols = [cols]
    cols = list(cols)

    def _fn(df: pd.DataFrame) -> pd.DataFrame:
        # No outputs: returning an empty DataFrame keeps the pipeline contract
        # (Transform → DataFrame of new columns) while signalling "nothing
        # added".  Actual removal happens via drop_inputs / global rule.
        return pd.DataFrame(index=df.index)

    return Transform(
        fn          = _fn,
        input_cols  = cols,
        symbol      = "ø",
        drop_inputs = True,
        skip_models = skip_models,
        name        = f"remove({', '.join(cols)})",
    )


# ---------------------------------------------------------------------------
# Outlier handling
# ---------------------------------------------------------------------------

def clip_outliers(
    cols:        Sequence[str],
    low_pct:     float = 1.0,
    high_pct:    float = 99.0,
    drop_inputs: bool = False,
    skip_models: Optional[Collection[str]] = None,
) -> Transform:
    """
    Clip each column to [low_pct, high_pct] percentiles.

    Percentiles are estimated from the first batch and held constant.
    Output columns: ``{col}⌈⌋`` (clipped symbol).
    """
    _bounds: dict = {}

    def _fn(df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in cols:
            if c not in df.columns:
                continue
            v = df[c].to_numpy(dtype=np.float64)
            if c not in _bounds:
                finite = v[np.isfinite(v)]
                _bounds[c] = (
                    (float(np.percentile(finite, low_pct)),
                     float(np.percentile(finite, high_pct)))
                    if len(finite) > 0 else (-np.inf, np.inf)
                )
            lo, hi = _bounds[c]
            out[f"{c}⌈⌋"] = np.clip(v, lo, hi)
        return pd.DataFrame(out, index=df.index)

    return Transform(
        fn          = _fn,
        input_cols  = list(cols),
        symbol      = "⌈⌋",
        drop_inputs = drop_inputs,
        skip_models = skip_models,
        name        = f"clip({', '.join(cols)}, [{low_pct},{high_pct}]%)",
    )


# ---------------------------------------------------------------------------
# Running z-score (for when StreamingScaler is not configured)
# ---------------------------------------------------------------------------

def rolling_zscore(
    cols:        Sequence[str],
    drop_inputs: bool = False,
    skip_models: Optional[Collection[str]] = None,
) -> Transform:
    """
    Online z-score standardisation via Welford's algorithm.

    Output columns: ``{col}z``.

    Only use this when a StreamingScaler is NOT configured — it runs inside
    the transform pipeline and is included in save/load via cloudpickle.
    Tree models are scale-invariant; pass
    ``skip_models={"Random Forest", "Extra Trees"}`` accordingly.
    """
    _n:    dict = {c: 0   for c in cols}
    _mean: dict = {c: 0.0 for c in cols}
    _M2:   dict = {c: 0.0 for c in cols}

    def _update(c, x):
        for xi in x:
            _n[c]    += 1
            delta     = xi - _mean[c]
            _mean[c] += delta / _n[c]
            _M2[c]   += delta * (xi - _mean[c])

    def _fn(df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in cols:
            if c not in df.columns:
                continue
            v = df[c].to_numpy(dtype=np.float64)
            _update(c, v[np.isfinite(v)])
            std = math.sqrt(_M2[c] / max(_n[c] - 1, 1))
            out[f"{c}z"] = (v - _mean[c]) / (std + 1e-9)
        return pd.DataFrame(out, index=df.index)

    return Transform(
        fn          = _fn,
        input_cols  = list(cols),
        symbol      = "z",
        drop_inputs = drop_inputs,
        skip_models = skip_models,
        name        = f"zscore({', '.join(cols)})",
    )


# ---------------------------------------------------------------------------
# Convenience wrappers (thin shims over func_transform for readability)
# ---------------------------------------------------------------------------

def log1p_transform(
    cols:        Sequence[str],
    clip_min:    float = 0.0,
    drop_inputs: bool = False,
    skip_models: Optional[Collection[str]] = None,
) -> Transform:
    """log(1 + max(x, clip_min)) — output columns: ``{col}㏒``."""
    def _fn_inner(v: np.ndarray) -> np.ndarray:
        return np.log1p(np.maximum(v, clip_min))

    _fn_inner.__name__ = "㏒"
    return func_transform(cols, _fn_inner, symbol="㏒",
                          drop_inputs=drop_inputs, skip_models=skip_models)


def sqrt_transform(
    cols:        Sequence[str],
    clip_min:    float = 0.0,
    drop_inputs: bool = False,
    skip_models: Optional[Collection[str]] = None,
) -> Transform:
    """sqrt(max(x, clip_min)) — output columns: ``{col}√``."""
    def _fn_inner(v: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(v, clip_min))

    _fn_inner.__name__ = "√"
    return func_transform(cols, _fn_inner, symbol="√",
                          drop_inputs=drop_inputs, skip_models=skip_models)
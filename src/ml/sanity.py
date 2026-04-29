"""
sanity.py — opt-in pipeline diagnostics for the streaming ML training loop.

Catches the kinds of silent failures that produce 1e19-scale R² scores:
  1. Feature columns missing or unfitted in the StreamingScaler.
  2. Per-batch X / y containing NaN, Inf, or extreme magnitudes (likely
     unscaled cyclical / sentinel values).
  3. Linear-model coefficients diverging within the first few SGD steps.
  4. Predictions blowing up in evaluate() / predict().

All checks are no-ops unless explicitly enabled via `enable()` (or by setting
the LST_SANITY env var).  Default mode logs a warning/error per failure;
strict mode raises SanityCheckFailure instead, which is what the dedicated
sanity test script wants so it can fail loudly.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence

import numpy as np

log = logging.getLogger("lst_models.sanity")


# ---------------------------------------------------------------------------
# Module-level toggle
# ---------------------------------------------------------------------------
ENABLED: bool = os.environ.get("LST_SANITY", "").lower() in ("1", "true", "yes")
STRICT:  bool = os.environ.get("LST_SANITY_STRICT", "").lower() in ("1", "true", "yes")


class SanityCheckFailure(RuntimeError):
    """Raised in strict mode when a check fails."""


def enable(strict: bool = False) -> None:
    """Turn sanity checks on for the rest of the process."""
    global ENABLED, STRICT
    ENABLED, STRICT = True, strict
    log.info("sanity checks enabled (strict=%s)", strict)


def disable() -> None:
    global ENABLED, STRICT
    ENABLED, STRICT = False, False


def is_enabled() -> bool:
    return ENABLED


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------
def _report(level: str, label: str, msg: str) -> None:
    full = f"[sanity:{label}] {msg}"
    if STRICT and level == "error":
        raise SanityCheckFailure(full)
    if level == "error":
        log.error(full)
    elif level == "warn":
        log.warning(full)
    else:
        log.info(full)


# ---------------------------------------------------------------------------
# 1. Scaler / feature-column alignment
# ---------------------------------------------------------------------------
def check_scaler_alignment(
    scaler: Any,
    cols: Sequence[str],
    model_name: str = "",
) -> None:
    """
    Verify that every active feature column has a fitted per-column scaler.

    Most divergence cases trace back to a feature that the scaler never saw
    (e.g. transform output column added after the scaler was fitted), leaving
    raw integer-valued columns flowing into SGD with squared loss.
    """
    if not ENABLED or scaler is None:
        return
    fitted = getattr(scaler, "_scalers", {}) or {}

    def _has_stats(s: Any) -> bool:
        return any(hasattr(s, a) for a in ("mean_", "data_min_", "max_abs_"))

    missing  = [c for c in cols if c not in fitted]
    unfitted = [c for c in cols
                if c in fitted and fitted[c] is not None and not _has_stats(fitted[c])]
    n_features = len(cols)
    n_seen = sum(1 for c in cols
                 if (s := fitted.get(c)) is not None and _has_stats(s))

    if missing or unfitted:
        head = lambda xs: f"{xs[:5]}{'…' if len(xs) > 5 else ''}"
        _report(
            "error", f"scaler/{model_name}",
            f"feature/scaler mismatch: {n_features} feature columns, "
            f"{n_seen} fitted in scaler.  "
            f"missing={head(missing)}  unfitted={head(unfitted)}"
        )
    else:
        _report("info", f"scaler/{model_name}",
                f"OK: {n_seen}/{n_features} feature columns are fitted in scaler")


# ---------------------------------------------------------------------------
# 2. Per-batch input array sanity
# ---------------------------------------------------------------------------
def check_input_batch(
    X: np.ndarray,
    y: np.ndarray,
    cols: Sequence[str],
    batch_idx: int,
    model_name: str = "",
    *,
    max_abs_x: float = 50.0,
    max_abs_y: float = 1e3,
    sample_zero_var_on_first_batch: bool = True,
) -> None:
    """
    After scaling, X should be roughly N(0,1)-ish and y in physical units.
    Anything wildly outside those ranges, or any non-finite value, is the
    typical precursor to a divergent SGD weight update.
    """
    if not ENABLED or X.size == 0:
        return
    label = f"batch/{model_name}"

    if not np.isfinite(X).all():
        bad = int((~np.isfinite(X)).sum())
        _report("error", label, f"batch {batch_idx}: X has {bad} non-finite values")
    if not np.isfinite(y).all():
        bad = int((~np.isfinite(y)).sum())
        _report("error", label, f"batch {batch_idx}: y has {bad} non-finite values")

    abs_x = np.abs(X)
    x_max = float(abs_x.max())
    if x_max > max_abs_x:
        col_max = abs_x.max(axis=0)
        i = int(np.argmax(col_max))
        col = cols[i] if i < len(cols) else f"col[{i}]"
        _report("error", label,
                f"batch {batch_idx}: max |X|={x_max:.2f} > {max_abs_x} "
                f"(column '{col}') — likely unscaled feature reaching SGD")

    y_max = float(np.abs(y).max())
    if y_max > max_abs_y:
        _report("error", label,
                f"batch {batch_idx}: max |y|={y_max:.2g} > {max_abs_y} — "
                f"sentinel / unmasked missing-value in target?")

    if sample_zero_var_on_first_batch and batch_idx == 0:
        col_std = X.std(axis=0)
        zeros = [cols[i] for i, s in enumerate(col_std)
                 if s < 1e-8 and i < len(cols)]
        if zeros:
            _report("warn", label,
                    f"batch 0: zero-variance columns after scaling: "
                    f"{zeros[:5]}{'…' if len(zeros) > 5 else ''}")


# ---------------------------------------------------------------------------
# 3. Post-update weight sanity (linear / SGD models only)
# ---------------------------------------------------------------------------
def check_post_step(
    model: Any,
    batch_idx: int,
    model_name: str = "",
    *,
    coef_max: float = 50.0,
) -> None:
    """
    Inspect coef_ on the underlying sklearn estimator after partial_fit.
    A jump from O(1) to O(1e2) within the first few batches is the
    unambiguous sign of squared-loss SGD divergence.
    """
    if not ENABLED:
        return
    inner = getattr(model, "_model", model)
    coef  = getattr(inner, "coef_", None)
    if coef is None:
        return
    cmax = float(np.abs(coef).max())
    if cmax > coef_max:
        _report(
            "error", f"weights/{model_name}",
            f"batch {batch_idx}: max |coef_|={cmax:.2g} > {coef_max} — "
            f"weights diverging; lower eta0 or check feature scaling"
        )


# ---------------------------------------------------------------------------
# 4. Prediction sanity (evaluate / predict)
# ---------------------------------------------------------------------------
def check_predictions(
    y_pred: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    *,
    label: str = "predict",
    model_name: str = "",
    pred_max: float = 1e3,
    bias_max: float = 50.0,
) -> None:
    """
    Sanity-check predictions against finiteness and physical plausibility
    (LST temperatures roughly in -50..70 °C; |ŷ| > 1e3 means the model
    is broken regardless of how the loss is configured).
    """
    if not ENABLED or y_pred.size == 0:
        return
    full_label = f"{label}/{model_name}"

    if not np.isfinite(y_pred).all():
        bad = int((~np.isfinite(y_pred)).sum())
        _report("error", full_label, f"{bad} non-finite predictions")
    pmax = float(np.abs(y_pred).max())
    if pmax > pred_max:
        _report("error", full_label,
                f"max |ŷ|={pmax:.2g} > {pred_max} — model output diverged")
    if y_true is not None and y_true.size:
        bias = float(y_pred.mean() - y_true.mean())
        if abs(bias) > bias_max:
            _report("warn", full_label,
                    f"bias={bias:+.2f} (mean ŷ={y_pred.mean():.2f}, "
                    f"mean y={y_true.mean():.2f}) — likely target/feature misalignment")

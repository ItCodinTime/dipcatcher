"""Distribution wrappee reselection cache.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.distribution import (
    ScaledGaussianDistribution,
    ScaledStudentTDistribution,
    fit_scaled_wrappee,
    select_wrappee_family_name,
)

from .state import (
    _CONFORMAL_CACHE,
    _GARCH_ASOF_CACHE,
    _GARCH_NAME_ASOF_CACHE,
    _GARCH_SPEC_CACHE,
    _PANEL_ASOF_CACHE,
    _RANKER_CACHE,
    _REALIZED_GARCH_ASOF_CACHE,
    _REALIZED_GARCH_SPEC_CACHE,
    _RL_POLICY_CACHE,
    _WRAPPEE_CACHE,
)


def _array_content_digest(*arrays: NDArray[np.float64]) -> str:
    """Return a deterministic digest for numeric inputs used by cached fits."""
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def clear_wrappee_cache() -> None:
    """Drop cached operational wrappee fits (Student-t / scaled Gaussian)."""
    _WRAPPEE_CACHE.clear()


def wrappee_cache_size() -> int:
    """Number of entries in the process-local wrappee cache (tests / diagnostics)."""
    return len(_WRAPPEE_CACHE)


def wrappee_cal_fingerprint(
    *,
    train_date_keys: tuple[object, ...],
    cal_date_keys: tuple[object, ...] = (),
    alpha: float,
    n_tr: int,
    n_cal: int = 0,
    y_tr_mean: float,
    scale_tr_mean: float,
    label: str = "",
    include_cal: bool = False,
    train_content_digest: str = "",
) -> tuple:
    """Fingerprint for ``_WRAPPEE_CACHE``.

    **When reuse is valid (train-primary, default ``include_cal=False``)**

    Train-fit reuse is keyed by this fingerprint **plus** the selected family
    name (see ``wrappee_fit_cache_key`` / ``resolve_wrappee_reselect_cached``).
    Family selection always uses the current cal/holdout; the fit itself is
    train-only. Adjacent asofs that share the train window may reuse the fitted
    wrappee when the re-selected family matches. Cal-window slides alone do
    **not** invalidate the train-primary key (family may still change).

    **Stale-reuse guards (always in the key)**

    Alpha (interval config), label column, train date keys, train sample size,
    and train y/scale means always participate. Production callers also pass
    a content digest so same-mean data cannot collide. Pass
    ``include_cal=True`` for a strict train+cal identity key (diagnostics / tests
    only).
    """
    base = (
        tuple(train_date_keys),
        float(alpha),
        str(label),
        int(n_tr),
        float(y_tr_mean),
        float(scale_tr_mean),
        str(train_content_digest),
    )
    if include_cal:
        return base + (tuple(cal_date_keys), int(n_cal))
    return base


def wrappee_fit_cache_key(
    *,
    train_date_keys: tuple[object, ...],
    alpha: float,
    n_tr: int,
    y_tr_mean: float,
    scale_tr_mean: float,
    label: str,
    family: str,
    cal_date_keys: tuple[object, ...] = (),
    n_cal: int = 0,
    include_cal: bool = False,
    train_content_digest: str = "",
) -> tuple:
    """Train-primary fit key including selected family (Wave 7).

    Cal keys are ignored unless ``include_cal=True`` (diagnostics). Alpha/label
    and train identity always participate so config changes invalidate.
    """
    fp = wrappee_cal_fingerprint(
        train_date_keys=train_date_keys,
        cal_date_keys=cal_date_keys,
        alpha=alpha,
        n_tr=n_tr,
        n_cal=n_cal,
        y_tr_mean=y_tr_mean,
        scale_tr_mean=scale_tr_mean,
        label=label,
        include_cal=include_cal,
        train_content_digest=train_content_digest,
    )
    return fp + (str(family),)


def _require_wrappee_resolve_inputs(
    taus: list[float],
    y_train: NDArray[np.float64],
    scale_train: NDArray[np.float64],
    y_cal: NDArray[np.float64],
    scale_cal: NDArray[np.float64],
    *,
    alpha: float,
    min_coverage: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Wave 50: fail-closed guards for wrappee resolve (research diagnostic path).

    Rejects empty taus, empty/mismatched y/scale arrays, and alpha/min_coverage
    outside (0, 1). Returns contiguous float64 views. Not a live fill claim.
    """
    if not taus:
        raise ValueError("taus must be a non-empty list")
    yt = np.asarray(y_train, dtype=np.float64).reshape(-1)
    st = np.asarray(scale_train, dtype=np.float64).reshape(-1)
    yc = np.asarray(y_cal, dtype=np.float64).reshape(-1)
    sc = np.asarray(scale_cal, dtype=np.float64).reshape(-1)
    if yt.size == 0 or st.size == 0:
        raise ValueError("y_train and scale_train must be non-empty")
    if yc.size == 0 or sc.size == 0:
        raise ValueError("y_cal and scale_cal must be non-empty")
    if yt.size != st.size:
        raise ValueError("y_train and scale_train length mismatch")
    if yc.size != sc.size:
        raise ValueError("y_cal and scale_cal length mismatch")
    a = float(alpha)
    if not np.isfinite(a) or not (0.0 < a < 1.0):
        raise ValueError("alpha must be finite and in (0, 1)")
    mc = float(min_coverage)
    if not np.isfinite(mc) or not (0.0 < mc < 1.0):
        raise ValueError("min_coverage must be finite and in (0, 1)")
    return yt, st, yc, sc


def resolve_wrappee_reselect_cached(
    taus: list[float],
    y_train: NDArray[np.float64],
    scale_train: NDArray[np.float64],
    y_cal: NDArray[np.float64],
    scale_cal: NDArray[np.float64],
    *,
    train_date_keys: tuple[object, ...],
    cal_date_keys: tuple[object, ...] = (),
    alpha: float,
    label: str = "",
    min_coverage: float = 0.85,
    nominal_coverage: float = 0.90,
) -> tuple[str, ScaledGaussianDistribution | ScaledStudentTDistribution, dict[str, object]]:
    """Re-select wrappee family on current cal; reuse train fit when fingerprint allows.

    Selection always runs on the provided cal/holdout (freshness when cal grows).
    The expensive train MLE is reused from ``_WRAPPEE_CACHE`` when the train-primary
    fingerprint and selected family match a prior entry.

    Returns ``(family_name, fitted_model, meta)`` with ``meta`` keys:
    ``cache_hit``, ``family``, ``reselected``, ``live_pnl_claim`` (always False).
    """
    _ = nominal_coverage
    y_train, scale_train, y_cal, scale_cal = _require_wrappee_resolve_inputs(
        taus,
        y_train,
        scale_train,
        y_cal,
        scale_cal,
        alpha=alpha,
        min_coverage=min_coverage,
    )
    family = select_wrappee_family_name(
        y_train,
        scale_train,
        y_cal,
        scale_cal,
        min_coverage=min_coverage,
    )
    fit_key = wrappee_fit_cache_key(
        train_date_keys=train_date_keys,
        cal_date_keys=cal_date_keys,
        alpha=float(alpha),
        n_tr=int(y_train.size),
        n_cal=int(y_cal.size),
        y_tr_mean=float(np.nanmean(y_train)),
        scale_tr_mean=float(np.nanmean(scale_train)),
        label=str(label),
        family=family,
        include_cal=False,
        train_content_digest=_array_content_digest(y_train, scale_train),
    )
    cached = _WRAPPEE_CACHE.get(fit_key)
    meta: dict[str, object] = {
        "family": family,
        "reselected": True,
        "live_pnl_claim": False,
        "research_only": True,
    }
    if isinstance(cached, (ScaledGaussianDistribution, ScaledStudentTDistribution)):
        _WRAPPEE_CACHE.move_to_end(fit_key)
        meta["cache_hit"] = True
        return family, cached, meta
    model = fit_scaled_wrappee(family, taus, y_train, scale_train)
    # Keep the bounded cache warm across broad multi-asof runs. Clearing the
    # entire cache at the capacity boundary creates a needless thundering herd
    # of refits; evict the least-recently-used fit instead.
    if len(_WRAPPEE_CACHE) >= 64:
        _WRAPPEE_CACHE.popitem(last=False)
    _WRAPPEE_CACHE[fit_key] = model
    meta["cache_hit"] = False
    return family, model, meta


def clear_forecast_caches() -> None:
    """Clear ranker / conformal / wrappee / GARCH process-local caches (panel cache separate)."""
    _RANKER_CACHE.clear()
    _RL_POLICY_CACHE.clear()
    _GARCH_SPEC_CACHE.clear()
    _GARCH_ASOF_CACHE.clear()
    _GARCH_NAME_ASOF_CACHE.clear()
    _REALIZED_GARCH_SPEC_CACHE.clear()
    _REALIZED_GARCH_ASOF_CACHE.clear()
    _PANEL_ASOF_CACHE.clear()
    _CONFORMAL_CACHE.clear()
    _WRAPPEE_CACHE.clear()


__all__ = [
    "clear_forecast_caches",
    "clear_wrappee_cache",
    "resolve_wrappee_reselect_cached",
    "wrappee_cache_size",
    "wrappee_cal_fingerprint",
    "wrappee_fit_cache_key",
]

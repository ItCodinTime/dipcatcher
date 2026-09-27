"""Regime-conditional evaluation gate: per-regime proper-score reporting.

Complements ``quant_fund.validation.gates`` (promotion evidence) with a
statistical question the promotion gate does not ask: does the pooled
out-of-sample score hide regime heterogeneity? Three pieces:

1. ``regime_conditional_summary`` — per-regime count/mean/standard error of a
   proper score series (pinball, CRPS, QLIKE, Brier — never Sharpe/Sortino/
   P&L, per the honesty contract), pooled mean/SE, and the per-regime-minus-
   pooled difference with a stationary-block-bootstrap percentile interval
   (Politis & Romano, 1994, JASA 89(428):1303-1313). Time-uniform reading of
   the comparison is legitimate because the e-processes below control
   crossing probabilities at every time (Howard, Ramdas, McAuliffe & Sekhon,
   2021, Ann. Statist. 49(2):1055-1080, arXiv:1810.08240); the bootstrap CI
   is a fixed-horizon companion, not a license to peek.
2. ``regime_stratified_evalue`` — per-regime AND pooled e-processes for the
   claim "regime r favors forecaster A over B" (smaller expected loss), built
   on the signed loss differential via ``quant_fund.metrics.evalues.
   e_process_loss_diff`` (bounded exponential betting in the style of
   Waudby-Smith & Ramdas, 2024, Ann. Statist. 52(3):1119-1148,
   arXiv:2010.09686; supermartingale e-processes per Ramdas, Ruf, Larsson &
   Koolen, 2022, Ann. Statist. arXiv:2009.03167; Ville, 1939). Because each
   path is a nonnegative supermartingale under its null, the 1/alpha level
   can be inspected at ANY stopping time — anytime-valid interpretation per
   regime and pooled.
3. ``regime_eval_gate`` — fail-closed gate: fails when a regime has share <
   min_regime_share with n >= 30 (too little evidence to pool), when the
   per-regime mean-score CV exceeds max_regime_cv (heterogeneity makes the
   pooled number misleading), or when a per-regime e-value crosses 1/alpha
   for the claim while the pooled e-value does not (a regime contradicts the
   pooled conclusion). Regime-stratified out-of-sample reporting of this kind
   is the 2025-26 practice in quant research audits (see
   docs/AUDIT_FRONTIER.md); regime-conditional forecast evaluation follows
   Nystrup, Madsen & Lindström, 2018, J. Forecasting 37(6):621-637.

Bootstrap design: per-regime independent stationary resamples keep the
observed regime proportions fixed (fixed-design bootstrap of the stratified
mean contrast); the pooled mean is re-derived from the SAME per-regime
resample means each replicate, so the bootstrap reproduces the true
covariance between a regime mean and the pooled mean. Mean block length is
adaptive per regime via the Politis–White (2004, Econometric Reviews
23(1):53-70) automatic selector (as ``quant_fund.metrics.inference.
optimal_block_length``), falling back to the cubic-root rule max(2,
n_r^(1/3)); replicate count is the spec-fixed B = 200.

Fail-closed everywhere: length mismatches, empty inputs, non-finite scores,
or a single unique regime label raise ValueError.

Convention: "regime r favors A over B" means E[L_B - L_A] > 0 in regime r
(smaller loss is better), so the e-process runs on d_t = L_B,t - L_A,t and
crossing 1/alpha is evidence FOR the claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.evalues import e_process_loss_diff
from quant_fund.metrics.inference import optimal_block_length, stationary_bootstrap_indices

Array = NDArray[np.float64]
IntArray = NDArray[np.intp]

__all__ = [
    "RegimeGateResult",
    "RegimeSummary",
    "regime_conditional_summary",
    "regime_eval_gate",
    "regime_stratified_evalue",
]

# Spec-fixed bootstrap size (B = 200 replicates).
_BOOT_REPLICATES = 200
# Default e-level for the anytime-valid claim threshold (Ville: 1/alpha).
_E_LEVEL = 0.05
# Deterministic default seed; callers can inject their own Generator.
_DEFAULT_SEED = 20260927
# Mean-block floor for the cubic-root fallback; matches the circular-block
# default in quant_fund.metrics.inference.bootstrap_mean_ci.
_MEAN_BLOCK_MIN = 2.0


@dataclass(frozen=True)
class RegimeSummary:
    """Output of :func:`regime_conditional_summary`.

    Attributes
    ----------
    names:
        Display name of each regime, in first-appearance (date-sorted) order.
    counts:
        Per-regime observation counts.
    means, stderrs:
        Per-regime mean proper score and its standard error (ddof=1 std /
        sqrt(n); NaN when the regime has a single observation).
    pooled_mean, pooled_stderr, pooled_n:
        Pooled (all-regime) mean, standard error, and count.
    diffs:
        Per-regime ``mean - pooled_mean`` on the observed series.
    diff_ci_low, diff_ci_high:
        Stationary-bootstrap percentile bounds for ``diffs`` at
        ``diff_ci_level`` (spec-fixed B = 200 replicates; independent
        per-regime stationary bootstraps, Politis & Romano 1994, with the
        pooled mean re-derived from the resampled regime means so the
        regime/pooled covariance is reproduced).
    e_value_report:
        Optional attached output of :func:`regime_stratified_evalue` (attach
        with ``dataclasses.replace``); consumed by :func:`regime_eval_gate`
        for the pooled-vs-regime e-value contradiction check.
    """

    names: tuple[str, ...]
    counts: IntArray
    means: Array
    stderrs: Array
    pooled_mean: float
    pooled_stderr: float
    pooled_n: int
    diffs: Array
    diff_ci_low: Array
    diff_ci_high: Array
    diff_ci_level: float
    n_boot: int
    mean_block: float
    e_value_report: dict[str, Any] | None = field(default=None, repr=False)


@dataclass(frozen=True)
class RegimeGateResult:
    """Output of :func:`regime_eval_gate`. ``passed`` is False when any
    fail-closed reason fired; ``reasons`` lists every violated check."""

    passed: bool
    reasons: tuple[str, ...]


def _mean_block_fallback(n: int) -> float:
    return max(_MEAN_BLOCK_MIN, float(int(round(float(n) ** (1.0 / 3.0)))))


def _adaptive_block(seg: Array) -> float:
    """Politis–White (2004) automatic mean block, cubic-root fallback."""
    n = int(seg.size)
    try:
        blk = float(optimal_block_length(seg))
    except (ValueError, IndexError, FloatingPointError):
        blk = float("nan")
    if not np.isfinite(blk) or blk < 1.0:
        blk = _mean_block_fallback(n)
    return float(min(max(1.0, blk), float(max(n, 1))))


def _as_vector(name: str, values: Any) -> NDArray[Any]:
    arr = np.asarray(values)
    if arr.ndim == 0:
        raise ValueError(f"{name} must be a one-dimensional sequence")
    return arr.reshape(-1)


def _prepare(
    dates: Any,
    regime_ids: Any,
    values: Any,
    value_name: str,
    *,
    require_finite: bool = True,
) -> tuple[NDArray[np.intp], list[str], Array]:
    """Validate lengths/non-emptiness and sort by date.

    Returns (order, regime key list, float values) where regime keys are
    ``str(id)`` in first-appearance order after date sorting. With
    ``require_finite``, non-finite values raise ValueError; otherwise they
    pass through for the caller to filter pairwise.
    """
    d_arr = _as_vector("dates", dates)
    r_arr = _as_vector("regime_ids", regime_ids)
    v_arr = np.asarray(values, dtype=float).reshape(-1)
    n = int(v_arr.size)
    if not (d_arr.size == n and r_arr.size == n):
        raise ValueError(
            f"length mismatch: dates={d_arr.size}, regime_ids={r_arr.size}, {value_name}={n}"
        )
    if n == 0:
        raise ValueError(f"{value_name} series must be nonempty")
    if require_finite and not bool(np.all(np.isfinite(v_arr))):
        raise ValueError(f"{value_name} must be finite (NaN/inf rejected)")
    try:
        order = np.argsort(d_arr, kind="stable")
    except TypeError as exc:
        raise ValueError("dates must be sortable (e.g. datetime64 or strings)") from exc
    order_i = np.asarray(order, dtype=np.intp)
    keys = [str(r_arr[i]) for i in order_i]
    return order_i, keys, v_arr


def _regime_masks(keys: list[str], unique: list[str]) -> list[IntArray]:
    idx = np.arange(len(keys), dtype=np.intp)
    return [idx[np.asarray([k == label for k in keys], dtype=bool)] for label in unique]


def _unique_checked(keys: list[str]) -> list[str]:
    unique: list[str] = []
    for k in keys:
        if k not in unique:
            unique.append(k)
    if len(unique) < 2:
        raise ValueError("regime_ids has a single unique value; nothing to stratify")
    return unique


def regime_conditional_summary(
    dates: Any,
    regime_ids: Any,
    scores: Any,
    regime_names: dict[Any, str] | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> RegimeSummary:
    """Per-regime proper-score summary with pooled comparison and bootstrap CIs.

    Parameters
    ----------
    dates:
        Sortable date-like sequence (datetime64, strings, ...). Rows are
        sorted by date so the block bootstrap respects time order.
    regime_ids:
        Regime label per observation (compared as str).
    scores:
        Finite proper-score series (pinball, CRPS, QLIKE, Brier; smaller is
        better). NaN/inf raise ValueError.
    regime_names:
        Optional mapping from regime id (original or str form) to display
        name; applied positionally per unique regime.
    rng:
        Optional Generator; default is deterministically seeded.

    Fail-closed: length mismatches, empty input, non-finite scores, or a
    single unique regime label raise ValueError.
    """
    order, keys, vals = _prepare(dates, regime_ids, scores, "scores")
    v = vals[order]
    unique = _unique_checked(keys)
    if regime_names:
        lookup = {str(k): str(v_name) for k, v_name in regime_names.items()}
        names = [lookup.get(k, k) for k in unique]
    else:
        names = list(unique)
    gen = rng if rng is not None else np.random.default_rng(_DEFAULT_SEED)
    masks = _regime_masks(keys, unique)
    counts_l: list[int] = []
    means_l: list[float] = []
    stderrs_l: list[float] = []
    for mask in masks:
        seg = v[mask]
        counts_l.append(int(seg.size))
        means_l.append(float(np.mean(seg)))
        stderrs_l.append(
            float(np.std(seg, ddof=1) / np.sqrt(seg.size)) if seg.size > 1 else float("nan")
        )
    counts = np.asarray(counts_l, dtype=np.intp)
    means = np.asarray(means_l, dtype=float)
    stderrs = np.asarray(stderrs_l, dtype=float)
    pooled_mean = float(np.mean(v))
    pooled_stderr = float(np.std(v, ddof=1) / np.sqrt(v.size))
    diffs = means - pooled_mean

    # Stationary bootstrap (Politis & Romano 1994) of the per-regime-minus-
    # pooled difference: independent per-regime resamples keep the observed
    # regime proportions fixed, and the pooled mean is rebuilt from the same
    # resampled regime means, so the regime/pooled covariance is reproduced.
    n_total = int(v.size)
    boot_means = np.empty((len(unique), _BOOT_REPLICATES), dtype=float)
    for r, mask in enumerate(masks):
        seg = v[mask]
        block = _adaptive_block(seg)
        idx = stationary_bootstrap_indices(int(seg.size), _BOOT_REPLICATES, block, gen)
        boot_means[r] = np.mean(seg[idx], axis=1)
    weights = counts.astype(float) / float(n_total)
    boot_pooled = weights @ boot_means
    boot_diffs = boot_means - boot_pooled[np.newaxis, :]
    level = 0.90
    tail = (1.0 - level) / 2.0
    ci_low = np.quantile(boot_diffs, tail, axis=1)
    ci_high = np.quantile(boot_diffs, 1.0 - tail, axis=1)
    return RegimeSummary(
        names=tuple(names),
        counts=counts,
        means=means,
        stderrs=stderrs,
        pooled_mean=pooled_mean,
        pooled_stderr=pooled_stderr,
        pooled_n=n_total,
        diffs=np.asarray(diffs, dtype=float),
        diff_ci_low=np.asarray(ci_low, dtype=float),
        diff_ci_high=np.asarray(ci_high, dtype=float),
        diff_ci_level=level,
        n_boot=_BOOT_REPLICATES,
        mean_block=_adaptive_block(v),
    )


def regime_stratified_evalue(
    dates: Any,
    regime_ids: Any,
    loss_a: Any,
    loss_b: Any,
    block_length: float | None = None,
    *,
    lam: float = 0.25,
    initial_bound: float = 1.0,
    level: float = _E_LEVEL,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Per-regime and pooled e-processes for "regime r favors A over B".

    The claim "favors A over B" means forecaster A has smaller expected loss
    in the regime, so each e-process runs on the signed differential
    ``d_t = L_B,t - L_A,t`` via ``e_process_loss_diff``; crossing 1/level is
    evidence FOR the claim. Each per-regime path is a nonnegative
    supermartingale under ``E_regime[d] <= 0`` and the pooled path under
    ``E[d] <= 0``, so both admit anytime-valid interpretation (inspect at any
    stopping time; Ville 1939; Ramdas, Ruf, Larsson & Koolen 2022).

    Parameters
    ----------
    dates, regime_ids:
        As in :func:`regime_conditional_summary` (rows date-sorted).
    loss_a, loss_b:
        Aligned per-observation loss series (smaller is better); non-finite
        pairs are dropped pairwise by the underlying e-process builder.
    block_length:
        Optional stationary-bootstrap mean block for the reported CI on each
        per-regime mean differential; default is the Politis–White adaptive
        selector (cubic-root fallback), as in :func:`regime_conditional_summary`.
    lam, initial_bound, level:
        Betting fraction, predictable-bound floor, and e-level (default
        0.05 -> threshold 20).
    rng:
        Optional Generator; default is deterministically seeded.

    Returns
    -------
    dict
        ``claim``, ``signed_diff``, ``level``, ``threshold``, ``pooled``
        (keys ``e`` path, ``e_final``, ``mean_diff``, ``n``, ``first_cross``,
        ``crossed``) and ``regimes`` mapping each regime name to the same
        fields plus ``diff_ci90`` (stationary-bootstrap interval for the mean
        differential, B = 200).
    """
    order, keys, a = _prepare(dates, regime_ids, loss_a, "loss_a", require_finite=False)
    b_raw = np.asarray(loss_b, dtype=float).reshape(-1)
    if b_raw.size != a.size:
        raise ValueError(f"length mismatch: loss_a={a.size}, loss_b={b_raw.size} (after date sort)")
    b = b_raw[order]
    a = a[order]
    names = _unique_checked(keys)
    gen = rng if rng is not None else np.random.default_rng(_DEFAULT_SEED)
    threshold = 1.0 / float(level)
    masks = _regime_masks(keys, names)

    def _path_report(d: Array) -> dict[str, Any]:
        path = e_process_loss_diff(d=d, lam=lam, initial_bound=initial_bound)
        hit = np.nonzero(path >= threshold)[0]
        return {
            "e": path,
            "e_final": float(path[-1]),
            "mean_diff": float(np.mean(d)) if d.size else float("nan"),
            "n": int(d.size),
            "first_cross": int(hit[0]) if hit.size else None,
            "crossed": bool(hit.size > 0),
        }

    d_all = b - a
    regimes: dict[str, Any] = {}
    for name, mask in zip(names, masks, strict=True):
        finite = d_all[mask]
        finite = finite[np.isfinite(finite)]
        report = _path_report(finite)
        if finite.size:
            block = float(block_length) if block_length is not None else _adaptive_block(finite)
            idx = stationary_bootstrap_indices(int(finite.size), _BOOT_REPLICATES, block, gen)
            boot_means = np.mean(finite[idx], axis=1)
            report["diff_ci90"] = (
                float(np.quantile(boot_means, 0.05)),
                float(np.quantile(boot_means, 0.95)),
            )
        else:
            report["diff_ci90"] = (float("nan"), float("nan"))
        regimes[name] = report
    pooled = _path_report(d_all[np.isfinite(d_all)])
    return {
        "claim": "regime {name} favors A over B (smaller expected loss, d = L_B - L_A)",
        "signed_diff": "loss_b - loss_a",
        "level": float(level),
        "threshold": float(threshold),
        "pooled": pooled,
        "regimes": regimes,
        "n_total": int(d_all.size),
    }


def regime_eval_gate(
    summary: RegimeSummary,
    min_regime_share: float = 0.05,
    max_regime_cv: float = 3.0,
) -> RegimeGateResult:
    """Fail-closed gate on a :class:`RegimeSummary`.

    Fails (with one reason per violated check) when:

    1. any regime has share < ``min_regime_share`` AND count >= 30 — too
       little per-regime evidence for the pooled number to be trusted;
    2. the CV (ddof=1 std / |mean|) of the per-regime mean scores exceeds
       ``max_regime_cv`` — heterogeneity makes the pooled score misleading;
    3. a per-regime e-value for the "favors A over B" claim crosses
       1/level while the pooled e-value does not — a regime contradicts the
       pooled conclusion (requires ``summary.e_value_report`` attached via
       ``dataclasses.replace``; skipped when absent, since the summary alone
       carries no e-process evidence).

    Fail-closed: invalid thresholds raise ValueError; an unexpected internal
    error appends ``gate_error`` and fails rather than passing.
    """
    share = float(min_regime_share)
    cv_max = float(max_regime_cv)
    if not np.isfinite(share) or not 0.0 < share < 1.0:
        raise ValueError("min_regime_share must lie in (0, 1)")
    if not np.isfinite(cv_max) or cv_max <= 0.0:
        raise ValueError("max_regime_cv must be finite and > 0")
    reasons: list[str] = []
    try:
        total = int(np.sum(summary.counts))
        for name, count in zip(summary.names, summary.counts, strict=True):
            regime_share = float(count) / float(total)
            if regime_share < share and int(count) >= 30:
                reasons.append(
                    f"insufficient_regime_share:{name}"
                    f"({regime_share:.4f}<{share:.4f},n={int(count)})"
                )
        means = np.asarray(summary.means, dtype=float)
        denom = abs(float(np.mean(means)))
        cv = float(np.std(means, ddof=1) / denom) if denom > 1e-12 else float("inf")
        if cv > cv_max:
            reasons.append(f"regime_mean_cv_exceeded:{cv:.3f}>{cv_max:.3f}")
        report = summary.e_value_report
        if report is not None:
            pooled_crossed = bool(report.get("pooled", {}).get("crossed", False))
            regimes = report.get("regimes", {})
            if isinstance(regimes, dict):
                for name, info in regimes.items():
                    if (
                        isinstance(info, dict)
                        and bool(info.get("crossed", False))
                        and not pooled_crossed
                    ):
                        reasons.append(f"regime_contradicts_pooled:{name}")
    except Exception as exc:  # defensive: never pass on internal error
        reasons.append(f"gate_error:{type(exc).__name__}")
    return RegimeGateResult(passed=not reasons, reasons=tuple(reasons))

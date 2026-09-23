"""Data-snooping tests over a universe of strategies (research-only).

White's Reality Check (2000), Hansen's SPA (2005; lower / consistent / upper),
Romano–Wolf StepM (2005), and the Hansen–Lunde–Nason (2011) model confidence
set, all on a ``T × K`` matrix of performance differentials (strategy minus
benchmark; **larger is better**, e.g. excess return or IC). Resampling uses the
stationary bootstrap (Politis–Romano 1994) with the Politis–White (2004)
automatic block length.

Conventions
-----------
- ``f[t, k]`` is the period-``t`` performance differential of strategy ``k``.
- One-sided nulls: H0 = ``max_k E[f_k] <= 0`` (Reality Check / SPA) — a small
  p-value is evidence that *some* strategy beats the benchmark after accounting
  for the full universe of trials.
- Bootstrap p-values use ``(1 + #{stat* >= stat}) / (B + 1)`` (never 0).
- Fail-closed: shape violations raise ``ValueError``; non-finite rows are
  dropped with an explicit count; too-short panels and zero-variance columns
  yield honest NaN results (never a fabricated pass).

Research diagnostic only — never a live Sharpe / P&L claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.inference import optimal_block_length, stationary_bootstrap_indices

Array = NDArray[np.float64]

MIN_OBS = 10
MIN_STRATEGIES = 2


@dataclass(frozen=True)
class SnoopingResult:
    """White's Reality Check outcome (H0: max_k E[f_k] <= 0)."""

    statistic: float
    p_value: float
    n_obs: int
    n_strategies: int
    n_rows_dropped: int
    block: float
    n_boot: int
    best_index: int
    best_mean: float


@dataclass(frozen=True)
class SpaResult:
    """Hansen's SPA outcome: lower / consistent / upper recentering variants.

    ``p_upper`` never recenters (conservative); ``p_lower`` recenters every
    column at its sample mean; ``p_consistent`` (the recommended variant)
    recenters only columns that are not significantly negative. There is no
    universal ordering between the three: ``p_consistent == p_lower`` exactly
    when no column is significantly negative, and ``p_consistent`` can fall
    below ``p_lower`` when bad columns inflate the all-recentered null.
    """

    statistic: float
    p_lower: float
    p_consistent: float
    p_upper: float
    n_obs: int
    n_strategies: int
    n_rows_dropped: int
    n_dropped: int
    block: float
    n_boot: int
    best_index: int
    best_mean: float
    studentized: tuple[float, ...]


@dataclass(frozen=True)
class StepMResult:
    """Romano–Wolf StepM: FWER-controlled rejections + step-down adjusted p-values."""

    rejected: tuple[bool, ...]
    adjusted_p: tuple[float, ...]
    n_rejected: int
    alpha: float
    n_obs: int
    n_strategies: int
    n_rows_dropped: int
    n_dropped: int
    block: float
    n_boot: int
    order: tuple[int, ...]


@dataclass(frozen=True)
class McsResult:
    """Hansen–Lunde–Nason model confidence set (range statistic)."""

    included: tuple[bool, ...]
    p_values: tuple[float, ...]
    n_included: int
    alpha: float
    n_obs: int
    n_strategies: int
    n_rows_dropped: int
    n_dropped: int
    block: float
    n_boot: int
    elimination_order: tuple[int, ...]


def _prepare(f: Array, *, drop_constant: bool) -> tuple[Array, int, int, int]:
    """Validate the differential matrix and drop non-finite rows.

    Returns ``(matrix, n_obs, n_rows_dropped, n_dropped_constant)``.
    """
    arr = np.asarray(f, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("f must be 1-D (single strategy) or 2-D (T x K)")
    n_rows_raw = int(arr.shape[0])
    n_cols = int(arr.shape[1])
    if n_rows_raw == 0:
        return np.empty((0, n_cols)), 0, 0, 0
    finite_rows = np.isfinite(arr).all(axis=1)
    n_rows_dropped = int(n_rows_raw - int(finite_rows.sum()))
    arr = arr[finite_rows]
    n_dropped = 0
    if drop_constant and arr.shape[0] >= 2 and arr.shape[1] > 0:
        # ptp is exactly 0 for a constant column; std is not (floating-point
        # variance of identical values can be ~1e-19 and would mint a huge
        # studentized statistic). Near-constant columns are dropped too.
        spread = np.ptp(arr, axis=0)
        scale_ref = np.maximum(np.abs(arr).max(axis=0), 1e-12)
        keep = spread > 1e-10 * scale_ref
        n_dropped = int((~keep).sum())
        arr = arr[:, keep]
    return arr, int(arr.shape[0]), n_rows_dropped, n_dropped


def _resolve_block(arr: Array, block: float | None) -> float:
    """Median per-column Politis–White block length (or the supplied value)."""
    n = int(arr.shape[0])
    if block is None:
        candidates = [optimal_block_length(arr[:, k]) for k in range(arr.shape[1])]
        finite = [b for b in candidates if np.isfinite(b)]
        block = float(np.median(finite)) if finite else max(2.0, float(np.round(n ** (1.0 / 3.0))))
    if not np.isfinite(block) or float(block) < 1.0:
        raise ValueError("block must be finite and >= 1")
    return float(min(max(float(block), 1.0), float(max(1, n))))


def _bootstrap_means(arr: Array, idx: NDArray[np.intp]) -> Array:
    """Per-bootstrap-sample column means, shape (n_boot, K).

    Built as a draw-count matrix times the data (one matmul) rather than
    fancy-indexing each draw: ~10x faster for realistic B and T, and exactly
    linear in ``arr`` (so scale invariance is preserved to machine precision).
    """
    n_boot, n = int(idx.shape[0]), int(idx.shape[1])
    counts = np.zeros((n_boot, n), dtype=float)
    for b in range(n_boot):
        counts[b] = np.bincount(idx[b], minlength=n)
    return (counts @ arr) / float(n)


def _nan_rc(n_obs: int, n_strategies: int, n_rows_dropped: int, n_boot: int) -> SnoopingResult:
    return SnoopingResult(
        statistic=float("nan"),
        p_value=float("nan"),
        n_obs=n_obs,
        n_strategies=n_strategies,
        n_rows_dropped=n_rows_dropped,
        block=float("nan"),
        n_boot=int(n_boot),
        best_index=-1,
        best_mean=float("nan"),
    )


def reality_check(
    f: Array,
    *,
    n_boot: int = 2000,
    block: float | None = None,
    seed: int = 7,
) -> SnoopingResult:
    """White's Reality Check (2000) with the stationary bootstrap.

    Statistic ``V = max_k sqrt(T) · f̄_k``; the bootstrap distribution recenters
    every column at its full-sample mean (so the null distribution is invariant
    to a common location shift). ``p = (1 + #{V* >= V}) / (B + 1)``.
    """
    if int(n_boot) < 1:
        raise ValueError("n_boot must be >= 1")
    arr, n, n_rows_dropped, _ = _prepare(f, drop_constant=False)
    k = int(arr.shape[1])
    if n < MIN_OBS or k < 1:
        return _nan_rc(n, k, n_rows_dropped, n_boot)
    block_v = _resolve_block(arr, block)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, int(n_boot), block_v, rng)
    means = arr.mean(axis=0)
    scale = float(np.sqrt(n))
    boot = _bootstrap_means(arr, idx)
    v = float(np.max(scale * means))
    v_star = (scale * (boot - means)).max(axis=1)
    p = float((1.0 + float(np.sum(v_star >= v))) / (int(n_boot) + 1.0))
    best = int(np.argmax(means))
    return SnoopingResult(
        statistic=v,
        p_value=p,
        n_obs=n,
        n_strategies=k,
        n_rows_dropped=n_rows_dropped,
        block=block_v,
        n_boot=int(n_boot),
        best_index=best,
        best_mean=float(means[best]),
    )


def _studentized_inputs(
    arr: Array, idx: NDArray[np.intp], n: int
) -> tuple[Array, Array, Array, Array]:
    """Return (means, sigma, t_stats, t_boot) for the studentized tests.

    ``sigma`` is Hansen's bootstrap long-run standard deviation of
    ``sqrt(T) · f̄_k``; ``t_boot`` holds the null (recentered) bootstrap
    studentized statistics, shape (n_boot, K).
    """
    means = arr.mean(axis=0)
    scale = float(np.sqrt(n))
    boot = _bootstrap_means(arr, idx)
    dev = scale * (boot - means)
    sigma = np.sqrt(np.mean(dev**2, axis=0))
    t_stats = scale * means / sigma
    t_boot = dev / sigma[None, :]
    return means, sigma, t_stats, t_boot


def spa_test(
    f: Array,
    *,
    n_boot: int = 2000,
    block: float | None = None,
    seed: int = 7,
) -> SpaResult:
    """Hansen's SPA (2005) — studentized Reality Check with three p-values.

    ``T^SPA = max_k sqrt(T) f̄_k / σ̂_k``. Null bootstrap statistics recenter at

    - ``lower``      : ``g_k = f̄_k`` for every k (all columns recentered),
    - ``consistent`` : ``g_k = f̄_k`` only for columns passing
      ``sqrt(T) f̄_k/σ̂_k >= -sqrt(2 log log T)`` (bad columns recentered at 0),
    - ``upper``      : ``g_k = 0`` (no recentering — conservative).

    Zero-variance columns are dropped (reported in ``n_dropped``) because they
    cannot be studentized.
    """
    if int(n_boot) < 1:
        raise ValueError("n_boot must be >= 1")
    arr, n, n_rows_dropped, n_dropped = _prepare(f, drop_constant=True)
    k = int(arr.shape[1])
    nan = float("nan")
    if n < MIN_OBS or k < 1:
        return SpaResult(
            statistic=nan,
            p_lower=nan,
            p_consistent=nan,
            p_upper=nan,
            n_obs=n,
            n_strategies=k,
            n_rows_dropped=n_rows_dropped,
            n_dropped=n_dropped,
            block=nan,
            n_boot=int(n_boot),
            best_index=-1,
            best_mean=nan,
            studentized=(),
        )
    block_v = _resolve_block(arr, block)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, int(n_boot), block_v, rng)
    means, sigma, t_stats, t_boot = _studentized_inputs(arr, idx, n)
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        return SpaResult(
            statistic=nan,
            p_lower=nan,
            p_consistent=nan,
            p_upper=nan,
            n_obs=n,
            n_strategies=k,
            n_rows_dropped=n_rows_dropped,
            n_dropped=n_dropped,
            block=block_v,
            n_boot=int(n_boot),
            best_index=-1,
            best_mean=nan,
            studentized=(),
        )
    stat = float(np.max(t_stats))
    scale = float(np.sqrt(n))
    threshold = float(np.sqrt(2.0 * np.log(np.log(n)) / n))

    def _p_for(g: Array) -> float:
        # T*_b = max_k sqrt(T)(f̄*_{k,b} - g_k)/σ_k
        #      = max_k (t_boot_{k,b} + t_stats_k - sqrt(T) g_k/σ_k).
        adj = (scale * g) / sigma
        t_star = (t_boot + t_stats[None, :] - adj[None, :]).max(axis=1)
        return float((1.0 + float(np.sum(t_star >= stat))) / (int(n_boot) + 1.0))

    g_lower = means
    g_upper = np.zeros_like(means)
    g_consistent = np.where(means >= -sigma * threshold, means, 0.0)
    p_lower = _p_for(g_lower)
    p_consistent = _p_for(g_consistent)
    p_upper = _p_for(g_upper)
    best = int(np.argmax(t_stats))
    return SpaResult(
        statistic=stat,
        p_lower=p_lower,
        p_consistent=p_consistent,
        p_upper=p_upper,
        n_obs=n,
        n_strategies=k,
        n_rows_dropped=n_rows_dropped,
        n_dropped=n_dropped,
        block=block_v,
        n_boot=int(n_boot),
        best_index=best,
        best_mean=float(means[best]),
        studentized=tuple(float(x) for x in t_stats),
    )


def stepm(
    f: Array,
    *,
    n_boot: int = 2000,
    block: float | None = None,
    alpha: float = 0.05,
    seed: int = 7,
) -> StepMResult:
    """Romano–Wolf StepM (2005): step-down FWER control with adjusted p-values.

    Hypotheses are ordered by ascending studentized statistic; at each step the
    null distribution is the bootstrap maximum over the remaining (less
    significant) hypotheses. Adjusted p-values are the running maximum of the
    per-step p-values, so rejections are exactly the hypotheses with
    ``adjusted_p <= alpha`` (a suffix of the ordering).
    """
    if int(n_boot) < 1:
        raise ValueError("n_boot must be >= 1")
    if not np.isfinite(alpha) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    arr, n, n_rows_dropped, n_dropped = _prepare(f, drop_constant=True)
    k = int(arr.shape[1])
    if n < MIN_OBS or k < 1:
        return StepMResult(
            rejected=tuple(False for _ in range(k)),
            adjusted_p=tuple(float("nan") for _ in range(k)),
            n_rejected=0,
            alpha=float(alpha),
            n_obs=n,
            n_strategies=k,
            n_rows_dropped=n_rows_dropped,
            n_dropped=n_dropped,
            block=float("nan"),
            n_boot=int(n_boot),
            order=(),
        )
    block_v = _resolve_block(arr, block)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, int(n_boot), block_v, rng)
    _means, sigma, t_stats, t_boot = _studentized_inputs(arr, idx, n)
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        return StepMResult(
            rejected=tuple(False for _ in range(k)),
            adjusted_p=tuple(float("nan") for _ in range(k)),
            n_rejected=0,
            alpha=float(alpha),
            n_obs=n,
            n_strategies=k,
            n_rows_dropped=n_rows_dropped,
            n_dropped=n_dropped,
            block=block_v,
            n_boot=int(n_boot),
            order=(),
        )
    order = np.argsort(t_stats, kind="stable")
    t_sorted = t_stats[order]
    running_max = np.maximum.accumulate(t_boot[:, order], axis=1)
    p_raw = (1.0 + (running_max >= t_sorted[None, :]).sum(axis=0)) / (int(n_boot) + 1.0)
    # Step-down stopping rule: H_(j) is rejected only when every step from the
    # most significant down to j rejects, so the adjusted p-value is the
    # cumulative maximum taken from the TOP of the ordering.
    p_adj_sorted = np.maximum.accumulate(p_raw[::-1])[::-1]
    rejected_sorted = p_adj_sorted <= float(alpha)
    rejected = np.zeros(k, dtype=bool)
    adjusted = np.full(k, float("nan"))
    rejected[order] = rejected_sorted
    adjusted[order] = p_adj_sorted
    return StepMResult(
        rejected=tuple(bool(x) for x in rejected),
        adjusted_p=tuple(float(x) for x in adjusted),
        n_rejected=int(rejected.sum()),
        alpha=float(alpha),
        n_obs=n,
        n_strategies=k,
        n_rows_dropped=n_rows_dropped,
        n_dropped=n_dropped,
        block=block_v,
        n_boot=int(n_boot),
        order=tuple(int(x) for x in order[::-1]),
    )


def model_confidence_set(
    f: Array,
    *,
    n_boot: int = 2000,
    block: float | None = None,
    alpha: float = 0.10,
    seed: int = 7,
) -> McsResult:
    """Hansen–Lunde–Nason (2011) model confidence set, range statistic.

    Within the current set the differentials are demeaned cross-sectionally
    (H0: all remaining means equal); ``T_max = max_i t_i − min_i t_i`` with
    studentized demeaned means. While the bootstrap p-value is below ``alpha``
    the worst remaining model (smallest demeaned mean) is eliminated. Each
    model's ``p_values`` entry is the p-value at the step it left the set (or
    the final step's p-value when it survives), so the set
    ``{i : p_values[i] >= alpha}`` is the MCS.
    """
    if int(n_boot) < 1:
        raise ValueError("n_boot must be >= 1")
    if not np.isfinite(alpha) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    arr, n, n_rows_dropped, n_dropped = _prepare(f, drop_constant=True)
    k = int(arr.shape[1])
    if n < MIN_OBS or k < 1:
        return McsResult(
            included=tuple(False for _ in range(k)),
            p_values=tuple(float("nan") for _ in range(k)),
            n_included=0,
            alpha=float(alpha),
            n_obs=n,
            n_strategies=k,
            n_rows_dropped=n_rows_dropped,
            n_dropped=n_dropped,
            block=float("nan"),
            n_boot=int(n_boot),
            elimination_order=(),
        )
    block_v = _resolve_block(arr, block)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, int(n_boot), block_v, rng)
    scale = float(np.sqrt(n))
    active = np.ones(k, dtype=bool)
    p_values = np.full(k, float("nan"))
    elimination: list[int] = []
    while True:
        cols = np.nonzero(active)[0]
        if cols.size <= 1:
            for c in cols:
                p_values[int(c)] = 1.0
            break
        sub = arr[:, cols]
        demeaned = sub - sub.mean(axis=1, keepdims=True)
        d_mean = demeaned.mean(axis=0)
        boot = _bootstrap_means(demeaned, idx)
        dev = scale * (boot - d_mean)
        sigma = np.sqrt(np.mean(dev**2, axis=0))
        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
            p_values[cols] = float("nan")
            break
        t_stats = scale * d_mean / sigma
        stat = float(t_stats.max() - t_stats.min())
        t_boot = dev / sigma[None, :]
        stat_star = t_boot.max(axis=1) - t_boot.min(axis=1)
        p = float((1.0 + float(np.sum(stat_star >= stat))) / (int(n_boot) + 1.0))
        for c in cols:
            p_values[int(c)] = p
        if p >= float(alpha):
            break
        worst = int(cols[int(np.argmin(d_mean))])
        active[worst] = False
        elimination.append(worst)
    return McsResult(
        included=tuple(bool(x) for x in active),
        p_values=tuple(float(x) for x in p_values),
        n_included=int(active.sum()),
        alpha=float(alpha),
        n_obs=n,
        n_strategies=k,
        n_rows_dropped=n_rows_dropped,
        n_dropped=n_dropped,
        block=block_v,
        n_boot=int(n_boot),
        elimination_order=tuple(elimination),
    )

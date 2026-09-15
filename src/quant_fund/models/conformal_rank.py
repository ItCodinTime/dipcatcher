"""Conformal prediction sets for ranking / top-k. No Sharpe.

Split conformalized selection (Jin & Candès 2023 cfBH; Bates et al. 2021
conformal p-values) and split conformal on the k-th order-statistic threshold.

Cross-section is per date: ranks, percentiles, oracle top-k, and BH never stack
raw names across timestamps. See ADR-018.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

from quant_fund.metrics.conformal import conformal_quantile
from quant_fund.metrics.cross_section import _date_keys
from quant_fund.metrics.inference import benjamini_hochberg

Array = NDArray[np.float64]
Guarantee = Literal["fdr", "set_coverage"]


@dataclass(frozen=True)
class TopKConformalResult:
    """Selection on the chronological test dates. Metrics are date-grouped."""

    selected: NDArray[np.bool_]
    p_values: Array
    qhat: float
    guarantee: str
    alpha: float
    k: int
    set_size: float
    fdr: float
    coverage: float
    n_dates: int
    n_cal_dates: int


def within_date_percentiles(values: Array) -> Array:
    """Average-rank percentiles on one date. Matches ``cs_pct``: (rank - 0.5) / n."""
    v = np.asarray(values, dtype=float).reshape(-1)
    out = np.full(v.size, np.nan)
    finite = np.isfinite(v)
    n = int(finite.sum())
    if n == 0:
        return out
    ranks = rankdata(v[finite], method="average")
    out[finite] = (ranks - 0.5) / float(n)
    return out


def oracle_topk_mask(y: Array, k: int, higher_is_better: bool = True) -> NDArray[np.bool_]:
    """Exactly ``min(k, n_finite)`` names. Ties broken by stable argsort."""
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.zeros(y.size, dtype=bool)
    finite = np.isfinite(y)
    n = int(finite.sum())
    if n == 0 or int(k) < 1:
        return mask
    kk = min(int(k), n)
    vals = y[finite]
    if not higher_is_better:
        vals = -vals
    take = np.argsort(vals, kind="mergesort")[-kk:]
    mask[np.flatnonzero(finite)[take]] = True
    return mask


def conformal_selection_pvalues(test_scores: Array, null_scores: Array) -> Array:
    """Conservative Bates / Jin conformal p-values.

    Higher ``test_scores`` are more extreme against the null pool:

        p_j = (1 + #{i : u_i ≥ u_j}) / (n + 1).

    This is formula (4) in Jin & Candès (2023) with U_j = 1 (no extra randomness).
    """
    test = np.asarray(test_scores, dtype=float).reshape(-1)
    null = np.asarray(null_scores, dtype=float).reshape(-1)
    null = null[np.isfinite(null)]
    p = np.ones(test.size, dtype=float)
    n = int(null.size)
    if n == 0:
        return p
    finite = np.isfinite(test)
    if not bool(finite.any()):
        return p
    ge = np.sum(null[None, :] >= test[finite, None], axis=1)
    p[finite] = (1.0 + ge) / float(n + 1)
    return p


def _unique_date_masks(dates: NDArray[Any] | list[object]) -> list[tuple[str, NDArray[np.bool_]]]:
    keys = _date_keys(dates)
    order: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key not in seen:
            seen.add(key)
            order.append(key)
    return [(key, np.array([k == key for k in keys], dtype=bool)) for key in order]


def _lower_conformal_quantile(values: Array, alpha: float) -> float:
    """Finite-sample lower (1-α) quantile: −q̂ of the negated scores."""
    v = np.asarray(values, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    return float(-conformal_quantile(-v, alpha))


def _cal_count(n_dates: int, cal_frac: float) -> int:
    if n_dates < 2:
        raise ValueError("need at least two dates for a chronological split")
    n_cal = int(np.floor(float(cal_frac) * n_dates))
    return int(min(max(n_cal, 1), n_dates - 1))


def _date_fdp_cover(
    selected: NDArray[np.bool_], oracle: NDArray[np.bool_], idx: NDArray[np.bool_]
) -> tuple[float, float, int]:
    sel = selected[idx]
    ora = oracle[idx]
    n_sel = int(sel.sum())
    n_ora = int(ora.sum())
    fdp = 0.0 if n_sel == 0 else float(np.sum(sel & ~ora)) / float(n_sel)
    cover = 1.0 if n_ora == 0 else float(np.all(ora <= sel))
    return fdp, cover, n_sel


def conformal_topk(
    scores: Array,
    labels_or_true_ranks: Array,
    k: int,
    alpha: float,
    dates: NDArray[Any] | list[object],
    *,
    guarantee: Guarantee = "fdr",
    higher_is_better: bool = True,
    cal_frac: float = 0.5,
) -> TopKConformalResult:
    """Select names that belong in the date's top-k with a finite-sample guarantee.

    ``guarantee="fdr"`` (default)
        Split conformalized selection. H0 for name i on a date: *not* in that
        date's oracle top-k (true label is not among the k best). When
        ``k = ⌊n/2⌋`` this is H0: not better than the median. Calibration
        scores are within-date percentiles of names that satisfy H0
        (true rank > k). Test p-values are Bates/Jin conformal p-values; BH
        runs **per test date**. Finite-sample FDR ≤ α under exchangeability of
        those percentiles between a test null and the calibration null pool
        (Jin & Candès 2023, Thm. 3; Bates et al. 2021).

    ``guarantee="set_coverage"``
        Split conformal on the threshold. Per calibration date, residual
        t_d = min {score percentile of the oracle top-k}. The set is
        {i : u_i ≥ q̂} with q̂ the finite-sample lower (1-α) quantile of
        {t_d}. Then P(oracle top-k ⊆ set) ≥ 1-α over exchangeable dates.

    Scores are always higher-is-better (as in ``models.ranking``). Pass
    ``higher_is_better=False`` when ``labels_or_true_ranks`` are 1=best ranks.
    """
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if int(k) < 1:
        raise ValueError("k must be >= 1")
    if not 0.0 < float(cal_frac) < 1.0:
        raise ValueError("cal_frac must be in (0, 1)")
    if guarantee not in ("fdr", "set_coverage"):
        raise ValueError("guarantee must be 'fdr' or 'set_coverage'")

    s = np.asarray(scores, dtype=float).reshape(-1)
    y = np.asarray(labels_or_true_ranks, dtype=float).reshape(-1)
    date_list = list(dates)
    if s.size != y.size or s.size != len(date_list):
        raise ValueError("scores, labels_or_true_ranks, and dates must align")

    groups = _unique_date_masks(date_list)
    n_cal = _cal_count(len(groups), cal_frac)
    cal_groups = groups[:n_cal]
    te_groups = groups[n_cal:]

    oracle = np.zeros(s.size, dtype=bool)
    pct = np.full(s.size, np.nan)
    for _key, idx in groups:
        oracle[idx] = oracle_topk_mask(y[idx], int(k), higher_is_better)
        pct[idx] = within_date_percentiles(s[idx])

    selected = np.zeros(s.size, dtype=bool)
    p_values = np.full(s.size, np.nan)
    qhat = 0.0

    if guarantee == "fdr":
        null_chunks = [pct[idx][~oracle[idx]] for _key, idx in cal_groups]
        null = np.concatenate(null_chunks) if null_chunks else np.asarray([], dtype=float)
        null = null[np.isfinite(null)]
        cutoffs: list[float] = []
        for _key, idx in te_groups:
            p = conformal_selection_pvalues(pct[idx], null)
            p_values[idx] = p
            reject, cutoff = benjamini_hochberg(p, float(alpha))
            selected[idx] = reject
            cutoffs.append(float(cutoff))
        qhat = float(np.mean(cutoffs)) if cutoffs else 0.0
    else:
        t_d: list[float] = []
        for _key, idx in cal_groups:
            u = pct[idx][oracle[idx]]
            u = u[np.isfinite(u)]
            if u.size:
                t_d.append(float(np.min(u)))
        qhat = _lower_conformal_quantile(np.asarray(t_d, dtype=float), float(alpha))
        for _key, idx in te_groups:
            u = pct[idx]
            selected[idx] = np.isfinite(u) & (u >= qhat)

    fdps: list[float] = []
    covers: list[float] = []
    sizes: list[float] = []
    for _key, idx in te_groups:
        fdp, cover, n_sel = _date_fdp_cover(selected, oracle, idx)
        fdps.append(fdp)
        covers.append(cover)
        sizes.append(float(n_sel))

    return TopKConformalResult(
        selected=selected,
        p_values=p_values,
        qhat=qhat,
        guarantee=str(guarantee),
        alpha=float(alpha),
        k=int(k),
        set_size=float(np.mean(sizes)) if sizes else 0.0,
        fdr=float(np.mean(fdps)) if fdps else float("nan"),
        coverage=float(np.mean(covers)) if covers else float("nan"),
        n_dates=int(len(te_groups)),
        n_cal_dates=int(n_cal),
    )


def planted_rank_panel(
    n_dates: int,
    n_names: int,
    noise: float,
    seed: int,
    date_shift: float = 0.0,
) -> tuple[Array, Array, NDArray[np.int64]]:
    """Scores = true + noise. Optional date-level offset (must not be stacked)."""
    rng = np.random.default_rng(int(seed))
    true = rng.normal(size=(int(n_dates), int(n_names)))
    if date_shift != 0.0:
        drift = date_shift * (
            np.arange(int(n_dates), dtype=float)[:, None] / max(int(n_dates) - 1, 1)
        )
        true = true + drift
    scores = true + rng.normal(scale=float(noise), size=true.shape)
    dates = np.repeat(np.arange(int(n_dates), dtype=np.int64), int(n_names))
    return scores.ravel(), true.ravel(), dates


def bench_conformal_topk(
    n_dates: int = 80,
    n_names: int = 40,
    k: int = 5,
    alpha: float = 0.20,
    noise: float = 0.20,
    seed: int = 18,
    guarantee: Guarantee = "fdr",
    cal_frac: float = 0.5,
) -> dict[str, float | str]:
    """Planted scores = true + noise. Set size, FDR or oracle-topk coverage, n_dates."""
    scores, labels, dates = planted_rank_panel(n_dates, n_names, noise, seed)
    result = conformal_topk(
        scores,
        labels,
        int(k),
        float(alpha),
        dates,
        guarantee=guarantee,
        cal_frac=float(cal_frac),
    )
    return {
        "set_size": result.set_size,
        "fdr": result.fdr,
        "coverage": result.coverage,
        "n_dates": float(result.n_dates),
        "alpha": float(alpha),
        "k": float(k),
        "seed": float(seed),
        "qhat": result.qhat,
        "guarantee": result.guarantee,
    }

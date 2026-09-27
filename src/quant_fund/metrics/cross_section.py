"""Date-grouped information coefficients and decile portfolios.

Cross-sectional statistics are never computed by stacking dates. Each
event_time is a group; inference is HAC on the resulting time series.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from quant_fund.metrics.inference import mean_tstat
from quant_fund.models.asset_pricing import date_groups

_IC_STD_FLOOR = 1e-15

Array = NDArray[np.float64]


def _date_keys(dates: NDArray[Any] | list[object] | Array) -> list[str]:
    keys: list[str] = []
    for d in dates:
        if isinstance(d, np.datetime64):
            keys.append(str(d))
        elif hasattr(d, "isoformat"):
            keys.append(d.isoformat())
        else:
            keys.append(str(d))
    return keys


@dataclass
class DateICResult:
    dates: list[object]
    pearson: Array
    spearman: Array
    mean_pearson: float
    mean_spearman: float
    t_pearson: float
    t_spearman: float
    p_pearson: float
    p_spearman: float
    icir_pearson: float
    icir_ann_pearson: float
    n_dates: int
    min_names: int


def _block_key(value: object) -> object:
    """Same label ``_date_keys`` would emit for one stamp."""
    if isinstance(value, np.datetime64):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _average_rank(x: Array) -> Array:
    """Average ranks, mergesort for ties. Matches ``scoring._rankdata``."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.shape[0], dtype=float)
    ranks[order] = np.arange(1, x.shape[0] + 1, dtype=float)
    _uniq, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.bincount(inverse, weights=ranks)
        ranks = sums[inverse] / counts[inverse]
    return ranks


def _pearson_pair(pred: Array, realized: Array) -> float:
    """Pearson on one date. Centered dot product; std gate matches ``pearson_ic``.

    ``np.corrcoef`` and this reduction differ by at most an ulp on finite
    inputs (two-pass sum of squares versus NumPy's covariance kernel).
    """
    mask = np.isfinite(pred) & np.isfinite(realized)
    if int(mask.sum()) < 3:
        return float("nan")
    left = pred[mask]
    right = realized[mask]
    if float(np.std(left)) < _IC_STD_FLOOR or float(np.std(right)) < _IC_STD_FLOOR:
        return float("nan")
    left = left - left.mean()
    right = right - right.mean()
    left_ss = float(np.dot(left, left))
    right_ss = float(np.dot(right, right))
    return float(np.dot(left, right) / np.sqrt(left_ss * right_ss))


def _aligned_groups(
    scores: Array, y: Array, dates: Array
) -> tuple[Array, Array, list[NDArray[np.intp]], np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(dates) != len(scores) or len(y) != len(scores):
        raise ValueError("scores, y, dates must align")
    date_arr = np.asarray(dates)
    groups = date_groups(date_arr)
    return scores.reshape(-1), y.reshape(-1), groups, date_arr.reshape(-1)


def _row_average_ranks(block: Array) -> Array:
    """Average ranks per row. Ties use the same mergesort rule as ``_average_rank``."""
    order = np.argsort(block, axis=1, kind="mergesort")
    ranks = np.empty(block.shape, dtype=np.float64)
    ranks[np.arange(block.shape[0])[:, None], order] = np.arange(
        1, block.shape[1] + 1, dtype=np.float64
    )
    sorted_vals = np.take_along_axis(block, order, axis=1)
    tied = np.any(sorted_vals[:, 1:] == sorted_vals[:, :-1], axis=1)
    for i in np.flatnonzero(tied):
        ranks[i] = _average_rank(block[i])
    return ranks


def _row_pearson(left: Array, right: Array) -> Array:
    """One Pearson per row. Std gate matches ``_pearson_pair`` (``ddof=0``)."""
    centered_l = left - left.mean(axis=1, keepdims=True)
    centered_r = right - right.mean(axis=1, keepdims=True)
    left_ss = np.sum(centered_l * centered_l, axis=1)
    right_ss = np.sum(centered_r * centered_r, axis=1)
    n = left.shape[1]
    ok = (np.sqrt(left_ss / n) >= _IC_STD_FLOOR) & (np.sqrt(right_ss / n) >= _IC_STD_FLOOR)
    out = np.full(left.shape[0], np.nan, dtype=np.float64)
    out[ok] = np.sum(centered_l * centered_r, axis=1)[ok] / np.sqrt(left_ss[ok] * right_ss[ok])
    return out


def _balanced_ic(
    groups: list[NDArray[np.intp]],
    scores: Array,
    y: Array,
    date_arr: np.ndarray,
    min_names: int,
) -> tuple[list[object], Array, Array] | None:
    """Vectorized IC when every date has the same count of finite names.

    Unequal dates and non-finite rows stay on the per-date loop. The centered
    dot product matches ``np.corrcoef`` to about one ulp.
    """
    if not groups:
        return None
    width = int(groups[0].size)
    if width < min_names or any(int(idx.size) != width for idx in groups):
        return None
    index = np.stack(groups)
    left = scores[index]
    right = y[index]
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return None
    pearson = _row_pearson(left, right)
    spearman = _row_pearson(_row_average_ranks(left), _row_average_ranks(right))
    kept = [_block_key(date_arr[int(row[0])]) for row in index]
    return kept, pearson, spearman


def date_ic_series(
    scores: Array,
    y: Array,
    dates: Array,
    *,
    min_names: int = 5,
    hac_lags: int | None = None,
) -> DateICResult:
    """Pearson/Spearman IC per date, then HAC t-stats on the IC series."""
    scores, y, groups, date_arr = _aligned_groups(scores, y, dates)
    balanced = _balanced_ic(groups, scores, y, date_arr, min_names)
    if balanced is not None:
        kept, p, r = balanced
    else:
        pearsons: list[float] = []
        spearmans: list[float] = []
        kept = []
        for idx in groups:
            if idx.size < min_names:
                continue
            s = scores[idx]
            t = y[idx]
            mask = np.isfinite(s) & np.isfinite(t)
            if int(mask.sum()) < min_names:
                continue
            s = s[mask]
            t = t[mask]
            pearsons.append(_pearson_pair(s, t))
            spearmans.append(_pearson_pair(_average_rank(s), _average_rank(t)))
            kept.append(_block_key(date_arr[int(idx[0])]))
        p = np.asarray(pearsons, dtype=float)
        r = np.asarray(spearmans, dtype=float)
    if p.size < 3:
        nan = float("nan")
        return DateICResult(
            dates=kept,
            pearson=p,
            spearman=r,
            mean_pearson=nan,
            mean_spearman=nan,
            t_pearson=nan,
            t_spearman=nan,
            p_pearson=nan,
            p_spearman=nan,
            icir_pearson=nan,
            icir_ann_pearson=nan,
            n_dates=int(p.size),
            min_names=min_names,
        )
    mp, tp, pp = mean_tstat(p, hac_lags)
    ms, ts, ps = mean_tstat(r, hac_lags)
    sd = float(np.std(p, ddof=1))
    icir = float(mp / sd) if sd > 0 else float("nan")
    icir_ann = float(icir * np.sqrt(252.0)) if np.isfinite(icir) else float("nan")
    return DateICResult(
        dates=kept,
        pearson=p,
        spearman=r,
        mean_pearson=mp,
        mean_spearman=ms,
        t_pearson=tp,
        t_spearman=ts,
        p_pearson=pp,
        p_spearman=ps,
        icir_pearson=icir,
        icir_ann_pearson=icir_ann,
        n_dates=int(p.size),
        min_names=min_names,
    )


@dataclass
class DecileResult:
    n_buckets: int
    mean_returns: list[float]
    monotonicity: float
    long_short: Array
    mean_ls: float
    t_ls: float
    p_ls: float
    hit_rate: float
    dates: list[object] = field(default_factory=list)


def _spearman_vs_index(means: list[float]) -> float:
    x = np.arange(1, len(means) + 1, dtype=float)
    y = np.asarray(means, dtype=float)
    mask = np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    rho, _ = stats.spearmanr(x[mask], y[mask])
    return float(rho)


def decile_portfolios(
    scores: Array,
    y: Array,
    dates: Array,
    *,
    n_buckets: int = 5,
    min_names: int = 5,
    hac_lags: int | None = None,
) -> DecileResult:
    """Equal-weight bucket returns by within-date score rank. Long high, short low."""
    if n_buckets < 2:
        raise ValueError("n_buckets must be >= 2")
    if min_names < 1:
        raise ValueError("min_names must be >= 1")
    scores, y, groups, date_arr = _aligned_groups(scores, y, dates)
    need = max(min_names, n_buckets)
    bucket_rets: list[list[float]] = [[] for _ in range(n_buckets)]
    ls: list[float] = []
    kept: list[object] = []
    for idx in groups:
        if idx.size < need:
            continue
        s = scores[idx]
        target = y[idx]
        mask = np.isfinite(s) & np.isfinite(target)
        if int(mask.sum()) < need:
            continue
        s, target = s[mask], target[mask]
        ranks = stats.rankdata(s, method="average")
        # 0 = lowest score, n_buckets-1 = highest
        edges = np.floor((ranks - 1.0) / ranks.size * n_buckets).astype(int)
        edges = np.clip(edges, 0, n_buckets - 1)
        means = []
        ok = True
        for b in range(n_buckets):
            sel = target[edges == b]
            if sel.size == 0:
                ok = False
                break
            means.append(float(np.mean(sel)))
        if not ok:
            continue
        for b, m in enumerate(means):
            bucket_rets[b].append(m)
        ls.append(means[-1] - means[0])
        kept.append(_block_key(date_arr[int(idx[0])]))
    mean_returns = [float(np.mean(v)) if v else float("nan") for v in bucket_rets]
    series = np.asarray(ls, dtype=float)
    if series.size >= 3:
        mu, t_ls, p_ls = mean_tstat(series, hac_lags)
    else:
        mu, t_ls, p_ls = float("nan"), float("nan"), float("nan")
    hit = float(np.mean(series > 0)) if series.size else float("nan")
    return DecileResult(
        n_buckets=n_buckets,
        mean_returns=mean_returns,
        monotonicity=_spearman_vs_index(mean_returns),
        long_short=series,
        mean_ls=mu,
        t_ls=t_ls,
        p_ls=p_ls,
        hit_rate=hit,
        dates=kept,
    )

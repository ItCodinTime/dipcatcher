"""Date-grouped information coefficients and decile portfolios.

Cross-sectional statistics are never computed by stacking dates. Each
event_time is a group; inference is HAC on the resulting time series.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats

from quant_fund.metrics.inference import mean_tstat
from quant_fund.metrics.scoring import pearson_ic, rank_ic

Array = NDArray[np.float64]


def _date_keys(dates: NDArray[Any] | list[object] | Array) -> list[str]:
    keys: list[str] = []
    for d in dates:
        if isinstance(d, np.datetime64):
            keys.append(str(d))
        elif hasattr(d, "isoformat"):
            keys.append(d.isoformat())  # type: ignore[no-untyped-call]
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


def date_ic_series(
    scores: Array,
    y: Array,
    dates: Array,
    *,
    min_names: int = 5,
    hac_lags: int | None = None,
) -> DateICResult:
    """Pearson/Spearman IC per date, then HAC t-stats on the IC series."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(dates) != len(scores) or len(y) != len(scores):
        raise ValueError("scores, y, dates must align")
    frame = pl.DataFrame({"score": scores, "y": y, "date": _date_keys(dates)}).drop_nulls()
    pearsons: list[float] = []
    spearmans: list[float] = []
    kept: list[object] = []
    for date, grp in frame.group_by("date", maintain_order=True):
        if grp.height < min_names:
            continue
        s = grp["score"].to_numpy().astype(float)
        t = grp["y"].to_numpy().astype(float)
        mask = np.isfinite(s) & np.isfinite(t)
        if int(mask.sum()) < min_names:
            continue
        pearsons.append(pearson_ic(s[mask], t[mask]))
        spearmans.append(rank_ic(s[mask], t[mask]))
        kept.append(date[0] if isinstance(date, tuple) else date)
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
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(dates) != len(scores) or len(y) != len(scores):
        raise ValueError("scores, y, dates must align")
    frame = pl.DataFrame({"score": scores, "y": y, "date": _date_keys(dates)}).drop_nulls()
    bucket_rets: list[list[float]] = [[] for _ in range(n_buckets)]
    ls: list[float] = []
    kept: list[object] = []
    for date, grp in frame.group_by("date", maintain_order=True):
        if grp.height < max(min_names, n_buckets):
            continue
        s = grp["score"].to_numpy().astype(float)
        target = grp["y"].to_numpy().astype(float)
        mask = np.isfinite(s) & np.isfinite(target)
        if int(mask.sum()) < max(min_names, n_buckets):
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
        kept.append(date[0] if isinstance(date, tuple) else date)
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

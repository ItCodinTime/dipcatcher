"""Krauss (2017) statistical-arbitrage sliding window on a CS panel.

davidalmeida90/machine-learning-for-finance used DNN/RF/GBT on next-day
return signs. ADR-007 keeps neural nets optional; this module is the
linear baseline: walk-forward logistic on features known at ``t``,
predicting whether next-day idio return is above the date median.
Research only. Trees already live in catalog ``gbrt``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression

Array = NDArray[np.float64]


def krauss_walk_forward_proba(
    x: ArrayLike,
    y: ArrayLike,
    dates: ArrayLike,
    *,
    window_dates: int = 252,
    min_names: int = 8,
    C: float = 1.0,
) -> Array:
    """OOS predicted P(y > date-median) using a trailing date window.

    ``x`` is ``(n_rows, n_features)`` aligned with ``y`` and ``dates``.
    Training uses rows with ``date < d`` inside the window; labels are
    the sign of ``y`` versus that training date's cross-sectional median.
    """
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float).reshape(-1)
    dd = np.asarray(dates)
    if xx.shape[0] != yy.size:
        raise ValueError("x and y length mismatch")
    uniq = np.unique(dd)
    uniq = np.sort(uniq)
    out = np.full(yy.size, np.nan, dtype=float)
    for i, d in enumerate(uniq):
        if i < window_dates:
            continue
        train_dates = uniq[i - window_dates : i]
        train_m = np.isin(dd, train_dates)
        test_m = dd == d
        if train_m.sum() < min_names * 8 or test_m.sum() < min_names:
            continue
        y_tr = yy[train_m]
        labels = np.zeros(train_m.sum(), dtype=int)
        train_d = dd[train_m]
        for td in train_dates:
            m = train_d == td
            block = y_tr[m]
            if block.size == 0:
                continue
            med = np.nanmedian(block)
            labels[m] = (block > med).astype(int)
        x_tr = xx[train_m]
        finite = np.isfinite(x_tr).all(axis=1) & np.isfinite(y_tr)
        if finite.sum() < min_names * 4 or len(np.unique(labels[finite])) < 2:
            continue
        clf = LogisticRegression(C=C, max_iter=200, solver="lbfgs")
        clf.fit(x_tr[finite], labels[finite])
        x_te = xx[test_m]
        te_ok = np.isfinite(x_te).all(axis=1)
        proba = np.full(test_m.sum(), np.nan)
        if te_ok.any():
            proba[te_ok] = clf.predict_proba(x_te[te_ok])[:, 1]
        out[test_m] = proba
    return out


def krauss_hit_rate(y: ArrayLike, proba: ArrayLike, dates: ArrayLike) -> dict[str, Any]:
    """Date-level hit rate of predicted CS winners vs realised above-median."""
    yy = np.asarray(y, dtype=float).reshape(-1)
    pp = np.asarray(proba, dtype=float).reshape(-1)
    dd = np.asarray(dates)
    hits = []
    for d in np.unique(dd):
        m = dd == d
        if not np.isfinite(pp[m]).any():
            continue
        med = np.nanmedian(yy[m])
        pred = pp[m] > 0.5
        real = yy[m] > med
        ok = np.isfinite(pp[m])
        if ok.sum() == 0:
            continue
        hits.append(float(np.mean(pred[ok] == real[ok])))
    arr = np.asarray(hits, dtype=float)
    return {
        "n_dates": int(arr.size),
        "hit_rate": float(np.nanmean(arr)) if arr.size else float("nan"),
        "research_only": True,
        "live_pnl_claim": False,
    }

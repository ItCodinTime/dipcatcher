"""Frozen sleeves on the file tape. Research only.

Pre-registered before the fit (dipcatcher.sota.v2 follow-on, not a new
ranker champion):

1. Ungated top-5 12–1 (already raced).
2. Gaussian HMM gate: fit on SPY through the selection cut, forward-filter
   only, hold the top-5 book only in the selection-sample high-mean state.
   Size is 0 or 1. No sign flip.
3. Linear sleeve policy: actions ``{cash, SPY, top-5}``. Ridge of next-bar
   sleeve return on lagged features, fit on selection rows only.
4. Declared lead rule: hold QQQ when the trailing 10-day XLK excess over
   SPY is positive; else TLT when its own 10-day return is positive; else
   cash.

Holdout starts 2025-01-02 and is not used to choose a state, a coefficient,
or the lead rule. ``blend_weight`` stays 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
IntArray = NDArray[np.intp]
_EPS = 1e-12


def trailing_mean(returns: Array, window: int) -> Array:
    """Mean of ``returns[t-window+1:t+1]``. Earlier bars are NaN."""
    r = np.asarray(returns, dtype=float).reshape(-1)
    out = np.full(r.size, np.nan)
    w = max(int(window), 1)
    for i in range(w - 1, r.size):
        seg = r[i - w + 1 : i + 1]
        if np.isfinite(seg).all():
            out[i] = float(seg.mean())
    return out


def trailing_vol(returns: Array, window: int) -> Array:
    r = np.asarray(returns, dtype=float).reshape(-1)
    out = np.full(r.size, np.nan)
    w = max(int(window), 2)
    for i in range(w - 1, r.size):
        seg = r[i - w + 1 : i + 1]
        if np.isfinite(seg).all():
            out[i] = float(np.std(seg, ddof=1))
    return out


def sleeve_features(spy: Array, book: Array) -> Array:
    """Features known at the close of bar t. Row t must not be used to trade bar t."""
    spy = np.asarray(spy, dtype=float).reshape(-1)
    book = np.asarray(book, dtype=float).reshape(-1)
    if spy.shape != book.shape:
        raise ValueError("spy and book returns must align")
    return np.column_stack(
        [
            trailing_mean(spy, 21),
            trailing_mean(spy, 63),
            trailing_vol(spy, 21),
            trailing_mean(book, 21),
        ]
    )


def fit_sleeve_policy(
    features: Array,
    sleeves: Array,
    train: Array,
) -> tuple[Array, Array]:
    """Ridge map from features at t-1 to the three sleeve returns at t.

    ``train[t]`` is True when bar t may be used as a label. The feature row
    is ``t-1``, which must also be inside the training span.
    """
    x = np.asarray(features, dtype=float)
    y = np.asarray(sleeves, dtype=float)
    mask = np.asarray(train, dtype=bool).reshape(-1)
    if y.ndim != 2 or y.shape[1] != 3 or y.shape[0] != x.shape[0] or mask.shape[0] != x.shape[0]:
        raise ValueError("features, sleeves (T, 3), and train mask must align")
    rows_x: list[Array] = []
    rows_y: list[Array] = []
    for t in range(1, x.shape[0]):
        if not mask[t] or not mask[t - 1]:
            continue
        if not (np.isfinite(x[t - 1]).all() and np.isfinite(y[t]).all()):
            continue
        rows_x.append(x[t - 1])
        rows_y.append(y[t])
    if len(rows_x) < 30:
        raise ValueError("sleeve policy needs at least 30 training rows")
    design = np.column_stack([np.ones(len(rows_x)), np.vstack(rows_x)])
    target = np.vstack(rows_y)
    coef, *_ = np.linalg.lstsq(design, target, rcond=1e-3)
    intercept = np.asarray(coef[0], dtype=float)
    slopes = np.asarray(coef[1:], dtype=float)
    return intercept, slopes


def apply_sleeve_policy(
    features: Array,
    sleeves: Array,
    intercept: Array,
    slopes: Array,
    *,
    one_way_cost: float = 0.001,
) -> tuple[Array, IntArray]:
    """Trade bar t with features through t-1. Cost is charged when the action changes."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(sleeves, dtype=float)
    pnl = np.zeros(y.shape[0], dtype=float)
    actions = np.zeros(y.shape[0], dtype=np.intp)
    prev = 0
    for t in range(1, y.shape[0]):
        row = x[t - 1]
        if not np.isfinite(row).all():
            choice = 0
        else:
            pred = intercept + row @ slopes
            choice = int(np.argmax(pred))
        actions[t] = choice
        turn = 0.0 if choice == prev else 1.0
        prev = choice
        pnl[t] = float(y[t, choice]) - float(one_way_cost) * turn
    return pnl, actions


def regime_gate_returns(
    market: Array,
    book: Array,
    train: Array,
    *,
    n_states: int = 2,
    vol_window: int = 21,
    one_way_cost: float = 0.001,
    seed: int = 7,
) -> Array:
    """Hold ``book`` only in the selection-sample high-mean HMM state.

    The model is fit on training rows. The path is a forward filter.
    The scale at t uses the filtered state at t-1.
    """
    from quant_fund.models.regime import GaussianHMMRegime

    mkt = np.asarray(market, dtype=float).reshape(-1)
    bk = np.asarray(book, dtype=float).reshape(-1)
    mask = np.asarray(train, dtype=bool).reshape(-1)
    if mkt.shape != bk.shape or mask.shape != mkt.shape:
        raise ValueError("market, book, and train mask must align")
    vol = trailing_vol(mkt, vol_window)
    obs = np.column_stack([mkt, np.where(np.isfinite(vol), vol, 0.0)])
    fit_rows = mask & np.isfinite(mkt) & np.isfinite(vol)
    if int(fit_rows.sum()) < 80:
        raise ValueError("HMM gate needs at least 80 training rows")
    model = GaussianHMMRegime(n_states=n_states, seed=seed)
    model.fit(obs[fit_rows])
    filtered = model.predict_proba(obs)
    state = np.argmax(filtered, axis=1)
    means = np.full(n_states, -np.inf)
    for k in range(n_states):
        chosen = mask & (np.roll(state, 1) == k)
        chosen[0] = False
        sample = bk[chosen]
        sample = sample[np.isfinite(sample)]
        if sample.size >= 20:
            means[k] = float(sample.mean())
    allowed = int(np.argmax(means)) if np.isfinite(means).any() else -1
    if allowed < 0 or not np.isfinite(means[allowed]) or means[allowed] <= 0.0:
        allowed = -1
    scale = np.zeros(mkt.size, dtype=float)
    pnl = np.zeros(mkt.size, dtype=float)
    prev = 0.0
    for t in range(1, mkt.size):
        on = 1.0 if state[t - 1] == allowed else 0.0
        scale[t] = on
        turn = abs(on - prev)
        prev = on
        pnl[t] = on * float(bk[t]) - float(one_way_cost) * turn
    return pnl


def lead_rule_returns(
    xlk_excess: Array,
    qqq: Array,
    tlt: Array,
    *,
    window: int = 10,
    one_way_cost: float = 0.001,
) -> Array:
    """QQQ if lagged XLK excess is positive, else TLT if lagged TLT is positive, else cash."""
    excess = trailing_mean(np.asarray(xlk_excess, dtype=float), window)
    tlt_mean = trailing_mean(np.asarray(tlt, dtype=float), window)
    q = np.asarray(qqq, dtype=float).reshape(-1)
    b = np.asarray(tlt, dtype=float).reshape(-1)
    pnl = np.zeros(q.size, dtype=float)
    prev = 0
    for t in range(1, q.size):
        if np.isfinite(excess[t - 1]) and excess[t - 1] > 0.0:
            choice = 1
            ret = q[t]
        elif np.isfinite(tlt_mean[t - 1]) and tlt_mean[t - 1] > 0.0:
            choice = 2
            ret = b[t]
        else:
            choice = 0
            ret = 0.0
        turn = 0.0 if choice == prev else 1.0
        prev = choice
        pnl[t] = float(ret) - float(one_way_cost) * turn
    return pnl


def policy_snapshot(actions: Array | IntArray) -> dict[str, Any]:
    a = np.asarray(actions, dtype=int)
    names = ("cash", "spy", "topk")
    counts = {names[i]: int(np.sum(a == i)) for i in range(3)}
    return {"action_counts": counts, "live_pnl_claim": False}


def etf_dual_momentum(
    prices: Array,
    *,
    defensive: int,
    top_k: int = 1,
    lookback: int = 252,
    skip: int = 21,
    delay: int = 1,
    rebalance_every: int = 21,
    one_way_cost: float = 0.001,
) -> Array:
    """GEM-style rotation. Risky names must beat both cash and the defensive asset.

    ``defensive`` is a column index. Decision at t uses prices through t-delay.
    """
    px = np.asarray(prices, dtype=float)
    if px.ndim != 2:
        raise ValueError("prices must be (T, N)")
    if not 0 <= int(defensive) < px.shape[1]:
        raise ValueError("defensive column is out of range")
    rets = np.zeros_like(px)
    rets[1:] = px[1:] / np.maximum(px[:-1], _EPS) - 1.0
    rets[~np.isfinite(rets)] = 0.0
    t_len, n_names = px.shape
    w = np.zeros((t_len, n_names), dtype=float)
    step = max(int(rebalance_every), 1)
    delay = max(int(delay), 0)
    start = int(lookback) + delay
    held: Array | None = None
    for t in range(start, t_len):
        if held is not None and (t - start) % step != 0:
            w[t] = held
            continue
        end = t - delay
        past = end - int(lookback)
        skip_i = end - int(skip)
        raw = np.zeros(n_names, dtype=float)
        if past >= 0 and skip_i > past and px[past, int(defensive)] > _EPS:
            mom = px[skip_i] / np.maximum(px[past], _EPS) - 1.0
            d_mom = float(mom[int(defensive)])
            risky = [
                i
                for i in range(n_names)
                if i != int(defensive) and np.isfinite(mom[i]) and mom[i] > 0.0 and mom[i] > d_mom
            ]
            risky.sort(key=lambda i: float(mom[i]), reverse=True)
            take = risky[: max(int(top_k), 1)]
            if take:
                raw[take] = 1.0 / float(len(take))
            elif np.isfinite(d_mom) and d_mom > 0.0:
                raw[int(defensive)] = 1.0
        w[t] = raw
        held = raw
    pnl = np.sum(w * rets, axis=1)
    turn = np.zeros(t_len, dtype=float)
    turn[1:] = np.sum(np.abs(w[1:] - w[:-1]), axis=1)
    return pnl - float(one_way_cost) * turn


def ucb_then_freeze(
    sleeves: Array,
    train: Array,
    *,
    c: float = 2.0,
    one_way_cost: float = 0.001,
) -> tuple[Array, int]:
    """UCB1 on training labels, then hold the empirical-best arm through the holdout.

    The arm for bar t is chosen before that bar's reward is seen. Holdout
    rewards are not used to update.
    """
    from quant_fund.models.bandits import UCB1

    y = np.asarray(sleeves, dtype=float)
    mask = np.asarray(train, dtype=bool).reshape(-1)
    if y.ndim != 2 or mask.shape[0] != y.shape[0]:
        raise ValueError("sleeves (T, K) and train mask must align")
    bandit = UCB1(int(y.shape[1]), c=c)
    pnl = np.zeros(y.shape[0], dtype=float)
    prev = -1
    frozen: int | None = None
    for t in range(1, y.shape[0]):
        if mask[t]:
            arm = bandit.select()
            reward = float(y[t, arm])
            if np.isfinite(reward):
                bandit.update(arm, reward)
        else:
            if frozen is None:
                frozen = int(np.argmax(bandit.values))
            arm = frozen
        turn = 0.0 if arm == prev else 1.0
        prev = arm
        reward = float(y[t, arm])
        pnl[t] = (reward if np.isfinite(reward) else 0.0) - float(one_way_cost) * turn
    if frozen is None:
        frozen = int(np.argmax(bandit.values))
    return pnl, frozen

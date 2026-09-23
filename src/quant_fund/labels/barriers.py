"""Event-driven labeling and sample weights (AFML ch. 2-4).

Pure-numpy implementations of the *Advances in Financial Machine Learning*
labeling canon: cUSUM event filters, the triple-barrier method, meta-labeling,
trend-scanning labels, and the concurrency/uniqueness sample-weight machinery
that honest walk-forward evaluation needs.

References:
- Lopez de Prado (2018). *Advances in Financial Machine Learning* - ch. 2
  (cUSUM filter), ch. 3 (triple barrier, meta-labeling, trend scanning),
  ch. 4 (average uniqueness, sequential bootstrap, time-decay weights).
- Lam, Yam (1997). CUSUM techniques for technical trading.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
IdxArray = NDArray[np.intp]


def _as_vector(x: Array, name: str = "x", *, min_obs: int = 2) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size < min_obs:
        raise ValueError(f"{name} must contain at least {min_obs} finite observations")
    return v


def cusum_filter(x: Array, h: float) -> IdxArray:
    """Symmetric CUSUM event filter (AFML 2.5.2.1; Lam-Yam 1997).

    Emits indices where the cumulative deviation of ``x`` from its running
    reference crosses ``h`` (positive or negative).  The reference resets to
    the value at each event, so events mark *structural* moves, not noise
    around a rolling mean.
    """
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("h must be positive and finite")
    v = _as_vector(x)
    events: list[int] = []
    ref = v[0]
    s_pos = s_neg = 0.0
    for t in range(1, v.size):
        s_pos = max(0.0, s_pos + v[t] - ref)
        s_neg = min(0.0, s_neg + v[t] - ref)
        if s_neg < -h or s_pos > h:
            events.append(t)
            ref = v[t]
            s_pos = s_neg = 0.0
    return np.asarray(events, dtype=np.intp)


def triple_barrier(
    close: Array,
    events: IdxArray,
    pt: float,
    sl: float,
    horizon: int,
    vol: Array | None = None,
    *,
    min_ret: float = 0.0,
) -> dict[str, Array]:
    """Triple-barrier labeling (AFML ch. 3).

    For each event index ``t``, the label is determined by which barrier the
    forward price path hits first:

    - ``+1`` - upper barrier ``close[t] * (1 + pt * sigma_t)``
    - ``-1`` - lower barrier ``close[t] * (1 - sl * sigma_t)``
    - ``0``  - vertical barrier (horizon elapsed) when ``min_ret > 0`` and the
      path return is smaller than ``min_ret``; otherwise the sign of the path
      return.

    ``vol`` supplies a per-bar ``sigma_t`` scale (e.g. EWMA of returns);
    without it barriers are absolute fractions of the event price.  Returns
    per-event ``label``, ``ret`` (touch/horizon return), ``touch`` (barrier
    id: +1/-1/0) and ``t_touch`` (index of first touch or horizon end).
    """
    c = _as_vector(close, "close")
    if not np.isfinite(pt) or pt <= 0.0 or not np.isfinite(sl) or sl <= 0.0:
        raise ValueError("pt and sl must be positive and finite")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if not np.isfinite(min_ret) or min_ret < 0.0:
        raise ValueError("min_ret must be non-negative and finite")
    ev = np.asarray(events, dtype=np.intp).reshape(-1)
    ev = ev[(ev >= 0) & (ev < c.size)]
    if ev.size == 0:
        raise ValueError("events must contain valid indices")
    sigma = np.ones(c.size)
    if vol is not None:
        sigma = np.asarray(vol, dtype=float).reshape(-1)
        if sigma.size != c.size or not np.isfinite(sigma).all() or np.any(sigma <= 0.0):
            raise ValueError("vol must match close, be finite and positive")
    labels = np.zeros(ev.size)
    touched = np.zeros(ev.size)
    ret_out = np.zeros(ev.size)
    t_touch = np.zeros(ev.size, dtype=np.intp)
    for i, t in enumerate(ev):
        end = min(int(t) + horizon, c.size - 1)
        path = c[int(t) + 1 : end + 1] / c[int(t)] - 1.0
        if path.size == 0:
            ret_out[i] = 0.0
            t_touch[i] = t
            continue
        up = pt * sigma[t]
        dn = -sl * sigma[t]
        label, touch, idx = 0.0, 0.0, path.size - 1
        for j, r in enumerate(path):
            if r >= up:
                label, touch, idx = 1.0, 1.0, j
                break
            if r <= dn:
                label, touch, idx = -1.0, -1.0, j
                break
        r_final = float(path[idx])
        if touch == 0.0:
            label = np.sign(r_final) if min_ret <= 0.0 or abs(r_final) >= min_ret else 0.0
        labels[i] = label
        touched[i] = touch
        ret_out[i] = r_final
        t_touch[i] = t + 1 + idx
    return {
        "label": labels,
        "touch": touched,
        "ret": ret_out,
        "t_touch": t_touch.astype(float),
        "event_idx": ev.astype(float),
    }


def meta_labels(side: Array, barrier_labels: Array) -> Array:
    """Meta-labeling (AFML 3.7): binary label = ``side`` *is correct*.

    ``side`` is a primary model's position sign in {-1, 0, +1}; the meta-label
    is 1 when the barrier label agrees with the side, else 0.
    """
    s = np.asarray(side, dtype=float).reshape(-1)
    label = np.asarray(barrier_labels, dtype=float).reshape(-1)
    if s.size != label.size or s.size == 0:
        raise ValueError("side and barrier_labels must be non-empty and equal length")
    if not np.isfinite(s).all() or not np.isfinite(label).all():
        raise ValueError("side and barrier_labels must be finite")
    if not np.all(np.isin(np.unique(s), (-1.0, 0.0, 1.0))):
        raise ValueError("side must take values in {-1, 0, +1}")
    return (s * label > 0.0).astype(float)


def trend_scanning_labels(
    close: Array, window: int = 20, *, min_t: float = 0.0
) -> Array:
    """Trend-scanning labels (AFML 5.4.1): sign of the best-fit trend t-stat.

    For each origin, regress log-price on time over trailing windows of length
    ``2..window``; the label is the sign of the slope t-stat of the *best* (most
    significant) fit - ``nan`` until the first full window is available.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 3:
        raise ValueError("window must be an integer >= 3")
    if not np.isfinite(min_t) or min_t < 0.0:
        raise ValueError("min_t must be non-negative and finite")
    c = _as_vector(close, "close", min_obs=window + 1)
    lp = np.log(c)
    n = c.size
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        best_t, best_abs = 0.0, -1.0
        for w in range(2, window + 1):
            seg = lp[t - w + 1 : t + 1]
            tt = np.arange(w, dtype=float)
            tt -= tt.mean()
            denom = float(tt @ tt)
            beta = float(tt @ (seg - seg.mean()) / denom)
            resid = seg - (seg.mean() + beta * tt)
            dof = w - 2
            s2 = float(resid @ resid / dof) if dof > 0 else 0.0
            se = np.sqrt(s2 / denom) if denom > 0 else np.inf
            t_stat = beta / se if se > 0 else 0.0
            if abs(t_stat) > best_abs:
                best_abs, best_t = abs(t_stat), t_stat
        out[t] = np.sign(best_t) if abs(best_t) >= min_t else 0.0
    return out


def label_concurrency(t_start: np.ndarray, t_end: np.ndarray) -> Array:
    """AFML 4.1 concurrency count: overlapping labels active at each bar.

    ``t_start``/``t_end`` are integer bar indices of label windows (inclusive
    start, exclusive end).  Returns the count of windows covering each bar.
    """
    s = np.asarray(t_start, dtype=np.intp).reshape(-1)
    e = np.asarray(t_end, dtype=np.intp).reshape(-1)
    if s.size != e.size or s.size == 0:
        raise ValueError("t_start and t_end must be non-empty equal-length")
    if np.any(e < s):
        raise ValueError("t_end must be >= t_start")
    n = int(e.max()) + 1
    counts = np.zeros(n)
    for a, b in zip(s, e, strict=True):
        counts[a:b] += 1.0
    return counts


def average_uniqueness(t_start: np.ndarray, t_end: np.ndarray) -> Array:
    """AFML 4.2 average uniqueness ``u_i = mean_t 1/c_t`` over each label's span."""
    s = np.asarray(t_start, dtype=np.intp).reshape(-1)
    e = np.asarray(t_end, dtype=np.intp).reshape(-1)
    if s.size != e.size or s.size == 0:
        raise ValueError("t_start and t_end must be non-empty equal-length")
    if np.any(e < s):
        raise ValueError("t_end must be >= t_start")
    counts = label_concurrency(s, e)
    u = np.empty(s.size)
    for i, (a, b) in enumerate(zip(s, e, strict=True)):
        u[i] = float(np.mean(1.0 / counts[a:b]))
    return u


def sequential_bootstrap(
    t_start: Array, t_end: Array, n_samples: int, seed: int = 0
) -> IdxArray:
    """AFML 4.3 sequential bootstrap: draw labels proportional to uniqueness.

    Each iteration picks an index with probability proportional to its current
    average uniqueness, then *removes* that window's contribution to the
    concurrency matrix so overlapping labels lose draw weight on later picks.
    """
    u = average_uniqueness(t_start, t_end)
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    rng = np.random.default_rng(seed)
    s = np.asarray(t_start, dtype=np.intp)
    e = np.asarray(t_end, dtype=np.intp)
    indicator = np.zeros((u.size, int(e.max()) + 1))
    for i, (a, b) in enumerate(zip(s, e, strict=True)):
        indicator[i, a:b] = 1.0
    counts = indicator.sum(axis=0)
    chosen = np.empty(n_samples, dtype=np.intp)
    for k in range(n_samples):
        probs = u / u.sum()
        pick = int(rng.choice(u.size, p=probs))
        chosen[k] = pick
        counts -= indicator[pick]
        counts = np.maximum(counts, 1.0)
        for i in range(u.size):
            u[i] = float(np.mean(1.0 / counts[s[i] : e[i]]))
    return chosen


def time_decay_weights(ages: Array, decay: float = 0.5) -> Array:
    """AFML 4.10 piecewise time-decay weights.

    Weights decay linearly from 1 (newest) toward ``decay`` across the observed
    age span, floored at ``decay``.  ``ages`` are non-negative recency offsets
    (``age = last_index - event_index``); equal ages get equal weights.
    """
    a = np.asarray(ages, dtype=float).reshape(-1)
    if a.size == 0 or not np.isfinite(a).all():
        raise ValueError("ages must be non-empty and finite")
    if np.any(a < 0.0):
        raise ValueError("ages must be non-negative")
    if not np.isfinite(decay) or not (0.0 <= decay <= 1.0):
        raise ValueError("decay must be in [0, 1]")
    span = float(a.max()) + 1.0
    w = 1.0 - (1.0 - decay) * (a / span)
    return np.clip(w, decay, 1.0)


def return_attribution_weights(labels: Array, t_start: Array, t_end: Array) -> Array:
    """AFML 4.5 return-attribution weights: ``|ret_i|`` shared across concurrency.

    Weight proportional to the label's absolute return divided by how many
    overlapping labels are active over its span - events concurrent with many
    others get down-weighted.
    """
    r = np.asarray(labels, dtype=float).reshape(-1)
    s = np.asarray(t_start, dtype=np.intp).reshape(-1)
    e = np.asarray(t_end, dtype=np.intp).reshape(-1)
    if r.size != s.size or s.size != e.size or r.size == 0:
        raise ValueError("labels, t_start and t_end must be non-empty equal-length")
    if not np.isfinite(r).all():
        raise ValueError("labels must be finite returns")
    counts = label_concurrency(s, e)
    w = np.empty(r.size)
    for i, (a, b) in enumerate(zip(s, e, strict=True)):
        share = np.abs(r[i]) / np.maximum(counts[a:b], 1.0)
        w[i] = float(share.sum())
    return w

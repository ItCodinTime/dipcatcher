"""Technical-analysis indicator canon (pure NumPy, NaN warmup).

All functions take/return float64 vectors; the first ``window - 1`` entries
are NaN (warmup).  Inputs are fail-closed: non-finite values or
``window > n`` raise ``ValueError``.

References (originators):
- Wilder (1978). *New Concepts in Technical Trading Systems* — RSI, ATR,
  DMI/ADX, Parabolic SAR.
- Appel (1979). MACD; Pring — KST, ROC summaries.
- Bollinger — Bollinger Bands; Keltner (1960) channels.
- Kaufman (1995). *Smarter Trading* — KAMA, efficiency ratio.
- Lane — stochastics; Williams — %R; Lambert (1980) — CCI.
- Hosoda — Ichimoku Kinko Hyo; Donchian channels.
- Granville — OBV; Chaikin — A/D line, CMF, Chaikin oscillator.
- Elder — Force Index, Elder Ray (bull/bear power); Arms — TRIN.
- Ehlers — Fisher transform; Bressert — (not included).
- Etienne Botes, Douglas Siepman — Vortex indicator.
- Colby, Meyers — Aroon (Tushar Chande).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy.signal import lfilter

Array = NDArray[np.float64]


def _as_vec(x: Array, name: str, window: int = 1, min_n: int | None = None) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    n = v.size
    need = max(window, min_n or window)
    if n < need:
        raise ValueError(f"{name} needs >= {need} observations")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be finite")
    return v


def _check_window(window: int, n: int, name: str = "window") -> int:
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError(f"{name} must be a positive integer")
    if window > n:
        raise ValueError(f"{name}={window} exceeds series length {n}")
    return window


def _ohlcv(high: Array, low: Array, close: Array, window: int = 1) -> tuple[Array, Array, Array]:
    h = _as_vec(high, "high")
    lo = _as_vec(low, "low")
    c = _as_vec(close, "close")
    if h.size != lo.size or h.size != c.size:
        raise ValueError("high/low/close must have equal length")
    if np.any(h < lo) or np.any(h < c) or np.any(lo > c):
        raise ValueError("invalid OHLC: require low <= close <= high")
    _check_window(window, h.size)
    return h, lo, c


def _vol(v: Array, n: int) -> Array:
    vv = np.asarray(v, dtype=float).reshape(-1)
    if vv.size != n or not np.all(np.isfinite(vv)) or np.any(vv < 0.0):
        raise ValueError("volume must be finite, non-negative, same length")
    return vv


def _nan_pad(n: int) -> Array:
    return np.full(n, np.nan)


def _lfilter_recur(seed: float, alpha: float, tail: Array) -> Array:
    """``y = alpha * x + (1 - alpha) * y_prev`` with ``y`` seeded before ``tail``.

    ``scipy.signal.lfilter`` uses that recurrence. For EMA (``alpha = 2/(n+1)``)
    it matches the scalar loop bit for bit. Wilder's ``(prev*(n-1) + x) / n``
    is the same formula written with ``alpha = 1/n``; the two associations
    differ by a few ulps on long series.
    """
    if tail.size == 0:
        return tail
    decay = 1.0 - alpha
    filtered, _ = lfilter([alpha], [1.0, -decay], tail, zi=np.array([decay * seed]))
    return np.asarray(filtered, dtype=float)


def _ema(x: Array, n: int) -> Array:
    """Wilder-exponential moving average, NaN warmup (alpha = 2/(n+1))."""
    v = np.asarray(x, dtype=float)
    out = _nan_pad(v.size)
    if n < 1 or v.size < n:
        return out
    alpha = 2.0 / (n + 1.0)
    seed = float(v[:n].mean())
    out[n - 1] = seed
    out[n:] = _lfilter_recur(seed, alpha, v[n:])
    return out


def _ema_causal(x: Array, n: int) -> Array:
    """EMA on a series that may start with NaNs.

    The seed is the mean of the first ``n`` finite bars only. A later gap
    stops the recursion. No full-sample fill.
    """
    v = np.asarray(x, dtype=float)
    out = _nan_pad(v.size)
    if n < 1 or v.size < n:
        return out
    finite = np.flatnonzero(np.isfinite(v))
    if finite.size < n:
        return out
    start = int(finite[0])
    seed_slice = v[start : start + n]
    if seed_slice.size < n or not np.isfinite(seed_slice).all():
        return out
    alpha = 2.0 / (n + 1.0)
    seed = float(seed_slice.mean())
    out[start + n - 1] = seed
    rest = v[start + n :]
    if rest.size == 0:
        return out
    bad = np.flatnonzero(~np.isfinite(rest))
    end = int(bad[0]) if bad.size else int(rest.size)
    if end:
        out[start + n : start + n + end] = _lfilter_recur(seed, alpha, rest[:end])
    return out


def _sma_causal(x: Array, window: int) -> Array:
    """SMA only when every bar in the window is finite. Leading NaNs stay NaN."""
    v = np.asarray(x, dtype=float)
    out = _nan_pad(v.size)
    if window < 1 or v.size < window:
        return out
    view = sliding_window_view(v, window)
    ok = np.isfinite(view).all(axis=1)
    # ``ndarray.sum`` on the window matches ``slice.sum``; divide by ``window``
    # matches ``slice.mean`` for these short windows (exact on the unit grid).
    out[window - 1 :] = np.where(ok, view.sum(axis=1) / window, np.nan)
    return out


def _wilder_smooth(x: Array, n: int) -> Array:
    """Wilder smoothing (alpha = 1/n), NaN warmup."""
    v = np.asarray(x, dtype=float)
    out = _nan_pad(v.size)
    if n < 1 or v.size < n:
        return out
    seed = float(v[:n].mean())
    out[n - 1] = seed
    out[n:] = _lfilter_recur(seed, 1.0 / n, v[n:])
    return out


def _rolling_sum(x: Array, window: int) -> Array:
    """Trailing sum. ``out[i] = sum(x[i-window+1:i+1])`` for ``i >= window-1``."""
    out = _nan_pad(x.size)
    if window < 1 or x.size < window:
        return out
    out[window - 1 :] = sliding_window_view(x, window).sum(axis=1)
    return out


def _rolling_extreme(x: Array, window: int, *, high: bool) -> Array:
    out = _nan_pad(x.size)
    if window < 1 or x.size < window:
        return out
    view = sliding_window_view(x, window)
    out[window - 1 :] = view.max(axis=1) if high else view.min(axis=1)
    return out


# ---------------------------------------------------------------- averages


def sma(close: Array, window: int) -> Array:
    c = _as_vec(close, "close", window)
    _check_window(window, c.size)
    out = _nan_pad(c.size)
    cs = np.cumsum(np.insert(c, 0, 0.0))
    out[window - 1 :] = (cs[window:] - cs[:-window]) / window
    return out


def ema(close: Array, window: int) -> Array:
    c = _as_vec(close, "close", window)
    return _ema(c, _check_window(window, c.size))


def wma(close: Array, window: int) -> Array:
    c = _as_vec(close, "close", window)
    w = _check_window(window, c.size)
    weights = np.arange(1, w + 1, dtype=float)
    out = _nan_pad(c.size)
    # correlate(c, weights)[k] = dot(c[k:k+w], weights), same as the scalar dot.
    out[w - 1 :] = np.correlate(c, weights, mode="valid") / weights.sum()
    return out


def dema(close: Array, window: int) -> Array:
    """Mulloy DEMA: ``2*EMA - EMA(EMA)``. Nested EMA uses only past bars."""
    c = _as_vec(close, "close", window)
    e1 = _ema(c, window)
    e2 = _ema_causal(e1, window)
    return 2.0 * e1 - e2


def tema(close: Array, window: int) -> Array:
    c = _as_vec(close, "close", window)
    e1 = _ema(c, window)
    e2 = _ema_causal(e1, window)
    e3 = _ema_causal(e2, window)
    return 3.0 * e1 - 3.0 * e2 + e3


def hma(close: Array, window: int) -> Array:
    """Hull MA: ``WMA(2*WMA(n/2) - WMA(n), sqrt(n))``."""
    c = _as_vec(close, "close", window)
    w = _check_window(window, c.size)
    w2 = max(1, w // 2)
    wsq = max(1, int(round(math.sqrt(w))))
    wma_h = wma(c, w2)
    wma_f = wma(c, w)
    raw = 2.0 * wma_h - wma_f
    out = _nan_pad(c.size)
    weights = np.arange(1, wsq + 1, dtype=float)
    if c.size >= wsq:
        view = sliding_window_view(raw, wsq)
        ok = np.isfinite(view).all(axis=1)
        scored = np.where(ok, view @ weights / weights.sum(), np.nan)
        out[wsq - 1 :] = scored
    out[: w + wsq - 2] = np.nan
    return out


def kama(close: Array, window: int = 10, fast: int = 2, slow: int = 30) -> Array:
    """Kaufman (1995) adaptive MA with efficiency-ratio scaling."""
    c = _as_vec(close, "close", window)
    w = _check_window(window, c.size)
    out = _nan_pad(c.size)
    er = _nan_pad(c.size)
    if c.size > w:
        change = np.abs(c[w:] - c[:-w])
        # sum |Δ| over the w steps ending at i equals the window of abs(diff).
        vol = sliding_window_view(np.abs(np.diff(c)), w).sum(axis=1)
        er[w:] = np.where(vol > 0.0, change / vol, 0.0)
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # ``c[w]`` is out of range when ``n == w``, matching the previous loop.
    out[w] = c[w]
    for i in range(w + 1, c.size):
        out[i] = out[i - 1] + sc[i] * (c[i] - out[i - 1])
    return out


def zlema(close: Array, window: int) -> Array:
    """Zero-lag EMA: EMA of ``close + (close - close[lag])``."""
    c = _as_vec(close, "close", window)
    lag = (window - 1) // 2
    adj = np.concatenate([c[:lag], c[lag:] + (c[lag:] - c[:-lag])])
    return _ema(adj, window)


# ------------------------------------------------------------------ trend


def macd(close: Array, fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, Array]:
    """Appel MACD line, signal, histogram."""
    c = _as_vec(close, "close", slow)
    ef = _ema(c, fast)
    es = _ema(c, slow)
    line = ef - es
    sig = _ema_causal(line, signal)
    return {"macd": line, "signal": sig, "histogram": line - sig}


def ppo(close: Array, fast: int = 12, slow: int = 26) -> Array:
    """Percentage price oscillator: MACD / EMA_slow."""
    c = _as_vec(close, "close", slow)
    ef = _ema(c, fast)
    es = _ema(c, slow)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ef - es) / es


def true_range(high: Array, low: Array, close: Array) -> Array:
    h, lo, c = _ohlcv(high, low, close)
    pc = np.concatenate([[c[0]], c[:-1]])
    return np.asarray(
        np.maximum(h - lo, np.maximum(np.abs(h - pc), np.abs(lo - pc))),
        dtype=float,
    )


def atr(high: Array, low: Array, close: Array, window: int = 14) -> Array:
    """Wilder ATR."""
    h, lo, c = _ohlcv(high, low, close, window)
    tr = true_range(h, lo, c)
    return _wilder_smooth(tr, window)


def natr(high: Array, low: Array, close: Array, window: int = 14) -> Array:
    """Normalized ATR = ATR / close."""
    _, _, c = _ohlcv(high, low, close, window)
    a = atr(high, low, close, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        return a / c


def dmi(high: Array, low: Array, close: Array, window: int = 14) -> dict[str, Array]:
    """Wilder directional movement: +DI, -DI, DX, ADX."""
    h, lo, c = _ohlcv(high, low, close, window)
    n = h.size
    up = np.diff(h, prepend=h[0])
    dn = -np.diff(lo, prepend=lo[0])
    plus_dm = np.where((up > dn) & (up > 0.0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0.0), dn, 0.0)
    tr_s = _wilder_smooth(true_range(h, lo, c), window)
    plus_di = 100.0 * _wilder_smooth(plus_dm, window) / np.where(tr_s > 0, tr_s, np.nan)
    minus_di = 100.0 * _wilder_smooth(minus_dm, window) / np.where(tr_s > 0, tr_s, np.nan)
    dx = (
        100.0
        * np.abs(plus_di - minus_di)
        / np.where(plus_di + minus_di > 0, plus_di + minus_di, np.nan)
    )
    adx = _nan_pad(n)
    first = 2 * window - 1
    if first < n:
        segment = dx[window - 1 : first + 1]
        # Finite OHLC makes this slice finite, so nanmean == mean. A gap falls
        # back to the scalar Wilder step so a NaN still stops the seed.
        if np.isfinite(segment).all():
            seed = float(segment.mean())
            adx[first] = seed
            adx[first + 1 :] = _lfilter_recur(seed, 1.0 / window, dx[first + 1 :])
        else:
            adx[first] = float(np.nanmean(segment))
            for i in range(first + 1, n):
                adx[i] = (adx[i - 1] * (window - 1) + dx[i]) / window
    return {"plus_di": plus_di, "minus_di": minus_di, "dx": dx, "adx": adx}


def aroon(high: Array, low: Array, window: int = 25) -> dict[str, Array]:
    """Chande Aroon up/down/oscillator."""
    h = _as_vec(high, "high", window)
    lo = _as_vec(low, "low", window)
    if h.size != lo.size:
        raise ValueError("high and low must match length")
    w = _check_window(window, h.size)
    up = _nan_pad(h.size)
    dn = _nan_pad(h.size)
    if w == 1:
        # Scalar ``/ (w - 1)`` raises; keep that failure instead of emitting NaN.
        raise ZeroDivisionError("division by zero")
    # Position of the most extreme point: near end -> high Aroon.
    up[w - 1 :] = 100.0 * np.argmax(sliding_window_view(h, w), axis=1) / (w - 1)
    dn[w - 1 :] = 100.0 * np.argmin(sliding_window_view(lo, w), axis=1) / (w - 1)
    return {"up": up, "down": dn, "oscillator": up - dn}


def vortex(high: Array, low: Array, close: Array, window: int = 14) -> dict[str, Array]:
    """Botes–Siepman Vortex indicator."""
    h, lo, c = _ohlcv(high, low, close, window)
    prev_lo = np.concatenate([[lo[0]], lo[:-1]])
    prev_hi = np.concatenate([[h[0]], h[:-1]])
    vm_plus = np.abs(h - prev_lo)
    vm_minus = np.abs(lo - prev_hi)
    tr = true_range(h, lo, c)
    w = _check_window(window, h.size)
    sp = _rolling_sum(vm_plus, w)
    sm = _rolling_sum(vm_minus, w)
    st = _rolling_sum(tr, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"plus": sp / st, "minus": sm / st}


def trix(close: Array, window: int = 15) -> Array:
    """1-day ROC of a triple-smoothed EMA."""
    c = _as_vec(close, "close", window)
    e1 = _ema(c, window)
    e2 = _ema_causal(e1, window)
    e3 = _ema_causal(e2, window)
    out = _nan_pad(c.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[1:] = np.diff(e3) / np.abs(e3[:-1])
    return out


def cci(high: Array, low: Array, close: Array, window: int = 20) -> Array:
    """Lambert (1980) CCI: ``(TP - SMA(TP)) / (0.015 * MD)``."""
    h, lo, c = _ohlcv(high, low, close, window)
    tp = (h + lo + c) / 3.0
    ma = sma(tp, window)
    out = _nan_pad(h.size)
    view = sliding_window_view(tp, window)
    centered = view - view.mean(axis=1, keepdims=True)
    md = np.mean(np.abs(centered), axis=1)
    out[window - 1 :] = np.where(
        md > 0.0, (tp[window - 1 :] - ma[window - 1 :]) / (0.015 * md), 0.0
    )
    return out


def ichimoku(
    high: Array, low: Array, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52
) -> dict[str, Array]:
    """Hosoda Ichimoku lines (no forward shift — causal placement)."""
    h = _as_vec(high, "high", senkou_b)
    lo = _as_vec(low, "low", senkou_b)
    if h.size != lo.size:
        raise ValueError("high and low must match length")

    def _hh_ll(w: int) -> Array:
        return 0.5 * (_rolling_extreme(h, w, high=True) + _rolling_extreme(lo, w, high=False))

    t = _hh_ll(tenkan)
    k = _hh_ll(kijun)
    sa = (t + k) / 2.0
    sb = _hh_ll(senkou_b)
    return {"tenkan": t, "kijun": k, "senkou_a": sa, "senkou_b": sb}


def donchian(high: Array, low: Array, window: int = 20) -> dict[str, Array]:
    h = _as_vec(high, "high", window)
    lo = _as_vec(low, "low", window)
    _check_window(window, h.size)
    up = _rolling_extreme(h, window, high=True)
    dn = _rolling_extreme(lo, window, high=False)
    return {"upper": up, "lower": dn, "mid": (up + dn) / 2.0}


def keltner(
    high: Array, low: Array, close: Array, window: int = 20, mult: float = 2.0
) -> dict[str, Array]:
    h, lo, c = _ohlcv(high, low, close, window)
    mid = _ema(c, window)
    band = atr(h, lo, c, window)
    return {"mid": mid, "upper": mid + mult * band, "lower": mid - mult * band}


def bollinger(close: Array, window: int = 20, num_sd: float = 2.0) -> dict[str, Array]:
    c = _as_vec(close, "close", window)
    w = _check_window(window, c.size)
    if not np.isfinite(num_sd) or num_sd <= 0.0:
        raise ValueError("num_sd must be positive")
    mid = sma(c, w)
    sd = _nan_pad(c.size)
    sd[w - 1 :] = sliding_window_view(c, w).std(axis=1, ddof=0)
    upper = mid + num_sd * sd
    lower = mid - num_sd * sd
    with np.errstate(invalid="ignore", divide="ignore"):
        pctb = (c - lower) / (upper - lower)
        bandwidth = (upper - lower) / mid
    return {
        "mid": mid,
        "upper": upper,
        "lower": lower,
        "pct_b": pctb,
        "bandwidth": bandwidth,
    }


# --------------------------------------------------------------- momentum


def rsi(close: Array, window: int = 14) -> Array:
    """Wilder RSI."""
    c = _as_vec(close, "close", window)
    w = _check_window(window, c.size)
    d = np.diff(c, prepend=c[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    au = _wilder_smooth(up, w)
    ad = _wilder_smooth(dn, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = au / ad
        out = 100.0 - 100.0 / (1.0 + rs)
    out[ad == 0.0] = np.where(au[ad == 0.0] > 0, 100.0, 50.0)
    return out


def stochastic(
    high: Array, low: Array, close: Array, k_window: int = 14, d_window: int = 3
) -> dict[str, Array]:
    """Lane stochastic %K and %D."""
    h, lo, c = _ohlcv(high, low, close, k_window)
    w = _check_window(k_window, h.size)
    k = _nan_pad(h.size)
    hh = _rolling_extreme(h, w, high=True)
    ll = _rolling_extreme(lo, w, high=False)
    span = hh - ll
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_k = 100.0 * (c - ll) / span
    k[w - 1 :] = np.where(span[w - 1 :] > 0.0, raw_k[w - 1 :], 50.0)
    d = _sma_causal(k, _check_window(d_window, k.size, "%D"))
    return {"k": k, "d": d}


def williams_r(high: Array, low: Array, close: Array, window: int = 14) -> Array:
    h, lo, c = _ohlcv(high, low, close, window)
    w = _check_window(window, h.size)
    out = _nan_pad(h.size)
    hh = _rolling_extreme(h, w, high=True)
    ll = _rolling_extreme(lo, w, high=False)
    span = hh - ll
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = -100.0 * (hh - c) / span
    out[w - 1 :] = np.where(span[w - 1 :] > 0.0, raw[w - 1 :], -50.0)
    return out


def roc(close: Array, window: int = 10) -> Array:
    c = _as_vec(close, "close", window)
    w = _check_window(window, c.size)
    out = _nan_pad(c.size)
    out[w:] = c[w:] / c[:-w] - 1.0
    return out


def momentum(close: Array, window: int = 10) -> Array:
    c = _as_vec(close, "close", window)
    w = _check_window(window, c.size)
    out = _nan_pad(c.size)
    out[w:] = c[w:] - c[:-w]
    return out


def cmo(close: Array, window: int = 14) -> Array:
    """Chande momentum oscillator."""
    c = _as_vec(close, "close", window)
    w = _check_window(window, c.size)
    d = np.diff(c, prepend=c[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    out = _nan_pad(c.size)
    # The first window starts at index 1: the prepended zero change is excluded.
    if c.size > w:
        su = sliding_window_view(up, w).sum(axis=1)[1:]
        sd = sliding_window_view(dn, w).sum(axis=1)[1:]
        denom = su + sd
        out[w:] = np.where(denom > 0.0, 100.0 * (su - sd) / denom, 0.0)
    return out


def tsi(close: Array, slow: int = 25, fast: int = 13) -> Array:
    """True strength index: double-smoothed momentum / |momentum|."""
    c = _as_vec(close, "close", slow)
    m = np.diff(c, prepend=c[0])
    e1 = _ema(m, slow)
    e2 = _ema_causal(e1, fast)
    a1 = _ema(np.abs(m), slow)
    a2 = _ema_causal(a1, fast)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100.0 * e2 / a2


def ultimate_oscillator(
    high: Array,
    low: Array,
    close: Array,
    s1: int = 7,
    s2: int = 14,
    s3: int = 28,
) -> Array:
    """Larry Williams Ultimate Oscillator."""
    h, lo, c = _ohlcv(high, low, close, s3)
    pc = np.concatenate([[c[0]], c[:-1]])
    bp = c - np.minimum(lo, pc)
    tr = np.maximum(h, pc) - np.minimum(lo, pc)
    out = _nan_pad(h.size)

    def _avg(w: int) -> Array:
        s_bp = _rolling_sum(bp, w)
        s_tr = _rolling_sum(tr, w)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = s_bp / s_tr
        return np.where(s_tr > 0.0, ratio, 0.0)

    a7 = _avg(s1)
    a14 = _avg(s2)
    a28 = _avg(s3)
    out[s3 - 1 :] = 100.0 * (4.0 * a7[s3 - 1 :] + 2.0 * a14[s3 - 1 :] + a28[s3 - 1 :]) / 7.0
    return out


def awesome_oscillator(high: Array, low: Array, fast: int = 5, slow: int = 34) -> Array:
    """Bill Williams AO: SMA_fast(median) - SMA_slow(median)."""
    h = _as_vec(high, "high", slow)
    lo = _as_vec(low, "low", slow)
    med = (h + lo) / 2.0
    return sma(med, fast) - sma(med, slow)


def fisher_transform(high: Array, low: Array, window: int = 10) -> dict[str, Array]:
    """Ehlers Fisher transform of the median price."""
    h = _as_vec(high, "high", window)
    lo = _as_vec(low, "low", window)
    w = _check_window(window, h.size)
    med = (h + lo) / 2.0
    out = _nan_pad(h.size)
    hh = _rolling_extreme(med, w, high=True)
    ll = _rolling_extreme(med, w, high=False)
    prev_val = 0.0
    prev_fish = 0.0
    for i in range(w - 1, h.size):
        rng_ = hh[i] - ll[i]
        v = 0.0 if rng_ <= 0.0 else 2.0 * (med[i] - ll[i]) / rng_ - 1.0
        val = min(max(0.33 * v + 0.67 * prev_val, -0.999), 0.999)
        fish = 0.5 * math.log((1.0 + val) / (1.0 - val)) + 0.5 * prev_fish
        prev_val = val
        prev_fish = fish
        out[i] = fish
    sig = np.roll(out, 1)
    sig[:w] = np.nan
    return {"fisher": out, "signal": sig}


def elder_ray(high: Array, low: Array, close: Array, window: int = 13) -> dict[str, Array]:
    """Elder bull/bear power vs EMA."""
    h, lo, c = _ohlcv(high, low, close, window)
    e = _ema(c, window)
    return {"bull_power": h - e, "bear_power": lo - e}


def kst(close: Array) -> dict[str, Array]:
    """Pring Know Sure Thing: weighted sum of 4 smoothed ROCs."""
    c = _as_vec(close, "close", 30)
    r1 = sma(np.nan_to_num(roc(c, 10)), 10)
    r2 = sma(np.nan_to_num(roc(c, 15)), 10)
    r3 = sma(np.nan_to_num(roc(c, 20)), 10)
    r4 = sma(np.nan_to_num(roc(c, 30)), 15)
    line = (
        np.nan_to_num(r1) * 1.0
        + np.nan_to_num(r2) * 2.0
        + np.nan_to_num(r3) * 3.0
        + np.nan_to_num(r4) * 4.0
    )
    line[:44] = np.nan
    sig = _sma_causal(line, 9)
    return {"kst": line, "signal": sig}


# ----------------------------------------------------------------- volume


def obv(close: Array, volume: Array) -> Array:
    """Granville on-balance volume."""
    c = _as_vec(close, "close")
    v = _vol(volume, c.size)
    d = np.sign(np.diff(c, prepend=c[0]))
    return np.cumsum(d * v)


def ad_line(high: Array, low: Array, close: Array, volume: Array) -> Array:
    """Chaikin accumulation/distribution line."""
    h, lo, c = _ohlcv(high, low, close)
    v = _vol(volume, h.size)
    rng_ = h - lo
    mfm = np.where(rng_ > 0, ((c - lo) - (h - c)) / np.where(rng_ > 0, rng_, 1.0), 0.0)
    return np.cumsum(mfm * v)


def chaikin_money_flow(
    high: Array, low: Array, close: Array, volume: Array, window: int = 20
) -> Array:
    h, lo, c = _ohlcv(high, low, close, window)
    v = _vol(volume, h.size)
    rng_ = h - lo
    mfm = np.where(rng_ > 0, ((c - lo) - (h - c)) / np.where(rng_ > 0, rng_, 1.0), 0.0)
    mfv = mfm * v
    out = _nan_pad(h.size)
    w = _check_window(window, h.size)
    sv = _rolling_sum(v, w)
    sm = _rolling_sum(mfv, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = sm / sv
    out[w - 1 :] = np.where(sv[w - 1 :] > 0.0, ratio[w - 1 :], 0.0)
    return out


def chaikin_oscillator(
    high: Array,
    low: Array,
    close: Array,
    volume: Array,
    fast: int = 3,
    slow: int = 10,
) -> Array:
    adl = ad_line(high, low, close, volume)
    return _ema(adl, fast) - _ema(adl, slow)


def mfi(high: Array, low: Array, close: Array, volume: Array, window: int = 14) -> Array:
    """Money flow index (volume-weighted RSI)."""
    h, lo, c = _ohlcv(high, low, close, window)
    v = _vol(volume, h.size)
    tp = (h + lo + c) / 3.0
    rmf = tp * v
    d = np.diff(tp, prepend=tp[0])
    pos = np.where(d > 0, rmf, 0.0)
    neg = np.where(d < 0, rmf, 0.0)
    out = _nan_pad(h.size)
    if h.size > window:
        sp = sliding_window_view(pos, window).sum(axis=1)[1:]
        sn = sliding_window_view(neg, window).sum(axis=1)[1:]
        with np.errstate(invalid="ignore", divide="ignore"):
            raw = 100.0 - 100.0 / (1.0 + sp / sn)
        out[window:] = np.where(sn > 0.0, raw, 100.0)
    return out


def force_index(close: Array, volume: Array, window: int = 13) -> Array:
    """Elder Force Index: EMA of ``diff(close) * volume``."""
    c = _as_vec(close, "close", window)
    v = _vol(volume, c.size)
    raw = np.diff(c, prepend=c[0]) * v
    return _ema(raw, window)


def eom(high: Array, low: Array, volume: Array, window: int = 14) -> Array:
    """Ease of movement."""
    h = _as_vec(high, "high", window)
    lo = _as_vec(low, "low", window)
    v = _vol(volume, h.size)
    mid_move = np.diff((h + lo) / 2.0, prepend=(h[0] + lo[0]) / 2.0)
    rng_ = np.where(h - lo > 0, h - lo, np.nan)
    box_ratio = (v / 1e6) / rng_
    raw = mid_move / np.where(box_ratio > 0, box_ratio, np.nan)
    return sma(np.nan_to_num(raw), window)


def vwap(high: Array, low: Array, close: Array, volume: Array) -> Array:
    """Cumulative VWAP over the full series (no session reset)."""
    h, lo, c = _ohlcv(high, low, close)
    v = _vol(volume, h.size)
    tp = (h + lo + c) / 3.0
    cv = np.cumsum(v)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.cumsum(tp * v) / np.where(cv > 0, cv, np.nan)


def arms_index(advances: Array, declines: Array, adv_volume: Array, dec_volume: Array) -> Array:
    """Arms TRIN: ``(adv/dec) / (adv_vol/dec_vol)``."""
    a = _as_vec(advances, "advances")
    d = _as_vec(declines, "declines")
    av = _as_vec(adv_volume, "adv_volume")
    dv = _as_vec(dec_volume, "dec_volume")
    if not (a.size == d.size == av.size == dv.size):
        raise ValueError("all breadth inputs must match length")
    if np.any(a <= 0) or np.any(d <= 0) or np.any(av <= 0) or np.any(dv <= 0):
        raise ValueError("breadth inputs must be positive")
    return (a / d) / (av / dv)


def chaikin_volatility(high: Array, low: Array, window: int = 10, roc_window: int = 10) -> Array:
    """Chaikin volatility: ROC of EMA(high - low)."""
    h = _as_vec(high, "high", window + roc_window)
    lo = _as_vec(low, "low", window + roc_window)
    e = _ema(h - lo, window)
    out = _nan_pad(h.size)
    start = window + roc_window - 1
    if start < h.size:
        prev = e[start - roc_window : h.size - roc_window]
        cur = e[start:]
        with np.errstate(invalid="ignore", divide="ignore"):
            roc = (cur - prev) / prev * 100.0
        out[start:] = np.where(prev != 0.0, roc, 0.0)
    return out


def stoch_rsi(close: Array, rsi_window: int = 14, stoch_window: int = 14) -> Array:
    """Stochastic RSI."""
    c = _as_vec(close, "close", rsi_window + stoch_window)
    r = rsi(c, rsi_window)
    out = _nan_pad(c.size)
    start = rsi_window + stoch_window - 1
    if start < c.size:
        view = sliding_window_view(r, stoch_window)
        # view[j] ends at index j + stoch_window - 1. The first used end is ``start``.
        offset = start - (stoch_window - 1)
        hi = np.nanmax(view[offset:], axis=1)
        lo = np.nanmin(view[offset:], axis=1)
        span = hi - lo
        cur = r[start:]
        with np.errstate(invalid="ignore", divide="ignore"):
            raw = (cur - lo) / span
        out[start:] = np.where(span > 0.0, raw, 0.5)
    return out

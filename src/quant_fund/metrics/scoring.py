"""Proper scoring rules and forecast metrics. See docs/MATH_SPEC.md."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import betaln, erf
from scipy.stats import t as student_t

from quant_fund.metrics.probability import pit_ks

Array = NDArray[np.float64]

_ZERO_STD = 1e-15


def _as_1d(name: str, x: Array) -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.ndim > 1:
        raise ValueError(f"{name} must be 1d")
    return arr.reshape(-1)


def _require_same_length(*named: tuple[str, Array]) -> None:
    lengths = {name: arr.shape[0] for name, arr in named}
    if len(set(lengths.values())) > 1:
        parts = ", ".join(f"{k}={v}" for k, v in lengths.items())
        raise ValueError(f"length mismatch: {parts}")


def pinball_loss(y: Array, q: Array, tau: float) -> Array:
    """Elementwise pinball loss for quantile tau."""
    if not 0 < tau < 1:
        raise ValueError("tau must be in (0, 1)")
    y = _as_1d("y", y)
    q = _as_1d("q", q)
    _require_same_length(("y", y), ("q", q))
    diff = y - q
    return np.maximum(tau * diff, (tau - 1.0) * diff)


def mean_pinball(y: Array, q: Array, tau: float) -> float:
    """Mean pinball. Empty after length check → honest NaN (research only)."""
    losses = pinball_loss(y, q, tau)
    if losses.size == 0:
        return float("nan")
    return float(np.mean(losses))


def coverage(y: Array, lower: Array, upper: Array) -> float:
    y = _as_1d("y", y)
    lower = _as_1d("lower", lower)
    upper = _as_1d("upper", upper)
    _require_same_length(("y", y), ("lower", lower), ("upper", upper))
    if y.size == 0:
        return float("nan")
    return float(np.mean((y >= lower) & (y <= upper)))


def interval_width(lower: Array, upper: Array) -> float:
    lower = _as_1d("lower", lower)
    upper = _as_1d("upper", upper)
    _require_same_length(("lower", lower), ("upper", upper))
    if lower.size == 0:
        return float("nan")
    valid = np.isfinite(lower) & np.isfinite(upper) & (upper >= lower)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(upper[valid] - lower[valid]))


def quantile_crossing_rate(quantiles: Array, taus: Array) -> float:
    """Fraction of rows with any Q_tau1 > Q_tau2 for tau1 < tau2.

    quantiles: (n, k) aligned with increasing taus.
    """
    q = np.asarray(quantiles, dtype=float)
    t = np.asarray(taus, dtype=float)
    if q.ndim != 2 or q.shape[1] != t.size:
        raise ValueError("quantiles must be (n, k) matching taus")
    if not np.isfinite(t).all() or np.any(np.diff(t) < 0.0) or np.any((t <= 0.0) | (t >= 1.0)):
        raise ValueError("taus must be finite, strictly inside (0, 1), and nondecreasing")
    if q.shape[0] == 0:
        return float("nan")
    valid_rows = np.isfinite(q).all(axis=1)
    if not np.any(valid_rows):
        return float("nan")
    diffs = np.diff(q[valid_rows], axis=1)
    crossed = np.any(diffs < -1e-15, axis=1)
    return float(np.mean(crossed))


def rearrange_quantiles(quantiles: Array) -> Array:
    """Sort along the quantile axis (rearrangement). Does not hide raw crossing rate."""
    return np.sort(np.asarray(quantiles, dtype=float), axis=1)


def crps_from_quantiles(y: Array, quantiles: Array, taus: Array) -> float:
    """Riemann-sum CRPS approximation from pinball losses (Gneiting-Raftery)."""
    q = np.asarray(quantiles, dtype=float)
    t = np.asarray(taus, dtype=float)
    y = _as_1d("y", y)
    if q.ndim != 2:
        raise ValueError("quantiles must be 2d")
    if q.shape[1] != t.size:
        raise ValueError("quantiles must be (n, k) matching taus")
    if not np.isfinite(t).all() or np.any(np.diff(t) < 0.0) or np.any((t <= 0.0) | (t >= 1.0)):
        raise ValueError("taus must be finite, strictly inside (0, 1), and nondecreasing")
    if y.shape[0] != q.shape[0]:
        raise ValueError(f"length mismatch: y={y.shape[0]}, quantiles rows={q.shape[0]}")
    if y.size == 0:
        return float("nan")
    dt = np.diff(np.concatenate([[0.0], t]))
    total = np.zeros(y.shape[0], dtype=float)
    for k, tau in enumerate(t):
        # CRPS = 2 * integral_0^1 pinball_tau(y, q_tau) dtau (Gneiting-Raftery).
        total += 2.0 * pinball_loss(y, q[:, k], float(tau)) * dt[k]
    return float(np.mean(total))


def crps_gaussian(y: Array, mu: Array, sigma: Array) -> Array:
    r"""Elementwise closed-form CRPS for N(μ, σ²) forecasts (research-only).

    Gneiting & Raftery / standard Gaussian CRPS:

    .. math::

        \mathrm{CRPS}\bigl(N(\mu,\sigma^2), y\bigr)
        = \sigma\bigl[z\,(2\Phi(z)-1) + 2\varphi(z) - 1/\sqrt{\pi}\bigr]

    with :math:`z=(y-\mu)/\sigma`, :math:`\varphi` / :math:`\Phi` the standard
    normal pdf / cdf. Not a live capital claim.

    Empty inputs → empty array. Length mismatch → ValueError. Non-positive or
    non-finite ``sigma`` (or non-finite ``y``/``mu``) → NaN at those indices.
    """
    y_arr = _as_1d("y", y)
    mu_arr = _as_1d("mu", mu)
    sig_arr = _as_1d("sigma", sigma)
    _require_same_length(("y", y_arr), ("mu", mu_arr), ("sigma", sig_arr))
    if y_arr.size == 0:
        return np.asarray([], dtype=float)
    out = np.full(y_arr.shape, np.nan, dtype=float)
    ok = np.isfinite(y_arr) & np.isfinite(mu_arr) & np.isfinite(sig_arr) & (sig_arr > 0.0)
    if not np.any(ok):
        return out
    z = (y_arr[ok] - mu_arr[ok]) / sig_arr[ok]
    # φ(z), Φ(z) via numpy (no scipy dep in this module)
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    Phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    out[ok] = sig_arr[ok] * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
    return out


def mean_crps_gaussian(y: Array, mu: Array, sigma: Array) -> float:
    """Mean closed-form Gaussian CRPS. Empty / all-NaN → honest NaN.

    Research-diagnostic only — never live Sharpe / promotion evidence.
    """
    scores = crps_gaussian(y, mu, sigma)
    if scores.size == 0:
        return float("nan")
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def crps_student_t(y: Array, mu: Array, sigma: Array, nu: float | Array) -> Array:
    r"""Elementwise closed-form CRPS for location-scale Student-t (research-only).

    Jordan, Krüger & Lerch / scoringRules (Gneiting–Raftery CRPS). With
    :math:`z=(y-\mu)/\sigma` and standard-t pdf/cdf :math:`f_\nu`/:math:`F_\nu`
    (:math:`\nu>2`):

    .. math::

        \mathrm{CRPS}
        = \sigma\Biggl[
          z\,(2F_\nu(z)-1)
          + \frac{2}{\nu-1}\Bigl(
            f_\nu(z)\,(\nu+z^2)
            - \sqrt{\nu}\,
            \frac{B(\tfrac12,\nu-\tfrac12)}{B(\tfrac12,\tfrac{\nu}{2})^2}
          \Bigr)
        \Biggr]

    Lab contract: ``nu<=2`` (or non-finite) → honest NaN (stricter than the
    finite-mean ``nu>1`` domain — fail-closed with finite variance). Non-positive
    / non-finite ``sigma`` → NaN. Empty → empty array. Length mismatch →
    ValueError. Not a live capital claim.
    """
    y_arr = _as_1d("y", y)
    mu_arr = _as_1d("mu", mu)
    sig_arr = _as_1d("sigma", sigma)
    _require_same_length(("y", y_arr), ("mu", mu_arr), ("sigma", sig_arr))
    nu_arr = np.asarray(nu, dtype=float).reshape(-1)
    if nu_arr.size == 1:
        nu_arr = np.full(y_arr.shape, float(nu_arr[0]), dtype=float)
    else:
        _require_same_length(("y", y_arr), ("nu", nu_arr))
    if y_arr.size == 0:
        return np.asarray([], dtype=float)
    out = np.full(y_arr.shape, np.nan, dtype=float)
    ok = (
        np.isfinite(y_arr)
        & np.isfinite(mu_arr)
        & np.isfinite(sig_arr)
        & (sig_arr > 0.0)
        & np.isfinite(nu_arr)
        & (nu_arr > 2.0)
    )
    if not np.any(ok):
        return out
    z = (y_arr[ok] - mu_arr[ok]) / sig_arr[ok]
    nu_ok = nu_arr[ok]
    # scoringRules bfrac = B(1/2, ν−1/2) / B(1/2, ν/2)^2  (stable via betaln)
    bfrac = np.exp(betaln(0.5, nu_ok - 0.5) - 2.0 * betaln(0.5, 0.5 * nu_ok))
    F = student_t.cdf(z, nu_ok)
    f = student_t.pdf(z, nu_ok)
    unit = z * (2.0 * F - 1.0) + (2.0 / (nu_ok - 1.0)) * (
        f * (nu_ok + z * z) - np.sqrt(nu_ok) * bfrac
    )
    out[ok] = sig_arr[ok] * unit
    return out


def mean_crps_student_t(y: Array, mu: Array, sigma: Array, nu: float | Array) -> float:
    """Mean closed-form Student-t CRPS. Empty / all-NaN → honest NaN.

    Research-diagnostic only — never live Sharpe / promotion evidence.
    """
    scores = crps_student_t(y, mu, sigma, nu)
    if scores.size == 0:
        return float("nan")
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


# 19-quantile grid used by the one-step GARCH/TSFM evaluation lanes.
GARCH_ONE_STEP_CRPS_TAUS: tuple[float, ...] = tuple(
    float(x) for x in np.linspace(0.05, 0.95, 19)
)


def log_score_gaussian(y: Array, mu: Array, sigma: Array) -> Array:
    r"""Elementwise logarithmic score of :math:`N(\mu,\sigma^2)` (research-only).

    Gneiting & Raftery logarithmic score :math:`\log f(y)` (higher is better):

    .. math::

        \log\varphi_{\mu,\sigma}(y)
        = -\tfrac12\log(2\pi) - \log\sigma - \tfrac12 z^2,
        \qquad z=(y-\mu)/\sigma.

    Empty → empty array. Length mismatch → ValueError. Non-positive or
    non-finite ``sigma`` (or non-finite ``y``/``mu``) → NaN at those indices.
    This is a proper density score, not a live-performance claim.
    """
    y_arr = _as_1d("y", y)
    mu_arr = _as_1d("mu", mu)
    sig_arr = _as_1d("sigma", sigma)
    _require_same_length(("y", y_arr), ("mu", mu_arr), ("sigma", sig_arr))
    if y_arr.size == 0:
        return np.asarray([], dtype=float)
    out = np.full(y_arr.shape, np.nan, dtype=float)
    ok = np.isfinite(y_arr) & np.isfinite(mu_arr) & np.isfinite(sig_arr) & (sig_arr > 0.0)
    if not np.any(ok):
        return out
    z = (y_arr[ok] - mu_arr[ok]) / sig_arr[ok]
    out[ok] = -0.5 * np.log(2.0 * np.pi) - np.log(sig_arr[ok]) - 0.5 * z * z
    return out


def mean_log_score_gaussian(y: Array, mu: Array, sigma: Array) -> float:
    """Mean Gaussian logarithmic score. Empty / all-NaN → honest NaN.

    Research-diagnostic only — never live Sharpe / promotion evidence.
    """
    scores = log_score_gaussian(y, mu, sigma)
    if scores.size == 0:
        return float("nan")
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def crps_gaussian_mixture(y: Array, weights: Array, mu: Array, sigma: Array) -> Array:
    r"""Elementwise closed-form CRPS for a Gaussian-mixture forecast (research-only).

    Grimit, Gneiting & Berrocal (2006, eq. for normal mixtures; also Jordan,
    Krüger & Lerch 2019). For :math:`F = \sum_k w_k\,N(\mu_k,\sigma_k^2)`:

    .. math::

        \mathrm{CRPS}(F, y)
        = \sum_k w_k\,\mathrm{CRPS}\bigl(N(\mu_k,\sigma_k^2), y\bigr)
          - \tfrac12 \sum_{i,j} w_i w_j\, A(\mu_i-\mu_j,\ \sigma_i^2+\sigma_j^2)

    with :math:`A(m, s^2) = s\bigl[2\varphi(m/s) + (m/s)(2\Phi(m/s)-1)\bigr]`,
    the expected absolute difference of two independent component draws.

    ``weights``/``mu``/``sigma`` are shared component arrays (1d, length K);
    ``y`` is 1d. A degenerate mixture (non-finite params, any sigma<=0, any
    negative weight, or zero weight mass) → NaN for every observation. Weights
    are renormalized only when their sum is within 1e-6 of 1 — anything larger
    is a caller bug and fails closed to NaN. Not a live capital claim.
    """
    y_arr = _as_1d("y", y)
    w = _as_1d("weights", weights)
    mu_arr = _as_1d("mu", mu)
    sig_arr = _as_1d("sigma", sigma)
    _require_same_length(("weights", w), ("mu", mu_arr), ("sigma", sig_arr))
    if y_arr.size == 0:
        return np.asarray([], dtype=float)
    wsum = float(w.sum())
    valid_mix = (
        w.size > 0
        and np.all(np.isfinite(w))
        and np.all(np.isfinite(mu_arr))
        and np.all(np.isfinite(sig_arr))
        and np.all(sig_arr > 0.0)
        and np.all(w >= 0.0)
        and wsum > 0.0
        and abs(wsum - 1.0) < 1e-6
    )
    out = np.full(y_arr.shape, np.nan, dtype=float)
    ok = np.isfinite(y_arr)
    if not valid_mix or not np.any(ok):
        return out
    w = w / wsum
    z = (y_arr[ok, None] - mu_arr[None, :]) / sig_arr[None, :]  # (n, K)
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    Phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    # E|X_k - y| for X_k ~ N(mu_k, sigma_k^2) — no 1/sqrt(pi) term; that
    # constant lives in the pairwise-spread term, not the first moment.
    single = sig_arr[None, :] * (z * (2.0 * Phi - 1.0) + 2.0 * phi)
    m_ij = mu_arr[:, None] - mu_arr[None, :]  # (K, K)
    s_ij = np.sqrt(sig_arr[:, None] ** 2 + sig_arr[None, :] ** 2)
    d_ij = m_ij / s_ij
    a_ij = s_ij * (
        2.0 * np.exp(-0.5 * d_ij * d_ij) / np.sqrt(2.0 * np.pi) + d_ij * erf(d_ij / np.sqrt(2.0))
    )
    spread = float((w[:, None] * w[None, :] * a_ij).sum())
    out[ok] = single @ w - 0.5 * spread
    return out


def gaussian_mixture_quantiles(weights: Array, mu: Array, sigma: Array, taus: Array) -> Array:
    """Quantiles of a Gaussian mixture by bisection on the monotone CDF.

    Returns NaN for a degenerate mixture (same contract as
    ``crps_gaussian_mixture``). Brackets span min(μ)-12σ to max(μ)+12σ over
    component scales; 80 iterations land within ~1e-24 of the bracket width.
    """
    w = _as_1d("weights", weights)
    mu_arr = _as_1d("mu", mu)
    sig_arr = _as_1d("sigma", sigma)
    tau_arr = _as_1d("taus", taus)
    _require_same_length(("weights", w), ("mu", mu_arr), ("sigma", sig_arr))
    wsum = float(w.sum())
    valid_mix = (
        w.size > 0
        and np.all(np.isfinite(w))
        and np.all(np.isfinite(mu_arr))
        and np.all(np.isfinite(sig_arr))
        and np.all(sig_arr > 0.0)
        and np.all(w >= 0.0)
        and wsum > 0.0
        and abs(wsum - 1.0) < 1e-6
    )
    out = np.full(tau_arr.shape, np.nan, dtype=float)
    if not valid_mix or np.any((tau_arr <= 0.0) | (tau_arr >= 1.0)):
        return out
    w = w / wsum
    lo = float((mu_arr - 12.0 * sig_arr).min())
    hi = float((mu_arr + 12.0 * sig_arr).max())
    for k, tau in enumerate(tau_arr):
        a, b = lo, hi
        for _ in range(80):
            mid = 0.5 * (a + b)
            z = (mid - mu_arr) / sig_arr
            cdf = float((w * (0.5 * (1.0 + erf(z / np.sqrt(2.0))))).sum())
            if cdf < tau:
                a = mid
            else:
                b = mid
        out[k] = 0.5 * (a + b)
    return out


def crps_empirical(y: float | Array, sample: Array) -> float:
    r"""Empirical CRPS from an ensemble sample (research-only).

    For observation ``y`` and i.i.d. draws ``X_1..X_n`` from the predictive:

    .. math::

        \widehat{\mathrm{CRPS}}
        = \frac{1}{n}\sum_i |X_i - y|
        - \frac{1}{2n^2}\sum_{i,j}|X_i - X_j|

    ``y`` may be a scalar or length-1 array; ``sample`` must be 1d with n≥1.
    Empty / all-non-finite sample → NaN. Multi-row ``y`` → ValueError (use a
    loop or extend later). Not a live capital claim.
    """
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if y_arr.size != 1:
        raise ValueError("crps_empirical expects a single observation y")
    y0 = float(y_arr[0])
    x = _as_1d("sample", sample)
    x = x[np.isfinite(x)]
    if x.size == 0 or not np.isfinite(y0):
        return float("nan")
    term1 = float(np.mean(np.abs(x - y0)))
    # Pairwise |Xi-Xj| via broadcasting; O(n²) — fine for research ensembles
    pairwise = np.abs(x[:, None] - x[None, :])
    term2 = float(np.mean(pairwise)) / 2.0
    return term1 - term2


def qlike(realized_var: Array, forecast_var: Array, floor: float = 1e-12) -> float:
    """QLIKE on variance: y/yhat - log(y/yhat) - 1.

    Fail-closed on variance contract: a variance must be finite, realized ≥ 0,
    forecast > 0. Negative or non-finite entries are invalid observations and
    are masked out honestly rather than clipped to ``floor`` — clipping a
    negative variance to a tiny positive floor would fabricate a plausible
    finite QLIKE from malformed input. Empty/all-invalid → NaN (research only).
    """
    y = _as_1d("realized_var", realized_var)
    yhat = _as_1d("forecast_var", forecast_var)
    _require_same_length(("realized_var", y), ("forecast_var", yhat))
    if y.size == 0:
        return float("nan")
    valid = np.isfinite(y) & np.isfinite(yhat) & (y >= 0.0) & (yhat > 0.0)
    if int(valid.sum()) < 1:
        return float("nan")
    yv = np.clip(y[valid], floor, None)
    hv = np.clip(yhat[valid], floor, None)
    ratio = yv / hv
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def date_level_equal_weight(dates: object, values: Array) -> tuple[NDArray[np.object_], Array]:
    """Equal-weight mean of finite values by date, sorted by date.

    Dates with no finite observations are omitted. Length mismatch fails closed.
    Research-diagnostic only — not a live-performance claim.
    """
    raw_dates = np.asarray(dates)
    raw_values = _as_1d("values", np.asarray(values, dtype=float))
    if raw_dates.shape[0] != raw_values.shape[0]:
        raise ValueError("dates and values must have the same length")
    if raw_dates.size == 0:
        return np.asarray([], dtype=object), np.asarray([], dtype=float)
    ordered = sorted(set(raw_dates.tolist()))
    out_dates: list[Any] = []
    out_values: list[float] = []
    for date in ordered:
        finite = raw_values[raw_dates == date]
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            continue
        out_dates.append(date)
        out_values.append(float(np.mean(finite)))
    return np.asarray(out_dates, dtype=object), np.asarray(out_values, dtype=float)


def nonoverlapping_origin_mask(session_positions: object, horizon_bars: int) -> NDArray[np.bool_]:
    """Keep origins whose *h*-bar realized windows do not overlap.

    ``session_positions`` are strictly increasing integer session indices
    aligned with the origin series. Origin *i* is kept if its session index is
    at least ``horizon_bars`` after the previously kept origin (Hansen–Lunde
    nonoverlapping subsample). Empty input returns an empty mask.
    """
    horizon = _require_positive_int("horizon_bars", horizon_bars)
    pos = np.asarray(session_positions)
    if pos.ndim != 1:
        raise ValueError("session_positions must be 1d")
    if pos.size == 0:
        return np.asarray([], dtype=bool)
    if not np.issubdtype(pos.dtype, np.integer):
        if not np.issubdtype(pos.dtype, np.floating) or not np.all(pos == np.floor(pos)):
            raise ValueError("session_positions must be integer-valued")
        pos = pos.astype(int)
    if not np.isfinite(np.asarray(pos, dtype=float)).all():
        raise ValueError("session_positions must be finite")
    if pos.size >= 2 and np.any(np.diff(pos.astype(int)) <= 0):
        raise ValueError("session_positions must be strictly increasing")
    keep = np.zeros(pos.size, dtype=bool)
    next_allowed = int(pos[0])
    for i, session in enumerate(pos.tolist()):
        index = int(session)
        if index >= next_allowed:
            keep[i] = True
            next_allowed = index + horizon
    return keep


def overlap_aware_qlike(
    dates: object,
    forecast: Array,
    realized: Array,
    *,
    horizon_bars: int,
    floor: float = 1e-12,
    session_index: dict[Any, int] | None = None,
) -> dict[str, Any]:
    """Date-level QLIKE for a market forecast with nonoverlapping *h*-step origins.

    A date-level (equal-weight) forecast must be unique within each date;
    name-level realized variance is collapsed with the same equal-weight mean.
    Primary ``qlike`` uses the nonoverlapping origin subsample so consecutive
    *h*-bar realized windows are not double-counted. ``qlike_overlapping`` is
    the diagnostic on every origin date. Empty nonoverlapping subsamples fail
    closed rather than silently scoring overlapping windows as independent.
    """
    horizon = _require_positive_int("horizon_bars", horizon_bars)
    if not np.isfinite(floor) or float(floor) <= 0.0:
        raise ValueError("floor must be finite and positive")
    raw_dates = np.asarray(dates)
    yhat = _as_1d("forecast", np.asarray(forecast, dtype=float))
    y = _as_1d("realized", np.asarray(realized, dtype=float))
    if raw_dates.shape[0] != yhat.shape[0] or yhat.shape[0] != y.shape[0]:
        raise ValueError("dates, forecast, and realized must have the same length")
    if raw_dates.size == 0:
        raise ValueError("overlap-aware QLIKE requires at least one observation")
    for date in sorted(set(raw_dates.tolist())):
        finite = yhat[raw_dates == date]
        finite = finite[np.isfinite(finite)]
        if finite.size > 1 and int(np.unique(finite).size) > 1:
            raise ValueError("forecast must be unique within each date")
    date_keys, forecast_d = date_level_equal_weight(raw_dates, yhat)
    realized_keys, realized_d = date_level_equal_weight(raw_dates, y)
    if date_keys.size == 0 or realized_keys.size == 0:
        raise ValueError("overlap-aware QLIKE has no finite date-level observations")
    if date_keys.shape != realized_keys.shape or not np.array_equal(date_keys, realized_keys):
        shared = [key for key in date_keys.tolist() if key in set(realized_keys.tolist())]
        if not shared:
            raise ValueError("overlap-aware QLIKE has no overlapping finite dates")
        forecast_lookup = {
            key: float(val) for key, val in zip(date_keys.tolist(), forecast_d, strict=True)
        }
        realized_lookup = {
            key: float(val) for key, val in zip(realized_keys.tolist(), realized_d, strict=True)
        }
        date_keys = np.asarray(shared, dtype=object)
        forecast_d = np.asarray([forecast_lookup[key] for key in shared], dtype=float)
        realized_d = np.asarray([realized_lookup[key] for key in shared], dtype=float)
    if session_index is None:
        positions = np.arange(date_keys.size, dtype=int)
    else:
        try:
            positions = np.asarray(
                [int(session_index[key]) for key in date_keys.tolist()], dtype=int
            )
        except KeyError as exc:
            raise ValueError("session_index missing origin date") from exc
        if positions.size >= 2 and np.any(np.diff(positions) <= 0):
            raise ValueError("session_index positions must be strictly increasing along dates")
    keep = nonoverlapping_origin_mask(positions, horizon)
    if int(keep.sum()) < 1:
        raise ValueError("nonoverlapping origin subsample is empty")
    return {
        "qlike": qlike(realized_d[keep], forecast_d[keep], floor),
        "qlike_overlapping": qlike(realized_d, forecast_d, floor),
        "n_origins_nonoverlapping": int(keep.sum()),
        "n_origins_overlapping": int(date_keys.size),
        "horizon_bars": horizon,
        "scoring_scope": "date_level_equal_weight",
    }


def _mean_finite(values: Array) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def one_step_density_summary(
    dates: object,
    log_scores: Array,
    crps: Array,
    pits: Array,
    *,
    horizon_bars: int,
    session_index: dict[Any, int] | None = None,
) -> dict[str, Any]:
    """Aggregate one-step density scores on unique date-level origins.

    Primary ``log_score_one_step`` / ``crps_one_step`` average every origin:
    a one-step return density does not double-count an *h*-bar realized window.
    Companion ``*_qlike_origins`` keys reuse the Hansen–Lunde stride so the
    density diagnostics can be compared on the same origin subset as QLIKE.
    Empty input fails closed. Research-diagnostic only.
    """
    horizon = _require_positive_int("horizon_bars", horizon_bars)
    raw_dates = np.asarray(dates)
    log_arr = _as_1d("log_scores", np.asarray(log_scores, dtype=float))
    crps_arr = _as_1d("crps", np.asarray(crps, dtype=float))
    pit_arr = _as_1d("pits", np.asarray(pits, dtype=float))
    if raw_dates.shape[0] != log_arr.shape[0] or log_arr.shape[0] != crps_arr.shape[0]:
        raise ValueError("dates, log_scores, and crps must have the same length")
    if pit_arr.shape[0] != log_arr.shape[0]:
        raise ValueError("pits must have the same length as log_scores")
    if raw_dates.size == 0:
        raise ValueError("one-step density summary requires at least one origin")
    if len(set(raw_dates.tolist())) != raw_dates.size:
        raise ValueError("one-step density origins must be unique dates")
    if session_index is None:
        positions = np.arange(raw_dates.size, dtype=int)
    else:
        try:
            positions = np.asarray(
                [int(session_index[key]) for key in raw_dates.tolist()], dtype=int
            )
        except KeyError as exc:
            raise ValueError("session_index missing origin date") from exc
        if positions.size >= 2 and np.any(np.diff(positions) <= 0):
            raise ValueError("session_index positions must be strictly increasing along dates")
    keep = nonoverlapping_origin_mask(positions, horizon)
    log_mean = _mean_finite(log_arr)
    pit_stat, pit_p = pit_ks(pit_arr)
    return {
        "log_score_one_step": log_mean,
        "ignorance_one_step": (-log_mean if np.isfinite(log_mean) else float("nan")),
        "crps_one_step": _mean_finite(crps_arr),
        "pit_ks_one_step": float(pit_stat),
        "pit_ks_p_one_step": float(pit_p),
        "n_density_origins": int(raw_dates.size),
        "n_density_origins_qlike_stride": int(keep.sum()),
        "log_score_one_step_qlike_origins": _mean_finite(log_arr[keep]),
        "crps_one_step_qlike_origins": _mean_finite(crps_arr[keep]),
        "density_horizon": 1,
        "density_target": "date_level_ret_1",
        "scoring_scope": "date_level_equal_weight",
        "horizon_bars": horizon,
    }


def _security_id_list(security_ids: object) -> list[str]:
    raw = np.asarray(security_ids)
    if raw.ndim != 1:
        raise ValueError("security_ids must be 1d")
    out: list[str] = []
    for item in raw.tolist():
        if isinstance(item, bool) or item is None or not isinstance(item, str):
            raise ValueError("security_ids must be strings")
        sid = item.strip()
        if not sid:
            raise ValueError("security_ids must be non-empty")
        out.append(sid)
    return out


def _require_unique_name_dates(security_ids: list[str], dates: np.ndarray) -> None:
    keys = list(zip(security_ids, dates.tolist(), strict=True))
    if len(keys) != len(set(keys)):
        raise ValueError("name-level origins must be unique (security_id, event_time)")


def _per_name_nonoverlapping_mask(
    security_ids: list[str],
    dates: np.ndarray,
    *,
    horizon_bars: int,
    session_index: dict[Any, int] | None,
) -> NDArray[np.bool_]:
    """Hansen–Lunde stride applied independently on each name's calendar."""
    n = len(security_ids)
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return keep
    calendar = (
        {date: i for i, date in enumerate(sorted(set(dates.tolist())))}
        if session_index is None
        else session_index
    )
    by_name: dict[str, list[int]] = {}
    for i, sid in enumerate(security_ids):
        by_name.setdefault(sid, []).append(i)
    for idxs in by_name.values():
        ordered = sorted(idxs, key=lambda i: (dates[i], i))
        try:
            positions = np.asarray([int(calendar[dates[i]]) for i in ordered], dtype=int)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("session_index missing origin date") from exc
        if positions.size >= 2 and np.any(np.diff(positions) <= 0):
            raise ValueError("session_index positions must be strictly increasing along dates")
        name_keep = nonoverlapping_origin_mask(positions, horizon_bars)
        for flag, index in zip(name_keep.tolist(), ordered, strict=True):
            keep[index] = bool(flag)
    return keep


def name_level_qlike(
    security_ids: object,
    dates: object,
    forecast: Array,
    realized: Array,
    *,
    horizon_bars: int,
    floor: float = 1e-12,
    session_index: dict[Any, int] | None = None,
) -> dict[str, Any]:
    """Per-name QLIKE with Hansen–Lunde stride applied independently per name.

    Unlike ``overlap_aware_qlike``, this does not equal-weight collapse names
    within a date. Duplicate ``(security_id, event_time)`` keys fail closed.
    Primary ``qlike`` pools the union of per-name nonoverlapping origins.
    ``qlike_overlapping`` scores every name-date pair. Research-diagnostic
    only — not a live-performance or market-overlay claim.
    """
    horizon = _require_positive_int("horizon_bars", horizon_bars)
    if not np.isfinite(floor) or float(floor) <= 0.0:
        raise ValueError("floor must be finite and positive")
    ids = _security_id_list(security_ids)
    raw_dates = np.asarray(dates)
    yhat = _as_1d("forecast", np.asarray(forecast, dtype=float))
    y = _as_1d("realized", np.asarray(realized, dtype=float))
    if raw_dates.ndim != 1:
        raise ValueError("dates must be 1d")
    if raw_dates.shape[0] != len(ids) or yhat.shape[0] != len(ids) or y.shape[0] != len(ids):
        raise ValueError("security_ids, dates, forecast, and realized must have the same length")
    if raw_dates.size == 0:
        raise ValueError("name-level QLIKE requires at least one observation")
    _require_unique_name_dates(ids, raw_dates)
    keep = _per_name_nonoverlapping_mask(
        ids, raw_dates, horizon_bars=horizon, session_index=session_index
    )
    if int(keep.sum()) < 1:
        raise ValueError("nonoverlapping origin subsample is empty")
    return {
        "qlike": qlike(y[keep], yhat[keep], floor),
        "qlike_overlapping": qlike(y, yhat, floor),
        "n_origins_nonoverlapping": int(keep.sum()),
        "n_origins_overlapping": int(raw_dates.size),
        "n_names": int(len(set(ids))),
        "horizon_bars": horizon,
        "scoring_scope": "security_level_ret_1",
    }


def name_level_one_step_density_summary(
    security_ids: object,
    dates: object,
    log_scores: Array,
    crps: Array,
    pits: Array,
    *,
    horizon_bars: int,
    session_index: dict[Any, int] | None = None,
) -> dict[str, Any]:
    """Aggregate one-step density scores on unique ``(security_id, event_time)``.

    Primary ``log_score_one_step`` / ``crps_one_step`` average every name-origin
    pair. Companion ``*_qlike_origins`` keys reuse the per-name Hansen–Lunde
    stride so density diagnostics can be compared on the same subsample as
    name-level QLIKE. PIT KS is a pooled calibration diagnostic; contemporaneous
    names remain cross-sectionally dependent. Empty input fails closed.
    Research-diagnostic only.
    """
    horizon = _require_positive_int("horizon_bars", horizon_bars)
    ids = _security_id_list(security_ids)
    raw_dates = np.asarray(dates)
    log_arr = _as_1d("log_scores", np.asarray(log_scores, dtype=float))
    crps_arr = _as_1d("crps", np.asarray(crps, dtype=float))
    pit_arr = _as_1d("pits", np.asarray(pits, dtype=float))
    if raw_dates.ndim != 1:
        raise ValueError("dates must be 1d")
    if raw_dates.shape[0] != len(ids) or log_arr.shape[0] != len(ids):
        raise ValueError("security_ids, dates, and log_scores must have the same length")
    if crps_arr.shape[0] != len(ids) or pit_arr.shape[0] != len(ids):
        raise ValueError("crps and pits must have the same length as security_ids")
    if raw_dates.size == 0:
        raise ValueError("name-level one-step density summary requires at least one origin")
    _require_unique_name_dates(ids, raw_dates)
    keep = _per_name_nonoverlapping_mask(
        ids, raw_dates, horizon_bars=horizon, session_index=session_index
    )
    log_mean = _mean_finite(log_arr)
    pit_stat, pit_p = pit_ks(pit_arr)
    return {
        "log_score_one_step": log_mean,
        "ignorance_one_step": (-log_mean if np.isfinite(log_mean) else float("nan")),
        "crps_one_step": _mean_finite(crps_arr),
        "pit_ks_one_step": float(pit_stat),
        "pit_ks_p_one_step": float(pit_p),
        "n_density_origins": int(raw_dates.size),
        "n_density_origins_qlike_stride": int(keep.sum()),
        "n_names": int(len(set(ids))),
        "log_score_one_step_qlike_origins": _mean_finite(log_arr[keep]),
        "crps_one_step_qlike_origins": _mean_finite(crps_arr[keep]),
        "density_horizon": 1,
        "density_target": "security_level_ret_1",
        "scoring_scope": "security_level_ret_1",
        "horizon_bars": horizon,
    }


def pearson_ic(pred: Array, realized: Array) -> float:
    p = _as_1d("pred", pred)
    r = _as_1d("realized", realized)
    _require_same_length(("pred", p), ("realized", r))
    mask = np.isfinite(p) & np.isfinite(r)
    if mask.sum() < 3:
        return float("nan")
    p, r = p[mask], r[mask]
    if np.std(p) < _ZERO_STD or np.std(r) < _ZERO_STD:
        return float("nan")
    return float(np.corrcoef(p, r)[0, 1])


def rank_ic(pred: Array, realized: Array) -> float:
    p = _as_1d("pred", pred)
    r = _as_1d("realized", realized)
    _require_same_length(("pred", p), ("realized", r))
    mask = np.isfinite(p) & np.isfinite(r)
    if mask.sum() < 3:
        return float("nan")
    pr = _rankdata(p[mask])
    rr = _rankdata(r[mask])
    return pearson_ic(pr, rr)


def _rankdata(x: Array) -> Array:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.bincount(inv, weights=ranks)
        ranks = sums[inv] / counts[inv]
    return ranks


def icir(ics: Array, annualize: bool = False, periods_per_year: float = 252.0) -> float:
    x = _as_1d("ics", ics)
    x = x[np.isfinite(x)]
    if x.size < 2 or np.std(x, ddof=1) < _ZERO_STD:
        return float("nan")
    ir = float(np.mean(x) / np.std(x, ddof=1))
    if annualize:
        ir *= np.sqrt(periods_per_year)
    return ir


def pit_values(y: Array, quantiles: Array, taus: Array) -> Array:
    """Approximate PIT by interpolating the empirical CDF defined by quantiles."""
    q = np.asarray(quantiles, dtype=float)
    t = np.asarray(taus, dtype=float)
    y = _as_1d("y", y)
    if q.ndim != 2:
        raise ValueError("quantiles must be 2d")
    if q.shape[1] != t.size:
        raise ValueError("quantiles must be (n, k) matching taus")
    if y.shape[0] != q.shape[0]:
        raise ValueError(f"length mismatch: y={y.shape[0]}, quantiles rows={q.shape[0]}")
    if not np.isfinite(t).all() or np.any(np.diff(t) < 0.0) or np.any((t <= 0.0) | (t >= 1.0)):
        raise ValueError("taus must be finite, strictly inside (0, 1), and nondecreasing")
    out = np.full(y.shape[0], np.nan, dtype=float)
    for i in range(y.shape[0]):
        if not np.isfinite(y[i]) or not np.isfinite(q[i]).all():
            continue
        if np.any(np.diff(q[i]) < 0.0):
            continue
        out[i] = float(np.interp(y[i], q[i], t, left=0.0, right=1.0))
    return out


def _require_alpha_coverage(alpha: float) -> None:
    """VaR coverage level α ∈ (0, 1), e.g. 0.95 — matches MATH_SPEC VaR_α."""
    if not np.isfinite(alpha) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")


def fissler_ziegel_loss(losses: Array, var: Array, es: Array, alpha: float) -> Array:
    """Elementwise Fissler–Ziegel 0-homogeneous joint (VaR, ES) score (research-only).

    Loss convention: positive losses ``L = -R`` (same as Acerbi / MATH_SPEC).
    ``alpha`` is the VaR **coverage** level (e.g. 0.95), so ``VaR_α = q_α(L)`` and
    breach rate is ``1 - alpha``. Forecasts ``var`` / ``es`` must be aligned 1d;
    ``es`` must be strictly positive for the log term.

    Nolde–Ziegel / Fissler–Ziegel **FZ0** member (G₁≡0, G₂=log) for level α
    (Fissler & Ziegel 2016, eq. 3.3 with G₂=log; Nolde & Ziegel 2017):

    .. math::

        S(v,e;L) = \\frac{1}{1-\\alpha} 1_{L>v}\\frac{L-v}{e}
                   + \\frac{v}{e} - 1 + \\log(e)

    Lower expected score is better. Jointly **consistent (proper)** for
    ``(VaR_α, ES_α)``: the expected score is uniquely minimized at the true
    quantile/ES pair. An earlier variant without the ``1/(1-α)`` scaling on the
    hit term (and with an extra ``(1-α)`` factor on ``v/e``) was *not* proper —
    it minimized at ``e = (1-α)·ES`` instead of ``ES`` — and was corrected here.
    Not a live capital claim.
    Empty → empty array; length mismatch / bad α → ValueError; non-positive or
    non-finite ``es`` (or non-finite inputs) → NaN at those indices.
    """
    _require_alpha_coverage(alpha)
    L = _as_1d("losses", losses)
    V = _as_1d("var", var)
    E = _as_1d("es", es)
    _require_same_length(("losses", L), ("var", V), ("es", E))
    if L.size == 0:
        return np.asarray([], dtype=float)
    out = np.full(L.shape, np.nan, dtype=float)
    ok = np.isfinite(L) & np.isfinite(V) & np.isfinite(E) & (E > 0.0)
    if not np.any(ok):
        return out
    Lo, Vo, Eo = L[ok], V[ok], E[ok]
    pb = 1.0 - float(alpha)
    hit = Lo > Vo
    s = np.zeros_like(Lo)
    s = s + hit.astype(float) * (Lo - Vo) / Eo / pb
    s = s + Vo / Eo - 1.0 + np.log(Eo)
    out[ok] = s
    return out


def mean_fissler_ziegel(losses: Array, var: Array, es: Array, alpha: float) -> float:
    """Mean Fissler–Ziegel FZ0 joint VaR/ES score. Empty / all-NaN → honest NaN.

    Research-diagnostic only — never live Sharpe / promotion evidence.
    """
    scores = fissler_ziegel_loss(losses, var, es, alpha)
    if scores.size == 0:
        return float("nan")
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))

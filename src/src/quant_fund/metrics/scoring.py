"""Proper scoring rules and forecast metrics. See docs/MATH_SPEC.md."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import betaln, erf
from scipy.stats import t as student_t

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


GARCH_ONE_STEP_CRPS_TAUS: tuple[float, ...] = (
    0.01,
    0.025,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.975,
    0.99,
)


def log_score_gaussian(y: Array, mu: Array, sigma: Array) -> Array:
    """Elementwise Gaussian log density with fail-closed invalid rows."""
    y = _as_1d("y", y)
    mu = _as_1d("mu", mu)
    sigma = _as_1d("sigma", sigma)
    _require_same_length(("y", y), ("mu", mu), ("sigma", sigma))
    out = np.full(y.shape, np.nan, dtype=float)
    valid = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0.0)
    if np.any(valid):
        z = (y[valid] - mu[valid]) / sigma[valid]
        out[valid] = -0.5 * (np.log(2.0 * np.pi) + z * z) - np.log(sigma[valid])
    return out


def mean_log_score_gaussian(y: Array, mu: Array, sigma: Array) -> float:
    """Mean finite Gaussian log score, or NaN when no row is valid."""
    scores = log_score_gaussian(y, mu, sigma)
    finite = scores[np.isfinite(scores)]
    return float(np.mean(finite)) if finite.size else float("nan")


def date_level_equal_weight(dates: Array, values: Array) -> tuple[Array, Array]:
    """Return sorted dates and the finite cross-sectional mean for each date."""
    d = np.asarray(dates, dtype=object).reshape(-1)
    v = _as_1d("values", values)
    if d.size != v.size:
        raise ValueError("dates and values must have the same length")
    grouped: dict[object, list[float]] = {}
    for date, value in zip(d.tolist(), v.tolist(), strict=True):
        if date is None or not np.isfinite(value):
            continue
        grouped.setdefault(date, []).append(float(value))
    if not grouped:
        return np.asarray([], dtype=object), np.asarray([], dtype=float)
    try:
        keys = sorted(cast(Any, grouped))
    except TypeError:
        keys = sorted(grouped, key=repr)
    return np.asarray(keys, dtype=object), np.asarray(
        [np.mean(grouped[key]) for key in keys], dtype=float
    )


def _validate_horizon(horizon_bars: int) -> int:
    if isinstance(horizon_bars, bool) or not isinstance(horizon_bars, (int, np.integer)):
        raise ValueError("horizon_bars must be a positive integer")
    horizon = int(horizon_bars)
    if horizon <= 0:
        raise ValueError("horizon_bars must be a positive integer")
    return horizon


def nonoverlapping_origin_mask(
    session_positions: NDArray[np.integer[Any]], horizon_bars: int
) -> NDArray[np.bool_]:
    """Keep origins separated by at least the forecast horizon."""
    horizon = _validate_horizon(horizon_bars)
    positions = np.asarray(session_positions)
    if positions.ndim != 1:
        raise ValueError("session_positions must be one-dimensional")
    if positions.size and (
        not np.issubdtype(positions.dtype, np.integer)
        or not np.isfinite(positions.astype(float)).all()
        or np.any(np.diff(positions) <= 0)
    ):
        raise ValueError("session_positions must be strictly increasing integers")
    keep = np.zeros(positions.size, dtype=bool)
    if positions.size:
        keep[0] = True
        last = int(positions[0])
        for i in range(1, positions.size):
            if int(positions[i]) >= last + horizon:
                keep[i] = True
                last = int(positions[i])
    return keep


def _session_positions(
    dates: Array, session_index: dict[object, int] | None
) -> NDArray[np.int_]:
    d = np.asarray(dates, dtype=object).reshape(-1)
    unique = list(dict.fromkeys(d.tolist()))
    if session_index is None:
        try:
            ordered = sorted(unique)
        except TypeError:
            ordered = sorted(unique, key=repr)
        mapping = {date: i for i, date in enumerate(ordered)}
    else:
        missing = [date for date in unique if date not in session_index]
        if missing:
            raise ValueError("session_index missing date")
        mapping = session_index
    return np.asarray([int(mapping[date]) for date in d.tolist()], dtype=int)


def overlap_aware_qlike(
    dates: Array,
    forecast_var: Array,
    realized_var: Array,
    *,
    horizon_bars: int,
    floor: float = 1e-12,
    session_index: dict[object, int] | None = None,
) -> dict[str, object]:
    """Score date-level forecasts, with a non-overlapping-origin companion."""
    horizon = _validate_horizon(horizon_bars)
    d = np.asarray(dates, dtype=object).reshape(-1)
    f = _as_1d("forecast_var", forecast_var)
    y = _as_1d("realized_var", realized_var)
    _require_same_length(("dates", d), ("forecast_var", f), ("realized_var", y))
    if d.size == 0:
        raise ValueError("at least one observation is required")
    if len(set(d.tolist())) != d.size:
        for date in dict.fromkeys(d.tolist()):
            mask = d == date
            if np.unique(f[mask][np.isfinite(f[mask])]).size > 1:
                raise ValueError("forecast must be unique within each date")
    date_keys, f_date = date_level_equal_weight(d, f)
    realized_keys, y_date = date_level_equal_weight(d, y)
    if date_keys.size == 0 or not np.array_equal(date_keys, realized_keys):
        raise ValueError("no finite date-level observations")
    all_score = qlike(y_date, f_date, floor=floor)
    positions = _session_positions(date_keys, session_index)
    keep = nonoverlapping_origin_mask(positions, horizon)
    nonoverlap = qlike(y_date[keep], f_date[keep], floor=floor)
    return {
        "qlike": float(nonoverlap),
        "qlike_overlapping": float(all_score),
        "n_origins_overlapping": int(date_keys.size),
        "n_origins_nonoverlapping": int(keep.sum()),
        "horizon_bars": horizon,
        "scoring_scope": "date_level_equal_weight",
    }


def _density_summary(
    log_scores: Array,
    crps: Array,
    pits: Array,
    keep: NDArray[np.bool_],
    *,
    density_target: str,
    scoring_scope: str,
    n_names: int | None = None,
    horizon: int,
) -> dict[str, object]:
    from quant_fund.metrics.probability import pit_ks

    log_s = _as_1d("log_scores", log_scores)
    crps_a = _as_1d("crps", crps)
    pit_a = _as_1d("pits", pits)
    _require_same_length(("log_scores", log_s), ("crps", crps_a), ("pits", pit_a))
    valid = np.isfinite(log_s) & np.isfinite(crps_a) & np.isfinite(pit_a)
    if not np.any(valid):
        raise ValueError("at least one finite density origin is required")
    selected = valid & keep
    if not np.any(selected):
        selected = valid
    ks, p = pit_ks(pit_a[valid])
    result: dict[str, object] = {
        "log_score_one_step": float(np.mean(log_s[valid])),
        "ignorance_one_step": float(-np.mean(log_s[valid])),
        "crps_one_step": float(np.mean(crps_a[valid])),
        "pit_ks_one_step": ks,
        "pit_ks_p_one_step": p,
        "n_density_origins": int(valid.sum()),
        "n_density_origins_qlike_stride": int(selected.sum()),
        "log_score_one_step_qlike_origins": float(np.mean(log_s[selected])),
        "crps_one_step_qlike_origins": float(np.mean(crps_a[selected])),
        "density_horizon": 1,
        "density_target": density_target,
        "scoring_scope": scoring_scope,
        "horizon_bars": horizon,
    }
    if n_names is not None:
        result["n_names"] = int(n_names)
    return result


def one_step_density_summary(
    dates: Array,
    log_scores: Array,
    crps: Array,
    pits: Array,
    *,
    horizon_bars: int,
    session_index: dict[object, int] | None = None,
) -> dict[str, object]:
    horizon = _validate_horizon(horizon_bars)
    d = np.asarray(dates, dtype=object).reshape(-1)
    if d.size == 0:
        raise ValueError("at least one origin is required")
    if len(set(d.tolist())) != d.size:
        raise ValueError("density origins must have unique dates")
    positions = _session_positions(d, session_index)
    keep = nonoverlapping_origin_mask(positions, horizon)
    return _density_summary(
        log_scores,
        crps,
        pits,
        keep,
        density_target="date_level_ret_1",
        scoring_scope="date_level_equal_weight",
        horizon=horizon,
    )


def name_level_qlike(
    security_ids: Array,
    dates: Array,
    forecast_var: Array,
    realized_var: Array,
    *,
    horizon_bars: int,
    floor: float = 1e-12,
    session_index: dict[object, int] | None = None,
) -> dict[str, object]:
    horizon = _validate_horizon(horizon_bars)
    ids = np.asarray(security_ids, dtype=object).reshape(-1)
    d = np.asarray(dates, dtype=object).reshape(-1)
    f = _as_1d("forecast_var", forecast_var)
    y = _as_1d("realized_var", realized_var)
    _require_same_length(("security_ids", ids), ("dates", d), ("forecast_var", f), ("realized_var", y))
    if ids.size == 0:
        raise ValueError("at least one observation is required")
    if any(not isinstance(value, str) for value in ids.tolist()):
        raise ValueError("security_ids must contain strings")
    pairs = list(zip(ids.tolist(), d.tolist(), strict=True))
    if len(set(pairs)) != len(pairs):
        raise ValueError("security/date observations must be unique")
    valid = np.isfinite(f) & np.isfinite(y)
    if not np.any(valid):
        raise ValueError("at least one finite observation is required")
    all_score = qlike(y[valid], f[valid], floor=floor)
    keep = np.zeros(ids.size, dtype=bool)
    for security in dict.fromkeys(ids.tolist()):
        row = np.flatnonzero((ids == security) & valid)
        order = np.argsort(_session_positions(d[row], session_index), kind="mergesort")
        row = row[order]
        keep[row] = nonoverlapping_origin_mask(_session_positions(d[row], session_index), horizon)
    selected = valid & keep
    return {
        "qlike": float(qlike(y[selected], f[selected], floor=floor)),
        "qlike_overlapping": float(all_score),
        "n_origins_overlapping": int(valid.sum()),
        "n_origins_nonoverlapping": int(selected.sum()),
        "n_names": int(len(set(ids[valid].tolist()))),
        "horizon_bars": horizon,
        "scoring_scope": "security_level_return_series",
    }


def name_level_one_step_density_summary(
    security_ids: Array,
    dates: Array,
    log_scores: Array,
    crps: Array,
    pits: Array,
    *,
    horizon_bars: int,
    session_index: dict[object, int] | None = None,
) -> dict[str, object]:
    horizon = _validate_horizon(horizon_bars)
    ids = np.asarray(security_ids, dtype=object).reshape(-1)
    d = np.asarray(dates, dtype=object).reshape(-1)
    _require_same_length(("security_ids", ids), ("dates", d), ("log_scores", _as_1d("log_scores", log_scores)))
    if ids.size == 0:
        raise ValueError("at least one origin is required")
    if any(not isinstance(value, str) for value in ids.tolist()):
        raise ValueError("security_ids must contain strings")
    keep = np.zeros(ids.size, dtype=bool)
    for security in dict.fromkeys(ids.tolist()):
        row = np.flatnonzero(ids == security)
        order = np.argsort(_session_positions(d[row], session_index), kind="mergesort")
        row = row[order]
        keep[row] = nonoverlapping_origin_mask(_session_positions(d[row], session_index), horizon)
    return _density_summary(
        log_scores,
        crps,
        pits,
        keep,
        density_target="security_level_ret_1",
        scoring_scope="security_level_return_series",
        n_names=len(set(ids.tolist())),
        horizon=horizon,
    )


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

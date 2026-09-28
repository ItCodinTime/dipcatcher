"""Information-form and numerically robust Kalman filter variants.

Companion to ``nonlinear_filters`` (EKF/UKF/particle): this module supplies
linear-Gaussian filters in numerically hardened parameterizations for

    x_t = F x_{t-1} + w_t,  w_t ~ N(0, Q)
    y_t = H x_t + v_t,      v_t ~ N(0, R)

- ``information_filter``: Anderson & Moore (1979) information (inverse
  covariance) recursion — propagates I = P^-1 and i = P^-1 x. Update is
  additive: I += H^T R^-1 H, i += H^T R^-1 y.
- ``huber_kalman``: Masreliez (1975) / Schick & Mitter (1994) robust filter —
  the score contribution of each observation is winsorized at Huber's c,
  giving bounded-influence filtering under heavy-tailed measurement noise.
- ``sqrt_kalman``: square-root (Potter) filter — propagates S with
  P = S S^T so the covariance stays symmetric PSD by construction.

All return filtered means, filtered covariances, and the Gaussian log
predictive likelihood. Fail-closed on dimension mismatch / non-finite input.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _ssm(
    y: Array,
    f: Array,
    h: Array,
    q: Array,
    r: Array,
    x0: Array,
    p0: Array,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, int]:
    yy = np.asarray(y, dtype=float)
    if yy.ndim == 1:
        yy = yy[:, None]
    if yy.ndim != 2 or yy.shape[0] < 2 or not np.isfinite(yy).all():
        raise ValueError("y must be a finite (T, m) array with T >= 2")
    t_len, m = yy.shape
    f = np.asarray(f, dtype=float)
    h = np.asarray(h, dtype=float)
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)
    x0 = np.asarray(x0, dtype=float).ravel()
    p0 = np.asarray(p0, dtype=float)
    n = x0.size
    if f.shape != (n, n):
        raise ValueError("F must be (n, n)")
    if h.shape != (m, n):
        raise ValueError("H must be (m, n)")
    if q.shape != (n, n) or r.shape != (m, m):
        raise ValueError("Q must be (n, n), R must be (m, m)")
    if p0.shape != (n, n):
        raise ValueError("P0 must be (n, n)")
    for name, a in (("F", f), ("H", h), ("Q", q), ("R", r), ("P0", p0), ("x0", x0)):
        if not np.isfinite(a).all():
            raise ValueError(f"non-finite {name}")
    return yy, f, h, q, r, x0, p0, t_len


def _innovation_nll(x_pred: Array, p_pred: Array, y_t: Array, h: Array, r: Array) -> float:
    e = y_t - h @ x_pred
    s = h @ p_pred @ h.T + r
    sign, logdet = np.linalg.slogdet(s)
    if sign <= 0:
        return float("inf")
    return float(0.5 * (logdet + e @ np.linalg.solve(s, e) + e.size * np.log(2.0 * np.pi)))


def information_filter(
    y: Array,
    f: Array,
    h: Array,
    q: Array,
    r: Array,
    x0: Array,
    p0: Array,
) -> dict[str, Array | float]:
    """Anderson & Moore information filter (reciprocal-covariance recursion)."""
    yy, f, h, q, r, x0, p0, t_len = _ssm(y, f, h, q, r, x0, p0)
    n = x0.size
    r_inv = np.linalg.inv(r)
    info_m = np.linalg.inv(p0)
    info_v = info_m @ x0
    xs = np.empty((t_len, n))
    ps = np.empty((t_len, n, n))
    nll = 0.0
    x_prev = x0.copy()
    for t in range(t_len):
        if t > 0:
            p_prev = np.linalg.inv(info_m)
            p_pred = f @ p_prev @ f.T + q
            info_m = np.linalg.inv(p_pred)
            info_v = info_m @ (f @ x_prev)
            nll += _innovation_nll(f @ x_prev, p_pred, yy[t], h, r)
        else:
            nll += _innovation_nll(x0, p0, yy[t], h, r)
        info_m = info_m + h.T @ r_inv @ h
        info_v = info_v + h.T @ r_inv @ yy[t]
        p_new = np.linalg.inv(info_m)
        x_prev = p_new @ info_v
        xs[t] = x_prev
        ps[t] = p_new
    return {"x": xs, "P": ps, "loglik": -nll, "info_matrix": info_m, "info_vector": info_v}


def huber_kalman(
    y: Array,
    f: Array,
    h: Array,
    q: Array,
    r: Array,
    x0: Array,
    p0: Array,
    c: float = 1.345,
) -> dict[str, Array | float]:
    """Masreliez robust Kalman filter; winsorizes standardized innovations."""
    if not np.isfinite(c) or c <= 0:
        raise ValueError("c must be a positive finite clip level")
    yy, f, h, q, r, x0, p0, t_len = _ssm(y, f, h, q, r, x0, p0)
    n = x0.size
    xs = np.empty((t_len, n))
    ps = np.empty((t_len, n, n))
    x, p = x0.copy(), p0.copy()
    nll = 0.0
    for t in range(t_len):
        if t > 0:
            x = f @ x
            p = f @ p @ f.T + q
        nll += _innovation_nll(x, p, yy[t], h, r)
        e = yy[t] - h @ x
        s = h @ p @ h.T + r
        k = np.linalg.solve(s.T, (p @ h.T).T).T
        # winsorize the standardized innovation (bounded influence)
        d = np.sqrt(np.clip(np.diag(s), 1e-300, None))
        e_std = e / d
        e_w = np.where(np.abs(e_std) > c, np.sign(e_std) * c, e_std) * d
        ratio = np.divide(np.abs(e_w), np.abs(e), out=np.ones_like(e), where=np.abs(e) > 1e-300)
        scale = float(np.clip(np.min(ratio), 0.0, 1.0))
        x = x + k @ e_w
        k_eff = k * scale
        p = p - k_eff @ s @ k_eff.T
        p = (p + p.T) / 2.0
        xs[t] = x
        ps[t] = p
    return {"x": xs, "P": ps, "loglik": -nll}


def sqrt_kalman(
    y: Array,
    f: Array,
    h: Array,
    q: Array,
    r: Array,
    x0: Array,
    p0: Array,
) -> dict[str, Array | float]:
    """Square-root (Potter) Kalman filter; propagates S with P = S S^T."""
    yy, f, h, q, r, x0, p0, t_len = _ssm(y, f, h, q, r, x0, p0)
    n = x0.size
    # whiten the observation equation so v ~ N(0, I)
    r_ch = np.linalg.cholesky(r)
    r_ch_inv = np.linalg.inv(r_ch)
    yw = yy @ r_ch_inv.T
    hw = r_ch_inv @ h
    q_ch = np.linalg.cholesky(q)
    s = np.linalg.cholesky(p0)
    x = x0.copy()
    xs = np.empty((t_len, n))
    ps = np.empty((t_len, n, n))
    nll = 0.0
    for t in range(t_len):
        if t > 0:
            x = f @ x
            # time update: P^- = F P F^T + Q => S^- from QR of [(F S) | chol(Q)]^T
            big = np.hstack([f @ s, q_ch]).T
            s = np.linalg.qr(big, mode="r").T[:, :n]
        nll += _innovation_nll(x, s @ s.T, yy[t], h, r)
        # Potter sequential measurement update on whitened scalar obs
        for j in range(hw.shape[0]):
            phi = s.T @ hw[j]
            b = 1.0 / (float(phi @ phi) + 1.0)
            gamma = 1.0 / (1.0 + np.sqrt(b))
            s_phi = s @ phi
            e = yw[t, j] - hw[j] @ x
            x = x + b * e * s_phi
            s = s - (b * gamma) * np.outer(s_phi, phi)
        ps[t] = s @ s.T
        xs[t] = x
    return {"x": xs, "P": ps, "loglik": -nll, "sqrtP": s}

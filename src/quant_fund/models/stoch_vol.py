"""Stochastic volatility: Harvey & Shephard (1996) lognormal-linear
approximation estimated by Kalman QMLE.

Model:  y_t = exp(h_t / 2) e_t,   h_t = phi h_{t-1} + sigma_eta eta_t.
Transform z_t = ln y_t^2 = mu + h_t + eps_t with eps_t = ln e_t^2 -
E[ln e_t^2]; for Gaussian e, E[ln e^2] = -1.2704 (Euler + ln 2) and
Var eps = pi^2 / 2.

State-space form: observation z_t = mu + h_t + eps (R = pi^2/2),
transition h_t = phi h_{t-1} + eta (Q = sigma_eta^2). mu absorbs the
chi-square mean correction.

Fail-closed: |phi| >= 1, non-finite input, y == 0 rows (log invalid).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

Array = NDArray[np.float64]

_E_LNE2 = -1.2704  # E[ln chi^2_1] approx -1.2704
_VAR_LNE2 = np.pi**2 / 2.0  # Var ln chi^2_1


def _kalman_svfilt(z: Array, mu: float, phi: float, s_eta2: float) -> tuple[Array, Array, float]:
    """Kalman filter for the transformed SV system.

    Returns (filtered h_t, predicted variances, -2 loglik/2).
    """
    n = z.size
    h_filt = np.empty(n)
    p_filt = np.empty(n)
    ll = 0.0
    # stationary init for |phi| < 1
    h_pred = 0.0
    p_pred = s_eta2 / max(1.0 - phi * phi, 1e-6)
    for t in range(n):
        # update
        f = p_pred + _VAR_LNE2
        kgain = p_pred / f
        innov = z[t] - mu - h_pred
        ll += -0.5 * (np.log(2 * np.pi * f) + innov * innov / f)
        h_upd = h_pred + kgain * innov
        p_upd = (1.0 - kgain) * p_pred
        h_filt[t] = h_upd
        p_filt[t] = p_upd
        # predict next
        h_pred = phi * h_upd
        p_pred = phi * phi * p_upd + s_eta2
    return h_filt, p_filt, ll


def stoch_vol_fit(y: Array) -> dict[str, Array | float]:
    """SV QMLE via Kalman on z = ln y^2.

    Returns phi, sigma_eta, mu (mean log-variance), filtered h path,
    conditional vol path exp(h/2), loglik.
    """
    yy = np.asarray(y, dtype=float).ravel()
    if yy.size < 100 or not np.isfinite(yy).all():
        raise ValueError("y must be finite with >= 100 obs")
    if (yy == 0).any():
        raise ValueError("y contains zeros; ln y^2 undefined")
    z = np.log(yy * yy) - _E_LNE2

    def nll(th: Array) -> float:
        mu, phi_raw, s_eta = float(th[0]), float(th[1]), float(th[2])
        phi = np.tanh(phi_raw)  # |phi| < 1
        if s_eta <= 0:
            return 1e12
        _, _, ll = _kalman_svfilt(z, mu, phi, s_eta * s_eta)
        return float(-ll) if np.isfinite(ll) else 1e12

    phi0 = np.arctanh(min(0.9, max(-0.9, float(np.corrcoef(z[:-1], z[1:])[0, 1]))))
    th0 = np.array([float(z.mean()), phi0, 0.3])
    res = optimize.minimize(nll, th0, method="Nelder-Mead", options={"maxiter": 4000})
    mu = float(res.x[0])
    phi = float(np.tanh(res.x[1]))
    s_eta = float(res.x[2])
    h_filt, p_filt, ll = _kalman_svfilt(z, mu, phi, s_eta * s_eta)
    return {
        "phi": phi,
        "sigma_eta": abs(s_eta),
        "mu": mu,
        "h": h_filt,
        "h_var": p_filt,
        "vol": np.exp(0.5 * h_filt),
        "loglik": float(ll),
    }

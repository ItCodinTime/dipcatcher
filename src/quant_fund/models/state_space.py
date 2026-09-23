"""Linear-Gaussian state-space machinery: Kalman filter/smoother + OU.

References:
- Kalman (1960). A new approach to linear filtering and prediction problems.
- Rauch, Tung, Striebel (1965). Maximum likelihood estimates of linear dynamic
  systems (RTS smoother).
- Durbin, Koopman (2012). *Time Series Analysis by State Space Methods*.
- Harvey (1989). *Forecasting, Structural Time Series Models and the Kalman
  Filter*.
- Uhlenbeck, Ornstein (1930) / Vasicek (1977) - OU process; MLE via the exact
  AR(1) bridge of the discretized transition.
- Elliott, van der Hoek, Malcolm (2005). Pairs trading via Kalman hedge ratios.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

Array = NDArray[np.float64]


def _as_vector(x: Array, name: str = "x", *, min_obs: int = 8) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size < min_obs:
        raise ValueError(f"{name} must contain at least {min_obs} finite observations")
    return v


def kalman_filter(
    y: Array,
    F: Array,
    H: Array,
    Q: Array,
    R: Array,
    x0: Array | None = None,
    P0: Array | None = None,
) -> dict[str, Array]:
    """Kalman filter for ``x_t = F x_{t-1} + w``, ``y_t = H x_t + v``.

    ``F`` (d,d) transition, ``H`` (m,d) observation, ``Q`` (d,d) state noise,
    ``R`` (m,m) observation noise; ``y`` is (T,m).  Returns filtered/smoothed
    arrays ``x_filt``, ``P_filt``, ``x_pred``, ``P_pred``, ``loglik``
    (sum of Gaussian innovations) and the innovation sequence ``v_t``.
    """
    y_arr = np.atleast_2d(np.asarray(y, dtype=float))
    if y_arr.shape[0] < y_arr.shape[1] and y_arr.shape[0] == 1:
        y_arr = y_arr.T
    if y_arr.ndim != 2 or y_arr.shape[0] < 2:
        raise ValueError("y must be a (T, m) array with T >= 2")
    if not np.isfinite(y_arr).all():
        raise ValueError("y must contain only finite values")
    F_a = np.asarray(F, dtype=float)
    H_a = np.asarray(H, dtype=float)
    Q_a = np.asarray(Q, dtype=float)
    R_a = np.asarray(R, dtype=float)
    d = F_a.shape[0]
    m = y_arr.shape[1]
    for name, mat, shape in (
        ("F", F_a, (d, d)),
        ("H", H_a, (m, d)),
        ("Q", Q_a, (d, d)),
        ("R", R_a, (m, m)),
    ):
        if mat.shape != shape or not np.isfinite(mat).all():
            raise ValueError(f"{name} must be a finite {shape} matrix")
    t_n = y_arr.shape[0]
    x = np.zeros(d) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)
    p_mat = np.eye(d) * 1e6 if P0 is None else np.asarray(P0, dtype=float)
    if x.size != d or p_mat.shape != (d, d):
        raise ValueError("x0/P0 shapes must match the state dimension")
    x_filt = np.empty((t_n, d))
    p_filt = np.empty((t_n, d, d))
    x_pred = np.empty((t_n, d))
    p_pred = np.empty((t_n, d, d))
    innov = np.empty((t_n, m))
    loglik = 0.0
    for t in range(t_n):
        xp = F_a @ x
        pp = F_a @ p_mat @ F_a.T + Q_a
        v = y_arr[t] - H_a @ xp
        s = H_a @ pp @ H_a.T + R_a
        s_inv = np.linalg.pinv(s)
        k = pp @ H_a.T @ s_inv
        x = xp + k @ v
        p_mat = pp - k @ H_a @ pp
        x_filt[t] = x
        p_filt[t] = p_mat
        x_pred[t] = xp
        p_pred[t] = pp
        innov[t] = v
        sign, logdet = np.linalg.slogdet(0.5 * (s + s.T))
        if sign > 0:
            loglik += -0.5 * (m * np.log(2.0 * np.pi) + logdet + float(v @ s_inv @ v))
    return {
        "x_filt": x_filt,
        "P_filt": p_filt,
        "x_pred": x_pred,
        "P_pred": p_pred,
        "innovations": innov,
        "loglik": np.array([loglik]),
    }


def kalman_smoother(filt: dict[str, Array], F: Array) -> dict[str, Array]:
    """Rauch–Tung–Striebel (1965) backward smoother over ``kalman_filter`` output.

    Uses future filtered states. A trading state is ``kalman_filter``, not this.
    """
    F_a = np.asarray(F, dtype=float)
    x_f = np.asarray(filt["x_filt"], dtype=float)
    p_f = np.asarray(filt["P_filt"], dtype=float)
    x_p = np.asarray(filt["x_pred"], dtype=float)
    p_p = np.asarray(filt["P_pred"], dtype=float)
    t_n, d = x_f.shape
    xs = x_f.copy()
    ps = p_f.copy()
    for t in range(t_n - 2, -1, -1):
        j = p_f[t] @ F_a.T @ np.linalg.pinv(p_p[t + 1])
        xs[t] = x_f[t] + j @ (xs[t + 1] - x_p[t + 1])
        ps[t] = p_f[t] + j @ (ps[t + 1] - p_p[t + 1]) @ j.T
    return {"x_smooth": xs, "P_smooth": ps}


def local_level_mle(y: Array) -> dict[str, float]:
    """Fit the local-level (random walk + noise) model by MLE.

    ``y_t = mu_t + eps``; ``mu_t = mu_{t-1} + eta``.  Returns the fitted
    variances and the filtered level path (Harvey 1989, Durbin–Koopman 2012).
    """
    v = _as_vector(y)
    y2 = v.reshape(-1, 1)

    def _nll(theta: Array) -> float:
        sig_e, sig_n = np.exp(theta)
        out = kalman_filter(
            y2, np.array([[1.0]]), np.array([[1.0]]), np.array([[sig_n]]), np.array([[sig_e]])
        )
        return -float(out["loglik"][0])

    var0 = float(np.var(v)) or 1.0
    res = optimize.minimize(_nll, np.log([var0 * 0.5, var0 * 0.1]), method="Nelder-Mead")
    sig_e, sig_n = np.exp(res.x)
    filt = kalman_filter(
        y2, np.array([[1.0]]), np.array([[1.0]]), np.array([[sig_n]]), np.array([[sig_e]])
    )
    sm = kalman_smoother(filt, np.array([[1.0]]))
    return {
        "sigma_obs": float(np.sqrt(sig_e)),
        "sigma_level": float(np.sqrt(sig_n)),
        "loglik": -float(res.fun),
        "converged": float(bool(res.success)),
        "level_final": float(filt["x_filt"][-1, 0]),
        "level_smooth_final": float(sm["x_smooth"][-1, 0]),
        "signal_to_noise": float(sig_n / sig_e) if sig_e > 0 else float("inf"),
    }


def local_linear_trend_mle(y: Array) -> dict[str, float]:
    """Fit the local-linear-trend model (level + slope states) by MLE."""
    v = _as_vector(y)
    y2 = v.reshape(-1, 1)
    f_mat = np.array([[1.0, 1.0], [0.0, 1.0]])
    h_mat = np.array([[1.0, 0.0]])

    def _nll(theta: Array) -> float:
        sig_e, sig_l, sig_s = np.exp(theta)
        q = np.diag([sig_l, sig_s])
        out = kalman_filter(y2, f_mat, h_mat, q, np.array([[sig_e]]))
        return -float(out["loglik"][0])

    var0 = float(np.var(v)) or 1.0
    res = optimize.minimize(
        _nll, np.log([var0 * 0.5, var0 * 0.1, var0 * 0.01]), method="Nelder-Mead"
    )
    sig_e, sig_l, sig_s = np.exp(res.x)
    filt = kalman_filter(y2, f_mat, h_mat, np.diag([sig_l, sig_s]), np.array([[sig_e]]))
    return {
        "sigma_obs": float(np.sqrt(sig_e)),
        "sigma_level": float(np.sqrt(sig_l)),
        "sigma_slope": float(np.sqrt(sig_s)),
        "loglik": -float(res.fun),
        "converged": float(bool(res.success)),
        "level_final": float(filt["x_filt"][-1, 0]),
        "slope_final": float(filt["x_filt"][-1, 1]),
    }


def kalman_hedge_ratio(x: Array, y: Array, delta: float = 1e-5) -> dict[str, Array | float]:
    """Dynamic hedge ratio via Kalman regression (Elliott et al. 2005).

    State ``[alpha_t, beta_t]`` with near-random-walk dynamics (``delta`` =
    state-noise variance fraction of var(y)); observation ``y_t = alpha_t +
    beta_t x_t + eps``.  Returns the filtered beta path and final value.
    """
    xv = _as_vector(x, "x")
    yv = _as_vector(y, "y")
    n = min(xv.size, yv.size)
    xv, yv = xv[:n], yv[:n]
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta must be positive and finite")
    r_var = float(np.var(yv)) * 0.01 + 1e-12
    q = np.eye(2) * delta * float(np.var(yv))
    f_mat = np.eye(2)
    # Time-varying H: run the filter manually per step (H_t = [1, x_t]).
    state = np.zeros(2)
    state[0] = yv[0]
    p_mat = np.eye(2) * 1e4
    betas = np.empty(n)
    alphas = np.empty(n)
    for t in range(n):
        h_t = np.array([[1.0, xv[t]]])
        xp = f_mat @ state
        pp = f_mat @ p_mat @ f_mat.T + q
        v = float((yv[t] - (h_t @ xp)).item())
        s = float((h_t @ pp @ h_t.T).item()) + r_var
        k = (pp @ h_t.T).reshape(-1) / s
        state = xp + k * v
        p_mat = pp - np.outer(k, (h_t @ pp).reshape(-1))
        alphas[t], betas[t] = state[0], state[1]
    out: dict[str, Array | float] = {
        "beta": betas,
        "alpha": alphas,
        "beta_final": float(betas[-1]),
    }
    return out


def ou_mle(x: Array, dt: float = 1.0) -> dict[str, float]:
    """Ornstein–Uhlenbeck MLE via the exact AR(1) transition.

    ``dX = theta (mu - X) dt + sigma dW``; discretely ``X_{t+1} = mu + phi
    (X_t - mu) + eps`` with ``phi = exp(-theta dt)`` and
    ``sigma^2 = sigma_OU^2 (1 - phi^2) / (2 theta)``.  Returns theta, mu,
    sigma_OU, half-life ``ln(2)/theta`` and the implied long-run sd.
    """
    v = _as_vector(x, min_obs=10)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    x0, x1 = v[:-1], v[1:]
    n = x0.size
    s_x = float(x0.sum())
    s_y = float(x1.sum())
    s_xx = float(x0 @ x0)
    s_xy = float(x0 @ x1)
    denom = n * s_xx - s_x * s_x
    if abs(denom) < 1e-18:
        raise ValueError("x has no variation for OU fit")
    phi = (n * s_xy - s_x * s_y) / denom
    phi = float(np.clip(phi, 1e-6, 0.999999))
    mu = (s_y - phi * s_x) / (n * (1.0 - phi))
    resid = x1 - (mu + phi * (x0 - mu))
    sigma_eps2 = float(resid @ resid / n)
    theta = -np.log(phi) / dt
    sigma_ou = np.sqrt(sigma_eps2 * 2.0 * theta / (1.0 - phi**2))
    half_life = np.log(2.0) / theta if theta > 0 else float("inf")
    return {
        "theta": float(theta),
        "mu": float(mu),
        "phi": float(phi),
        "sigma": float(sigma_ou),
        "half_life": float(half_life),
        "long_run_sd": float(sigma_ou / np.sqrt(2.0 * theta)) if theta > 0 else float("nan"),
    }


def ou_expected_excursion(x: Array, level: float) -> float:
    """Expected first-passage time to ``level`` under the fitted OU process.

    Ricciardi–Sato (1988) integral for the OU mean first-passage time to a
    level ``z`` measured in stationary-standard-deviation units from the mean:
    ``E[T] = sqrt(pi)/theta * integral_0^z exp(u^2) (1 + erf(u)) du``.
    """
    from scipy import integrate, special

    fit = ou_mle(x)
    theta = fit["theta"]
    mu = fit["mu"]
    sd = fit["long_run_sd"]
    if sd <= 0.0 or not np.isfinite(sd) or theta <= 0.0:
        return float("nan")
    z = (float(level) - mu) / sd
    if not np.isfinite(z):
        return float("nan")
    if z <= 0.0:
        # Below the mean the process hits it quickly (mass concentrates there).
        lo, hi = z, 0.0
    else:
        lo, hi = 0.0, z
    val, _ = integrate.quad(
        lambda u: float(np.exp(u * u) * (1.0 + special.erf(u))), lo, hi, limit=50
    )
    return float(np.sqrt(np.pi) / theta * val)

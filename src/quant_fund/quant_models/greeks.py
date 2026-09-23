"""First-, second- and third-order BSM greeks.

Formulas and desk scaling match davidalmeida90/quant-models
``advanced-greeks/model.py``. Raw derivatives live in ``greeks()``;
``SCALE`` converts to desk units (vega per vol point, theta per day).
Analytic identities are checked against finite differences of the
price surface in ``validate_greeks``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from quant_fund.quant_models.black_scholes import bs_price, d1, d2

# raw derivative -> desk quote (same table as the source notebook)
SCALE: dict[str, tuple[float, str]] = {
    "delta": (1.0, "per $1"),
    "gamma": (1.0, "delta per $1"),
    "vega": (0.01, "per 1 vol pt"),
    "theta": (1 / 365.0, "per day"),
    "rho": (0.01, "per 1 rate pt"),
    "vanna": (0.01, "delta per 1 vol pt"),
    "volga": (0.0001, "vega per 1 vol pt"),
    "charm": (1 / 365.0, "delta per day"),
    "veta": (0.01 / 365.0, "vega per day"),
    "speed": (1.0, "gamma per $1"),
    "zomma": (0.01, "gamma per 1 vol pt"),
    "color": (1 / 365.0, "gamma per day"),
    "ultima": (1e-6, "volga per 1 vol pt"),
}

GREEK_NAMES = tuple(SCALE)


def greeks(
    S: ArrayLike,
    K: ArrayLike,
    tau: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sig: ArrayLike,
    kind: str = "call",
) -> dict[str, np.ndarray]:
    """Raw BSM greeks (no desk scaling). Time derivatives are calendar ``d/dt``."""
    option = str(kind).lower()
    S, K, tau, r, q, sig = (np.asarray(z, dtype=float) for z in (S, K, tau, r, q, sig))
    a = d1(S, K, tau, r, q, sig)
    b = d2(S, K, tau, r, q, sig)
    rt = np.sqrt(tau)
    pa = norm.pdf(a)
    eq = np.exp(-q * tau)
    er = np.exp(-r * tau)
    vega = S * eq * pa * rt
    gamma = eq * pa / (S * sig * rt)
    if option == "call":
        delta = eq * norm.cdf(a)
        rho = K * tau * er * norm.cdf(b)
        theta = -S * pa * sig * eq / (2 * rt) + q * S * eq * norm.cdf(a) - r * K * er * norm.cdf(b)
        charm = q * eq * norm.cdf(a) - eq * pa * (2 * (r - q) * tau - b * sig * rt) / (
            2 * tau * sig * rt
        )
    elif option == "put":
        delta = eq * (norm.cdf(a) - 1.0)
        rho = -K * tau * er * norm.cdf(-b)
        theta = -S * pa * sig * eq / (2 * rt) - q * S * eq * norm.cdf(-a) + r * K * er * norm.cdf(-b)
        charm = -q * eq * norm.cdf(-a) - eq * pa * (2 * (r - q) * tau - b * sig * rt) / (
            2 * tau * sig * rt
        )
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    vanna = -eq * pa * b / sig
    volga = vega * a * b / sig
    veta = S * eq * pa * rt * (q + (r - q) * a / (sig * rt) - (1 + a * b) / (2 * tau))
    speed = -gamma / S * (a / (sig * rt) + 1.0)
    zomma = gamma * (a * b - 1.0) / sig
    color = (
        eq
        * pa
        / (2 * S * tau * sig * rt)
        * (2 * q * tau + 1.0 + (2 * (r - q) * tau - b * sig * rt) / (sig * rt) * a)
    )
    ultima = -vega / sig**2 * (a * b * (1 - a * b) + a * a + b * b)
    price = bs_price(S, K, tau, r, q, sig, option)
    return {
        "d1": np.asarray(a, dtype=float),
        "d2": np.asarray(b, dtype=float),
        "price": np.asarray(price, dtype=float),
        "delta": np.asarray(delta, dtype=float),
        "vega": np.asarray(vega, dtype=float),
        "theta": np.asarray(theta, dtype=float),
        "rho": np.asarray(rho, dtype=float),
        "gamma": np.asarray(gamma, dtype=float),
        "vanna": np.asarray(vanna, dtype=float),
        "volga": np.asarray(volga, dtype=float),
        "charm": np.asarray(charm, dtype=float),
        "veta": np.asarray(veta, dtype=float),
        "speed": np.asarray(speed, dtype=float),
        "zomma": np.asarray(zomma, dtype=float),
        "color": np.asarray(color, dtype=float),
        "ultima": np.asarray(ultima, dtype=float),
    }


def scaled_greeks(
    S: ArrayLike,
    K: ArrayLike,
    tau: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sig: ArrayLike,
    kind: str = "call",
    *,
    multiplier: float = 1.0,
) -> dict[str, float]:
    """Desk-unit greeks for a scalar contract."""
    raw = greeks(S, K, tau, r, q, sig, kind)
    out: dict[str, float] = {"price": float(np.asarray(raw["price"]).reshape(-1)[0])}
    for name, (scale, _unit) in SCALE.items():
        out[name] = float(np.asarray(raw[name]).reshape(-1)[0]) * scale * multiplier
    return out


def _fd(f, x: float, h: float, order: int = 1) -> float:
    if order == 1:
        return (f(x + h) - f(x - h)) / (2 * h)
    if order == 2:
        return (f(x + h) - 2 * f(x) + f(x - h)) / h**2
    return (f(x + 2 * h) - 2 * f(x + h) + 2 * f(x - h) - f(x - 2 * h)) / (2 * h**3)


def _numeric_greeks(
    S: float,
    K: float,
    tau: float,
    r: float,
    q: float,
    sig: float,
    kind: str,
) -> dict[str, float]:
    """Finite-difference greeks for one contract. Locals are bound, not loop vars."""

    def price(s: float = S, t: float = tau, v: float = sig) -> float:
        return float(bs_price(s, K, t, r, q, v, kind))

    def greek(name: str, s: float = S, t: float = tau, v: float = sig) -> float:
        return float(np.asarray(greeks(s, K, t, r, q, v, kind)[name]).reshape(-1)[0])

    h_s, h_v, h_t = S * 2e-4, 1e-4, 1e-5
    return {
        "delta": _fd(lambda s: price(s=s), S, h_s),
        "vega": _fd(lambda v: price(v=v), sig, h_v),
        "theta": -_fd(lambda t: price(t=t), tau, h_t),
        "gamma": _fd(lambda s: price(s=s), S, h_s, order=2),
        "vanna": _fd(lambda v: _fd(lambda s: price(s=s, v=v), S, h_s), sig, h_v),
        "volga": _fd(lambda v: price(v=v), sig, h_v, order=2),
        "charm": -_fd(lambda t: greek("delta", t=t), tau, h_t),
        "veta": -_fd(lambda t: greek("vega", t=t), tau, h_t),
        "speed": _fd(lambda s: greek("gamma", s=s), S, h_s),
        "zomma": _fd(lambda v: greek("gamma", v=v), sig, h_v),
        "color": -_fd(lambda t: greek("gamma", t=t), tau, h_t),
        "ultima": _fd(lambda v: greek("volga", v=v), sig, h_v),
    }


def validate_greeks(*, verbose: bool = False) -> float:
    """Worst relative error of analytic greeks vs finite differences of BSM."""
    cases = [
        dict(S=100.0, K=100.0, tau=0.25, r=0.042, q=0.005, sig=0.28, kind="call"),
        dict(S=100.0, K=110.0, tau=0.08, r=0.042, q=0.000, sig=0.45, kind="call"),
        dict(S=100.0, K=92.0, tau=1.00, r=0.042, q=0.020, sig=0.20, kind="put"),
        dict(S=180.0, K=175.0, tau=0.04, r=0.042, q=0.000, sig=0.55, kind="put"),
    ]
    worst = 0.0
    for c in cases:
        analytic = greeks(**c)  # type: ignore[arg-type]
        numeric = _numeric_greeks(**c)  # type: ignore[arg-type]
        for name, nv in numeric.items():
            av = float(np.asarray(analytic[name]).reshape(-1)[0])
            scale = max(abs(av), abs(nv), 1e-12)
            err = abs(av - nv) / scale
            worst = max(worst, err)
            if verbose:
                flag = "" if err < 2e-4 else " <-- CHECK"
                print(
                    f"{c['kind']} K{c['K']:g} T{c['tau']:g} {name:<8} "
                    f"{av:.6g} {nv:.6g} {err:.2e}{flag}"
                )
    return float(worst)


def taylor_attribution(
    S: float,
    K: float,
    tau: float,
    r: float,
    q: float,
    sig: float,
    kind: str,
    dS: float,
    dsig: float,
    days: float,
) -> dict[str, float]:
    """How much of a BSM reprice each order of the expansion explains."""
    g = {k: float(np.asarray(v).reshape(-1)[0]) for k, v in greeks(S, K, tau, r, q, sig, kind).items()}
    dt_yr = days / 365.0
    exact = float(bs_price(S + dS, K, tau - dt_yr, r, q, sig + dsig, kind)) - g["price"]
    t1 = g["delta"] * dS + g["vega"] * dsig + g["theta"] * dt_yr
    t2 = (
        0.5 * g["gamma"] * dS**2
        + 0.5 * g["volga"] * dsig**2
        + g["vanna"] * dS * dsig
        + g["charm"] * dS * dt_yr
        + g["veta"] * dsig * dt_yr
    )
    t3 = (
        (1 / 6) * g["speed"] * dS**3
        + 0.5 * g["zomma"] * dS**2 * dsig
        + 0.5 * g["color"] * dS**2 * dt_yr
        + (1 / 6) * g["ultima"] * dsig**3
    )
    return {
        "exact": exact,
        "o1": t1,
        "o2": t1 + t2,
        "o3": t1 + t2 + t3,
        "d_o1": t1,
        "d_o2": t2,
        "d_o3": t3,
        "err1": exact - t1,
        "err2": exact - (t1 + t2),
        "err3": exact - (t1 + t2 + t3),
    }


def strangle_volga(
    S: float,
    k_call: float,
    k_put: float,
    tau: float,
    r: float,
    q: float,
    sig_call: float,
    sig_put: float,
    *,
    n_call: float = 1.0,
    n_put: float = 1.0,
) -> dict[str, Any]:
    """Volga of a long strangle (quant-models volga-convexity-spread)."""
    c = greeks(S, k_call, tau, r, q, sig_call, "call")
    p = greeks(S, k_put, tau, r, q, sig_put, "put")
    volga = float(c["volga"]) * n_call + float(p["volga"]) * n_put
    vega = float(c["vega"]) * n_call + float(p["vega"]) * n_put
    return {
        "volga_raw": volga,
        "vega_raw": vega,
        "volga_per_vol_pt": volga * SCALE["volga"][0],
        "convex_in_vol": volga > 0,
    }

"""Proper scores for probabilities, PIT, and VaR hit tests.

Research-diagnostic metrics only — not live P&L claims.
``live_pnl_claim`` stays false; empty / all-NaN / bad inputs → honest NaN
or fail-closed ValueError (length mismatch).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.float64]


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


def _require_alpha_as(alpha: float) -> None:
    """Validate the tail probability used by Acerbi--Székely diagnostics."""
    if not np.isfinite(alpha) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")


def brier_score(prob: Array, y: Array) -> float:
    """Mean squared error of probabilities. Empty/all-NaN/all-invalid → NaN.

    Probabilities outside ``[0, 1]`` are treated as invalid observations
    (masked out honestly, like NaN) rather than clipped into range — a proper
    score requires inputs that are actually probabilities, and an out-of-range
    forecast must never be reshaped into a low-looking Brier. Research only.
    """
    p = _as_1d("prob", prob)
    t = _as_1d("y", y)
    _require_same_length(("prob", p), ("y", t))
    mask = np.isfinite(p) & np.isfinite(t) & (p >= 0.0) & (p <= 1.0)
    if int(mask.sum()) < 1:
        return float("nan")
    return float(np.mean((p[mask] - t[mask]) ** 2))


def log_loss(prob: Array, y: Array, eps: float = 1e-12) -> float:
    """Binary cross-entropy. Empty/all-NaN/all-invalid → NaN.

    Out-of-range probabilities are masked out (honest invalid) instead of
    clipped into ``(0, 1)``; the ``eps`` interior clip still guards log(0) for
    valid endpoints. Research only.
    """
    p = _as_1d("prob", prob)
    t = _as_1d("y", y)
    _require_same_length(("prob", p), ("y", t))
    mask = np.isfinite(p) & np.isfinite(t) & (p >= 0.0) & (p <= 1.0)
    if int(mask.sum()) < 1:
        return float("nan")
    pv = np.clip(p[mask], eps, 1.0 - eps)
    yy = np.clip(t[mask], 0.0, 1.0)
    return float(-np.mean(yy * np.log(pv) + (1.0 - yy) * np.log(1.0 - pv)))


def expected_calibration_error(prob: Array, y: Array, n_bins: int = 10) -> float:
    """Binned ECE. Empty/short/all-NaN or n_bins<1 → NaN (research only)."""
    if n_bins < 1 or not np.isfinite(n_bins):
        return float("nan")
    p = _as_1d("prob", prob)
    t = _as_1d("y", y)
    _require_same_length(("prob", p), ("y", t))
    mask = np.isfinite(p) & np.isfinite(t) & (p >= 0.0) & (p <= 1.0)
    p, t = p[mask], t[mask]
    if p.size < n_bins:
        return float("nan")
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for i in range(int(n_bins)):
        sel = (p >= edges[i]) & (p < edges[i + 1] if i < int(n_bins) - 1 else p <= edges[i + 1])
        if not sel.any():
            continue
        ece += float(sel.mean()) * abs(float(t[sel].mean()) - float(p[sel].mean()))
    return float(ece)


def kupiec_pof(hits: Array, alpha: float) -> tuple[float, float, float]:
    """Kupiec proportion-of-failures test. Returns (hit_rate, lr, p_value).

    Empty/short (n<10) / alpha∉(0,1) → (NaN, NaN, NaN). Research-diagnostic only.
    """
    h = _as_1d("hits", hits)
    h = h[np.isfinite(h)]
    n = int(h.size)
    x = int(np.sum(h > 0.5))
    if n < 10 or not (np.isfinite(alpha) and 0 < alpha < 1):
        return float("nan"), float("nan"), float("nan")
    rate = x / n
    if x == 0 or x == n:
        return rate, float("nan"), float("nan")
    lr = -2.0 * (
        (n - x) * np.log(1.0 - alpha)
        + x * np.log(alpha)
        - (n - x) * np.log(1.0 - rate)
        - x * np.log(rate)
    )
    p = float(stats.chi2.sf(lr, df=1))
    return float(rate), float(lr), p


def pit_ks(pits: Array) -> tuple[float, float]:
    """Kolmogorov–Smirnov test of PIT against Uniform(0, 1).

    Empty/short (n<8) / all-NaN → (NaN, NaN). Research-diagnostic only.
    """
    u = _as_1d("pits", pits)
    u = u[np.isfinite(u)]
    u = np.clip(u, 1e-9, 1.0 - 1e-9)
    if u.size < 8:
        return float("nan"), float("nan")
    stat, p = stats.kstest(u, "uniform")
    return float(stat), float(p)


def christoffersen_independence(hits: Array) -> tuple[float, float, dict[str, float]]:
    """Christoffersen (1998) independence test of VaR hit clustering.

    Returns (lr_ind, p_value, transition_counts). Null: hits are i.i.d. Bernoulli.
    Empty/short (n<12) → NaN. Research-diagnostic only — not a live P&L claim.
    """
    h = _as_1d("hits", hits)
    h = h[np.isfinite(h)]
    h = (h > 0.5).astype(int)
    n = int(h.size)
    if n < 12:
        return float("nan"), float("nan"), {}
    # Transition counts n_ij: i→j
    n00 = n01 = n10 = n11 = 0
    for i in range(n - 1):
        a, b = int(h[i]), int(h[i + 1])
        if a == 0 and b == 0:
            n00 += 1
        elif a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
        else:
            n11 += 1
    counts = {"n00": float(n00), "n01": float(n01), "n10": float(n10), "n11": float(n11)}
    # Unconditional hit rate
    n0 = n00 + n01
    n1 = n10 + n11
    if n0 + n1 == 0:
        return float("nan"), float("nan"), counts
    pi_hat = (n01 + n11) / (n0 + n1)
    # Conditional probs
    pi01 = n01 / n0 if n0 > 0 else 0.0
    pi11 = n11 / n1 if n1 > 0 else 0.0

    # Likelihoods (avoid log0)
    def _ll_ind() -> float:
        # under independence: pi01 = pi11 = pi_hat
        if pi_hat <= 0.0 or pi_hat >= 1.0:
            return float("nan")
        return float(
            n00 * np.log(1.0 - pi_hat)
            + n01 * np.log(pi_hat)
            + n10 * np.log(1.0 - pi_hat)
            + n11 * np.log(pi_hat)
        )

    def _ll_dep() -> float:
        ll = 0.0
        if n0 > 0:
            if n00:
                if pi01 >= 1.0:
                    return float("nan")
                ll += n00 * np.log(1.0 - pi01)
            if n01:
                if pi01 <= 0.0:
                    return float("nan")
                ll += n01 * np.log(pi01)
        if n1 > 0:
            if n10:
                if pi11 >= 1.0:
                    return float("nan")
                ll += n10 * np.log(1.0 - pi11)
            if n11:
                if pi11 <= 0.0:
                    return float("nan")
                ll += n11 * np.log(pi11)
        return ll

    ll0 = _ll_ind()
    ll1 = _ll_dep()
    if not np.isfinite(ll0) or not np.isfinite(ll1):
        return float("nan"), float("nan"), counts
    lr = float(-2.0 * (ll0 - ll1))
    p = float(stats.chi2.sf(max(lr, 0.0), df=1))
    return lr, p, counts


def christoffersen_cc(hits: Array, alpha: float) -> tuple[float, float, dict[str, float]]:
    """Christoffersen conditional coverage = Kupiec POF + independence.

    ``alpha`` here is the *expected hit rate* (e.g. 0.05 for 95% VaR).
    Returns (lr_cc, p_value, extras). Empty/bad → NaN. Research-diagnostic only.
    """
    rate, lr_uc, p_uc = kupiec_pof(hits, alpha)
    lr_ind, p_ind, counts = christoffersen_independence(hits)
    extras = {
        "hit_rate": float(rate),
        "kupiec_lr": float(lr_uc),
        "kupiec_p": float(p_uc),
        "ind_lr": float(lr_ind),
        "ind_p": float(p_ind),
        **counts,
    }
    if not np.isfinite(lr_uc) or not np.isfinite(lr_ind):
        return float("nan"), float("nan"), extras
    lr_cc = float(lr_uc + lr_ind)
    p = float(stats.chi2.sf(max(lr_cc, 0.0), df=2))
    return lr_cc, p, extras


def acerbi_szekely_z1(
    losses: Array,
    var: Array,
    es: Array,
    alpha: float,
) -> tuple[float, int]:
    """Acerbi–Székely (2014) Z1 ES backtest statistic (unconditional).

    Loss convention: positive losses. Hit when ``loss > VaR``.
    ``Z1 = (1/(N α)) Σ 1_{L>VaR} (L/ES) − 1``. Under a correct forecast, E[Z1]=0.
    Research-diagnostic only — not a live capital claim.

    Returns ``(Z1, n_hits)``. Empty / length mismatch / bad alpha / non-positive ES
    on hits → honest ``(nan, 0)`` or fail-closed ValueError for alpha.
    """
    _require_alpha_as(alpha)
    L = _as_1d("losses", losses)
    V = _as_1d("var", var)
    E = _as_1d("es", es)
    _require_same_length(("losses", L), ("var", V), ("es", E))
    n = int(L.size)
    if n == 0:
        return float("nan"), 0
    hits = L > V
    n_hits = int(np.sum(hits))
    if n_hits == 0:
        return float("nan"), 0
    es_h = E[hits]
    L_h = L[hits]
    if not np.isfinite(es_h).all() or not np.isfinite(L_h).all() or np.any(es_h <= 0):
        return float("nan"), n_hits
    z1 = float(np.sum(L_h / es_h) / (n * float(alpha)) - 1.0)
    return z1, n_hits


def acerbi_szekely_z2(
    losses: Array,
    var: Array,
    es: Array,
) -> tuple[float, int]:
    """Acerbi–Székely Z2: mean excess ratio on VaR breaches − 1.

    ``Z2 = mean_{t: L_t>VaR_t}(L_t/ES_t) − 1``. Research-diagnostic only.
    Empty / no hits / non-positive ES on hits → honest ``(nan, n_hits)``.
    """
    L = _as_1d("losses", losses)
    V = _as_1d("var", var)
    E = _as_1d("es", es)
    _require_same_length(("losses", L), ("var", V), ("es", E))
    if L.size == 0:
        return float("nan"), 0
    hits = L > V
    n_hits = int(np.sum(hits))
    if n_hits == 0:
        return float("nan"), 0
    es_h = E[hits]
    L_h = L[hits]
    if not np.isfinite(es_h).all() or not np.isfinite(L_h).all() or np.any(es_h <= 0):
        return float("nan"), n_hits
    return float(np.mean(L_h / es_h) - 1.0), n_hits

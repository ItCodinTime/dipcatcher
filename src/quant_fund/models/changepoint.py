"""Changepoint detection: Bayesian online detection, CUSUM, segmentation.

References:
- Adams, MacKay (2007). Bayesian online changepoint detection.
  arXiv:0710.3742.
- Page (1954). Continuous inspection schemes (CUSUM). *Biometrika* 41.
- Page (1955) / Hinkley (1971) — Page-Hinkley cumulative-sum test.
- Scott, Knott (1974). A cluster analysis method for grouping means —
  binary segmentation.
- Killick, Fearnhead, Eckley (2012). Optimal detection of changepoints with
  a linear computational cost (PELT). *JASA* 107.  The pruning is optional;
  the plain optimal-partitioning core is implemented here.
- Jackson et al. (2005). An algorithm for optimal partitioning of data on an
  interval (OP).  *IEEE Signal Process. Lett.* 12.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

Array = NDArray[np.float64]
IdxArray = NDArray[np.intp]


def _as_vector(x: Array, name: str = "x", *, min_obs: int = 8) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size < min_obs:
        raise ValueError(f"{name} must contain at least {min_obs} finite observations")
    return v


def bocpd_gaussian(
    x: Array,
    hazard: float = 1.0 / 250.0,
    mu0: float | None = None,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float | None = None,
    *,
    prune: float = 1e-8,
) -> dict[str, Array]:
    """Adams–MacKay (2007) Bayesian online changepoint detection.

    Conjugate prior Normal-Gamma on (mean, precision); the per-run-length
    predictive is Student-t.  Constant hazard ``hazard`` = 1/E[run length].

    Returns:
    - ``cp_prob[t]``: posterior probability a changepoint occurred at t
      (run length reset to 0).
    - ``expected_run_length[t]``: posterior mean run length.
    - ``changepoints``: indices where ``cp_prob`` exceeds 0.5.
    - ``run_length_map[t, r]``: full posterior (rows sum to 1).
    """
    v = _as_vector(x)
    if not np.isfinite(hazard) or not (0.0 < hazard < 1.0):
        raise ValueError("hazard must be in (0, 1)")
    if kappa0 <= 0.0 or alpha0 <= 0.0:
        raise ValueError("kappa0 and alpha0 must be positive")
    m0 = float(v.mean()) if mu0 is None else float(mu0)
    b0 = float(v.var(ddof=1)) * kappa0 if beta0 is None else float(beta0)
    if not np.isfinite(m0) or not np.isfinite(b0) or b0 <= 0.0:
        raise ValueError("prior hyperparameters must be finite (beta0 > 0)")
    n = v.size
    run_probs = np.zeros((n + 1, n + 1))
    run_probs[0, 0] = 1.0
    # Normal-Gamma hyperparams indexed by run length (index r = posterior from
    # the r most recent observations; index 0 is always the prior).
    mu = np.full(n + 1, m0)
    kappa = np.full(n + 1, kappa0)
    alpha = np.full(n + 1, alpha0)
    beta = np.full(n + 1, b0)
    cp_prob = np.zeros(n + 1)
    expected_rl = np.zeros(n + 1)
    rls_map = np.zeros((n + 1, n + 1))
    for t in range(1, n + 1):
        obs = v[t - 1]
        # Student-t predictive for each live run length.
        df = 2.0 * alpha[:t]
        scale = np.sqrt(beta[:t] * (kappa[:t] + 1.0) / (alpha[:t] * kappa[:t]))
        scale = np.maximum(scale, 1e-12)
        z = (obs - mu[:t]) / scale
        pred_log = (
            gammaln((df + 1.0) / 2.0)
            - gammaln(df / 2.0)
            - 0.5 * np.log(df * np.pi)
            - np.log(scale)
            - ((df + 1.0) / 2.0) * np.log1p(z * z / df)
        )
        pred = np.exp(np.clip(pred_log, -700.0, 700.0))
        growth = run_probs[t - 1, :t] * pred * (1.0 - hazard)
        # Changepoint mass: a new run predicts under the PRIOR predictive,
        # marginalized over the run-length posterior (Adams–MacKay eq. 9).
        df0 = 2.0 * alpha0
        scale0 = math.sqrt(b0 * (kappa0 + 1.0) / (alpha0 * kappa0))
        z0 = (obs - m0) / scale0
        prior_pred = float(
            np.exp(
                np.clip(
                    gammaln((df0 + 1.0) / 2.0)
                    - gammaln(df0 / 2.0)
                    - 0.5 * math.log(df0 * math.pi)
                    - math.log(scale0)
                    - ((df0 + 1.0) / 2.0) * math.log1p(z0 * z0 / df0),
                    -700.0,
                    700.0,
                )
            )
        )
        cp_mass = hazard * prior_pred * float(run_probs[t - 1, :t].sum())
        new_probs = np.zeros(t + 1)
        new_probs[0] = cp_mass
        new_probs[1 : t + 1] = growth
        total = float(new_probs.sum())
        if total <= 0.0 or not np.isfinite(total):
            new_probs = np.zeros(t + 1)
            new_probs[0] = 1.0
            total = 1.0
        new_probs /= total
        if prune > 0.0:
            new_probs[new_probs < prune] = 0.0
            s = new_probs.sum()
            if s > 0.0:
                new_probs /= s
        run_probs[t, : t + 1] = new_probs
        rls_map[t, : t + 1] = new_probs
        cp_prob[t] = float(new_probs[0])
        expected_rl[t] = float(np.dot(np.arange(t + 1), new_probs))
        # Normal-Gamma updates, shifted by one: a run of length r that survives
        # becomes length r+1, so index r+1 receives the updated params of r and
        # index 0 keeps the prior (a new run has seen only ``obs`` so far).
        k_new = kappa[:t] + 1.0
        mu_new = (kappa[:t] * mu[:t] + obs) / k_new
        beta_new = beta[:t] + kappa[:t] * (obs - mu[:t]) ** 2 / k_new
        alpha_new = alpha[:t] + 0.5
        mu[1 : t + 1] = mu_new
        kappa[1 : t + 1] = k_new
        alpha[1 : t + 1] = alpha_new
        beta[1 : t + 1] = beta_new
        mu[0], kappa[0], alpha[0], beta[0] = m0, kappa0, alpha0, b0
    return {
        "cp_prob": cp_prob[1:],
        "expected_run_length": expected_rl[1:],
        "changepoints": np.flatnonzero(cp_prob[1:] > 0.5).astype(np.intp),
        "run_length_map": rls_map[1:, :],
    }


def cusum_detect(
    x: Array, drift: float | None = None, threshold: float | None = None
) -> dict[str, Array]:
    """Page (1954) two-sided CUSUM detector for mean shifts.

    ``drift`` is the allowance k (defaults to ``0.5 * std`` — tuned to detect
    shifts of one std); ``threshold`` defaults to ``5 * std(x)``.  Returns
    alarm indices and the cumulative positive/negative statistics paths.
    """
    v = _as_vector(x)
    sd = float(v.std(ddof=1))
    d = 0.5 * sd if drift is None else float(drift)
    if not np.isfinite(d):
        raise ValueError("drift must be finite")
    h = 5.0 * sd if threshold is None else float(threshold)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("threshold must be positive and finite")
    ref = float(v.mean())
    sp = np.zeros(v.size)
    sn = np.zeros(v.size)
    alarms: list[int] = []
    for t in range(1, v.size):
        sp[t] = max(0.0, sp[t - 1] + (v[t] - ref - d))
        sn[t] = max(0.0, sn[t - 1] - (v[t] - ref + d))
        if sp[t] > h or sn[t] > h:
            alarms.append(t)
            sp[t] = 0.0
            sn[t] = 0.0
    return {
        "alarms": np.asarray(alarms, dtype=np.intp),
        "s_pos": sp,
        "s_neg": sn,
        "threshold": np.array([h]),
    }


def page_hinkley(x: Array, delta: float = 0.0, lam: float | None = None) -> dict[str, Array]:
    """Page–Hinkley test: cumulative sum of ``(x_t - mean - delta)`` vs ``lam``.

    ``lam`` defaults to ``0.5 * std``.  Returns alarm times and the running
    ``U_t = m_t - min m_t`` decision statistic.
    """
    v = _as_vector(x)
    if not np.isfinite(delta):
        raise ValueError("delta must be finite")
    sd = float(v.std(ddof=1))
    lam_v = 0.5 * sd if lam is None else float(lam)
    if not np.isfinite(lam_v) or lam_v <= 0.0:
        raise ValueError("lam must be positive and finite")
    m = 0.0
    m_min = 0.0
    u = np.zeros(v.size)
    alarms: list[int] = []
    mean = 0.0
    for t in range(v.size):
        mean += (v[t] - mean) / (t + 1.0)
        m += v[t] - mean - delta
        m_min = min(m_min, m)
        u[t] = m - m_min
        if u[t] > lam_v:
            alarms.append(t)
            m = 0.0
            m_min = 0.0
    return {
        "alarms": np.asarray(alarms, dtype=np.intp),
        "u": u,
        "lambda": np.array([lam_v]),
    }


def _seg_cost(v: Array, a: int, b: int) -> float:
    """Within-segment cost = sum of squared deviations (Gaussian mean-shift)."""
    seg = v[a:b]
    if seg.size <= 0:
        return 0.0
    d = seg - seg.mean()
    return float(d @ d)


def binary_segmentation(
    x: Array, min_size: int = 10, penalty: float | None = None, max_cps: int = 10
) -> IdxArray:
    """Scott–Knott (1974) binary segmentation on mean shifts.

    Recursively splits at the argmax CUSUM split point while the split reduces
    total cost by more than ``penalty`` (default ``2 * log(n) * var`` — a BIC
    analogue).  Returns sorted changepoint indices.
    """
    v = _as_vector(x)
    if isinstance(min_size, bool) or not isinstance(min_size, int) or min_size < 2:
        raise ValueError("min_size must be an integer >= 2")
    var = float(v.var(ddof=1))
    pen = 2.0 * math.log(v.size) * max(var, 1e-12) * 2 if penalty is None else float(penalty)
    if not np.isfinite(pen) or pen < 0.0:
        raise ValueError("penalty must be non-negative and finite")
    cps: list[int] = []

    def _split(a: int, b: int) -> None:
        if len(cps) >= max_cps or b - a < 2 * min_size:
            return
        best_gain, best_t = 0.0, -1
        base = _seg_cost(v, a, b)
        for t in range(a + min_size, b - min_size + 1):
            gain = base - _seg_cost(v, a, t) - _seg_cost(v, t, b)
            if gain > best_gain:
                best_gain, best_t = gain, t
        if best_t > 0 and best_gain > pen:
            cps.append(best_t)
            _split(a, best_t)
            _split(best_t, b)

    _split(0, v.size)
    return np.asarray(sorted(cps), dtype=np.intp)


def optimal_partition_mean(
    x: Array, penalty: float | None = None, min_size: int = 2
) -> tuple[IdxArray, float]:
    """Jackson et al. (2005) optimal partitioning (PELT without pruning).

    Exact minimizer of ``sum_seg cost + penalty * n_changepoints`` for the
    Gaussian mean-shift cost, O(n^2).  ``penalty`` defaults to BIC
    ``2 * log(n) * var``.  Returns (changepoints, objective).
    """
    v = _as_vector(x)
    var = float(v.var(ddof=1))
    pen = 2.0 * math.log(v.size) * max(var, 1e-12) if penalty is None else float(penalty)
    if not np.isfinite(pen) or pen < 0.0:
        raise ValueError("penalty must be non-negative and finite")
    n = v.size
    cumsum = np.concatenate([[0.0], np.cumsum(v)])
    cumsum2 = np.concatenate([[0.0], np.cumsum(v**2)])

    def _cost(a: int, b: int) -> float:
        # sum of squares within [a, b)
        s1 = cumsum[b] - cumsum[a]
        s2 = cumsum2[b] - cumsum2[a]
        m = b - a
        return s2 - s1 * s1 / m

    f = np.full(n + 1, np.inf)
    f[0] = -pen
    last = np.zeros(n + 1, dtype=np.intp)
    for b in range(min_size, n + 1):
        for a in range(0, b - min_size + 1):
            cand = f[a] + _cost(a, b) + pen
            if cand < f[b]:
                f[b] = cand
                last[b] = a
    cps: list[int] = []
    b = n
    while last[b] > 0:
        cps.append(int(last[b]))
        b = int(last[b])
    return np.asarray(sorted(cps), dtype=np.intp), float(f[n])

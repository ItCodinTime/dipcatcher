"""Anytime-valid sequential changepoint detection via e-detectors.

Implements the e-detector framework of Shin, Ramdas & Rinaldo (2023),
"E-detectors: a nonparametric framework for sequential change detection",
Annals of Statistics 51(4):1233-1268 (arXiv:2203.03532). An e-detector is a
nonnegative process (M_n) with E[M_tau] <= E[tau] under any pre-change
distribution and any stopping time tau (their Def. 2.1); thresholding at
1/alpha yields ARL >= 1/alpha (their Thm. 2.4, Eq. 15). The constructions
below strengthen this to a genuine time-uniform type-I guarantee
P(exists n: M_n >= 1/alpha) <= alpha, by mixing the baseline Shiryaev-Roberts
recursion M_n = L_n * (M_{n-1} + 1) (their Eq. 13) with a Geometric(1-gamma)
prior over the changepoint time (the "mixing e-detectors" of their Sec. 2.4;
the geometric-prior mixture is the Bayes/Shiryaev weighting, cf. Shiryaev
1963, Theory Probab. Appl. 8(1):22-46). Concretely, per mixture component k
with one-step e-value e_{k,n} (E[e_{k,n} | F_{n-1}] <= 1 under the null):

    A_{k,n} = e_{k,n} * A_{k,n-1} + (1 - gamma) * gamma^{n-1},   A_{k,0} = 0
    M_n     = sum_k w_k * A_{k,n} + gamma^n,                     sum_k w_k = 1

The trailing gamma^n term is the unspent prior mass on "no changepoint yet";
it keeps (M_n) a true nonnegative supermartingale with M_0 = 1, so Ville's
inequality (Ville 1939, "Etude critique de la notion de collectif",
Gauthier-Villars) delivers the time-uniform bound exactly. Mixing over a grid
of one-step e-values (the "e-detector with predictable plugins" / mixture
likelihood-ratio construction of Sec. 2.4 and Example 1) lets one detector
cover a composite post-change class: the product process inside each A_{k,n}
grows at the best-matched component's rate, unlike a per-step mixture whose
log-growth is diluted toward the average component.

One-step e-values (all satisfy E[e | F_{n-1}] <= 1 under the stated null):
- Gaussian, known variance sigma^2, H0: mean <= 0: sub-Gaussian betting
  e_k = exp(lambda_k * x - lambda_k^2 * sigma^2 / 2), lambda_k on a fixed
  grid (Howard, Ramdas, McAuliffe & Sekhon 2021, Ann. Statist. 49(2):
  1055-1080, arXiv:1810.08240, Sec. 3 one-sided sub-Gaussian boundary).
- Bounded x in [lower, upper], H0: mean <= m0: Hoeffding-style
  e_k = exp(lambda_k * (y - m) - lambda_k^2 / 8) with y = (x - lower) /
  (upper - lower) in [0, 1], m = (m0 - lower) / (upper - lower)
  (Hoeffding 1963, JASA 58(301):13-30, Thm. 2; Shin et al. 2023 Sec. 2.6
  Example 7, change in mean of bounded variables).
- Bernoulli hit rate, H0: violation rate <= p0:
  e_k = (x / p0) * lambda_k + ((1 - x) / (1 - p0)) * (1 - lambda_k),
  lambda_k in (0, 1) (Ramdas / Grunwald / Shafer-Vovk style two-point
  e-value, cf. quant_fund.metrics.evalues; expectation <= 1 on [0, p0]).

All statistics are anytime-valid evidence processes; no Sharpe/P&L content.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

__all__ = [
    "DetectionResult",
    "EDetectorGaussian",
    "EDetectorBounded",
    "EDetectorBernoulli",
    "alarm_threshold",
    "run_detector",
]

_E_MAX = 1e300
_EPS = 1e-12


def alarm_threshold(alpha: float) -> float:
    """E-detector alarm level 1/alpha (Shin et al. 2023, Def. 2.12, Eq. 15)."""
    a = float(alpha)
    if not np.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return 1.0 / a


@dataclass(frozen=True)
class DetectionResult:
    """Output of run_detector: first alarm index (0-based) and detector path."""

    alarm_time: int | None
    detector_path: Array


class _EDetector(Protocol):
    """Structural type for run_detector (structural subtyping over classes)."""

    @property
    def alarmed(self) -> bool: ...

    def update(self, x: float) -> float: ...

    def reset(self) -> None: ...


class _MixtureSRDetector:
    """Geometric-prior mixture Shiryaev-Roberts e-detector (see module doc).

    Maintains K parallel mixture components A_{k,n} in log space; update(x)
    returns the current detector value M_n and advances the recursion.
    """

    def __init__(self, alpha: float, gamma: float) -> None:
        a = float(alpha)
        if not np.isfinite(a) or not 0.0 < a < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        g = float(gamma)
        if not np.isfinite(g) or not 0.0 < g < 1.0:
            raise ValueError("gamma must be in (0, 1)")
        self.alpha = a
        self.gamma = g
        self._log_gamma = float(np.log(g))
        self._log_1m_gamma = float(np.log1p(-g))
        self._log_w: Array = np.array([], dtype=float)
        self._n_components: int = 0
        self.reset()

    def _set_grid(self, n_components: int) -> None:
        k = int(n_components)
        if k < 1:
            raise ValueError("lambda grid must be non-empty")
        self._n_components = k
        self._log_w = np.full(k, -np.log(k), dtype=float)
        self.reset()

    def reset(self) -> None:
        """Restore the detector to its initial state (M_0 = 1, no data seen)."""
        self._log_active = np.full(self._n_components, -np.inf, dtype=float)
        self._c = 0.0  # (n - 1) * log(gamma) before the n-th update
        self._log_m = 0.0  # log M_0 = log 1
        self._n = 0

    def _log_step(self, x: float) -> Array:
        raise NotImplementedError

    def update(self, x: float) -> float:
        """Incorporate one observation; return the detector value M_n.

        Raises ValueError on non-finite input (fail-closed); subclasses may
        add further range checks.
        """
        v = float(x)
        if not np.isfinite(v):
            raise ValueError("x must be finite")
        log_e = self._log_step(v)
        inject = self._log_1m_gamma + self._c
        self._log_active = np.logaddexp(log_e + self._log_active, inject)
        self._c += self._log_gamma
        terms = self._log_w + self._log_active
        finite = np.isfinite(terms)
        if bool(np.any(finite)):
            t = terms[finite]
            mx = float(np.max(t))
            log_mix = mx + float(np.log(np.sum(np.exp(t - mx))))
        else:
            log_mix = -np.inf
        self._log_m = min(float(np.logaddexp(log_mix, self._c)), float(np.log(_E_MAX)))
        self._n += 1
        return float(np.exp(self._log_m))

    @property
    def value(self) -> float:
        """Current detector value M_n (1.0 before any update)."""
        return float(np.exp(self._log_m))

    @property
    def alarmed(self) -> bool:
        """True once M_n >= 1/alpha (time-uniform type-I error <= alpha)."""
        return self._log_m >= float(np.log(alarm_threshold(self.alpha)))

    @property
    def n_seen(self) -> int:
        """Number of observations incorporated since the last reset."""
        return self._n


def _as_grid(grid: Iterable[float] | Array, name: str) -> Array:
    g = np.asarray(list(grid) if not isinstance(grid, np.ndarray) else grid, dtype=float)
    g = np.atleast_1d(g).astype(float, copy=False).ravel()
    if g.size == 0 or not bool(np.all(np.isfinite(g))):
        raise ValueError(f"{name} must be a non-empty grid of finite values")
    return g


class EDetectorGaussian(_MixtureSRDetector):
    """Known-variance Gaussian mean-shift detector, H0: mean <= 0.

    One-step e-values e_k = exp(lambda_k * x - lambda_k^2 * sigma^2 / 2)
    (sub-Gaussian boundary, Howard et al. 2021, Sec. 3). The default grid
    lambda_k = linspace(0.25, 4.0, 16) / sigma targets post-change means in
    roughly [0.25 sigma, 4 sigma]; pass lambda_grid (absolute lambdas, not
    divided by sigma) to target other shifts, e.g. negative lambdas for
    downward shifts. Mixing follows the module-doc construction.
    """

    def __init__(
        self,
        sigma: float,
        alpha: float = 0.05,
        lambda_grid: Iterable[float] | Array | None = None,
        *,
        gamma: float = 0.999,
    ) -> None:
        s = float(sigma)
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError("sigma must be finite and > 0")
        super().__init__(alpha, gamma)
        self.sigma = s
        if lambda_grid is None:
            lam = np.linspace(0.25, 4.0, 16) / s
        else:
            lam = _as_grid(lambda_grid, "lambda_grid")
        self._lam = lam
        self._set_grid(int(lam.size))

    def _log_step(self, x: float) -> Array:
        return self._lam * x - 0.5 * (self._lam * self.sigma) ** 2


class EDetectorBounded(_MixtureSRDetector):
    """Bounded-stream mean-shift detector for x in [lower, upper].

    H0: mean <= m0 (default m0 = midpoint). Hoeffding-style one-step
    e-values e_k = exp(lambda_k * (y - m) - lambda_k^2 / 8) on the rescaled
    observation y = (x - lower) / (upper - lower) in [0, 1]
    (Hoeffding 1963, Thm. 2; Shin et al. 2023, Sec. 2.6 Example 7). The
    lambda grid {0.5, 1.0, 1.5, 2.0} covers shifts delta with 4*delta in that
    range (the per-component optimal plugin is lambda* = 4*delta on the
    rescaled scale). Out-of-range observations raise ValueError (fail-closed):
    the Hoeffding exponent is only valid on the declared support.
    """

    _LAMBDA_GRID: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)

    def __init__(
        self,
        lower: float,
        upper: float,
        alpha: float = 0.05,
        *,
        m0: float | None = None,
        gamma: float = 0.999,
    ) -> None:
        lo = float(lower)
        hi = float(upper)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError("lower/upper must be finite with lower < upper")
        super().__init__(alpha, gamma)
        if m0 is None:
            mid = 0.5 * (lo + hi)
        else:
            mid = float(m0)
            if not np.isfinite(mid) or not lo <= mid <= hi:
                raise ValueError("m0 must be finite and within [lower, upper]")
        self.lower = lo
        self.upper = hi
        self.m0 = mid
        self._scale = hi - lo
        self._lam = np.asarray(self._LAMBDA_GRID, dtype=float)
        self._m_scaled = (mid - lo) / self._scale
        self._set_grid(int(self._lam.size))

    def _log_step(self, x: float) -> Array:
        if not self.lower <= x <= self.upper:
            raise ValueError("x outside declared [lower, upper] support")
        y = (x - self.lower) / self._scale
        return self._lam * (y - self._m_scaled) - (self._lam**2) / 8.0


class EDetectorBernoulli(_MixtureSRDetector):
    """Hit-rate detector, H0: violation rate <= p0 (e.g. VaR misses at p0).

    One-step e-values e_k = (x / p0) * lambda_k + ((1 - x) / (1 - p0)) *
    (1 - lambda_k), lambda_k in (0, 1); E_p[e_k] <= 1 for all p <= p0 with
    equality at p = p0 (two-point e-value, Ramdas / Grunwald / Shafer-Vovk;
    same construction as quant_fund.metrics.evalues.e_value_bernoulli).
    The default grid lambda_k = clip(linspace(1.5, 3.0, 4) * p0, eps, 1 - eps)
    targets post-change rates in [1.5 p0, 3 p0], e.g. a 95% VaR miss rate
    drifting from p0 = 0.05 toward 0.075-0.15. Inputs are clipped to [0, 1]
    (soft/fractional indicators allowed; cf. evalues.py); non-finite input
    still raises via the shared update() check.
    """

    def __init__(self, p0: float, alpha: float = 0.05, *, gamma: float = 0.999) -> None:
        p = float(p0)
        if not np.isfinite(p) or not 0.0 < p < 1.0:
            raise ValueError("p0 must be in (0, 1)")
        super().__init__(alpha, gamma)
        self.p0 = p
        self._lam = np.clip(np.linspace(1.5, 3.0, 4) * p, _EPS, 1.0 - _EPS)
        self._set_grid(int(self._lam.size))

    def _log_step(self, x: float) -> Array:
        v = float(np.clip(x, 0.0, 1.0))
        e = (v / self.p0) * self._lam + ((1.0 - v) / (1.0 - self.p0)) * (1.0 - self._lam)
        return np.log(e)


def run_detector(detector: _EDetector, stream: Iterable[float], alpha: float) -> DetectionResult:
    """Feed stream into detector; return first alarm index and detector path.

    The detector is reset() first so the result depends only on
    (detector class/config, stream, alpha). Iteration stops at the first
    alarm (the detector stays alarmed thereafter, so later values carry no
    new information); alarm_time is the 0-based index of the crossing
    observation, or None if the stream ends unalarmed.
    """
    thr = alarm_threshold(alpha)
    detector.reset()
    path: list[float] = []
    alarm_time: int | None = None
    for i, x in enumerate(stream):
        path.append(detector.update(float(x)))
        if path[-1] >= thr:
            alarm_time = i
            break
    return DetectionResult(alarm_time=alarm_time, detector_path=np.asarray(path, dtype=float))

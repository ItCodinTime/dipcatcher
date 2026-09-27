"""Anytime-valid false discovery rate (FDR) control on e-values.

1. e-BH (Wang & Ramdas, 2022, JRSS-B 84(5):1667-1684, arXiv:2009.02824):
   Benjamini-Hochberg run on inverted e-values. FDR <= alpha is guaranteed
   under ARBITRARY dependence between the e-values (Wang & Ramdas 2022,
   Theorem 1) — no PRDS or independence condition is needed.
2. Stopped e-BH (se-BH; Wang, Dandapanthula & Ramdas, 2025, arXiv:2502.08539,
   Statistics & Probability Letters): running plain e-BH on stopped local
   e-processes can inflate the stopped FDR by up to ~log G because a global
   stopping time need not be a local one (their Section 5.5, Eqs. 42-49).
   The paper's correction (their Section 4, Eq. 26 / Corollary 4.3, building
   on the e-lifting of Choe & Ramdas, 2024, Theorem 2, and the adjusters of
   Shafer, Shen, Vereshchagin & Vovk, 2011) lifts each local e-process
   through a compound adjuster applied to its running maximum, which turns it
   into a global compound e-process; e-BH on the lifted stopped values then
   controls the stopped FDR at ANY global stopping time (their Theorem 2.5).
3. e-LOND (Xu & Ramdas, 2024, AISTATS, pp. 3997-4005, arXiv:2311.06412,
   Section 3.2): online FDR control on a stream of e-values, FDR <= alpha at
   every time t under arbitrary dependence (their Theorem 1). Test level
   alpha_t = alpha * gamma_t * (|R_{t-1}| + 1); reject hypothesis t iff
   e_t >= 1/alpha_t.

No Sharpe/Sortino/P&L content — discovery counts and FDR only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

__all__ = ["EbhResult", "ELond", "StoppedEbhResult", "e_bh", "stopped_e_bh"]

_INF = float("inf")


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not np.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError("alpha must lie in the open interval (0, 1)")
    return a


def _check_e_values(e_values: Array | Iterable[float]) -> Array:
    e = np.asarray(e_values, dtype=float).reshape(-1)
    if e.size == 0:
        raise ValueError("e_values must be nonempty")
    if not bool(np.all(np.isfinite(e))):
        raise ValueError("e-values must be finite (NaN/inf rejected)")
    if bool(np.any(e < 0.0)):
        raise ValueError("e-values must be nonnegative")
    return e


def _ebh_core(e: Array, alpha: float) -> tuple[BoolArray, int, float]:
    """Shared e-BH step: sort, find k-hat, reject e_i >= n/(alpha*k-hat)."""
    n = int(e.size)
    e_sorted = np.sort(e)[::-1]
    ranks = np.arange(1, n + 1, dtype=float)
    thresholds = n / (alpha * ranks)
    hits = e_sorted >= thresholds
    if not bool(np.any(hits)):
        return np.zeros(n, dtype=bool), 0, _INF
    k_hat = int(np.max(ranks[hits]))
    critical_value = float(n / (alpha * float(k_hat)))
    rejected = np.asarray(e >= critical_value, dtype=bool)
    return rejected, int(rejected.sum()), critical_value


@dataclass(frozen=True)
class EbhResult:
    """Output of :func:`e_bh`.

    ``critical_value`` is n/(alpha*k-hat); it is ``inf`` when nothing is
    rejected. ``rejected`` aligns positionally with the input e-values.
    """

    rejected: BoolArray
    num_rejected: int
    critical_value: float


def e_bh(e_values: Array | Iterable[float], alpha: float) -> EbhResult:
    """e-BH: FDR control at level ``alpha`` under arbitrary dependence.

    Sort the n e-values descending, e_(1) >= ... >= e_(n), and set
    k-hat = max{k : e_(k) >= n/(alpha*k)} (zero if no such k). Reject every
    hypothesis with e_i >= n/(alpha*k-hat). Wang & Ramdas (2022, JRSS-B,
    Section 4.1, Theorem 1): the false discovery rate is at most ``alpha``
    for ANY joint dependence between the e-values — the dependence-robustness
    BH lacks. FDR <= alpha requires each input to be a genuine e-value
    (E_P[e_i] <= 1 under its null); garbage inputs void the guarantee.
    """
    a = _check_alpha(alpha)
    e = _check_e_values(e_values)
    rejected, num_rejected, critical_value = _ebh_core(e, a)
    return EbhResult(
        rejected=rejected,
        num_rejected=num_rejected,
        critical_value=critical_value,
    )


@dataclass(frozen=True)
class StoppedEbhResult:
    """Output of :func:`stopped_e_bh`.

    ``stopped_e`` are the lifted stopped e-values actually fed to e-BH
    (adjuster applied to each stream's running maximum, evaluated at its stop
    time); ``critical_value`` is the e-BH threshold n/(alpha*k-hat).
    """

    rejected: BoolArray
    num_rejected: int
    critical_value: float
    stopped_e: Array
    stop_times: IntArray


def stopped_e_bh(
    e_paths: Array,
    alpha: float,
    stop_times: Iterable[int] | IntArray | None = None,
    lift_power: float = 0.5,
) -> StoppedEbhResult:
    """Corrected stopped e-BH (se-BH) of Wang, Dandapanthula & Ramdas (2025).

    Naive se-BH — plain e-BH on the raw stopped values M^g_{tau} of local
    e-processes — can inflate the stopped FDR by up to ~log G, because the
    global stopping time tau need not be a stopping time of stream g's local
    filtration (arXiv:2502.08539, Section 5.5, Eqs. 42-49). The paper's fix
    (Section 4, Eq. 26, Corollary 4.3) is e-lifting: apply a compound adjuster
    to each stream's running maximum,

        M^{ga}_t = A(max{M^g_s : s <= t}),   A(x) = k * x^(1-k),

    where A is a Shafer et al. (2011) adjuster (here k*x^(1-k), k in (0,1),
    for which int_1^inf A(x)/x^2 dx = 1 exactly) and k = ``lift_power``.
    Choe & Ramdas (2024, Theorem 2) show the lifted process is an e-process
    under ANY refinement of the original filtration, hence a global compound
    e-process, so e-BH on the lifted stopped values {M^{ga}_{tau_g}} controls
    the stopped FDR at level ``alpha`` for ANY global stopping times
    (arXiv:2502.08539, Theorem 2.5) — even with cross-stream dependence or
    data-dependent (e.g. wealth-crossing) stop rules.

    Parameters
    ----------
    e_paths : (n_hyp, n_time) array of local e-process values (row g is the
        path of hypothesis g; M_0 = 1 is implicit).
    alpha : FDR level in (0, 1).
    stop_times : optional per-hypothesis 0-based index of the last observed
        time, in [0, n_time-1]; default stops every stream at the final time.
    lift_power : adjuster exponent k in (0, 1); A(x) = k*x^(1-k).
    """
    a = _check_alpha(alpha)
    paths = np.asarray(e_paths, dtype=float)
    if paths.ndim != 2 or paths.shape[0] == 0 or paths.shape[1] == 0:
        raise ValueError("e_paths must be a nonempty 2-D (n_hyp, n_time) array")
    if not bool(np.all(np.isfinite(paths))):
        raise ValueError("e-paths must be finite (NaN/inf rejected)")
    if bool(np.any(paths < 0.0)):
        raise ValueError("e-process values must be nonnegative")
    k = float(lift_power)
    if not np.isfinite(k) or not 0.0 < k < 1.0:
        raise ValueError("lift_power must lie in the open interval (0, 1)")
    n_hyp, n_time = int(paths.shape[0]), int(paths.shape[1])
    if stop_times is None:
        tau = np.full(n_hyp, n_time - 1, dtype=np.int64)
    else:
        tau_raw = np.asarray(stop_times, dtype=float).reshape(-1)
        if tau_raw.size != n_hyp:
            raise ValueError("stop_times must have one entry per hypothesis")
        if bool(np.any(~np.isfinite(tau_raw))):
            raise ValueError("stop_times must be finite integers")
        if bool(np.any(tau_raw != np.floor(tau_raw))):
            raise ValueError("stop_times must be integers")
        if bool(np.any(tau_raw < 0)) or bool(np.any(tau_raw > n_time - 1)):
            raise ValueError("stop_times must lie in [0, n_time-1]")
        tau = tau_raw.astype(np.int64)
    running_max = np.maximum.accumulate(
        np.concatenate([np.ones((n_hyp, 1)), paths], axis=1), axis=1
    )[:, 1:]
    lifted = k * np.power(np.maximum(running_max, 1.0), 1.0 - k)
    stopped = np.asarray(lifted[np.arange(n_hyp), tau], dtype=float)
    rejected, num_rejected, critical_value = _ebh_core(stopped, a)
    return StoppedEbhResult(
        rejected=rejected,
        num_rejected=num_rejected,
        critical_value=critical_value,
        stopped_e=stopped,
        stop_times=tau,
    )


def _default_gamma(j: int) -> float:
    """e-LOND default spending: gamma_j = 0.07*ln(max(j,2)) / (j*e^{sqrt(ln(max(j,2)))}).

    Sums to ~0.41 < 1 over j >= 1 (verified numerically), so it is a valid
    discount sequence (Xu & Ramdas 2024 require sum_j gamma_j <= 1).
    """
    m = float(max(int(j), 2))
    return float(0.07 * np.log(m) / (m * np.exp(np.sqrt(np.log(m)))))


class ELond:
    """e-LOND: online FDR control on a stream of e-values (Xu & Ramdas 2024).

    Section 3.2 of arXiv:2311.06412: with discount sequence (gamma_j),
    gamma_j >= 0, sum_j gamma_j <= 1, the test level of hypothesis j is

        alpha_j = alpha * gamma_j * (|R_{j-1}| + 1),

    where |R_{j-1}| is the number of discoveries among the first j-1
    hypotheses, and hypothesis j is rejected iff e_j >= 1/alpha_j. Theorem 1
    of the paper: FDR(R_t) <= alpha for every t under ARBITRARY dependence
    between the e-values. Each rejection earns +1 of "wealth" — the
    multiplier (|R_{j-1}| + 1) plays the role of an alpha-investing wealth
    and is >= 1 at all times, so the wealth never goes negative (unlike
    LORD's, it cannot be depleted; it only grows on discoveries).

    Default gamma_j = 0.07*ln(max(j,2)) / (max(j,2)*e^{sqrt(ln(max(j,2)))}),
    a summable (~0.41 total) LORD-style sequence; pass ``gamma`` to override
    (callable ``gamma(j)`` with j = 1, 2, ... or an indexable sequence with
    gamma[0] = gamma_1).

    Fail-closed: ``alpha`` outside (0, 1), a negative gamma, or a submitted
    e-value that is negative or non-finite raises ValueError.
    """

    def __init__(
        self,
        alpha: float,
        e_values: Iterable[float] | None = None,
        gamma: Callable[[int], float] | Iterable[float] | None = None,
    ) -> None:
        self._alpha = _check_alpha(alpha)
        if gamma is not None and not callable(gamma):
            seq = np.asarray(list(gamma), dtype=float)
            if seq.ndim != 1 or seq.size == 0:
                raise ValueError("gamma sequence must be a nonempty 1-D sequence")
            gamma = seq
        self._gamma = gamma
        self._n = 0
        self._rejected = 0
        self._levels: list[float] = []
        self._rejections: list[bool] = []
        if e_values is not None:
            for e in e_values:
                self.submit(e)

    def _gamma_j(self, j: int) -> float:
        g = self._gamma
        if g is None:
            return _default_gamma(j)
        val = float(g(j)) if callable(g) else float(g[j - 1])
        if not np.isfinite(val) or val < 0.0:
            raise ValueError("gamma sequence entries must be finite and nonnegative")
        return val

    @property
    def wealth(self) -> float:
        """Current wealth multiplier |R| + 1 (>= 1; grows by 1 per discovery)."""
        return float(self._rejected + 1)

    @property
    def num_submitted(self) -> int:
        return self._n

    @property
    def num_rejected(self) -> int:
        return self._rejected

    @property
    def levels(self) -> tuple[float, ...]:
        """The realized test levels alpha_j = alpha*gamma_j*(|R_{j-1}| + 1)."""
        return tuple(self._levels)

    @property
    def rejections(self) -> tuple[bool, ...]:
        """Per-hypothesis rejection decisions, in submission order."""
        return tuple(self._rejections)

    def submit(self, e_value: float) -> bool:
        """Test the next hypothesis in the stream; True iff it is rejected."""
        e = float(e_value)
        if not np.isfinite(e) or e < 0.0:
            raise ValueError("submitted e-values must be finite and nonnegative")
        self._n += 1
        gamma_j = self._gamma_j(self._n)
        alpha_t = self._alpha * gamma_j * float(self._rejected + 1)
        reject = bool(e >= 1.0 / alpha_t)
        if reject:
            self._rejected += 1
        self._levels.append(alpha_t)
        self._rejections.append(reject)
        return reject

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
4. e-LORD (Zhang, Wei, Ren & Zou, 2025, ICML, arXiv:2506.01452, Algorithm 1):
   generalized alpha-investing (e-GAI) on e-values. Instead of e-LOND's
   pre-specified gamma sequence, a wealth-alpha_investing recursion spends a
   fraction omega_t of the remaining alpha-wealth,
   alpha_t = omega_t * (alpha - sum_{j<t} alpha_j/(R_{j-1}+1)) * (R_{t-1}+1),
   with omega_{t+1} = omega_1 * (1 + sum_{j<=t-R_t} phi^j - sum_{j<=R_t} psi^j)
   (their Eq. 9, the risk-averse-investing update).
5. e-SAFFRON (same paper, Algorithm 2): the adaptive counterpart that
   discounts the FDP estimate Storey-style — only tests with weak evidence
   (e_j < 1/lambda) drain the wealth budget alpha*(1-lambda):
   alpha_t = omega_t * (alpha(1-lambda) - spent_t) * (R_{t-1}+1), where
   spent_t = sum_{j<t} alpha_j * 1{e_j<1/lambda} / (R_{j-1}+1). Following the
   authors' reference implementation (eSAFFRON.R), the level is capped at
   lambda so a rejected hypothesis is never charged as a likely null; this
   makes the procedure strictly more conservative than the uncapped Eq. (10)
   and is the reason the paper's lambda->0 reduction to e-LORD does not hold
   verbatim here.

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

__all__ = [
    "ELond",
    "ELord",
    "ESaffron",
    "EbhResult",
    "StoppedEbhResult",
    "e_bh",
    "stopped_e_bh",
]

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


def _check_omega1(omega1: float) -> float:
    w = float(omega1)
    if not np.isfinite(w) or not 0.0 < w < 0.5:
        raise ValueError("omega1 must lie in the open interval (0, 0.5)")
    return w


def _check_intensity(name: str, value: float) -> float:
    v = float(value)
    if not np.isfinite(v) or not 0.0 <= v <= 0.5:
        raise ValueError(f"{name} must lie in the closed interval [0, 0.5]")
    return v


class _EGaiBase:
    """Shared e-GAI machinery (Zhang, Wei, Ren & Zou 2025, arXiv:2506.01452).

    All e-GAI procedures test hypothesis t at a level
    ``alpha_t = cap(omega_t * remaining_wealth_t * (R_{t-1} + 1))`` and reject
    iff ``e_t >= 1/alpha_t``; the subclass hooks below fix the wealth budget,
    which submitted e-values drain it, and whether the level is capped. The
    allocation coefficient follows the RAI recursion (their Eq. 9)

        omega_{t+1} = omega_1 * (1 + sum_{j=1}^{t-R_t} phi^j
                                     - sum_{j=1}^{R_t} psi^j),

    maintained here through incremental geometric sums: on a non-rejection
    sum_phi <- phi*(1 + sum_phi) appends the next power; on a rejection
    sum_psi <- psi*(1 + sum_psi) likewise. With omega_1 in (0, 0.5) and
    phi, psi in [0, 0.5] (their Remark 3.3 sufficient box) omega_t stays
    strictly inside (0, 1), so levels stay nonnegative; a nonpositive or
    non-finite level is additionally treated as alpha-death — the hypothesis
    is not rejected and no wealth is charged — rather than evaluated against
    the 1/alpha_t threshold.
    """

    def __init__(
        self,
        alpha: float,
        omega1: float,
        phi: float,
        psi: float,
        e_values: Iterable[float] | None = None,
    ) -> None:
        self._alpha = _check_alpha(alpha)
        self._omega1 = _check_omega1(omega1)
        self._phi = _check_intensity("phi", phi)
        self._psi = _check_intensity("psi", psi)
        self._n = 0
        self._rejected = 0
        self._sum_phi = 0.0
        self._sum_psi = 0.0
        self._omega = self._omega1
        self._wealth_used = 0.0
        self._levels: list[float] = []
        self._rejections: list[bool] = []
        if e_values is not None:
            for e in e_values:
                self.submit(e)

    def _budget(self) -> float:
        """Total alpha-wealth the procedure may ever spend."""
        return self._alpha

    def _cap(self, level: float) -> float:
        """Post-process the raw level (identity for e-LORD)."""
        return level

    def _charges(self, e_value: float) -> bool:
        """Whether this submission drains wealth (all do for e-LORD)."""
        return True

    @property
    def omega(self) -> float:
        """Allocation coefficient omega_{t+1} for the next test, in (0, 1)."""
        return float(self._omega)

    @property
    def remaining_wealth(self) -> float:
        """Unspent alpha-wealth (budget minus charged levels)."""
        return float(self._budget() - self._wealth_used)

    @property
    def num_submitted(self) -> int:
        return self._n

    @property
    def num_rejected(self) -> int:
        return self._rejected

    @property
    def levels(self) -> tuple[float, ...]:
        """The realized test levels alpha_t (post-cap)."""
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
        raw_level = self._omega * self.remaining_wealth * float(self._rejected + 1)
        alpha_t = self._cap(raw_level)
        if not np.isfinite(alpha_t) or alpha_t <= 0.0:
            alpha_t = 0.0  # alpha-death: never reject on a nonpositive level
        reject = alpha_t > 0.0 and bool(e >= 1.0 / alpha_t)
        if self._charges(e):
            self._wealth_used += alpha_t / float(self._rejected + 1)
        if reject:
            self._rejected += 1
            self._sum_psi = self._psi * (1.0 + self._sum_psi)
        else:
            self._sum_phi = self._phi * (1.0 + self._sum_phi)
        self._omega = self._omega1 * (1.0 + self._sum_phi - self._sum_psi)
        self._levels.append(alpha_t)
        self._rejections.append(reject)
        return reject


class ELord(_EGaiBase):
    """e-LORD: generalized alpha-investing FDR control on a stream of e-values
    (Zhang, Wei, Ren & Zou 2025, Algorithm 1).

    alpha_1 = alpha*omega_1 and for t >= 2,

        alpha_t = omega_t * (alpha - sum_{j<t} alpha_j/(R_{j-1}+1))
                  * (R_{t-1} + 1),

    i.e. every test — rejected or not — spends alpha_j/(R_{j-1}+1) of the
    total wealth alpha. Theorem 3.1 of the paper: choosing levels so that
    this charged-wealth estimate stays <= alpha yields FDR <= alpha at every
    t for valid online e-values (E[e_t | F_{t-1}] <= 1 under the null),
    under arbitrary cross-hypothesis dependence. Unlike e-LOND the spend is
    data-driven, so long stretches without discoveries can exhaust the
    wealth (alpha-death); the paper recommends omega_1 = O(1/T).

    Parameters
    ----------
    alpha : FDR level in (0, 1).
    omega1 : initial allocation coefficient in (0, 0.5); default 1e-3 suits
        streams of order 10^3 hypotheses (paper suggests omega_1 ~ 1/T).
    phi : non-rejection stimulation intensity in [0, 0.5] (default 0.5);
        phi = 0 freezes omega_t = omega_1.
    psi : rejection risk-regulation intensity in [0, 0.5] (default 0.5).
    """

    def __init__(
        self,
        alpha: float,
        omega1: float = 1e-3,
        phi: float = 0.5,
        psi: float = 0.5,
        e_values: Iterable[float] | None = None,
    ) -> None:
        super().__init__(alpha, omega1, phi, psi, e_values)


class ESaffron(_EGaiBase):
    """e-SAFFRON: adaptive e-GAI with a Storey-style FDP estimate
    (Zhang, Wei, Ren & Zou 2025, Algorithm 2).

    alpha_1 = alpha*(1-lambda)*omega_1 and for t >= 2,

        alpha_t = min(lambda, omega_t * rw_t * (R_{t-1} + 1)),

        rw_t = alpha*(1-lambda)
               - sum_{j<t} alpha_j * 1{e_j < 1/lambda} / (R_{j-1} + 1).

    Only submissions with e_j < 1/lambda — weak evidence, likely nulls —
    drain the budget alpha*(1-lambda); the min(lambda, .) cap follows the
    authors' reference implementation (functions/eSAFFRON.R) so a rejected
    hypothesis is never charged, matching SAFFRON's candidacy logic. This is
    strictly more conservative than the paper's uncapped Eq. (10) — under
    the uncapped formula e-SAFFRON with lambda = 0 reduces to e-LORD
    (their remark after Algorithm 2), which the cap deliberately breaks.
    Proposition 3.4: the adaptive estimate still overestimates FDP in
    expectation, so FDR <= alpha at every t under arbitrary dependence for
    valid online e-values. e-SAFFRON dominates e-LORD when the alternative
    proportion is non-negligible, because rejections stop draining wealth.

    Parameters
    ----------
    alpha : FDR level in (0, 1).
    lam : candidacy threshold lambda in (0, 1); the paper's default 0.1
        (Remark 3.5) — smaller than p-SAFFRON's 0.5 because the e-value
        budget alpha*(1-lambda) is never replenished.
    omega1, phi, psi : as for :class:`ELord`.
    """

    def __init__(
        self,
        alpha: float,
        lam: float = 0.1,
        omega1: float = 1e-3,
        phi: float = 0.5,
        psi: float = 0.5,
        e_values: Iterable[float] | None = None,
    ) -> None:
        lam_val = float(lam)
        if not np.isfinite(lam_val) or not 0.0 < lam_val < 1.0:
            raise ValueError("lam must lie in the open interval (0, 1)")
        self._lam = lam_val
        super().__init__(alpha, omega1, phi, psi, e_values)

    def _budget(self) -> float:
        return self._alpha * (1.0 - self._lam)

    def _cap(self, level: float) -> float:
        return min(self._lam, level)

    def _charges(self, e_value: float) -> bool:
        return bool(e_value < 1.0 / self._lam)

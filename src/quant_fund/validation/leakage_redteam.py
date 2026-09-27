"""Leaky-oracle red-team battery: documenting the DSR/PBO blind spot on SYNTHETIC data.

Encodes the Gençay (2026, arXiv:2608.27734) finding that a deliberately leaky
oracle can survive the standard overfitting gates — Deflated Sharpe Ratio
(DSR) and Probability of Backtest Overfitting (PBO) — so the harness needs
(a) a demonstration battery proving the blind spot on SYNTHETIC data,
(b) a structural look-ahead audit of the feature registry, and (c)
search-trial-count deflation of significance thresholds. Gençay's
recommended structural fix is exactly (b): a registry-validated,
look-ahead-free feature space, so no amount of return-resampling statistics
can rescue a strategy whose inputs already contain the future.

HARD STAMPS (honesty contract; see ``quant_fund.research.catalog``
``FORBIDDEN_RESEARCH_METRIC_KEYS`` and ``quant_fund.metrics.overfitting``):
every artifact this module emits carries ``data_source='SYNTHETIC'`` and
``research_only=True``. This is a CORRECTNESS / RED-TEAM DIAGNOSTIC — it
proves that PSR/DSR/PBO, called here as gate functions on synthetic returns
(exactly their purpose per ``quant_fund.metrics.overfitting``), can be
passed by a strategy with zero genuine skill. It is NEVER market evidence,
never a performance claim, and no headline Sharpe/Sortino/Calmar/P&L/NAV
metric is produced anywhere (the underlying per-period ratios feed PSR/DSR
as gate inputs only, mirroring the research-only docstrings there).

Leaky-oracle construction. The oracle observes the MAGNITUDE of the next
period's shock and takes its profitable side (a pure look-ahead position):

    r_t = eps_t + leakage * |eps_{t+1}|,   eps iid N(0, 1)

The stream has POSITIVE mean leakage*sqrt(2/pi) and ZERO genuine predictive
skill — the apparent skill is entirely future information. (A literal
zero-mean leak r_t = eps_t + leakage*eps_{t+1} has mean exactly zero and
would defeat no Sharpe-based gate; the half-normal leak is the
economically meaningful Gençay-style oracle in which the leaked information
is actually traded.) The control oracle is r_t = eps_t: no leak, zero
skill. At leakage >= 0.5 the leaky stream's per-period ratio is far above
the DSR search-deflation hurdle E[max SR | n_trials] (Bailey & López de
Prado, 2014, Journal of Portfolio Management 40(5):94-98,
"The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
Overfitting and Non-Normality"; probabilistic Sharpe from Bailey & López de
Prado, 2012, Journal of Investment Management), so DSR promotes it at a
rate near 1 while the control promotes at the search-count-deflated false
rate — the blind-spot contrast this module regression-locks.

PBO is the CSCV statistic of Bailey & López de Prado (López de Prado &
Bailey, 2014, Journal of Portfolio Management 40(4):56-69, "The Probability
of Backtest Overfitting", arXiv:1405.3421; computed by
``quant_fund.metrics.overfitting.probability_of_backtest_overfitting``). On
a trial matrix of equally leaky oracles, in-sample-best selection is noise
with respect to the held-out block, so PBO sits near its null 0.5 — neither
flagged nor informative; the leak is invisible to CSCV because the leak
works in-sample AND out-of-sample.

Trial-count deflation (``trial_count_deflation``) is the conservative
small-team approximation of search-count deflation:
``base_alpha / (1 + log(max(n_trials, 1)))``, a Bonferroni-style
(Bonferroni, 1935; 1936, Pubbl. Ist. Sup. Econ. Com. Firenze 8:3-62)
log-count correction for the number of search trials a strategy survived.

Fail-closed everywhere: n_days < 252 -> ValueError for the oracle makers
(the 1-year minimum for any annualized-skill statistic), n_trials < 2 for
the battery -> ValueError, empty feature list -> ValueError, non-finite
returns -> ValueError.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.overfitting import (
    deflated_sharpe,
    moments_from_returns,
    probabilistic_sharpe,
    probability_of_backtest_overfitting,
)

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

__all__ = [
    "AuditResult",
    "BatteryResult",
    "StudyResult",
    "dsr_pbo_battery",
    "make_control_oracle",
    "make_leaky_oracle",
    "run_leaky_oracle_study",
    "structural_lookahead_audit",
    "trial_count_deflation",
]

# Minimum history for any annualized-skill statistic (1 trading year).
_MIN_DAYS = 252
# Default DSR promotion level (probability the true ratio exceeds the
# search-deflated hurdle).
_DSR_LEVEL = 0.95
# Default number of CSCV contiguous blocks for the battery PBO.
_N_SPLITS = 16
# Default per-period ratio sampling variance for DSR under iid Gaussian
# returns: Var(SR_hat) ~= 1 / (n_obs - 1).
# Default search size the oracle must survive in the study (n_search > 1 is
# what makes the DSR hurdle positive; with a single trial the hurdle is 0
# and both oracles promote at rate ~0.5, which would be a meaningless gate).
_STUDY_SEARCH_TRIALS = 16
# Default size of the per-replicate trial matrix fed to PBO in the study.
_STUDY_MATRIX_TRIALS = 8
# Default study history: 2 trading years.
_STUDY_DAYS = 504

# Look-ahead substrings for the structural audit (lowercased containment).
_LOOKAHEAD_PATTERNS: tuple[str, ...] = (
    "future",
    "lead",
    "next_",
    "shift(-",
    "ahead",
    "lookahead",
    "forward_fill",
)


def _check_days(n_days: int) -> int:
    n = int(n_days)
    if n < _MIN_DAYS:
        raise ValueError(
            f"n_days={n} < {_MIN_DAYS}: a 1-year minimum history is required "
            "for any annualized-skill statistic"
        )
    return n


def _check_finite_returns(r: Array, name: str = "returns") -> None:
    if not bool(np.all(np.isfinite(r))):
        raise ValueError(f"{name} must be finite (NaN/inf rejected)")


def _per_period_ratio(r: Array) -> float:
    """Mean/std (ddof=1) per-period ratio; honest NaN when degenerate."""
    mu = float(np.mean(r))
    sig = float(np.std(r, ddof=1))
    if sig <= 0.0 or not np.isfinite(sig):
        return float("nan")
    return mu / sig


def make_leaky_oracle(
    n_days: int,
    leakage: float,
    rng: np.random.Generator,
    n_assets: int = 1,
) -> dict[str, Any]:
    """SYNTHETIC leaky oracle: r_t = eps_t + leakage * |eps_{t+1}|.

    PURE look-ahead with zero genuine skill: the oracle observes the
    magnitude of the NEXT period's shock and takes its profitable side, so
    the stream has positive mean ``leakage * sqrt(2/pi)`` that survives
    Sharpe-based gates entirely because it is future information
    (Gençay, 2026, arXiv:2608.27734). ``eps`` is iid standard normal.
    ``leakage = 0`` degenerates to the control stream.

    Parameters
    ----------
    n_days:
        History length; must be >= 252 (fail-closed: below the 1-year
        minimum for any annualized-skill statistic).
    leakage:
        Non-negative, finite look-ahead weight on the next period's shock
        magnitude.
    rng:
        Seeded Generator; all randomness flows through it.
    n_assets:
        Number of independent leaky streams; 1 (default) returns a flat
        (n_days,) array, otherwise (n_assets, n_days).

    Returns
    -------
    dict
        ``returns`` ((n_days,) or (n_assets, n_days)), ``has_leakage: True``,
        ``leakage``, ``n_days``, ``n_assets``, ``data_source: 'SYNTHETIC'``,
        ``research_only: True``. SYNTHETIC correctness-diagnostic artifact —
        never market evidence.

    Fail-closed: n_days < 252, non-finite or negative leakage, n_assets < 1.
    """
    n = _check_days(n_days)
    lam = float(leakage)
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError("leakage must be finite and >= 0")
    k = int(n_assets)
    if k < 1:
        raise ValueError("n_assets must be >= 1")
    eps = rng.standard_normal((k, n + 1))
    leak = np.abs(eps[:, 1:])
    rets = eps[:, :n] + lam * leak
    out: Array = rets[0] if k == 1 else rets
    return {
        "returns": out,
        "has_leakage": True,
        "leakage": lam,
        "n_days": n,
        "n_assets": k,
        "data_source": "SYNTHETIC",
        "research_only": True,
    }


def make_control_oracle(n_days: int, rng: np.random.Generator) -> dict[str, Any]:
    """SYNTHETIC control oracle: r_t = eps_t — no leak, zero skill.

    Same marginal law as the leaky oracle at ``leakage = 0``; the honest
    baseline for the blind-spot contrast. Stamped ``data_source='SYNTHETIC'``,
    ``research_only=True``; correctness diagnostic only, never market
    evidence. Fail-closed: n_days < 252.
    """
    n = _check_days(n_days)
    return {
        "returns": rng.standard_normal(n),
        "has_leakage": False,
        "leakage": 0.0,
        "n_days": n,
        "n_assets": 1,
        "data_source": "SYNTHETIC",
        "research_only": True,
    }


@dataclass(frozen=True)
class BatteryResult:
    """Output of :func:`dsr_pbo_battery`.

    SYNTHETIC red-team artifact (``data_source='SYNTHETIC'``,
    ``research_only=True``): per-trial PSR and DSR gate values, per-trial
    DSR promotion flags, the battery-wide promotion rate, and the CSCV PBO
    of the trial matrix. Diagnostic only — never market evidence; the
    per-period ratios are PSR/DSR gate inputs, not headline metrics.
    """

    psr: Array
    dsr: Array
    dsr_promoted: BoolArray
    promotion_rate: float
    pbo: float
    n_days: int
    n_trials: int
    n_splits: int
    dsr_level: float
    data_source: str = "SYNTHETIC"
    research_only: bool = True


def _cscv_block_sharpes(mat: Array, n_splits: int) -> tuple[Array, Array]:
    """Contiguous-block CSCV: (n_splits, n_trials) IS and OOS ratios."""
    blocks = [
        np.asarray(b, dtype=np.intp) for b in np.array_split(np.arange(mat.shape[0]), n_splits)
    ]
    is_rows: list[Array] = []
    oos_rows: list[Array] = []
    for s in range(n_splits):
        oos_idx = blocks[s]
        is_idx = np.concatenate([blocks[j] for j in range(n_splits) if j != s])
        is_rows.append(np.array([_per_period_ratio(mat[is_idx, t]) for t in range(mat.shape[1])]))
        oos_rows.append(np.array([_per_period_ratio(mat[oos_idx, t]) for t in range(mat.shape[1])]))
    return np.vstack(is_rows), np.vstack(oos_rows)


def dsr_pbo_battery(
    trial_returns: Array | Iterable[Iterable[float]],
    n_splits: int = _N_SPLITS,
    dsr_level: float = _DSR_LEVEL,
) -> BatteryResult:
    """Run PSR/DSR gates + CSCV PBO on a SYNTHETIC trial-return matrix.

    Wraps (does not reimplement) ``probabilistic_sharpe``,
    ``deflated_sharpe``, and ``probability_of_backtest_overfitting`` from
    ``quant_fund.metrics.overfitting`` — calling those gate functions on
    synthetic returns is their stated purpose. For each of the ``n_trials``
    columns: the per-period ratio and its third/fourth moments feed PSR at
    hurdle 0 and DSR at the search-deflated hurdle E[max SR | n_trials]
    with ``var_sr`` estimated as the cross-trial variance of the observed
    per-period ratios (Bailey & López de Prado, 2014). PBO splits the
    n_days rows into ``n_splits`` contiguous blocks, holds one out per
    split, and asks whether the in-sample-best trial lands below the
    held-out median (López de Prado & Bailey, 2014).

    Parameters
    ----------
    trial_returns:
        (n_days, n_trials) finite SYNTHETIC return matrix, n_trials >= 2.
    n_splits:
        Number of contiguous CSCV blocks, >= 2 and <= n_days // 2.
    dsr_level:
        Promotion threshold on the DSR probability in [0.5, 1).

    Fail-closed: wrong shape, n_trials < 2, non-finite entries, or invalid
    n_splits/dsr_level raise ValueError. SYNTHETIC artifact, stamped
    ``data_source='SYNTHETIC'``, ``research_only=True``; correctness
    diagnostic only, never market evidence.
    """
    mat = np.asarray(trial_returns, dtype=float)
    if mat.ndim != 2 or mat.shape[1] < 2:
        raise ValueError("trial_returns must be 2-D (n_days, n_trials) with n_trials >= 2")
    n_days, n_trials = int(mat.shape[0]), int(mat.shape[1])
    _check_finite_returns(mat, "trial_returns")
    s = int(n_splits)
    if s < 2 or s > n_days // 2:
        raise ValueError("n_splits must lie in [2, n_days // 2]")
    lvl = float(dsr_level)
    if not np.isfinite(lvl) or not 0.5 <= lvl < 1.0:
        raise ValueError("dsr_level must lie in [0.5, 1)")

    psr = np.empty(n_trials, dtype=float)
    dsr = np.empty(n_trials, dtype=float)
    ratios = np.empty(n_trials, dtype=float)
    for t in range(n_trials):
        col = mat[:, t]
        sig, skew, kurt = moments_from_returns(col)
        ratios[t] = float(np.mean(col) / sig) if sig > 0.0 else float("nan")
        psr[t] = probabilistic_sharpe(ratios[t], 0.0, n_days, skew, kurt)
    var_sr = float(np.var(ratios, ddof=1))
    for t in range(n_trials):
        sig, skew, kurt = moments_from_returns(mat[:, t])
        dsr[t] = deflated_sharpe(ratios[t], n_days, skew, kurt, n_trials, var_sr)

    is_sharpes, oos_sharpes = _cscv_block_sharpes(mat, s)
    pbo = probability_of_backtest_overfitting(is_sharpes, oos_sharpes)
    promoted = np.asarray(dsr >= lvl, dtype=bool)
    return BatteryResult(
        psr=psr,
        dsr=dsr,
        dsr_promoted=promoted,
        promotion_rate=float(np.mean(promoted)),
        pbo=float(pbo),
        n_days=n_days,
        n_trials=n_trials,
        n_splits=s,
        dsr_level=lvl,
    )


@dataclass(frozen=True)
class StudyResult:
    """Output of :func:`run_leaky_oracle_study`.

    SYNTHETIC red-team artifact (``data_source='SYNTHETIC'``,
    ``research_only=True``): per leakage level, the Monte Carlo DSR
    promotion rate of the leaky oracle vs the control (the blind-spot
    contrast — the STUDY HEADLINE), with Monte Carlo standard errors, plus
    per-level mean PBO of leaky and control trial matrices. ``headline``
    is a fixed-format human-readable contrast string. Correctness
    diagnostic only — never market evidence.
    """

    leakage_levels: tuple[float, ...]
    promotion_rate_leaky: Array
    promotion_rate_control: Array
    promotion_rate_leaky_se: Array
    promotion_rate_control_se: Array
    pbo_leaky: Array
    pbo_control: Array
    blind_spot_contrast: Array
    headline: str
    n_mc: int
    n_days: int
    n_search_trials: int
    dsr_level: float
    data_source: str = "SYNTHETIC"
    research_only: bool = True
    diagnostic_kind: str = field(default="leaky_oracle_blind_spot", repr=False)


def run_leaky_oracle_study(
    leakage_levels: Iterable[float],
    n_mc: int = 200,
    seed: int = 0,
    *,
    n_days: int = _STUDY_DAYS,
    n_search_trials: int = _STUDY_SEARCH_TRIALS,
    matrix_trials: int = _STUDY_MATRIX_TRIALS,
    dsr_level: float = _DSR_LEVEL,
) -> StudyResult:
    """Monte Carlo blind-spot study: leaky vs control oracle under DSR/PBO.

    For each leakage level, ``n_mc`` replicates draw one leaky oracle and
    one control oracle (SYNTHETIC, ``r_t = eps_t + leakage*|eps_{t+1}|`` vs
    ``r_t = eps_t``) and evaluate each as one candidate out of a
    ``n_search_trials``-trial search: the DSR hurdle is
    ``expected_max_sharpe(n_search_trials, 1/(n_days-1))`` — search-count
    deflation is exactly what DSR adds over PSR, and it is what keeps the
    CONTROL's promotion rate small while the leaky oracle sails through
    (Gençay, 2026, arXiv:2608.27734). Each replicate also builds a
    (n_days, ``matrix_trials``) trial matrix of leaky (resp. control)
    streams and computes its CSCV PBO; per-level means are reported. The
    STUDY HEADLINE is the blind-spot contrast — leaky promotion rate vs
    control — and the whole artifact is stamped SYNTHETIC /
    research_only: a correctness/red-team diagnostic, never market
    evidence.

    Deterministic: per-level generators derive from
    ``SeedSequence([seed, level_index])``.

    Fail-closed: empty/non-finite/negative leakage levels, n_mc < 1,
    n_days < 252, n_search_trials < 2, matrix_trials < 2, or dsr_level
    outside [0.5, 1) raise ValueError.
    """
    levels = tuple(float(x) for x in leakage_levels)
    if not levels:
        raise ValueError("leakage_levels must be nonempty")
    if not all(np.isfinite(x) and x >= 0.0 for x in levels):
        raise ValueError("leakage_levels must be finite and >= 0")
    m = int(n_mc)
    if m < 1:
        raise ValueError("n_mc must be >= 1")
    n = _check_days(n_days)
    n_search = int(n_search_trials)
    if n_search < 2:
        raise ValueError("n_search_trials must be >= 2")
    m_trials = int(matrix_trials)
    if m_trials < 2:
        raise ValueError("matrix_trials must be >= 2")
    lvl = float(dsr_level)
    if not np.isfinite(lvl) or not 0.5 <= lvl < 1.0:
        raise ValueError("dsr_level must lie in [0.5, 1)")

    var_sr = 1.0 / float(n - 1)
    promo_leaky = np.empty(len(levels), dtype=float)
    promo_ctrl = np.empty(len(levels), dtype=float)
    pbo_leaky = np.empty(len(levels), dtype=float)
    pbo_ctrl = np.empty(len(levels), dtype=float)

    for i, lam in enumerate(levels):
        gen = np.random.default_rng(np.random.SeedSequence([int(seed), i]))
        leaky_flags = np.empty(m, dtype=float)
        ctrl_flags = np.empty(m, dtype=float)
        leaky_pbos = np.empty(m, dtype=float)
        ctrl_pbos = np.empty(m, dtype=float)
        for r in range(m):
            leaky_flags[r] = _dsr_promotion(
                make_leaky_oracle(n, lam, gen)["returns"], n_search, var_sr, n, lvl
            )
            ctrl_flags[r] = _dsr_promotion(
                make_control_oracle(n, gen)["returns"], n_search, var_sr, n, lvl
            )
            leaky_mat = np.column_stack(
                [make_leaky_oracle(n, lam, gen)["returns"] for _ in range(m_trials)]
            )
            ctrl_mat = np.column_stack(
                [make_control_oracle(n, gen)["returns"] for _ in range(m_trials)]
            )
            leaky_pbos[r] = _matrix_pbo(leaky_mat)
            ctrl_pbos[r] = _matrix_pbo(ctrl_mat)
        promo_leaky[i] = float(np.mean(leaky_flags))
        promo_ctrl[i] = float(np.mean(ctrl_flags))
        pbo_leaky[i] = float(np.mean(leaky_pbos))
        pbo_ctrl[i] = float(np.mean(ctrl_pbos))

    contrast = promo_leaky - promo_ctrl
    headline = (
        "SYNTHETIC red-team diagnostic (never market evidence): at leakage "
        f"{max(levels):.2f}, leaky-oracle DSR promotion "
        f"{promo_leaky[-1]:.3f} vs control {promo_ctrl[-1]:.3f} "
        f"(n_mc={m}, n_days={n}, search_trials={n_search}) — DSR/PBO blind "
        "spot reproduced per Gençay (2026, arXiv:2608.27734)"
    )
    return StudyResult(
        leakage_levels=levels,
        promotion_rate_leaky=promo_leaky,
        promotion_rate_control=promo_ctrl,
        promotion_rate_leaky_se=np.sqrt(promo_leaky * (1.0 - promo_leaky) / m),
        promotion_rate_control_se=np.sqrt(promo_ctrl * (1.0 - promo_ctrl) / m),
        pbo_leaky=pbo_leaky,
        pbo_control=pbo_ctrl,
        blind_spot_contrast=contrast,
        headline=headline,
        n_mc=m,
        n_days=n,
        n_search_trials=n_search,
        dsr_level=lvl,
    )


def _dsr_promotion(
    returns: Array, n_search: int, var_sr: float, n_days: int, dsr_level: float
) -> float:
    """1.0 if the stream's DSR (as 1-of-n_search candidate) clears dsr_level."""
    sig, skew, kurt = moments_from_returns(returns)
    if not (np.isfinite(sig) and sig > 0.0):
        return 0.0
    ratio = float(np.mean(returns) / sig)
    d = deflated_sharpe(ratio, n_days, skew, kurt, n_search, var_sr)
    return 1.0 if (np.isfinite(d) and d >= dsr_level) else 0.0


def _matrix_pbo(mat: Array, n_splits: int = 8) -> float:
    """CSCV PBO of a trial matrix; honest NaN passes through as NaN."""
    is_sharpes, oos_sharpes = _cscv_block_sharpes(mat, n_splits)
    return float(probability_of_backtest_overfitting(is_sharpes, oos_sharpes))


@dataclass(frozen=True)
class AuditResult:
    """Output of :func:`structural_lookahead_audit`.

    SYNTHETIC red-team artifact (``data_source='SYNTHETIC'``,
    ``research_only=True``): ``fail`` is True when any feature name matches
    a look-ahead pattern; ``flagged`` lists the offending names. This is
    the Gençay-recommended structural fix — a registry-validated,
    look-ahead-free feature space — implemented as a fail-closed gate.
    Correctness diagnostic only, never market evidence.
    """

    fail: bool
    flagged: tuple[str, ...]
    n_features: int
    patterns: tuple[str, ...]
    extra_patterns: tuple[str, ...]
    data_source: str = "SYNTHETIC"
    research_only: bool = True


def structural_lookahead_audit(
    feature_names: Iterable[str],
    extra_patterns: Iterable[str] | None = None,
) -> AuditResult:
    """Fail-closed structural look-ahead audit of a feature registry.

    Flags (case-insensitive substring match, lowercase) any feature name
    containing a look-ahead token: 'future', 'lead', 'next_', 'shift(-',
    'ahead', 'lookahead', 'forward_fill' — the structural signatures of
    future information in a feature pipeline — plus any caller-supplied
    ``extra_patterns``. This operationalizes the Gençay (2026,
    arXiv:2608.27734) recommended structural fix: no return-resampling
    statistic can rescue a strategy whose inputs already contain the
    future, so the feature space itself must be registry-validated and
    look-ahead-free. The audit is SYNTHETIC-agnostic (it audits names, not
    data) but every artifact is stamped ``data_source='SYNTHETIC'``,
    ``research_only=True`` as a correctness/red-team diagnostic, never
    market evidence.

    Fail-closed: empty feature list, or an empty/whitespace extra pattern,
    raises ValueError. ``fail`` is True iff at least one name is flagged.
    """
    names = tuple(str(x) for x in feature_names)
    if not names:
        raise ValueError("feature_names must be nonempty")
    extras = tuple(str(x) for x in extra_patterns) if extra_patterns is not None else ()
    for p in extras:
        if not p or not p.strip():
            raise ValueError("extra_patterns must be nonempty, non-whitespace strings")
    flagged = tuple(
        name
        for name in names
        if any(pat in name.lower() for pat in _LOOKAHEAD_PATTERNS)
        or any(pat in name.lower() for pat in extras)
    )
    return AuditResult(
        fail=bool(flagged),
        flagged=flagged,
        n_features=len(names),
        patterns=_LOOKAHEAD_PATTERNS,
        extra_patterns=extras,
    )


def trial_count_deflation(n_trials: int, base_alpha: float) -> float:
    """Search-trial-count deflation of a significance threshold.

    Conservative small-team approximation of search-count deflation:

        alpha_adj = base_alpha / (1 + log(max(n_trials, 1)))

    A Bonferroni-style log-count factor: the more trials a strategy
    survived, the lower the threshold at which its evidence counts
    (Bonferroni, 1935; 1936). At n_trials = 1 the factor is 1 and the
    threshold is unchanged; it decreases monotonically in n_trials.
    Companion to DSR's E[max SR] hurdle (Bailey & López de Prado, 2014)
    for p-value-style gates that have no built-in search deflation.

    Fail-closed: n_trials < 1 or non-integer-valued, or base_alpha outside
    (0, 1), raises ValueError. SYNTHETIC red-team diagnostic only.
    """
    n = float(n_trials)
    if not np.isfinite(n) or n < 1.0 or n != float(int(n)):
        raise ValueError("n_trials must be a positive integer")
    alpha = float(base_alpha)
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("base_alpha must lie in (0, 1)")
    return float(alpha / (1.0 + np.log(max(int(n), 1))))

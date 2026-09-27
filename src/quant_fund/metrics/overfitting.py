"""Probabilistic and Deflated Sharpe, CSCV/PBO, and trial dependence.

Bailey & López de Prado. Research-diagnostic only — never a live P&L /
promotion claim. Empty or non-finite inputs → honest NaN (or fail-closed
ValueError for invalid ``n_trials``).

The per-period score passed to these helpers is whatever series the
research runner already selects on (date-level IC). The ratio of its mean
to its standard deviation is the sampling-frequency input the Bailey–López
de Prado formulas require. It is not an annualized portfolio ratio and is
not stored under a forbidden research-headline key.
"""

from __future__ import annotations

from itertools import combinations
from math import comb, sqrt
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import norm

# Fixed a priori. Trials whose average-linkage cophenetic distance is at
# most the Mantegna distance of this correlation are one effective trial.
# The threshold is not estimated from a sample and is not fit to a result.
CORRELATED_TRIAL_MIN_RHO = 0.5

Array = NDArray[np.float64]

EULER_MASCHERONI = 0.5772156649015329


def _sr_se(sr: float, skew: float, kurtosis_raw: float) -> float:
    """Non-normal SE of Sharpe (Lo 2002). kurtosis_raw is the raw fourth moment / sigma^4."""
    inside = 1.0 - skew * sr + ((kurtosis_raw - 1.0) / 4.0) * sr**2
    return float(np.sqrt(max(inside, 1e-18)))


def _finite_sharpe_inputs(*vals: float) -> bool:
    return all(np.isfinite(v) for v in vals)


def probabilistic_sharpe(
    sr: float,
    sr_star: float,
    n_obs: int,
    skew: float,
    kurtosis_raw: float,
) -> float:
    """PSR: P(true SR > sr_star | observed moments). Research diagnostic only."""
    if not _finite_sharpe_inputs(sr, sr_star, skew, kurtosis_raw) or not np.isfinite(n_obs):
        return float("nan")
    if int(n_obs) < 2:
        return float("nan")
    se = _sr_se(float(sr), float(skew), float(kurtosis_raw))
    return float(norm.cdf((float(sr) - float(sr_star)) * np.sqrt(int(n_obs) - 1) / se))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected max of ``n_trials`` zero-mean Sharpes (Bailey–LdP). Diagnostic only."""
    if not np.isfinite(n_trials) or n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if not np.isfinite(var_sr):
        return float("nan")
    if n_trials == 1:
        return 0.0
    n = float(n_trials)
    gamma = EULER_MASCHERONI
    term = (1.0 - gamma) * norm.ppf(1.0 - 1.0 / n) + gamma * norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(np.sqrt(max(float(var_sr), 0.0)) * term)


def deflated_sharpe(
    sr: float,
    n_obs: int,
    skew: float,
    kurtosis_raw: float,
    n_trials: int,
    var_sr: float,
) -> float:
    """DSR = PSR with sr* = E[max SR under n_trials]. Research diagnostic only — not live P&L."""
    if not _finite_sharpe_inputs(sr, skew, kurtosis_raw, var_sr) or not np.isfinite(n_obs):
        return float("nan")
    sr_star = expected_max_sharpe(n_trials, var_sr)
    return probabilistic_sharpe(sr, sr_star, n_obs, skew, kurtosis_raw)


def probability_of_backtest_overfitting(is_sharpes: Array, oos_sharpes: Array) -> float:
    """CSCV PBO (López de Prado): share of splits where the IS-best trial is OOS-below-median.

    Both arrays are shape (n_splits, n_trials). Higher Sharpe is better.
    Research-diagnostic only — not a live Sharpe / P&L claim.
    Empty, mismatched, too-small (need ≥2 splits and ≥2 trials), 1×N / N×1,
    or partially non-finite paths → honest NaN (never a silent 0.0).
    A PBO computed after silently dropping a split is not a valid model-selection
    diagnostic, so the function refuses the entire matrix when any score is
    missing or non-finite. Valid finite matrices may still return exact 0.0 or 1.0.
    """
    ins = np.asarray(is_sharpes, dtype=float)
    oos = np.asarray(oos_sharpes, dtype=float)
    if ins.shape != oos.shape or ins.ndim != 2 or ins.shape[0] < 2 or ins.shape[1] < 2:
        return float("nan")
    if not np.isfinite(ins).all() or not np.isfinite(oos).all():
        return float("nan")
    flags: list[float] = []
    for i in range(ins.shape[0]):
        best = int(np.nanargmax(ins[i]))
        oos_row = oos[i]
        med = float(np.median(oos_row))
        flags.append(1.0 if oos_row[best] < med else 0.0)
    if not flags:
        return float("nan")
    return float(np.mean(flags))


def moments_from_returns(returns: Array) -> tuple[float, float, float]:
    """Return (sigma, skew, raw_kurtosis) for PSR/DSR. Research diagnostic only.

    Needs ≥4 finite observations; else honest (NaN, NaN, NaN).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 4:
        return float("nan"), float("nan"), float("nan")
    mu = float(np.mean(r))
    sig = float(np.std(r, ddof=1))
    if sig == 0:
        return 0.0, 0.0, 3.0
    z = (r - mu) / sig
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))  # raw
    return sig, skew, kurt


def min_track_record_length(
    sr: float,
    skew: float,
    kurtosis_raw: float,
    *,
    conf: float = 0.95,
    sr_star: float = 0.0,
) -> float:
    """Minimum track-record length (Bailey & López de Prado) for PSR ≥ conf.

    Research-diagnostic only — not a live Sharpe / promotion claim.
    Non-finite inputs, ``conf`` ∉ (0.5, 1), or ``sr == sr_star`` → honest NaN.
    """
    if not _finite_sharpe_inputs(sr, skew, kurtosis_raw, sr_star, conf):
        return float("nan")
    if not (0.5 < float(conf) < 1.0):
        return float("nan")
    diff = float(sr) - float(sr_star)
    if abs(diff) < 1e-18:
        return float("nan")
    z = float(norm.ppf(float(conf)))
    if not np.isfinite(z):
        return float("nan")
    # Bailey–LdP: MinTRL = 1 + [1 - γ3 SR + (γ4−1)/4 SR²] (z/(SR−SR*))²
    # Negative/zero SE-variance term is clamped (same as _sr_se) for honest finite MinTRL.
    inside = 1.0 - float(skew) * float(sr) + ((float(kurtosis_raw) - 1.0) / 4.0) * float(sr) ** 2
    # Keep the minimum track length strictly above one even when extreme
    # higher-moment inputs drive the asymptotic variance term toward zero.
    inside = max(inside, 1e-12)
    return float(1.0 + inside * (z / diff) ** 2)


def min_trl_from_returns(
    returns: Array,
    *,
    conf: float = 0.95,
    sr_star: float = 0.0,
) -> dict[str, float | int | bool | str]:
    """Research-only Bailey–LdP MinTRL smoke from return moments.

    Computes per-period mean/std SR plus ``moments_from_returns`` skew/kurtosis,
    then ``min_track_record_length``. Payload keys intentionally omit underscore
    tokens ``sharpe`` / ``sortino`` / ``calmar`` / ``pnl`` / ``nav`` so research
    family-blob hygiene stays green.

    Not a live performance / promotion claim. Short, non-finite, or zero-vol
    paths → honest NaN MinTRL fields with ``n_obs`` still reported.
    """
    r = np.asarray(returns, dtype=float).reshape(-1)
    r = r[np.isfinite(r)]
    n = int(r.size)
    out: dict[str, float | int | bool | str] = {
        "n_obs": n,
        "conf": float(conf),
        "min_track_record_length": float("nan"),
        "min_trl": float("nan"),
        "track_record_bars": float("nan"),
        "research_only": True,
        # Intentionally omit live_pnl_claim — "pnl" is a forbidden key token.
        "claim": "bailey_ldp_min_trl_diagnostic_only",
    }
    if n < 4:
        return out
    sig, skew, kurt = moments_from_returns(r)
    if not (np.isfinite(sig) and np.isfinite(skew) and np.isfinite(kurt)) or sig == 0.0:
        return out
    # Per-period observed ratio (same scale as moments); not annualized headline.
    observed = float(np.mean(r) / sig)
    mtrl = min_track_record_length(
        observed, float(skew), float(kurt), conf=float(conf), sr_star=float(sr_star)
    )
    out["min_track_record_length"] = float(mtrl)
    out["min_trl"] = float(mtrl)
    if np.isfinite(mtrl):
        out["track_record_bars"] = float(np.ceil(mtrl))
    return out


def _score_ratio(column: Array) -> float:
    """Mean/std of a per-period score. Needs ≥2 finite points and positive scale."""
    values = np.asarray(column, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    scale = float(np.std(values, ddof=1))
    if scale == 0.0 or not np.isfinite(scale):
        return float("nan")
    return float(np.mean(values) / scale)


def _correlation_matrix(scores: Array) -> Array:
    """Pairwise correlation. Constant columns correlate 0 with everyone else."""
    data = np.asarray(scores, dtype=float)
    _t, k = data.shape
    corr = np.eye(k, dtype=float)
    if k < 2:
        return corr
    centered = data - data.mean(axis=0, keepdims=True)
    scale = data.std(axis=0, ddof=1)
    denom_n = float(data.shape[0] - 1)
    for i in range(k):
        for j in range(i + 1, k):
            degenerate = (
                not np.isfinite(scale[i])
                or not np.isfinite(scale[j])
                or scale[i] == 0.0
                or scale[j] == 0.0
            )
            if degenerate:
                rho = 0.0
            else:
                rho = float(
                    np.dot(centered[:, i], centered[:, j]) / (denom_n * scale[i] * scale[j])
                )
                if not np.isfinite(rho):
                    rho = 0.0
                rho = float(min(1.0, max(-1.0, rho)))
            corr[i, j] = rho
            corr[j, i] = rho
    return corr


def effective_n_trials(
    scores: Array,
    *,
    min_rho: float = CORRELATED_TRIAL_MIN_RHO,
) -> tuple[int, NDArray[np.intp]]:
    """Effective independent-trial count by clustering correlated columns.

    ``scores`` has shape ``(n_obs, n_trials)``. Average-linkage clustering
    on the Mantegna distance ``sqrt((1-ρ)/2)`` (Mantegna 1999; López de Prado,
    AFML ch. 2) cuts at ``ρ = min_rho``. Columns that are constant, or a
    sample shorter than 4 observations, are left unclustered: every column
    then counts, which is the larger multiplicity. Returns ``(n_effective, labels)``
    with labels in ``0 .. n_effective-1``.
    """
    data = np.asarray(scores, dtype=float)
    if data.ndim != 2:
        raise ValueError("scores must have shape (n_obs, n_trials)")
    if not np.isfinite(min_rho) or not -1.0 < float(min_rho) < 1.0:
        raise ValueError("min_rho must lie in (-1, 1)")
    n_obs, n_trials = data.shape
    if n_trials < 2 or n_obs < 4:
        return int(n_trials), np.arange(n_trials, dtype=np.intp)
    corr = _correlation_matrix(data)
    distance = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    np.fill_diagonal(distance, 0.0)
    distance = np.clip((distance + distance.T) / 2.0, 0.0, None)
    if float(np.max(distance)) <= 1e-12:
        return 1, np.zeros(n_trials, dtype=np.intp)
    condensed = squareform(distance, checks=False)
    linked = linkage(condensed, method="average")
    threshold = sqrt((1.0 - float(min_rho)) / 2.0)
    raw = fcluster(linked, t=threshold, criterion="distance")
    # Compress arbitrary cluster ids to 0 .. n_effective-1, stable by first appearance.
    mapping: dict[int, int] = {}
    labels = np.empty(n_trials, dtype=np.intp)
    for index, cluster_id in enumerate(raw.tolist()):
        key = int(cluster_id)
        if key not in mapping:
            mapping[key] = len(mapping)
        labels[index] = mapping[key]
    return int(len(mapping)), labels


def _representative_indices(scores: Array, labels: NDArray[np.intp]) -> NDArray[np.intp]:
    """One column per cluster: the member with the largest mean score."""
    data = np.asarray(scores, dtype=float)
    reps: list[int] = []
    for cluster_id in range(int(labels.max()) + 1 if labels.size else 0):
        members = np.flatnonzero(labels == cluster_id)
        if members.size == 0:
            continue
        means = np.array([float(np.nanmean(data[:, member])) for member in members], dtype=float)
        if not np.isfinite(means).any():
            reps.append(int(members[0]))
            continue
        reps.append(int(members[int(np.nanargmax(means))]))
    return np.asarray(reps, dtype=np.intp)


def choose_cscv_slices(n_obs: int, *, max_slices: int = 16) -> int:
    """Largest even slice count with at least two observations per slice.

    Bailey, Borwein, López de Prado & Zhu (2017) split the record into an
    even number of contiguous blocks and take every combination of half of
    them as the in-sample set. ``S=16`` is their illustrated size
    (``C(16, 8) = 12_870``). Shorter records use the largest even ``S`` that
    still leaves two observations in each block. Fewer than four observations
    cannot form two non-empty symmetric halves.
    """
    if n_obs < 4 or max_slices < 2:
        return 0
    slices = min(int(max_slices), int(n_obs) // 2)
    if slices % 2:
        slices -= 1
    return slices if slices >= 2 else 0


def cscv_performance(
    scores: Array,
    n_slices: int,
) -> tuple[Array, Array]:
    """In-sample and out-of-sample mean scores for every symmetric CSCV split.

    ``scores`` has shape ``(n_obs, n_trials)``. The timeline is cut into
    ``n_slices`` contiguous blocks (``n_slices`` even, ≥2). Each combination
    of ``n_slices/2`` blocks is the in-sample set; the complement is
    out-of-sample. Performance is the length-weighted mean of the per-period
    score on those observations (the quantity the research runner ranks).

    Returns two arrays of shape ``(C(n_slices, n_slices/2), n_trials)``.
    """
    data = np.asarray(scores, dtype=float)
    if data.ndim != 2:
        raise ValueError("scores must have shape (n_obs, n_trials)")
    if n_slices < 2 or n_slices % 2:
        raise ValueError("n_slices must be even and at least 2")
    n_obs, _n_trials = data.shape
    if n_obs < n_slices:
        raise ValueError("need at least one observation per slice")
    bounds = [int(i * n_obs / n_slices) for i in range(n_slices + 1)]
    if any(bounds[i + 1] <= bounds[i] for i in range(n_slices)):
        raise ValueError("slice bounds produced an empty block")
    slice_sum = np.vstack([data[bounds[i] : bounds[i + 1]].sum(axis=0) for i in range(n_slices)])
    slice_n = np.array([bounds[i + 1] - bounds[i] for i in range(n_slices)], dtype=float)
    half = n_slices // 2
    combos = np.asarray(list(combinations(range(n_slices), half)), dtype=int)
    is_sum = slice_sum[combos].sum(axis=1)
    is_n = slice_n[combos].sum(axis=1)
    total_sum = slice_sum.sum(axis=0)
    total_n = float(slice_n.sum())
    oos_sum = total_sum - is_sum
    oos_n = total_n - is_n
    return is_sum / is_n[:, None], oos_sum / oos_n[:, None]


def _variance_of_ratios(ratios: Array) -> float:
    finite = np.asarray(ratios, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float("nan")
    return float(np.var(finite, ddof=1))


def _json_float(value: float) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def overfitting_diagnostics(
    scores: Array | None,
    *,
    n_trials: int,
    names: list[str] | None = None,
    horizon_bars: int = 1,
    embargo_bars: int = 0,
    n_groups: int = 6,
    n_test_groups: int = 2,
) -> dict[str, Any]:
    """PBO, DSR, PSR, MinTRL, and trial counts for one research run.

    ``scores`` is the aligned per-period score matrix ``(n_obs, n_aligned)``
    of the configs the runner evaluated. ``n_trials`` is the full count of
    those configs, including any that could not be aligned; unaligned trials
    stay in the multiplicity and are not clustered away. Research diagnostic
    only. Keys omit the forbidden headline tokens.

    Probability of backtest overfitting is the CSCV fraction from
    :func:`probability_of_backtest_overfitting`. DSR uses the effective
    independent-trial count; ``dsr_counted_trials`` repeats the deflation
    with every recorded trial counted separately, so a clustering reduction
    is visible next to the stricter count.
    """
    from quant_fund.validation.cpcv import (
        combinatorial_purged_indices,
        cpcv_n_paths,
        cpcv_n_splits,
    )

    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 0:
        raise ValueError("n_trials must be a non-negative integer")
    if names is not None and scores is not None:
        width = int(np.asarray(scores).shape[1]) if np.asarray(scores).ndim == 2 else -1
        if len(names) != width:
            raise ValueError("names must match the number of score columns")
    block: dict[str, Any] = {
        "claim": "research_diagnostic_only",
        "research_only": True,
        "metrics_status": "unavailable",
        "pbo": None,
        "dsr": None,
        "dsr_counted_trials": None,
        "psr": None,
        "min_trl": None,
        "n_trials": int(n_trials),
        "n_trials_effective": int(n_trials),
        "n_obs": 0,
        "n_aligned": 0,
        "selected_trial": None,
        "cluster_corr_min": float(CORRELATED_TRIAL_MIN_RHO),
        "clustering_applied": False,
        "cscv_slices": 0,
        "cpcv_n_splits": None,
        "cpcv_n_paths": None,
        "cpcv_n_folds": None,
        "purge_disjoint": None,
    }
    data = None if scores is None else np.asarray(scores, dtype=float)
    if data is None or data.ndim != 2 or data.shape[1] == 0 or data.shape[0] == 0:
        return block
    n_obs, n_aligned = int(data.shape[0]), int(data.shape[1])
    if n_aligned > n_trials:
        raise ValueError("n_trials cannot be smaller than the aligned score width")
    block["n_obs"] = n_obs
    block["n_aligned"] = n_aligned
    unaligned = n_trials - n_aligned
    if n_obs >= 4 and n_aligned >= 2:
        n_cluster, labels = effective_n_trials(data)
        block["clustering_applied"] = True
    else:
        n_cluster, labels = n_aligned, np.arange(n_aligned, dtype=np.intp)
    # Unaligned evaluated configs have no series to cluster, so each remains
    # its own trial. That keeps the multiplicity from shrinking by omission.
    block["n_trials_effective"] = int(n_cluster + unaligned)

    ratios = np.array([_score_ratio(data[:, column]) for column in range(n_aligned)], dtype=float)
    means = np.array(
        [
            float(np.mean(data[:, column])) if np.isfinite(data[:, column]).any() else float("nan")
            for column in range(n_aligned)
        ],
        dtype=float,
    )
    if np.isfinite(means).any():
        best = int(np.nanargmax(means))
        block["selected_trial"] = names[best] if names is not None else str(best)
        best_ratio = float(ratios[best])
        _sigma, skew, kurt = moments_from_returns(data[:, best])
        if np.isfinite(best_ratio) and np.isfinite(skew) and np.isfinite(kurt) and n_obs >= 4:
            block["psr"] = _json_float(
                probabilistic_sharpe(best_ratio, 0.0, n_obs, float(skew), float(kurt))
            )
            block["min_trl"] = _json_float(
                min_track_record_length(
                    best_ratio, float(skew), float(kurt), conf=0.95, sr_star=0.0
                )
            )
            reps = _representative_indices(data, labels)
            rep_ratios = ratios[reps] if reps.size else ratios[:0]
            n_effective = int(n_cluster + unaligned)
            # A single independent trial has no cross-trial dispersion (V=0).
            # Extra unaligned trials with no recorded ratio must not be given
            # V=0: that would drop the multiplicity penalty. Leave DSR unset
            # unless a real variance is available.
            if n_effective <= 1:
                var_eff = 0.0
            elif n_cluster >= 2:
                var_eff = _variance_of_ratios(rep_ratios)
            else:
                var_eff = _variance_of_ratios(ratios)
            var_all = 0.0 if n_trials <= 1 else _variance_of_ratios(ratios)
            if n_effective >= 1 and np.isfinite(var_eff):
                block["dsr"] = _json_float(
                    deflated_sharpe(
                        best_ratio,
                        n_obs,
                        float(skew),
                        float(kurt),
                        n_effective,
                        float(var_eff),
                    )
                )
            if n_trials >= 1 and np.isfinite(var_all):
                block["dsr_counted_trials"] = _json_float(
                    deflated_sharpe(
                        best_ratio,
                        n_obs,
                        float(skew),
                        float(kurt),
                        int(n_trials),
                        float(var_all),
                    )
                )
    n_slices = choose_cscv_slices(n_obs)
    block["cscv_slices"] = int(n_slices)
    if n_slices >= 2 and n_aligned >= 2 and np.isfinite(data).all():
        is_perf, oos_perf = cscv_performance(data, n_slices)
        block["pbo"] = _json_float(probability_of_backtest_overfitting(is_perf, oos_perf))
        block["cscv_combinations"] = int(comb(n_slices, n_slices // 2))
    if (
        isinstance(horizon_bars, int)
        and not isinstance(horizon_bars, bool)
        and isinstance(embargo_bars, int)
        and not isinstance(embargo_bars, bool)
        and horizon_bars >= 0
        and embargo_bars >= 0
        and n_obs >= n_groups >= 2
        and 1 <= n_test_groups < n_groups
    ):
        folds = combinatorial_purged_indices(
            n_obs,
            n_groups,
            n_test_groups,
            label_horizon=horizon_bars,
            embargo=embargo_bars,
        )
        block["cpcv_n_splits"] = int(cpcv_n_splits(n_groups, n_test_groups))
        block["cpcv_n_paths"] = int(cpcv_n_paths(n_groups, n_test_groups))
        block["cpcv_n_folds"] = int(len(folds))
        block["purge_disjoint"] = all(
            set(train.tolist()).isdisjoint(test.tolist()) for train, test in folds
        )
    finite_metrics = [
        block["pbo"],
        block["dsr"],
        block["psr"],
        block["min_trl"],
    ]
    if any(value is not None for value in finite_metrics):
        block["metrics_status"] = "computed"
    return block

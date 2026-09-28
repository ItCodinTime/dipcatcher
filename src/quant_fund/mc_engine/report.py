"""Assemble the scenario-risk report. Headline figures are proper scores and tails.

Sharpe, Sortino, Calmar, P&L, and NAV are not report keys. Loss is the drop
of a unit numeraire. Intervals are simulation error.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from quant_fund.mc_engine.aggregate import P2Quantile, welford_variance
from quant_fund.mc_engine.engine_config import EngineConfig
from quant_fund.mc_engine.summary import MergedSummary
from quant_fund.mc_engine.tails import (
    NUMERAIRE_START,
    batch_means_es_interval,
    normal_z,
    pot_from_exceedances,
    pot_gpd,
    spectral_es,
    weighted_expected_shortfall,
    wilson_interval,
)
from quant_fund.mc_engine.variance import antithetic_mean_vrf, control_variate_from_stats

FloatArray = NDArray[np.float64]

LIMITATIONS: tuple[str, ...] = (
    "Built-in generators are synthetic. A plug-in generator is only as real as the data source it declares. Scenario output is not market evidence.",
    "Intervals measure Monte Carlo error under the scenario design, not sampling error of a historical market estimate.",
    "This engine does not submit orders and does not claim live profit.",
    "t-digest and P² figures are sketches. Exact mode headlines the empirical quantile and reports the sketch error beside it.",
    "P² is a single stream. It is applied in path order to retained losses and is not merged across workers.",
    "Variance-reduction factors are separate measurements. They are not multiplied, and a factor below 1 is left below 1.",
    "Importance-sampling tail probabilities and expected shortfall are self-normalized. Peaks-over-threshold is not fit on a weighted sample.",
    "Quasi-Monte Carlo does not get an independence-based interval. With two or more scrambles, the standard error across scrambles is reported instead.",
    "Bit-identical output requires the same chunk size, seed, generator, and platform arithmetic, with chunks folded in chunk-id order. Worker count is not an input.",
)

_DD_QUANTILES = (0.5, 0.9, 0.95, 0.99)


def limitations() -> list[str]:
    return list(LIMITATIONS)


def array_fingerprint(merged: MergedSummary) -> str:
    digest = hashlib.sha256()
    if merged.retain_samples:
        arrays: list[tuple[bytes, np.ndarray]] = [
            (b"loss", np.asarray(merged.loss)),
            (b"max_drawdown", np.asarray(merged.max_drawdown)),
            (b"ruined", np.asarray(merged.ruined)),
            (b"recovery_steps", np.asarray(merged.recovery_steps)),
        ]
        for name, arr in arrays:
            digest.update(name)
            digest.update(np.ascontiguousarray(arr).tobytes())
        if merged.weight is None:
            digest.update(b"weight:uniform")
        else:
            digest.update(b"weight")
            digest.update(np.ascontiguousarray(merged.weight).tobytes())
    else:
        digest.update(b"sketch")
        digest.update(np.ascontiguousarray(merged.loss_digest.means).tobytes())
        digest.update(np.ascontiguousarray(merged.loss_digest.weights).tobytes())
        digest.update(np.asarray(merged.welford_loss[1], dtype=np.float64).tobytes())
        digest.update(str(merged.ruin_count).encode())
        digest.update(str(merged.n_paths).encode())
    return digest.hexdigest()


def _sample_variance(n: int, total: float, total_sq: float) -> float | None:
    if n < 2:
        return None
    return float((total_sq - (total * total) / n) / (n - 1))


def _weighted_quantile(values: FloatArray, weights: FloatArray, probability: float) -> float:
    order = np.argsort(values, kind="mergesort")
    ordered = np.asarray(values, dtype=np.float64)[order]
    ordered_w = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(ordered_w)
    target = float(probability) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(index, int(ordered.size) - 1)
    return float(ordered[index])


def _empirical_quantile(values: FloatArray, probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability, method="linear"))


def _level_key(level: float) -> str:
    return format(float(level), ".10g")


def _es_block(
    merged: MergedSummary,
    config: EngineConfig,
    level: float,
    *,
    weighted: bool,
    allow_batch_ci: bool,
) -> dict[str, object]:
    if merged.retain_samples and merged.loss is not None:
        if weighted and merged.weight is not None:
            estimate = weighted_expected_shortfall(merged.loss, merged.weight, level)
            method = "self_normalized_weighted_spectral"
        else:
            estimate = spectral_es(merged.loss, level)
            method = "spectral_empirical"
        approximate = False
        if allow_batch_ci:
            interval = batch_means_es_interval(
                merged.loss,
                level,
                config.ci_level,
                weights=merged.weight if weighted else None,
            )
            ci_method = "batch_means"
            ci_reason = interval["reason"]
            ci_low = interval["ci_low"]
            ci_high = interval["ci_high"]
            se = interval["standard_error"]
            batch_count = interval["batch_count"]
        else:
            ci_method = None
            ci_low = None
            ci_high = None
            se = None
            batch_count = None
            ci_reason = (
                "independence-based intervals are not reported for quasi-Monte Carlo; "
                "use n_scrambles >= 2 for a standard error across scrambles"
                if config.shock_mode == "qmc_sobol"
                else "batch interval was not applied"
            )
    else:
        estimate = merged.loss_digest.expected_shortfall(level)
        method = "t_digest_centroid"
        approximate = True
        ci_method = None
        ci_low = None
        ci_high = None
        se = None
        batch_count = None
        ci_reason = (
            "expected-shortfall intervals need retained scenario losses "
            "(memory_mode='exact'); the t-digest figure is a sketch"
        )
    return {
        "estimate": estimate,
        "estimate_method": method,
        "approximate": approximate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": config.ci_level,
        "ci_method": ci_method,
        "ci_reason": ci_reason,
        "batch_count": batch_count,
        "standard_error": se,
    }


def _var_block(
    merged: MergedSummary,
    level: float,
    *,
    weighted: bool,
) -> dict[str, object]:
    digest_q = merged.loss_digest.quantile(level)
    psquare: float | None = None
    exact: float | None = None
    if merged.retain_samples and merged.loss is not None:
        if weighted and merged.weight is not None:
            exact = _weighted_quantile(merged.loss, merged.weight, level)
        else:
            exact = _empirical_quantile(merged.loss, level)
            estimator = P2Quantile(level)
            estimator.add_all(merged.loss)
            psquare = estimator.value
    estimate = exact if exact is not None else digest_q
    error = None if exact is None else abs(digest_q - exact)
    psquare_error = None if psquare is None or exact is None else abs(psquare - exact)
    return {
        "estimate": estimate,
        "approximate": exact is None,
        "method": "empirical" if exact is not None else "t_digest_centroid",
        "t_digest": digest_q,
        "psquare": psquare,
        "abs_error_t_digest": error,
        "abs_error_psquare": psquare_error,
    }


def _distribution_quantiles(
    values: FloatArray | None,
    digest_quantile: Callable[[float], float],
    probabilities: tuple[float, ...],
    *,
    weights: FloatArray | None,
) -> dict[str, object]:
    out: dict[str, object] = {}
    for probability in probabilities:
        key = _level_key(probability)
        if values is None:
            out[key] = {"estimate": digest_quantile(probability), "approximate": True}
        elif weights is None:
            out[key] = {
                "estimate": _empirical_quantile(values, probability),
                "approximate": False,
                "t_digest": digest_quantile(probability),
            }
        else:
            out[key] = {
                "estimate": _weighted_quantile(values, weights, probability),
                "approximate": False,
            }
    return out


def _evt_block(
    merged: MergedSummary,
    config: EngineConfig,
    *,
    weighted: bool,
) -> dict[str, object]:
    if weighted:
        return {
            "available": False,
            "diagnostics_ok": False,
            "reason": (
                "peaks-over-threshold is fit to an unweighted sample; "
                "this engine does not claim a weighted GPD fit"
            ),
        }
    if merged.evt_truncated:
        return {
            "available": False,
            "diagnostics_ok": False,
            "reason": "exceedance buffer was truncated; the fit is refused",
        }
    if merged.retain_samples and merged.loss is not None:
        evt = pot_gpd(merged.loss, threshold=config.evt_threshold)
    elif config.evt_threshold is not None:
        evt = pot_from_exceedances(merged.exceedances, config.evt_threshold, merged.n_paths)
    else:
        return {
            "available": False,
            "diagnostics_ok": False,
            "reason": (
                "sketch mode needs an absolute evt_threshold so each worker can "
                "retain every exceedance; otherwise retain the losses"
            ),
        }
    if config.shock_mode != "crude":
        evt = dict(evt)
        evt["iid_inference"] = False
        evt["iid_note"] = (
            "Kolmogorov-Smirnov p-values and Hessian standard errors assume "
            "i.i.d. excesses. That assumption is only the crude Monte Carlo design."
        )
    else:
        evt = dict(evt)
        evt["iid_inference"] = True
    return evt


def _variance_block(
    merged: MergedSummary,
    config: EngineConfig,
    *,
    qmc_means: list[float] | None,
    qmc_crude_variance: float | None,
) -> dict[str, object]:
    block: dict[str, object] = {
        "measured": bool(config.measure_variance_reduction),
        "note": (
            "Each technique is measured on its own. Factors are not multiplied. "
            "A factor below 1 means the technique increased variance on this sample."
        ),
    }
    if not config.measure_variance_reduction:
        block["antithetic"] = None
        block["control_variate"] = None
        block["importance_sampling"] = None
        block["quasi_monte_carlo"] = None
        block["reason"] = "variance reduction measurement disabled"
        return block
    if config.shock_mode == "antithetic" and merged.loss is not None:
        block["antithetic"] = antithetic_mean_vrf(merged.loss)
    elif config.shock_mode == "antithetic":
        block["antithetic"] = {
            "variance_reduction_factor": None,
            "variance_reduction_factor_infinite": False,
            "reason": "antithetic factor needs retained path losses",
        }
    else:
        block["antithetic"] = None
    if config.control_variate:
        if not merged.has_control or merged.control_mean is None:
            block["control_variate"] = {
                "variance_reduction_factor": None,
                "variance_reduction_factor_infinite": False,
                "reason": "generator did not supply a control variate with a known mean",
            }
        else:
            block["control_variate"] = control_variate_from_stats(
                merged.pilot, merged.eval_cross, merged.control_mean
            )
    else:
        block["control_variate"] = None
    if config.shock_mode == "importance":
        block["importance_sampling"] = _importance_block(merged)
    else:
        block["importance_sampling"] = None
    if config.shock_mode == "qmc_sobol":
        block["quasi_monte_carlo"] = _qmc_block(config, qmc_means, qmc_crude_variance)
    else:
        block["quasi_monte_carlo"] = None
    return block


def _importance_block(merged: MergedSummary) -> dict[str, object]:
    ess = (merged.sum_weight**2) / merged.sum_weight_sq if merged.sum_weight_sq > 0.0 else 0.0
    degenerate = ess < 0.1 * merged.n_paths
    var_crude = _sample_variance(merged.crude_n, merged.crude_sum, merged.crude_sumsq)
    var_weighted = _sample_variance(
        merged.n_paths, merged.sum_weighted_loss, merged.sum_weighted_loss_sq
    )
    factor: float | None = None
    infinite = False
    reason: str | None = None
    if merged.crude_n < 2 or var_crude is None or var_weighted is None:
        reason = "matched crude reference was not retained"
    elif var_weighted == 0.0 and var_crude > 0.0:
        infinite = True
        reason = "importance-weighted outcomes have zero sample variance"
    elif var_weighted <= 0.0:
        reason = "importance-sampling variance is not positive"
    else:
        factor = var_crude / var_weighted
    if degenerate and factor is not None:
        reason = (
            "effective sample size is below 10% of the path count; "
            "the ratio is reported and is not a claim of a successful reduction"
        )
    return {
        "method": "importance_sampling_unbiased_mean",
        "estimator": "unbiased_mean",
        "effective_sample_size": ess,
        "weight_degeneracy": degenerate,
        "claim": factor is not None and not degenerate and not infinite,
        "variance_crude_estimator": None if var_crude is None else var_crude / merged.n_paths,
        "variance_reduced_estimator": None
        if var_weighted is None
        else var_weighted / merged.n_paths,
        "variance_reduction_factor": factor,
        "variance_reduction_factor_infinite": infinite,
        "reason": reason,
        "note": (
            "The crude reference uses the same seed without the mean shift and is "
            "not part of the headline distribution. The factor is for mean(weight * loss)."
        ),
    }


def _qmc_block(
    config: EngineConfig,
    qmc_means: list[float] | None,
    qmc_crude_variance: float | None,
) -> dict[str, object]:
    if qmc_means is None or len(qmc_means) < 2:
        return {
            "method": "rqmc_vs_crude_mean",
            "variance_reduction_factor": None,
            "variance_reduction_factor_infinite": False,
            "rqmc_mean_standard_error": None,
            "reason": (
                "a single Sobol sequence has no internal i.i.d. variance; "
                "set n_scrambles >= 2 to measure the standard error across scrambles"
            ),
        }
    means = np.asarray(qmc_means, dtype=np.float64)
    var_scramble = float(np.var(means, ddof=1))
    se = float(math.sqrt(var_scramble / means.size)) if var_scramble >= 0.0 else None
    factor: float | None = None
    infinite = False
    reason: str | None = None
    var_crude_mean: float | None = None
    if qmc_crude_variance is None:
        reason = "matched crude Monte Carlo reference was not run"
    elif var_scramble == 0.0 and qmc_crude_variance > 0.0:
        infinite = True
        reason = "scramble means have zero sample variance"
        var_crude_mean = qmc_crude_variance / config.n_paths
    elif var_scramble <= 0.0:
        reason = "RQMC scramble variance is not positive"
    else:
        var_crude_mean = qmc_crude_variance / config.n_paths
        factor = var_crude_mean / var_scramble
    return {
        "method": "rqmc_vs_crude_mean",
        "estimator": "mean",
        "n_scrambles": int(means.size),
        "scramble_means": [float(v) for v in means.tolist()],
        "variance_one_scramble_mean": var_scramble,
        "variance_crude_estimator": var_crude_mean,
        "rqmc_mean_standard_error": se,
        "variance_reduction_factor": factor,
        "variance_reduction_factor_infinite": infinite,
        "reason": reason,
        "note": (
            "Headline paths are scramble 0 only. Other scrambles and the crude "
            "reference are measurement instruments and are not pooled into the tails."
        ),
    }


def _recovery_block(
    merged: MergedSummary,
    config: EngineConfig,
    *,
    weighted: bool,
) -> dict[str, object]:
    n_with = merged.drawdown_count
    n_recovered = merged.recovered_count
    n_censored = n_with - n_recovered
    ci: tuple[float | None, float | None]
    ci_method: str | None
    ci_reason: str | None
    if weighted:
        probability = (
            None
            if merged.sum_weight_drawdown <= 0.0
            else merged.sum_weight_recovered / merged.sum_weight_drawdown
        )
        mean_steps = (
            None
            if merged.sum_weight_recovered <= 0.0
            else merged.sum_weighted_recovery / merged.sum_weight_recovered
        )
        ci = (None, None)
        ci_method = None
        ci_reason = "Wilson intervals need unweighted counts"
    else:
        probability = None if n_with == 0 else n_recovered / n_with
        mean_steps = None if n_recovered == 0 else merged.recovery_step_sum / n_recovered
        if n_with == 0:
            ci = (None, None)
            ci_method = None
            ci_reason = "no path drew down"
        else:
            z = normal_z(config.ci_level)
            ci = wilson_interval(float(n_recovered), int(n_with), z)
            ci_method = "wilson"
            ci_reason = None
    quantiles: dict[str, object] | None
    if merged.recovery_steps is not None and merged.recovered is not None:
        steps = merged.recovery_steps[np.asarray(merged.recovered, dtype=np.uint8).astype(bool)]
        if steps.size == 0:
            quantiles = None
        else:
            quantiles = {
                _level_key(p): _empirical_quantile(steps.astype(np.float64), p) for p in (0.5, 0.9)
            }
    elif merged.recovery_digest.total_weight > 0.0:
        quantiles = {_level_key(p): merged.recovery_digest.quantile(p) for p in (0.5, 0.9)}
    else:
        quantiles = None
    return {
        "n_with_drawdown": n_with,
        "n_recovered": n_recovered,
        "n_censored": n_censored,
        "probability_recovered_given_drawdown": probability,
        "probability_ci_low": ci[0],
        "probability_ci_high": ci[1],
        "probability_ci_method": ci_method,
        "probability_ci_reason": ci_reason,
        "mean_steps_given_recovery": mean_steps,
        "quantiles_given_recovery": quantiles,
        "conditional_on_recovery": True,
        "censored_excluded_from_mean": True,
        "approximate_quantiles": merged.recovery_steps is None and quantiles is not None,
    }


def _ruin_block(
    merged: MergedSummary,
    config: EngineConfig,
    *,
    weighted: bool,
) -> dict[str, object]:
    if weighted:
        estimate = merged.sum_weighted_ruin / merged.sum_weight if merged.sum_weight > 0.0 else None
        return {
            "estimate": estimate,
            "ruin_level": config.ruin_level,
            "n_ruined": merged.ruin_count,
            "estimator": "self_normalized_importance_weights",
            "ci_low": None,
            "ci_high": None,
            "ci_method": None,
            "ci_reason": (
                "a Wilson interval needs unweighted Bernoulli trials; "
                "the weighted ruin rate is reported without one"
            ),
        }
    estimate = merged.ruin_count / merged.n_paths
    z = normal_z(config.ci_level)
    low, high = wilson_interval(float(merged.ruin_count), merged.n_paths, z)
    return {
        "estimate": estimate,
        "ruin_level": config.ruin_level,
        "n_ruined": merged.ruin_count,
        "estimator": "sample_frequency",
        "ci_low": low,
        "ci_high": high,
        "ci_level": config.ci_level,
        "ci_method": "wilson",
        "ci_reason": None,
    }


def _chunk_es_diagnostic(
    merged: MergedSummary, levels: tuple[float, ...]
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for index, level in enumerate(levels):
        column = merged.es_chunks[:, index]
        finite = column[np.isfinite(column)]
        out[_level_key(level)] = None if finite.size == 0 else float(finite.mean())
    return out


def scenario_mean(merged: MergedSummary, shock_mode: str) -> float:
    """Target-measure mean loss. Self-normalized when the shock mode is importance sampling."""
    if shock_mode == "importance":
        if merged.sum_weight <= 0.0:
            raise ValueError("importance weights have no mass")
        return float(merged.sum_weighted_loss / merged.sum_weight)
    if merged.retain_samples and merged.loss is not None:
        return float(np.mean(merged.loss))
    return float(merged.welford_loss[1])


def scenario_variance(merged: MergedSummary, shock_mode: str) -> float | None:
    """Variance that pairs with :func:`scenario_mean` for a crude reference."""
    if shock_mode == "importance":
        return _sample_variance(
            merged.n_paths, merged.sum_weighted_loss, merged.sum_weighted_loss_sq
        )
    if merged.retain_samples and merged.loss is not None:
        if merged.loss.size < 2:
            return None
        return float(np.var(merged.loss, ddof=1))
    return welford_variance(merged.welford_loss)


def build_report(
    merged: MergedSummary,
    config: EngineConfig,
    *,
    generator_name: str,
    generator_type: str,
    data_source: str,
    n_steps: int,
    n_factors: int,
    elapsed_s: float,
    workers_used: int,
    qmc_means: list[float] | None = None,
    qmc_crude_variance: float | None = None,
    replicate_elapsed_s: float | None = None,
) -> dict[str, object]:
    weighted = merged.weight is not None or (
        config.shock_mode == "importance" and merged.sum_weight > 0.0 and not merged.retain_samples
    )
    # Exact uniform weights are stored as None. Importance mode with retained
    # weights is weighted. Sketch importance has no weight array; the sums carry it.
    if config.shock_mode == "importance":
        weighted = True
    allow_batch_ci = config.shock_mode in {"crude", "antithetic", "importance"}
    mean_loss = scenario_mean(merged, config.shock_mode)
    variance = scenario_variance(merged, config.shock_mode)
    if config.shock_mode == "importance":
        variance_name = "variance_of_weight_times_loss"
    elif merged.retain_samples and merged.loss is not None:
        variance_name = "sample_variance_ddof1"
    else:
        variance_name = "welford_chan_merge"
    simulated_mean = float(merged.welford_loss[1])
    es = {
        _level_key(level): _es_block(
            merged, config, level, weighted=weighted, allow_batch_ci=allow_batch_ci
        )
        for level in config.es_levels
    }
    var = {
        _level_key(level): _var_block(merged, level, weighted=weighted)
        for level in config.es_levels
    }
    dd_quantiles = _distribution_quantiles(
        merged.max_drawdown,
        merged.dd_digest.quantile,
        _DD_QUANTILES,
        weights=merged.weight if weighted and merged.weight is not None else None,
    )
    if weighted and merged.weight is not None and merged.max_drawdown is not None:
        dd_mean = float(np.sum(merged.weight * merged.max_drawdown) / np.sum(merged.weight))
        dd_approx = False
    elif weighted:
        dd_mean = merged.sum_weighted_dd / merged.sum_weight
        dd_approx = True
    elif merged.max_drawdown is not None:
        dd_mean = float(np.mean(merged.max_drawdown))
        dd_approx = False
    else:
        dd_mean = float(merged.welford_dd[1])
        dd_approx = True
    power_of_two = config.n_paths > 0 and (config.n_paths & (config.n_paths - 1)) == 0
    sobol = None
    if config.shock_mode == "qmc_sobol":
        sobol = {
            "scramble": config.qmc_scramble,
            "n_scrambles_measured": 1 if qmc_means is None else len(qmc_means),
            "balance_power_of_two": power_of_two,
            "balance_note": (
                "Sobol balance properties are exact at powers of two. "
                "Other lengths are still deterministic; the report does not pretend otherwise."
            ),
        }
    importance = None
    if config.shock_mode == "importance":
        ess = effective_sample_size_from_sums(merged.sum_weight, merged.sum_weight_sq)
        importance = {
            "shift_on_factor_0": config.importance_shift,
            "mean_weight": merged.sum_weight / merged.n_paths,
            "effective_sample_size": ess,
            "effective_sample_size_fraction": ess / merged.n_paths,
            "mean_loss_unbiased": merged.sum_weighted_loss / merged.n_paths,
            "mean_loss_self_normalized": mean_loss,
            "generator_evaluations_per_path": 2 if config.measure_variance_reduction else 1,
        }
    return {
        "mc_engine_version": 1,
        "status": "complete",
        "research_only": True,
        "live_pnl_claim": False,
        "market_evidence": False,
        "data_source": data_source,
        "simulation_claim": "scenario_simulation_not_market_evidence",
        "generator": {"name": generator_name, "type": generator_type},
        "seed": config.seed,
        "n_paths": config.n_paths,
        "n_steps": n_steps,
        "n_factors": n_factors,
        "chunk_size": config.chunk_size,
        "n_chunks": len(merged.chunks),
        "workers_requested": config.workers,
        "workers_used": workers_used,
        "backend": config.backend,
        "shock_mode": config.shock_mode,
        "memory_mode": config.memory_mode,
        "control_variate": config.control_variate,
        "numeraire_start": NUMERAIRE_START,
        "fingerprint": array_fingerprint(merged),
        "elapsed_s": elapsed_s,
        "paths_per_s_wall_clock": config.n_paths / elapsed_s if elapsed_s > 0.0 else None,
        "replicate_elapsed_s": replicate_elapsed_s,
        "moments": {
            "n": merged.n_paths,
            "mean_loss": mean_loss,
            "mean_loss_simulated_sample": simulated_mean,
            "variance_loss": variance,
            "variance_definition": variance_name,
            "mean_terminal_level": NUMERAIRE_START - mean_loss,
        },
        "expected_shortfall": es,
        "value_at_risk": var,
        "max_drawdown": {
            "mean": dd_mean,
            "quantiles": dd_quantiles,
            "approximate": dd_approx,
        },
        "probability_of_ruin": _ruin_block(merged, config, weighted=weighted),
        "time_to_recovery": _recovery_block(merged, config, weighted=weighted),
        "evt": _evt_block(merged, config, weighted=weighted),
        "variance_reduction": _variance_block(
            merged,
            config,
            qmc_means=qmc_means,
            qmc_crude_variance=qmc_crude_variance,
        ),
        "mean_of_chunk_es": {
            "values": _chunk_es_diagnostic(merged, config.es_levels),
            "note": (
                "Average of within-chunk expected shortfall. "
                "It is not the pooled expected shortfall."
            ),
        },
        "quantile_sketch": {
            "t_digest_compression": config.tdigest_compression,
            "t_digest_centroids": int(merged.loss_digest.means.size),
            "merge": "chunk_id_order_then_compress",
            "psquare_applied": merged.retain_samples and not weighted,
        },
        "sobol": sobol,
        "importance": importance,
        "limitations": limitations(),
    }


def effective_sample_size_from_sums(sum_weight: float, sum_weight_sq: float) -> float:
    if sum_weight_sq <= 0.0:
        return 0.0
    return float((sum_weight * sum_weight) / sum_weight_sq)


def incomplete_report(
    *,
    chunks_completed: int,
    chunks_total: int,
    checkpoint_dir: str | None,
) -> dict[str, object]:
    return {
        "mc_engine_version": 1,
        "status": "incomplete",
        "research_only": True,
        "live_pnl_claim": False,
        "market_evidence": False,
        "simulation_claim": "scenario_simulation_not_market_evidence",
        "chunks_completed": chunks_completed,
        "chunks_total": chunks_total,
        "checkpoint_dir": checkpoint_dir,
        "limitations": limitations(),
    }

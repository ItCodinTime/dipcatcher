"""Stylized-fact tests for one synthetic agent-market tape.

Rules below were written before the measurement. A miss is a miss. Short
samples are inconclusive. Asymptotic p-values treat observations as
independent, so they are anti-conservative when the tape is dependent.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from quant_fund.market_sim.config import EVIDENCE, EcologyConfig, validation_config
from quant_fund.market_sim.impact import measure_impact
from quant_fund.market_sim.simulator import SimResult, run_ecology
from quant_fund.metrics.fractal import dfa_hurst
from quant_fund.metrics.serial import autocorrelation, engle_arch_lm, jarque_bera, ljung_box

# Sample gates and decision thresholds. Fixed before looking at the tape.
MIN_RETURNS = 200
MIN_DFA = 256
MIN_SHAPE = 100
FAT_TAIL_P = 0.05
FAT_TAIL_KURTOSIS = 1.0
ARCH_LAGS = 5
ARCH_P = 0.05
LJUNG_LAG = 10
LJUNG_P = 0.05
HURST_ABS_MIN = 0.60
HURST_SIGNED_LO = 0.35
HURST_SIGNED_HI = 0.65

PVALUE_NOTE = (
    "Asymptotic p-values assume independent observations and are "
    "anti-conservative under dependence."
)


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _fact(
    name: str,
    status: str,
    *,
    test: str,
    rule: str,
    n: int,
    reason: str = "",
    **stats_out: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": name,
        "status": status,
        "test": test,
        "rule": rule,
        "n": int(n),
        "reason": reason,
    }
    payload.update(stats_out)
    return payload


def _fact_n(fact: dict[str, object]) -> int:
    value = fact.get("n", 0)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _shape_series(values: np.ndarray, label: str) -> dict[str, object]:
    sample = np.asarray(values, dtype=float).reshape(-1)
    sample = sample[np.isfinite(sample) & (sample > 0.0)]
    n = int(sample.size)
    if n < MIN_SHAPE:
        return _fact(
            label,
            "inconclusive",
            test="skewtest_and_aic",
            rule="n >= 100, positive skew at 5%, lognormal AIC beats normal",
            n=n,
            reason="sample",
        )
    skewness = float(stats.skew(sample))
    skew_p = _finite(float(stats.skewtest(sample).pvalue))
    try:
        norm_params = stats.norm.fit(sample)
        log_params = stats.lognorm.fit(sample, floc=0)
        ll_norm = float(np.sum(stats.norm.logpdf(sample, *norm_params)))
        ll_log = float(np.sum(stats.lognorm.logpdf(sample, *log_params)))
    except (ValueError, FloatingPointError):
        return _fact(
            label,
            "inconclusive",
            test="skewtest_and_aic",
            rule="n >= 100, positive skew at 5%, lognormal AIC beats normal",
            n=n,
            reason="fit_failed",
            skew=skewness,
        )
    if not math.isfinite(ll_norm) or not math.isfinite(ll_log) or skew_p is None:
        return _fact(
            label,
            "inconclusive",
            test="skewtest_and_aic",
            rule="n >= 100, positive skew at 5%, lognormal AIC beats normal",
            n=n,
            reason="non_finite",
            skew=skewness,
        )
    aic_norm = 4.0 - 2.0 * ll_norm
    aic_log = 4.0 - 2.0 * ll_log
    positive = skewness > 0.0 and skew_p < FAT_TAIL_P
    lognormal_better = aic_log < aic_norm
    status = "pass" if positive and lognormal_better else "fail"
    return _fact(
        label,
        status,
        test="skewtest_and_aic",
        rule="n >= 100, positive skew at 5%, lognormal AIC beats normal",
        n=n,
        skew=skewness,
        skew_pvalue=skew_p,
        aic_normal=aic_norm,
        aic_lognormal=aic_log,
    )


def _returns_ready(returns: np.ndarray) -> np.ndarray:
    values = np.asarray(returns, dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def validate_stylized_facts(
    returns: np.ndarray,
    spreads: np.ndarray,
    depths: np.ndarray,
    impact: dict[str, object] | None = None,
) -> dict[str, object]:
    """Apply the pre-registered tests. Does not rerun or retune the tape."""
    series = _returns_ready(returns)
    n = int(series.size)
    facts: dict[str, object] = {}

    if n < MIN_RETURNS:
        for name, test, rule in (
            ("fat_tails", "jarque_bera", "p < 0.05 and excess kurtosis > 1"),
            ("volatility_clustering", "engle_arch_lm", "p < 0.05 at 5 lags"),
            (
                "no_return_autocorrelation",
                "ljung_box",
                "do not reject signed-return Ljung-Box at 5%, lag 10",
            ),
        ):
            facts[name] = _fact(name, "inconclusive", test=test, rule=rule, n=n, reason="sample")
    else:
        jb = jarque_bera(series)
        kurtosis = _finite(jb["excess_kurtosis"])
        jb_p = _finite(jb["pvalue"])
        if kurtosis is None or jb_p is None:
            facts["fat_tails"] = _fact(
                "fat_tails",
                "inconclusive",
                test="jarque_bera",
                rule="p < 0.05 and excess kurtosis > 1",
                n=n,
                reason="non_finite",
            )
        else:
            passed = jb_p < FAT_TAIL_P and kurtosis > FAT_TAIL_KURTOSIS
            facts["fat_tails"] = _fact(
                "fat_tails",
                "pass" if passed else "fail",
                test="jarque_bera",
                rule="p < 0.05 and excess kurtosis > 1",
                n=n,
                stat=_finite(jb["stat"]),
                pvalue=jb_p,
                excess_kurtosis=kurtosis,
                skew=_finite(jb["skew"]),
            )
        arch = engle_arch_lm(series, ARCH_LAGS)
        arch_p = _finite(arch["pvalue"])
        if arch_p is None:
            facts["volatility_clustering"] = _fact(
                "volatility_clustering",
                "inconclusive",
                test="engle_arch_lm",
                rule="p < 0.05 at 5 lags",
                n=n,
                reason="non_finite",
            )
        else:
            facts["volatility_clustering"] = _fact(
                "volatility_clustering",
                "pass" if arch_p < ARCH_P else "fail",
                test="engle_arch_lm",
                rule="p < 0.05 at 5 lags",
                n=n,
                stat=_finite(arch["stat"]),
                pvalue=arch_p,
            )
        try:
            lb = ljung_box(series, LJUNG_LAG)
            lb_p = _finite(lb["pvalue"])
            rho1 = abs(float(autocorrelation(series, 1)[0]))
        except ValueError as exc:
            facts["no_return_autocorrelation"] = _fact(
                "no_return_autocorrelation",
                "inconclusive",
                test="ljung_box",
                rule="do not reject signed-return Ljung-Box at 5%, lag 10",
                n=n,
                reason=str(exc),
            )
            lb_p = None
            rho1 = float("nan")
        if "no_return_autocorrelation" in facts:
            pass
        elif lb_p is None:
            facts["no_return_autocorrelation"] = _fact(
                "no_return_autocorrelation",
                "inconclusive",
                test="ljung_box",
                rule="do not reject signed-return Ljung-Box at 5%, lag 10",
                n=n,
                reason="non_finite",
                abs_rho1=_finite(rho1),
            )
        else:
            facts["no_return_autocorrelation"] = _fact(
                "no_return_autocorrelation",
                "pass" if lb_p >= LJUNG_P else "fail",
                test="ljung_box",
                rule="do not reject signed-return Ljung-Box at 5%, lag 10",
                n=n,
                stat=_finite(lb["stat"]),
                pvalue=lb_p,
                abs_rho1=_finite(rho1),
            )

    if n < MIN_DFA:
        facts["long_memory_absolute_returns"] = _fact(
            "long_memory_absolute_returns",
            "inconclusive",
            test="dfa_hurst_and_ljung_box",
            rule="H(|r|) >= 0.60, H(r) in [0.35, 0.65], Ljung-Box on |r| rejects at 5%",
            n=n,
            reason="sample",
        )
    else:
        try:
            hurst_abs = float(dfa_hurst(np.abs(series))["hurst"])
            hurst_signed = float(dfa_hurst(series)["hurst"])
            lb_abs = ljung_box(np.abs(series), LJUNG_LAG)
            lb_abs_p = _finite(lb_abs["pvalue"])
        except ValueError as exc:
            facts["long_memory_absolute_returns"] = _fact(
                "long_memory_absolute_returns",
                "inconclusive",
                test="dfa_hurst_and_ljung_box",
                rule="H(|r|) >= 0.60, H(r) in [0.35, 0.65], Ljung-Box on |r| rejects at 5%",
                n=n,
                reason=str(exc),
            )
        else:
            if lb_abs_p is None or not math.isfinite(hurst_abs) or not math.isfinite(hurst_signed):
                status = "inconclusive"
                reason = "non_finite"
            else:
                passed = (
                    hurst_abs >= HURST_ABS_MIN
                    and HURST_SIGNED_LO <= hurst_signed <= HURST_SIGNED_HI
                    and lb_abs_p < LJUNG_P
                )
                status = "pass" if passed else "fail"
                reason = ""
            facts["long_memory_absolute_returns"] = _fact(
                "long_memory_absolute_returns",
                status,
                test="dfa_hurst_and_ljung_box",
                rule="H(|r|) >= 0.60, H(r) in [0.35, 0.65], Ljung-Box on |r| rejects at 5%",
                n=n,
                reason=reason,
                hurst_absolute=_finite(hurst_abs),
                hurst_signed=_finite(hurst_signed),
                absolute_ljung_box_pvalue=lb_abs_p,
            )

    spread_fact = _shape_series(spreads, "spread")
    depth_fact = _shape_series(depths, "depth")
    statuses = {str(spread_fact["status"]), str(depth_fact["status"])}
    if "inconclusive" in statuses:
        shape_status = "inconclusive"
    elif statuses == {"pass"}:
        shape_status = "pass"
    else:
        shape_status = "fail"
    facts["spread_and_depth_shape"] = _fact(
        "spread_and_depth_shape",
        shape_status,
        test="skewtest_and_aic",
        rule="both spread and depth: positive skew at 5% and lognormal AIC beats normal",
        n=_fact_n(spread_fact) + _fact_n(depth_fact),
        spread=spread_fact,
        depth=depth_fact,
    )

    if impact is None:
        facts["square_root_impact"] = _fact(
            "square_root_impact",
            "inconclusive",
            test="log_log_ols",
            rule="95% CI contains 0.5 and excludes 0, with at least 12 positive points",
            n=0,
            reason="not_measured",
        )
    else:
        fit = impact.get("fit")
        if not isinstance(fit, dict):
            facts["square_root_impact"] = _fact(
                "square_root_impact",
                "inconclusive",
                test="log_log_ols",
                rule="95% CI contains 0.5 and excludes 0, with at least 12 positive points",
                n=0,
                reason="missing_fit",
            )
        else:
            facts["square_root_impact"] = _fact(
                "square_root_impact",
                str(fit.get("status", "inconclusive")),
                test="log_log_ols",
                rule="95% CI contains 0.5 and excludes 0, with at least 12 positive points",
                n=int(fit.get("n") or 0),
                reason=str(fit.get("reason") or ""),
                slope=fit.get("slope"),
                intercept=fit.get("intercept"),
                ci_low=fit.get("ci_low"),
                ci_high=fit.get("ci_high"),
                r_squared=fit.get("r_squared"),
                n_attempted=impact.get("n_attempted"),
                n_dropped=impact.get("n_dropped"),
                n_positive=impact.get("n_positive"),
                drop_reasons=impact.get("drop_reasons"),
            )

    report: dict[str, object] = dict(EVIDENCE)
    report.update(
        {
            "pvalue_note": PVALUE_NOTE,
            "thresholds": {
                "min_returns": MIN_RETURNS,
                "min_dfa": MIN_DFA,
                "min_shape": MIN_SHAPE,
                "fat_tail_p": FAT_TAIL_P,
                "fat_tail_excess_kurtosis": FAT_TAIL_KURTOSIS,
                "arch_lags": ARCH_LAGS,
                "ljung_lag": LJUNG_LAG,
                "hurst_absolute_min": HURST_ABS_MIN,
                "hurst_signed_lo": HURST_SIGNED_LO,
                "hurst_signed_hi": HURST_SIGNED_HI,
            },
            "facts": facts,
        }
    )
    return report


def run_stylized_validation(
    cfg: EcologyConfig | None = None,
    *,
    impact: dict[str, object] | None = None,
    measure: bool = True,
) -> dict[str, object]:
    """Run the pre-registered ecology and attach the impact experiment."""
    config = cfg or validation_config()
    tape: SimResult = run_ecology(config)
    if impact is None and measure:
        impact = measure_impact()
    report = validate_stylized_facts(tape.returns, tape.spreads, tape.depths, impact)
    report["tape"] = tape.summary()
    report["config_seed"] = config.seed
    report["config_max_events"] = config.max_events
    return report

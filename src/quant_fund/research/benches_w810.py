"""Benchmark batteries for SOTA canon waves 8–10 (anytime-valid inference,
multivariate proper scores, time-series conformal, regime-conditional
evaluation, the leaky-oracle red team, and distributional-ML baselines).

Every battery here runs on seeded SYNTHETIC streams only — no panel or
vendor data is consumed, and no payload contains headline performance
ratios. Each bench returns a flat dict of proper diagnostic statistics
(conformal coverage, pinball/CRPS improvements, FDR control rates,
detection delays), or ``{}`` if its synthetic setup cannot be constructed.
"""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from quant_fund.metrics.anytime_fdr import (
    ELond,
    ELord,
    ESaffron,
    e_bh,
    stopped_e_bh,
)
from quant_fund.metrics.conformal_martingale import WatchMonitor
from quant_fund.metrics.e_detectors import EDetectorGaussian, run_detector
from quant_fund.metrics.energy_score import energy_score
from quant_fund.metrics.scoring import crps_gaussian
from quant_fund.models.enbpi import EnbPI, EnbPIResult
from quant_fund.models.ngboost_lite import NGBoostGaussian
from quant_fund.models.qrf import QuantileRegressionForest
from quant_fund.validation.leakage_redteam import (
    run_leaky_oracle_study,
    structural_lookahead_audit,
    trial_count_deflation,
)
from quant_fund.validation.regime_eval import (
    regime_conditional_summary,
    regime_eval_gate,
)

_SEED = 20260927


def bench_anytime_valid() -> dict[str, float]:
    """Anytime-valid inference battery (wave 8).

    Gaussian location e-variables exp(Z - 1/2) are exact null e-values
    (E=1); signal uses Z ~ N(1.5, 1). e-BH/stopped e-BH (Wang & Ramdas
    2022/2025), e-LOND (Xu & Ramdas 2024), e-LORD/e-SAFFRON e-GAI
    alpha-investing (Zhang, Wei, Ren & Zou 2025), mixture-SR e-detector
    (Shin, Ramdas & Rinaldo 2023).
    """
    try:
        rng = np.random.default_rng(_SEED)
        n_null, n_signal, horizon = 40, 10, 30

        def _evalues(means: np.ndarray) -> np.ndarray:
            z = rng.standard_normal(means.size) + means
            return np.asarray(np.exp(z - 0.5), dtype=float)

        nulls = _evalues(np.zeros(n_null))
        signals = _evalues(np.full(n_signal, 1.5))
        # Floor the planted signals: one-shot e-BH needs e-values at or beyond
        # the n/(alpha*k) critical value (the harness's own anytime_fdr tests
        # use the same 1e6 floor for planted signals).
        signals = np.maximum(signals, 1e6)
        battery = np.concatenate([nulls, signals])
        alpha = 0.05
        res = e_bh(battery, alpha)
        # stopped variant: cumulative-product streams evaluated at t=horizon
        z_paths = rng.standard_normal((n_null + n_signal, horizon))
        z_paths[n_null:, :] += 1.5
        paths = np.exp(np.cumsum(z_paths - 0.5, axis=1))
        stopped = stopped_e_bh(paths, alpha)
        lond = ELond(alpha)
        lond_rejects = sum(int(lond.submit(e)) for e in battery)
        # e-GAI (Zhang, Wei, Ren & Zou 2025): data-driven alpha-investing
        # levels; omega1 ~ 1/T with T = stream length.
        t_stream = n_null + n_signal
        lord = ELord(alpha, omega1=1.0 / t_stream)
        lord_rejects = sum(int(lord.submit(e)) for e in battery)
        saffron = ESaffron(alpha, lam=0.1, omega1=1.0 / t_stream)
        saffron_rejects = sum(int(saffron.submit(e)) for e in battery)
        # empirical FDR under the global null (seeded MC)
        fdr_hits = 0
        saffron_fa_hits = 0
        reps = 200
        for _ in range(reps):
            if e_bh(_evalues(np.zeros(n_null)), alpha).num_rejected > 0:
                fdr_hits += 1
            null_proc = ESaffron(alpha, lam=0.1, omega1=1.0 / n_null)
            if any(null_proc.submit(float(e)) for e in _evalues(np.zeros(n_null))):
                saffron_fa_hits += 1
        # e-detector: 2-sigma mean shift, plus null FA rate
        shifted = np.concatenate([rng.standard_normal(600), rng.standard_normal(400) + 2.0])
        det = run_detector(EDetectorGaussian(sigma=1.0, alpha=alpha), shifted, alpha)
        fa_hits = 0
        fa_reps = 100
        for _ in range(fa_reps):
            null_stream = rng.standard_normal(600)
            if (
                run_detector(
                    EDetectorGaussian(sigma=1.0, alpha=alpha), null_stream, alpha
                ).alarm_time
                is not None
            ):
                fa_hits += 1
        return {
            "alpha": alpha,
            "n_null": float(n_null),
            "n_signal": float(n_signal),
            "e_bh_rejected": float(res.num_rejected),
            "stopped_e_bh_rejected": float(stopped.num_rejected),
            "elond_rejected": float(lond_rejects),
            "elord_rejected": float(lord_rejects),
            "esaffron_rejected": float(saffron_rejects),
            "esaffron_null_fa_rate": saffron_fa_hits / reps,
            "esaffron_remaining_wealth": float(saffron.remaining_wealth),
            "null_fdr_empirical": fdr_hits / reps,
            "detector_alarm_time": float(det.alarm_time if det.alarm_time is not None else -1.0),
            "detector_null_fa_rate": fa_hits / fa_reps,
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return {}


def bench_energy_score() -> dict[str, float]:
    """Multivariate energy score propriety gap (wave 8; Szekely 2003,
    Gneiting & Raftery 2007). Sharp ensemble must score below the
    over-dispersed one averaged over seeded observations."""
    try:
        rng = np.random.default_rng(_SEED)
        n_samples, dim, n_obs = 2000, 4, 64
        sharp = rng.standard_normal((n_samples, dim))
        wide = rng.standard_normal((n_samples, dim)) * 3.0
        obs = rng.standard_normal((n_obs, dim))
        sharp_mean = float(np.mean([energy_score(sharp, o) for o in obs]))
        wide_mean = float(np.mean([energy_score(wide, o) for o in obs]))
        return {
            "dim": float(dim),
            "n_samples": float(n_samples),
            "sharp_mean": sharp_mean,
            "wide_mean": wide_mean,
            "propriety_gap": wide_mean - sharp_mean,
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return {}


def _ar1_stream(seed: int, t_total: int, sigma_break: float | None) -> np.ndarray:
    """SYNTHETIC AR(1) stream (phi=0.6), optional mid-stream volatility break."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(t_total)
    sigma = np.ones(t_total)
    if sigma_break is not None:
        sigma[t_total // 2 :] = sigma_break
    y = np.zeros(t_total)
    for t in range(1, t_total):
        y[t] = 0.6 * y[t - 1] + sigma[t] * eps[t]
    return y


def _enbpi_online(y: np.ndarray, *, split: int, alpha: float) -> EnbPIResult:
    """EnbPI on the lagged-level design (predict y_t from y_{t-1}, y_{t-2}).

    A shallow tree keeps the battery about the conformal layer rather than
    about base-learner strength.
    """
    idx = np.arange(2, y.size)
    features = np.column_stack([y[idx - 1], y[idx - 2]])
    target = y[idx]
    model = EnbPI(
        lambda: DecisionTreeRegressor(max_depth=3, random_state=0),
        n_estimators=20,
        alpha=alpha,
        seed=7,
    )
    model.fit(features[:split], target[:split])
    return model.predict_online(features[split:], target[split:])


def bench_ts_conformal() -> dict[str, float]:
    """Time-series conformal battery (waves 9-10).

    Two EnbPI readings on SYNTHETIC AR(1) streams (Xu & Xie 2021/2023): the
    marginal-coverage guarantee is asserted on a stationary stream, and the
    same measurement is repeated after a mid-stream volatility break, where
    the paper's mixing assumption no longer holds and coverage is expected to
    degrade — reported as a diagnostic, never asserted to hold. Plus WATCH
    exchangeability monitoring on a mid-stream conformity shift (Prinster,
    Han & Saria 2025).
    """
    try:
        alpha = 0.2  # central 80% interval
        nominal = 1.0 - alpha
        split = 318
        t_total = 520
        stationary = _enbpi_online(_ar1_stream(_SEED, t_total, None), split=split, alpha=alpha)
        broken = _enbpi_online(_ar1_stream(_SEED + 1, t_total, 2.5), split=split, alpha=alpha)
        # WATCH: conformity scores, scale doubles mid-stream
        rng = np.random.default_rng(_SEED + 2)
        s = np.abs(rng.standard_normal(800))
        s[500:] *= 2.5
        watch = WatchMonitor(alpha=0.05).run(s)
        alarm_time = float(watch.alarms[0]) if watch.alarms else -1.0
        return {
            "enbpi_nominal_central": nominal,
            "enbpi_central_coverage": float(stationary.coverage),
            "enbpi_abs_coverage_error": float(abs(stationary.coverage - nominal)),
            "enbpi_vol_break_coverage_error": float(abs(broken.coverage - nominal)),
            "enbpi_mean_width": float(stationary.mean_width),
            "enbpi_n_test": float(t_total - 2 - split),
            "watch_detected": 1.0 if watch.alarms else 0.0,
            "watch_alarm_time": alarm_time,
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return {}


def bench_regime_eval() -> dict[str, float]:
    """Regime-conditional evaluation gate (wave 9): homogeneous regimes must
    pass the fail-closed gate; heterogeneous ones must fail it."""
    try:
        rng = np.random.default_rng(_SEED)
        n = 600
        regimes = np.repeat([0, 1, 2], n // 3)
        dates = np.arange(n)
        homo = rng.standard_normal(n) * 0.2 + 0.1
        hetero = homo + np.where(regimes == 1, 0.6, 0.0) - np.where(regimes == 2, 0.5, 0.0)
        summary_h = regime_conditional_summary(dates, regimes, homo)
        summary_x = regime_conditional_summary(dates, regimes, hetero)
        gate_h = regime_eval_gate(summary_h)
        gate_x = regime_eval_gate(summary_x)
        return {
            "n": float(n),
            "n_regimes": 3.0,
            "homo_gate_passed": 1.0 if gate_h.passed else 0.0,
            "hetero_gate_failed": 0.0 if gate_x.passed else 1.0,
            "homo_pooled_mean": float(summary_h.pooled_mean),
            "hetero_regime_mean_cv": float(
                np.std(summary_x.means) / max(abs(np.mean(summary_x.means)), 1e-12)
            ),
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return {}


def bench_leakage_redteam() -> dict[str, float]:
    """Leaky-oracle red team (wave 10; Gencay 2026): the structural audit
    passes the house registry and flags planted look-ahead names; the Monte
    Carlo contrast regression-locks the DSR blind spot on SYNTHETIC data."""
    try:
        clean = structural_lookahead_audit(["mom_12_1", "vol_20d", "basis_zscore", "turnover_adv"])
        planted = structural_lookahead_audit(["mom_12_1", "future_return_5d", "lead_1_volume"])
        study = run_leaky_oracle_study(leakage_levels=(0.6,), n_mc=60, seed=_SEED)
        return {
            "audit_clean_passed": 0.0 if clean.fail else 1.0,
            "audit_planted_flagged": 1.0 if planted.fail else 0.0,
            "deflated_alpha_trials10": trial_count_deflation(10, 0.05),
            "leaky_promotion_rate": float(study.promotion_rate_leaky[0]),
            "control_promotion_rate": float(study.promotion_rate_control[0]),
            "blindspot_contrast": float(study.blind_spot_contrast[0]),
            "data_source_syn": 1.0,
        }
    except (ValueError, RuntimeError, FloatingPointError, AttributeError, KeyError):
        return {}


def bench_distributional_ml() -> dict[str, float]:
    """Distributional-ML baselines (wave 9): NGBoostLite must beat a global
    constant location-scale baseline under CRPS; the quantile regression
    forest median must track the conditional median."""
    try:
        rng = np.random.default_rng(_SEED)
        n = 160
        x = rng.uniform(-1.0, 1.0, (n, 3))
        mu = np.sin(3.0 * x[:, 0]) + 0.5 * x[:, 1]
        sig = 0.3 + 0.4 * np.abs(x[:, 2])
        y = mu + sig * rng.standard_normal(n)
        model = NGBoostGaussian(n_estimators=80, max_depth=2, seed=3, score="crps").fit(x, y)
        mu_hat, sig_hat = model.predict_params(x)
        ngboost_crps = float(np.mean(crps_gaussian(y, mu_hat, sig_hat)))
        base_crps = float(np.mean(crps_gaussian(y, np.full(n, y.mean()), np.full(n, y.std()))))
        n2 = 300
        x2 = rng.uniform(0.0, 1.0, (n2, 2))
        y2 = 2.0 * x2[:, 0] + rng.standard_normal(n2) * (0.2 + x2[:, 1])
        qrf = QuantileRegressionForest(n_estimators=40, min_samples_leaf=5, seed=5).fit(x2, y2)
        med = qrf.predict_quantiles(x2, np.array([0.5]))[:, 0]
        corr = float(np.corrcoef(med, 2.0 * x2[:, 0])[0, 1])
        return {
            "ngboost_crps": ngboost_crps,
            "constant_baseline_crps": base_crps,
            "crps_improvement": base_crps - ngboost_crps,
            "qrf_median_corr": corr,
            "n_train": float(n),
            "n_train_qrf": float(n2),
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return {}

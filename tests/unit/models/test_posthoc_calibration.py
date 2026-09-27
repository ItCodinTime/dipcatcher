"""Post-hoc distributional recalibration: recovery, CRPS/PIT gains, fail-closed.

All experiments are synthetic correctness checks with seeded rngs — never
market evidence (lab honesty contract).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.metrics.scoring import crps_gaussian, crps_student_t
from quant_fund.models.posthoc_calibration import (
    IsotonicQuantileCalibrator,
    QuantileMappingCalibrator,
    VarianceScalingGaussian,
    VarianceScalingStudentT,
)

Array = np.ndarray

LEVELS = np.array([0.05, 0.25, 0.50, 0.75, 0.95])
Z_LEVELS = norm.ppf(LEVELS)


def _gaussian_pair(
    seed: int, n_cal: int, n_eval: int, scale_pred: float
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Calibration and eval draws with predictions off by ``scale_pred``."""
    rng = np.random.default_rng(seed)
    mu_c = rng.normal(0.0, 1.0, n_cal)
    y_c = mu_c + rng.normal(0.0, 1.0, n_cal)
    mu_e = rng.normal(0.0, 1.0, n_eval)
    y_e = mu_e + rng.normal(0.0, 1.0, n_eval)
    return mu_c, np.full(n_cal, scale_pred), y_c, mu_e, np.full(n_eval, scale_pred), y_e


def test_variance_scaling_recovers_inflated_scale() -> None:
    mu_c, sig_c, y_c, _, _, _ = _gaussian_pair(1, 2000, 500, scale_pred=2.0)
    cal = VarianceScalingGaussian().fit(mu_c, sig_c, y_c)
    assert cal.scale_ is not None
    assert cal.scale_ == pytest.approx(0.5, rel=0.10)


def test_variance_scaling_recovers_deflated_scale() -> None:
    mu_c, sig_c, y_c, _, _, _ = _gaussian_pair(2, 2000, 500, scale_pred=0.5)
    cal = VarianceScalingGaussian().fit(mu_c, sig_c, y_c)
    assert cal.scale_ is not None
    assert cal.scale_ == pytest.approx(2.0, rel=0.10)


def test_variance_scaling_crps_improves_over_and_under_dispersed() -> None:
    for seed, scale_pred in ((3, 2.0), (4, 0.5)):
        mu_c, sig_c, y_c, mu_e, sig_e, y_e = _gaussian_pair(seed, 2000, 2000, scale_pred)
        cal = VarianceScalingGaussian().fit(mu_c, sig_c, y_c)
        out = cal.transform(mu_e, sig_e)
        before = float(np.mean(crps_gaussian(y_e, mu_e, sig_e)))
        after = float(np.mean(crps_gaussian(y_e, out.mu, out.sigma)))
        assert after < before


def test_student_t_variance_scaling_improves_crps() -> None:
    rng = np.random.default_rng(5)
    n_cal, n_eval, df = 2000, 2000, 6.0
    mu_c = rng.normal(0.0, 1.0, n_cal)
    y_c = mu_c + rng.standard_t(df, n_cal) * np.sqrt((df - 2.0) / df)
    sig_c = np.full(n_cal, 2.5)
    mu_e = rng.normal(0.0, 1.0, n_eval)
    y_e = mu_e + rng.standard_t(df, n_eval) * np.sqrt((df - 2.0) / df)
    sig_e = np.full(n_eval, 2.5)
    cal = VarianceScalingStudentT(df=df).fit(mu_c, sig_c, y_c)
    out = cal.transform(mu_e, sig_e)
    assert out.df == df
    before = float(np.mean(crps_student_t(y_e, mu_e, sig_e, nu=df)))
    after = float(np.mean(crps_student_t(y_e, out.mu, out.sigma, nu=df)))
    assert after < before


def test_pit_coverage_closer_to_nominal_after_scaling() -> None:
    mu_c, sig_c, y_c, mu_e, sig_e, y_e = _gaussian_pair(6, 3000, 3000, scale_pred=2.0)
    cal = VarianceScalingGaussian().fit(mu_c, sig_c, y_c)
    out = cal.transform(mu_e, sig_e)
    z25, z75 = norm.ppf(0.25), norm.ppf(0.75)
    cov_before = float(np.mean((y_e >= mu_e + z25 * sig_e) & (y_e <= mu_e + z75 * sig_e)))
    cov_after = float(
        np.mean((y_e >= out.mu + z25 * out.sigma) & (y_e <= out.mu + z75 * out.sigma))
    )
    assert abs(cov_after - 0.5) < abs(cov_before - 0.5)
    assert cov_after == pytest.approx(0.5, abs=0.06)


def test_quantile_mapping_recovers_linear_distortion() -> None:
    # Predicted grid is the monotone distortion q_pred = (q_true - 1) / 2 of
    # the N(0,1) quantile curve; the calibrator should invert it exactly.
    rng = np.random.default_rng(7)
    n = 4000
    y_cal = rng.normal(0.0, 1.0, n)
    q_pred = np.tile((Z_LEVELS - 1.0) / 2.0, (n, 1))
    cal = QuantileMappingCalibrator(LEVELS).fit(q_pred, y_cal)
    q_eval = np.tile((Z_LEVELS - 1.0) / 2.0, (500, 1))
    corrected = cal.transform(q_eval)
    np.testing.assert_allclose(corrected[0], Z_LEVELS, atol=0.10)


def _piecewise_g(x: Array) -> Array:
    return np.where(x <= 0.0, x, np.where(x <= 1.0, 2.0 * x, x + 1.0))


def test_isotonic_recovers_piecewise_monotone_distortion() -> None:
    rng = np.random.default_rng(8)
    n = 4000
    x = rng.normal(0.0, 1.0, n)
    y = _piecewise_g(x) + rng.normal(0.0, 0.05, n)
    # Identity location prediction with modest spread: q[i, tau] = x_i + z_tau*0.5.
    q_pred = x.reshape(-1, 1) + Z_LEVELS.reshape(1, -1) * 0.5
    cal = IsotonicQuantileCalibrator(LEVELS).fit(q_pred, y)
    mid = int(np.argmin(np.abs(LEVELS - 0.5)))
    probes = np.array([-1.5, -0.5, 0.5, 0.75, 1.5, 2.0])
    probe_rows = probes.reshape(-1, 1) + Z_LEVELS.reshape(1, -1) * 0.5
    corrected = cal.transform(probe_rows)
    np.testing.assert_allclose(corrected[:, mid], _piecewise_g(probes), atol=0.20)


def test_quantile_mapping_output_non_crossing() -> None:
    rng = np.random.default_rng(9)
    n = 500
    y_cal = rng.normal(0.0, 1.0, n)
    q_pred = np.tile(Z_LEVELS, (n, 1)) + rng.normal(0.0, 0.1, (n, 5))
    cal = QuantileMappingCalibrator(LEVELS).fit(q_pred, y_cal)
    corrected = cal.transform(q_pred[:50])
    assert np.all(np.diff(corrected, axis=1) >= -1e-12)


def test_isotonic_output_non_crossing() -> None:
    rng = np.random.default_rng(10)
    n = 2000
    x = rng.normal(0.0, 1.0, n)
    y = _piecewise_g(x) + rng.normal(0.0, 0.05, n)
    q_pred = x.reshape(-1, 1) + Z_LEVELS.reshape(1, -1) * 0.5
    cal = IsotonicQuantileCalibrator(LEVELS).fit(q_pred, y)
    corrected = cal.transform(q_pred[:100])
    assert np.all(np.diff(corrected, axis=1) >= -1e-12)


def test_variance_scaling_output_sigma_positive() -> None:
    mu_c, sig_c, y_c, mu_e, sig_e, _ = _gaussian_pair(11, 500, 100, scale_pred=2.0)
    out = VarianceScalingGaussian().fit(mu_c, sig_c, y_c).transform(mu_e, sig_e)
    assert np.all(out.sigma > 0.0)
    assert out.df is None


def test_fail_closed_edges() -> None:
    rng = np.random.default_rng(12)
    n = 100
    mu = rng.normal(0.0, 1.0, n)
    sig = np.ones(n)
    y = mu + rng.normal(0.0, 1.0, n)
    q = np.tile(Z_LEVELS, (n, 1))

    with pytest.raises(ValueError):
        VarianceScalingGaussian().fit(mu[:29], sig[:29], y[:29])
    with pytest.raises(ValueError):
        QuantileMappingCalibrator(LEVELS).fit(q[:29], y[:29])
    with pytest.raises(ValueError):
        IsotonicQuantileCalibrator(LEVELS).fit(q[:29], y[:29])
    with pytest.raises(ValueError):
        QuantileMappingCalibrator(np.array([0.5, 0.25, 0.75])).fit(q, y)
    with pytest.raises(ValueError):
        QuantileMappingCalibrator(np.array([0.0, 0.5, 1.0])).fit(q, y)
    with pytest.raises(ValueError):
        QuantileMappingCalibrator(np.array([0.5])).fit(q, y)
    with pytest.raises(ValueError):
        VarianceScalingStudentT(df=2.0)
    with pytest.raises(ValueError):
        VarianceScalingStudentT(df=float("nan"))

    y_bad = y.copy()
    y_bad[0] = np.nan
    with pytest.raises(ValueError):
        VarianceScalingGaussian().fit(mu, sig, y_bad)
    sig_bad = sig.copy()
    sig_bad[0] = 0.0
    with pytest.raises(ValueError):
        VarianceScalingGaussian().fit(mu, sig_bad, y)
    mu_bad = mu.copy()
    mu_bad[0] = np.inf
    with pytest.raises(ValueError):
        VarianceScalingGaussian().fit(mu_bad, sig, y)
    q_bad = q.copy()
    q_bad[0, 0] = -np.inf
    with pytest.raises(ValueError):
        QuantileMappingCalibrator(LEVELS).fit(q_bad, y)
    with pytest.raises(ValueError):
        IsotonicQuantileCalibrator(LEVELS).fit(q_bad, y)
    with pytest.raises(RuntimeError):
        VarianceScalingGaussian().transform(mu, sig)  # not fitted
    with pytest.raises(RuntimeError):
        QuantileMappingCalibrator(LEVELS).transform(q)  # not fitted
    with pytest.raises(RuntimeError):
        IsotonicQuantileCalibrator(LEVELS).transform(q)  # not fitted


def test_isotonic_fit_split_requires_enough_rows() -> None:
    rng = np.random.default_rng(13)
    n = 40  # 70% of 40 = 28 < 30
    x = rng.normal(0.0, 1.0, n)
    y = x + rng.normal(0.0, 0.1, n)
    q = x.reshape(-1, 1) + Z_LEVELS.reshape(1, -1) * 0.1
    with pytest.raises(ValueError):
        IsotonicQuantileCalibrator(LEVELS).fit(q, y)

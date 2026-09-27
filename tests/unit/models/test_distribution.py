"""Tests for distributional shape and normality battery."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.distribution import (
    anderson_darling_normal,
    cramervonmises_normal,
    dagostino_k2,
    lilliefors,
    mardia_test,
    medcouple,
    normality_battery,
    pearson_chi2_normal,
    qn_scale,
    robust_shape,
    shapiro_wilk,
)


class TestUnivariateTests:
    def test_normal_accepted(self):
        rng = np.random.default_rng(0)
        v = rng.normal(size=400)
        assert shapiro_wilk(v)["pvalue"] > 0.05
        assert anderson_darling_normal(v)["reject_5pct"] == 0.0
        assert cramervonmises_normal(v)["pvalue"] > 0.05
        assert dagostino_k2(v)["pvalue"] > 0.05
        assert pearson_chi2_normal(v)["pvalue"] > 0.01
        assert lilliefors(v)["pvalue"] > 0.05

    def test_fat_tails_rejected(self):
        rng = np.random.default_rng(1)
        v = rng.standard_t(df=3, size=400) * 0.01
        assert shapiro_wilk(v)["pvalue"] < 0.01
        assert anderson_darling_normal(v)["reject_5pct"] == 1.0
        assert dagostino_k2(v)["pvalue"] < 0.01
        assert lilliefors(v)["pvalue"] < 0.05

    def test_skewed_rejected(self):
        rng = np.random.default_rng(2)
        v = rng.exponential(size=300)
        assert dagostino_k2(v)["pvalue"] < 0.01
        assert cramervonmises_normal(v)["pvalue"] < 0.01

    def test_battery(self):
        rng = np.random.default_rng(3)
        normal = normality_battery(rng.normal(size=300))
        assert normal["normal_5pct"] == 1.0
        fat = normality_battery(rng.standard_t(df=3, size=300))
        assert fat["normal_5pct"] == 0.0

    def test_failclosed(self):
        with pytest.raises(ValueError):
            shapiro_wilk(np.ones(2))
        with pytest.raises(ValueError):
            pearson_chi2_normal(np.ones(50), n_bins=30)


class TestMardia:
    def test_mvn_accepted(self):
        rng = np.random.default_rng(4)
        m = rng.normal(size=(400, 4))
        out = mardia_test(m)
        assert out["skew_pvalue"] > 0.05
        assert out["kurt_pvalue"] > 0.05

    def test_mvn_tails_rejected(self):
        rng = np.random.default_rng(5)
        m = rng.standard_t(df=3, size=(500, 3))
        out = mardia_test(m)
        assert out["kurt_pvalue"] < 0.01

    def test_failclosed(self):
        with pytest.raises(ValueError):
            mardia_test(np.ones((10, 3)))


class TestRobust:
    def test_medcouple_symmetric(self):
        rng = np.random.default_rng(6)
        v = rng.normal(size=200)
        assert abs(medcouple(v)) < 0.15

    def test_medcouple_skewed(self):
        rng = np.random.default_rng(7)
        v = rng.lognormal(0, 0.5, 300)
        assert medcouple(v) > 0.05

    def test_qn_scale(self):
        rng = np.random.default_rng(8)
        v = rng.normal(scale=2.0, size=300)
        assert abs(qn_scale(v) - 2.0) < 0.4
        # Robust: a few outliers barely move Qn.
        v2 = v.copy()
        v2[:5] += 100.0
        assert abs(qn_scale(v2) - 2.0) < 0.6

    def test_robust_shape_flags_outliers(self):
        rng = np.random.default_rng(9)
        v = rng.normal(size=200)
        v[10] = 50.0
        out = robust_shape(v)
        assert out["n_outlier_35"] >= 1.0
        assert out["medcouple"] < 0.3

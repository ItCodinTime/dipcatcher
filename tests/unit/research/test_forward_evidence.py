"""Coverage for research.forward_evidence — HAC sample-size planning."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.forward_evidence import evidence_plan, long_run_variance


class TestLongRunVariance:
    def test_white_noise_matches_sample_variance(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 0.01, 500)
        lrv = long_run_variance(x, lag=0)
        assert lrv == pytest.approx(float(np.var(x)), rel=1e-9)

    def test_positive_lag_captures_autocorrelation(self) -> None:
        rng = np.random.default_rng(1)
        n = 400
        shocks = rng.normal(0.0, 1.0, n)
        ar = np.zeros(n)
        for t in range(1, n):
            ar[t] = 0.8 * ar[t - 1] + shocks[t]
        # AR(1) long-run variance exceeds the raw variance.
        assert long_run_variance(ar, lag=20) > float(np.var(ar))

    def test_validation_matrix(self) -> None:
        good = np.zeros(50)
        good[::7] = 0.01
        with pytest.raises(ValueError, match="at least"):
            long_run_variance(np.zeros(10), 0)
        with pytest.raises(ValueError, match="at least"):
            long_run_variance(np.zeros((50, 2)), 0)
        with pytest.raises(ValueError, match="at least"):
            long_run_variance(np.array([np.nan] * 50), 0)
        with pytest.raises(ValueError, match="at least"):
            long_run_variance(good, True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="at least"):
            long_run_variance(good, -1)
        with pytest.raises(ValueError, match="at least"):
            long_run_variance(good, 40)  # need >= 4*(lag+1)

    def test_degenerate_variance_refused(self) -> None:
        with pytest.raises(ValueError, match="degenerate"):
            long_run_variance(np.zeros(50), 0)


class TestEvidencePlan:
    def test_happy_path_shape(self) -> None:
        rng = np.random.default_rng(2)
        calibration = list(rng.normal(0.0, 0.01, 200))
        plan = evidence_plan(calibration, effect_bps=5.0, lag=5)
        assert plan["method"] == "one_sided_normal_approximation_with_Bartlett_HAC"
        assert plan["required_sessions"] >= max(30, 4 * 6)
        assert plan["calibration_n"] == 200
        assert plan["effect_bps"] == 5.0
        assert "assumption" in plan

    def test_larger_effect_needs_fewer_sessions(self) -> None:
        rng = np.random.default_rng(3)
        calibration = list(rng.normal(0.0, 0.01, 200))
        small = evidence_plan(calibration, effect_bps=1.0, lag=0)
        big = evidence_plan(calibration, effect_bps=20.0, lag=0)
        assert big["required_sessions"] < small["required_sessions"]

    def test_floor_of_30(self) -> None:
        rng = np.random.default_rng(4)
        calibration = list(rng.normal(0.0, 0.001, 60))
        plan = evidence_plan(calibration, effect_bps=500.0, lag=0)
        assert plan["required_sessions"] == 30

    def test_parameter_validation(self) -> None:
        calibration = list(np.random.default_rng(5).normal(0, 0.01, 60))
        with pytest.raises(ValueError, match="effect"):
            evidence_plan(calibration, effect_bps=0.0, lag=0)
        with pytest.raises(ValueError, match="effect"):
            evidence_plan(calibration, effect_bps=-1.0, lag=0)
        with pytest.raises(ValueError, match="alpha"):
            evidence_plan(calibration, effect_bps=1.0, lag=0, alpha=0.7)
        with pytest.raises(ValueError, match="alpha"):
            evidence_plan(calibration, effect_bps=1.0, lag=0, alpha=0.0)
        with pytest.raises(ValueError, match="power"):
            evidence_plan(calibration, effect_bps=1.0, lag=0, power=0.4)
        with pytest.raises(ValueError, match="power"):
            evidence_plan(calibration, effect_bps=1.0, lag=0, power=1.0)
        with pytest.raises(ValueError, match="effect"):
            evidence_plan(calibration, effect_bps=float("nan"), lag=0)

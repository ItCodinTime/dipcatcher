"""Tests for changepoint detection (BOCPD, CUSUM, Page–Hinkley, segmentation)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.changepoint import (
    binary_segmentation,
    bocpd_gaussian,
    cusum_detect,
    optimal_partition_mean,
    page_hinkley,
)


def _two_regime(rng: np.random.Generator, n1: int = 200, n2: int = 200) -> np.ndarray:
    return np.concatenate([rng.normal(0.0, 1.0, n1), rng.normal(3.0, 1.0, n2)])


class TestBOCPD:
    def test_detects_mean_shift(self):
        rng = np.random.default_rng(0)
        x = _two_regime(rng)
        out = bocpd_gaussian(x, hazard=1.0 / 200.0)
        cp_prob = out["cp_prob"]
        # Peak changepoint probability should be near index 200 (±20).
        peak = int(np.argmax(cp_prob[80:])) + 80
        assert abs(peak - 200) < 30
        # Posterior changepoint mass concentrates near the true break: a
        # several-fold elevation over the baseline hazard (1/200).
        assert cp_prob[peak] > 4.0 / 200.0

    def test_stationary_series_few_cps(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0.0, 1.0, 300)
        out = bocpd_gaussian(x, hazard=1.0 / 300.0)
        assert len(out["changepoints"]) <= 3
        assert np.all(out["cp_prob"] >= 0.0)

    def test_output_shapes_and_failclosed(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0.0, 1.0, 60)
        out = bocpd_gaussian(x)
        assert out["cp_prob"].shape == (60,)
        assert out["run_length_map"].shape == (60, 61)
        with pytest.raises(ValueError):
            bocpd_gaussian(np.arange(3.0))
        with pytest.raises(ValueError):
            bocpd_gaussian(x, hazard=1.5)
        with pytest.raises(ValueError):
            bocpd_gaussian(x, kappa0=0.0)


class TestCUSUM:
    def test_detects_level_shift(self):
        rng = np.random.default_rng(3)
        x = _two_regime(rng, 150, 150)
        out = cusum_detect(x)
        alarms = out["alarms"]
        assert len(alarms) >= 1
        assert np.any(np.abs(alarms - 150) < 40)

    def test_no_shift_few_alarms(self):
        rng = np.random.default_rng(4)
        x = rng.normal(0.0, 1.0, 400)
        out = cusum_detect(x, threshold=8.0 * x.std(ddof=1))
        assert len(out["alarms"]) <= 2

    def test_failclosed(self):
        with pytest.raises(ValueError):
            cusum_detect(np.arange(4.0))
        with pytest.raises(ValueError):
            cusum_detect(np.random.default_rng(0).normal(size=50), threshold=0.0)


class TestPageHinkley:
    def test_detects_drift(self):
        rng = np.random.default_rng(5)
        x = np.concatenate([rng.normal(0.0, 1.0, 150), rng.normal(1.5, 1.0, 150)])
        out = page_hinkley(x, delta=0.0, lam=3.0 * x.std(ddof=1))
        assert len(out["alarms"]) >= 1

    def test_failclosed(self):
        with pytest.raises(ValueError):
            page_hinkley(np.arange(3.0))
        with pytest.raises(ValueError):
            page_hinkley(np.random.default_rng(0).normal(size=60), lam=-1.0)


class TestSegmentation:
    def test_binary_segmentation_finds_split(self):
        rng = np.random.default_rng(6)
        x = _two_regime(rng, 200, 200)
        cps = binary_segmentation(x, min_size=30)
        assert cps.size >= 1
        assert np.any(np.abs(cps - 200) < 30)

    def test_binary_segmentation_stationary(self):
        rng = np.random.default_rng(7)
        x = rng.normal(0.0, 1.0, 300)
        cps = binary_segmentation(x, min_size=40)
        assert cps.size <= 2

    def test_optimal_partition(self):
        rng = np.random.default_rng(8)
        x = np.concatenate(
            [rng.normal(0.0, 0.5, 60), rng.normal(4.0, 0.5, 60), rng.normal(-2.0, 0.5, 60)]
        )
        cps, obj = optimal_partition_mean(x, penalty=25.0, min_size=5)
        assert cps.size == 2
        assert abs(cps[0] - 60) < 10
        assert abs(cps[1] - 120) < 10
        assert np.isfinite(obj)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            binary_segmentation(np.arange(4.0))
        with pytest.raises(ValueError):
            binary_segmentation(np.random.default_rng(0).normal(size=60), min_size=1)
        with pytest.raises(ValueError):
            optimal_partition_mean(np.random.default_rng(0).normal(size=60), penalty=-5.0)

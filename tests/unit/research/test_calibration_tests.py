"""Tests for metrics/calibration_tests.py — Spiegelhalter's Z."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.calibration_tests import spiegelhalter_z


def test_well_calibrated_not_rejected() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.1, 0.9, 5000)
    y = (rng.random(5000) < p).astype(float)
    out = spiegelhalter_z(p, y)
    assert abs(out["z"]) < 2.5
    assert out["pvalue"] > 0.02


def test_miscalibrated_rejected() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 5000)
    # overconfident forecasts: true probability is shrunk toward 0.5, the
    # miscalibration Spiegelhalter's (1-2p)-weighted Z is designed to detect.
    true_p = 0.5 + 0.5 * (p - 0.5)
    y = (rng.random(5000) < true_p).astype(float)
    out = spiegelhalter_z(p, y)
    assert out["pvalue"] < 0.01


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        spiegelhalter_z(np.array([0.5, 1.2, 0.3, 0.4, 0.5]), np.array([0, 1, 0, 1, 0]))
    with pytest.raises(ValueError):
        spiegelhalter_z(np.full(10, 0.3), np.full(10, 0.5))  # non-binary outcome

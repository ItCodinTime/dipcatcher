"""Tests for metrics/mic.py — Maximal Information Coefficient."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.mic import maximal_information_coefficient


def test_mic_high_for_functional_relationship() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, 800)
    y = np.sin(2.0 * x) + 0.05 * rng.standard_normal(x.size)
    mic = maximal_information_coefficient(x, y)["mic"]
    assert mic > 0.4


def test_mic_low_for_independence() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(800)
    y = rng.standard_normal(800)
    mic = maximal_information_coefficient(x, y)["mic"]
    assert mic < 0.3


def test_mic_in_unit_interval() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal(500)
    y = 2.0 * x + 0.1 * rng.standard_normal(500)
    out = maximal_information_coefficient(x, y)
    assert 0.0 <= out["mic"] <= 1.0
    assert out["mic"] > 0.5  # strong linear dependence


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        maximal_information_coefficient(np.arange(5.0), np.arange(5.0))
    with pytest.raises(ValueError):
        maximal_information_coefficient(np.arange(50.0), np.arange(40.0))

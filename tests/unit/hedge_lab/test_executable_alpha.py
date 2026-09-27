"""Wave 9: executable_alpha / should_trade closed-form + boundary fixtures."""

import math

import pytest

from quant_fund.portfolio.executable_alpha import executable_alpha, should_trade


def test_executable_alpha_closed_form_subtraction() -> None:
    assert executable_alpha(0.10, 0.01, 0.02, 0.03, 0.01, 0.005) == pytest.approx(0.025)
    assert executable_alpha(0.05, 0.0, 0.0, 0.0, 0.0) == pytest.approx(0.05)
    # Costs can push alpha negative
    assert executable_alpha(0.02, 0.01, 0.01, 0.01, 0.01) == pytest.approx(-0.02)


def test_executable_alpha_zero_boundary() -> None:
    assert executable_alpha(0.04, 0.01, 0.01, 0.01, 0.01) == pytest.approx(0.0)


def test_executable_alpha_rejects_non_finite() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            executable_alpha(bad, 0.0, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError, match="finite"):
            executable_alpha(0.1, bad, 0.0, 0.0, 0.0)


def test_should_trade_threshold_boundaries() -> None:
    assert should_trade(0.05, 0.05) is True  # inclusive
    assert should_trade(0.049999, 0.05) is False
    assert should_trade(0.0, 0.0) is True
    assert should_trade(-0.01, 0.0) is False
    assert should_trade(1.0, -1.0) is True


def test_should_trade_non_finite_fails_closed() -> None:
    assert should_trade(float("nan"), 0.0) is False
    assert should_trade(0.1, float("nan")) is False
    assert should_trade(float("inf"), 0.0) is False
    assert should_trade(0.1, float("inf")) is False
    assert should_trade(float("-inf"), -1e9) is False


def test_should_trade_pairs_with_executable_alpha() -> None:
    ea = executable_alpha(0.08, 0.01, 0.01, 0.01, 0.01, uncertainty_penalty=0.02)
    assert ea == pytest.approx(0.02)
    assert should_trade(ea, 0.02) is True
    assert should_trade(ea, 0.021) is False
    assert math.isfinite(ea)

import numpy as np
import pytest

from quant_fund.portfolio.concentration import average_pairwise_corr, effective_bets
from quant_fund.portfolio.executable_alpha import executable_alpha, should_trade


def test_executable_alpha_and_trade_gate_are_finite_only() -> None:
    assert executable_alpha(0.10, 0.01, 0.01, 0.01, 0.01) == pytest.approx(0.06)
    assert should_trade(0.06, 0.05) is True
    assert should_trade(float("nan"), 0.0) is False
    with pytest.raises(ValueError, match="finite"):
        executable_alpha(float("inf"), 0.0, 0.0, 0.0, 0.0)


def test_concentration_requires_aligned_finite_inputs() -> None:
    corr = np.eye(2)
    assert effective_bets(np.array([0.5, -0.5]), corr) == pytest.approx(2.0)
    assert average_pairwise_corr(corr) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="aligned"):
        effective_bets(np.ones(2), np.eye(3))
    with pytest.raises(ValueError, match="finite"):
        average_pairwise_corr(np.array([[1.0, np.nan], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="symmetric"):
        average_pairwise_corr(np.array([[1.0, 0.2], [0.1, 1.0]]))

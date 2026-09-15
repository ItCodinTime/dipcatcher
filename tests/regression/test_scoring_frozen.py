"""Frozen numerical conventions — do not 'improve' without updating MATH_SPEC."""

import numpy as np

from quant_fund.metrics.scoring import pinball_loss, qlike


def test_pinball_formula_frozen() -> None:
    y = np.array([0.0])
    q = np.array([-1.0])
    assert pinball_loss(y, q, 0.05)[0] == 0.05


def test_qlike_formula_frozen() -> None:
    # y=e, yhat=1 → e - 1 - 1 = e-2
    assert np.isclose(qlike(np.array([np.e]), np.array([1.0])), np.e - 2.0)

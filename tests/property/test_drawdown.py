"""Need at least one test under tests/property."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.metrics.returns import max_drawdown


@given(
    st.lists(st.floats(0.0, 0.05, allow_nan=False, allow_infinity=False), min_size=2, max_size=40)
)
@settings(max_examples=25)
def test_nonneg_returns_zero_mdd(rs: list[float]) -> None:
    assert max_drawdown(np.array(rs)) == 0.0

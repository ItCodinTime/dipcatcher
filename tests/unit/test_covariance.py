import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.models.covariance import (
    is_symmetric,
    ledoit_wolf_cov,
    min_eigenvalue,
    repair_psd,
    sample_cov,
)


def test_ledoit_wolf_psd() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 6))
    s = ledoit_wolf_cov(x)
    assert is_symmetric(s)
    assert min_eigenvalue(s) >= -1e-8


def test_repair_indefinite() -> None:
    s = np.array([[1.0, 2.0], [2.0, 1.0]])
    r, info = repair_psd(s)
    assert info["repaired"] == 1.0
    assert min_eigenvalue(r) >= -1e-8
    assert is_symmetric(r)


@given(st.integers(3, 8), st.integers(20, 40))
@settings(max_examples=10, deadline=None)
def test_sample_cov_symmetric(k: int, n: int) -> None:
    rng = np.random.default_rng(k * 100 + n)
    x = rng.normal(size=(n, k))
    s = sample_cov(x)
    assert is_symmetric(s)

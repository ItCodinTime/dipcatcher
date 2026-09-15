import numpy as np

from quant_fund.models.covariance import dcc_gaussian, min_eigenvalue


def test_dcc_psd_and_params() -> None:
    rng = np.random.default_rng(0)
    t, n = 80, 4
    e = rng.normal(size=(t, n))
    r = np.cumsum(0.1 * e, axis=0)
    r = np.diff(r, axis=0, prepend=0)
    h, params = dcc_gaussian(e * 0.01)
    assert min_eigenvalue(h) >= -1e-8
    assert params["a"] >= 0
    assert params["b"] >= 0
    assert params["a"] + params["b"] < 1.0 + 1e-6

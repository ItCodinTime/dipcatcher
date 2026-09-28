"""Canon tests: input-output HMM."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.io_hmm import (
    io_hmm_filter,
    io_hmm_fit,
    io_hmm_nll,
    io_hmm_transitions,
)


def _driven_series() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two regimes; covariate x drives the switch."""
    rng = np.random.default_rng(6)
    t_len = 200
    x = np.zeros((t_len, 2))
    x[:, 0] = 1.0  # intercept
    drive = np.zeros(t_len)
    drive[60:140] = 1.0  # regime-1 pressure window
    x[:, 1] = drive
    states = (drive > 0.5).astype(int)
    y = rng.normal(0, 0.3, t_len) + states * 3.0
    return y, x, states


def test_transitions_softmax_rows() -> None:
    x = np.array([[1.0, 0.0], [1.0, 5.0]])
    w = np.array([[[1.0, 0.0], [0.0, 2.0]], [[0.0, 0.0], [0.0, 0.0]]])
    p = io_hmm_transitions(x, w)
    assert p.shape == (2, 2, 2)
    np.testing.assert_allclose(p.sum(axis=2), 1.0)
    # row state-0 under x=[1,5]: logits [1, 10] -> strong exit to j=1
    assert p[1, 0, 1] > 0.99


def test_nll_finite_and_prefers_signal() -> None:
    y, x, _ = _driven_series()
    k, p = 2, 2
    mu = np.array([0.0, 3.0])
    sigma = np.array([0.3, 0.3])
    # good w: state exits driven by drive covariate
    w_good = np.zeros((k, k, p))
    w_good[0, 0, 0] = 3.0
    w_good[0, 1, 1] = 3.0  # drive -> exit 0->1
    w_good[1, 1, 0] = 3.0
    w_good[1, 0, 1] = -3.0
    w_flat = np.zeros((k, k, p))
    nll_good = io_hmm_nll(y, x, w_good, mu, sigma)
    nll_flat = io_hmm_nll(y, x, w_flat, mu, sigma)
    assert np.isfinite(nll_good) and np.isfinite(nll_flat)
    assert nll_good < nll_flat


def test_io_hmm_filter_posts() -> None:
    y, x, _ = _driven_series()
    w = np.zeros((2, 2, 2))
    w[0, 0, 0] = 3.0
    w[0, 1, 1] = 3.0
    w[1, 1, 0] = 3.0
    w[1, 0, 1] = -3.0
    out = io_hmm_filter(y, x, w, np.array([0.0, 3.0]), np.array([0.3, 0.3]))
    np.testing.assert_allclose(out["state_prob"].sum(axis=1), 1.0, atol=1e-8)
    assert out["state_prob"][10, 0] > 0.8
    assert out["state_prob"][100, 1] > 0.8


def test_io_hmm_fit_recovers_switch() -> None:
    y, x, states = _driven_series()
    out = io_hmm_fit(y, x, n_states=2, max_iter=60)
    assert np.isfinite(out["nll"])
    post = io_hmm_filter(y, x, out["w"], out["mu"], out["sigma"])["state_prob"]
    agree = (post.argmax(axis=1) == states).mean()
    assert agree > 0.7 or (1 - agree) > 0.7  # label order is free


def test_io_hmm_validation() -> None:
    y, x, _ = _driven_series()
    w = np.zeros((2, 2, 2))
    with pytest.raises(ValueError):
        io_hmm_nll(y[:4], x[:4], w, np.zeros(2), np.ones(2))
    with pytest.raises(ValueError):
        io_hmm_nll(y, x, w, np.zeros(2), -np.ones(2))
    with pytest.raises(ValueError):
        io_hmm_nll(y, x, np.zeros((3, 3, 2)), np.zeros(2), np.ones(2))
    with pytest.raises(ValueError):
        io_hmm_transitions(x, np.zeros((2, 3, 2)))  # not square in states
    with pytest.raises(ValueError):
        io_hmm_fit(np.full(20, np.nan), x[:20], 2)
    with pytest.raises(ValueError):
        io_hmm_fit(y, x, n_states=1)

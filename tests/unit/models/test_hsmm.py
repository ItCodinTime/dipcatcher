"""Canon tests: explicit-duration HSMM."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.hsmm import hsmm_filter, hsmm_fit, hsmm_viterbi


def _two_state_params(d_max: int = 15):
    mu = np.array([0.0, 4.0])
    sigma = np.array([0.3, 0.4])
    trans = np.array([[0.0, 1.0], [1.0, 0.0]])  # strict alternation
    dur = np.zeros((2, d_max))
    dur[0, 5] = 1.0  # state 0 always lasts exactly 6 steps
    dur[1, 7] = 1.0  # state 1 always lasts exactly 8 steps
    return mu, sigma, trans, dur


def _alternating_series() -> np.ndarray:
    rng = np.random.default_rng(4)
    blocks = []
    for _ in range(6):
        blocks.append(rng.normal(0.0, 0.3, 6))
        blocks.append(rng.normal(4.0, 0.4, 8))
    return np.concatenate(blocks)


def test_hsmm_filter_state_probs() -> None:
    y = _alternating_series()
    mu, sigma, trans, dur = _two_state_params()
    out = hsmm_filter(y, mu, sigma, trans, dur)
    post = out["state_prob"]
    assert post.shape == (y.size, 2)
    np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-8)
    # mid-block assignments should be confident
    assert post[3, 0] > 0.9
    assert post[10, 1] > 0.9
    assert np.isfinite(out["loglik"])


def test_hsmm_viterbi_recovers_blocks() -> None:
    y = _alternating_series()
    mu, sigma, trans, dur = _two_state_params()
    states = hsmm_viterbi(y, mu, sigma, trans, dur)["states"]
    truth = np.concatenate([[0] * 6 + [1] * 8] * 6)
    assert (states == truth).mean() > 0.9


def test_hsmm_nll_prefers_true_duration() -> None:
    y = _alternating_series()
    mu, sigma, trans, dur = _two_state_params()
    good = hsmm_filter(y, mu, sigma, trans, dur)["loglik"]
    # wrong duration: put mass on d=2 instead of 6/8
    bad_dur = np.zeros((2, 15))
    bad_dur[:, 1] = 1.0
    bad = hsmm_filter(y, mu, sigma, trans, bad_dur)["loglik"]
    assert good > bad


def test_hsmm_geometric_is_hmm() -> None:
    # geometric duration pmf must collapse to HMM-like behavior: likelihood
    # finite and state probs well-formed on a sticky-switching series
    rng = np.random.default_rng(8)
    states = np.repeat(np.r_[0, 1, 0, 1], 30)
    y = rng.normal(0, 0.5, 120) + states * 3
    d_max = 30
    mu = np.array([0.0, 3.0])
    sigma = np.array([0.5, 0.5])
    trans = np.array([[0.0, 1.0], [1.0, 0.0]])
    dur = np.zeros((2, d_max))
    p_geo = 1 / 20.0
    dur[:] = p_geo * (1 - p_geo) ** np.arange(d_max)
    dur /= dur.sum(axis=1, keepdims=True)
    out = hsmm_filter(y, mu, sigma, trans, dur)
    assert np.isfinite(out["loglik"])
    assert (out["state_prob"].argmax(axis=1) == states).mean() > 0.85


def test_hsmm_fit_runs_and_learns_emissions() -> None:
    rng = np.random.default_rng(2)
    y = np.concatenate([rng.normal(0, 0.4, 100), rng.normal(3, 0.4, 100)])
    out = hsmm_fit(y, n_states=2, d_max=8, n_iter=5)
    assert np.isfinite(out["loglik"])
    assert sorted(out["mu"])[1] - sorted(out["mu"])[0] > 1.5
    assert (out["sigma"] > 0).all()


def test_hsmm_validation() -> None:
    mu, sigma, trans, dur = _two_state_params()
    y = _alternating_series()
    with pytest.raises(ValueError):
        hsmm_filter(np.arange(2.0), mu, sigma, trans, dur)
    with pytest.raises(ValueError):
        hsmm_filter(y, mu, -sigma, trans, dur)
    with pytest.raises(ValueError):
        hsmm_filter(y, mu, sigma, trans + 1.0, dur)  # rows sum to 2
    with pytest.raises(ValueError):
        hsmm_filter(y, mu, sigma, trans, dur * 2)  # rows sum to 2
    with pytest.raises(ValueError):
        hsmm_fit(np.full(20, np.nan), 2)
    with pytest.raises(ValueError):
        hsmm_fit(y, 0)

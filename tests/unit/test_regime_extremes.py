"""Wave 11: regime extremes — HMM probs, SingleState, VolThreshold, n_states fail-closed."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.regime import (
    GaussianHMMRegime,
    SingleStateRegime,
    VolThresholdRegime,
)


def test_gaussian_hmm_probs_sum_to_one_and_nonneg() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(120, 2))
    m = GaussianHMMRegime(n_states=3, seed=11).fit(x)
    p = m.predict_proba(x)
    assert p.shape == (120, 3)
    assert np.all(p >= -1e-12)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    # Smoothed path also row-stochastic
    ps = m.predict_smoothed_proba(x)
    assert ps.shape == (120, 3)
    assert np.allclose(ps.sum(axis=1), 1.0, atol=1e-6)


def test_single_state_regime_constant_one() -> None:
    x = np.zeros((7, 3))
    m = SingleStateRegime().fit(x)
    p = m.predict_proba(x)
    assert p.shape == (7, 1)
    assert np.allclose(p, 1.0)
    assert np.allclose(m.predict(x), 1.0)
    meta = m.metadata()
    assert meta.family == "regime"
    assert meta.name == "single_state"


def test_vol_threshold_regime_boundaries() -> None:
    # Fit cut at q=0.5 on [0,1,2,3,4] → cut=2.0
    x_fit = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=float)
    m = VolThresholdRegime(q=0.5).fit(x_fit)
    assert m.cut == pytest.approx(2.0)

    # Exact cut → high regime (prob low=0, high=1)
    at = np.array([[2.0]], dtype=float)
    p_at = m.predict_proba(at)
    assert p_at.shape == (1, 2)
    np.testing.assert_allclose(p_at, [[0.0, 1.0]])

    # Strictly below → low
    below = np.array([[1.999]], dtype=float)
    np.testing.assert_allclose(m.predict_proba(below), [[1.0, 0.0]])

    # Strictly above → high
    above = np.array([[2.001]], dtype=float)
    np.testing.assert_allclose(m.predict_proba(above), [[0.0, 1.0]])

    # Batch mixes
    batch = np.array([[0.0], [2.0], [5.0]], dtype=float)
    pb = m.predict_proba(batch)
    np.testing.assert_allclose(pb, [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    assert np.allclose(pb.sum(axis=1), 1.0)


@pytest.mark.parametrize("bad", [0, -1, -5])
def test_gaussian_hmm_invalid_n_states_fail_closed(bad: int) -> None:
    with pytest.raises(ValueError, match="n_states"):
        GaussianHMMRegime(n_states=bad)


def test_gaussian_hmm_n_states_rejects_non_integer() -> None:
    with pytest.raises(ValueError, match="n_states"):
        GaussianHMMRegime(n_states=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_states"):
        GaussianHMMRegime(n_states="3")  # type: ignore[arg-type]

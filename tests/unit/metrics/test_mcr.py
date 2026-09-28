"""Canon tests: PAVA + miscalibration/discrimination decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.mcr import (
    log_score_urc,
    mcr_decomposition,
    pava,
    reliability_isotonic,
)


@pytest.mark.parametrize(
    ("y", "expected"),
    [
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])),
        (np.array([3.0, 2.0, 1.0]), np.array([2.0, 2.0, 2.0])),
        (np.array([1.0, 3.0, 2.0]), np.array([1.0, 2.5, 2.5])),
        (np.array([2.0, 1.0, 4.0, 3.0]), np.array([1.5, 1.5, 3.5, 3.5])),
    ],
)
def test_pava_known_answers(y, expected) -> None:
    np.testing.assert_allclose(pava(y), expected)


def test_pava_weighted() -> None:
    # weights pull the pooled level toward the heavier block
    out = pava(np.array([2.0, 0.0]), w=np.array([3.0, 1.0]))
    np.testing.assert_allclose(out, [1.5, 1.5])


def test_pava_validation() -> None:
    with pytest.raises(ValueError):
        pava(np.array([]))
    with pytest.raises(ValueError):
        pava(np.array([1.0, np.nan]))
    with pytest.raises(ValueError):
        pava(np.array([1.0, 0.0]), w=np.array([0.0, 1.0]))


def test_mcr_identity_exact() -> None:
    rng = np.random.default_rng(3)
    prob = rng.uniform(0.05, 0.95, 300)
    y = (rng.uniform(size=300) < prob * 0.6 + 0.2).astype(float)
    out = mcr_decomposition(prob, y)
    recon = (
        out["within_group_variance"]
        + out["mcb"]
        - out["dsc"]
        + out["unc"]
        - 2.0 * out["cov_within"]
    )
    assert out["brier"] == pytest.approx(recon, abs=1e-10)
    assert abs(out["residual"]) < 1e-10
    assert out["mcb"] >= 0 and out["dsc"] >= 0 and out["unc"] >= 0


def test_mcr_perfect_forecast() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0])
    prob = y.copy()
    out = mcr_decomposition(prob, y)
    assert out["brier"] == pytest.approx(0.0)
    assert out["mcb"] == pytest.approx(0.0)


def test_mcr_miscalibration_detected() -> None:
    rng = np.random.default_rng(13)
    y = (rng.uniform(size=500) < 0.3).astype(float)
    calibrated = np.full(500, 0.3)
    biased = np.full(500, 0.6)
    assert mcr_decomposition(biased, y)["mcb"] > mcr_decomposition(calibrated, y)["mcb"]


def test_reliability_isotonic_monotone_in_prob() -> None:
    rng = np.random.default_rng(21)
    prob = rng.uniform(0, 1, 200)
    y = (rng.uniform(size=200) < prob).astype(float)
    iso = reliability_isotonic(prob, y)
    order = np.argsort(prob)
    assert np.all(np.diff(iso[order]) >= -1e-12)


def test_log_score_urc_identity() -> None:
    rng = np.random.default_rng(31)
    prob = np.clip(rng.normal(0.4, 0.15, 300), 0.01, 0.99)
    y = (rng.uniform(size=300) < prob).astype(float)
    out = log_score_urc(prob, y)
    assert abs(out["residual"]) < 1e-10
    assert out["log_score"] > 0


def test_prob_validation() -> None:
    y = np.array([0.0, 1.0])
    with pytest.raises(ValueError):
        mcr_decomposition(np.array([-0.1, 0.5]), y)
    with pytest.raises(ValueError):
        mcr_decomposition(np.array([0.5]), np.array([0.0]))
    with pytest.raises(ValueError):
        reliability_isotonic(np.array([0.5, np.nan]), y)

import numpy as np

from quant_fund.metrics.probability import (
    brier_score,
    expected_calibration_error,
    kupiec_pof,
    log_loss,
    pit_ks,
)


def test_brier_perfect() -> None:
    y = np.array([0.0, 1.0, 1.0, 0.0])
    assert brier_score(y, y) == 0.0


def test_log_loss_better_when_confident_and_right() -> None:
    y = np.array([1.0, 0.0])
    assert log_loss(np.array([0.9, 0.1]), y) < log_loss(np.array([0.6, 0.4]), y)


def test_kupiec_matches_when_rate_equals_alpha() -> None:
    rng = np.random.default_rng(1)
    hits = (rng.random(2000) < 0.05).astype(float)
    rate, _, p = kupiec_pof(hits, 0.05)
    assert abs(rate - 0.05) < 0.02
    assert p > 0.01


def test_pit_ks_uniform() -> None:
    u = np.linspace(0.01, 0.99, 200)
    stat, p = pit_ks(u)
    assert stat < 0.1
    assert p > 0.05


def test_ece_zero_when_aligned() -> None:
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0.1, 0.1, 0.9, 0.9])
    assert expected_calibration_error(p, y, n_bins=2) < 0.05

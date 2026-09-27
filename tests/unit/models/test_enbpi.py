"""Tests for quant_fund.models.enbpi — Xu & Xie EnbPI intervals."""

import numpy as np
import pytest

from quant_fund.models.enbpi import (
    EnbPI,
    block_bootstrap_indices,
    in_bag_mask,
    leave_one_out_predictions,
    optimal_beta,
    prediction_interval,
    signed_residuals,
)


def test_loo_aggregate_uses_only_out_of_bag_models() -> None:
    predictions = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    in_bag = np.array([[True, False], [False, True], [True, False]])
    out = leave_one_out_predictions(predictions, in_bag, aggregate="mean")
    assert np.allclose(out, [10.0, 2.0, 30.0])
    med = leave_one_out_predictions(predictions, in_bag, aggregate="median")
    assert np.allclose(med, out)


def test_loo_row_with_no_out_of_bag_model_fails_closed() -> None:
    predictions = np.ones((2, 2))
    in_bag = np.array([[True, True], [False, True]])
    with pytest.raises(ValueError, match="out-of-bag"):
        leave_one_out_predictions(predictions, in_bag)


def test_signed_residuals_are_y_minus_loo() -> None:
    y = np.array([1.0, 2.0, 4.0])
    loo = np.array([0.0, 2.0, 1.0])
    assert np.allclose(signed_residuals(y, loo), [1.0, 0.0, 3.0])


def test_gaussian_beta_is_near_equal_tailed() -> None:
    rng = np.random.default_rng(0)
    residuals = rng.normal(0.0, 1.0, 4000)
    beta = optimal_beta(residuals, alpha=0.1, n_grid=51)
    assert abs(beta - 0.05) < 0.02


def test_interval_covers_fresh_gaussian_draws() -> None:
    rng = np.random.default_rng(1)
    residuals = rng.normal(0.0, 1.0, 3000)
    fresh = rng.normal(0.0, 1.0, 3000)
    interval = prediction_interval(0.0, residuals, alpha=0.1, n_grid=41)
    covered = (fresh >= interval.lower) & (fresh <= interval.upper)
    assert abs(float(np.mean(covered)) - 0.9) < 0.03
    assert interval.upper > interval.lower


def test_skewed_residuals_shift_beta_and_still_cover() -> None:
    rng = np.random.default_rng(2)
    residuals = rng.exponential(1.0, 4000) - 1.0
    beta = optimal_beta(residuals, alpha=0.2, n_grid=41)
    assert beta < 0.08
    fresh = rng.exponential(1.0, 4000) - 1.0
    interval = prediction_interval(0.0, residuals, alpha=0.2, n_grid=41)
    covered = (fresh >= interval.lower) & (fresh <= interval.upper)
    assert abs(float(np.mean(covered)) - 0.8) < 0.04


def test_sliding_window_widens_after_a_variance_jump() -> None:
    rng = np.random.default_rng(3)
    model = EnbPI(rng.normal(0.0, 1.0, 300), alpha=0.1, batch_size=1)
    width0 = model.interval(0.0).upper - model.interval(0.0).lower
    for error in rng.normal(0.0, 4.0, 300):
        model.observe(float(error), 0.0)
    width1 = model.interval(0.0).upper - model.interval(0.0).lower
    assert width1 > width0 * 2.0
    assert model.n_pending == 0


def test_partial_batch_does_not_slide_the_window() -> None:
    model = EnbPI(np.array([0.0, 0.0, 0.0]), alpha=0.1, batch_size=2)
    model.observe(1.0, 0.0)
    assert model.n_pending == 1
    assert np.allclose(model.residuals_, [0.0, 0.0, 0.0])
    model.observe(1.0, 0.0)
    assert model.n_pending == 0
    assert np.allclose(model.residuals_, [0.0, 1.0, 1.0])


def test_block_bootstrap_indices_stay_in_range_and_form_blocks() -> None:
    rng = np.random.default_rng(4)
    iid = block_bootstrap_indices(20, n_bootstrap=5, rng=rng, block_length=1)
    assert iid.shape == (5, 20)
    assert iid.min() >= 0 and iid.max() < 20
    blocks = block_bootstrap_indices(12, n_bootstrap=3, rng=rng, block_length=4)
    assert blocks.shape == (3, 12)
    for row in blocks:
        for start in (0, 4, 8):
            chunk = row[start : start + 4]
            assert np.all(np.diff(chunk) == 1)
    mask = in_bag_mask(iid, n=20)
    assert mask.shape == (20, 5)
    assert bool(np.all(mask.any(axis=0)))


def test_fail_closed_edges() -> None:
    with pytest.raises(ValueError):
        prediction_interval(0.0, np.array([1.0]), alpha=0.0)
    with pytest.raises(ValueError):
        prediction_interval(np.nan, np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        EnbPI(np.array([]))
    with pytest.raises(ValueError):
        EnbPI(np.array([1.0, np.inf]))
    with pytest.raises(ValueError):
        leave_one_out_predictions(np.ones((2, 2)), np.zeros((2, 2), dtype=bool), aggregate="trim")
    model = EnbPI(np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        model.observe(np.nan, 0.0)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        block_bootstrap_indices(4, 2, rng, block_length=0)

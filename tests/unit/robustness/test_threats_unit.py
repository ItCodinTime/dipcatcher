"""Unit coverage for the threat-model perturbation primitives."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quant_fund.robustness.threats import (
    THREAT_NAMES,
    apply_jitter,
    apply_missing,
    apply_spike,
    apply_stale,
    apply_vol_scaled,
    apply_vol_scaled_prices,
    as_vector,
    ball_norm,
    causal_increment_vol,
    net_excess,
    path_gross,
    path_turnover,
    project,
)


def test_threat_names_are_stable() -> None:
    assert THREAT_NAMES == (
        "vol_scaled_path",
        "missing_bars",
        "stale_prints",
        "spikes",
        "timing_jitter",
        "cost_shock",
        "wasserstein_regime",
    )


class TestAsVector:
    def test_flattens_and_copies(self) -> None:
        base = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = as_vector(base, name="x")
        assert out.shape == (4,)
        out[0] = 99.0
        assert base[0, 0] == 1.0

    @pytest.mark.parametrize("bad", [np.array([]), np.array([np.nan]), np.array([np.inf])])
    def test_rejects_empty_and_nonfinite(self, bad: np.ndarray) -> None:
        with pytest.raises(ValueError, match="non-empty finite vector"):
            as_vector(bad, name="x")


class TestProject:
    def test_l2_inside_ball_returns_same_values(self) -> None:
        delta = np.array([0.1, -0.2])
        out = project(delta, 1.0, "l2")
        assert_allclose(out, delta)

    def test_l2_outside_ball_is_rescaled_to_boundary(self) -> None:
        delta = np.array([3.0, 4.0])
        out = project(delta, 1.0, "l2")
        assert_allclose(np.linalg.norm(out), 1.0)
        assert_allclose(out, delta / 5.0)

    def test_linf_clips_each_coordinate(self) -> None:
        delta = np.array([5.0, -0.5, 0.2])
        assert_allclose(project(delta, 1.0, "linf"), [1.0, -0.5, 0.2])

    def test_zero_radius_zeroes(self) -> None:
        assert_allclose(project(np.array([3.0]), 0.0, "l2"), [0.0])
        assert_allclose(project(np.array([3.0]), 0.0, "linf"), [0.0])

    def test_zero_length_vector_stays_zero(self) -> None:
        assert_allclose(project(np.zeros(3), 1.0, "l2"), np.zeros(3))

    def test_bad_norm_and_radius_and_delta(self) -> None:
        with pytest.raises(ValueError, match="l2.*linf"):
            project(np.ones(2), 1.0, "l1")
        with pytest.raises(ValueError, match="finite and non-negative"):
            project(np.ones(2), -1.0, "l2")
        with pytest.raises(ValueError, match="finite and non-negative"):
            project(np.ones(2), math.inf, "l2")
        with pytest.raises(ValueError, match="delta must be finite"):
            project(np.array([np.nan]), 1.0, "l2")


class TestBallNorm:
    def test_l2_and_linf(self) -> None:
        delta = np.array([3.0, -4.0])
        assert ball_norm(delta, "l2") == pytest.approx(5.0)
        assert ball_norm(delta, "linf") == pytest.approx(4.0)

    def test_linf_empty_is_zero(self) -> None:
        assert ball_norm(np.array([]), "linf") == 0.0

    def test_unknown_norm_raises(self) -> None:
        with pytest.raises(ValueError, match="l2.*linf"):
            ball_norm(np.ones(2), "l1")


class TestVolScaled:
    def test_unit_scale_defaults_to_raw_addition(self) -> None:
        base = np.array([1.0, -1.0])
        eta = np.array([0.5, 0.5])
        assert_allclose(apply_vol_scaled(base, eta), [1.5, -0.5])

    def test_scale_multiplies_shock(self) -> None:
        base = np.array([1.0])
        assert_allclose(apply_vol_scaled(base, np.array([1.0]), np.array([2.0])), [3.0])

    def test_misaligned_and_nonfinite_rejected(self) -> None:
        base = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="aligned"):
            apply_vol_scaled(base, np.array([1.0]))
        with pytest.raises(ValueError, match="aligned"):
            apply_vol_scaled(base, np.array([np.nan, 0.0]))
        with pytest.raises(ValueError, match="aligned"):
            apply_vol_scaled(base, np.ones(2), np.array([1.0]))
        with pytest.raises(ValueError, match="aligned"):
            apply_vol_scaled(base, np.ones(2), np.array([-1.0, 0.0]))
        with pytest.raises(ValueError, match="aligned"):
            apply_vol_scaled(base, np.ones(2), np.array([np.inf, 0.0]))

    def test_prices_rebuild_positive_path_from_log_increments(self) -> None:
        prices = np.array([100.0, 101.0, 102.0])
        eta = np.array([0.0, 0.5])
        vol = np.array([0.1, 0.1])
        out = apply_vol_scaled_prices(prices, eta, vol)
        assert out[0] == 100.0
        # increment 0 unchanged, increment 1 raised by 0.05 in log space
        assert out[1] == pytest.approx(101.0)
        expected_log2 = math.log(100.0) + math.log(101.0 / 100.0) + (math.log(102.0 / 101.0) + 0.05)
        assert out[2] == pytest.approx(math.exp(expected_log2))
        assert np.all(out > 0.0)

    def test_prices_reject_nonpositive_and_misaligned(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            apply_vol_scaled_prices(np.array([0.0, 1.0]), np.array([0.0]), np.array([0.0]))
        with pytest.raises(ValueError, match="align"):
            apply_vol_scaled_prices(np.array([1.0, 2.0]), np.array([0.0, 0.0]), np.array([0.0]))
        with pytest.raises(ValueError, match="non-negative"):
            apply_vol_scaled_prices(np.array([1.0, 2.0]), np.array([0.0]), np.array([-1.0]))
        with pytest.raises(ValueError, match="finite"):
            apply_vol_scaled_prices(np.array([1.0, 2.0]), np.array([np.inf]), np.array([0.0]))


class TestCausalIncrementVol:
    def test_first_scale_is_floor_then_lagged_std(self) -> None:
        prices = np.array([100.0, 101.0, 99.0, 100.5])
        vol = causal_increment_vol(prices, floor=1e-4)
        increments = np.diff(np.log(prices))
        assert vol[0] == 1e-4
        # vol[1] sees only increment[0] — fewer than 2 observations -> floor
        assert vol[1] == 1e-4
        assert vol[2] == pytest.approx(float(np.std(increments[:2], ddof=1)))

    def test_constant_prices_fall_back_to_floor(self) -> None:
        prices = np.full(5, 10.0)
        vol = causal_increment_vol(prices, floor=0.5)
        assert_allclose(vol, 0.5)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            causal_increment_vol(np.array([1.0, 2.0]), floor=0.0)
        with pytest.raises(ValueError, match="finite"):
            causal_increment_vol(np.array([1.0, 2.0]), floor=math.nan)
        with pytest.raises(ValueError, match="two positive"):
            causal_increment_vol(np.array([1.0]))
        with pytest.raises(ValueError, match="two positive"):
            causal_increment_vol(np.array([1.0, -1.0]))


class TestDiscreteCorruptions:
    def test_missing_zeroes_selected_bars(self) -> None:
        out = apply_missing(np.array([1.0, 2.0, 3.0]), [0, 2])
        assert_allclose(out, [0.0, 2.0, 0.0])

    def test_missing_index_validation(self) -> None:
        with pytest.raises(TypeError, match="integers"):
            apply_missing(np.ones(3), [True])
        with pytest.raises(TypeError, match="integers"):
            apply_missing(np.ones(3), [1.5])
        with pytest.raises(IndexError):
            apply_missing(np.ones(3), [3])
        with pytest.raises(IndexError):
            apply_missing(np.ones(3), [-1])

    def test_stale_repeats_previous_print_and_clips(self) -> None:
        out = apply_stale(np.array([1.0, 2.0, 3.0, 4.0]), 1, 10)
        assert_allclose(out, [1.0, 1.0, 1.0, 1.0])

    def test_stale_at_first_bar_fills_zero(self) -> None:
        out = apply_stale(np.array([1.0, 2.0]), 0, 1)
        assert_allclose(out, [0.0, 2.0])

    def test_stale_validation(self) -> None:
        with pytest.raises(TypeError, match="integers"):
            apply_stale(np.ones(3), True, 1)
        with pytest.raises(TypeError, match="integers"):
            apply_stale(np.ones(3), 0, 1.5)
        with pytest.raises(ValueError, match="outside"):
            apply_stale(np.ones(3), -1, 1)
        with pytest.raises(ValueError, match="outside"):
            apply_stale(np.ones(3), 3, 1)
        with pytest.raises(ValueError, match="outside"):
            apply_stale(np.ones(3), 0, 0)

    def test_spike_adds_scaled_magnitude(self) -> None:
        out = apply_spike(np.ones(3), 1, 2.0, scale=3.0)
        assert_allclose(out, [1.0, 7.0, 1.0])

    def test_spike_validation(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            apply_spike(np.ones(3), True, 1.0)
        with pytest.raises(ValueError, match="outside"):
            apply_spike(np.ones(3), -1, 1.0)
        with pytest.raises(ValueError, match="outside"):
            apply_spike(np.ones(3), 3, 1.0)
        with pytest.raises(ValueError, match="finite"):
            apply_spike(np.ones(3), 0, math.inf)
        with pytest.raises(ValueError, match="non-negative"):
            apply_spike(np.ones(3), 0, 1.0, scale=-1.0)

    def test_jitter_rolls_circularly(self) -> None:
        assert_allclose(apply_jitter(np.array([1.0, 2.0, 3.0]), 1), [3.0, 1.0, 2.0])
        assert_allclose(apply_jitter(np.array([1.0, 2.0, 3.0]), -1), [2.0, 3.0, 1.0])

    def test_jitter_empty_and_type(self) -> None:
        # Empty input cannot pass as_vector (non-empty required), so the
        # size-0 early return is unreachable — verify the type gate instead.
        with pytest.raises(TypeError, match="integer"):
            apply_jitter(np.ones(3), True)
        with pytest.raises(TypeError, match="integer"):
            apply_jitter(np.ones(3), 0.5)


class TestPathAggregates:
    def test_path_turnover_counts_initial_change(self) -> None:
        assert path_turnover(np.array([0.0, 1.0, -1.0]), initial=0.5) == pytest.approx(3.5)

    def test_path_turnover_validation(self) -> None:
        with pytest.raises(ValueError, match="non-empty and finite"):
            path_turnover(np.array([]))
        with pytest.raises(ValueError, match="non-empty and finite"):
            path_turnover(np.array([np.nan]))
        with pytest.raises(ValueError, match="non-empty and finite"):
            path_turnover(np.array([1.0]), initial=math.inf)

    def test_path_gross_alignment(self) -> None:
        assert path_gross(np.array([1.0, -1.0]), np.array([0.5, 0.5])) == pytest.approx(0.0)
        with pytest.raises(ValueError, match="aligned"):
            path_gross(np.array([1.0]), np.array([1.0, 1.0]))
        with pytest.raises(ValueError, match="finite"):
            path_gross(np.array([np.nan]), np.array([1.0]))

    def test_net_excess_subtracts_cost_times_turnover(self) -> None:
        # gross = 1.0 + 0.5 = 1.5; turnover = |1-0| + |0.5-1| = 1.5
        net = net_excess(np.array([1.0, 0.5]), np.array([1.0, 1.0]), cost=0.4)
        assert net == pytest.approx(1.5 - 0.4 * 1.5)
        with pytest.raises(ValueError, match="non-negative"):
            net_excess(np.array([1.0]), np.array([1.0]), cost=-0.1)

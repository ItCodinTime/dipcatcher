"""Edge and boundary cases for the analytic closed forms."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.robustness.analytic import (
    clopper_pearson_lower,
    cost_shock_radius,
    gelbrich_worst_case_ratio,
    linear_l2_radius,
    linear_margin,
    linear_positive_probability,
    linear_spike_radius,
    linf_radius_from_l2,
    lipschitz_worst_case,
    plug_in_moments,
    smoothing_radius,
    wasserstein_worst_case_mean,
)


class TestClopperPearson:
    def test_zero_successes_is_zero(self) -> None:
        assert clopper_pearson_lower(0, 100, 0.001) == 0.0

    def test_validation(self) -> None:
        with pytest.raises(TypeError):
            clopper_pearson_lower(True, 10, 0.05)
        with pytest.raises(TypeError):
            clopper_pearson_lower(1, 10.0, 0.05)
        with pytest.raises(ValueError, match="0..trials"):
            clopper_pearson_lower(11, 10, 0.05)
        with pytest.raises(ValueError, match="0..trials"):
            clopper_pearson_lower(1, 0, 0.05)
        with pytest.raises(ValueError, match="0, 1"):
            clopper_pearson_lower(5, 10, 0.0)
        with pytest.raises(ValueError, match="0, 1"):
            clopper_pearson_lower(5, 10, 1.0)


class TestSmoothingRadius:
    def test_abstains_at_or_below_half(self) -> None:
        assert smoothing_radius(0.5, 0.4, 1.0) == 0.0
        assert smoothing_radius(0.4, 0.3, 1.0) == 0.0

    def test_infinite_when_runner_zero_and_top_certain(self) -> None:
        assert smoothing_radius(1.0, 0.0, 1.0) == math.inf

    def test_infinite_when_runner_is_zero(self) -> None:
        # p_lower < 1 with a zero runner bound: the -inf ppf term drives
        # the radius to +inf through the finite-check fallback.
        assert smoothing_radius(0.9, 0.0, 1.0) == math.inf

    def test_binary_simplification(self) -> None:
        from scipy import stats

        radius = smoothing_radius(0.9, 0.1, 2.0)
        assert radius == pytest.approx(2.0 * stats.norm.ppf(0.9))

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="sigma"):
            smoothing_radius(0.9, 0.1, 0.0)
        with pytest.raises(ValueError, match="sigma"):
            smoothing_radius(0.9, 0.1, math.inf)
        with pytest.raises(ValueError, match="p_lower"):
            smoothing_radius(-0.1, 0.0, 1.0)
        with pytest.raises(ValueError, match="p_runner_upper"):
            smoothing_radius(0.9, 1.1, 1.0)
        with pytest.raises(ValueError, match="at least p_runner"):
            smoothing_radius(0.5, 0.6, 1.0)


class TestLinfFromL2:
    def test_scales_by_sqrt_dimension(self) -> None:
        assert linf_radius_from_l2(4.0, 4) == pytest.approx(2.0)

    def test_infinite_rejected_by_finiteness_gate(self) -> None:
        # The isinf passthrough is unreachable: the finite gate fires first.
        with pytest.raises(ValueError, match="non-negative"):
            linf_radius_from_l2(math.inf, 4)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            linf_radius_from_l2(1.0, 0)
        with pytest.raises(ValueError, match="positive integer"):
            linf_radius_from_l2(1.0, True)
        with pytest.raises(ValueError, match="non-negative"):
            linf_radius_from_l2(-1.0, 4)


class TestLinearForms:
    def test_margin_shape_and_finiteness(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            linear_margin(np.ones(2), 0.0, np.ones(3))
        with pytest.raises(ValueError, match="finite"):
            linear_margin(np.array([np.nan]), 0.0, np.ones(1))
        with pytest.raises(ValueError, match="finite"):
            linear_margin(np.ones(1), math.inf, np.ones(1))

    def test_l2_radius_zero_weights(self) -> None:
        w = np.zeros(3)
        x = np.ones(3)
        assert linear_l2_radius(w, 0.0, x) == 0.0
        assert linear_l2_radius(w, 1.0, x) == math.inf

    def test_positive_probability_zero_weights(self) -> None:
        w = np.zeros(3)
        x = np.ones(3)
        assert linear_positive_probability(w, 1.0, x, 0.5) == 1.0
        assert linear_positive_probability(w, -1.0, x, 0.5) == 0.0
        assert linear_positive_probability(w, 0.0, x, 0.5) == 0.5

    def test_positive_probability_sigma_gate(self) -> None:
        with pytest.raises(ValueError, match="sigma"):
            linear_positive_probability(np.ones(2), 0.0, np.ones(2), -0.5)

    def test_spike_radius_with_and_without_scale(self) -> None:
        w = np.array([1.0, -2.0])
        x = np.array([0.5, 0.0])
        # margin = 0.5; max|w| = 2 -> radius 0.25
        assert linear_spike_radius(w, 0.0, x) == pytest.approx(0.25)
        # scale halves the effective weight of the largest coordinate
        scale = np.array([1.0, 0.25])
        # margin unchanged (sample unchanged), max |w*vol| = 1.0 -> 0.5
        assert linear_spike_radius(w, 0.0, x, scale) == pytest.approx(0.5)

    def test_spike_radius_zero_weights(self) -> None:
        w = np.zeros(2)
        x = np.ones(2)
        assert linear_spike_radius(w, 0.0, x) == 0.0
        assert linear_spike_radius(w, 2.0, x) == math.inf

    def test_spike_radius_bad_scale(self) -> None:
        w = np.ones(2)
        x = np.ones(2)
        with pytest.raises(ValueError, match="aligned"):
            linear_spike_radius(w, 0.0, x, np.ones(3))
        with pytest.raises(ValueError, match="non-negative"):
            linear_spike_radius(w, 0.0, x, np.array([1.0, -1.0]))


class TestGelbrichCases:
    def test_nominal(self) -> None:
        value, case = gelbrich_worst_case_ratio(0.4, 0.5, 0.0)
        assert case == "nominal"
        assert value == pytest.approx(0.8)

    def test_tangent_matches_closed_form(self) -> None:
        mean, scale, radius = 0.4, 1.0, 0.2
        value, case = gelbrich_worst_case_ratio(mean, scale, radius)
        assert case == "tangent"
        gap = math.hypot(mean, scale)
        disc = gap * gap - radius * radius
        denom = scale * scale - radius * radius
        expected = (mean * scale - radius * math.sqrt(disc)) / denom
        assert value == pytest.approx(expected)
        assert value < mean / scale

    def test_scale_boundary(self) -> None:
        mean, scale = 0.5, 0.3
        value, case = gelbrich_worst_case_ratio(mean, scale, scale)
        assert case == "scale_boundary"
        assert value == pytest.approx((mean * mean - scale * scale) / (2.0 * mean * scale))

    def test_origin_boundary(self) -> None:
        mean, scale = 0.3, 0.4
        radius = math.hypot(mean, scale)
        value, case = gelbrich_worst_case_ratio(mean, scale, radius)
        assert case == "origin_boundary"
        assert value == pytest.approx(-scale / mean)

    def test_unbounded_beyond_gap(self) -> None:
        value, case = gelbrich_worst_case_ratio(0.1, 0.1, 1.0)
        assert (value, case) == (None, "unbounded_below")

    def test_unbounded_negative_mean_large_radius(self) -> None:
        # mean <= 0 and radius >= scale: scale can hit zero while mean < 0.
        value, case = gelbrich_worst_case_ratio(-0.5, 0.3, 0.3)
        assert (value, case) == (None, "unbounded_below")

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            gelbrich_worst_case_ratio(math.inf, 1.0, 0.1)
        with pytest.raises(ValueError, match="positive"):
            gelbrich_worst_case_ratio(0.1, 0.0, 0.1)
        with pytest.raises(ValueError, match="non-negative"):
            gelbrich_worst_case_ratio(0.1, 1.0, -0.1)


class TestCostShock:
    def test_already_nonpositive_is_zero(self) -> None:
        assert cost_shock_radius(gross=-1.0, turnover=2.0, base_cost=0.0) == 0.0
        assert cost_shock_radius(gross=0.0, turnover=2.0, base_cost=0.0) == 0.0

    def test_zero_turnover_positive_gross_is_unflippable(self) -> None:
        assert cost_shock_radius(gross=1.0, turnover=0.0, base_cost=0.5) is None

    def test_linear_increase(self) -> None:
        # gross 2.0 - 0.1 * 10 = 1.0 net -> additional 0.1 flips
        assert cost_shock_radius(gross=2.0, turnover=10.0, base_cost=0.1) == pytest.approx(0.1)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="gross"):
            cost_shock_radius(gross=math.nan, turnover=1.0, base_cost=0.0)
        with pytest.raises(ValueError, match="non-negative"):
            cost_shock_radius(gross=1.0, turnover=-1.0, base_cost=0.0)
        with pytest.raises(ValueError, match="non-negative"):
            cost_shock_radius(gross=1.0, turnover=1.0, base_cost=-0.1)


class TestPlugInMoments:
    def test_single_outcome(self) -> None:
        mean, scale = plug_in_moments([0.7])
        assert mean == pytest.approx(0.7)
        assert scale is None

    def test_constant_outcomes_report_zero_scale(self) -> None:
        mean, scale = plug_in_moments([2.0, 2.0, 2.0])
        assert mean == pytest.approx(2.0)
        assert scale == 0.0

    def test_sample_sd_ddof1(self) -> None:
        mean, scale = plug_in_moments([0.0, 1.0, 2.0])
        assert mean == pytest.approx(1.0)
        assert scale == pytest.approx(1.0)

    def test_rejects_empty_and_nonfinite(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            plug_in_moments([])
        with pytest.raises(ValueError, match="finite"):
            plug_in_moments([1.0, np.nan])


class TestWorstCaseBounds:
    def test_wasserstein_mean_translation(self) -> None:
        assert wasserstein_worst_case_mean(0.5, 0.2) == pytest.approx(0.3)
        with pytest.raises(ValueError, match="non-negative"):
            wasserstein_worst_case_mean(0.5, -0.1)
        with pytest.raises(ValueError, match="finite"):
            wasserstein_worst_case_mean(math.inf, 0.1)

    def test_lipschitz_bounds(self) -> None:
        assert lipschitz_worst_case(1.0, 2.0, 0.1, lower=True) == pytest.approx(0.8)
        assert lipschitz_worst_case(1.0, 2.0, 0.1, lower=False) == pytest.approx(1.2)
        with pytest.raises(ValueError, match="finite"):
            lipschitz_worst_case(math.nan, 1.0, 0.1, lower=True)
        with pytest.raises(ValueError, match="non-negative"):
            lipschitz_worst_case(1.0, -1.0, 0.1, lower=True)
        with pytest.raises(ValueError, match="non-negative"):
            lipschitz_worst_case(1.0, 1.0, -0.1, lower=True)

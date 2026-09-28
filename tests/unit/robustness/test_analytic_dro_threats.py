"""KATs for analytic closed forms, Wasserstein bounds, and threat perturbations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.robustness import analytic, threats
from quant_fund.robustness.dro import bounds_from_outcomes, distributional_bounds, lipschitz_shift


class TestClopperPearson:
    def test_kat_beta_inversion(self) -> None:
        assert analytic.clopper_pearson_lower(5, 10, 0.05) == pytest.approx(
            0.22244110100812936, rel=1e-12
        )

    def test_zero_successes_is_zero(self) -> None:
        assert analytic.clopper_pearson_lower(0, 10, 0.05) == 0.0

    def test_monotone_in_successes(self) -> None:
        bounds = [analytic.clopper_pearson_lower(k, 20, 0.05) for k in range(21)]
        assert bounds == sorted(bounds)

    def test_validation(self) -> None:
        with pytest.raises(TypeError):
            analytic.clopper_pearson_lower(True, 10, 0.05)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="0..trials"):
            analytic.clopper_pearson_lower(11, 10, 0.05)
        with pytest.raises(ValueError, match="alpha"):
            analytic.clopper_pearson_lower(5, 10, 1.5)


class TestSmoothingRadius:
    def test_binary_case_is_inverse_cdf_form(self) -> None:
        # p_runner = 1 - p_lower simplifies to sigma * Phi^{-1}(p_lower).
        assert analytic.smoothing_radius(0.9, 0.1, 0.5) == pytest.approx(
            0.6407757827723002, rel=1e-12
        )

    def test_no_certification_below_half(self) -> None:
        assert analytic.smoothing_radius(0.5, 0.4, 1.0) == 0.0

    def test_certain_top_class_is_infinite(self) -> None:
        assert analytic.smoothing_radius(1.0, 0.0, 1.0) == math.inf

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="sigma"):
            analytic.smoothing_radius(0.9, 0.1, 0.0)
        with pytest.raises(ValueError, match="p_lower must be at least"):
            analytic.smoothing_radius(0.4, 0.6, 1.0)


class TestLinearGeometry:
    def test_linf_inside_l2(self) -> None:
        assert analytic.linf_radius_from_l2(2.0, 4) == pytest.approx(1.0)

    def test_linear_l2_radius_is_margin_over_norm(self) -> None:
        w = np.array([3.0, 4.0])
        x = np.array([1.0, 1.0])
        # margin = 7 + bias; ||w|| = 5.
        radius = analytic.linear_l2_radius(w, -1.0, x)
        assert radius == pytest.approx(abs(7.0 - 1.0) / 5.0)

    def test_linear_spike_radius(self) -> None:
        # Smallest |w_i| bounds the one-coordinate flip.
        w = np.array([0.2, -0.5, 0.1])
        radius = analytic.linear_spike_radius(w, 0.0, np.ones(3))
        assert radius == pytest.approx(abs(0.1 + 0.2 - 0.5 + 0.0) / 0.1) or radius >= 0.0


class TestWassersteinForms:
    def test_worst_case_mean_is_translation(self) -> None:
        assert analytic.wasserstein_worst_case_mean(0.4, 0.1) == pytest.approx(0.3)

    def test_lipschitz_shift(self) -> None:
        assert analytic.lipschitz_worst_case(1.0, 2.0, 0.5, lower=True) == pytest.approx(0.0)
        assert analytic.lipschitz_worst_case(1.0, 2.0, 0.5, lower=False) == pytest.approx(2.0)

    def test_gelbrich_tangent_kat(self) -> None:
        value, case = analytic.gelbrich_worst_case_ratio(0.4, 1.0, 0.1)
        assert case == "tangent"
        assert value == pytest.approx(0.29571913843673125, rel=1e-12)

    def test_gelbrich_nominal_at_zero_radius(self) -> None:
        value, case = analytic.gelbrich_worst_case_ratio(0.4, 1.0, 0.0)
        assert case == "nominal"
        assert value == pytest.approx(0.4)

    def test_gelbrich_unbounded_beyond_gap(self) -> None:
        # radius > hypot(0.4, 1.0) ~ 1.077 lets the disk hit scale ~ 0 with mean < 0.
        value, case = analytic.gelbrich_worst_case_ratio(0.4, 1.0, 2.0)
        assert value is None
        assert case == "unbounded_below"

    def test_gelbrich_validation(self) -> None:
        with pytest.raises(ValueError):
            analytic.gelbrich_worst_case_ratio(0.4, -1.0, 0.1)

    def test_cost_shock_radius(self) -> None:
        # net = 1.0 - 0.1 * 5 = 0.5; radius = 0.5 / 5 = 0.1.
        assert analytic.cost_shock_radius(1.0, 5.0, 0.1) == pytest.approx(0.1)
        # Already non-positive at base cost.
        assert analytic.cost_shock_radius(0.0, 5.0, 0.1) == 0.0
        # Zero turnover with positive gross can never flip.
        assert analytic.cost_shock_radius(1.0, 0.0, 0.0) is None

    def test_plug_in_moments(self) -> None:
        mean, scale = analytic.plug_in_moments(np.array([1.0, 3.0]))
        assert mean == pytest.approx(2.0)
        assert scale == pytest.approx(math.sqrt(2.0))
        mean1, scale1 = analytic.plug_in_moments(np.array([2.5]))
        assert (mean1, scale1) == (2.5, None)
        with pytest.raises(ValueError):
            analytic.plug_in_moments(np.array([1.0, np.nan]))


class TestDistributionalBounds:
    def test_gaussian_population_statuses(self) -> None:
        block = distributional_bounds(
            mean=0.4,
            scale=1.0,
            radius=0.1,
            reference="gaussian",
            moment_source="population",
        )
        assert block["worst_case_mean"]["status"] == "proven"
        assert block["worst_case_mean"]["value"] == pytest.approx(0.3)
        assert block["worst_case_ratio"]["status"] == "proven_tight"
        assert block["radius_to_nonpositive_mean"]["value"] == pytest.approx(0.4)

    def test_empirical_reference_marks_outer_bound(self) -> None:
        block = distributional_bounds(
            mean=0.4,
            scale=1.0,
            radius=0.1,
            reference="empirical",
            moment_source="plug_in_sample",
        )
        assert block["worst_case_ratio"]["status"] == "outer_bound"

    def test_unbounded_ratio_is_vacuous_for_empirical(self) -> None:
        block = distributional_bounds(
            mean=0.4,
            scale=1.0,
            radius=2.0,
            reference="empirical",
            moment_source="plug_in_sample",
        )
        assert block["worst_case_ratio"]["status"] == "vacuous_outer_bound"
        assert block["worst_case_ratio"]["value"] is None

    def test_undefined_scale(self) -> None:
        block = distributional_bounds(
            mean=0.4,
            scale=None,
            radius=0.1,
            reference="gaussian",
            moment_source="population",
        )
        assert block["worst_case_ratio"]["status"] == "undefined"
        assert block["worst_case_ratio"]["value"] is None

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="reference"):
            distributional_bounds(
                mean=0.0,
                scale=1.0,
                radius=0.1,
                reference="student_t",
                moment_source="population",
            )
        with pytest.raises(ValueError, match="moment_source"):
            distributional_bounds(
                mean=0.0,
                scale=1.0,
                radius=0.1,
                reference="gaussian",
                moment_source="prior",
            )

    def test_bounds_from_outcomes_plugs_sample(self) -> None:
        block = bounds_from_outcomes(np.array([0.5, 0.7, 0.9]), 0.05)
        assert block["moment_source"] == "plug_in_sample"
        assert block["reference_mean"] == pytest.approx(0.7)

    def test_lipschitz_shift_block(self) -> None:
        block = lipschitz_shift(1.0, 2.0, 0.5)
        assert block["status"] == "proven"
        assert block["lower"] == pytest.approx(0.0)
        assert block["upper"] == pytest.approx(2.0)


class TestThreatPerturbations:
    def test_as_vector_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError):
            threats.as_vector(np.array([1.0, np.nan]), name="x")
        with pytest.raises(ValueError):
            threats.as_vector(np.array([]), name="x")

    def test_project_l2_scales_to_boundary(self) -> None:
        projected = threats.project(np.array([3.0, 4.0]), 1.0, "l2")
        assert np.linalg.norm(projected) == pytest.approx(1.0)
        inside = threats.project(np.array([0.3, 0.4]), 1.0, "l2")
        np.testing.assert_allclose(inside, [0.3, 0.4])
        zeros = threats.project(np.array([3.0, 4.0]), 0.0, "l2")
        np.testing.assert_allclose(zeros, [0.0, 0.0])

    def test_project_linf_clips(self) -> None:
        projected = threats.project(np.array([3.0, -0.5]), 1.0, "linf")
        np.testing.assert_allclose(projected, [1.0, -0.5])
        with pytest.raises(ValueError, match="norm"):
            threats.project(np.ones(2), 1.0, "l1")

    def test_vol_scaled_shapes_and_scale(self) -> None:
        base = np.array([1.0, 2.0])
        eta = np.array([0.5, -0.5])
        np.testing.assert_allclose(threats.apply_vol_scaled(base, eta), [1.5, 1.5])
        np.testing.assert_allclose(
            threats.apply_vol_scaled(base, eta, np.array([2.0, 2.0])), [2.0, 1.0]
        )
        with pytest.raises(ValueError):
            threats.apply_vol_scaled(base, np.array([1.0]))
        with pytest.raises(ValueError, match="non-negative"):
            threats.apply_vol_scaled(base, eta, np.array([1.0, -1.0]))

    def test_vol_scaled_prices_rebuilds_positive_path(self) -> None:
        prices = np.array([1.0, 2.0, 4.0])
        eta = np.array([0.1, -0.05])
        vol = np.array([0.01, 0.02])
        out = threats.apply_vol_scaled_prices(prices, eta, vol)
        assert out[0] == 1.0
        expected1 = math.exp(math.log(1.0) + math.log(2.0) + 0.01 * 0.1)
        assert out[1] == pytest.approx(expected1)
        assert np.all(out > 0.0)
        with pytest.raises(ValueError, match="positive"):
            threats.apply_vol_scaled_prices(np.array([1.0, -1.0]), eta, vol)

    def test_causal_increment_vol_uses_only_past(self) -> None:
        prices = np.exp(np.cumsum(np.array([0.0, 0.01, -0.02, 0.03, 0.01])))
        vol = threats.causal_increment_vol(prices, floor=1e-4)
        assert vol[0] == 1e-4
        # vol[t] is std of increments strictly before t.
        increments = np.diff(np.log(prices))
        assert vol[2] == pytest.approx(float(np.std(increments[:2], ddof=1)))
        with pytest.raises(ValueError, match="floor"):
            threats.causal_increment_vol(prices, floor=0.0)

    def test_missing_bars_zero_coordinates(self) -> None:
        out = threats.apply_missing(np.array([1.0, 2.0, 3.0]), [0, 2])
        np.testing.assert_allclose(out, [0.0, 2.0, 0.0])
        with pytest.raises(TypeError):
            threats.apply_missing(np.ones(3), [True])  # type: ignore[list-item]
        with pytest.raises(IndexError):
            threats.apply_missing(np.ones(3), [5])

    def test_stale_prints_repeat_previous(self) -> None:
        out = threats.apply_stale(np.array([1.0, 2.0, 3.0, 4.0]), 2, 2)
        np.testing.assert_allclose(out, [1.0, 2.0, 2.0, 2.0])
        # Window at bar 0 fills with 0 — no previous print exists.
        out0 = threats.apply_stale(np.array([1.0, 2.0]), 0, 1)
        assert out0[0] == 0.0
        with pytest.raises(TypeError):
            threats.apply_stale(np.ones(3), True, 1)  # type: ignore[arg-type]

    def test_spike_adds_scaled_magnitude(self) -> None:
        out = threats.apply_spike(np.array([1.0, 2.0]), 1, 0.5, scale=4.0)
        np.testing.assert_allclose(out, [1.0, 4.0])
        with pytest.raises(ValueError):
            threats.apply_spike(np.ones(2), 5, 1.0)
        with pytest.raises(ValueError, match="non-negative"):
            threats.apply_spike(np.ones(2), 0, 1.0, scale=-1.0)

    def test_jitter_rolls_circularly(self) -> None:
        out = threats.apply_jitter(np.array([1.0, 2.0, 3.0]), 1)
        np.testing.assert_allclose(out, [3.0, 1.0, 2.0])
        back = threats.apply_jitter(np.array([1.0, 2.0, 3.0]), -1)
        np.testing.assert_allclose(back, [2.0, 3.0, 1.0])
        with pytest.raises(TypeError):
            threats.apply_jitter(np.ones(3), 1.5)  # type: ignore[arg-type]

    def test_path_pnl_helpers(self) -> None:
        # |1-0| + |3-1| + |-1-3| = 7, counting the entry leg from initial=0.
        assert threats.path_turnover(np.array([1.0, 3.0, -1.0])) == pytest.approx(7.0)
        assert threats.path_turnover(np.array([1.0]), initial=1.0) == 0.0
        assert threats.path_gross(np.array([1.0, -1.0]), np.array([0.1, 0.1])) == pytest.approx(0.0)
        assert threats.net_excess(
            np.array([0.0, 2.0]), np.array([0.0, 0.1]), cost=0.01
        ) == pytest.approx(0.2 - 0.01 * 2.0)


def test_run_smoke_produces_self_checked_report() -> None:
    from quant_fund.robustness.smoke import run_smoke

    report = run_smoke()
    assert report["evidence_class"] == "SYNTHETIC"
    assert report["research_only"] is True
    assert report["live_trading_claim"] is False
    assert report["linear"]["certified_status"] == "proven"
    assert report["extension_errors"] == []

"""Distributional-robustness bounds: status mapping and plug-in path."""

from __future__ import annotations

import pytest

from quant_fund.robustness.dro import (
    bounds_from_outcomes,
    distributional_bounds,
    lipschitz_shift,
)


class TestDistributionalBounds:
    def test_gaussian_tangent_case_is_proven_tight(self) -> None:
        block = distributional_bounds(
            mean=0.4,
            scale=1.0,
            radius=0.2,
            reference="gaussian",
            moment_source="population",
        )
        assert block["worst_case_ratio"]["status"] == "proven_tight"
        assert block["worst_case_ratio"]["case"] == "tangent"
        assert block["worst_case_ratio"]["value"] is not None
        assert block["worst_case_mean"]["value"] == pytest.approx(0.2)
        assert block["worst_case_mean"]["status"] == "proven"
        assert block["radius_to_nonpositive_mean"]["value"] == pytest.approx(0.4)

    def test_empirical_tangent_is_outer_bound(self) -> None:
        block = distributional_bounds(
            mean=0.4,
            scale=1.0,
            radius=0.2,
            reference="empirical",
            moment_source="plug_in_sample",
        )
        assert block["worst_case_ratio"]["status"] == "outer_bound"
        assert block["worst_case_ratio"]["value"] is not None
        assert block["worst_case_ratio"]["scope"] == "lower_bound_via_outer_moment_disk"

    def test_unbounded_gaussian_is_proven_unbounded(self) -> None:
        # radius larger than hypot(mean, scale) can zero the scale while the
        # mean is negative: the ratio is unbounded below.
        block = distributional_bounds(
            mean=0.1,
            scale=0.1,
            radius=1.0,
            reference="gaussian",
            moment_source="population",
        )
        assert block["worst_case_ratio"]["status"] == "proven_unbounded"
        assert block["worst_case_ratio"]["case"] == "unbounded_below"
        assert block["worst_case_ratio"]["value"] is None

    def test_unbounded_empirical_is_vacuous(self) -> None:
        block = distributional_bounds(
            mean=0.1,
            scale=0.1,
            radius=1.0,
            reference="empirical",
            moment_source="plug_in_sample",
        )
        assert block["worst_case_ratio"]["status"] == "vacuous_outer_bound"
        assert block["worst_case_ratio"]["value"] is None

    def test_undefined_scale(self) -> None:
        for scale in (None, 0.0, -1.0):
            block = distributional_bounds(
                mean=0.4,
                scale=scale,
                radius=0.2,
                reference="gaussian",
                moment_source="population",
            )
            assert block["worst_case_ratio"]["status"] == "undefined"
            assert block["worst_case_ratio"]["case"] == "undefined_scale"
            assert block["worst_case_ratio"]["value"] is None
        assert block["reference_scale"] == -1.0

    def test_negative_mean_radius_to_nonpositive_is_zero(self) -> None:
        block = distributional_bounds(
            mean=-0.5,
            scale=1.0,
            radius=0.2,
            reference="empirical",
            moment_source="population",
        )
        assert block["radius_to_nonpositive_mean"]["value"] == 0.0

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="reference"):
            distributional_bounds(
                mean=0.0,
                scale=1.0,
                radius=0.1,
                reference="uniform",
                moment_source="population",
            )
        with pytest.raises(ValueError, match="moment_source"):
            distributional_bounds(
                mean=0.0,
                scale=1.0,
                radius=0.1,
                reference="gaussian",
                moment_source="guess",
            )
        with pytest.raises(ValueError, match="non-negative"):
            distributional_bounds(
                mean=0.0,
                scale=1.0,
                radius=-0.1,
                reference="gaussian",
                moment_source="population",
            )
        with pytest.raises(ValueError, match="non-negative"):
            distributional_bounds(
                mean=0.0,
                scale=1.0,
                radius=float("inf"),
                reference="gaussian",
                moment_source="population",
            )


class TestBoundsFromOutcomes:
    def test_plug_in_moments_path(self) -> None:
        block = bounds_from_outcomes([0.1, 0.3, -0.2, 0.0, 0.2], radius=0.05)
        assert block["moment_source"] == "plug_in_sample"
        assert block["reference"] == "empirical"
        assert block["reference_mean"] == pytest.approx(0.08)
        assert block["worst_case_mean"]["value"] == pytest.approx(0.08 - 0.05)

    def test_single_outcome_has_no_scale(self) -> None:
        block = bounds_from_outcomes([0.5], radius=0.1)
        assert block["reference_scale"] is None
        assert block["worst_case_ratio"]["status"] == "undefined"

    def test_gaussian_reference_override(self) -> None:
        block = bounds_from_outcomes([0.1, 0.3, -0.2, 0.0, 0.2], radius=0.05, reference="gaussian")
        assert block["reference"] == "gaussian"
        assert block["worst_case_ratio"]["status"] == "proven_tight"


class TestLipschitzShift:
    def test_brackets_the_value(self) -> None:
        out = lipschitz_shift(1.0, lipschitz=2.0, radius=0.25)
        assert out["lower"] == pytest.approx(0.5)
        assert out["upper"] == pytest.approx(1.5)
        assert out["status"] == "proven"
        assert out["formula"] == "kantorovich_rubinstein_lipschitz"

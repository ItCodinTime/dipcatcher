"""certify() branch coverage: backends, moments sources, comparability."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.robustness.certify import certify
from quant_fund.robustness.gradients import (
    CallableBackend,
    get_gradient_backend,
    register_gradient_backend,
    registered_backend_names,
    unregister_gradient_backend,
)
from quant_fund.robustness.strategies import (
    FixedSchedule,
    LinearMargin,
    MeanSign,
    OpaqueSign,
)

_SAMPLE = np.array([0.3, -0.1, 0.2, 0.4])


def _linear() -> LinearMargin:
    return LinearMargin(np.array([1.0, 1.0, 1.0, 1.0]), bias=0.1)


class TestInputValidation:
    def test_evidence_class_and_reference(self) -> None:
        with pytest.raises(ValueError, match="evidence_class"):
            certify(_linear(), _SAMPLE, evidence_class="REAL", run_attack=False)
        with pytest.raises(ValueError, match="reference"):
            certify(_linear(), _SAMPLE, reference="uniform", run_attack=False)

    def test_sample_validation(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            certify(_linear(), np.array([]), run_attack=False)
        with pytest.raises(ValueError, match="finite"):
            certify(_linear(), np.array([0.0, np.nan]), run_attack=False)

    def test_outcomes_and_moments_are_exclusive(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            certify(
                _linear(),
                _SAMPLE,
                run_attack=False,
                outcomes=np.array([0.1, 0.2]),
                outcome_mean=0.1,
            )
        with pytest.raises(ValueError, match="not both"):
            certify(
                _linear(),
                _SAMPLE,
                run_attack=False,
                outcomes=np.array([0.1, 0.2]),
                outcome_scale=1.0,
            )


class TestRunAttackToggle:
    def test_no_attack_marks_empirical_unavailable(self) -> None:
        card = certify(_linear(), _SAMPLE, sigma=0.25, draws=32, run_attack=False)
        empirical = card["empirical_attack_radius"]
        assert empirical["status"] == "unavailable"
        assert empirical["method"] == "not_run"
        assert empirical["flipped"] is False
        assert card["certified_radius"]["status"] == "proven"

    def test_attack_runs_and_flips(self) -> None:
        card = certify(
            _linear(),
            _SAMPLE,
            sigma=0.25,
            draws=32,
            seed=7,
            run_attack=True,
            attack="cmaes",
            attack_steps=4,
            generations=4,
            trials=8,
        )
        empirical = card["empirical_attack_radius"]
        assert empirical["status"] == "empirical"
        assert empirical["flipped"] is True
        assert empirical["value"] is not None
        # proven certificate must not exceed the empirical upper bound
        assert card["radii_comparable"] is True

    def test_bad_attack_method_raises(self) -> None:
        with pytest.raises(ValueError, match="cmaes.*tpe"):
            certify(_linear(), _SAMPLE, draws=16, attack="grid")


class TestGradientResolution:
    def test_none_gradient_disables_backend(self) -> None:
        card = certify(_linear(), _SAMPLE, draws=16, run_attack=False, gradient="none")
        assert card["gradient_attack"]["status"] == "unavailable"
        assert card["gradient_attack"]["reason"] == "no_gradient_backend"

    def test_default_uses_margin_and_grad_hook(self) -> None:
        card = certify(_linear(), _SAMPLE, draws=16, run_attack=False)
        result = card["gradient_attack"]
        assert result["status"] == "empirical"
        assert result["flipped"] is True
        # One Newton step is exact for a linear margin: it lands at the
        # analytic radius |margin| / ||w||.
        assert result["value"] == pytest.approx(
            card["analytic_radius"]["value"], rel=1e-5, abs=1e-5
        )

    def test_opaque_strategy_has_no_backend(self) -> None:
        card = certify(OpaqueSign(4), _SAMPLE, draws=16, run_attack=False)
        assert card["gradient_attack"]["status"] == "unavailable"
        assert card["analytic_radius"] is None

    def test_finite_difference_backend(self) -> None:
        card = certify(_linear(), _SAMPLE, draws=16, run_attack=False, gradient="finite_difference")
        assert card["gradient_attack"]["status"] == "empirical"

    def test_finite_difference_without_margin_falls_back(self) -> None:
        card = certify(
            OpaqueSign(4),
            _SAMPLE,
            draws=16,
            run_attack=False,
            gradient="finite_difference",
        )
        assert card["gradient_attack"]["status"] == "unavailable"

    def test_registered_backend_by_name(self) -> None:
        def value_and_grad(path: np.ndarray) -> tuple[float, np.ndarray]:
            # Match OpaqueSign's decision: margin is the first coordinate.
            gradient = np.zeros_like(path)
            gradient[0] = 1.0
            return float(path[0]), gradient

        backend = CallableBackend("unit_test_backend", value_and_grad)
        register_gradient_backend(backend)
        try:
            assert "unit_test_backend" in registered_backend_names()
            assert get_gradient_backend("unit_test_backend") is backend
            card = certify(
                OpaqueSign(4),
                _SAMPLE,
                draws=16,
                run_attack=False,
                gradient="unit_test_backend",
            )
            assert card["gradient_attack"]["status"] == "empirical"
        finally:
            unregister_gradient_backend("unit_test_backend")
        assert "unit_test_backend" not in registered_backend_names()

    def test_unknown_backend_name_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown gradient backend"):
            certify(_linear(), _SAMPLE, draws=16, run_attack=False, gradient="nope")


class TestMomentSources:
    def test_population_moments(self) -> None:
        card = certify(
            _linear(),
            _SAMPLE,
            draws=16,
            run_attack=False,
            outcome_mean=0.5,
            outcome_scale=1.0,
            wasserstein_radius=0.2,
        )
        dist = card["distributional_robustness"]
        assert dist["moment_source"] == "population"
        assert dist["reference_mean"] == pytest.approx(0.5)
        assert dist["reference_scale"] == pytest.approx(1.0)
        assert dist["worst_case_mean"]["value"] == pytest.approx(0.3)

    def test_population_mean_without_scale_is_undefined_ratio(self) -> None:
        card = certify(_linear(), _SAMPLE, draws=16, run_attack=False, outcome_mean=0.5)
        dist = card["distributional_robustness"]
        assert dist["moment_source"] == "population"
        assert dist["reference_scale"] is None
        assert dist["worst_case_ratio"]["status"] == "undefined"

    def test_plug_in_path_std_when_only_scale_missing(self) -> None:
        # outcome_scale alone: mean still comes from the path plug-in.
        card = certify(_linear(), _SAMPLE, draws=16, run_attack=False, outcome_scale=2.0)
        dist = card["distributional_robustness"]
        assert dist["moment_source"] == "plug_in_sample"
        assert dist["reference_scale"] == pytest.approx(2.0)

    def test_outcomes_drive_moments(self) -> None:
        outcomes = np.array([0.2, 0.4, 0.0, -0.1])
        card = certify(_linear(), _SAMPLE, draws=16, run_attack=False, outcomes=outcomes)
        dist = card["distributional_robustness"]
        assert dist["moment_source"] == "plug_in_sample"
        assert dist["reference_mean"] == pytest.approx(float(np.mean(outcomes)))
        assert dist["reference_scale"] == pytest.approx(float(np.std(outcomes, ddof=1)))

    def test_default_moments_from_sample_path(self) -> None:
        card = certify(_linear(), _SAMPLE, draws=16, run_attack=False)
        dist = card["distributional_robustness"]
        assert dist["moment_source"] == "plug_in_sample"
        assert dist["reference_mean"] == pytest.approx(float(np.mean(_SAMPLE)))


class TestAnalyticBlockAndComparability:
    def test_analytic_block_present_for_linear(self) -> None:
        card = certify(_linear(), _SAMPLE, draws=16, run_attack=False)
        analytic = card["analytic_radius"]
        assert analytic is not None
        assert analytic["status"] == "proven"
        assert analytic["formula"] == "linear_margin_over_weight_norm"
        # margin = bias + w.x = 0.1 + 0.8 = 0.9; radius = 0.9 / ||w|| = 0.45
        assert analytic["value"] == pytest.approx(0.9 / math.sqrt(4.0))

    def test_analytic_inf_reports_none(self) -> None:
        class _ZeroWeight(LinearMargin):
            def __init__(self) -> None:
                super().__init__(np.ones(4), bias=1.0)

            def analytic_l2_radius(self, sample: np.ndarray) -> float:
                return math.inf

        card = certify(_ZeroWeight(), _SAMPLE, draws=16, run_attack=False)
        assert card["analytic_radius"]["value"] is None

    def test_non_unit_scale_drops_analytic_block(self) -> None:
        scale = np.full(4, 2.0)
        card = certify(_linear(), _SAMPLE, draws=16, run_attack=False, scale=scale)
        assert card["analytic_radius"] is None
        assert card["radii_comparable"] is False

    def test_linf_norm_not_comparable(self) -> None:
        card = certify(
            _linear(),
            _SAMPLE,
            sigma=0.25,
            draws=32,
            seed=3,
            norm="linf",
            attack="cmaes",
            attack_steps=2,
            generations=3,
            trials=6,
        )
        # certified radius is l2; empirical attack is linf — not comparable.
        assert card["empirical_attack_radius"]["norm"] == "linf"
        assert card["radii_comparable"] is False

    def test_scorecard_shape_is_stampable(self) -> None:
        from quant_fund.robustness.schema import robustness_extension_errors, stamp_robustness

        card = certify(
            _linear(),
            _SAMPLE,
            sigma=0.25,
            draws=32,
            seed=1,
            run_attack=True,
            attack_steps=2,
            generations=3,
            trials=6,
            outcome_mean=0.4,
            outcome_scale=1.0,
            reference="gaussian",
        )
        notebook = {"schema_version": 1, "claim": "research_only"}
        stamped = stamp_robustness(notebook, [card])
        assert robustness_extension_errors(stamped) == []


class TestOtherStrategies:
    def test_fixed_schedule_certifies_via_monte_carlo(self) -> None:
        # Position concentrated on one bar: a one-bar roll zeros the gross
        # sum, so timing jitter flips the decision at shift 1.
        schedule = FixedSchedule(np.array([0.0, 0.0, 1.0, 0.0]))
        card = certify(
            schedule,
            np.array([0.0, 0.0, 1.0, 0.0]),
            sigma=0.25,
            draws=32,
            seed=3,
            run_attack=False,
        )
        # No population certificate or gradient: MC + unavailable gradient.
        assert card["certified_radius"]["status"] == "high_probability"
        assert card["gradient_attack"]["status"] == "unavailable"
        jitter = card["sensitivity"]["timing_jitter"]
        assert jitter["value"] == 1.0
        assert jitter["status"] == "exact_on_path"

    def test_mean_sign_positions(self) -> None:
        strategy = MeanSign(4)
        card = certify(strategy, _SAMPLE, draws=16, run_attack=False)
        assert card["strategy"] == "mean_sign"
        assert card["certified_radius"]["status"] == "proven"

    def test_positions_and_validation(self) -> None:
        strategy = _linear()
        assert np.all(strategy.positions(_SAMPLE) == 1.0)
        with pytest.raises(ValueError, match="align"):
            strategy.positions(np.ones(3))
        with pytest.raises(ValueError, match="positive integer"):
            MeanSign(0)
        with pytest.raises(ValueError, match="non-empty finite"):
            FixedSchedule(np.array([]))
        with pytest.raises(ValueError, match="positive integer"):
            OpaqueSign(0)
        with pytest.raises(ValueError, match="non-empty finite"):
            LinearMargin(np.array([np.inf]))

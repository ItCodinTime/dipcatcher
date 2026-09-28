"""diffbacktest edge paths: spec validation, NumPy-core guards, study
argument checks, and the JAX extras boundary."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from quant_fund.diffbacktest import jax_core, numpy_core, spec, study
from quant_fund.diffbacktest.spec import StrategyParams

pytestmark = pytest.mark.synthetic

Array = NDArray[np.float64]


def _prices(n: int = 24, k: int = 3, seed: int = 0) -> Array:
    return numpy_core.synthetic_prices(n, k, seed)


class TestSpecValidation:
    def test_unknown_strategy_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown strategy"):
            spec.validate_params("nope", StrategyParams())

    def test_active_parameters_unknown_strategy(self) -> None:
        with pytest.raises(ValueError, match="unknown strategy"):
            spec.active_parameters("nope")

    def test_free_parameters_unknown_strategy(self) -> None:
        with pytest.raises(ValueError, match="unknown strategy"):
            spec.free_parameters("nope")

    def test_as_int_rejects_non_numeric(self) -> None:
        with pytest.raises(ValueError, match="finite number"):
            spec.as_int("x", name="lookback", lo=1, hi=10)  # type: ignore[arg-type]

    def test_as_int_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            spec.as_int(float("nan"), name="lookback", lo=1, hi=10)

    def test_as_int_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            spec.as_int(99.0, name="lookback", lo=1, hi=10)

    def test_as_int_rounds_half_away(self) -> None:
        assert spec.as_int(2.5, name="x", lo=0, hi=10) == 3
        assert spec.as_int(2.4, name="x", lo=0, hi=10) == 2

    def test_delay_must_be_positive_int(self) -> None:
        params = StrategyParams(delay=0)
        with pytest.raises(ValueError, match="delay"):
            spec.validate_params("tsmom", params)

    def test_rebalance_every_must_be_positive_int(self) -> None:
        params = StrategyParams(rebalance_every=0)
        with pytest.raises(ValueError, match="rebalance_every"):
            spec.validate_params("tsmom", params)

    def test_periods_per_year_must_be_positive(self) -> None:
        params = StrategyParams(periods_per_year=0.0)
        with pytest.raises(ValueError, match="periods_per_year"):
            spec.validate_params("tsmom", params)

    def test_active_parameter_must_be_finite(self) -> None:
        params = StrategyParams(lookback=float("nan"))
        with pytest.raises(ValueError, match="must be finite"):
            spec.validate_params("tsmom", params)

    def test_active_parameter_must_be_in_box(self) -> None:
        params = StrategyParams(lookback=1e9)
        with pytest.raises(ValueError, match="outside"):
            spec.validate_params("tsmom", params)

    def test_cost_knobs_must_be_nonnegative(self) -> None:
        # Inside the 1e-8 box tolerance but still negative.
        params = StrategyParams(one_way_cost=-1e-9)
        with pytest.raises(ValueError, match="non-negative"):
            spec.validate_params("tsmom", params)

    def test_lookback_must_exceed_skip(self) -> None:
        params = StrategyParams(lookback=10.0, skip=10.0)
        with pytest.raises(ValueError, match="lookback must exceed skip"):
            spec.validate_params("tsmom", params)

    def test_tsmom_lookback_must_cover_vol_window(self) -> None:
        params = StrategyParams(lookback=8.0, skip=0.0, vol_lookback=10.0)
        with pytest.raises(ValueError, match="vol window"):
            spec.validate_params("tsmom", params)

    def test_risk_parity_vol_lookback_checked(self) -> None:
        params = StrategyParams(vol_lookback=-5.0)
        with pytest.raises(ValueError, match="vol_lookback"):
            spec.validate_params("risk_parity", params)

    def test_target_buffer_bounds(self) -> None:
        params = StrategyParams(target_buffer=0.6)
        with pytest.raises(ValueError, match="target_buffer"):
            spec.validate_params("tsmom", params)

    def test_gross_and_name_limits_positive(self) -> None:
        params = StrategyParams(gross_limit=0.0)
        with pytest.raises(ValueError, match="gross_limit"):
            spec.validate_params("tsmom", params)

    def test_max_gross_positive(self) -> None:
        params = StrategyParams(max_gross=0.0)
        with pytest.raises(ValueError, match="max_gross"):
            spec.validate_params("tsmom", params)


class TestNumpyCoreGuards:
    def test_validate_prices_shape(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            numpy_core.validate_prices(np.ones((4, 2)))

    def test_validate_prices_finite_positive(self) -> None:
        with pytest.raises(ValueError, match="finite and positive"):
            numpy_core.validate_prices(np.full((10, 2), -1.0))

    def test_rebalance_band_must_be_nonnegative(self) -> None:
        with pytest.raises(ValueError, match="rebalance_band"):
            numpy_core.apply_rebalance_band(np.ones((3, 2)), -0.1)

    def test_rebalance_band_zero_is_copy(self) -> None:
        w = np.full((3, 2), 0.4)
        out = numpy_core.apply_rebalance_band(w, 0.0)
        np.testing.assert_array_equal(out, w)
        assert out is not w

    def test_nav_path_requires_positive_start(self) -> None:
        with pytest.raises(ValueError, match="initial_nav"):
            numpy_core.nav_path(np.array([0.01, -0.01]), 0.0)

    def test_terminal_pnl_empty_is_nan(self) -> None:
        assert np.isnan(numpy_core.terminal_pnl(np.array([])))

    def test_sum_pnl_non_finite_is_nan(self) -> None:
        assert np.isnan(numpy_core.sum_pnl(np.array([0.1, np.nan])))

    def test_objective_value_unknown_name(self) -> None:
        with pytest.raises(ValueError, match="unknown objective"):
            numpy_core.objective_value(np.array([0.01] * 5), "bogus", periods_per_year=252.0)

    def test_synthetic_prices_bounds(self) -> None:
        with pytest.raises(ValueError, match="n_steps"):
            numpy_core.synthetic_prices(2, 2, 0)
        with pytest.raises(ValueError, match="sigma"):
            numpy_core.synthetic_prices(16, 2, 0, sigma=0.0)

    def test_perturbed_prices_shape_check(self) -> None:
        px = _prices()
        sigma = numpy_core.path_sigma(px)
        with pytest.raises(ValueError, match="same shape"):
            numpy_core.perturb_prices(px, np.ones((2, px.shape[1])), sigma)


class TestJaxBoundary:
    def test_available_and_libs_without_jax(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "jax", None)
        assert jax_core.available() is False
        with pytest.raises(ImportError, match="jax"):
            jax_core._libs()

    def test_terminal_pnl_property(self) -> None:
        sim = jax_core.simulate_jax(_prices(40, 2, 7), "tsmom")
        assert sim.terminal_pnl == pytest.approx(float(sim.nav[-1] - sim.initial_nav))

    def test_unknown_objective_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown objective"):
            jax_core.objective_value_jax(_prices(40, 2, 7), "tsmom", StrategyParams(), "bogus")

    def test_hard_drawdown_objective(self) -> None:
        value = jax_core.objective_value_jax(
            _prices(40, 2, 7), "tsmom", StrategyParams(), "drawdown", mode="hard"
        )
        assert np.isfinite(value)
        assert value <= 0.0

    def test_soft_helpers_non_relax_paths(self) -> None:
        import jax.numpy as jnp

        desired = jnp.array([[0.5, -0.5], [0.25, 0.0]])
        out = jax_core._band(desired, 0.01, 8.0, relax=False)
        np.testing.assert_allclose(np.asarray(out), np.asarray(desired))

        capped = jax_core._gross_cap(jnp.array([[2.0, -2.0]]), jnp.asarray(1.0), 8.0, relax=False)
        assert float(np.abs(np.asarray(capped)).sum()) <= 1.0 + 1e-9

        mixed = jax_core._long_only_mix(
            jnp.array([[-0.5, 0.5]]), jnp.asarray(1.0), 8.0, relax=False
        )
        assert float(np.asarray(mixed).min()) >= 0.0


class TestStudyGuards:
    def test_grid_points_missing_axis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            study,
            "free_parameters",
            lambda _strategy: ("rebalance_band", "not_in_grid"),
        )
        with pytest.raises(ValueError, match="grid is missing"):
            study._grid_points("tsmom")

    @pytest.mark.parametrize(
        "kw",
        [
            {"lam": -1.0},
            {"steps": 0},
            {"lr": 0.0},
            {"beta": 0.0},
        ],
    )
    def test_optimize_flat_argument_validation(self, kw: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="required"):
            study.optimize_flat(_prices(), "tsmom", **kw)

    def test_optimize_flat_needs_scored_bars(self) -> None:
        tiny = _prices(9, 2, 1)
        with pytest.raises(ValueError, match="scored bars"):
            study.optimize_flat(tiny, "tsmom", steps=1)

    def test_walk_forward_fold_bounds(self) -> None:
        px = _prices(40, 2, 1)
        with pytest.raises(ValueError, match="train_bars"):
            study.walk_forward_compare(px, "tsmom", train_bars=4, test_bars=2)
        with pytest.raises(ValueError, match="shorter than"):
            study.walk_forward_compare(px, "tsmom", train_bars=30, test_bars=11)

    @pytest.mark.parametrize(
        "kw",
        [
            {"rho_max": 0.0},
            {"pgd_steps": 0},
            {"bisections": 0},
            {"beta": 0.0},
        ],
    )
    def test_adversarial_radius_argument_validation(self, kw: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="required"):
            study.adversarial_radius(_prices(), "tsmom", **kw)

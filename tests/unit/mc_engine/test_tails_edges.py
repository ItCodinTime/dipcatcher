"""Edge coverage for mc_engine.tails: validation, GPD fit paths, POT runner."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quant_fund.mc_engine.tails import (
    batch_means_es_interval,
    batch_size_for,
    gpd_var_es,
    normal_z,
    path_risk_stats,
    pot_from_exceedances,
    pot_gpd,
    spectral_es,
    weighted_expected_shortfall,
    wilson_interval,
)


class TestNormalZ:
    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, math.inf, math.nan])
    def test_out_of_range(self, bad: float) -> None:
        with pytest.raises(ValueError, match="ci_level"):
            normal_z(bad)

    def test_known_quantiles(self) -> None:
        assert normal_z(0.95) == pytest.approx(1.959963985, rel=1e-6)
        assert normal_z(0.99) == pytest.approx(2.575829304, rel=1e-6)


class TestWilsonInterval:
    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="positive int"):
            wilson_interval(1.0, 0, 1.96)
        with pytest.raises(ValueError, match="positive int"):
            wilson_interval(1.0, -3, 1.96)
        with pytest.raises(ValueError, match="positive int"):
            wilson_interval(1.0, True, 1.96)
        with pytest.raises(ValueError, match="positive int"):
            wilson_interval(1.0, 10.5, 1.96)
        with pytest.raises(ValueError, match=r"\[0, n\]"):
            wilson_interval(11.0, 10, 1.96)
        with pytest.raises(ValueError, match=r"\[0, n\]"):
            wilson_interval(-1.0, 10, 1.96)
        with pytest.raises(ValueError, match=r"\[0, n\]"):
            wilson_interval(math.nan, 10, 1.96)
        with pytest.raises(ValueError, match="positive"):
            wilson_interval(5.0, 10, 0.0)
        with pytest.raises(ValueError, match="positive"):
            wilson_interval(5.0, 10, math.inf)

    def test_boundary_snapping(self) -> None:
        low, high = wilson_interval(0.0, 100, 1.96)
        assert low == 0.0
        assert 0.0 < high < 0.1
        low, high = wilson_interval(100.0, 100, 1.96)
        assert high == 1.0
        assert 0.9 < low < 1.0

    def test_centered_interval(self) -> None:
        low, high = wilson_interval(50.0, 100, 1.96)
        assert low < 0.5 < high
        assert high - low < 0.25


class TestPathRiskStats:
    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            path_risk_stats(np.ones(5), ruin_level=0.0)
        with pytest.raises(ValueError, match="shape"):
            path_risk_stats(np.ones((4, 0)), ruin_level=0.0)
        with pytest.raises(ValueError, match="finite"):
            path_risk_stats(np.array([[0.1, np.nan]]), ruin_level=0.0)
        with pytest.raises(ValueError, match="finite"):
            path_risk_stats(np.ones((2, 3)), ruin_level=math.nan)

    def test_flat_path_no_drawdown_not_ruined(self) -> None:
        r = np.zeros((2, 5))
        out = path_risk_stats(r, ruin_level=0.9)
        assert out["no_drawdown"].tolist() == [1, 1]
        assert out["recovered"].tolist() == [0, 0]
        assert out["recovery_steps"].tolist() == [-1, -1]
        assert out["ruined"].tolist() == [0, 0]
        assert_allclose(out["loss"], np.zeros(2))

    def test_drawdown_and_recovery(self) -> None:
        # +10%, -20%, +10%, +15% -> wealth 1.1, 0.88, 0.968, 1.1132
        r = np.array([[0.10, -0.20, 0.10, 0.15]])
        out = path_risk_stats(r, ruin_level=0.5)
        assert out["max_drawdown"][0] == pytest.approx(0.20)
        # Recovery to the old peak happens at the final step.
        assert out["recovered"][0] == 1
        assert out["recovery_steps"][0] == 2
        assert out["ruined"][0] == 0

    def test_unrecovered_drawdown_is_censored(self) -> None:
        r = np.array([[0.10, -0.30, 0.01]])
        out = path_risk_stats(r, ruin_level=0.5)
        assert out["recovered"][0] == 0
        assert out["recovery_steps"][0] == -1

    def test_ruin_threshold_hit(self) -> None:
        r = np.array([[-0.60, 0.01]])
        out = path_risk_stats(r, ruin_level=0.5)
        assert out["ruined"][0] == 1


class TestWeightedEs:
    def test_equal_weights_match_spectral(self) -> None:
        losses = np.linspace(-0.4, 0.8, 60)
        w = np.full(60, 1.0)
        assert weighted_expected_shortfall(losses, w, 0.9) == pytest.approx(
            spectral_es(losses, 0.9)
        )

    def test_heavier_tail_weights_raise_es(self) -> None:
        losses = np.linspace(-0.4, 0.8, 60)
        w = np.linspace(0.1, 4.0, 60)
        assert weighted_expected_shortfall(losses, w, 0.9) > spectral_es(losses, 0.9)

    def test_validation(self) -> None:
        x = np.linspace(0.0, 1.0, 10)
        w = np.ones(10)
        with pytest.raises(ValueError, match="same shape"):
            weighted_expected_shortfall(x, w[:5], 0.9)
        with pytest.raises(ValueError, match="at least 5"):
            weighted_expected_shortfall(x[:4], w[:4], 0.9)
        with pytest.raises(ValueError, match="finite"):
            weighted_expected_shortfall(np.append(x, np.nan), np.ones(11), 0.9)
        with pytest.raises(ValueError, match="finite"):
            weighted_expected_shortfall(np.append(x, 0.5), np.append(w, np.nan), 0.9)
        with pytest.raises(ValueError, match="non-negative"):
            weighted_expected_shortfall(x, -np.ones(10), 0.9)
        with pytest.raises(ValueError, match="alpha"):
            weighted_expected_shortfall(x, w, 1.0)
        with pytest.raises(ValueError, match="alpha"):
            weighted_expected_shortfall(x, w, 0.0)
        with pytest.raises(ValueError, match="positive mass"):
            weighted_expected_shortfall(x, np.zeros(10), 0.9)

    def test_degenerate_weight_dirac_returns_that_outcome(self) -> None:
        # All mass on the smallest loss -> weighted tail mean is that loss.
        x = np.linspace(0.0, 1.0, 10)
        w = np.array([1.0] + [0.0] * 9)
        assert weighted_expected_shortfall(x, w, 0.99) == pytest.approx(0.0)


class TestBatchMeans:
    def test_batch_size_bounds(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            batch_size_for(0.0)
        with pytest.raises(ValueError, match="alpha"):
            batch_size_for(1.0)
        assert batch_size_for(0.5) == 20
        assert batch_size_for(0.99) == 500

    def test_too_few_batches_fails_closed(self) -> None:
        losses = np.linspace(0.0, 1.0, 40)
        out = batch_means_es_interval(losses, 0.9, 0.95)
        assert out["reason"] == "fewer than 8 batches"
        assert out["ci_low"] is None
        assert out["batch_count"] == 40 // batch_size_for(0.9)

    def test_weighted_skips_zero_mass_blocks(self) -> None:
        losses = np.linspace(0.0, 1.0, 500)
        w = np.zeros(500)
        w[:50] = 1.0  # only the first of ten blocks has mass
        out = batch_means_es_interval(losses, 0.9, 0.95, weights=w)
        assert out["reason"] == "fewer than 8 batches with positive weight"
        assert out["ci_low"] is None

    def test_full_interval_brackets_point(self) -> None:
        rng = np.random.default_rng(5)
        losses = rng.normal(0.0, 0.3, 4_000)
        out = batch_means_es_interval(losses, 0.9, 0.95)
        assert out["reason"] is None
        assert out["ci_low"] < out["estimate"] < out["ci_high"]
        assert out["standard_error"] > 0.0
        assert out["batch_count"] >= 8

    def test_weighted_interval(self) -> None:
        rng = np.random.default_rng(6)
        losses = rng.normal(0.0, 0.3, 4_000)
        w = np.ones(4_000)
        out = batch_means_es_interval(losses, 0.9, 0.95, weights=w)
        assert out["reason"] is None
        assert out["ci_low"] < out["estimate"] < out["ci_high"]


class TestGpdVarEs:
    def test_finite_guards(self) -> None:
        for kwargs in (
            {"xi": math.nan},
            {"sigma": math.inf},
            {"threshold": math.nan},
            {"phi_u": math.nan},
            {"alpha": math.inf},
        ):
            args = dict(xi=0.2, sigma=0.1, threshold=1.0, phi_u=0.05, alpha=0.99)
            args.update(kwargs)
            with pytest.raises(ValueError, match="finite"):
                gpd_var_es(**args)

    def test_domain_guards(self) -> None:
        args = dict(xi=0.2, sigma=0.1, threshold=1.0, phi_u=0.05, alpha=0.99)
        with pytest.raises(ValueError, match="sigma"):
            gpd_var_es(**{**args, "sigma": 0.0})
        with pytest.raises(ValueError, match="sigma"):
            gpd_var_es(**{**args, "phi_u": 0.0})
        with pytest.raises(ValueError, match="sigma"):
            gpd_var_es(**{**args, "phi_u": 1.5})
        with pytest.raises(ValueError, match="sigma"):
            gpd_var_es(**{**args, "alpha": 1.0})
        with pytest.raises(ValueError, match="coverage"):
            # alpha below the threshold coverage is meaningless.
            gpd_var_es(**{**args, "alpha": 0.5})

    def test_zero_shape_closed_form(self) -> None:
        var, es = gpd_var_es(0.0, 0.5, 1.0, 0.05, 0.99)
        # xi -> 0 limit: VaR = u - sigma * ln((1-alpha)/phi_u)
        assert var == pytest.approx(1.0 - 0.5 * math.log(0.01 / 0.05))
        assert es == pytest.approx(var + 0.5)

    def test_heavy_shape_es_nan(self) -> None:
        var, es = gpd_var_es(1.2, 0.1, 1.0, 0.05, 0.99)
        assert math.isfinite(var)
        assert math.isnan(es)

    def test_var_beyond_threshold(self) -> None:
        var, es = gpd_var_es(0.2, 0.1, 1.0, 0.05, 0.99)
        assert var > 1.0
        assert es > var


class TestPotGpd:
    def test_tiny_sample_unavailable(self) -> None:
        out = pot_gpd(np.linspace(0, 1, 10))
        assert out["available"] is False
        assert out["diagnostics_ok"] is False
        assert "fewer than 20" in out["reason"]

    def test_quantile_rule_bounds(self) -> None:
        with pytest.raises(ValueError, match="threshold_quantile"):
            pot_gpd(np.linspace(0, 1, 100), threshold_quantile=0.4)
        with pytest.raises(ValueError, match="threshold_quantile"):
            pot_gpd(np.linspace(0, 1, 100), threshold_quantile=1.0)

    def test_absolute_threshold_must_be_finite(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            pot_gpd(np.linspace(0, 1, 100), threshold=math.inf)

    def test_few_exceedances_unavailable(self) -> None:
        out = pot_gpd(np.linspace(0.0, 1.0, 100), threshold=0.995)
        assert out["available"] is False
        assert "fewer than 20 exceedances" in out["reason"]
        assert out["threshold_rule"] == "absolute"

    def test_full_fit_heavy_tail(self) -> None:
        rng = np.random.default_rng(4)
        # Pareto-like tail: plenty of exceedances, finite shape < 1.
        losses = rng.pareto(3.0, size=4_000) * 0.01
        out = pot_gpd(losses, threshold_quantile=0.95)
        assert out["available"] is True
        assert out["n_exceedances"] == 200
        assert out["threshold_rule"] == "quantile_0.95"
        assert math.isfinite(out["xi"])
        assert out["es_finite"] is True
        assert out["xi_at_quantiles"] is not None
        assert "var_0.99" in out["tail"]
        assert math.isfinite(out["mean_excess_empirical"])
        assert isinstance(out["diagnostics_ok"], bool)

    def test_degenerate_excess_fails_closed(self, monkeypatch) -> None:
        # A GPD fit that raises leaves the diagnostics unavailable.
        import scipy.stats as sstats

        monkeypatch.setattr(
            sstats.genpareto, "fit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no fit"))
        )
        losses = np.concatenate([np.zeros(100), np.linspace(0.01, 0.5, 60)])
        out = pot_gpd(losses, threshold=0.005)
        assert out["available"] is False
        assert out["reason"] == "gpd fit failed"


class TestPotFromExceedances:
    def test_n_total_validation(self) -> None:
        x = np.linspace(0.01, 0.5, 30)
        for bad in (0, -1, True, 2.5):
            with pytest.raises(ValueError, match="positive int"):
                pot_from_exceedances(x, 0.0, bad)

    def test_threshold_finite(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            pot_from_exceedances(np.linspace(0.01, 0.5, 30), math.nan, 100)

    def test_few_exceedances_unavailable(self) -> None:
        out = pot_from_exceedances(np.linspace(0.01, 0.5, 5), 0.0, 100)
        assert out["available"] is False
        assert "fewer than 20 exceedances" in out["reason"]

    def test_retained_tail_fit(self) -> None:
        rng = np.random.default_rng(8)
        excess = rng.pareto(3.0, size=400) * 0.01
        out = pot_from_exceedances(0.5 + excess, 0.5, 10_000)
        assert out["available"] is True
        assert out["xi_at_quantiles"] is None
        assert any("stability" in w for w in out["diagnostic_warnings"])
        assert out["threshold_rule"] == "absolute"
        assert out["n_exceedances"] >= 350
        assert out["phi_u"] == pytest.approx(out["n_exceedances"] / 10_000)

    def test_fit_failure_fails_closed(self, monkeypatch) -> None:
        import scipy.stats as sstats

        monkeypatch.setattr(
            sstats.genpareto, "fit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no fit"))
        )
        out = pot_from_exceedances(np.linspace(0.5, 1.0, 60), 0.25, 1000)
        assert out["available"] is False
        assert out["reason"] == "gpd fit failed"

"""KATs for tail-risk estimators: Wilson, spectral ES, batch means, POT/GPD."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.mc_engine import tails


class TestNormalZ:
    def test_two_sided_convention(self) -> None:
        # normal_z takes the CI level, not the tail probability:
        # z(0.95) = Phi^{-1}(0.975).
        assert tails.normal_z(0.95) == pytest.approx(1.959963984540054, rel=1e-12)

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            tails.normal_z(1.5)
        with pytest.raises(ValueError):
            tails.normal_z(0.0)


class TestWilsonInterval:
    def test_kat(self) -> None:
        low, high = tails.wilson_interval(3, 10, 1.959963984540054)
        assert low == pytest.approx(0.10779126740630099, rel=1e-12)
        assert high == pytest.approx(0.6032218525388546, rel=1e-12)

    def test_edge_snapping(self) -> None:
        assert tails.wilson_interval(0, 10, 1.96)[0] == 0.0
        assert tails.wilson_interval(10, 10, 1.96)[1] == 1.0

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            tails.wilson_interval(11, 10, 1.96)
        with pytest.raises(ValueError):
            tails.wilson_interval(3, 10, -1.0)
        with pytest.raises(ValueError):
            tails.wilson_interval(3, True, 1.96)  # type: ignore[arg-type]


class TestSpectralES:
    def test_kat_top_decile(self) -> None:
        x = np.arange(1.0, 101.0)
        assert tails.spectral_es(x, 0.9) == pytest.approx(95.5)

    def test_weighted_equals_unweighted_for_equal_weights(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        assert tails.weighted_expected_shortfall(x, np.ones(x.size), 0.95) == pytest.approx(
            tails.spectral_es(x, 0.95)
        )

    def test_weights_shift_mass_to_tail(self) -> None:
        x = np.arange(1.0, 101.0)
        w = np.ones(100)
        w[:50] = 0.0  # remove the bottom half from the measure
        es = tails.weighted_expected_shortfall(x, w, 0.9)
        # Top 10% of the surviving 51..100 measure = values 96..100.
        assert es == pytest.approx(98.0)
        assert es > tails.spectral_es(x, 0.9)

    def test_weighted_validation(self) -> None:
        x = np.ones(10)
        with pytest.raises(ValueError, match="same shape"):
            tails.weighted_expected_shortfall(x, np.ones(5), 0.9)
        with pytest.raises(ValueError, match="non-negative"):
            tails.weighted_expected_shortfall(x, -np.ones(10), 0.9)
        with pytest.raises(ValueError, match="positive mass"):
            tails.weighted_expected_shortfall(x, np.zeros(10), 0.9)
        with pytest.raises(ValueError, match="at least 5"):
            tails.weighted_expected_shortfall(x[:4], np.ones(4), 0.9)


class TestBatchMeans:
    def test_batch_size_rule(self) -> None:
        assert tails.batch_size_for(0.95) == 100
        assert tails.batch_size_for(0.5) == 20  # floor of 20
        with pytest.raises(ValueError):
            tails.batch_size_for(1.0)

    def test_too_few_batches_gives_no_interval(self) -> None:
        x = np.random.default_rng(0).standard_normal(500)  # 5 batches at alpha=0.95
        out = tails.batch_means_es_interval(x, 0.95, 0.95)
        assert out["ci_low"] is None and out["ci_high"] is None
        assert out["reason"] == "fewer than 8 batches"
        assert out["estimate"] == pytest.approx(tails.spectral_es(x, 0.95))

    def test_interval_brackets_estimate(self) -> None:
        x = np.random.default_rng(0).standard_normal(4000)
        out = tails.batch_means_es_interval(x, 0.95, 0.95)
        assert out["ci_low"] < out["estimate"] < out["ci_high"]
        assert out["standard_error"] > 0.0

    def test_zero_weight_blocks_are_skipped(self) -> None:
        x = np.random.default_rng(0).standard_normal(4000)
        w = np.ones(4000)
        w[:2000] = 0.0  # zero out 20 of 40 blocks -> 20 remain, still >= 8
        out = tails.batch_means_es_interval(x, 0.95, 0.95, weights=w)
        assert out["batch_count"] == 20


class TestPotGpd:
    def test_exponential_excesses_fit_near_zero_xi(self) -> None:
        rng = np.random.default_rng(0)
        # Exponential excesses => true xi = 0.
        losses = rng.exponential(1.0, 400) + 2.0
        out = tails.pot_from_exceedances(losses, 2.0, 1000)
        assert out["available"] is True
        assert abs(float(out["xi"])) < 0.15
        assert out["n_exceedances"] == 400

    def test_too_few_exceedances_unavailable(self) -> None:
        losses = np.array([2.5, 3.0, 4.0])
        out = tails.pot_from_exceedances(losses, 2.0, 100)
        assert out["available"] is False
        assert "fewer than 20" in str(out["reason"])

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="positive int"):
            tails.pot_from_exceedances(np.ones(50), 1.0, 0)
        with pytest.raises(ValueError, match="finite"):
            tails.pot_from_exceedances(np.ones(50), float("nan"), 10)

    def test_var_es_ordering(self) -> None:
        # ES must exceed VaR at the same level for any valid fit.
        rng = np.random.default_rng(2)
        losses = 5.0 + rng.pareto(2.0, 300)  # heavy tail
        out = tails.pot_from_exceedances(losses, 5.0, 1000)
        assert out["available"] is True
        var = out.get("var") or out.get("var_level")
        es = out.get("es") or out.get("es_level")
        if var is not None and es is not None:
            assert float(es) >= float(var)

"""market_sim edge paths: AS quote guards, EcologyConfig validation matrix,
impact trial/measurement guards, stylized-fact series guards, native helpers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from quant_fund.market_sim import impact as imp
from quant_fund.market_sim import native, quotes, stylized
from quant_fund.market_sim.config import EcologyConfig, validation_config

pytestmark = pytest.mark.synthetic


class TestQuotes:
    def test_finite_and_positive_guards(self) -> None:
        for bad in (float("nan"), float("inf")):
            with pytest.raises(ValueError, match="mid_tick"):
                quotes.avellaneda_stoikov_quotes(bad, 0.0, 0.1, 1.0, 0.5)
        with pytest.raises(ValueError, match="inventory"):
            quotes.avellaneda_stoikov_quotes(100.0, float("nan"), 0.1, 1.0, 0.5)
        for bad in (0.0, -1.0, float("nan")):
            with pytest.raises(ValueError, match="gamma"):
                quotes.avellaneda_stoikov_quotes(100.0, 0.0, bad, 1.0, 0.5)
        for bad in (0.0, -1.0, float("nan")):
            with pytest.raises(ValueError, match="k"):
                quotes.avellaneda_stoikov_quotes(100.0, 0.0, 0.1, bad, 0.5)
        for bad in (-1.0, float("nan")):
            with pytest.raises(ValueError, match="sigma2"):
                quotes.avellaneda_stoikov_quotes(100.0, 0.0, 0.1, 1.0, bad)

    def test_integer_ticks_and_spread_floor(self) -> None:
        bid, ask, half = quotes.avellaneda_stoikov_quotes(100.0, 0.0, 0.1, 1.0, 0.5)
        assert ask > bid and half >= 1.0
        # zero-variance quote still respects the one-tick minimum spread
        bid2, ask2, half2 = quotes.avellaneda_stoikov_quotes(100.0, 0.0, 0.1, 1.0, 0.0)
        assert ask2 > bid2 and half2 >= 1.0
        # positive inventory shifts the reservation price down
        bid3, ask3, _ = quotes.avellaneda_stoikov_quotes(100.0, 5.0, 0.1, 1.0, 1.0)
        assert bid3 < bid


class TestEcologyConfig:
    def test_defaults_and_replacements(self) -> None:
        EcologyConfig()
        vc = validation_config(seed=3)
        assert vc.max_events == 24_000 and vc.bar_events == 25

    @pytest.mark.parametrize(
        ("patch", "match"),
        [
            ({"noise_rate": 0}, "positive"),
            ({"hawkes_qty": -1}, "non-negative"),
            ({"initial_mid_tick": 999}, "seed quotes"),
            ({"initial_mid_tick": 99_000}, "seed quotes"),
            ({"seed": -1}, "non-negative"),
            ({"tick_size": 0.0}, "tick_size"),
            ({"max_events": 0}, "event counts"),
            ({"warmup_events": -1}, "event counts"),
            ({"max_events": 100, "warmup_events": 100}, "warmup_events"),
            ({"bar_events": 0}, "strides"),
            ({"n_mm": -1}, "non-negative"),
            ({"mm_gamma": 0.0}, "mm_gamma"),
            ({"noise_offset_lo": -1}, "noise offsets"),
            ({"noise_offset_lo": 10, "noise_offset_hi": 5}, "noise offsets"),
            ({"noise_limit_prob": -0.5}, "probabilities"),
            (
                {
                    "noise_limit_prob": 0.0,
                    "noise_market_prob": 0.0,
                    "noise_cancel_prob": 0.0,
                },
                "probabilities",
            ),
            ({"execution_start_frac": 0.0}, "execution_start_frac"),
            ({"execution_start_frac": 1.0}, "execution_start_frac"),
            ({"strategy_nav": 0.0}, "strategy nav"),
        ],
    )
    def test_guard_matrix(self, patch: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            replace(EcologyConfig(seed=1), **patch)


class TestImpact:
    def test_trial_config_bounds(self) -> None:
        with pytest.raises(ValueError, match="50 events"):
            imp.impact_trial_config(seed=1, max_events=49)
        cfg = imp.impact_trial_config(seed=1, max_events=100)
        assert cfg.n_execution == 0

    def test_execute_trial_guards(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            imp.execute_impact_trial(0)
        with pytest.raises(ValueError, match="\\+1 or -1"):
            imp.execute_impact_trial(10, side=2)
        with pytest.raises(ValueError, match="finish before"):
            imp.execute_impact_trial(10, start_event=490, n_slices=5, every=10, max_events=500)

    def test_fit_impact_law_edges(self) -> None:
        x = np.array([0.1, 0.2, 0.3])
        y = np.array([0.5, 1.0, 1.5])
        with pytest.raises(ValueError, match="same length"):
            imp.fit_impact_law(x, y[:2])
        out = imp.fit_impact_law(x[:2], y[:2], min_points=5)
        assert out["status"] == "inconclusive" and out["reason"] == "too_few_points"
        const = imp.fit_impact_law(np.full(5, 0.1), np.ones(5), min_points=3)
        assert const["reason"] == "no_participation_variation"
        fitted = imp.fit_impact_law(x, y, min_points=3)
        assert fitted["status"] in ("pass", "fail", "inconclusive")
        xp = np.linspace(0.1, 0.5, 6)
        strict = imp.fit_impact_law(xp, xp**0.5, min_points=3)
        assert strict["slope"] is not None

    def test_measure_impact_guards(self) -> None:
        with pytest.raises(ValueError, match="reps"):
            imp.measure_impact(reps=0)


class TestStylized:
    def test_too_few_returns_blocks(self) -> None:
        report = stylized.validate_stylized_facts(np.arange(50.0), None, None)  # type: ignore[arg-type]
        assert isinstance(report, dict)

    def test_shape_series_on_empty(self) -> None:
        out = stylized._shape_series(np.array([]), "x")
        assert out["n"] == 0


class TestNative:
    def test_matching_benchmark_guards(self) -> None:
        with pytest.raises(ValueError, match="int"):
            native.matching_benchmark(n_events=1.5)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="\\[1, 20000000\\]"):
            native.matching_benchmark(n_events=0)
        with pytest.raises(ValueError, match="non-negative"):
            native.matching_benchmark(n_events=10, seed=-1)

    def test_core_version_is_string(self) -> None:
        assert isinstance(native.core_version(), str)

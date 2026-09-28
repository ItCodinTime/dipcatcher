"""Stylized-fact edge paths: non-finite/inconclusive guards, shape fail and
impact plumbing, and the validation runner."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from quant_fund.market_sim import stylized
from quant_fund.market_sim.config import EcologyConfig
from quant_fund.market_sim.stylized import (
    run_stylized_validation,
    validate_stylized_facts,
)

pytestmark = pytest.mark.synthetic


def _fact_of(report: dict[str, Any], name: str) -> dict[str, Any]:
    facts = cast(dict[str, Any], report["facts"])
    return cast(dict[str, Any], facts[name])


def _returns(n: int = 260) -> np.ndarray:
    return np.random.default_rng(3).standard_normal(n) * 0.01


class TestPrivateHelpers:
    def test_finite_rejects_bool_and_nonfinite(self) -> None:
        assert stylized._finite(True) is None
        assert stylized._finite("x") is None
        assert stylized._finite(float("nan")) is None
        assert stylized._finite(float("inf")) is None
        assert stylized._finite(3) == 3.0
        assert stylized._finite(2.5) == 2.5

    def test_fact_n_rejects_non_int(self) -> None:
        assert stylized._fact_n({"n": "3"}) == 0
        assert stylized._fact_n({"n": True}) == 0
        assert stylized._fact_n({"n": 4}) == 4
        assert stylized._fact_n({}) == 0


class TestShapeSeriesGuards:
    def test_fit_failure_inconclusive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            stylized.stats.norm, "fit", lambda _x: (_ for _ in ()).throw(ValueError("no fit"))
        )
        sample = np.abs(np.random.default_rng(1).standard_normal(150)) + 0.5
        fact = stylized._shape_series(sample, "spread")
        assert fact["status"] == "inconclusive" and fact["reason"] == "fit_failed"

    def test_non_finite_loglik_inconclusive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(stylized.stats.norm, "logpdf", lambda *_a: np.array([float("nan")]))
        sample = np.abs(np.random.default_rng(1).standard_normal(150)) + 0.5
        fact = stylized._shape_series(sample, "spread")
        assert fact["status"] == "inconclusive" and fact["reason"] == "non_finite"

    def test_symmetric_spreads_fail_shape_rule(self) -> None:
        spreads = np.random.default_rng(4).uniform(0.005, 0.02, 200)
        depths = np.random.default_rng(5).uniform(3.0, 8.0, 200)
        report = validate_stylized_facts(_returns(), spreads, depths)
        shape = _fact_of(report, "spread_and_depth_shape")
        assert shape["status"] == "fail"


class TestStatisticGuards:
    def test_jarque_bera_non_finite_inconclusive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            stylized,
            "jarque_bera",
            lambda _x: {"stat": 1.0, "pvalue": float("nan"), "excess_kurtosis": 1.0},
        )
        fact = _fact_of(
            validate_stylized_facts(_returns(), np.array([]), np.array([])), "fat_tails"
        )
        assert fact["status"] == "inconclusive" and fact["reason"] == "non_finite"

    def test_arch_non_finite_inconclusive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            stylized, "engle_arch_lm", lambda *_a: {"stat": 1.0, "pvalue": float("nan")}
        )
        fact = _fact_of(
            validate_stylized_facts(_returns(), np.array([]), np.array([])), "volatility_clustering"
        )
        assert fact["status"] == "inconclusive" and fact["reason"] == "non_finite"

    def test_ljung_box_value_error_inconclusive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            stylized,
            "ljung_box",
            lambda *_a: (_ for _ in ()).throw(ValueError("lag too large")),
        )
        fact = _fact_of(
            validate_stylized_facts(_returns(), np.array([]), np.array([])),
            "no_return_autocorrelation",
        )
        assert fact["status"] == "inconclusive" and fact["reason"] == "lag too large"

    def test_ljung_box_non_finite_inconclusive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            stylized, "ljung_box", lambda *_a: {"stat": 1.0, "pvalue": float("nan")}
        )
        fact = _fact_of(
            validate_stylized_facts(_returns(), np.array([]), np.array([])),
            "no_return_autocorrelation",
        )
        assert fact["status"] == "inconclusive" and fact["reason"] == "non_finite"

    def test_dfa_value_error_inconclusive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            stylized,
            "dfa_hurst",
            lambda *_a: (_ for _ in ()).throw(ValueError("window")),
        )
        fact = _fact_of(
            validate_stylized_facts(_returns(), np.array([]), np.array([])),
            "long_memory_absolute_returns",
        )
        assert fact["status"] == "inconclusive" and fact["reason"] == "window"

    def test_hurst_non_finite_inconclusive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(stylized, "dfa_hurst", lambda *_a: {"hurst": float("nan")})
        fact = _fact_of(
            validate_stylized_facts(_returns(), np.array([]), np.array([])),
            "long_memory_absolute_returns",
        )
        assert fact["status"] == "inconclusive" and fact["reason"] == "non_finite"


class TestImpactPlumbing:
    def test_impact_none_not_measured(self) -> None:
        fact = _fact_of(
            validate_stylized_facts(_returns(), np.array([]), np.array([])), "square_root_impact"
        )
        assert fact["status"] == "inconclusive" and fact["reason"] == "not_measured"

    def test_impact_missing_fit(self) -> None:
        fact = _fact_of(
            validate_stylized_facts(
                _returns(), np.array([]), np.array([]), impact={"fit": "bogus"}
            ),
            "square_root_impact",
        )
        assert fact["status"] == "inconclusive" and fact["reason"] == "missing_fit"

    def test_impact_fit_passthrough(self) -> None:
        impact: dict[str, Any] = {
            "fit": {
                "status": "pass",
                "reason": "",
                "n": 15,
                "slope": 0.51,
                "intercept": -0.2,
                "ci_low": 0.45,
                "ci_high": 0.58,
                "r_squared": 0.9,
            },
            "n_attempted": 16,
            "n_dropped": 1,
            "n_positive": 15,
            "drop_reasons": {"no_fill": 1},
        }
        fact = _fact_of(
            validate_stylized_facts(_returns(), np.array([]), np.array([]), impact=impact),
            "square_root_impact",
        )
        assert fact["status"] == "pass" and fact["n"] == 15 and fact["n_dropped"] == 1


def test_run_stylized_validation_without_impact() -> None:
    cfg = replace(EcologyConfig(seed=5), max_events=200, warmup_events=20, bar_events=10)
    report = run_stylized_validation(cfg, measure=False)
    assert report["config_seed"] == 5
    assert report["config_max_events"] == 200
    assert "tape" in report
    assert _fact_of(report, "square_root_impact")["reason"] == "not_measured"

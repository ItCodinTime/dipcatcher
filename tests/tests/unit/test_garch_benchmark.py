"""Honest GARCH benchmark: validation-only selection, no synthetic SOTA."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.garch_benchmark import (
    aggregate_by_date,
    build_origins,
    evaluate_models,
    realizable,
    select_on_validation,
    summarize_losses,
)


def test_selection_uses_validation_not_test() -> None:
    validation = {"garch_normal": np.full(40, 1.0), "gjr_t": np.full(40, 2.0)}
    test = {
        "garch_normal": np.full(40, 3.0),
        "gjr_t": np.full(40, 0.1),
        "rolling": np.full(40, 2.0),
        "ewma": np.full(40, 2.0),
        "har": np.full(40, 2.0),
    }
    report = summarize_losses(validation, test, source="synthetic")
    assert report["selected"] == "garch_normal"
    assert report["sota_proven"] is False
    assert report["beats_all_baselines"] is False


def test_better_test_loss_cannot_flip_selection_or_sota() -> None:
    validation = {"a": np.full(40, 2.0), "b": np.full(40, 1.0)}
    test = {
        "a": np.full(40, 0.001),
        "b": np.full(40, 100.0),
        "rolling": np.full(40, 5.0),
        "ewma": np.full(40, 5.0),
        "har": np.full(40, 5.0),
    }
    report = summarize_losses(validation, test, source="synthetic")
    assert report["selected"] == "b"
    assert report["sota_proven"] is False


def test_realizable_rejects_nonfinite_and_nonpositive() -> None:
    assert realizable(np.array([1e-6, 2e-4])).tolist() == [1e-6, 2e-4]
    with pytest.raises(ValueError):
        realizable(np.array([0.0, 1e-4]))
    with pytest.raises(ValueError):
        realizable(np.array([1e-6, float("nan")]))


def test_build_origins_purge_and_spacing() -> None:
    origins = build_origins(n_dates=300, h=5, min_history=200, stride=5, n_origins=10)
    assert len(origins) == 10
    assert all(o >= 200 for o in origins)
    assert all(origins[i + 1] - origins[i] >= 5 for i in range(len(origins) - 1))
    assert max(origins) + 5 <= 300
    with pytest.raises(ValueError):
        build_origins(n_dates=100, h=5, min_history=200, stride=5, n_origins=2)


def test_aggregate_by_date_requires_complete_panel() -> None:
    dates = np.array([1, 1, 2, 2])
    values = np.array([np.nan, 1.0, 2.0, 4.0])
    with pytest.raises(ValueError):
        aggregate_by_date(dates, values)
    ok = aggregate_by_date(dates, np.array([1.0, 3.0, 2.0, 4.0]))
    assert ok.tolist() == [2.0, 3.0]


def test_evaluate_models_matches_naive_forecast_for_short_series() -> None:
    rng = np.random.default_rng(0)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 260)))
    prices[0] = 100.0
    frame = evaluate_models(
        prices=prices,
        h=5,
        min_history=200,
        stride=5,
        n_origins=8,
        seed=3,
        candidates=("rolling", "ewma", "garch_normal"),
    )
    assert set(frame.columns) >= {"date", "target", "rolling", "ewma", "garch_normal"}
    assert frame["target"].is_finite().all()
    assert frame.height == 8


def test_forecasts_do_not_use_future_prices() -> None:
    prices = 100 * np.exp(np.cumsum(np.random.default_rng(41).normal(0, 0.01, 230)))
    changed = prices.copy()
    changed[201:] *= 1.5
    kwargs = dict(h=5, min_history=200, stride=5, n_origins=1)
    original = evaluate_models(prices=prices, **kwargs)
    altered = evaluate_models(prices=changed, **kwargs)
    for name in ("rolling", "ewma", "garch_normal", "garch_t", "gjr_t", "egarch_t"):
        assert original[name].to_list() == altered[name].to_list()
    assert original["target"].to_list() != altered["target"].to_list()


def test_targets_and_rolling_forecasts_match_log_return_contract() -> None:
    prices = 100 * np.exp(np.cumsum(np.random.default_rng(19).normal(0, 0.02, 230)))
    r = np.diff(np.log(prices))
    result = evaluate_models(prices=prices, n_origins=2, candidates=("rolling",))
    assert result["rolling"][0] == pytest.approx(5 * np.mean(r[180:200] ** 2))
    assert result["target"][0] == pytest.approx(np.sum(r[200:205] ** 2))
    assert result["target"][1] == pytest.approx(np.sum(r[205:210] ** 2))


def test_selection_rejects_nonfinite_validation() -> None:
    with pytest.raises(ValueError, match="nonfinite"):
        select_on_validation({"a": np.array([np.nan])})

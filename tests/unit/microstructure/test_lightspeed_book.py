"""Lightspeed book helpers: 3x reconstruction, IC cards, windows, gates."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from quant_fund.hedge_lab.lightspeed_book import (
    _align_ic,
    _date_key,
    _gates,
    _ic_card,
    _momentum_spec_on_tape,
    _window_cards,
    close_panel,
    reconstruct_3x,
    risk_book_returns,
    split_by_holdout,
)


def test_reconstruct_3x_known_path() -> None:
    px = np.array([100.0, 101.0, 99.0])
    out = reconstruct_3x(px)
    assert out[0] == 100.0
    assert out[1] == pytest.approx(103.0)
    assert out[2] == pytest.approx(103.0 * (1.0 + 3.0 * (99.0 / 101.0 - 1.0)))


def test_reconstruct_3x_bad_input_fails_soft() -> None:
    assert np.isnan(reconstruct_3x(np.array([]))).all() or reconstruct_3x(np.array([])).size == 0
    out = reconstruct_3x(np.array([np.nan, 1.0]))
    assert np.isnan(out).all()
    # Gap in the middle: value carries forward, not reset to NaN.
    px = np.array([100.0, np.nan, 101.0])
    out = reconstruct_3x(px)
    assert out[1] == 100.0
    assert np.isfinite(out[2])


def test_close_panel_inner_join(tmp_path) -> None:
    d1, d2 = dt.date(2025, 1, 1), dt.date(2025, 1, 2)
    gold = pl.DataFrame(
        {
            "event_time": [d1, d1, d2, d2, d2],
            "security_id": ["A", "B", "A", "B", "C"],
            "close": [1.0, 2.0, 1.5, 2.5, 9.0],
            "close_total_return": [1.1, 2.1, 1.6, 2.6, 9.1],
        }
    )
    dates, closes = close_panel(gold, ("A", "B", "ZZZ"))
    assert set(closes) == {"A", "B"}
    # Prefers total-return close.
    assert closes["A"].tolist() == [1.1, 1.6]
    with pytest.raises(ValueError, match="none of the requested"):
        close_panel(gold, ("ZZZ",))
    with pytest.raises(ValueError, match="missing"):
        close_panel(gold, ("A",), price_col="nope")


def test_risk_book_returns_costs_and_residual() -> None:
    closes = {"AAA": np.array([100.0, 102.0, 103.0]), "SGOV": np.full(3, 1.0)}
    weights = {"AAA": np.array([0.0, 0.5, 0.5])}
    r = risk_book_returns(weights, closes, one_way_cost=0.0)
    assert r.shape == (2,)
    # bar1: w=0.5 * (102/100-1) = 0.01 ; bar2: 0.5*(103/102-1)
    assert r[0] == pytest.approx(0.5 * 0.02)
    assert r[1] == pytest.approx(0.5 * (103 / 102 - 1))
    # Turnover charged on weight change.
    r_cost = risk_book_returns(weights, closes, one_way_cost=0.01)
    assert r_cost[0] < r[0]


def test_risk_book_returns_bad_cost_raises() -> None:
    with pytest.raises(ValueError):
        risk_book_returns({"A": np.ones(3)}, {"A": np.ones(3)}, one_way_cost=-0.1)
    with pytest.raises(ValueError):
        risk_book_returns({"A": np.ones(3)}, {"A": np.ones(3)}, one_way_cost=np.nan)


def test_risk_book_returns_no_risk_legs() -> None:
    out = risk_book_returns({"SGOV": np.ones(3)}, {"SGOV": np.ones(3)}, defensive="SGOV")
    assert out.size == 0


def _card(name: str, dates: list[str], series: list[float]) -> dict:
    return {
        "name": name,
        "mean_ic": float(np.mean(series)),
        "ic_series": [float(v) for v in series],
        "ic_dates": list(dates),
        "n_dates": len(series),
    }


def test_align_ic_intersects_dates() -> None:
    a = _card("a", ["2024-01-01", "2024-01-02", "2024-01-03"], [0.1, 0.2, 0.3])
    b = _card("b", ["2024-01-02", "2024-01-03"], [0.4, 0.5])
    out = _align_ic([a, b])
    assert out["a"].tolist() == [0.2, 0.3]
    assert out["b"].tolist() == [0.4, 0.5]


def test_align_ic_disjoint_returns_empty() -> None:
    a = _card("a", ["2024-01-01"], [0.1])
    b = _card("b", ["2024-01-02"], [0.2])
    out = _align_ic([a, b])
    assert out["a"].size == 0 and out["b"].size == 0


def test_ic_card_computes_cross_sectional_ic() -> None:
    rng = np.random.default_rng(0)
    n_dates, n_names = 30, 8
    dates = np.repeat([f"2024-02-{d + 1:02d}" for d in range(n_dates)], n_names)
    signal = rng.normal(size=dates.size)
    y = signal + rng.normal(scale=0.3, size=dates.size)
    card = _ic_card("m", signal, y, dates, hac=1)
    assert card["name"] == "m"
    assert card["n_dates"] == n_dates
    assert card["mean_ic"] > 0.5  # planted signal
    assert len(card["ic_series"]) == n_dates == len(card["ic_dates"])
    assert card["p_ic"] < 0.05


def test_ic_card_sparse_dates_fail_closed() -> None:
    # Fewer than 3 usable dates -> NaN stats, no crash.
    card = _ic_card(
        "m",
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
        np.array(["2024-01-01", "2024-01-01"]),
        hac=None,
    )
    assert np.isnan(card["mean_ic"])
    assert card["n_dates"] <= 2


def test_gates_benchmark_missing_and_short() -> None:
    with pytest.raises(ValueError, match="benchmark"):
        _gates({"a": np.ones(20)}, benchmark="ridge", n_boot=50)
    out = _gates({"ridge": np.zeros(5), "g": np.ones(5)}, benchmark="ridge", n_boot=50)
    assert out["promote"] is False
    assert np.isnan(out["reality_check_p"])


def test_gates_dominant_challenger_clears() -> None:
    rng = np.random.default_rng(1)
    aligned = {
        "ridge": rng.normal(0.0, 0.01, 60),
        "champ": rng.normal(0.08, 0.01, 60),
    }
    out = _gates(aligned, benchmark="ridge", n_boot=200, lags=1)
    assert out["promote"] is True
    assert out["cleared"] == ["champ"]
    assert out["dm"]["champ"]["preferred"] == "champ"


def test_date_key_truncates_and_isoformat() -> None:
    assert _date_key(dt.date(2025, 1, 2)) == "2025-01-02"
    assert _date_key("2025-01-02T12:00:00") == "2025-01-02"
    assert _date_key("short") == "short"


def test_split_by_holdout_aligns_and_checks_length() -> None:
    dates = ["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"]
    r = np.array([0.01, 0.02, 0.03])  # aligned to dates[1:]
    parts = split_by_holdout(dates, r)
    assert parts["is"].tolist() == [0.01]
    assert parts["holdout"].tolist() == [0.02, 0.03]
    assert parts["full"].tolist() == [0.01, 0.02, 0.03]
    with pytest.raises(ValueError, match="align"):
        split_by_holdout(dates, np.array([0.01]))


def test_window_cards_keys() -> None:
    dates = [(dt.date(2024, 8, 1) + dt.timedelta(days=i)).isoformat() for i in range(300)]
    cards = _window_cards(dates, np.random.default_rng(0).normal(0.0, 0.01, 299))
    for k in ("full", "is", "holdout"):
        assert k in cards
        assert cards[k]["research_only"] is True
    assert cards["selection_end"] == "2024-12-31"
    assert cards["holdout_start"] == "2025-01-02"


def test_momentum_spec_on_tape_freezes_live() -> None:
    spec = _momentum_spec_on_tape(("AAPL", "MSFT"), "stock")
    assert spec.live_disabled is True
    assert spec.universe.risk == ("AAPL", "MSFT")
    assert spec.universe.defensive == "SGOV"

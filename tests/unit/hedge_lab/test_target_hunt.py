"""Target-hunt primitives: Kalman beta, pairs, and the composite receipt."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from quant_fund.hedge_lab.target_hunt import (
    ECONOMIC_PAIRS,
    _card,
    _pivot,
    kalman_beta,
    pair_spread_returns,
    run_target_hunt,
)
from quant_fund.risk.gates import dd_halt, vol_target


def test_economic_pairs_declared_not_scan() -> None:
    assert ("SPY", "QQQ") in ECONOMIC_PAIRS
    assert len(ECONOMIC_PAIRS) <= 20  # declared list, not a 55-choose-2 scan
    for a, b in ECONOMIC_PAIRS:
        assert a != b


def test_kalman_beta_converges_to_true_beta() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 0.01, 600)
    y = 2.0 * x + rng.normal(0.0, 1e-4, 600)
    b = kalman_beta(y, x)
    assert b.shape == (600,)
    # Estimator needs burn-in; check only the tail mean.
    assert float(np.nanmean(b[-200:])) == pytest.approx(2.0, abs=0.05)


def test_kalman_beta_nan_rows_hold_last() -> None:
    x = np.linspace(-1, 1, 200)
    y = 1.5 * x
    y[50] = np.nan
    b = kalman_beta(y, x)
    assert np.isfinite(b[50])
    assert b[50] == b[49]  # state carried forward, not updated


def test_pair_spread_returns_shape_and_warmup() -> None:
    rng = np.random.default_rng(1)
    # Cointegrated-ish pair: y tracks x with mean-reverting spread.
    x = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, 400))
    noise = np.zeros(400)
    for i in range(1, 400):
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0.0, 0.5)
    y = x + noise
    pnl = pair_spread_returns(y, x, z_win=40)
    assert pnl.shape == (400,)
    assert np.all(pnl[:41] == 0.0)  # warmup + delay are flat
    assert np.isfinite(pnl).all()


def test_pivot_inner_join() -> None:
    gold = pl.DataFrame(
        {
            "event_time": [dt.date(2025, 1, 1)] * 2 + [dt.date(2025, 1, 2)] * 2,
            "security_id": ["A", "B", "A", "B"],
            "close": [1.0, 2.0, 1.5, 2.5],
        }
    )
    dates, closes = _pivot(gold, "close")
    assert len(dates) == 2
    assert closes["A"].tolist() == [1.0, 1.5]
    assert closes["B"].tolist() == [2.0, 2.5]


def test_card_contains_holdout_split() -> None:
    dates = [dt.date(2024, 6, 3) + dt.timedelta(days=i) for i in range(400)]
    r = np.random.default_rng(7).normal(0.0, 0.01, 400)
    c = _card("synthetic", r, dates)
    assert c["name"] == "synthetic"
    assert c["research_only"] is True
    assert "holdout_sharpe" in c and "selection_sharpe" in c
    assert c["n_returns"] == 400


def test_vol_target_scales_and_caps() -> None:
    # Constant returns have zero trailing vol -> gate stays flat (fail-safe).
    flat = vol_target(np.full(200, 0.01), target=0.10, lookback=20)
    assert (flat == 0.0).all()
    r = np.resize(np.array([0.02, -0.01]), 200)  # alternating, ~1.5% daily vol
    out = vol_target(r, target=0.10, lookback=20)
    assert np.isfinite(out).all()
    assert out[:21].tolist() == [0.0] * 21
    # sig ≈ std * sqrt(252) ~ 0.238; lev ≈ 0.42 -> scaled |r| ≈ 0.0084
    assert abs(out[50]) == pytest.approx(
        abs(r[50]) * (0.10 / (np.std(r[:21]) * np.sqrt(252))), rel=0.05
    )
    with pytest.raises(ValueError):
        vol_target(r, target=0.0)


def test_dd_halt_flattens_and_stays_cash() -> None:
    r = np.array([0.0, -0.03, -0.03, 0.5, 0.5])
    out = dd_halt(r, limit=0.05)
    # equity: 1.0 -> 0.97 -> 0.9409 (dd=-5.91% < -5% -> halt at i=2)
    assert out[:3].tolist() == pytest.approx(r[:3].tolist())
    assert out[3:].tolist() == [0.0, 0.0]


def _gold_labels(tmp_path, tickers, n=420) -> str:
    rng = np.random.default_rng(3)
    start = dt.date(2024, 4, 1)
    rows_t, rows_s, rows_c, rows_o = [], [], [], []
    for t in tickers:
        px = 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.008, n))
        for i, d in enumerate([start + dt.timedelta(days=k) for k in range(n)]):
            rows_t.append(d)
            rows_s.append(t)
            rows_c.append(float(px[i]))
            rows_o.append(float(px[i] * 0.999))
    path = tmp_path / "labels.parquet"
    pl.DataFrame(
        {
            "event_time": rows_t,
            "security_id": rows_s,
            "close": rows_c,
            "open": rows_o,
        }
    ).write_parquet(path)
    return str(path)


def test_run_target_hunt_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # artifact lands under tmp/artifacts
    labels = _gold_labels(tmp_path, ["SPY", "TLT", "QQQ", "JPM", "GS", "XOM", "CVX"])
    out = run_target_hunt(labels)
    assert out["catalog"] == "hedge_lab_analytics"
    assert out["research_only"] is True and out["live_pnl_claim"] is False
    assert out["n_names"] == 7
    assert "ew_close" in {c["name"] for c in out["cards"]}
    # Overnight books need open+close — we supplied both.
    assert "ew_overnight_20bp" in {c["name"] for c in out["cards"]}
    # Declared pairs that exist on the tape were traded.
    assert "JPM/GS" in out["pairs"] and "XOM/CVX" in out["pairs"]
    assert "dir_riskparity_voltgt" in {c["name"] for c in out["cards"]}
    assert (tmp_path / "artifacts" / "hedge_lab" / "target_hunt.json").is_file()
    # Cards carry the honesty stamps.
    for c in out["cards"]:
        assert c["research_only"] is True


def test_run_target_hunt_minimal_tape(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    labels = _gold_labels(tmp_path, ["AAA", "BBB"], n=120)
    out = run_target_hunt(labels)
    # Only ew_close + overnight cards survive; no declared pairs on tape.
    names = {c["name"] for c in out["cards"]}
    assert "ew_close" in names
    assert out["pairs"] == []

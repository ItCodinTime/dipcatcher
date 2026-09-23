"""Lightspeed engines: TQQQ rotation, nautica momentum, metalabel, CLI."""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.config import load_config
from quant_fund.lightspeed.ema import sma_seeded_ema
from quant_fund.lightspeed.metalabel import meta_label_gate, metalabel_multiplier
from quant_fund.lightspeed.momentum import momentum_scores, momentum_target_weights
from quant_fund.lightspeed.ranker import NauticaRanker
from quant_fund.lightspeed.rotation import tqqq_target_weights
from quant_fund.lightspeed.specs import (
    MomentumParams,
    MomentumSpec,
    MomentumUniverse,
    frozen_families,
    nautica_momentum_v1,
    stock_momentum_v1,
    tqqq_long_full_v1,
)
from quant_fund.pipeline.train import RANKING_MODEL_NAMES, _make_ranker


def test_sma_seeded_ema_first_value_is_sma() -> None:
    closes = np.linspace(10.0, 20.0, 40)
    ema = sma_seeded_ema(closes, 10)
    assert not np.isfinite(ema[8])
    assert float(ema[9]) == pytest.approx(float(np.mean(closes[:10])), abs=1e-12)
    assert np.all(np.isfinite(ema[9:]))


def test_frozen_specs_match_lightspeed_lock() -> None:
    tqqq = tqqq_long_full_v1()
    assert tqqq.params.fast_ema == 20
    assert tqqq.params.slow_ema == 180
    assert tqqq.params.flatten_when_fast_below_slow is True
    assert tqqq.params.max_tqqq_weight == pytest.approx(0.98)
    assert tqqq.params.vol_budget == pytest.approx(10.0)
    assert tqqq.params.rebalance_every == 5
    assert tqqq.params.signal_delay_sessions == 1
    assert tqqq.live_disabled is True
    stock = stock_momentum_v1()
    nautica = nautica_momentum_v1()
    for book in (stock, nautica):
        assert book.params.mom_fast == 63
        assert book.params.mom_blend == pytest.approx(1.0)
        assert book.params.trend_sma == 200
        assert book.params.top_k == 1
        assert book.params.vol_budget == pytest.approx(0.6)
        assert book.params.crash_lookback == 10
        assert book.params.crash_return == pytest.approx(-0.2)
        assert book.params.signal_delay_sessions == 1
        assert book.live_disabled is True
    assert "PLTR" not in stock.universe.risk
    assert "PLTR" in nautica.universe.risk
    blob = frozen_families()
    assert blob["live_disabled"] is True
    assert blob["champion"] == "ridge"
    assert blob["blend_weight"] == 0
    assert blob["broker"] is None


def test_tqqq_uptrend_takes_risk_and_delay_is_zero() -> None:
    n = 240
    t = np.arange(n, dtype=float)
    qqq = 100.0 * np.exp(0.0015 * t)
    tqqq = 30.0 * np.exp(0.004 * t)
    weights = tqqq_target_weights(qqq, tqqq)
    assert set(weights) == {"TQQQ", "SGOV"}
    delay = tqqq_long_full_v1().params.signal_delay_sessions
    assert delay == 1
    assert float(weights["TQQQ"][0]) == pytest.approx(0.0)
    assert np.all(weights["TQQQ"][:delay] == 0.0)
    last = float(weights["TQQQ"][-1])
    assert last > 0.5
    assert last <= 0.98 + 1e-12
    row = weights["TQQQ"][-1] + weights["SGOV"][-1]
    assert float(row) == pytest.approx(1.0, abs=1e-12)


def test_tqqq_downtrend_flattens_to_sgov() -> None:
    n = 240
    t = np.arange(n, dtype=float)
    qqq = 200.0 * np.exp(-0.002 * t)
    tqqq = 80.0 * np.exp(-0.006 * t)
    weights = tqqq_target_weights(qqq, tqqq)
    warmup = tqqq_long_full_v1().warmup
    assert np.max(weights["TQQQ"][warmup + 5 :]) == pytest.approx(0.0, abs=1e-12)
    assert float(weights["SGOV"][-1]) == pytest.approx(1.0, abs=1e-12)


def test_risk_multiplier_can_only_reduce() -> None:
    n = 240
    t = np.arange(n, dtype=float)
    qqq = 100.0 * np.exp(0.0015 * t)
    tqqq = 30.0 * np.exp(0.004 * t)
    base = tqqq_target_weights(qqq, tqqq)
    half = tqqq_target_weights(qqq, tqqq, risk_multiplier=np.full(n, 0.5))
    off = tqqq_target_weights(qqq, tqqq, risk_multiplier=np.zeros(n))
    assert float(half["TQQQ"][-1]) <= float(base["TQQQ"][-1]) + 1e-12
    assert float(off["TQQQ"][-1]) == pytest.approx(0.0, abs=1e-12)


def _short_momentum_spec() -> MomentumSpec:
    return MomentumSpec(
        family="unit-momentum",
        name="unit",
        status="test",
        universe=MomentumUniverse(risk=("AAA", "BBB"), defensive="SGOV"),
        params=MomentumParams(
            mom_fast=5,
            mom_slow=10,
            mom_blend=1.0,
            trend_sma=8,
            min_score=0.0,
            top_k=1,
            vol_window=5,
            vol_budget=0.6,
            max_position_weight=0.95,
            max_gross_weight=0.98,
            rebalance_every=1,
            signal_delay_sessions=1,
            weight_quantum=0.01,
            no_trade_band=0.0,
            crash_lookback=5,
            crash_return=-0.2,
        ),
        live_disabled=True,
    )


def test_nautica_picks_positive_mom_above_sma() -> None:
    spec = _short_momentum_spec()
    n = 40
    t = np.arange(n, dtype=float)
    closes = {
        "AAA": 50.0 * np.exp(0.01 * t),
        "BBB": 50.0 * np.exp(-0.004 * t),
        "SGOV": 100.0 + 0.01 * t,
    }
    scores = momentum_scores(closes, spec.universe.risk, 5, 10, 1.0)
    assert float(scores["AAA"][-1]) > 0.0
    assert float(scores["BBB"][-1]) < 0.0
    weights = momentum_target_weights(closes, spec)
    assert float(weights["AAA"][0]) == pytest.approx(0.0)
    assert float(weights["AAA"][-1]) > 0.0
    assert float(weights["BBB"][-1]) == pytest.approx(0.0, abs=1e-12)
    assert float(weights["AAA"][-1] + weights["SGOV"][-1]) == pytest.approx(1.0, abs=1e-12)


def test_crash_gate_flattens_held_name() -> None:
    spec = _short_momentum_spec()
    n = 40
    t = np.arange(n, dtype=float)
    aaa = 50.0 * np.exp(0.01 * t)
    crash = np.linspace(0.92, 0.60, 8)
    aaa[-8:] = aaa[-9] * crash
    closes = {
        "AAA": aaa,
        "BBB": np.full(n, 40.0),
        "SGOV": 100.0 + 0.01 * t,
    }
    weights = momentum_target_weights(closes, spec)
    assert float(np.max(weights["AAA"][spec.params.warmup : n - 6])) > 0.0
    assert float(weights["AAA"][-1]) == pytest.approx(0.0, abs=1e-12)


def test_metalabel_multiplier_is_reduce_only() -> None:
    p = np.array([0.1, 0.55, 0.8, 1.2, -0.3])
    m = metalabel_multiplier(p, threshold=0.55)
    assert np.all(m >= 0.0) and np.all(m <= 1.0)
    assert float(m[0]) == pytest.approx(0.0)
    assert float(m[2]) == pytest.approx(0.6)
    rng = np.random.default_rng(3)
    sig = rng.normal(size=80)
    ret = 0.4 * sig + 0.2 * rng.normal(size=80)
    gate = meta_label_gate(sig, ret, threshold=0.55)
    assert np.all((gate.multiplier >= 0.0) & (gate.multiplier <= 1.0))
    assert float(np.max(gate.multiplier)) <= 1.0


def test_nautica_ranker_uses_mom_60() -> None:
    names = ["cs_z_reversal_1", "cs_z_mom_60", "cs_z_vol_20"]
    x = np.array(
        [
            [1.0, 2.0, 0.0],
            [0.0, -1.0, 0.5],
            [-2.0, 0.5, 1.0],
        ],
        dtype=float,
    )
    y = np.array([0.1, -0.2, 0.0])
    ranker = NauticaRanker().fit(x, y, features=names)
    pred = ranker.predict(x)
    assert pred == pytest.approx(x[:, 1])
    assert ranker.metadata().name == "nautica"
    cfg = load_config("configs/research.yaml")
    assert "nautica" in RANKING_MODEL_NAMES
    model = _make_ranker("nautica", cfg)
    assert model.metadata().name == "nautica"


def test_vol_target_and_dd_halt_are_causal() -> None:
    from quant_fund.hedge_lab.target_hunt import dd_halt, kalman_beta, vol_target

    r = np.concatenate([np.full(80, 0.001), np.full(20, -0.02)])
    vt = vol_target(r, target=0.025, lookback=60)
    assert float(vt[60]) == pytest.approx(0.0)
    halted = dd_halt(r, limit=0.05)
    assert float(np.sum(np.abs(halted[90:]))) == pytest.approx(0.0)
    x = np.linspace(1.0, 2.0, 40)
    y = 3.0 + 1.5 * x + 0.01 * np.sin(np.arange(40))
    b = kalman_beta(y, x)
    assert np.all(np.isfinite(b[10:]))
    assert float(b[-1]) == pytest.approx(1.5, abs=0.3)


def test_holdout_split_uses_frozen_cut() -> None:
    from datetime import datetime

    from quant_fund.hedge_lab.lightspeed_book import split_by_holdout
    from quant_fund.lightspeed.specs import HOLDOUT_START

    dates = [
        datetime(2024, 12, 30),
        datetime(2024, 12, 31),
        datetime(2025, 1, 2),
        datetime(2025, 1, 3),
    ]
    rets = np.array([0.1, 0.2, 0.3])
    parts = split_by_holdout(dates, rets)
    assert HOLDOUT_START == "2025-01-02"
    assert parts["is"].tolist() == [0.1]
    assert parts["holdout"].tolist() == [0.2, 0.3]


def test_reconstruct_3x_and_risk_book_cost() -> None:
    from quant_fund.hedge_lab.lightspeed_book import reconstruct_3x, risk_book_returns

    qqq = np.array([100.0, 101.0, 99.0, 99.0], dtype=float)
    tqqq = reconstruct_3x(qqq)
    assert float(tqqq[0]) == pytest.approx(100.0)
    assert float(tqqq[1]) == pytest.approx(103.0)
    assert float(tqqq[2]) == pytest.approx(103.0 * (1.0 + 3.0 * (99.0 / 101.0 - 1.0)))
    w = {"AAA": np.array([0.0, 0.5, 0.5, 0.0]), "SGOV": np.array([1.0, 0.5, 0.5, 1.0])}
    closes = {"AAA": np.array([10.0, 11.0, 12.0, 12.0])}
    r0 = risk_book_returns(w, closes, one_way_cost=0.0)
    r_c = risk_book_returns(w, closes, one_way_cost=0.01)
    assert r_c[0] < r0[0]


def test_ls_cli_specs_and_demo() -> None:
    runner = CliRunner()
    specs = runner.invoke(app, ["ls", "specs"])
    assert specs.exit_code == 0, specs.output
    blob = json.loads(specs.output)
    assert blob["live_disabled"] is True
    assert blob["champion"] == "ridge"
    assert "tqqq-long-full-v1" in blob["families"]
    assert "nautica-momentum-v1" in blob["families"]
    demo = runner.invoke(app, ["ls", "demo", "--seed", "7"])
    assert demo.exit_code == 0, demo.output
    card = json.loads(demo.output)
    assert card["research_only"] is True
    assert card["live_pnl_claim"] is False
    assert card["broker"] is None
    assert card["champion"] == "ridge"
    assert card["blend_weight"] == 0
    help_race = runner.invoke(app, ["ls", "race", "--help"])
    assert help_race.exit_code == 0, help_race.output
    assert "Not live" in help_race.output or "risk gates" in help_race.output.lower()


def test_doctor_echoes_ls_hunt_race() -> None:
    result = CliRunner().invoke(app, ["doctor", "--config", "configs/research.yaml"])
    assert "hunt|book|race|confirm" in result.output

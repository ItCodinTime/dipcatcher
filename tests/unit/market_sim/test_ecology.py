"""Agents, accounting, and the scenario library on short synthetic tapes."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from quant_fund.market_sim.agents import (
    Action,
    ExecutionAgent,
    InformedAgent,
    MarketMaker,
    MarketView,
    MomentumAgent,
)
from quant_fund.market_sim.config import EcologyConfig
from quant_fund.market_sim.honesty import diagnostic_keys_ok
from quant_fund.market_sim.quotes import avellaneda_stoikov_quotes
from quant_fund.market_sim.scenarios import SCENARIOS, run_scenario
from quant_fund.market_sim.simulator import SEED_AGENT, STRATEGY_AGENT, Simulator, run_ecology


def _view(**overrides: object) -> MarketView:
    base = dict(
        bid=100,
        ask=102,
        bid_qty=10,
        ask_qty=10,
        mid=101.0,
        spread=2,
        anchor=101,
        mid_history=[100, 100, 100, 101],
        trailing_var=1.0,
        halted=False,
        inventory=0,
        event_index=10,
        force_anchor=False,
        price_max=100_000,
    )
    base.update(overrides)
    return MarketView(**base)  # type: ignore[arg-type]


def test_half_spread_rises_with_risk_aversion() -> None:
    _, _, narrow = avellaneda_stoikov_quotes(100.0, 0.0, 0.2, 1.5, 4.0)
    _, _, wide = avellaneda_stoikov_quotes(100.0, 0.0, 0.8, 1.5, 4.0)
    assert wide > narrow
    _, _, floored = avellaneda_stoikov_quotes(100.0, 0.0, 0.2, 1.5, 0.01)
    assert floored == 1.0


def test_long_inventory_shifts_quotes_down() -> None:
    flat_bid, flat_ask, _ = avellaneda_stoikov_quotes(100.0, 0.0, 0.5, 1.5, 1.0)
    long_bid, long_ask, _ = avellaneda_stoikov_quotes(100.0, 4.0, 0.5, 1.5, 1.0)
    assert long_bid < flat_bid
    assert long_ask < flat_ask


def test_quote_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        avellaneda_stoikov_quotes(100.0, 0.0, 0.0, 1.5, 1.0)
    with pytest.raises(ValueError):
        EcologyConfig(seed=-1)
    with pytest.raises(ValueError):
        EcologyConfig(warmup_events=10, max_events=10)


def test_market_maker_quotes_both_sides_until_the_cap() -> None:
    maker = MarketMaker(kind="mm", agent_id=1, rate=1.0, size=6, inventory_cap=20.0)
    actions = maker.propose(_view(), np.random.default_rng(1))
    sides = {action.side for action in actions if action.kind == "limit"}
    assert sides == {1, -1}
    capped = maker.propose(_view(inventory=1_000), np.random.default_rng(1))
    assert all(action.side != 1 for action in capped if action.kind == "limit")


def test_momentum_and_informed_and_execution_rules() -> None:
    history = [100] * 20 + [110]
    momentum = MomentumAgent(kind="momentum", agent_id=2, rate=1.0, lookback=8, size=4)
    assert momentum.propose(_view(mid_history=history), np.random.default_rng(1)) == [
        Action("market", side=1, qty=4)
    ]
    informed = InformedAgent(kind="informed", agent_id=3, rate=1.0, noise_ticks=0.0)
    informed.observe(120)
    buy = informed.propose(_view(ask=102, bid=100), np.random.default_rng(1))
    assert buy and buy[0].side == 1
    child = ExecutionAgent(
        kind="execution",
        agent_id=4,
        rate=1.0,
        target_qty=10,
        slice_qty=4,
        side=-1,
        start_event=50,
    )
    assert child.propose(_view(event_index=10), np.random.default_rng(1)) == []
    assert child.propose(_view(event_index=50), np.random.default_rng(1))[0].qty == 4


def test_accounts_stay_zero_sum() -> None:
    cfg = replace(EcologyConfig(seed=3), max_events=180, warmup_events=20, bar_events=30)
    plain = run_ecology(cfg)
    assert plain.position_sum == 0
    assert plain.cash_ticks_sum == 0
    assert plain.n_trades > 0
    assert plain.research_only is True
    assert plain.live_pnl_claim is False
    assert diagnostic_keys_ok(plain.summary())

    def _weight(closes: np.ndarray) -> float:
        return 0.2 if closes.size else 0.0

    stressed = run_ecology(cfg, strategy=_weight)
    starting = int(round(cfg.strategy_nav / cfg.tick_size))
    assert stressed.position_sum == 0
    assert stressed.cash_ticks_sum == starting
    assert stressed.strategy_equity.size >= 1


def test_same_seed_is_the_same_tape() -> None:
    cfg = replace(EcologyConfig(seed=9), max_events=120, warmup_events=10)
    left = run_ecology(cfg)
    right = run_ecology(cfg)
    assert left.checksum == right.checksum
    assert left.n_trades == right.n_trades
    assert np.array_equal(left.returns, right.returns)


def test_hawkes_wakes_do_not_reschedule() -> None:
    cfg = replace(EcologyConfig(seed=1), max_events=30, warmup_events=5, hawkes_extra=2)
    sim = Simulator(cfg)
    sim._excite_noise(100)
    one_shots = [item for item in sim._heap if item[3] == 0]
    assert len(one_shots) == cfg.hawkes_extra
    sim.close()


def test_cancel_all_includes_the_opening_ladder() -> None:
    cfg = replace(EcologyConfig(seed=1), max_events=20, warmup_events=5)
    sim = Simulator(cfg)
    assert sim.seed_ids
    assert all(sim.book.order_qty(order_id) > 0 for order_id in sim.seed_ids)
    pulled = sim.cancel_all()
    assert pulled >= len(sim.seed_ids)
    assert sim.seed_ids == []
    sim.close()
    assert pulled > 0


def _short(seed: int = 4) -> EcologyConfig:
    return replace(EcologyConfig(seed=seed), max_events=220, warmup_events=20, bar_events=40)


@pytest.mark.synthetic
def test_flash_crash_pulls_makers_and_sells() -> None:
    result = run_scenario("flash_crash", _short())
    assert result.hook["scenario"] == "flash_crash"
    assert int(result.hook["shock_filled"] or 0) > 0
    assert float(result.hook["shock_drop_ticks"] or 0.0) > 0.0
    assert result.position_sum == 0


@pytest.mark.synthetic
def test_halt_auction_prints_only_at_the_reopen() -> None:
    result = run_scenario("halt_auction", _short())
    assert result.hook["trades_while_halted"] == 0
    assert int(result.hook["auction_qty"] or 0) > 0
    assert result.position_sum == 0
    assert result.cash_ticks_sum == 0


@pytest.mark.synthetic
def test_gap_open_reopens_above_the_old_mid() -> None:
    result = run_scenario("gap_open", _short())
    gap = result.hook.get("gap_ticks")
    assert isinstance(gap, float)
    assert gap > 200.0
    assert result.position_sum == 0
    assert result.cash_ticks_sum == 0


@pytest.mark.synthetic
def test_crowded_unwind_trades_both_bursts() -> None:
    result = run_scenario("crowded_unwind", _short())
    assert int(result.hook["entry_filled"] or 0) > 0
    assert int(result.hook["exit_filled"] or 0) > 0
    assert result.position_sum == 0


def test_scenario_names_are_the_library() -> None:
    assert SCENARIOS == (
        "flash_crash",
        "halt_auction",
        "liquidity_drought",
        "gap_open",
        "crowded_unwind",
    )
    with pytest.raises(ValueError):
        run_scenario("not_a_scenario")


def test_seed_and_strategy_ids_stay_out_of_the_population() -> None:
    assert SEED_AGENT == 900_001
    assert STRATEGY_AGENT == 900_003


@pytest.mark.parametrize(
    "field",
    ["tick_size", "mm_gamma", "mm_k", "strategy_nav", "noise_market_prob", "fundamental_rate"],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_config_rejects_nonfinite_values(field, value):
    with pytest.raises(ValueError, match="finite"):
        EcologyConfig(**{field: value})


@pytest.mark.parametrize("value", [True, 2.5, float("inf")])
def test_config_rejects_noninteger_event_counts(value):
    with pytest.raises(ValueError, match="integer"):
        EcologyConfig(max_events=value)

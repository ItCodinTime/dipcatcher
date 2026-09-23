"""Conformance: ``run_backtest_fast`` must reproduce ``run_backtest`` exactly.

The fast path exists for incumbent-benchmark latency. It is only trustworthy
if it is the *same* engine semantically — so these tests assert bitwise NAV /
fill / counter equality on deterministic panels and a seeded fuzz sweep over
costs, gates, fill conventions, shorts, missing bars, and stale marks.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from quant_fund.backtest.engine import StaleValuationError, run_backtest
from quant_fund.backtest.fast_replay import run_backtest_fast
from quant_fund.config.models import (
    AppConfig,
    CostConfig,
    ExecutionConfig,
    FillConvention,
    KillSwitchConfig,
    RiskGateConfig,
)

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _bars(
    sids: list[str], n_days: int, rng: np.random.Generator | None = None, missing: float = 0.0
) -> pl.DataFrame:
    rows = []
    for k, sid in enumerate(sids):
        px = 100.0 * (1 + 0.05 * k)
        for i in range(n_days):
            if rng is not None and rng.random() < missing and i > 2:
                continue  # hole: no bar row at all
            drift = 0.0004 * ((i + 3 * k) % 7 - 3)
            noise = float(rng.normal(0, 0.02)) if rng is not None else 0.0
            close = px * (1 + drift + noise)
            rows.append(
                {
                    "security_id": sid,
                    "event_time": T0 + timedelta(days=i),
                    "open": px * (1 + noise / 2),
                    "close": close,
                    "close_total_return": close,
                    "volume": 1_000_000.0 + 10_000 * i,
                    "adv": close * (1_000_000.0 + 10_000 * i),
                    "vol_20": 0.02 + 0.001 * k,
                    "source": "file",
                }
            )
            px = close
    return pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))


def _weights(
    sids: list[str],
    n_days: int,
    rng: np.random.Generator,
    lo: float = -0.3,
    hi: float = 1.2,
    sparse: float = 0.0,
) -> pl.DataFrame:
    rows = []
    for i in range(n_days):
        for sid in sids:
            if rng.random() < sparse:
                continue
            rows.append(
                {
                    "event_time": T0 + timedelta(days=i),
                    "security_id": sid,
                    "target_weight": float(rng.uniform(lo, hi)),
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))


def _cfg(
    *,
    commission_bps: float = 10.0,
    half_spread_bps: float = 0.0,
    impact_y: float = 0.0,
    bps_per_turnover: float = 0.0,
    borrow_bps_per_year: float = 0.0,
    frictionless: bool = False,
    participation_limit: float = 1.0,
    max_order_notional: float = 1e12,
    max_gross: float = 100.0,
    max_net: float = 100.0,
    max_name: float = 1.0,
    max_participation: float = 1.0,
    max_predicted_vol: float = 100.0,
    stale_price_bars: int = 3,
    fill: FillConvention = FillConvention.NEXT_OPEN,
    allow_close_auction: bool = False,
    kill: str = "ENABLED",
) -> AppConfig:
    return AppConfig(
        costs=CostConfig(
            commission_bps=commission_bps,
            half_spread_bps=half_spread_bps,
            impact_y=impact_y,
            bps_per_turnover=bps_per_turnover,
            borrow_bps_per_year=borrow_bps_per_year,
            frictionless=frictionless,
            participation_limit=participation_limit,
        ),
        risk_gate=RiskGateConfig(
            max_order_notional=max_order_notional,
            max_gross=max_gross,
            max_net=max_net,
            max_name=max_name,
            max_participation=max_participation,
            max_predicted_vol=max_predicted_vol,
            stale_price_bars=stale_price_bars,
            stale_model_hours=1e9,
        ),
        execution=ExecutionConfig(fill=fill, allow_close_auction=allow_close_auction),
        kill_switch=KillSwitchConfig(state=kill),
    )


def _assert_identical(ref, fast) -> None:
    assert ref.equity.height == fast.equity.height
    if ref.equity.height:
        nav_r = np.asarray(ref.equity["nav"].to_list())
        nav_f = np.asarray(fast.equity["nav"].to_list())
        assert np.array_equal(nav_r, nav_f), (
            f"NAV divergence: max abs {np.abs(nav_r - nav_f).max():.3e}"
        )
        for col in ("gross", "net", "turnover"):
            a = np.asarray(ref.equity[col].to_list(), dtype=float)
            b = np.asarray(fast.equity[col].to_list(), dtype=float)
            assert np.array_equal(a, b), f"{col} mismatch"
    assert ref.fills.height == fast.fills.height
    if ref.fills.height:
        assert ref.fills["security_id"].to_list() == fast.fills["security_id"].to_list()
        for col in ("quantity", "price", "fee", "spread_cost", "impact_cost"):
            a = np.asarray(ref.fills[col].to_list(), dtype=float)
            b = np.asarray(fast.fills[col].to_list(), dtype=float)
            assert np.array_equal(a, b), f"fill {col} mismatch"
        dp_r = [v for v in ref.fills["decision_price"].to_list()]
        dp_f = [v for v in fast.fills["decision_price"].to_list()]
        assert len(dp_r) == len(dp_f)
        for x, y in zip(dp_r, dp_f, strict=True):
            assert (x is None and y is None) or x == y
    for key in (
        "total_return",
        "sharpe",
        "max_drawdown",
        "commission",
        "spread",
        "impact",
        "risk_gate_rejects",
        "cash_rejects",
        "kill_switch_halts",
    ):
        a, b = ref.metrics.get(key), fast.metrics.get(key)
        if isinstance(a, float) and np.isnan(a):
            assert isinstance(b, float) and np.isnan(b), key
        else:
            assert a == b, f"metric {key}: {a} != {b}"


def test_small_panel_equivalence():
    rng = np.random.default_rng(7)
    bars = _bars(["AAA", "BBB", "CCC"], 80, rng, missing=0.05)
    weights = _weights(["AAA", "BBB", "CCC"], 80, rng)
    cfg = _cfg(commission_bps=10.0)
    _assert_identical(run_backtest(bars, weights, cfg), run_backtest_fast(bars, weights, cfg))


def test_fuzz_random_panels():
    for seed in range(30):
        rng = np.random.default_rng(seed)
        n_assets = int(rng.integers(2, 9))
        n_days = int(rng.integers(30, 120))
        sids = [f"S{k:02d}" for k in range(n_assets)]
        bars = _bars(sids, n_days, rng, missing=float(rng.uniform(0, 0.15)))
        weights = _weights(
            sids,
            n_days,
            rng,
            lo=float(rng.uniform(-0.5, 0.0)),
            hi=float(rng.uniform(0.5, 1.5)),
            sparse=float(rng.uniform(0, 0.3)),
        )
        cfg = _cfg(
            commission_bps=float(rng.uniform(0, 20)),
            half_spread_bps=float(rng.uniform(0, 10)),
            impact_y=float(rng.uniform(0, 0.5)),
            bps_per_turnover=float(rng.uniform(0, 5)),
            borrow_bps_per_year=float(rng.uniform(0, 300)),
            frictionless=bool(rng.random() < 0.15),
            participation_limit=float(rng.uniform(0.05, 1.0)),
            max_order_notional=float(rng.choice([1e5, 1e6, 1e9, 1e12])),
            max_gross=float(rng.choice([0.5, 1.0, 2.0, 100.0])),
            max_net=float(rng.choice([0.1, 0.5, 1.0, 100.0])),
            max_name=float(rng.choice([0.05, 0.2, 0.5, 1.0])),
            max_participation=float(rng.uniform(0.05, 1.0)),
            max_predicted_vol=float(rng.choice([0.01, 0.05, 0.4, 100.0])),
            stale_price_bars=int(rng.integers(0, 7)),
            fill=(FillConvention.NEXT_OPEN if rng.random() < 0.5 else FillConvention.CLOSE_AUCTION),
            kill="ENABLED" if rng.random() < 0.8 else "HALT_NEW_ORDERS",
        )
        try:
            ref = run_backtest(bars, weights, cfg)
        except Exception as e:  # noqa: BLE001 - whatever ref does, fast must do
            with pytest.raises(type(e)):
                run_backtest_fast(bars, weights, cfg)
            continue
        fast = run_backtest_fast(bars, weights, cfg)
        _assert_identical(ref, fast)


def test_stale_held_position_both_raise():
    rng = np.random.default_rng(11)
    bars = _bars(["AAA", "BBB"], 40, rng)
    # AAA stops printing ANY prices for the final 10 bars while held — no
    # exec mark either, so the position cannot be exited and must go stale.
    cutoff = T0 + timedelta(days=30)
    for col in ("open", "close", "close_total_return"):
        bars = bars.with_columns(
            pl.when((pl.col("security_id") == "AAA") & (pl.col("event_time") > cutoff))
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )
    weights = _weights(["AAA"], 40, rng, lo=0.8, hi=0.8)
    cfg = _cfg(stale_price_bars=3)
    with pytest.raises(StaleValuationError):
        run_backtest(bars, weights, cfg)
    with pytest.raises(StaleValuationError):
        run_backtest_fast(bars, weights, cfg)


def test_unmarked_held_position_parity():
    # AAA keeps an exec open but loses close marks while held. Engines that
    # flatten unmarked names exit it before staleness; engines that don't
    # raise StaleValuationError. Either way fast must match ref bitwise.
    rng = np.random.default_rng(11)
    bars = _bars(["AAA", "BBB"], 40, rng)
    cutoff = T0 + timedelta(days=30)
    for col in ("close", "close_total_return"):
        bars = bars.with_columns(
            pl.when((pl.col("security_id") == "AAA") & (pl.col("event_time") > cutoff))
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )
    weights = _weights(["AAA"], 40, rng, lo=0.8, hi=0.8)
    cfg = _cfg(stale_price_bars=3)
    try:
        ref = run_backtest(bars, weights, cfg)
    except StaleValuationError:
        with pytest.raises(StaleValuationError):
            run_backtest_fast(bars, weights, cfg)
    else:
        _assert_identical(ref, run_backtest_fast(bars, weights, cfg))


def test_allow_close_auction_refused():
    rng = np.random.default_rng(3)
    bars = _bars(["AAA"], 30, rng)
    weights = _weights(["AAA"], 30, rng, lo=0.5, hi=0.5)
    cfg = _cfg(fill=FillConvention.NEXT_OPEN, allow_close_auction=True)
    with pytest.raises(ValueError, match="allow_close_auction"):
        run_backtest_fast(bars, weights, cfg)


def test_duplicate_weight_rows_rejected():
    rng = np.random.default_rng(5)
    bars = _bars(["AAA"], 30, rng)
    weights = _weights(["AAA"], 30, rng, lo=0.5, hi=0.5)
    dup = pl.concat([weights, weights.head(1)])
    cfg = _cfg()
    with pytest.raises(ValueError, match="duplicate"):
        run_backtest_fast(bars, dup, cfg)


def test_kill_switch_halt_parity():
    rng = np.random.default_rng(9)
    bars = _bars(["AAA", "BBB"], 60, rng)
    weights = _weights(["AAA", "BBB"], 60, rng, lo=0.4, hi=0.9)
    cfg = _cfg(kill="HALT_NEW_ORDERS")
    _assert_identical(run_backtest(bars, weights, cfg), run_backtest_fast(bars, weights, cfg))


def test_sparse_rebalance_grid():
    """Whole dates with no weight rows — the deployed engine's semantics
    (carry-forward on the overlay lineage, zero-fill on the older one) must
    be replicated exactly. Plus a held name losing its mark mid-run."""
    rng = np.random.default_rng(21)
    bars = _bars(["AAA", "BBB", "CCC"], 120, rng, missing=0.06)
    weights = _weights(["AAA", "BBB", "CCC"], 120, rng, lo=-0.2, hi=0.9)
    keep = sorted(set(weights["event_time"].to_list()))[::4]
    weights = weights.filter(pl.col("event_time").is_in(keep))
    cfg = _cfg(commission_bps=5.0)
    _assert_identical(run_backtest(bars, weights, cfg), run_backtest_fast(bars, weights, cfg))


def test_empty_weights():
    rng = np.random.default_rng(13)
    bars = _bars(["AAA"], 30, rng)
    weights = pl.DataFrame(
        {
            "event_time": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "security_id": pl.Series([], dtype=pl.String),
            "target_weight": pl.Series([], dtype=pl.Float64),
        }
    )
    cfg = _cfg()
    _assert_identical(run_backtest(bars, weights, cfg), run_backtest_fast(bars, weights, cfg))

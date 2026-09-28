"""Minimized regression: duplicate bars fail closed on both engines.

Hypothesis differential fuzzer
(``tests/property/test_differential_engine_fast_replay.py``) found that
``run_backtest_fast`` refused duplicate ``(event_time, security_id)`` keys
while ``_run_backtest_event_loop`` accepted them and last-valid-write-won —
order-dependent fills for conflicting opens.

Fix: the event loop now raises ``ValueError("duplicate bars …")`` matching
the paper-loop guard. Both engines therefore refuse with ``ValueError``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from quant_fund.backtest.engine import _run_backtest_event_loop
from quant_fund.backtest.fast_replay import run_backtest_fast
from quant_fund.config.models import AppConfig, DataConfig, KillSwitchConfig

T0 = datetime(2024, 1, 2, tzinfo=UTC)


def _row(day: int, *, open_px: float, close: float = 100.0) -> dict[str, object]:
    return {
        "security_id": "A",
        "event_time": T0 + timedelta(days=day),
        "open": open_px,
        "close": close,
        "close_total_return": close,
        "volume": 1_000_000.0,
        "adv": close * 1_000_000.0,
        "vol_20": 0.02,
        "source": "synthetic",
    }


def _panel(opens: list[float]) -> tuple[pl.DataFrame, pl.DataFrame, AppConfig]:
    """One name, four days; day 1 carries every open in ``opens`` (duplicates)."""
    rows = [_row(0, open_px=100.0), _row(2, open_px=100.0), _row(3, open_px=100.0)]
    rows.extend(_row(1, open_px=px) for px in opens)
    bars = pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))
    weights = pl.DataFrame(
        {
            "event_time": [T0 + timedelta(days=i) for i in range(4)],
            "security_id": ["A"] * 4,
            "target_weight": [0.5] * 4,
        }
    ).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))
    cfg = AppConfig(
        data=DataConfig(root=Path("/tmp/dipcatcher-differential-fuzzer")),
        kill_switch=KillSwitchConfig(state="ENABLED"),
    )
    cfg.costs.frictionless = True
    cfg.costs.participation_limit = 1.0
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    return bars, weights, cfg


@pytest.mark.parametrize("opens", [[100.0, 110.0], [110.0, 100.0], [100.0, 100.0]])
def test_duplicate_bars_refused_by_both_engines(opens: list[float]) -> None:
    bars, weights, cfg = _panel(opens)
    with pytest.raises(ValueError, match="duplicate bars"):
        _run_backtest_event_loop(bars, weights, cfg)
    with pytest.raises(ValueError, match="duplicate bars"):
        run_backtest_fast(bars, weights, cfg)

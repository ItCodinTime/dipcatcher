"""Duplicate bar keys fail closed in the paper loop.

Before the guard, ``_bar_maps`` kept the last row for a repeated
``(event_time, security_id)``. Two unique panels with the same close and
different opens (frictionless, target weight 0.5, initial NAV 1_000_000)
filled different share counts and marked NAVs. Both duplicate orders now
raise, and the simulation session still refuses a conflicting tape before
it calls the loop.

Minimized reproduction: one name, one buy, two opens on the fill bar.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from quant_fund.config.loader import load_config
from quant_fund.paper.loop import run_paper_loop
from quant_fund.simtest.faults import Fault, FaultSchedule
from quant_fund.simtest.feed import Bar, normalize_bars
from quant_fund.simtest.invariants import check_invariants
from quant_fund.simtest.session import run_session


def _cfg(tmp_path):
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    cfg.costs.participation_limit = 1.0
    cfg.costs.frictionless = True
    return cfg


def _row(moment: datetime, open_px: float) -> dict[str, object]:
    return {
        "security_id": "A",
        "event_time": moment,
        "open": open_px,
        "close": 100.0,
        "close_total_return": 100.0,
        "volume": 1_000_000.0,
        "adv": 100_000_000.0,
        "vol_20": 0.02,
        "source": "synthetic",
    }


def _frame(fill_opens: list[float]) -> pl.DataFrame:
    """Jan 2 decision, conflicting opens on the Jan 3 fill bar, Jan 4 mark."""
    decision = datetime(2024, 1, 2, tzinfo=UTC)
    fill_day = datetime(2024, 1, 3, tzinfo=UTC)
    tail = datetime(2024, 1, 4, tzinfo=UTC)
    ordered = [_row(decision, 100.0)]
    ordered.extend(_row(fill_day, open_px) for open_px in fill_opens)
    ordered.append(_row(tail, 100.0))
    return pl.DataFrame(ordered).with_columns(
        pl.col("event_time").cast(pl.Datetime(time_zone="UTC"))
    )


def _weights() -> pl.DataFrame:
    moment = datetime(2024, 1, 2, tzinfo=UTC)
    return pl.DataFrame(
        [{"event_time": moment, "security_id": "A", "target_weight": 0.5}]
    ).with_columns(pl.col("event_time").cast(pl.Datetime(time_zone="UTC")))


def test_paper_loop_rejects_duplicate_bars_in_either_order(tmp_path) -> None:
    for name, opens in (("low", [100.0, 110.0]), ("high", [110.0, 100.0])):
        with pytest.raises(ValueError, match="duplicate bars"):
            run_paper_loop(
                _frame(opens),
                _cfg(tmp_path / name),
                champion_weights=_weights(),
                initial_nav=1_000_000.0,
                run_id=f"dup-{name}",
                max_steps=1,
                prefer_latest=False,
            )


def test_session_rejects_conflicting_duplicates_independent_of_order() -> None:
    first = Bar("AAA", datetime(2026, 3, 3, 21, tzinfo=UTC), 100.0, 101.0, 99.0, 100.0, 1.0)
    conflict = Bar("AAA", datetime(2026, 3, 3, 21, tzinfo=UTC), 110.0, 111.0, 109.0, 110.0, 1.0)
    left, left_status = normalize_bars([first, conflict])
    right, right_status = normalize_bars([conflict, first])
    assert left_status == right_status == "conflict"
    assert left == right == []

    schedule = FaultSchedule((Fault(step=0, kind="feed_duplicate", target="AAA", conflict=True),))
    result = run_session(4, n_days=3, schedule=schedule)
    report = check_invariants(result)
    assert report.ok, report.failures
    assert result.outcomes[0] == "fail_closed"
    assert result.cash is None

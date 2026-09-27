"""Paper-loop order rows are behaviours of the lifecycle spec."""

from datetime import UTC, datetime

import polars as pl

from quant_fund.config.loader import load_config
from quant_fund.formal.order_lifecycle import check_trace
from quant_fund.formal.traces import events_from_order_rows
from quant_fund.paper.loop import run_paper_loop


def test_paper_loop_order_trace_is_allowed(tmp_path) -> None:
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = True
    cfg.costs.participation_limit = 1.0
    cfg.risk_gate.max_name = 0.5
    cfg.risk_gate.max_net = 0.5
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    cfg.paper.promote_min_steps = 2
    cfg.paper.promote_max_mean_l1 = 1.0
    rows = []
    for day in range(6):
        when = datetime(2024, 1, 1 + day, tzinfo=UTC)
        for sid, px in (("A", 100.0 + day), ("B", 50.0 + 0.5 * day)):
            rows.append(
                {
                    "security_id": sid,
                    "event_time": when,
                    "open": px,
                    "close": px,
                    "close_total_return": px,
                    "volume": 1_000_000.0,
                    "adv": 100_000_000.0,
                    "vol_20": 0.02,
                    "source": "synthetic",
                }
            )
    bars = pl.DataFrame(rows)
    weights = []
    shadow = []
    for day in range(5):
        when = datetime(2024, 1, 1 + day, tzinfo=UTC)
        weights.append({"event_time": when, "security_id": "A", "target_weight": 0.1})
        weights.append({"event_time": when, "security_id": "B", "target_weight": -0.05})
        shadow.append({"event_time": when, "security_id": "A", "target_weight": 0.05})
        shadow.append({"event_time": when, "security_id": "B", "target_weight": -0.02})
    result = run_paper_loop(
        bars,
        cfg,
        champion_weights=pl.DataFrame(weights),
        shadow_weights=pl.DataFrame(shadow),
        initial_nav=100_000.0,
        max_steps=3,
        run_id="formal-paper-trace",
        prefer_latest=False,
    )
    assert result.metrics["live_pnl_claim"] is False
    assert result.metrics["research_only"] is True
    frame = result.orders
    assert frame.height > 0
    events = events_from_order_rows(frame.to_dicts())
    checked = check_trace(events)
    assert checked.ok, checked.violations
    assert any(order.status == "filled" for order in checked.book.orders.values())

"""NAV = cash + MV identity holds across paper resume (Wave 6)."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from quant_fund.config.loader import load_config
from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.paper.ledger import load_broker_state, validate_ledger_schema
from quant_fund.paper.loop import run_paper_loop


def _bars(n_days: int = 10):
    rows = []
    for d in range(n_days):
        t = datetime(2024, 1, 1 + d, tzinfo=UTC)
        for sid, px0 in [("A", 100.0 + d), ("B", 50.0 + 0.5 * d)]:
            rows.append(
                {
                    "security_id": sid,
                    "event_time": t,
                    "open": px0,
                    "close": px0,
                    "close_total_return": px0,
                    "volume": 1_000_000.0,
                    "adv": 100_000_000.0,
                    "vol_20": 0.02,
                    "source": "synthetic",
                }
            )
    return pl.DataFrame(rows)


def _weights(n_days: int = 9, scale: float = 1.0):
    rows = []
    for d in range(n_days):
        t = datetime(2024, 1, 1 + d, tzinfo=UTC)
        rows.append({"event_time": t, "security_id": "A", "target_weight": 0.1 * scale})
        rows.append({"event_time": t, "security_id": "B", "target_weight": -0.05 * scale})
    return pl.DataFrame(rows)


def _cfg(tmp_path: Path):
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 0.5
    cfg.risk_gate.max_net = 0.5
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    cfg.costs.frictionless = True
    cfg.paper.promote_min_steps = 2
    cfg.paper.promote_max_mean_l1 = 1.0
    return cfg


def test_nav_equals_cash_plus_mv_across_resume(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    run_id = "wave6-nav-resume"
    first = run_paper_loop(
        _bars(10),
        cfg,
        champion_weights=_weights(9),
        shadow_weights=_weights(9, 0.5),
        initial_nav=100_000.0,
        max_steps=3,
        run_id=run_id,
        prefer_latest=False,
    )
    assert first.metrics["n_steps"] == 3
    # Equity rows from first leg: residual ~ 0
    eq1 = first.champion_equity
    if "cash_nav_residual" in eq1.columns:
        for r in eq1["cash_nav_residual"].to_list():
            assert abs(float(r)) < 1e-6

    state = load_broker_state(tmp_path, run_id)
    assert state is not None
    restored = SimulatedBroker.from_state(cfg, state["champion"])
    marks = restored.last_marks or {"A": 103.0, "B": 51.5}
    ident = restored.cash_nav_identity(marks)
    assert ident["residual"] == pytest.approx(0.0, abs=1e-6)
    assert ident["nav"] == pytest.approx(ident["cash"] + ident["position_mv"], abs=1e-6)

    second = run_paper_loop(
        _bars(10),
        cfg,
        champion_weights=_weights(9),
        shadow_weights=_weights(9, 0.5),
        initial_nav=100_000.0,
        max_steps=3,
        resume=True,
        resume_run_id=run_id,
    )
    assert second.metrics["resumed"] is True
    assert second.metrics["n_steps"] >= 6
    eq2 = second.champion_equity
    if "cash_nav_residual" in eq2.columns:
        for r in eq2["cash_nav_residual"].to_list():
            assert abs(float(r)) < 1e-6

    state2 = load_broker_state(tmp_path, run_id)
    assert state2 is not None
    restored2 = SimulatedBroker.from_state(cfg, state2["champion"])
    marks2 = restored2.last_marks
    assert marks2
    ident2 = restored2.cash_nav_identity(marks2)
    assert ident2["residual"] == pytest.approx(0.0, abs=1e-6)
    assert ident2["nav"] == pytest.approx(ident2["cash"] + ident2["position_mv"], abs=1e-6)

    report = validate_ledger_schema(tmp_path / "metadata" / "paper" / run_id)
    assert report["ok"] is True, report["errors"]
    assert report["live_pnl_claim"] is False


def test_resume_preserves_all_durable_ledger_rows(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    run_id = "wave6-append-safe-ledger"
    first = run_paper_loop(
        _bars(10),
        cfg,
        champion_weights=_weights(9),
        shadow_weights=_weights(9, 0.5),
        initial_nav=100_000.0,
        max_steps=2,
        run_id=run_id,
        prefer_latest=False,
    )
    root = tmp_path / "metadata" / "paper" / run_id
    first_orders = pl.read_parquet(root / "orders.parquet")
    first_equity = pl.read_parquet(root / "equity.parquet")
    first_shadow = pl.read_parquet(root / "shadow_equity.parquet")
    first_positions = pl.read_parquet(root / "positions.parquet")
    first_cash = pl.read_parquet(root / "cash_ledger.parquet")

    second = run_paper_loop(
        _bars(10),
        cfg,
        champion_weights=_weights(9),
        shadow_weights=_weights(9, 0.5),
        initial_nav=100_000.0,
        max_steps=2,
        resume=True,
        resume_run_id=run_id,
    )
    assert second.metrics["resumed"] is True

    # Resume is append-safe: every first-leg durable row and receipt remains.
    for name, prior in (
        ("orders", first_orders),
        ("equity", first_equity),
        ("shadow_equity", first_shadow),
        ("positions", first_positions),
        ("cash_ledger", first_cash),
    ):
        current = pl.read_parquet(root / f"{name}.parquet")
        assert current.height >= prior.height, name

    prior_order_ids = set(first_orders["order_id"].cast(pl.String).to_list())
    current_order_ids = set(
        pl.read_parquet(root / "orders.parquet")["order_id"].cast(pl.String).to_list()
    )
    assert prior_order_ids <= current_order_ids
    assert second.orders.height >= first.orders.height
    assert validate_ledger_schema(root)["ok"] is True


def test_resume_preserves_broker_receipt_history(tmp_path: Path) -> None:
    split_cfg = _cfg(tmp_path / "split")
    split_run_id = "wave6-receipt-split"
    run_paper_loop(
        _bars(10),
        split_cfg,
        champion_weights=_weights(9),
        initial_nav=100_000.0,
        max_steps=2,
        run_id=split_run_id,
        prefer_latest=False,
    )
    split = run_paper_loop(
        _bars(10),
        split_cfg,
        champion_weights=_weights(9),
        initial_nav=100_000.0,
        max_steps=2,
        resume=True,
        resume_run_id=split_run_id,
    )

    full_cfg = _cfg(tmp_path / "full")
    full_run_id = "wave6-receipt-full"
    run_paper_loop(
        _bars(10),
        full_cfg,
        champion_weights=_weights(9),
        initial_nav=100_000.0,
        max_steps=4,
        run_id=full_run_id,
        prefer_latest=False,
    )

    split_root = split_cfg.data.root / "metadata" / "paper" / split_run_id
    full_root = full_cfg.data.root / "metadata" / "paper" / full_run_id
    split_state = load_broker_state(split_cfg.data.root, split_run_id)
    full_state = load_broker_state(full_cfg.data.root, full_run_id)
    assert split_state is not None and full_state is not None
    assert split_state["champion"]["n_orders"] == full_state["champion"]["n_orders"]
    assert split_state["champion"]["n_fills"] == full_state["champion"]["n_fills"]
    assert len(split_state["champion"]["history"]) == split_state["champion"]["n_orders"]
    assert split.metrics["n_fills"] == full_state["champion"]["n_fills"]
    assert validate_ledger_schema(split_root)["ok"] is True
    assert validate_ledger_schema(full_root)["ok"] is True


@given(
    n_first=st.integers(min_value=2, max_value=4),
    n_second=st.integers(min_value=1, max_value=3),
)
@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_property_nav_identity_resume_steps(tmp_path: Path, n_first: int, n_second: int) -> None:
    cfg = _cfg(tmp_path)
    run_id = f"wave6-prop-{n_first}-{n_second}"
    run_paper_loop(
        _bars(12),
        cfg,
        champion_weights=_weights(11),
        initial_nav=50_000.0,
        max_steps=n_first,
        run_id=run_id,
        prefer_latest=False,
    )
    second = run_paper_loop(
        _bars(12),
        cfg,
        champion_weights=_weights(11),
        initial_nav=50_000.0,
        max_steps=n_second,
        resume=True,
        resume_run_id=run_id,
    )
    state = load_broker_state(tmp_path, run_id)
    assert state is not None
    broker = SimulatedBroker.from_state(cfg, state["champion"])
    marks = broker.last_marks
    if not marks:
        return
    ident = broker.cash_nav_identity(marks)
    assert ident["residual"] == pytest.approx(0.0, abs=1e-5)
    assert second.metrics["n_steps"] >= n_first

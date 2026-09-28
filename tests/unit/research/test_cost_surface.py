"""P5.4 cost-surface lane: maker/taker fee delta + hysteresis band surface."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.config.models import CostConfig
from quant_fund.execution.costs import total_cost
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.research.cost_surface import (
    DEFAULT_BANDS,
    _churn_targets,
    _maker_probe,
    _synth_bars,
    run_cost_surface,
    write_cost_surface_receipt,
)


def _book(seed: int = 0, n_names: int = 6, n_bars: int = 48):
    bars = _synth_bars(seed, n_names, n_bars)
    return bars, _churn_targets(seed + 1, bars, amplitude=0.05)


def test_maker_cost_leg_zeroes_spread_and_impact() -> None:
    cfg = CostConfig(commission_bps=1.0, maker_commission_bps=0.25)
    taker = total_cost(10.0, 100.0, 1e7, 0.02, cfg)
    maker = total_cost(10.0, 100.0, 1e7, 0.02, cfg, maker=True)
    assert maker["commission"] == pytest.approx(10.0 * 100.0 * 0.25 / 1e4)
    assert maker["spread"] == 0.0
    assert maker["impact"] == 0.0
    assert maker["total"] < taker["total"]
    assert maker["label"] == "costed_maker"


def test_maker_none_rate_falls_back_to_taker() -> None:
    cfg = CostConfig(commission_bps=1.0, maker_commission_bps=None)
    maker = total_cost(10.0, 100.0, 1e7, 0.02, cfg, maker=True)
    assert maker["commission"] == pytest.approx(10.0 * 100.0 * 1.0 / 1e4)


def test_maker_commission_bps_validated() -> None:
    with pytest.raises(ValueError, match="maker_commission_bps"):
        CostConfig(maker_commission_bps=-0.5)


def test_hysteresis_surface_monotone() -> None:
    bars, targets = _book()
    frame, _ = run_cost_surface(bars, targets, bands=DEFAULT_BANDS)
    rows = {
        r["band"]: r for r in frame.filter(pl.col("probe") == "hysteresis").iter_rows(named=True)
    }
    # Provably monotone: sent-stream turnover (triangle inequality on
    # suppressed increments) and suppressed-row count. Realized turnover is
    # reported for context but can jitter via participation-cap splitting.
    target_turnover = [rows[b]["target_turnover"] for b in DEFAULT_BANDS]
    suppressed = [rows[b]["suppressed_rows"] for b in DEFAULT_BANDS]
    tracking = [rows[b]["tracking_rmse"] for b in DEFAULT_BANDS]
    assert all(a >= b - 1e-9 for a, b in zip(target_turnover, target_turnover[1:], strict=False))
    assert all(a <= b for a, b in zip(suppressed, suppressed[1:], strict=False))
    assert suppressed[0] == 0 and suppressed[-1] > 0
    assert all(a <= b + 1e-12 for a, b in zip(tracking, tracking[1:], strict=False))
    assert tracking[0] == pytest.approx(0.0)
    assert tracking[-1] > 0.0
    # Endpoints still separate: the widest band materially cuts realized cost.
    assert rows[DEFAULT_BANDS[-1]]["cost_total"] < rows[0.0]["cost_total"]


def test_maker_probe_records_fill_rate_and_fee_delta() -> None:
    bars, targets = _book()
    cfg = CostConfig(commission_bps=1.0, maker_commission_bps=0.5)
    probe = _maker_probe(bars, targets, cfg)
    assert 0.0 < probe["maker_fill_rate"] <= 1.0
    assert probe["maker_fee_bps_per_filled"] < probe["taker_fee_bps_per_filled"]
    assert probe["maker_fees"] < probe["taker_fees"]
    assert np.isfinite(probe["unfilled_notional_frac"])


def test_maker_fill_marks_is_maker_on_schema() -> None:
    from datetime import UTC, datetime

    from quant_fund.config.models import AppConfig
    from quant_fund.execution.simulated_broker import SimulatedBroker
    from quant_fund.schemas.orders import Order, OrderSide

    cfg = AppConfig(costs=CostConfig(maker_commission_bps=0.5))
    broker = SimulatedBroker(config=cfg)
    dt = datetime(2024, 1, 1, tzinfo=UTC)
    order = Order(
        order_id="o1",
        security_id="S00",
        symbol="S00",
        side=OrderSide.BUY,
        quantity=1.0,
        signal_time=dt,
        decision_time=dt,
        order_time=dt,
        limit_price=100.0,
    )
    rec = broker.submit(order, price=100.0, bar_open=101.0, bar_high=102.0, bar_low=99.0)
    assert rec.fill is not None
    assert rec.fill.is_maker is True
    rec2 = broker.submit(
        Order(
            order_id="o2",
            security_id="S01",
            symbol="S01",
            side=OrderSide.BUY,
            quantity=1.0,
            signal_time=dt,
            decision_time=dt,
            order_time=dt,
        ),
        price=50.0,
    )
    assert rec2.fill is not None and rec2.fill.is_maker is False


def test_receipt_seals_and_avoids_forbidden_metrics(tmp_path) -> None:
    bars, targets = _book()
    _, receipt = run_cost_surface(bars, targets, bands=(0.0, 0.01))
    assert receipt["kind"] == "cost_surface_eval"
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["schema"] == "receipt.v2"
    assert family_blob_forbidden_metrics_absent(receipt["payload"])
    path = write_cost_surface_receipt(receipt, tmp_path)
    sealed = json.loads(path.read_text())
    assert sealed["receipt_sha256"].startswith(path.stem.removeprefix("cost_surface_"))


def test_receipt_rejects_wrong_kind(tmp_path) -> None:
    bars, targets = _book()
    _, receipt = run_cost_surface(bars, targets, bands=(0.0,))
    bad = dict(receipt)
    bad["kind"] = "fleet_tournament"
    with pytest.raises(ValueError, match="honesty contract"):
        write_cost_surface_receipt(bad, tmp_path)


def test_bands_validated() -> None:
    bars, targets = _book()
    with pytest.raises(ValueError, match="bands"):
        run_cost_surface(bars, targets, bands=(0.01, -0.1))


def test_cli_requires_dev() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["cost-surface"])
    assert result.exit_code != 0

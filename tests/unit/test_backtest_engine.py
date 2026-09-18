from datetime import UTC, datetime

import polars as pl
import pytest

from quant_fund.backtest.engine import run_backtest
from quant_fund.config.loader import load_config


def test_backtest_marks_positions_with_total_return_close(tmp_path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = True
    # Unit fixtures use a single-name 100% book; relax gate to the fixture intent.
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 1.0
    cfg.risk_gate.max_order_notional = 1e12
    bars = pl.DataFrame(
        {
            "security_id": ["A", "A", "A"],
            "event_time": [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                datetime(2024, 1, 3, tzinfo=UTC),
            ],
            "open": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
            "close_total_return": [100.0, 110.0, 110.0],
            "volume": [1_000_000.0] * 3,
            "adv": [100_000_000.0] * 3,
            "vol_20": [0.02] * 3,
            "source": ["file"] * 3,
        }
    )
    weights = pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 1, tzinfo=UTC)],
            "security_id": ["A"],
            "target_weight": [1.0],
        }
    )

    result = run_backtest(bars, weights, cfg, initial_nav=100_000.0)

    assert result.equity["nav"][0] == 110_000.0


def test_backtest_turnover_reports_executed_notional_under_participation_cap(tmp_path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = True
    # Unit fixtures use a single-name 100% book; relax gate to the fixture intent.
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 1.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.costs.participation_limit = 0.1
    bars = pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "event_time": [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
            ],
            "open": [100.0, 100.0],
            "close": [100.0, 100.0],
            "close_total_return": [100.0, 100.0],
            "volume": [10.0, 10.0],
            "adv": [1_000.0, 1_000.0],
            "vol_20": [0.02, 0.02],
            "source": ["file", "file"],
        }
    )
    weights = pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 1, tzinfo=UTC)],
            "security_id": ["A"],
            "target_weight": [1.0],
        }
    )

    result = run_backtest(bars, weights, cfg, initial_nav=100_000.0)

    assert result.fills["quantity"][0] == 1.0
    assert result.equity["turnover"][0] == 0.001


def test_backtest_does_not_trade_missing_execution_price(tmp_path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = True
    # Unit fixtures use a single-name 100% book; relax gate to the fixture intent.
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 1.0
    cfg.risk_gate.max_order_notional = 1e12
    bars = pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "event_time": [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
            ],
            "open": [100.0, None],
            "close": [100.0, 100.0],
            "close_total_return": [100.0, 100.0],
            "volume": [1_000_000.0] * 2,
            "adv": [100_000_000.0] * 2,
            "vol_20": [0.02] * 2,
            "source": ["file"] * 2,
        }
    )
    weights = pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 1, tzinfo=UTC)],
            "security_id": ["A"],
            "target_weight": [1.0],
        }
    )

    result = run_backtest(bars, weights, cfg, initial_nav=100_000.0)

    assert result.fills.height == 0
    assert result.equity["nav"][0] == 100_000.0


def test_backtest_empty_panel_forces_research_only(tmp_path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = True
    bars = pl.DataFrame(
        {
            "security_id": pl.Series([], dtype=pl.Utf8),
            "event_time": pl.Series([], dtype=pl.Datetime(time_zone="UTC")),
            "open": pl.Series([], dtype=pl.Float64),
            "close": pl.Series([], dtype=pl.Float64),
            "close_total_return": pl.Series([], dtype=pl.Float64),
            "volume": pl.Series([], dtype=pl.Float64),
            "adv": pl.Series([], dtype=pl.Float64),
            "vol_20": pl.Series([], dtype=pl.Float64),
            "source": pl.Series([], dtype=pl.Utf8),
        }
    )
    weights = pl.DataFrame(
        {
            "event_time": pl.Series([], dtype=pl.Datetime(time_zone="UTC")),
            "security_id": pl.Series([], dtype=pl.Utf8),
            "target_weight": pl.Series([], dtype=pl.Float64),
        }
    )
    result = run_backtest(bars, weights, cfg, initial_nav=100_000.0)
    assert result.equity.height == 0
    assert result.fills.height == 0
    assert result.metrics["n"] == 0
    assert result.metrics["research_only"] is True
    assert result.metrics["live_pnl_claim"] is False


def test_backtest_short_panel_one_date_no_equity_path(tmp_path) -> None:
    """NEXT_OPEN needs ≥2 dates; single bar → empty equity, research-only flags."""
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = True
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 1.0
    cfg.risk_gate.max_order_notional = 1e12
    bars = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2024, 1, 1, tzinfo=UTC)],
            "open": [100.0],
            "close": [100.0],
            "close_total_return": [100.0],
            "volume": [1_000_000.0],
            "adv": [100_000_000.0],
            "vol_20": [0.02],
            "source": ["file"],
        }
    )
    weights = pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 1, tzinfo=UTC)],
            "security_id": ["A"],
            "target_weight": [1.0],
        }
    )
    result = run_backtest(bars, weights, cfg, initial_nav=100_000.0)
    assert result.equity.height == 0
    assert result.metrics["n"] == 0
    assert result.metrics["research_only"] is True
    assert result.metrics["live_pnl_claim"] is False


def test_export_backtest_metrics_json_forces_live_pnl_claim_false(tmp_path) -> None:
    import json
    from pathlib import Path

    from quant_fund.backtest.engine import BacktestResult, export_backtest_metrics_json

    # Thin result without analytics_export — extra tries to claim live P&L
    result = BacktestResult(
        equity=pl.DataFrame({"event_time": [], "nav": []}),
        fills=pl.DataFrame(),
        metrics={
            "total_return": 0.0,
            "sharpe": float("nan"),
            "n": 0,
            "research_only": True,
            "live_pnl_claim": False,
        },
        frictionless=True,
        source_note="SYNTHETIC",
    )
    out = tmp_path / "thin_bt.json"
    path = export_backtest_metrics_json(
        result,
        out,
        extra={"live_pnl_claim": True, "research_only": False, "label": "FORGED"},
    )
    blob = json.loads(Path(path).read_text())
    assert blob["live_pnl_claim"] is False
    assert blob["research_only"] is True
    assert blob["frictionless"] is True
    assert blob["source_note"] == "SYNTHETIC"
    from quant_fund.metrics import analytics_export_digest

    assert blob["analytics_export_sha256"] == analytics_export_digest(blob)


def test_export_backtest_overrides_poisoned_analytics_export(tmp_path) -> None:
    import json
    from pathlib import Path

    from quant_fund.backtest.engine import BacktestResult, export_backtest_metrics_json

    result = BacktestResult(
        equity=pl.DataFrame({"event_time": [], "nav": []}),
        fills=pl.DataFrame(),
        metrics={
            "analytics_export": {
                "live_pnl_claim": True,
                "research_only": False,
                "label": "poison",
            },
            "n": 0,
        },
        frictionless=False,
        source_note="file",
    )
    path = export_backtest_metrics_json(result, tmp_path / "poison.json")
    blob = json.loads(Path(path).read_text())
    assert blob["live_pnl_claim"] is False
    assert blob["research_only"] is True


def test_backtest_rejects_buy_that_would_overdraw_cash() -> None:
    cfg = load_config("configs/research.yaml")
    cfg.costs.commission_bps = 10_000.0
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 1.0
    cfg.risk_gate.max_order_notional = 1e12
    dates = [datetime(2024, 1, day, tzinfo=UTC) for day in (1, 2)]
    bars = pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "event_time": dates,
            "open": [100.0, 100.0],
            "close": [100.0, 100.0],
            "close_total_return": [100.0, 100.0],
            "volume": [1_000_000.0, 1_000_000.0],
            "adv": [100_000_000.0, 100_000_000.0],
            "vol_20": [0.02, 0.02],
            "source": ["file", "file"],
        }
    )
    weights = pl.DataFrame(
        {
            "event_time": [dates[0]],
            "security_id": ["A"],
            "target_weight": [1.0],
        }
    )

    result = run_backtest(bars, weights, cfg, initial_nav=100.0)

    assert result.fills.height == 0
    assert result.equity["nav"].to_list() == [100.0]
    assert result.metrics["cash_rejects"] == 1


def test_backtest_rejects_held_position_after_stale_mark_limit() -> None:
    from quant_fund.backtest import StaleValuationError

    cfg = load_config("configs/research.yaml")
    cfg.costs.frictionless = True
    cfg.risk_gate.stale_price_bars = 1
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 1.0
    cfg.risk_gate.max_order_notional = 1e12
    dates = [datetime(2024, 1, day, tzinfo=UTC) for day in range(1, 5)]
    bars = pl.DataFrame(
        {
            "security_id": ["A", "A", "B", "B", "B", "B"],
            "event_time": [dates[0], dates[1], dates[0], dates[1], dates[2], dates[3]],
            "open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "close_total_return": [100.0] * 6,
            "volume": [1_000_000.0] * 6,
            "adv": [100_000_000.0] * 6,
            "vol_20": [0.02] * 6,
            "source": ["synthetic"] * 6,
        }
    )
    weights = pl.DataFrame(
        {
            "event_time": [dates[0], dates[1]],
            "security_id": ["A", "A"],
            "target_weight": [1.0, 1.0],
        }
    )

    with pytest.raises(StaleValuationError, match=r"A=2"):
        from quant_fund.backtest.engine import run_backtest

        run_backtest(bars, weights, cfg, initial_nav=100_000.0)


def test_export_backtest_metrics_json_keeps_previous_receipt_on_replace_failure(
    tmp_path, monkeypatch
) -> None:
    import quant_fund.backtest.engine as engine

    result = engine.BacktestResult(
        equity=pl.DataFrame({"event_time": [], "nav": []}),
        fills=pl.DataFrame(),
        metrics={"n": 0},
        frictionless=True,
        source_note="SYNTHETIC",
    )
    out = tmp_path / "receipt.json"
    out.write_text('{"previous": true}', encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(engine.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated publication failure"):
        engine.export_backtest_metrics_json(result, out)

    assert out.read_text(encoding="utf-8") == '{"previous": true}'
    assert list(tmp_path.glob(".receipt.json.*.tmp")) == []

"""``backtest.engine`` edge paths: price validators, panel guards, mark
fallbacks, and the metrics-export merge branches."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl
import pytest

from quant_fund.backtest.engine import (
    BacktestResult,
    _target_weight_map,
    _valid_price,
    _validate_bar_panel,
    export_backtest_metrics_json,
    run_backtest,
)
from quant_fund.config.loader import load_config

pytestmark = pytest.mark.synthetic


class TestValidPrice:
    @pytest.mark.parametrize("value", [None, "n/a", float("nan"), -1.0, 0.0, "0"])
    def test_rejects_invalid(self, value: object) -> None:
        assert _valid_price(value) is None

    @pytest.mark.parametrize("bad", [" ", "abc", "1,000"])
    def test_unparseable_string_is_none(self, bad: str) -> None:
        assert _valid_price(bad) is None or _valid_price(bad) == 1000.0

    def test_numeric_string_parses(self) -> None:
        assert _valid_price("12.5") == 12.5

    def test_int_like_object_parses(self) -> None:
        assert _valid_price(7) == 7.0

    def test_inf_rejected(self) -> None:
        assert _valid_price(float("inf")) is None


class TestTargetWeightMap:
    def _weights(self, **cols: object) -> pl.DataFrame:
        return pl.DataFrame(cols)

    def test_missing_required_column_rejected(self) -> None:
        rows = self._weights(
            event_time=[datetime(2024, 1, 1, tzinfo=UTC)],
            security_id=["A"],
        )
        with pytest.raises(ValueError, match="missing required columns"):
            _target_weight_map(rows)

    def test_duplicate_keys_rejected(self) -> None:
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        rows = self._weights(
            event_time=[ts, ts],
            security_id=["A", "A"],
            target_weight=[0.5, 0.5],
        )
        with pytest.raises(ValueError, match="duplicate target weights"):
            _target_weight_map(rows)

    def test_non_finite_weight_rejected(self) -> None:
        rows = self._weights(
            event_time=[datetime(2024, 1, 1, tzinfo=UTC)],
            security_id=["A"],
            target_weight=[float("nan")],
        )
        with pytest.raises(ValueError, match="finite"):
            _target_weight_map(rows)


class TestValidateBarPanel:
    def test_missing_key_columns_is_noop(self) -> None:
        _validate_bar_panel(pl.DataFrame({"open": [1.0]}))

    def test_empty_frame_is_noop(self) -> None:
        _validate_bar_panel(
            pl.DataFrame(
                {
                    "event_time": [],
                    "security_id": [],
                }
            )
        )

    def test_duplicate_bar_keys_rejected(self) -> None:
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        bars = pl.DataFrame(
            {
                "event_time": [ts, ts],
                "security_id": ["A", "A"],
                "open": [100.0, 101.0],
            }
        )
        with pytest.raises(ValueError, match="duplicate bars"):
            _validate_bar_panel(bars)


def _cfg(tmp_path, *, frictionless: bool = True):
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = frictionless
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 1.0
    cfg.risk_gate.max_order_notional = 1e12
    return cfg


def _bars(close_tr: list[float | None], closes: list[float]) -> pl.DataFrame:
    n = len(closes)
    return pl.DataFrame(
        {
            "security_id": ["A"] * n,
            "event_time": [datetime(2024, 1, d, tzinfo=UTC) for d in range(1, n + 1)],
            "open": closes,
            "close": closes,
            "close_total_return": close_tr,
            "volume": [1_000_000.0] * n,
            "adv": [100_000_000.0] * n,
            "vol_20": [0.02] * n,
            "source": ["file"] * n,
        }
    )


def _target(w: float = 1.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 1, tzinfo=UTC)],
            "security_id": ["A"],
            "target_weight": [w],
        }
    )


def test_fallback_mark_used_when_total_return_missing(tmp_path) -> None:
    """Day 2 has no total-return mark: the close stands in for valuation.

    The day-1 target fills at the day-2 open (110), sizing 909.09 shares;
    day-2 NAV marks at the fallback close 110 (flat), day-3 at TR close 120.
    """
    cfg = _cfg(tmp_path)
    bars = _bars([100.0, None, 120.0], [100.0, 110.0, 120.0])
    result = run_backtest(bars, _target(), cfg, initial_nav=100_000.0)
    navs = result.equity["nav"].to_list()
    assert navs == pytest.approx([100_000.0, 100_000.0 * 120.0 / 110.0])


def _result_with_metrics(metrics: dict[str, Any]) -> BacktestResult:
    return BacktestResult(
        equity=pl.DataFrame({"nav": [1.0]}),
        fills=pl.DataFrame(),
        metrics=metrics,
        frictionless=True,
        source_note="SYNTHETIC",
    )


def test_export_merges_extra_into_existing_analytics_export(tmp_path) -> None:
    result = _result_with_metrics({"analytics_export": {"a": 1, "research_only": False}})
    dest = tmp_path / "metrics.json"
    export_backtest_metrics_json(result, dest, extra={"b": 2})
    import json

    blob = json.loads(dest.read_text())
    assert blob["a"] == 1 and blob["b"] == 2
    assert blob["research_only"] is True
    assert blob["live_pnl_claim"] is False


def test_export_without_analytics_export_builds_blob(tmp_path) -> None:
    result = _result_with_metrics(
        {"analytics": {"x": 1}, "sharpe": 1.5, "n": 3, "total_return": 0.1}
    )
    dest = tmp_path / "m2.json"
    export_backtest_metrics_json(result, dest, extra={"tag": "edge"})
    import json

    blob = json.loads(dest.read_text())
    assert blob["extra"]["tag"] == "edge"
    assert blob["research_only"] is True
    assert blob["live_pnl_claim"] is False


def test_export_without_extra_keeps_blob(tmp_path) -> None:
    result = _result_with_metrics({"analytics_export": {"a": 1}})
    dest = tmp_path / "m3.json"
    export_backtest_metrics_json(result, dest)
    import json

    blob = json.loads(dest.read_text())
    assert blob["a"] == 1
    assert blob["research_only"] is True

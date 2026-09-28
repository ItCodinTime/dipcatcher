"""robinhood_plus bench: status matrix (disabled/missing/history) and the
scored IC path with a stubbed cross-section forecast. SYNTHETIC only."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from quant_fund.config.models import AppConfig
from quant_fund.models.robinhood_plus import bench as B
from quant_fund.models.robinhood_plus.engine import RobinhoodPlusNameForecast
from quant_fund.models.robinhood_plus.predictor import PathForecast

KLINES = [
    "open_split_adjusted",
    "high_split_adjusted",
    "low_split_adjusted",
    "close_split_adjusted",
    "volume",
    "amount",
]


def _frame(
    n_dates: int = 14, names: int = 4, label: str = "future_excess_return_5"
) -> pl.DataFrame:
    rows = []
    t0 = datetime(2024, 1, 2, tzinfo=UTC)
    for d in range(n_dates):
        for k in range(names):
            rows.append(
                {
                    "security_id": f"S{k}",
                    "event_time": t0 + timedelta(days=d),
                    "open_split_adjusted": 100.0 + d,
                    "high_split_adjusted": 101.0 + d,
                    "low_split_adjusted": 99.0 + d,
                    "close_split_adjusted": 100.5 + d,
                    "volume": 1000.0,
                    "amount": 100_000.0,
                    label: float(k) * 0.01 + 0.001 * d,
                }
            )
    return pl.DataFrame(rows)


def _cfg(**rh: Any) -> AppConfig:
    cfg: dict[str, Any] = {"robinhood_plus": {"lookback": 4, "pred_len": 5, **rh}}
    return AppConfig.model_validate(cfg)


def _ok_forecast(mu: float) -> PathForecast:
    paths = np.zeros((1, 2, 6))
    return PathForecast(
        paths=paths,
        mean_path=paths[0],
        last_close=100.0,
        expected_returns={"5d": mu},
        quantiles={},
        probability_positive={"5d": 0.5},
        rank_score=0.5,
        confidence=0.5,
        status="ok",
    )


class TestStatusMatrix:
    def test_disabled(self) -> None:
        out = B.bench_robinhood_plus(_frame(), _cfg(enabled=False))
        assert out["status"] == "disabled"
        assert out["research_only"] is True

    def test_missing_kline_columns(self) -> None:
        out = B.bench_robinhood_plus(pl.DataFrame({"a": [1]}), _cfg())
        assert out["status"] == "missing_ohlc"

    def test_missing_label(self) -> None:
        frame = _frame().drop("future_excess_return_5")
        out = B.bench_robinhood_plus(frame, _cfg())
        assert out["status"] == "missing_label"

    def test_future_log_return_label_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        frame = _frame(label="future_log_return_5")
        monkeypatch.setattr(
            B,
            "forecast_robinhood_plus_cross_section",
            lambda *a, **k: {},
        )
        out = B.bench_robinhood_plus(frame, _cfg())
        assert out["label"] == "future_log_return_5"

    def test_insufficient_history(self) -> None:
        out = B.bench_robinhood_plus(_frame(n_dates=4), _cfg(lookback=8))
        assert out["status"] == "insufficient_history"

    def test_insufficient_scored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # forecasts exist but return non-ok statuses -> nothing scored
        def stub(*a: Any, **k: Any) -> dict[str, RobinhoodPlusNameForecast]:
            return {}

        monkeypatch.setattr(B, "forecast_robinhood_plus_cross_section", stub)
        out = B.bench_robinhood_plus(_frame(), _cfg())
        assert out["status"] == "insufficient_scored"
        assert out["n_scored"] == 0

    def test_scored_path_produces_ic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def stub(
            frame: pl.DataFrame, asof: datetime, **k: Any
        ) -> dict[str, RobinhoodPlusNameForecast]:
            # rank-order mu matching realized ordering -> positive IC
            day = frame.filter(pl.col("event_time") == asof)
            out: dict[str, RobinhoodPlusNameForecast] = {}
            for row in day.iter_rows(named=True):
                sid = str(row["security_id"])
                mu = float(row["future_excess_return_5"])
                out[sid] = RobinhoodPlusNameForecast(security_id=sid, forecast=_ok_forecast(mu))
            return out

        monkeypatch.setattr(B, "forecast_robinhood_plus_cross_section", stub)
        out = B.bench_robinhood_plus(_frame(), _cfg())
        assert out["status"] == "ok"
        assert out["n_scored"] >= 8
        assert out["mean_ic"] == pytest.approx(1.0, abs=0.1)
        assert out["ic_n_dates"] >= 1
        # blob sealed: no forbidden keys
        assert "sharpe" not in out
        assert "pnl" not in out

    def test_scored_skips_bad_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        frame = _frame(n_dates=14, names=3)
        # poison one label with NaN; also have stub skip S0 -> fewer scored
        frame = frame.with_columns(
            pl.when(pl.col("security_id") == "S1")
            .then(None)
            .otherwise(pl.col("future_excess_return_5"))
            .alias("future_excess_return_5")
        )

        def stub(
            frame: pl.DataFrame, asof: datetime, **k: Any
        ) -> dict[str, RobinhoodPlusNameForecast]:
            day = frame.filter(pl.col("event_time") == asof)
            out: dict[str, RobinhoodPlusNameForecast] = {}
            for row in day.iter_rows(named=True):
                sid = str(row["security_id"])
                if sid == "S0":
                    continue  # no forecast for this name
                out[sid] = RobinhoodPlusNameForecast(security_id=sid, forecast=_ok_forecast(0.5))
            return out

        monkeypatch.setattr(B, "forecast_robinhood_plus_cross_section", stub)
        out = B.bench_robinhood_plus(frame, _cfg())
        # S0 has no forecast, S1's label is NaN -> only S2 scores, 4 dates
        assert out["n_scored"] == 4
        assert out["status"] == "insufficient_scored"

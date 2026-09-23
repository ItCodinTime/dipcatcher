from datetime import UTC, datetime, timedelta

import pandas as pd
import polars as pl
import pytest
from quant_fund.models.kronos import KronosAdapter, validate_ohlcv_frame

from quant_fund.config.models import AppConfig


def _bars(n: int = 8) -> pd.DataFrame:
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    return pd.DataFrame(
        {
            "event_time": [start + timedelta(minutes=i) for i in range(n)],
            "available_time": [start + timedelta(minutes=i) for i in range(n)],
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [1000.0] * n,
            "amount": [100_250.0 + i * 1000 for i in range(n)],
        }
    )


class _Predictor:
    def predict(self, df, x_timestamp, y_timestamp, pred_len, **kwargs):
        last = float(df["close"].iloc[-1])
        return pd.DataFrame(
            {
                "open": [last + 1.0] * pred_len,
                "high": [last + 2.0] * pred_len,
                "low": [last] * pred_len,
                "close": [last + 1.5] * pred_len,
                "volume": [1000.0] * pred_len,
                "amount": [100_000.0] * pred_len,
            },
            index=y_timestamp,
        )


def test_kronos_config_is_disabled_and_strict() -> None:
    cfg = AppConfig.model_validate({})
    assert cfg.train.kronos.enabled is False
    with pytest.raises(ValueError, match="extra_forbidden"):
        AppConfig.model_validate({"train": {"kronos": {"unknown": 1}}})


def test_validate_ohlcv_and_convert_prediction_to_asset_forecast() -> None:
    frame = _bars()
    validate_ohlcv_frame(frame)
    result = KronosAdapter(_Predictor(), pred_len=2, lookback=5).forecast_asset(
        frame, security_id="A", symbol="A", asof=frame["event_time"].iloc[-1]
    )
    assert result.expected_returns["2b"] > 0
    assert result.quantiles["2b"][0.05] < result.quantiles["2b"][0.95]
    assert result.model_version.startswith("kronos.")


def test_kronos_rejects_future_and_nonfinite_data() -> None:
    frame = _bars()
    frame.loc[0, "close"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_ohlcv_frame(frame)

    frame = _bars()
    frame.loc[len(frame) - 1, "available_time"] = frame["event_time"].iloc[-1] + timedelta(
        minutes=1
    )
    with pytest.raises(ValueError, match="point-in-time"):
        KronosAdapter(_Predictor(), pred_len=2, lookback=5).forecast_asset(
            frame, security_id="A", symbol="A", asof=frame["event_time"].iloc[-1]
        )


def test_kronos_pipeline_uses_causal_silver_rows() -> None:
    from quant_fund.pipeline.kronos import forecast_kronos_frame

    frame = pl.from_pandas(_bars()).with_columns(
        pl.lit("A").alias("security_id"), pl.lit("A").alias("symbol")
    )
    state = forecast_kronos_frame(
        AppConfig.model_validate({"train": {"kronos": {"lookback": 5, "pred_len": 2}}}),
        asof=_bars()["event_time"].iloc[-1],
        frame=frame,
        predictor=_Predictor(),
    )
    assert len(state.forecasts) == 1
    assert state.forecasts[0].security_id == "A"

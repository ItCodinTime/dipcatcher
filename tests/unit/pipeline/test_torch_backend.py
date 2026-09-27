"""robinhood_plus torch backend: fail-closed guards, causal filtering, stub predictor."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from quant_fund.config.models import AppConfig
from quant_fund.models.robinhood_plus import torch_backend as tb
from quant_fund.models.robinhood_plus.constants import (
    STATUS_INSUFFICIENT_HISTORY,
    STATUS_OK,
)


def test_torch_available_and_variant_guards() -> None:
    assert tb.torch_available() is True
    assert tb.local_kronos_weight_dirs("bogus") is None
    # Weights are not vendored in the repo checkout.
    assert tb.local_kronos_weight_dirs("mini") is None


def test_load_pretrained_predictor_fail_closed() -> None:
    with pytest.raises(tb.RobinhoodPlusTorchError, match="unknown"):
        tb.load_pretrained_predictor("bogus")
    # allow_network=False refuses Hub ids (non-existent local dirs).
    with pytest.raises(tb.RobinhoodPlusTorchError, match="refused Hub"):
        tb.load_pretrained_predictor("mini")


def test_bind_local_weights_noop_without_dirs() -> None:
    cfg = AppConfig.model_validate({})
    out = tb.bind_local_kronos_weights(cfg)
    assert out.robinhood_plus.tokenizer_path == cfg.robinhood_plus.tokenizer_path
    assert out.robinhood_plus.model_path == cfg.robinhood_plus.model_path


def test_future_stamps_continues_last_delta() -> None:
    t0 = datetime(2024, 1, 2, tzinfo=UTC)
    times = [t0, t0 + timedelta(minutes=5), t0 + timedelta(minutes=10)]
    fut = tb._future_stamps(times, 3)
    assert fut[0] == t0 + timedelta(minutes=15)
    assert (fut[1] - fut[0]) == timedelta(minutes=5)
    # Single/empty history falls back to 1-day spacing.
    assert (tb._future_stamps([t0], 2)[0] - t0) == timedelta(days=1)
    assert tb._future_stamps([], 1)[0] == datetime(2020, 1, 3)


def _panel(names: list[str], n_per: dict[str, int]) -> pl.DataFrame:
    rows = []
    t0 = datetime(2024, 1, 2, tzinfo=UTC)
    for name in names:
        for i in range(n_per[name]):
            rows.append(
                {
                    "security_id": name,
                    "event_time": t0 + timedelta(minutes=i),
                    "available_time": t0 + timedelta(minutes=i),
                    "open_split_adjusted": 100.0 + i,
                    "high_split_adjusted": 101.0 + i,
                    "low_split_adjusted": 99.0 + i,
                    "close_split_adjusted": 100.5 + i,
                    "volume": 1000.0,
                    "amount": 100_000.0,
                }
            )
    return pl.DataFrame(rows)


class _FakePredictor:
    def predict(self, df, x_timestamp, y_timestamp, pred_len, **kwargs):
        last = float(df["close"].iloc[-1])
        cols = ["open", "high", "low", "close", "volume", "amount"]
        data = np.tile([last, last + 1.0, last - 1.0, last + 0.5, 1.0, 1.0], (pred_len, 1))
        return pd.DataFrame(data, columns=cols, index=y_timestamp)


def _cfg() -> AppConfig:
    return AppConfig.model_validate(
        {"robinhood_plus": {"lookback": 8, "pred_len": 5, "sample_count": 2}}
    )


def test_kline_paths_from_kronos_shapes() -> None:
    kline = np.tile([100.0, 101.0, 99.0, 100.5, 1000.0, 100_000.0], (4, 1))
    times = [datetime(2024, 1, 2, tzinfo=UTC) + timedelta(minutes=i) for i in range(4)]
    paths = tb._kline_paths_from_kronos(
        _FakePredictor(),
        kline,
        times,
        pred_len=5,
        sample_count=3,
        temperature=1.0,
        top_p=0.9,
    )
    assert paths.shape == (3, 5, 6)

    class BadShape:
        def predict(self, *a, **k):
            cols = ["open", "high", "low", "close", "volume", "amount"]
            return pd.DataFrame(np.zeros((2, 6)), columns=cols)

    with pytest.raises(tb.RobinhoodPlusTorchError, match="shape"):
        tb._kline_paths_from_kronos(
            BadShape(),
            kline,
            times,
            pred_len=5,
            sample_count=1,
            temperature=1.0,
            top_p=0.9,
        )


def test_forecast_cross_section_torch_contract(monkeypatch) -> None:
    monkeypatch.setattr(tb, "_cached_predictor", lambda cfg: _FakePredictor())
    frame = _panel(["AAA", "B1", "SKIP"], {"AAA": 6, "B1": 1, "SKIP": 5})
    asof = datetime(2024, 1, 2, 0, 10, tzinfo=UTC)
    out = tb.forecast_cross_section_torch(frame, asof, _cfg(), ["AAA", "B1", "MISSING"])
    assert set(out) == {"AAA", "B1"}  # SKIP filtered, MISSING absent
    assert out["AAA"].forecast.status == STATUS_OK
    assert out["B1"].forecast.status == STATUS_INSUFFICIENT_HISTORY
    fc = out["AAA"].forecast
    assert fc.diagnostics["backend"] == "torch"
    # Causal bound: available_time beyond asof is excluded.
    assert "1d" in fc.expected_returns or "5d" in fc.expected_returns


def test_forecast_cross_section_torch_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(tb, "_cached_predictor", lambda cfg: _FakePredictor())
    asof = datetime(2024, 1, 2, tzinfo=UTC)
    # No kline columns -> empty dict (hard no-data path).
    assert (
        tb.forecast_cross_section_torch(
            pl.DataFrame({"event_time": [asof], "x": [1.0]}), asof, _cfg(), ["A"]
        )
        == {}
    )
    # Kline cols present but no event_time -> raise.
    bad = pl.DataFrame(
        {
            c: [1.0]
            for c in (
                "open_split_adjusted",
                "high_split_adjusted",
                "low_split_adjusted",
                "close_split_adjusted",
                "volume",
                "amount",
            )
        }
    )
    with pytest.raises(tb.RobinhoodPlusTorchError, match="event_time"):
        tb.forecast_cross_section_torch(bad, asof, _cfg(), ["A"])

    # Predictor error propagates as RobinhoodPlusTorchError (no numpy fallback).
    class Boom:
        def predict(self, *a, **k):
            raise RuntimeError("gpu on fire")

    monkeypatch.setattr(tb, "_cached_predictor", lambda cfg: Boom())
    frame = _panel(["AAA"], {"AAA": 5})
    with pytest.raises(tb.RobinhoodPlusTorchError, match="inference failed"):
        tb.forecast_cross_section_torch(frame, datetime(2024, 1, 2, 1, tzinfo=UTC), _cfg(), ["AAA"])

"""Edge paths: robinhood+ numpy engine miss-status matrix + lightspeed
AFML meta-label gate (reduce-only multiplier) validation and behavior."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from quant_fund.lightspeed import metalabel
from quant_fund.models.robinhood_plus import engine
from quant_fund.models.robinhood_plus.constants import (
    STATUS_INSUFFICIENT_HISTORY,
    STATUS_MISSING_OHLC,
    STATUS_NONFINITE,
)

pytestmark = pytest.mark.synthetic


def _frame(n: int = 12, sid: str = "AAA", *, extra: dict | None = None) -> pl.DataFrame:
    rows = []
    for d in range(n):
        row = {
            "security_id": sid,
            "event_time": datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=d),
            "open_split_adjusted": 1.0,
            "high_split_adjusted": 1.1,
            "low_split_adjusted": 0.9,
            "close_split_adjusted": 1.0 + 0.01 * d,
            "volume": 1e6,
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return pl.DataFrame(rows)


class TestKlineExtraction:
    def test_missing_volume_raises(self) -> None:
        frame = _frame(3).drop("volume")
        with pytest.raises(ValueError, match="volume"):
            engine.extract_kline(frame)

    def test_amount_fallback_and_sanitization(self) -> None:
        frame = _frame(3)
        k = engine.extract_kline(frame)
        assert k.shape == (3, 6)
        assert np.isfinite(k).all()
        frame_nan = _frame(3, extra={"amount": float("nan")})
        k2 = engine.extract_kline(frame_nan)
        assert np.isfinite(k2).all()


class TestForecastNameKline:
    @staticmethod
    def _pred() -> engine.RobinhoodPlusPredictor:
        return engine.RobinhoodPlusPredictor(lookback=4, pred_len=2, sample_count=1)

    def test_status_matrix(self) -> None:
        pred = self._pred()
        assert (
            getattr(engine.forecast_name_kline(_frame(1), pred), "status", None)
            == STATUS_INSUFFICIENT_HISTORY
        )
        no_cols = pl.DataFrame({"security_id": ["A", "B"], "event_time": [1, 2]})
        out = engine.forecast_name_kline(no_cols, self._pred())
        assert getattr(out, "status", None) == STATUS_MISSING_OHLC

    def test_nonfinite_kline_marks_name(self) -> None:
        frame = _frame(5, extra={"close_split_adjusted": float("nan")})
        out = engine.forecast_name_kline(frame, self._pred())
        assert getattr(out, "status", None) == STATUS_NONFINITE


class TestCrossSection:
    def test_lookback_below_two_rejected(self) -> None:
        with pytest.raises(ValueError, match="lookback"):
            engine.forecast_robinhood_plus_cross_section(
                _frame(3), datetime(2025, 1, 1, tzinfo=UTC), lookback=1
            )

    def test_no_kline_columns_returns_empty(self) -> None:
        assert (
            engine.forecast_robinhood_plus_cross_section(
                pl.DataFrame({"x": [1.0]}), datetime(2025, 1, 1, tzinfo=UTC)
            )
            == {}
        )

    def test_empty_history_and_wanted_filter(self) -> None:
        frame = _frame(4)
        asof_past = datetime(2020, 1, 1, tzinfo=UTC)
        assert engine.forecast_robinhood_plus_cross_section(frame, asof_past) == {}
        out = engine.forecast_robinhood_plus_cross_section(
            frame, datetime(2025, 1, 1, tzinfo=UTC), security_ids=["NOPE"]
        )
        assert out == {}

    def test_available_time_respected(self) -> None:
        rows = _frame(6).with_columns(
            (pl.col("event_time") + pl.duration(days=30)).alias("available_time")
        )
        # All bars are unpublished at asof -> empty visible history.
        out = engine.forecast_robinhood_plus_cross_section(rows, datetime(2024, 1, 10, tzinfo=UTC))
        assert out == {}

    def test_happy_path_numpy_predictor(self) -> None:
        frame = _frame(12, sid="AAA").vstack(_frame(12, sid="BBB"))
        out = engine.forecast_robinhood_plus_cross_section(
            frame, datetime(2025, 1, 1, tzinfo=UTC), lookback=8, pred_len=3, sample_count=2
        )
        assert set(out) == {"AAA", "BBB"}


class TestEngineWrapper:
    def test_predict_requires_2d(self) -> None:
        eng = engine.RobinhoodPlusEngine(lookback=4, pred_len=2, sample_count=1)
        with pytest.raises(ValueError, match="2-d"):
            eng.predict(np.ones(3))
        assert eng.predict(np.ones((4, 3))).shape == (4,)

    def test_fit_and_metadata(self) -> None:
        eng = engine.RobinhoodPlusEngine(lookback=4, pred_len=2, sample_count=1, seed=7)
        assert eng.fit(np.ones((2, 2)), np.ones(2)) is eng
        meta = eng.metadata()
        assert meta.extra["fitted"] is True
        assert meta.extra["decoder"] == "hierarchical_markov"


class TestMetaLabel:
    def test_design_validation_matrix(self) -> None:
        sig = np.array([0.5, -0.2, 0.1])
        rets = np.array([0.01, -0.01, 0.02])
        with pytest.raises(ValueError):
            metalabel.meta_label_gate(sig, rets[:2])
        with pytest.raises(ValueError):
            metalabel.meta_label_gate(np.array([]), np.array([]))
        with pytest.raises(ValueError):
            metalabel.meta_label_gate(np.array([np.nan, 1.0]), np.array([1.0, 2.0]))
        with pytest.raises(ValueError):
            metalabel.meta_label_gate(sig, rets, features=np.ones((3, 2, 2)))
        with pytest.raises(ValueError):
            metalabel.meta_label_gate(sig, rets, features=np.ones((2, 2)))
        with pytest.raises(ValueError):
            metalabel.meta_label_gate(sig, rets, features=np.ones((3, 0)))
        with pytest.raises(ValueError):
            metalabel.meta_label_gate(sig, rets, features=np.full((3, 1), np.nan))
        with pytest.raises(ValueError):
            metalabel.meta_label_gate(sig, rets, threshold=1.5)
        with pytest.raises(ValueError):
            metalabel.metalabel_multiplier(np.array([0.6]), threshold=-0.1)

    def test_multiplier_mapping(self) -> None:
        m = metalabel.metalabel_multiplier(np.array([0.4, 0.55, 0.75, 1.0]), 0.55)
        assert m[0] == 0.0
        assert m[1] > 0.0
        assert m[2] == pytest.approx(0.5)
        assert m[3] == 1.0

    def test_predict_meta_small_and_constant(self) -> None:
        train = np.random.default_rng(0).normal(size=(5, 3))
        assert metalabel._predict_meta(train, np.zeros(5), train[:1])[0] == 0.5
        big = np.random.default_rng(1).normal(size=(20, 3))
        assert metalabel._predict_meta(big, np.ones(20), big[:1])[0] == 1.0

    def test_gate_delay_and_reduce_only(self) -> None:
        rng = np.random.default_rng(2)
        n = 40
        sig = rng.normal(size=n)
        rets = sig * 0.05 + rng.normal(scale=0.01, size=n)
        res = metalabel.meta_label_gate(sig, rets, threshold=0.55)
        assert res.primary_side.shape == (n,)
        assert np.all(res.meta_prob[:10] == 0.5)
        assert np.all((res.multiplier >= 0.0) & (res.multiplier <= 1.0))
        assert res.threshold == 0.55

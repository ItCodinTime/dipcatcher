"""Kronos research adapter: artifact digests, OHLCV/PIT validation, predict seam."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_fund.models import kronos
from quant_fund.models.kronos import (
    KronosAdapter,
    load_local_predictor,
    validate_local_artifact,
    validate_ohlcv_frame,
)


def _frame(n: int = 6, *, start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, n))
    open_ = close * (1.0 + rng.normal(0, 0.002, n))
    return pd.DataFrame(
        {
            "open": np.minimum(open_ * 1.01, open_ * 1.05),
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": np.full(n, 1e6),
            "event_time": dates,
            "available_time": dates + pd.Timedelta(hours=1),
        }
    )


class _Predictor:
    """Deterministic stand-in for the upstream KronosPredictor protocol."""

    def __init__(self, out: pd.DataFrame | None = None) -> None:
        self.calls: list[dict] = []
        self._out = out

    def predict(self, df, x_timestamp, y_timestamp, pred_len, **kwargs):
        self.calls.append(
            {
                "df": df,
                "x": list(x_timestamp),
                "y": list(y_timestamp),
                "pred_len": pred_len,
                "kwargs": kwargs,
            }
        )
        if self._out is not None:
            return self._out
        base = float(df["close"].iloc[-1])
        idx = range(pred_len)
        return pd.DataFrame(
            {
                "open": [base * (1 + 0.001 * i) for i in idx],
                "high": [base * (1 + 0.002 + 0.001 * i) for i in idx],
                "low": [base * (1 - 0.002 + 0.001 * i) for i in idx],
                "close": [base * (1 + 0.001 * i) for i in idx],
                "volume": np.full(pred_len, 1e5),
                "amount": np.full(pred_len, 1e7),
            }
        )


class TestFiniteDigest:
    def test_file_digest_matches_content(self, tmp_path: Path) -> None:
        p = tmp_path / "blob.bin"
        p.write_bytes(b"abc" * 10)
        assert kronos._finite_digest(p) == hashlib.sha256(b"abc" * 10).hexdigest()

    def test_dir_digest_covers_names_and_bytes(self, tmp_path: Path) -> None:
        sub = tmp_path / "pkg"
        (sub / "a").mkdir(parents=True)
        (sub / "a" / "x.txt").write_bytes(b"xx")
        (sub / "b.txt").write_bytes(b"yy")
        d1 = kronos._finite_digest(sub)
        (sub / "b.txt").write_bytes(b"yz")
        assert kronos._finite_digest(sub) != d1

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            kronos._finite_digest(tmp_path / "nope")


class TestValidateLocalArtifact:
    def test_must_be_directory(self, tmp_path: Path) -> None:
        p = tmp_path / "file.txt"
        p.write_text("x")
        with pytest.raises(ValueError, match="existing local directory"):
            validate_local_artifact(p)

    def test_digest_mismatch_fails_closed(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="checksum mismatch"):
            validate_local_artifact(tmp_path, expected_sha256="0" * 64)

    def test_digest_match_resolves(self, tmp_path: Path) -> None:
        digest = kronos._finite_digest(tmp_path)
        assert validate_local_artifact(tmp_path, expected_sha256=digest) == tmp_path.resolve()

    def test_none_digest_skips_check(self, tmp_path: Path) -> None:
        assert validate_local_artifact(tmp_path) == tmp_path.resolve()


class TestValidateOhlcvFrame:
    def test_type_and_columns(self) -> None:
        with pytest.raises(TypeError, match="pandas DataFrame"):
            validate_ohlcv_frame(np.zeros((3, 4)))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="missing required columns"):
            validate_ohlcv_frame(pd.DataFrame({"open": [1.0]}))
        with pytest.raises(ValueError, match="must not be empty"):
            validate_ohlcv_frame(pd.DataFrame(columns=["open", "high", "low", "close"]))

    def test_volume_amount_defaults(self) -> None:
        out = validate_ohlcv_frame(
            pd.DataFrame({"open": [1.0], "high": [1.1], "low": [0.9], "close": [1.05]})
        )
        assert out["volume"].tolist() == [0.0]
        assert out["amount"].tolist() == [0.0]

    def test_non_finite_and_nonpositive_rejected(self) -> None:
        good = pd.DataFrame(
            {
                "open": [1.0],
                "high": [1.1],
                "low": [0.9],
                "close": [1.05],
                "volume": [1.0],
                "amount": [1.0],
            }
        )
        for col, bad in (
            ("open", np.nan),
            ("close", np.inf),
            ("open", 0.0),
            ("close", -1.0),
        ):
            frame = good.copy()
            frame[col] = bad
            with pytest.raises((ValueError, TypeError)):
                validate_ohlcv_frame(frame)

    def test_candle_geometry_enforced(self) -> None:
        base = {"open": 1.0, "close": 1.0, "volume": 1.0, "amount": 1.0}
        with pytest.raises(ValueError, match="high must cover"):
            validate_ohlcv_frame(pd.DataFrame([{**base, "high": 0.5, "low": 0.5}]))
        with pytest.raises(ValueError, match="low must not exceed"):
            validate_ohlcv_frame(pd.DataFrame([{**base, "high": 1.5, "low": 1.2}]))
        with pytest.raises(ValueError, match="non-negative"):
            validate_ohlcv_frame(pd.DataFrame([{**base, "high": 1.5, "low": 0.5, "volume": -1.0}]))

    def test_numeric_coercion_errors(self) -> None:
        frame = pd.DataFrame({"open": ["x"], "high": [1.1], "low": [0.9], "close": [1.0]})
        with pytest.raises((ValueError, TypeError)):
            validate_ohlcv_frame(frame)


class TestValidateTimestamps:
    def test_requires_both_columns(self) -> None:
        frame = _frame().drop(columns=["available_time"])
        with pytest.raises(ValueError, match="event_time and available_time"):
            kronos._validate_timestamps(frame, "2024-01-10")

    def test_availability_violation(self) -> None:
        frame = _frame()
        frame.loc[2, "available_time"] = pd.Timestamp("2030-01-01", tz="UTC")
        with pytest.raises(ValueError, match="point-in-time"):
            kronos._validate_timestamps(frame, "2024-01-10")

    def test_duplicate_event_times(self) -> None:
        frame = _frame()
        frame.loc[3, "event_time"] = frame.loc[2, "event_time"]
        with pytest.raises(ValueError, match="duplicate event_time"):
            kronos._validate_timestamps(frame, "2024-01-10")

    def test_causal_filter_and_naive_asof(self) -> None:
        frame = _frame(8)
        selected, decision = kronos._validate_timestamps(frame, "2024-01-05T12:00")
        assert decision.tzinfo is not None
        assert len(selected) == 5
        assert selected["event_time"].is_monotonic_increasing


class TestKronosAdapter:
    def test_constructor_validation(self) -> None:
        with pytest.raises(ValueError, match="lookback"):
            KronosAdapter(_Predictor(), pred_len=0, lookback=4)
        with pytest.raises(ValueError, match="lookback"):
            KronosAdapter(_Predictor(), pred_len=2, lookback=1)

    def test_forecast_asset_shape_and_semantics(self) -> None:
        predictor = _Predictor()
        adapter = KronosAdapter(
            predictor, pred_len=3, lookback=4, horizon_name="3b", top_k=5, sample_count=2
        )
        out = adapter.forecast_asset(_frame(10), security_id="S1", symbol="AAA", asof="2024-01-11")
        assert out.security_id == "S1"
        assert out.model_version == "kronos.adapter.v1"
        call = predictor.calls[0]
        assert call["pred_len"] == 3
        assert call["kwargs"]["top_k"] == 5
        assert len(call["x"]) == 4  # lookback cap
        assert len(call["y"]) == 3
        assert set(out.quantiles["3b"]) == {0.05, 0.5, 0.95}
        q = out.quantiles["3b"]
        assert q[0.05] <= q[0.5] <= q[0.95]
        assert out.volatility["3b"] >= 1e-12
        assert out.confidence["3b"] == 0.5
        assert out.diagnostics["backend"] == "kronos"

    def test_pred_len_one_confidence(self) -> None:
        out = KronosAdapter(_Predictor(), pred_len=1, lookback=3).forecast_asset(
            _frame(8), security_id="S1", symbol="AAA", asof="2024-01-09"
        )
        assert out.confidence["1b"] == 0.25

    def test_insufficient_causal_history(self) -> None:
        adapter = KronosAdapter(_Predictor(), pred_len=2, lookback=8)
        with pytest.raises(ValueError, match="at least 8 causal bars"):
            adapter.forecast_asset(_frame(6), security_id="S", symbol="S", asof="2024-01-07")

    def test_predictor_bad_output_rejected(self) -> None:
        bad = pd.DataFrame({"open": [1.0]})
        adapter = KronosAdapter(_Predictor(out=bad), pred_len=2, lookback=4)
        with pytest.raises(ValueError, match="six-column OHLCV"):
            adapter.forecast_asset(_frame(10), security_id="S", symbol="S", asof="2024-01-11")

    def test_predictor_wrong_horizon_rejected(self) -> None:
        out = _Predictor().predict(_frame(4), [], [], 1)
        adapter = KronosAdapter(_Predictor(out=out), pred_len=3, lookback=4)
        with pytest.raises(ValueError, match="unexpected horizon"):
            adapter.forecast_asset(_frame(10), security_id="S", symbol="S", asof="2024-01-11")

    def test_predictor_nonfinite_path_rejected(self) -> None:
        out = _Predictor().predict(_frame(3), [], [], 2)
        out.loc[0, "close"] = np.nan
        adapter = KronosAdapter(_Predictor(out=out), pred_len=2, lookback=4)
        with pytest.raises((ValueError, TypeError)):
            adapter.forecast_asset(_frame(10), security_id="S", symbol="S", asof="2024-01-11")


def test_load_local_predictor_missing_backend(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Kronos backend is unavailable"):
        load_local_predictor(model_path=tmp_path, tokenizer_path=tmp_path)


def test_load_local_predictor_validates_artifacts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="existing local directory"):
        load_local_predictor(model_path=tmp_path / "nope", tokenizer_path=tmp_path)

"""Optional, research-only adapter for the Kronos candlestick model.

The module intentionally has no torch or upstream-Kronos imports at module import
 time. A predictor implementing ``predict`` can be injected for deterministic
research and unit tests.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from quant_fund.schemas.forecast import AssetForecast

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume", "amount")
PRICE_COLUMNS = ("open", "high", "low", "close")


class KronosPredictor(Protocol):
    def predict(
        self,
        df: pd.DataFrame,
        x_timestamp: Sequence[Any],
        y_timestamp: Sequence[Any],
        pred_len: int,
        **kwargs: Any,
    ) -> pd.DataFrame: ...


def _finite_digest(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    elif path.is_dir():
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(str(child.relative_to(path)).encode("utf-8"))
            with child.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
    else:
        raise ValueError(f"Kronos artifact does not exist: {path}")
    return digest.hexdigest()


def validate_local_artifact(path: Path, expected_sha256: str | None = None) -> Path:
    """Validate a local Kronos directory and optional deterministic digest."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"Kronos artifact must be an existing local directory: {resolved}")
    if expected_sha256 is not None:
        actual = _finite_digest(resolved)
        if not __import__("hmac").compare_digest(actual, expected_sha256.lower()):
            raise ValueError(f"Kronos artifact checksum mismatch: {resolved}")
    return resolved


def validate_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy after strict OHLCV/amount validation."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Kronos input must be a pandas DataFrame")
    missing = [name for name in PRICE_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"Kronos input is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Kronos input must not be empty")
    result = frame.copy()
    if "volume" not in result.columns:
        result["volume"] = 0.0
    if "amount" not in result.columns:
        result["amount"] = result["volume"].astype(float) * result[list(PRICE_COLUMNS)].mean(axis=1)
    for name in OHLCV_COLUMNS:
        result[name] = pd.to_numeric(result[name], errors="raise")
        if not np.isfinite(result[name].to_numpy(dtype=float)).all():
            raise ValueError(f"Kronos {name} values must be finite")
    if (
        (result["open"] <= 0).any()
        or (result["high"] <= 0).any()
        or (result["low"] <= 0).any()
        or (result["close"] <= 0).any()
    ):
        raise ValueError("Kronos prices must be strictly positive")
    if (result["high"] < result[["open", "close"]].max(axis=1)).any():
        raise ValueError("Kronos high must cover open and close")
    if (result["low"] > result[["open", "close"]].min(axis=1)).any():
        raise ValueError("Kronos low must not exceed open or close")
    if (result["volume"] < 0).any() or (result["amount"] < 0).any():
        raise ValueError("Kronos volume and amount must be non-negative")
    return result


def _validate_timestamps(frame: pd.DataFrame, asof: Any) -> tuple[pd.DataFrame, pd.Timestamp]:
    if "event_time" not in frame.columns or "available_time" not in frame.columns:
        raise ValueError("Kronos input requires event_time and available_time columns")
    events = pd.to_datetime(frame["event_time"], errors="raise", utc=True)
    available = pd.to_datetime(frame["available_time"], errors="raise", utc=True)
    decision = pd.Timestamp(asof)
    if decision.tzinfo is None:
        decision = decision.tz_localize("UTC")
    else:
        decision = decision.tz_convert("UTC")
    causal = (events <= decision) & (available <= decision)
    if ((events <= decision) & (available > decision)).any():
        raise ValueError("Kronos data violates point-in-time availability")
    selected = frame.loc[causal].copy()
    selected["event_time"] = events.loc[causal].to_numpy()
    selected["available_time"] = available.loc[causal].to_numpy()
    selected = selected.sort_values("event_time", kind="stable")
    if selected["event_time"].duplicated().any():
        raise ValueError("Kronos input contains duplicate event_time rows")
    return selected, decision


class KronosAdapter:
    """Convert an injected Kronos predictor's candle path into AssetForecast."""

    def __init__(
        self,
        predictor: KronosPredictor,
        *,
        pred_len: int,
        lookback: int,
        horizon_name: str | None = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 1,
    ) -> None:
        if pred_len < 1 or lookback < 2:
            raise ValueError("Kronos pred_len must be positive and lookback must be >= 2")
        self.predictor = predictor
        self.pred_len = pred_len
        self.lookback = lookback
        self.horizon_name = horizon_name or f"{pred_len}b"
        # Upstream predict() takes no `clip` — it is a KronosPredictor
        # constructor argument, applied in load_local_predictor.
        self.kwargs = {
            "T": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "sample_count": sample_count,
            "verbose": False,
        }

    def forecast_asset(
        self, frame: pd.DataFrame, *, security_id: str, symbol: str, asof: Any
    ) -> AssetForecast:
        validated = validate_ohlcv_frame(frame)
        causal, decision = _validate_timestamps(validated, asof)
        if len(causal) < self.lookback:
            raise ValueError(f"Kronos requires at least {self.lookback} causal bars")
        history = causal.tail(self.lookback).reset_index(drop=True)
        # Upstream calc_time_stamps requires a pandas Series (.dt accessor),
        # not a DatetimeIndex.
        x_timestamp = pd.Series(pd.to_datetime(history["event_time"]))
        step = x_timestamp.iloc[-1] - x_timestamp.iloc[-2]
        y_timestamp = pd.Series(
            pd.date_range(start=x_timestamp.iloc[-1] + step, periods=self.pred_len, freq=step)
        )
        output = self.predictor.predict(
            history[list(OHLCV_COLUMNS)], x_timestamp, y_timestamp, self.pred_len, **self.kwargs
        )
        if not isinstance(output, pd.DataFrame) or list(output.columns) != list(OHLCV_COLUMNS):
            raise ValueError("Kronos predictor must return a six-column OHLCV DataFrame")
        if len(output) != self.pred_len:
            raise ValueError("Kronos predictor returned an unexpected horizon length")
        predicted = validate_ohlcv_frame(output.reset_index(drop=True))
        last_close = float(history["close"].iloc[-1])
        path_returns = predicted["close"].to_numpy(dtype=float) / last_close - 1.0
        low_returns = predicted["low"].to_numpy(dtype=float) / last_close - 1.0
        high_returns = predicted["high"].to_numpy(dtype=float) / last_close - 1.0
        if not (
            np.isfinite(path_returns).all()
            and np.isfinite(low_returns).all()
            and np.isfinite(high_returns).all()
        ):
            raise ValueError("Kronos predicted returns must be finite")
        # The 5%/95% band is the predicted candle envelope: the worst low and
        # best high the model drew across the horizon, never narrower than the
        # close path itself. With sample_count=1 the close path can be flat —
        # reporting degenerate quantiles as a "distribution" would overstate
        # predictive sharpness, so the band falls back to the candle range.
        q50 = float(np.median(path_returns))
        q05 = float(min(np.quantile(path_returns, 0.05), low_returns.min()))
        q95 = float(max(np.quantile(path_returns, 0.95), high_returns.max()))
        volatility = float(np.std(path_returns, ddof=0))
        expected = float(np.mean(path_returns))
        return AssetForecast(
            security_id=security_id,
            symbol=symbol,
            asof=decision.to_pydatetime(),
            model_version="kronos.adapter.v1",
            expected_returns={self.horizon_name: expected},
            quantiles={self.horizon_name: {0.05: q05, 0.5: q50, 0.95: q95}},
            probability_positive={self.horizon_name: float(np.mean(path_returns > 0))},
            volatility={self.horizon_name: max(volatility, 1e-12)},
            confidence={self.horizon_name: 0.25 if self.pred_len == 1 else 0.5},
            diagnostics={
                "backend": "kronos",
                "quantile_band": "candle_envelope",
                "lookback": float(self.lookback),
                "pred_len": float(self.pred_len),
                "last_close": last_close,
            },
        )


def load_local_predictor(
    *,
    model_path: Path,
    tokenizer_path: Path,
    model_sha256: str | None = None,
    tokenizer_sha256: str | None = None,
    device: str = "cpu",
    max_context: int = 512,
    clip: float = 5.0,
) -> KronosPredictor:
    """Load upstream Kronos strictly from local, pre-downloaded artifacts."""
    model_dir = validate_local_artifact(model_path, model_sha256)
    tokenizer_dir = validate_local_artifact(tokenizer_path, tokenizer_sha256)
    try:
        from model import Kronos, KronosTokenizer
        from model import KronosPredictor as UpstreamPredictor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Kronos backend is unavailable; install the optional dipcatcher[nn] environment and expose the upstream model package"
        ) from exc
    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    model = Kronos.from_pretrained(str(model_dir), local_files_only=True)
    model.eval()
    tokenizer.eval()
    return cast(
        KronosPredictor,
        UpstreamPredictor(model, tokenizer, device=device, max_context=max_context, clip=clip),
    )

"""robinhood+ as a Dipcatcher core forecast engine.

Consumes point-in-time split-adjusted K-lines, emits path-derived expected
returns / quantiles / rank scores, and never bypasses fusion or the risk gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.models.base import JoblibMixin, ModelMeta
from quant_fund.models.robinhood_plus.constants import (
    AMOUNT_COL,
    DEFAULT_CLIP,
    DEFAULT_LOOKBACK,
    DEFAULT_MAX_CONTEXT,
    DEFAULT_PRED_LEN,
    DEFAULT_S1_BITS,
    DEFAULT_S2_BITS,
    DEFAULT_SAMPLE_COUNT,
    ENGINE_DISPLAY,
    ENGINE_NAME,
    ENGINE_VERSION,
    FAMILY,
    MODEL_VERSION,
    PRICE_COLS,
    PRICE_SPACE,
    STATUS_INSUFFICIENT_HISTORY,
    STATUS_MISSING_OHLC,
    STATUS_NONFINITE,
    VOLUME_COL,
)
from quant_fund.models.robinhood_plus.predictor import (
    PathForecast,
    RobinhoodPlusPredictor,
    empty_forecast,
)

Array = NDArray[np.float64]


@dataclass(frozen=True)
class RobinhoodPlusNameForecast:
    security_id: str
    forecast: PathForecast


def kline_columns_present(columns: list[str] | tuple[str, ...]) -> bool:
    needed = set(PRICE_COLS) | {VOLUME_COL}
    return needed.issubset(columns)


def extract_kline(frame: pl.DataFrame) -> Array:
    """Build a (T, 6) OHLCV+amount matrix from a single-name PIT window."""
    if frame.is_empty():
        raise ValueError("K-line window is empty")
    missing = [c for c in PRICE_COLS if c not in frame.columns]
    if missing:
        raise ValueError(f"K-line missing price columns: {missing}")
    if VOLUME_COL not in frame.columns:
        raise ValueError("K-line missing volume")
    ordered = frame.sort("event_time") if "event_time" in frame.columns else frame
    close = ordered[PRICE_COLS[3]].to_numpy().astype(np.float64)
    volume = ordered[VOLUME_COL].to_numpy().astype(np.float64)
    if AMOUNT_COL in ordered.columns:
        amount = ordered[AMOUNT_COL].to_numpy().astype(np.float64)
        amount = np.where(np.isfinite(amount), amount, np.maximum(volume, 0.0) * np.abs(close))
    else:
        amount = np.maximum(volume, 0.0) * np.abs(close)
    open_px = ordered[PRICE_COLS[0]].to_numpy().astype(np.float64)
    high = ordered[PRICE_COLS[1]].to_numpy().astype(np.float64)
    low = ordered[PRICE_COLS[2]].to_numpy().astype(np.float64)
    return np.column_stack([open_px, high, low, close, volume, amount])


def _visible_history(frame: pl.DataFrame, asof: datetime) -> pl.DataFrame:
    if "event_time" not in frame.columns:
        raise ValueError("panel missing event_time")
    hist = frame.filter(pl.col("event_time") <= asof)
    if "available_time" in hist.columns:
        hist = hist.filter(pl.col("available_time") <= asof)
    return hist


def forecast_name_kline(
    history: pl.DataFrame,
    predictor: RobinhoodPlusPredictor,
) -> PathForecast:
    if history.height < 2:
        return empty_forecast(STATUS_INSUFFICIENT_HISTORY)
    if not kline_columns_present(history.columns):
        return empty_forecast(STATUS_MISSING_OHLC)
    try:
        kline = extract_kline(history)
    except ValueError:
        return empty_forecast(STATUS_MISSING_OHLC)
    if not np.isfinite(kline).all():
        return empty_forecast(STATUS_NONFINITE)
    if kline.shape[0] < 2:
        return empty_forecast(STATUS_INSUFFICIENT_HISTORY)
    return predictor.predict_kline(kline)


def forecast_robinhood_plus_cross_section(
    frame: pl.DataFrame,
    asof: datetime,
    *,
    lookback: int = DEFAULT_LOOKBACK,
    pred_len: int = DEFAULT_PRED_LEN,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    s1_bits: int = DEFAULT_S1_BITS,
    s2_bits: int = DEFAULT_S2_BITS,
    clip: float = DEFAULT_CLIP,
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_context: int = DEFAULT_MAX_CONTEXT,
    seed: int = 42,
    decoder: str = "hierarchical_markov",
    security_ids: list[str] | None = None,
    quantile_levels: tuple[float, ...] = (0.05, 0.50, 0.95),
    horizons: tuple[int, ...] = (1, 5, 20),
) -> dict[str, RobinhoodPlusNameForecast]:
    """Causal per-name robinhood+ forecasts at ``asof``.

    Only bars with ``event_time <= asof`` (and ``available_time <= asof`` when
    present) enter the lookback. Names without a clean K-line window are stamped
    with a miss status rather than a fabricated path.
    """
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if not kline_columns_present(frame.columns):
        return {}
    hist = _visible_history(frame, asof)
    if hist.is_empty() or "security_id" not in hist.columns:
        return {}
    predictor = RobinhoodPlusPredictor(
        s1_bits=s1_bits,
        s2_bits=s2_bits,
        clip=clip,
        lookback=lookback,
        pred_len=pred_len,
        sample_count=sample_count,
        temperature=temperature,
        top_p=top_p,
        max_context=max_context,
        seed=seed,
        decoder=decoder,  # type: ignore[arg-type]
        quantile_levels=quantile_levels,
        horizons=horizons,
    )
    wanted = set(security_ids) if security_ids is not None else None
    out: dict[str, RobinhoodPlusNameForecast] = {}
    for group in hist.partition_by("security_id", maintain_order=True):
        security_id = str(group["security_id"][0])
        if wanted is not None and security_id not in wanted:
            continue
        window = group.sort("event_time").tail(lookback)
        out[security_id] = RobinhoodPlusNameForecast(
            security_id=security_id,
            forecast=forecast_name_kline(window, predictor),
        )
    return out


class RobinhoodPlusEngine(JoblibMixin):
    """ForecastModel wrapper around the robinhood+ K-line predictor."""

    def __init__(
        self,
        *,
        lookback: int = DEFAULT_LOOKBACK,
        pred_len: int = DEFAULT_PRED_LEN,
        sample_count: int = DEFAULT_SAMPLE_COUNT,
        s1_bits: int = DEFAULT_S1_BITS,
        s2_bits: int = DEFAULT_S2_BITS,
        seed: int = 42,
        decoder: str = "hierarchical_markov",
    ) -> None:
        self.lookback = int(lookback)
        self.pred_len = int(pred_len)
        self.sample_count = int(sample_count)
        self.s1_bits = int(s1_bits)
        self.s2_bits = int(s2_bits)
        self.seed = int(seed)
        self.decoder = decoder
        self._fitted = False

    def fit(self, x: Array, y: Array, **kwargs: Any) -> RobinhoodPlusEngine:
        # Unsupervised tokenizer + lookback-conditioned decoder. Labels are
        # evaluation-only and must not enter the K-line codes.
        _ = (x, y, kwargs)
        self._fitted = True
        return self

    def predict(self, x: Array) -> Array:
        # Tabular feature rows are not K-lines. Callers that only have a
        # design matrix must use forecast_robinhood_plus_cross_section.
        rows = np.asarray(x, dtype=np.float64)
        if rows.ndim != 2:
            raise ValueError("robinhood+ predict expects a 2-d array")
        return np.zeros(rows.shape[0], dtype=np.float64)

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family=FAMILY,
            name=ENGINE_NAME,
            version=ENGINE_VERSION,
            extra={
                "display": ENGINE_DISPLAY,
                "model_version": MODEL_VERSION,
                "price_space": PRICE_SPACE,
                "lookback": self.lookback,
                "pred_len": self.pred_len,
                "sample_count": self.sample_count,
                "s1_bits": self.s1_bits,
                "s2_bits": self.s2_bits,
                "decoder": self.decoder,
                "fitted": self._fitted,
            },
        )

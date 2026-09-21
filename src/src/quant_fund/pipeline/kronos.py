"""Causal pipeline entry point for the optional Kronos candlestick adapter."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.models.kronos import KronosAdapter, KronosPredictor, load_local_predictor
from quant_fund.schemas.forecast import MarketState


def _as_datetime(value: Any) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _group_to_pandas(group: pl.DataFrame) -> pd.DataFrame:
    """Convert a small per-security Polars slice without requiring Arrow."""
    return pd.DataFrame(group.to_dicts())


def forecast_kronos_frame(
    config: AppConfig,
    *,
    asof: datetime,
    frame: pl.DataFrame,
    predictor: KronosPredictor | None = None,
) -> MarketState:
    """Produce causal Kronos forecasts for each security in ``frame``.

    Rows are filtered by both event and availability timestamps before any
    predictor sees them.  The adapter then performs its own strict OHLCV and
    timestamp validation.  Kronos remains opt-in; an injected predictor is
    accepted for deterministic research and tests even when the config flag is
    disabled.
    """
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("Kronos pipeline frame must be a Polars DataFrame")
    required = {"event_time", "available_time", "security_id", *{"open", "high", "low", "close"}}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Kronos pipeline frame is missing required columns: {missing}")
    if frame.is_empty():
        return MarketState(asof=_as_datetime(asof), forecasts=[])

    decision = _as_datetime(asof)
    causal = frame.filter(
        (pl.col("event_time") <= decision) & (pl.col("available_time") <= decision)
    )
    if causal.is_empty():
        return MarketState(asof=decision, forecasts=[])

    settings = config.train.kronos
    if predictor is None:
        if not settings.enabled:
            raise ValueError("Kronos is disabled; enable train.kronos or inject a predictor")
        if settings.model_path is None or settings.tokenizer_path is None:
            raise ValueError("enabled Kronos requires local model and tokenizer paths")
        predictor = load_local_predictor(
            model_path=Path(settings.model_path),
            tokenizer_path=Path(settings.tokenizer_path),
            model_sha256=settings.model_sha256,
            tokenizer_sha256=settings.tokenizer_sha256,
            device=settings.device,
            max_context=settings.max_context,
        )

    adapter = KronosAdapter(
        predictor,
        pred_len=settings.pred_len,
        lookback=settings.lookback,
        horizon_name=settings.horizon_name,
        temperature=settings.temperature,
        top_k=settings.top_k,
        top_p=settings.top_p,
        sample_count=settings.sample_count,
        clip=settings.clip,
    )
    forecasts = []
    for key, group in causal.sort(["security_id", "event_time"], maintain_order=True).group_by(
        "security_id", maintain_order=True
    ):
        security_id = str(key[0] if isinstance(key, tuple) else key)
        symbol = security_id
        if "symbol" in group.columns and group["symbol"].len() > 0:
            value = group["symbol"][0]
            if value is not None:
                symbol = str(value)
        forecasts.append(
            adapter.forecast_asset(
                _group_to_pandas(group),
                security_id=security_id,
                symbol=symbol,
                asof=decision,
            )
        )
    return MarketState(asof=decision, forecasts=forecasts)


__all__ = ["forecast_kronos_frame"]

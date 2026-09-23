"""Kronos candlestick forecasts as a MarketState (research-only, opt-in).

This module is the thin pipeline glue between the optional
:class:`~quant_fund.models.kronos.KronosAdapter` and the rest of the lab. It is
deliberately offline-safe: without ``train.kronos.enabled`` and local,
pre-downloaded artifacts there is no predictor, and nothing here reaches the
network. Tests and research callers may inject a deterministic predictor
implementing the ``KronosPredictor`` protocol.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.models.kronos import (
    PRICE_COLUMNS,
    KronosAdapter,
    KronosPredictor,
    load_local_predictor,
)
from quant_fund.schemas.forecast import AssetForecast, MarketState

REQUIRED_FRAME_COLUMNS = (
    "security_id",
    "event_time",
    "available_time",
    "open",
    "high",
    "low",
    "close",
)


def _resolve_predictor(config: AppConfig) -> KronosPredictor:
    """Load the upstream predictor from validated local artifacts only."""
    kronos = config.train.kronos
    if not kronos.enabled:
        raise ValueError(
            "Kronos forecasts require train.kronos.enabled=true with local "
            "model_path/tokenizer_path, or an injected predictor"
        )
    if kronos.model_path is None or kronos.tokenizer_path is None:
        raise ValueError("enabled Kronos requires local model_path and tokenizer_path")
    return load_local_predictor(
        model_path=kronos.model_path,
        tokenizer_path=kronos.tokenizer_path,
        model_sha256=kronos.model_sha256,
        tokenizer_sha256=kronos.tokenizer_sha256,
        device=kronos.device,
        max_context=kronos.max_context,
        clip=kronos.clip,
    )


def _resolve_asof(frame: pl.DataFrame, asof: Any) -> datetime:
    if asof is None:
        from quant_fund.pipeline.forecast import latest_decision

        return latest_decision(frame)
    if isinstance(asof, datetime):
        return asof
    timestamp = pd.Timestamp(asof)
    return timestamp.to_pydatetime()


def forecast_kronos_frame(
    config: AppConfig,
    *,
    asof: Any = None,
    frame: pl.DataFrame | None = None,
    predictor: KronosPredictor | None = None,
) -> MarketState:
    """Forecast every security's candle path with Kronos at ``asof``.

    ``frame`` is a silver-style bars panel with the REQUIRED_FRAME_COLUMNS
    (``volume``/``amount`` are optional; the adapter supplies honest defaults).
    When ``frame`` is omitted the lake's ``silver/bars.parquet`` is used. When
    ``predictor`` is omitted the adapter loads strictly local artifacts, which
    requires ``train.kronos.enabled`` — fail-closed by construction.
    """
    kronos = config.train.kronos
    if frame is None:
        from quant_fund.pipeline.dataset import ensure_silver

        frame = ensure_silver(config)
    missing = [name for name in REQUIRED_FRAME_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"Kronos frame is missing required columns: {missing}")
    if frame.is_empty():
        raise ValueError("Kronos frame must not be empty")
    if predictor is None:
        predictor = _resolve_predictor(config)
    adapter = KronosAdapter(
        predictor,
        pred_len=kronos.pred_len,
        lookback=kronos.lookback,
        horizon_name=kronos.horizon_name,
        temperature=kronos.temperature,
        top_k=kronos.top_k,
        top_p=kronos.top_p,
        sample_count=kronos.sample_count,
    )
    decision = _resolve_asof(frame, asof)
    optional = [name for name in ("volume", "amount") if name in frame.columns]
    columns = ["event_time", "available_time", *PRICE_COLUMNS, *optional]
    forecasts: list[AssetForecast] = []
    ordered = frame.sort("security_id", maintain_order=True)
    for key, group in ordered.group_by("security_id", maintain_order=True):
        sid = str(key[0] if isinstance(key, tuple) else key)
        symbol = (
            str(group["symbol"][0]) if "symbol" in group.columns and group["symbol"][0] else sid
        )
        forecasts.append(
            adapter.forecast_asset(
                group.select(columns).to_pandas(),
                security_id=sid,
                symbol=symbol,
                asof=decision,
            )
        )
    notes: list[str] = []
    if config.data.source == "synthetic":
        notes.append("SYNTHETIC")
    return MarketState(asof=decision, forecasts=forecasts, notes=notes)

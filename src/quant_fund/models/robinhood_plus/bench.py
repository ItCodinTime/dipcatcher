"""Optional robinhood+ research bench. Proper scores only — no Sharpe."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.metrics.cross_section import date_ic_series
from quant_fund.models.robinhood_plus.constants import (
    AFFILIATION_DISCLAIMER,
    ENGINE_DISPLAY,
    ENGINE_NAME,
    FAMILY,
    KRONOS_PAPER,
    KRONOS_REPO,
    MODEL_VERSION,
    PRICE_SPACE,
)
from quant_fund.models.robinhood_plus.engine import (
    forecast_robinhood_plus_cross_section,
    kline_columns_present,
)
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent


def bench_robinhood_plus(frame: pl.DataFrame, config: AppConfig) -> dict[str, Any]:
    """Score causal robinhood+ 5d expected returns with date-level IC.

    SYNTHETIC recovery is an engine-correctness diagnostic, not a live claim.
    """
    cfg = config.robinhood_plus
    source = "SYNTHETIC" if config.data.source == "synthetic" else str(config.data.source)
    blob: dict[str, Any] = {
        "family": ENGINE_NAME,
        "display": ENGINE_DISPLAY,
        "model_version": MODEL_VERSION,
        "catalog_family": FAMILY,
        "backend": cfg.backend,
        "decoder": cfg.decoder,
        "price_space": PRICE_SPACE,
        "research_only": True,
        "execution_claim": "research_only",
        "claim": "research_metric_only",
        "data_source": source,
        "kronos_paper": KRONOS_PAPER,
        "kronos_repo": KRONOS_REPO,
        "affiliation_disclaimer": AFFILIATION_DISCLAIMER,
        "n_scored": 0,
        "n_ok": 0,
        "mean_ic": float("nan"),
        "ic_n_dates": 0,
    }
    if not cfg.enabled:
        blob["status"] = "disabled"
        return _seal(blob)
    if not kline_columns_present(frame.columns):
        blob["status"] = "missing_ohlc"
        return _seal(blob)
    label = "future_excess_return_5"
    if label not in frame.columns:
        label = "future_log_return_5"
    if label not in frame.columns:
        blob["status"] = "missing_label"
        return _seal(blob)
    times = frame["event_time"].unique().sort().to_list()
    if len(times) < cfg.lookback + 2:
        blob["status"] = "insufficient_history"
        return _seal(blob)
    step = max(1, len(times) // 8)
    sample_times = [t for t in times[cfg.lookback :: step] if isinstance(t, datetime)][-4:]
    scores: list[float] = []
    realized: list[float] = []
    dates: list[datetime] = []
    n_ok = 0
    for asof in sample_times:
        day = frame.filter(pl.col("event_time") == asof)
        if day.is_empty():
            continue
        forecasts = forecast_robinhood_plus_cross_section(
            frame,
            asof,
            lookback=cfg.lookback,
            pred_len=cfg.pred_len,
            sample_count=cfg.sample_count,
            s1_bits=cfg.s1_bits,
            s2_bits=cfg.s2_bits,
            clip=cfg.clip,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_context=cfg.max_context,
            seed=config.train.random_seed,
            decoder=cfg.decoder,
            security_ids=[str(s) for s in day["security_id"].to_list()],
            quantile_levels=tuple(config.quantiles.levels[:3]) or (0.05, 0.50, 0.95),
            horizons=tuple(int(h) for h in config.horizons.bars),
        )
        for row in day.iter_rows(named=True):
            sid = str(row["security_id"])
            hit = forecasts.get(sid)
            if hit is None or hit.forecast.status != "ok":
                continue
            mu = hit.forecast.expected_returns.get("5d")
            y = row.get(label)
            if mu is None or y is None:
                continue
            try:
                yf = float(y)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(mu) or not np.isfinite(yf):
                continue
            scores.append(float(mu))
            realized.append(yf)
            dates.append(asof)
            n_ok += 1
    blob["n_scored"] = int(len(scores))
    blob["n_ok"] = int(n_ok)
    blob["label"] = label
    if len(scores) >= 8:
        ic = date_ic_series(
            np.asarray(scores, dtype=float),
            np.asarray(realized, dtype=float),
            np.asarray(dates, dtype=object),
            min_names=3,
        )
        blob["mean_ic"] = float(ic.mean_pearson)
        blob["ic_n_dates"] = int(ic.n_dates)
        blob["status"] = "ok" if ic.n_dates else "insufficient_dates"
    else:
        blob["status"] = "insufficient_scored"
    return _seal(blob)


def _seal(blob: dict[str, Any]) -> dict[str, Any]:
    if not family_blob_forbidden_metrics_absent(blob):
        raise AssertionError("robinhood+ bench leaked forbidden research keys")
    return blob

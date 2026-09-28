"""Score forecasts and a placeholder signal map.

Forecast metrics reuse :func:`quant_fund.metrics.scoring.pearson_ic`,
:func:`quant_fund.metrics.scoring.rank_ic`, and, when the sample is large
enough, :func:`quant_fund.metrics.direction.pesaran_timmermann`. Fold
stability reuses :func:`quant_fund.validation.walk_forward.fold_ic_stability`.

The placeholder signal mapping reports turnover and an assumed one-way cost
drag only. It does not report Sharpe, P&L, NAV, or drawdown as research scores.
This module does not submit orders.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, cast

import numpy as np
import polars as pl

from fx1.forecast.config import Fx1HarnessConfig
from fx1.forecast.features import normalize_bar_times
from fx1.forecast.schema import SchemaError, validate_forecast_schema
from fx1.forecast.signals import PLACEHOLDER_NOT_A_STRATEGY, map_signals
from quant_fund.config.models import ValidationConfig
from quant_fund.metrics.direction import pesaran_timmermann
from quant_fund.metrics.returns import turnover
from quant_fund.metrics.scoring import pearson_ic, rank_ic
from quant_fund.validation.walk_forward import (
    assert_no_label_overlap,
    fold_ic_stability,
    session_index,
    walk_forward,
)


def json_ready(value: Any) -> Any:
    """Replace non-finite floats so run metadata is strict JSON."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def forward_simple_returns(bars: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """Simple return from t to t+h. This is a label. Do not pass it to ``predict``."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    frame = normalize_bar_times(bars).sort(["security_id", "event_time"])
    return frame.with_columns(
        (pl.col("close").shift(-horizon).over("security_id") / pl.col("close") - 1.0).alias(
            "realized_return"
        ),
        pl.col("event_time").shift(-horizon).over("security_id").alias("target_time"),
    ).select(["security_id", "event_time", "close", "realized_return", "target_time"])


def _decision_times(forecasts: pl.DataFrame) -> list[datetime]:
    values = (
        forecasts.select("event_time")
        .unique()
        .sort("event_time")
        .get_column("event_time")
        .to_list()
    )
    times: list[datetime] = []
    for value in values:
        if not isinstance(value, datetime):
            raise SchemaError("forecast event_time values must be datetimes")
        times.append(value)
    return times


def _test_folds(
    times: list[datetime], config: Fx1HarnessConfig
) -> tuple[set[datetime], list[Any], dict[str, Any]]:
    wf = config.walk_forward
    folds = walk_forward(
        times,
        ValidationConfig(
            scheme=wf.scheme,
            train_bars=wf.train_bars,
            val_bars=wf.val_bars,
            test_bars=wf.test_bars,
            embargo_bars=wf.embargo_bars,
        ),
        horizon_bars=config.features.horizon_bars,
        embargo_bars=wf.embargo_bars,
        scheme=wf.scheme,
    )
    if not folds:
        need = wf.train_bars + wf.val_bars + wf.test_bars
        raise ValueError(
            f"walk-forward produced no folds from {len(times)} decision times; "
            f"need at least {need} (train+val+test)"
        )
    index = session_index(times)
    test: set[datetime] = set()
    for fold in folds:
        assert_no_label_overlap(fold, config.features.horizon_bars, index)
        test.update(fold.test_times)
    info = {"n_folds": len(folds), "n_test_times": len(test), "scheme": wf.scheme}
    return test, folds, info


def _score_column(frame: pl.DataFrame) -> pl.DataFrame:
    if "predicted_return" in frame.columns and "predicted_price" in frame.columns:
        score = (
            pl.when(pl.col("predicted_return").is_not_null())
            .then(pl.col("predicted_return"))
            .otherwise(pl.col("predicted_price") / pl.col("close") - 1.0)
        )
    elif "predicted_return" in frame.columns:
        score = pl.col("predicted_return")
    else:
        score = pl.col("predicted_price") / pl.col("close") - 1.0
    return frame.with_columns(score.alias("score"))


def _mae_rmse(pred: np.ndarray, realized: np.ndarray) -> tuple[float, float, int]:
    mask = np.isfinite(pred) & np.isfinite(realized)
    n = int(mask.sum())
    if n == 0:
        return float("nan"), float("nan"), 0
    err = pred[mask] - realized[mask]
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2))), n


def _hit_rate(pred: np.ndarray, realized: np.ndarray) -> dict[str, float | None]:
    mask = np.isfinite(pred) & np.isfinite(realized) & (pred != 0.0) & (realized != 0.0)
    n = int(mask.sum())
    if n == 0:
        return {"hit_rate": float("nan"), "n_hit": 0, "pt_stat": float("nan")}
    rate = float(np.mean(np.sign(pred[mask]) == np.sign(realized[mask])))
    pt = float("nan")
    if n >= 20:
        stats = pesaran_timmermann(np.sign(pred[mask]), np.sign(realized[mask]))
        pt = float(stats["pt_stat"])
        rate = float(stats["hit_rate"])
    return {"hit_rate": rate, "n_hit": n, "pt_stat": pt}


def _mean_cross_sectional_ic(frame: pl.DataFrame) -> float:
    ics: list[float] = []
    for group in frame.partition_by("event_time", maintain_order=True):
        if group.height < 3:
            continue
        ics.append(
            pearson_ic(
                group["score"].to_numpy().astype(float),
                group["realized_return"].to_numpy().astype(float),
            )
        )
    finite = [value for value in ics if math.isfinite(value)]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def _horizon_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    pred = frame["score"].to_numpy().astype(float)
    realized = frame["realized_return"].to_numpy().astype(float)
    mae, rmse, n = _mae_rmse(pred, realized)
    return {
        "n": n,
        "ic": pearson_ic(pred, realized),
        "rank_ic": rank_ic(pred, realized),
        "mean_cross_sectional_ic": _mean_cross_sectional_ic(frame),
        **_hit_rate(pred, realized),
        "mae": mae,
        "rmse": rmse,
    }


def _signal_diagnostics(
    frame: pl.DataFrame,
    *,
    cost_bps: float,
    stride: int,
) -> dict[str, Any]:
    times = _decision_times(frame)
    if stride > 1:
        times = times[::stride]
    ids = sorted(str(value) for value in frame["security_id"].unique().to_list())
    index = {sid: i for i, sid in enumerate(ids)}
    prev = np.zeros(len(ids), dtype=float)
    turns: list[float] = []
    for stamp in times:
        day = frame.filter(pl.col("event_time") == stamp)
        weights = np.zeros(len(ids), dtype=float)
        for row in day.iter_rows(named=True):
            slot = index[str(row["security_id"])]
            signal = row["signal"]
            if signal is None:
                continue
            if not math.isfinite(float(signal)):
                continue
            weights[slot] = float(signal)
        gross = float(np.sum(np.abs(weights)))
        if gross > 0.0:
            weights = weights / gross
        traded = turnover(weights, prev)
        turns.append(traded)
        prev = weights
    mean_turnover = float(np.mean(turns)) if turns else float("nan")
    return {
        "mean_turnover": mean_turnover,
        "cost_bps": float(cost_bps),
        "mean_one_way_cost_drag_bps": float(cost_bps * mean_turnover),
        "n_periods": len(times),
        "return_stride_bars": int(stride),
        "research_diagnostic_only": True,
        "placeholder_mapping": PLACEHOLDER_NOT_A_STRATEGY,
        "orders_submitted": False,
        "note": (
            "Turnover and assumed cost of a placeholder score-to-signal map. "
            "No performance or live-trading claim."
        ),
    }


def evaluate_forecasts(
    forecasts: pl.DataFrame,
    bars: pl.DataFrame,
    config: Fx1HarnessConfig,
) -> dict[str, Any]:
    """Join labels after the fact, score test folds, and map a placeholder signal."""
    validate_forecast_schema(forecasts)
    horizon = config.features.horizon_bars
    if forecasts.filter(pl.col("horizon_bars") == horizon).is_empty():
        raise SchemaError(f"forecasts have no rows at configured horizon_bars={horizon}")
    times = _decision_times(forecasts.filter(pl.col("horizon_bars") == horizon))
    test_times, folds, fold_info = _test_folds(times, config)
    by_horizon: dict[str, Any] = {}
    primary: pl.DataFrame | None = None
    fold_ics: list[float] = []
    for horizon_value in sorted(
        int(value) for value in forecasts["horizon_bars"].unique().to_list()
    ):
        block = forecasts.filter(pl.col("horizon_bars") == horizon_value)
        if "close" in block.columns:
            block = block.drop("close")
        joined = _score_column(
            block.join(
                forward_simple_returns(bars, horizon_value),
                on=["security_id", "event_time"],
                how="left",
            )
        )
        scored = joined.filter(
            pl.col("event_time").is_in(list(test_times)) & pl.col("realized_return").is_not_null()
        )
        if scored.is_empty():
            by_horizon[str(horizon_value)] = {"n": 0}
            continue
        by_horizon[str(horizon_value)] = _horizon_metrics(scored)
        if horizon_value == horizon:
            primary = scored
            for fold in folds:
                group = scored.filter(pl.col("event_time").is_in(fold.test_times))
                if group.height < 3:
                    continue
                fold_ics.append(
                    pearson_ic(
                        group["score"].to_numpy().astype(float),
                        group["realized_return"].to_numpy().astype(float),
                    )
                )
    if primary is None:
        raise SchemaError("configured horizon has no realized test-fold outcomes")
    primary = _score_column(primary).filter(pl.col("event_time").is_in(list(test_times)))
    primary = primary.filter(
        pl.col("score").is_not_null() & pl.col("realized_return").is_not_null()
    )
    mapped = map_signals(primary, config.signal.mapping, threshold=config.signal.threshold)
    if mapped.filter(pl.col("mapping_role") != PLACEHOLDER_NOT_A_STRATEGY).height:
        raise SchemaError("signal mapping lost its placeholder marker")
    diagnostics = _signal_diagnostics(
        mapped,
        cost_bps=config.signal.cost_bps,
        stride=horizon,
    )
    return cast(
        dict[str, Any],
        json_ready(
            {
                "schema": "fx1.harness.eval/v1",
                "research_only": True,
                "promotion_eligible": False,
                "live_pnl_claim": False,
                "placeholder_signal_mapping": True,
                "signal_mapping": config.signal.mapping,
                "mapping_role": PLACEHOLDER_NOT_A_STRATEGY,
                "orders_submitted": False,
                "folds": {**fold_info, "ic_stability": fold_ic_stability(fold_ics)},
                "forecast_metrics": {"by_horizon": by_horizon},
                "signal_diagnostics": diagnostics,
            }
        ),
    )

"""Strict input and output schemas for the forecasting harness.

Feature frames are an allow-list: identifiers, the decision close, availability,
and the declared numeric features. Label-like columns are rejected so a target
cannot ride along into ``predict``.
"""

from __future__ import annotations

import re

import polars as pl

from quant_fund.schemas.errors import LeakageError

FORECAST_SCHEMA = "fx1.harness.forecast/v1"

_CONTEXT_COLUMNS = frozenset({"event_time", "security_id", "close", "available_time"})
_FORBIDDEN_EXACT = frozenset(
    {
        "target",
        "target_return",
        "forward_return",
        "future_close",
        "future_return",
        "label",
        "y",
        "realized_return",
        "next_open",
        "next_close",
        "planted_signal",
    }
)
_FORBIDDEN_PREFIXES = ("target_", "future_", "fwd_", "label_")
_QUANTILE_COLUMN = re.compile(r"^q_(0\.\d+)$")


class SchemaError(ValueError):
    """A feature or forecast frame violated the harness contract."""


def forbidden_column(name: str) -> bool:
    lowered = name.lower()
    return lowered in _FORBIDDEN_EXACT or lowered.startswith(_FORBIDDEN_PREFIXES)


def validate_feature_schema(frame: pl.DataFrame, feature_columns: list[str]) -> None:
    """Reject missing keys, lookahead columns, and undeclared extras."""
    if frame.is_empty():
        raise SchemaError("feature frame is empty")
    missing = [
        name
        for name in ("event_time", "security_id", *feature_columns)
        if name not in frame.columns
    ]
    if missing:
        raise SchemaError(f"feature frame missing columns: {missing}")
    leaked = [name for name in frame.columns if forbidden_column(name)]
    if leaked:
        raise LeakageError(f"feature frame contains lookahead/label columns: {sorted(leaked)}")
    allowed = _CONTEXT_COLUMNS | set(feature_columns)
    extra = sorted(set(frame.columns) - allowed)
    if extra:
        raise SchemaError(f"feature frame has columns outside the schema allow-list: {extra}")
    dup = frame.select(["event_time", "security_id"]).is_duplicated().any()
    if dup:
        raise SchemaError("feature frame has duplicate (event_time, security_id) rows")
    for name in feature_columns:
        dtype = frame.schema[name]
        if dtype not in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
            raise SchemaError(f"feature {name!r} must be numeric, got {dtype}")
        if (
            frame[name].null_count()
            or frame.filter(pl.col(name).is_not_null() & ~pl.col(name).is_finite()).height
        ):
            raise SchemaError(f"feature {name!r} must be finite")


def _quantile_taus(columns: list[str]) -> list[tuple[str, float]]:
    found: list[tuple[str, float]] = []
    for name in columns:
        match = _QUANTILE_COLUMN.match(name)
        if match is None:
            continue
        tau = float(match.group(1))
        if not 0.0 < tau < 1.0:
            raise SchemaError(f"quantile column {name!r} tau must lie in (0, 1)")
        found.append((name, tau))
    found.sort(key=lambda item: item[1])
    return found


def validate_forecast_schema(frame: pl.DataFrame) -> None:
    """Require the forecast key, a return and/or a price, and optional quantiles."""
    if frame.is_empty():
        raise SchemaError("forecast frame is empty")
    required = ("event_time", "security_id", "horizon_bars")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise SchemaError(f"forecast frame missing columns: {missing}")
    if "predicted_return" not in frame.columns and "predicted_price" not in frame.columns:
        raise SchemaError("forecast frame needs predicted_return and/or predicted_price")
    leaked = [name for name in frame.columns if forbidden_column(name)]
    if leaked:
        raise LeakageError(f"forecast frame contains label columns: {sorted(leaked)}")
    keys = ["event_time", "security_id", "horizon_bars"]
    if frame.select(keys).is_duplicated().any():
        raise SchemaError("forecast frame has duplicate (event_time, security_id, horizon_bars)")
    horizons = frame["horizon_bars"]
    if horizons.null_count() or horizons.dtype not in (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt32,
        pl.UInt64,
    ):
        raise SchemaError("horizon_bars must be a non-null integer")
    if frame.filter(pl.col("horizon_bars") < 1).height:
        raise SchemaError("horizon_bars must be >= 1")
    neither = pl.lit(True)
    if "predicted_return" in frame.columns:
        neither = neither & pl.col("predicted_return").is_null()
        bad_ret = frame.filter(
            pl.col("predicted_return").is_not_null() & ~pl.col("predicted_return").is_finite()
        )
        if bad_ret.height:
            raise SchemaError("predicted_return must be finite when present")
    if "predicted_price" in frame.columns:
        neither = neither & pl.col("predicted_price").is_null()
        bad_px = frame.filter(
            pl.col("predicted_price").is_not_null()
            & (~pl.col("predicted_price").is_finite() | (pl.col("predicted_price") <= 0))
        )
        if bad_px.height:
            raise SchemaError("predicted_price must be finite and positive when present")
    if frame.filter(neither).height:
        raise SchemaError("each forecast row needs predicted_return and/or predicted_price")
    if "confidence" in frame.columns:
        bad_conf = frame.filter(
            pl.col("confidence").is_not_null()
            & (
                ~pl.col("confidence").is_finite()
                | (pl.col("confidence") < 0.0)
                | (pl.col("confidence") > 1.0)
            )
        )
        if bad_conf.height:
            raise SchemaError("confidence must be null or inside [0, 1]")
    quantiles = _quantile_taus(list(frame.columns))
    if len(quantiles) >= 2:
        exprs = [pl.col(name) for name, _tau in quantiles]
        # A row with any null quantile is skipped; present values must be ordered.
        ordered = frame.filter(pl.all_horizontal([expr.is_not_null() for expr in exprs]))
        decreasing = ordered.filter(
            pl.any_horizontal(
                [
                    pl.col(left) > pl.col(right)
                    for (left, _), (right, _) in zip(quantiles, quantiles[1:], strict=False)
                ]
            )
        )
        if decreasing.height:
            raise SchemaError("quantile columns must be nondecreasing in tau on each row")

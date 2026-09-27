"""Reference forecasters for tests and smoke runs.

These are not fx-1. They do not load market-trained weights and must not be
described as the forecasting model.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

from fx1.forecast.artifacts import load_artifact
from fx1.forecast.protocol import ForecastModel
from fx1.forecast.schema import SchemaError, forbidden_column, validate_feature_schema
from quant_fund.schemas.errors import LeakageError


def _guard_features(features: pl.DataFrame) -> None:
    if features.is_empty():
        raise ValueError("reference model received an empty feature frame")
    if "event_time" not in features.columns or "security_id" not in features.columns:
        raise SchemaError("feature frame missing event_time or security_id")
    leaked = [name for name in features.columns if forbidden_column(name)]
    if leaked:
        raise LeakageError(f"reference model refuses lookahead/label columns: {sorted(leaked)}")


def _base_forecast(features: pl.DataFrame, horizon_bars: int, predicted: pl.Series) -> pl.DataFrame:
    n = features.height
    return pl.DataFrame(
        {
            "event_time": features["event_time"],
            "security_id": features["security_id"],
            "horizon_bars": pl.Series([horizon_bars] * n, dtype=pl.Int64),
            "predicted_return": predicted,
            "predicted_price": pl.Series([None] * n, dtype=pl.Float64),
            "confidence": pl.Series([0.0] * n, dtype=pl.Float64),
        }
    )


class _ReferenceModel(ForecastModel):
    """Shared load path. ``role`` marks the object as not fx-1."""

    role = "reference_not_fx1"
    version = "reference-not-fx1"

    def __init__(self) -> None:
        self.horizon_bars = 1
        self.artifact_sha256: str | None = None
        self.artifact_format: str | None = None

    def load(
        self,
        checkpoint_path: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        cfg = dict(config or {})
        if cfg.get("horizon_bars") is not None:
            self.horizon_bars = int(cfg["horizon_bars"])
        if cfg.get("version"):
            self.version = str(cfg["version"])
        if checkpoint_path is None:
            return
        artifact = load_artifact(
            checkpoint_path,
            fmt=cfg.get("checkpoint_format"),
            allow_unsafe_deserialization=bool(cfg.get("allow_unsafe_deserialization", False)),
            trusted_checkpoint_sha256=cfg.get("trusted_checkpoint_sha256"),
        )
        self.artifact_sha256 = artifact.sha256
        self.artifact_format = artifact.format
        if artifact.version:
            self.version = artifact.version
        payload = artifact.payload
        if isinstance(payload, dict) and payload.get("horizon_bars") is not None:
            self.horizon_bars = int(payload["horizon_bars"])
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be >= 1")


class ZeroForecastModel(_ReferenceModel):
    """Predict a zero simple return at every row. A null baseline, not fx-1."""

    name = "dummy-zero"

    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        _guard_features(features)
        zeros = pl.Series("predicted_return", [0.0] * features.height, dtype=pl.Float64)
        return _base_forecast(features, self.horizon_bars, zeros)


class MomentumForecastModel(_ReferenceModel):
    """Predict that the latest trailing return persists. A naive baseline, not fx-1.

    Uses ``ret_1`` when present, otherwise the first ``mom_*`` column. This is
    a placeholder reference so the plumbing can be tested offline.
    """

    name = "dummy-momentum"

    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        _guard_features(features)
        column = "ret_1" if "ret_1" in features.columns else None
        if column is None:
            moms = [name for name in features.columns if name.startswith("mom_")]
            if not moms:
                raise ValueError("dummy-momentum needs ret_1 or a mom_* column")
            column = moms[0]
        # Declared features are whatever numeric trail the caller actually has.
        numeric = [
            name
            for name in features.columns
            if name not in {"event_time", "security_id", "close", "available_time"}
        ]
        validate_feature_schema(features, numeric)
        predicted = features[column].cast(pl.Float64)
        return _base_forecast(features, self.horizon_bars, predicted)

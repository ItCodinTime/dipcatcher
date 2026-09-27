"""Batch and walk-forward inference, then placeholder-signal evaluation.

Walk-forward calls ``predict`` with bars whose ``event_time`` and
``available_time`` are both at or before the decision timestamp. Batch mode
is row-local: feature values are causal, but the frame contains the whole
window, and it refuses bars that were released after their event time.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl
import pyarrow.parquet as pq

from fx1.forecast.artifacts import UntrustedArtifactError, probe_artifact
from fx1.forecast.config import Fx1HarnessConfig
from fx1.forecast.evaluate import evaluate_forecasts, json_ready
from fx1.forecast.features import (
    OhlcvFeaturePipeline,
    as_utc,
    normalize_bar_times,
    resample_ohlcv,
)
from fx1.forecast.protocol import FeaturePipeline, ForecastModel
from fx1.forecast.registry import create_model, load_symbol
from fx1.forecast.schema import SchemaError, validate_feature_schema, validate_forecast_schema
from quant_fund.schemas.errors import LeakageError
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

_META_KEY = b"fx1_harness"


@dataclass(frozen=True)
class InferenceResult:
    forecasts: pl.DataFrame
    parquet_path: Path
    meta_path: Path
    n_rows: int
    metadata: dict[str, Any]


def parse_bound(value: str | None) -> datetime | None:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    return as_utc(parsed)


def resolve_provider(config: Fx1HarnessConfig) -> Any:
    """Return an object with ``get_bars``. Synthetic and parquet use existing adapters."""
    section = config.data
    if section.provider == "synthetic":
        from quant_fund.data.adapters.synthetic import SyntheticMarketProvider

        return SyntheticMarketProvider(
            n_assets=section.synthetic_n_assets,
            n_days=section.synthetic_n_days,
            seed=section.synthetic_seed,
        )
    if section.provider == "parquet":
        from quant_fund.data.adapters.parquet import ParquetMarketProvider

        assert section.root is not None
        return ParquetMarketProvider(section.root)
    if not section.entrypoint:
        raise ValueError("data.entrypoint is required")
    obj = load_symbol(section.entrypoint)
    if isinstance(obj, type):
        obj = obj()
    if not hasattr(obj, "get_bars"):
        raise TypeError("data.entrypoint must provide get_bars(start, end, security_ids)")
    return obj


def resolve_pipeline(config: Fx1HarnessConfig) -> FeaturePipeline:
    if config.features.entrypoint:
        obj = load_symbol(config.features.entrypoint)
        if isinstance(obj, type):
            obj = obj()
        if not hasattr(obj, "build") or not hasattr(obj, "feature_columns"):
            raise TypeError("features.entrypoint must provide build() and feature_columns()")
        return cast(FeaturePipeline, obj)
    return OhlcvFeaturePipeline(list(config.features.lookbacks), config.features.vol_window)


def load_bars(provider: Any, config: Fx1HarnessConfig) -> pl.DataFrame:
    """Pull bars through the adapter. ``data.start`` filters decisions, not history."""
    symbols = list(config.data.symbols) or None
    end = parse_bound(config.data.end)
    bars = provider.get_bars(start=None, end=end, security_ids=symbols)
    if not isinstance(bars, pl.DataFrame):
        raise TypeError("get_bars must return a polars DataFrame")
    if bars.is_empty():
        raise ValueError("provider returned no bars")
    if config.data.resample:
        bars = resample_ohlcv(bars, config.data.resample)
    else:
        bars = normalize_bar_times(bars)
    if symbols:
        bars = bars.filter(pl.col("security_id").is_in(symbols))
    if end is not None:
        bars = bars.filter(pl.col("event_time") <= end)
    if bars.is_empty():
        raise ValueError("no bars remain after symbol and end filters")
    return bars


def data_label(bars: pl.DataFrame, provider_name: str) -> str:
    if "source" in bars.columns:
        values = sorted({str(value).lower() for value in bars["source"].unique().to_list()})
        if values == ["synthetic"]:
            return "SYNTHETIC"
        if len(values) == 1:
            return str(bars["source"].unique().to_list()[0])
        if values:
            return "mixed"
    if provider_name == "synthetic":
        return "SYNTHETIC"
    return provider_name


def _decision_times(bars: pl.DataFrame, config: Fx1HarnessConfig) -> list[datetime]:
    start = parse_bound(config.data.start)
    end = parse_bound(config.data.end)
    stamps = (
        bars.select("event_time").unique().sort("event_time").get_column("event_time").to_list()
    )
    chosen: list[datetime] = []
    for stamp in stamps:
        if not isinstance(stamp, datetime):
            raise SchemaError("bar event_time values must be datetimes")
        if start is not None and stamp < start:
            continue
        if end is not None and stamp > end:
            continue
        chosen.append(stamp)
    if not chosen:
        raise ValueError("no decision timestamps inside data.start/data.end")
    return chosen


def _has_late_release(bars: pl.DataFrame) -> bool:
    return bars.filter(pl.col("available_time") > pl.col("event_time")).height > 0


def _coerce_frame(frame: object) -> pl.DataFrame:
    if isinstance(frame, pl.DataFrame):
        return normalize_bar_times(frame) if "event_time" in frame.columns else frame
    module = type(frame).__module__
    if module.startswith("pandas"):
        return normalize_bar_times(pl.from_pandas(frame))
    raise TypeError("predict() must return a polars or pandas DataFrame")


def _at_decision(
    pred: pl.DataFrame, features: pl.DataFrame, decision_time: datetime
) -> pl.DataFrame:
    pred = _coerce_frame(pred)
    validate_forecast_schema(pred)
    cutoff = as_utc(decision_time)
    if pred.filter(pl.col("event_time") > cutoff).height:
        raise LeakageError("forecast event_time is after the walk-forward decision cutoff")
    part = pred.filter(pl.col("event_time") == cutoff)
    expected = set(features.filter(pl.col("event_time") == cutoff)["security_id"].to_list())
    got = set(part["security_id"].to_list())
    if not expected:
        return part
    if got != expected:
        raise SchemaError(
            "forecast securities do not match the visible cross-section at the decision time"
        )
    return part


def _match_panel(pred: pl.DataFrame, features: pl.DataFrame) -> pl.DataFrame:
    pred = _coerce_frame(pred)
    validate_forecast_schema(pred)
    if pred.filter(pl.col("event_time") > features["event_time"].max()).height:
        raise LeakageError("forecast event_time is after the latest feature row")
    expected = set(
        zip(features["event_time"].to_list(), features["security_id"].to_list(), strict=True)
    )
    got = set(zip(pred["event_time"].to_list(), pred["security_id"].to_list(), strict=True))
    if got != expected:
        raise SchemaError("batch forecast keys do not match the feature panel")
    return pred


def _predict_walk_forward(
    bars: pl.DataFrame,
    pipeline: FeaturePipeline,
    model: ForecastModel,
    decisions: list[datetime],
    config: Fx1HarnessConfig,
) -> pl.DataFrame:
    columns = pipeline.feature_columns()
    parts: list[pl.DataFrame] = []
    if _has_late_release(bars):
        slices: list[tuple[datetime, pl.DataFrame]] = []
        for stamp in decisions:
            feats = pipeline.build(bars, decision_time=stamp)
            if feats.is_empty() or feats.filter(pl.col("event_time") == stamp).is_empty():
                continue
            slices.append((stamp, feats))
    else:
        built = pipeline.build(bars, decision_time=decisions[-1])
        slices = []
        for stamp in decisions:
            feats = built.filter(pl.col("event_time") <= stamp)
            if feats.filter(pl.col("event_time") == stamp).is_empty():
                continue
            slices.append((stamp, feats))
    for stamp, feats in slices:
        validate_feature_schema(feats, columns)
        latest = feats["event_time"].max()
        if not isinstance(latest, datetime) or latest > stamp:
            raise LeakageError("feature frame extends past the decision time")
        part = _at_decision(model.predict(feats), feats, stamp)
        if not part.is_empty():
            parts.append(part)
    if not parts:
        raise ValueError("walk-forward inference produced no forecasts")
    return pl.concat(parts, how="vertical")


def _predict_batch(
    bars: pl.DataFrame,
    pipeline: FeaturePipeline,
    model: ForecastModel,
    config: Fx1HarnessConfig,
) -> pl.DataFrame:
    if _has_late_release(bars):
        raise LeakageError(
            "batch mode refuses bars released after their event_time; use inference.mode=walk_forward"
        )
    end = parse_bound(config.data.end)
    feats = pipeline.build(bars, decision_time=end)
    start = parse_bound(config.data.start)
    if start is not None:
        feats = feats.filter(pl.col("event_time") >= start)
    if feats.is_empty():
        raise ValueError("batch inference produced an empty feature frame")
    validate_feature_schema(feats, pipeline.feature_columns())
    return _match_panel(model.predict(feats), feats)


def _write_outputs(
    forecasts: pl.DataFrame, metadata: dict[str, Any], config: Fx1HarnessConfig
) -> tuple[Path, Path]:
    parquet_path = Path(config.inference.output_parquet)
    meta_path = Path(config.inference.output_meta)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_ready(metadata)
    table = forecasts.to_arrow()
    existing = dict(table.schema.metadata or {})
    existing[_META_KEY] = json.dumps(payload, sort_keys=True).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(existing), parquet_path)
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return parquet_path, meta_path


def run_inference(
    config: Fx1HarnessConfig,
    *,
    model: ForecastModel | None = None,
    provider: Any = None,
) -> InferenceResult:
    """Load a forecaster, run it over the configured window, and write parquet."""
    provider = provider if provider is not None else resolve_provider(config)
    bars = load_bars(provider, config)
    pipeline = resolve_pipeline(config)
    checkpoint = config.model.checkpoint_path
    stamp = (
        probe_artifact(checkpoint, config.model.checkpoint_format)
        if checkpoint is not None
        else None
    )
    if stamp is not None:
        expected = config.model.trusted_checkpoint_sha256
        if expected is not None and stamp["sha256"] != expected:
            raise UntrustedArtifactError("checkpoint SHA-256 does not match the trusted digest")
        if (
            stamp["format"] in {"pickle", "joblib"}
            and not config.model.allow_unsafe_deserialization
        ):
            raise UntrustedArtifactError(
                "pickle and joblib require allow_unsafe_deserialization=true "
                "and trusted_checkpoint_sha256"
            )
    bound = (
        model
        if model is not None
        else create_model(config.model.name, entrypoint=config.model.entrypoint)
    )
    bound.load(
        checkpoint,
        {
            "horizon_bars": config.features.horizon_bars,
            "version": config.model.version,
            "checkpoint_format": config.model.checkpoint_format,
            "trusted_checkpoint_sha256": config.model.trusted_checkpoint_sha256,
            "allow_unsafe_deserialization": config.model.allow_unsafe_deserialization,
            "name": config.model.name,
        },
    )
    if checkpoint is not None and stamp is not None:
        # An external loader can receive a path instead of the helper above.
        # Refuse to stamp a checkpoint that changed while it was loaded.
        after = probe_artifact(checkpoint, config.model.checkpoint_format)
        if after["sha256"] != stamp["sha256"]:
            raise UntrustedArtifactError("checkpoint changed while the model was loading")
    decisions = _decision_times(bars, config)
    if config.inference.mode == "walk_forward":
        forecasts = _predict_walk_forward(bars, pipeline, bound, decisions, config)
    else:
        forecasts = _predict_batch(bars, pipeline, bound, config)
    validate_forecast_schema(forecasts)
    role = getattr(bound, "role", None)
    model_role = role if isinstance(role, str) else "external"
    version = getattr(bound, "version", None)
    metadata: dict[str, Any] = {
        "schema": "fx1.harness.forecasts/v1",
        "run_id": uuid.uuid4().hex,
        "created_at": datetime.now(UTC).isoformat(),
        "model_name": getattr(bound, "name", config.model.name),
        "model_role": model_role,
        "model_version": version if isinstance(version, str) else None,
        "artifact_sha256": None if stamp is None else stamp["sha256"],
        "artifact_format": None if stamp is None else stamp["format"],
        "artifact_version": None if stamp is None else stamp["version"],
        "feature_columns": pipeline.feature_columns(),
        "horizon_bars": config.features.horizon_bars,
        "inference_mode": config.inference.mode,
        "n_rows": forecasts.height,
        "data_label": data_label(bars, config.data.provider),
        "research_only": True,
        "live_pnl_claim": False,
        "config_sha256": hash_bytes(
            canonical_json_bytes(config.model_dump(mode="json", by_alias=True))
        ),
    }
    artifact_sha = getattr(bound, "artifact_sha256", None)
    if isinstance(artifact_sha, str):
        metadata["artifact_sha256"] = artifact_sha
    model_version = getattr(bound, "version", None)
    if isinstance(model_version, str):
        metadata["model_version"] = model_version
    parquet_path, meta_path = _write_outputs(forecasts, metadata, config)
    return InferenceResult(
        forecasts=forecasts,
        parquet_path=parquet_path,
        meta_path=meta_path,
        n_rows=forecasts.height,
        metadata=json_ready(metadata),
    )


def run_signal_evaluation(
    config: Fx1HarnessConfig,
    *,
    forecasts: pl.DataFrame | None = None,
    provider: Any = None,
) -> dict[str, Any]:
    """Score an existing forecast parquet. Does not place orders."""
    if forecasts is None:
        path = Path(config.inference.output_parquet)
        if not path.is_file():
            raise FileNotFoundError(f"no forecasts at {path}; run `fx1 infer` first")
        forecasts = pl.read_parquet(path)
    provider = provider if provider is not None else resolve_provider(config)
    bars = load_bars(provider, config)
    report = evaluate_forecasts(forecasts, bars, config)
    report["data_label"] = data_label(bars, config.data.provider)
    report["model_name"] = config.model.name
    return report

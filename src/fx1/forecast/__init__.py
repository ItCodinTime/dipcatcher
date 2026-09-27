"""Infrastructure harness for an external price/return forecaster.

``fx-1`` the forecasting model is not implemented here. This package is the
plumbing: a typed model contract, a name-keyed registry, a format-agnostic
checkpoint loader, point-in-time feature hooks over the dipcatcher data
adapters, and inference / placeholder-signal evaluation runners.

Reference forecasters in :mod:`fx1.forecast.dummy` exist for tests and smoke
runs. They are not fx-1.
"""

from fx1.forecast.artifacts import (
    ArtifactBackendUnavailable,
    LoadedArtifact,
    UntrustedArtifactError,
    load_artifact,
    probe_artifact,
)
from fx1.forecast.config import Fx1HarnessConfig, load_harness_config
from fx1.forecast.dummy import MomentumForecastModel, ZeroForecastModel
from fx1.forecast.features import OhlcvFeaturePipeline, resample_ohlcv, visible_bars
from fx1.forecast.protocol import FeaturePipeline, ForecastModel, Fx1Model
from fx1.forecast.registry import ModelNotRegistered, create_model
from fx1.forecast.schema import FORECAST_SCHEMA, SchemaError, validate_forecast_schema

__all__ = [
    "FORECAST_SCHEMA",
    "ArtifactBackendUnavailable",
    "FeaturePipeline",
    "ForecastModel",
    "Fx1HarnessConfig",
    "Fx1Model",
    "LoadedArtifact",
    "UntrustedArtifactError",
    "ModelNotRegistered",
    "MomentumForecastModel",
    "OhlcvFeaturePipeline",
    "SchemaError",
    "ZeroForecastModel",
    "create_model",
    "load_artifact",
    "load_harness_config",
    "probe_artifact",
    "resample_ohlcv",
    "validate_forecast_schema",
    "visible_bars",
]

"""YAML/JSON config for inference and placeholder-signal evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataSection(_Strict):
    """Which adapter supplies bars. ``entrypoint`` is a generic ``get_bars`` object."""

    provider: Literal["synthetic", "parquet", "entrypoint"] = "parquet"
    root: Path | None = None
    entrypoint: str | None = None
    symbols: list[str] = Field(default_factory=list)
    start: str | None = None
    end: str | None = None
    resample: str | None = None
    synthetic_n_assets: int = 4
    synthetic_n_days: int = 80
    synthetic_seed: int = 7

    @model_validator(mode="after")
    def _provider_paths(self) -> DataSection:
        if self.provider == "parquet" and self.root is None:
            raise ValueError("data.root is required when provider is parquet")
        if self.provider == "entrypoint" and not self.entrypoint:
            raise ValueError("data.entrypoint is required when provider is entrypoint")
        if self.synthetic_n_assets < 2 or self.synthetic_n_days < 16:
            raise ValueError("synthetic panel must have n_assets >= 2 and n_days >= 16")
        return self


class FeatureSection(_Strict):
    lookbacks: list[int] = Field(default_factory=lambda: [1, 5, 20])
    vol_window: int = 20
    horizon_bars: int = 1
    entrypoint: str | None = None

    @model_validator(mode="after")
    def _windows(self) -> FeatureSection:
        if not self.lookbacks or any(int(k) < 1 for k in self.lookbacks):
            raise ValueError("features.lookbacks must be positive")
        if self.vol_window < 2:
            raise ValueError("features.vol_window must be >= 2")
        if self.horizon_bars < 1:
            raise ValueError("features.horizon_bars must be >= 1")
        return self


class ModelSection(_Strict):
    name: str = "dummy-zero"
    entrypoint: str | None = None
    checkpoint_path: Path | None = None
    checkpoint_format: Literal["auto", "pickle", "joblib", "torch", "onnx", "json"] = "auto"
    trusted_checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    allow_unsafe_deserialization: bool = False
    version: str | None = None

    @model_validator(mode="after")
    def _unsafe_checkpoint(self) -> ModelSection:
        if self.allow_unsafe_deserialization and self.trusted_checkpoint_sha256 is None:
            raise ValueError("allow_unsafe_deserialization requires trusted_checkpoint_sha256")
        return self


class InferenceSection(_Strict):
    mode: Literal["batch", "walk_forward"] = "walk_forward"
    output_parquet: Path = Path("data/fx1/forecasts.parquet")
    output_meta: Path = Path("data/fx1/forecasts.meta.json")


class SignalSection(_Strict):
    """Placeholder mapping and a flat one-way cost in basis points times turnover."""

    mapping: Literal["sign", "rank", "threshold"] = "sign"
    threshold: float = 0.0
    cost_bps: float = 0.0

    @model_validator(mode="after")
    def _costs(self) -> SignalSection:
        if self.threshold < 0 or self.cost_bps < 0:
            raise ValueError("threshold and cost_bps must be >= 0")
        return self


class WalkForwardSection(_Strict):
    scheme: Literal["expanding", "rolling"] = "expanding"
    train_bars: int = 20
    val_bars: int = 5
    test_bars: int = 5
    embargo_bars: int = 1

    @model_validator(mode="after")
    def _bars(self) -> WalkForwardSection:
        for name in ("train_bars", "val_bars", "test_bars"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"walk_forward.{name} must be positive")
        if self.embargo_bars < 0:
            raise ValueError("walk_forward.embargo_bars must be >= 0")
        return self


class Fx1HarnessConfig(_Strict):
    """Top-level harness config. The model name ``fx-1`` still needs an entrypoint."""

    schema_version: str = Field(default="fx1.harness.config/v1", alias="schema")
    data: DataSection
    features: FeatureSection = Field(default_factory=FeatureSection)
    model: ModelSection = Field(default_factory=ModelSection)
    inference: InferenceSection = Field(default_factory=InferenceSection)
    signal: SignalSection = Field(default_factory=SignalSection)
    walk_forward: WalkForwardSection = Field(default_factory=WalkForwardSection)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


def load_harness_config(path: Path) -> Fx1HarnessConfig:
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError("harness config must be a mapping")
    return Fx1HarnessConfig.model_validate(payload)

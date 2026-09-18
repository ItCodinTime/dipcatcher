"""Standard forecast schema. Every engine writes model_version."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

IntervalMethod = Literal["mondrian_cqr", "split_cqr"]


class AssetForecast(BaseModel):
    security_id: str
    symbol: str
    asof: datetime
    model_version: str
    expected_returns: dict[str, float] = Field(default_factory=dict)
    quantiles: dict[str, dict[float, float]] = Field(default_factory=dict)
    probability_positive: dict[str, float] = Field(default_factory=dict)
    alpha: dict[str, float] = Field(default_factory=dict)
    rank_score: dict[str, float] = Field(default_factory=dict)
    rank_percentile: dict[str, float] = Field(default_factory=dict)
    volatility: dict[str, float] = Field(default_factory=dict)
    regime_probabilities: dict[str, float] = Field(default_factory=dict)
    var: dict[str, float] = Field(default_factory=dict)
    expected_shortfall: dict[str, float] = Field(default_factory=dict)
    drawdown_probabilities: dict[str, float] = Field(default_factory=dict)
    expected_spread_bps: float | None = None
    expected_impact_bps: dict[str, float] | None = None
    confidence: dict[str, float] = Field(default_factory=dict)
    diagnostics: dict[str, float | str] = Field(default_factory=dict)
    aleatoric: dict[str, float] = Field(default_factory=dict)
    epistemic: dict[str, float] = Field(default_factory=dict)
    # Conformal prediction sets (optional; raw quantile PIT/CRPS unchanged).
    interval_lo: dict[str, float] = Field(default_factory=dict)
    interval_hi: dict[str, float] = Field(default_factory=dict)
    interval_alpha: float | None = None
    interval_method: IntervalMethod | None = None

    @field_validator("interval_alpha")
    @classmethod
    def validate_interval_alpha(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or not 0.0 < value < 1.0):
            raise ValueError("interval_alpha must be finite and in (0, 1)")
        return value

    @field_validator("probability_positive", "regime_probabilities", "drawdown_probabilities")
    @classmethod
    def validate_probability_map(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in value.values()):
            raise ValueError("probability values must be finite and in [0, 1]")
        return value

    @field_validator(
        "expected_returns",
        "alpha",
        "rank_score",
        "rank_percentile",
        "volatility",
        "var",
        "expected_shortfall",
        "confidence",
        "aleatoric",
        "epistemic",
    )
    @classmethod
    def validate_finite_numeric_map(cls, value: dict[str, float]) -> dict[str, float]:
        # NaN/inf here flow into the optimizer as garbage or trigger opaque
        # solver failures; reject at the schema boundary instead.
        if any(not math.isfinite(v) for v in value.values()):
            raise ValueError("numeric forecast values must be finite")
        return value

    @field_validator("expected_impact_bps")
    @classmethod
    def validate_finite_impact_map(cls, value: dict[str, float] | None) -> dict[str, float] | None:
        if value is not None and any(not math.isfinite(v) for v in value.values()):
            raise ValueError("expected_impact_bps values must be finite")
        return value

    @field_validator("quantiles")
    @classmethod
    def validate_quantiles(
        cls, value: dict[str, dict[float, float]]
    ) -> dict[str, dict[float, float]]:
        for levels in value.values():
            if any(
                not math.isfinite(float(level))
                or not 0.0 < float(level) < 1.0
                or not math.isfinite(float(quantile))
                for level, quantile in levels.items()
            ):
                raise ValueError(
                    "quantile levels and values must be finite; levels must be in (0, 1)"
                )
        return value

    @model_validator(mode="after")
    def validate_intervals(self) -> AssetForecast:
        keys = set(self.interval_lo) | set(self.interval_hi)
        for key in keys:
            if key not in self.interval_lo or key not in self.interval_hi:
                raise ValueError("interval_lo and interval_hi must have matching keys")
            lo = self.interval_lo[key]
            hi = self.interval_hi[key]
            if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
                raise ValueError("interval endpoints must be finite and ordered")
        if keys and self.interval_alpha is None:
            raise ValueError("interval_alpha is required when intervals are supplied")
        return self


class MarketState(BaseModel):
    asof: datetime
    forecasts: list[AssetForecast]
    regime_probabilities: dict[str, float] = Field(default_factory=dict)
    covariance_condition_number: float | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("covariance_condition_number")
    @classmethod
    def validate_covariance_condition_number(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ValueError("covariance_condition_number must be finite and strictly positive")
        return value

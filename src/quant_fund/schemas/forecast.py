"""Standard forecast schema. Every engine writes model_version."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

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


class MarketState(BaseModel):
    asof: datetime
    forecasts: list[AssetForecast]
    regime_probabilities: dict[str, float] = Field(default_factory=dict)
    covariance_condition_number: float | None = None
    notes: list[str] = Field(default_factory=list)

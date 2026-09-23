"""Portfolio and optimization schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PortfolioSnapshot(BaseModel):
    timestamp: datetime
    security_id: str
    symbol: str
    current_weight: float
    target_weight: float
    delta: float
    alpha: float | None = None
    risk_contribution: float | None = None
    marginal_risk: float | None = None
    expected_cost: float | None = None
    sector: str | None = None
    factor_exposures: dict[str, float] = Field(default_factory=dict)
    liquidity: float | None = None
    confidence: float | None = None
    reason_code: str | None = None
    forecast_attribution: dict[str, Any] = Field(default_factory=dict)


class OptimizationDiagnostics(BaseModel):
    status: str
    feasible: bool
    objective: float | None = None
    expected_alpha: float | None = None
    predicted_volatility: float | None = None
    predicted_cost: float | None = None
    turnover: float | None = None
    gross: float | None = None
    net: float | None = None
    sector_exposure: dict[str, float] = Field(default_factory=dict)
    factor_exposure: dict[str, float] = Field(default_factory=dict)
    constraint_slack: dict[str, float] = Field(default_factory=dict)
    duals: dict[str, float] = Field(default_factory=dict)
    message: str | None = None
    conflicting_constraints: list[str] = Field(default_factory=list)

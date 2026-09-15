"""Typed configuration. No magic numbers in model code."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class RuntimeMode(str, Enum):
    RESEARCH = "research"
    BACKTEST = "backtest"
    PAPER = "paper"
    SHADOW = "shadow"
    LIVE = "live"


class FillConvention(str, Enum):
    NEXT_OPEN = "next_open"
    CLOSE_AUCTION = "close_auction"


class MissingPolicy(str, Enum):
    TRAIN_MEDIAN = "train_median"
    INDICATOR = "indicator"
    NATIVE = "native"
    DROP_FEATURE = "drop_feature"
    DROP_ROW = "drop_row"


class RuntimeConfig(BaseModel):
    mode: RuntimeMode = RuntimeMode.RESEARCH
    allow_live: bool = False
    experiment_id: str | None = None
    output_dir: Path = Path("data/metadata")

    @model_validator(mode="after")
    def live_must_be_explicit(self) -> RuntimeConfig:
        if self.mode is RuntimeMode.LIVE and not self.allow_live:
            msg = "Live mode requires runtime.allow_live: true"
            raise ValueError(msg)
        return self


class UniverseConfig(BaseModel):
    min_price: float = 5.0
    min_adv: float = 1_000_000.0
    min_history_bars: int = 252
    exchanges: list[str] = Field(default_factory=lambda: ["XNYS", "XNAS"])
    security_types: list[str] = Field(default_factory=lambda: ["common_stock"])
    top_n_adv: int | None = 50
    include_delisted: bool = True


class CalendarConfig(BaseModel):
    name: str = "XNYS"
    weekend: tuple[int, int] = (5, 6)


class HorizonConfig(BaseModel):
    bars: list[int] = Field(default_factory=lambda: [1, 5, 20])
    names: list[str] = Field(default_factory=lambda: ["1d", "5d", "20d"])

    @model_validator(mode="after")
    def matching_lengths(self) -> HorizonConfig:
        if len(self.bars) != len(self.names):
            raise ValueError("horizons.bars and horizons.names must have equal length")
        return self


class QuantileConfig(BaseModel):
    levels: list[float] = Field(
        default_factory=lambda: [0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99]
    )

    @field_validator("levels")
    @classmethod
    def sorted_open_unit(cls, v: list[float]) -> list[float]:
        if any(x <= 0 or x >= 1 for x in v):
            raise ValueError("quantiles must be in (0, 1)")
        if v != sorted(v):
            raise ValueError("quantiles must be strictly increasing")
        return v


class CostConfig(BaseModel):
    commission_bps: float = 1.0
    half_spread_bps: float = 5.0
    impact_y: float = 0.1
    bps_per_turnover: float = 0.0
    borrow_bps_per_year: float = 50.0
    financing_bps_per_year: float = 0.0
    frictionless: bool = False
    participation_limit: float = 0.1


class ExecutionConfig(BaseModel):
    fill: FillConvention = FillConvention.NEXT_OPEN
    allow_close_auction: bool = False
    horizon_bars: int = 5
    n_slices: int = 5
    risk_aversion: float = 1e-6
    temporary_impact: float = 1e-6
    permanent_impact: float = 1e-7
    participation_rate: float = 0.05


class PortfolioConstraints(BaseModel):
    gross_leverage: float = 2.0
    net_exposure: float = 0.1
    name_max: float = 0.02
    name_min: float = -0.02
    long_max: float = 0.02
    short_max: float = 0.02
    turnover_limit: float = 0.5
    cash_buffer: float = 0.02
    sector_abs_max: float = 0.15
    factor_abs_max: float = 0.1
    beta_abs_max: float = 0.1
    max_positions: int | None = None
    cvar_limit: float | None = None
    cvar_alpha: float = 0.95
    predicted_vol_max: float | None = None


class OptimizerConfig(BaseModel):
    lambda_risk: float = 10.0
    lambda_tc: float = 1.0
    lambda_tail: float = 0.0
    solver: str = "CLARABEL"
    mode: str = "mean_variance"
    risk_aversion: float = 1.0


class ValidationConfig(BaseModel):
    scheme: str = "expanding"
    train_bars: int = 252
    val_bars: int = 63
    test_bars: int = 63
    embargo_bars: int | None = None
    n_groups: int = 6
    test_groups: int = 2
    n_optuna_trials: int = 20
    seed: int = 42


class FeatureConfig(BaseModel):
    winsor_p: float = 0.01
    ewma_lambda: float = 0.94
    amihud_lookback: int = 20
    adv_lookback: int = 20
    families: list[str] = Field(
        default_factory=lambda: [
            "returns",
            "momentum",
            "reversal",
            "volatility",
            "liquidity",
            "cross_sectional",
            "market",
        ]
    )


class MissingConfig(BaseModel):
    policy: MissingPolicy = MissingPolicy.NATIVE


class DataConfig(BaseModel):
    root: Path = Path("data")
    source: str = "synthetic"
    parquet_path: Path | None = None
    synthetic_n_assets: int = 12
    synthetic_n_days: int = 400
    synthetic_seed: int = 7
    synthetic_oracle_beta: float = 0.015
    synthetic_oracle_phi: float = 0.70
    benchmark_id: str = "SEC_MKT"


class TrainConfig(BaseModel):
    ranking_target: str = "future_excess_return_5"
    distribution_target: str = "future_log_return_5"
    volatility_target: str = "future_realized_var_5"
    random_seed: int = 42
    xgb_n_estimators: int = 80
    xgb_max_depth: int = 3
    lgbm_n_estimators: int = 80
    lgbm_num_leaves: int = 15
    ridge_alpha: float = 1.0
    elasticnet_l1: float = 0.5
    n_hmm_states: int = 3
    garch_p: int = 1
    garch_q: int = 1
    har_log: bool = True
    qlike_floor: float = 1e-12
    psd_eigen_tol: float = 1e-10


class RiskGateConfig(BaseModel):
    max_order_notional: float = 1_000_000.0
    max_gross: float = 2.0
    max_net: float = 0.2
    max_name: float = 0.03
    max_participation: float = 0.1
    max_predicted_vol: float = 0.4
    stale_price_bars: int = 3
    stale_model_hours: float = 24.0


class PromotionConfig(BaseModel):
    min_mean_ic: float = 0.0
    min_cost_adjusted_spread: float = 0.0
    max_turnover: float = 5.0
    require_leakage_pass: bool = True


class FusionConfig(BaseModel):
    alpha_weight: float = 1.0
    confidence_weight: float = 1.0
    regime_weight: float = 1.0
    risk_weight: float = 1.0
    tail_penalty: float = 1.0
    liquidity_penalty: float = 1.0
    min_executable_alpha: float = 0.0
    alpha_scale: float = 0.005


class MonitoringConfig(BaseModel):
    psi_alert: float = 0.25
    reference_bars: int = 63
    live_bars: int = 21


class KillSwitchConfig(BaseModel):
    state: str = "ENABLED"
    allow_auto_flatten: bool = False


class AppConfig(BaseModel):
    """Fully resolved experiment configuration."""

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    horizons: HorizonConfig = Field(default_factory=HorizonConfig)
    quantiles: QuantileConfig = Field(default_factory=QuantileConfig)
    costs: CostConfig = Field(default_factory=CostConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    constraints: PortfolioConstraints = Field(default_factory=PortfolioConstraints)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    missing: MissingConfig = Field(default_factory=MissingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    risk_gate: RiskGateConfig = Field(default_factory=RiskGateConfig)
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    kill_switch: KillSwitchConfig = Field(default_factory=KillSwitchConfig)
    inherit: str | None = None

    def embargo_bars(self) -> int:
        if self.validation.embargo_bars is not None:
            return self.validation.embargo_bars
        return max(self.horizons.bars)

    def dump(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

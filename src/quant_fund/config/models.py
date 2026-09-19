"""Typed configuration. No magic numbers in model code."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictConfigModel(BaseModel):
    """Reject silently ignored configuration keys at every nesting level."""

    model_config = ConfigDict(extra="forbid")


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


class GarchDist(str, Enum):
    """Allowed innovation distributions for GARCH-family volatility models."""

    NORMAL = "normal"
    T = "t"
    SKEWT = "skewt"


class GarchVolSpec(str, Enum):
    """Allowed variance specifications for GARCH-family volatility models."""

    GARCH = "garch"
    EGARCH = "egarch"
    GJR = "gjr"
    APARCH = "aparch"
    FIGARCH = "figarch"


class RobinhoodPlusBackend(str, Enum):
    """K-line foundation backends. Torch is the optional [nn] extra."""

    NUMPY = "numpy"
    TORCH = "torch"


class RobinhoodPlusDecoder(str, Enum):
    """NumPy decoders for hierarchical K-line tokens."""

    HIERARCHICAL_MARKOV = "hierarchical_markov"
    TRANSFORMER = "transformer"


class RobinhoodPlusVariant(str, Enum):
    """Kronos zoo sizes. Numpy uses the same names for research-scale maps."""

    MINI = "mini"
    SMALL = "small"
    BASE = "base"


class RuntimeConfig(StrictConfigModel):
    mode: RuntimeMode = RuntimeMode.RESEARCH
    allow_live: bool = False
    experiment_id: str | None = None
    output_dir: Path = Path("data/metadata")

    @model_validator(mode="after")
    def live_must_be_explicit(self) -> RuntimeConfig:
        if self.mode is RuntimeMode.LIVE and not self.allow_live:
            msg = "Live mode requires runtime.allow_live: true"
            raise ValueError(msg)
        if self.allow_live:
            if self.mode is not RuntimeMode.LIVE:
                raise ValueError("allow_live requires runtime.mode: live")
            # SimulatedBroker is paper-only; no live broker adapter ships here.
            raise ValueError("allow_live is unsupported: no live broker adapter in this repository")
        return self


class UniverseConfig(StrictConfigModel):
    min_price: float = 5.0
    min_adv: float = 1_000_000.0
    min_history_bars: int = 252
    exchanges: list[str] = Field(default_factory=lambda: ["XNYS", "XNAS"])
    security_types: list[str] = Field(default_factory=lambda: ["common_stock"])
    top_n_adv: int | None = 50
    include_delisted: bool = True

    @model_validator(mode="after")
    def valid_universe_filters(self) -> UniverseConfig:
        if not np.isfinite(self.min_price) or self.min_price <= 0:
            raise ValueError("min_price must be finite and positive")
        if not np.isfinite(self.min_adv) or self.min_adv < 0:
            raise ValueError("min_adv must be finite and non-negative")
        if self.min_history_bars < 1:
            raise ValueError("min_history_bars must be positive")
        if self.top_n_adv is not None and self.top_n_adv < 1:
            raise ValueError("top_n_adv must be positive when supplied")
        if not self.exchanges or not all(str(value).strip() for value in self.exchanges):
            raise ValueError("exchanges must contain non-empty identifiers")
        if not self.security_types or not all(str(value).strip() for value in self.security_types):
            raise ValueError("security_types must contain non-empty identifiers")
        return self


class CalendarConfig(StrictConfigModel):
    name: str = "XNYS"
    weekend: tuple[int, int] = (5, 6)

    @model_validator(mode="after")
    def valid_calendar(self) -> CalendarConfig:
        if not self.name.strip():
            raise ValueError("calendar name must be non-empty")
        if (
            len(self.weekend) != 2
            or len(set(self.weekend)) != 2
            or any(day < 0 or day > 6 for day in self.weekend)
        ):
            raise ValueError("weekend must contain two distinct weekday indices in [0, 6]")
        return self


class HorizonConfig(StrictConfigModel):
    bars: list[int] = Field(default_factory=lambda: [1, 5, 20])
    names: list[str] = Field(default_factory=lambda: ["1d", "5d", "20d"])

    @model_validator(mode="after")
    def matching_lengths(self) -> HorizonConfig:
        if len(self.bars) != len(self.names):
            raise ValueError("horizons.bars and horizons.names must have equal length")
        if (
            not self.bars
            or any(value < 1 for value in self.bars)
            or len(set(self.bars)) != len(self.bars)
        ):
            raise ValueError("horizon bars must be positive and unique")
        if any(not str(value).strip() for value in self.names) or len(set(self.names)) != len(
            self.names
        ):
            raise ValueError("horizon names must be non-empty and unique")
        return self


class QuantileConfig(StrictConfigModel):
    levels: list[float] = Field(
        default_factory=lambda: [0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99]
    )

    @field_validator("levels")
    @classmethod
    def sorted_open_unit(cls, v: list[float]) -> list[float]:
        if any(not np.isfinite(x) or x <= 0 or x >= 1 for x in v):
            raise ValueError("quantiles must be in (0, 1)")
        if v != sorted(v) or len(set(v)) != len(v):
            raise ValueError("quantiles must be strictly increasing")
        return v


class CostConfig(StrictConfigModel):
    commission_bps: float = 1.0
    half_spread_bps: float = 5.0
    impact_y: float = 0.1
    bps_per_turnover: float = 0.0
    borrow_bps_per_year: float = 50.0
    financing_bps_per_year: float = 0.0
    frictionless: bool = False
    participation_limit: float = 0.1

    @model_validator(mode="after")
    def non_negative_costs(self) -> CostConfig:
        for name in (
            "commission_bps",
            "half_spread_bps",
            "impact_y",
            "bps_per_turnover",
            "borrow_bps_per_year",
            "financing_bps_per_year",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0 < self.participation_limit <= 1:
            raise ValueError("participation_limit must be in (0, 1]")
        return self


class ExecutionConfig(StrictConfigModel):
    fill: FillConvention = FillConvention.NEXT_OPEN
    allow_close_auction: bool = False
    horizon_bars: int = 5
    n_slices: int = 5
    risk_aversion: float = 1e-6
    temporary_impact: float = 1e-6
    permanent_impact: float = 1e-7
    participation_rate: float = 0.05

    @model_validator(mode="after")
    def valid_execution_bounds(self) -> ExecutionConfig:
        if self.horizon_bars < 1 or self.n_slices < 1:
            raise ValueError("execution horizon_bars and n_slices must be positive")
        for name in ("risk_aversion", "temporary_impact", "permanent_impact"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0 < self.participation_rate <= 1:
            raise ValueError("participation_rate must be in (0, 1]")
        return self


class PortfolioConstraints(StrictConfigModel):
    """Gross is |w|_1. Sleeve caps are sum of long and sum of short.

    Defaults keep long_max + short_max == gross_leverage so the gross constraint
    is not a dead outer envelope under tiny accidental sleeve defaults. If you
    set sleeves tighter than gross on purpose, sleeve binds first; see
    ``effective_gross_cap``.
    """

    gross_leverage: float = 2.0
    net_exposure: float = 0.1
    name_max: float = 0.05
    name_min: float = -0.05
    long_max: float = 1.0
    short_max: float = 1.0
    turnover_limit: float = 0.5
    cash_buffer: float = 0.02
    sector_abs_max: float = 0.15
    factor_abs_max: float = 0.1
    beta_abs_max: float = 0.1
    max_positions: int | None = None
    cvar_limit: float | None = None
    cvar_alpha: float = 0.95
    predicted_vol_max: float | None = None
    # HARD capacity: per-name |Δw_i|*nav/ADV_i <= max_adv_participation when ADV
    # is provided to optimize_mean_variance. None disables the hard bound.
    # Soft/diagnostic participation is metrics.analytics.capacity_proxy (no weight change).
    max_adv_participation: float | None = 0.10

    @model_validator(mode="after")
    def valid_bounds(self) -> PortfolioConstraints:
        non_negative = (
            "gross_leverage",
            "net_exposure",
            "long_max",
            "short_max",
            "turnover_limit",
            "sector_abs_max",
            "factor_abs_max",
            "beta_abs_max",
            "cash_buffer",
            "cvar_alpha",
        )
        for name in non_negative:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not np.isfinite(self.name_min)
            or not np.isfinite(self.name_max)
            or self.name_min > self.name_max
            or self.name_min > 0
            or self.name_max < 0
        ):
            raise ValueError("name_min/name_max must bracket zero")
        if self.cash_buffer > 1 or not 0 < self.cvar_alpha < 1:
            raise ValueError("cash_buffer must be <= 1 and cvar_alpha must be in (0, 1)")
        if self.max_positions is not None and self.max_positions < 1:
            raise ValueError("max_positions must be positive when supplied")
        for name in ("cvar_limit", "predicted_vol_max", "max_adv_participation"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_adv_participation is not None and self.max_adv_participation > 1:
            raise ValueError("max_adv_participation must be <= 1")
        return self

    @property
    def effective_gross_cap(self) -> float:
        """Tightest book-level L1 envelope from gross and sleeve caps."""
        return float(min(self.gross_leverage, self.long_max + self.short_max))

    @property
    def gross_redundant(self) -> bool:
        """True when sleeve caps make ``gross_leverage`` non-binding."""
        return (self.long_max + self.short_max) + 1e-12 < self.gross_leverage


class OptimizerConfig(StrictConfigModel):
    lambda_risk: float = 10.0
    lambda_tc: float = 1.0
    # Explicit soft L1 turnover penalty on ||w - w_prev||_1 (in addition to
    # hard turnover_limit and linear TC via lambda_tc * tc @ |Δw|).
    lambda_turnover: float = 0.0
    lambda_tail: float = 0.0
    solver: str = "CLARABEL"
    mode: str = "mean_variance"
    risk_aversion: float = 1.0
    # Named covariance path for optimize_asof / /risk/portfolio. Default stays
    # trailing Ledoit–Wolf 2004 plus the GARCH/RGARCH overlay. dcc_gaussian,
    # dcc_student_t, adcc, ccc, agdcc, agdcc_full, and ewma are explicit
    # one-step paths. oas, ledoit_wolf_nonlinear, and sample are explicit
    # trailing paths plus the overlay. Generic dcc and catalog estimators
    # that are not optimizer-wired (factor) fail closed rather than
    # silently substituting. Named ledoit_wolf_nonlinear is analytical
    # 2020 spectral shrinkage and must not silently size as 2004 linear
    # Ledoit–Wolf. Named agdcc is diagonal CES AG-DCC; named agdcc_full
    # is unrestricted CES AG-DCC and must not silently size as diagonal
    # AG-DCC. Scalar CES ADCC is not diagonal AG-DCC. CCC is Bollerslev
    # constant correlation, not Engle DCC.
    covariance: str = "ledoit_wolf"

    @field_validator("covariance")
    @classmethod
    def implemented_optimizer_covariance(cls, value: object) -> str:
        from quant_fund.models.covariance import require_implemented_optimizer_covariance

        if not isinstance(value, str):
            raise ValueError("optimizer covariance must be a non-empty string")
        return require_implemented_optimizer_covariance(value)

    @model_validator(mode="after")
    def valid_optimizer_bounds(self) -> OptimizerConfig:
        for name in ("lambda_risk", "lambda_tc", "lambda_turnover", "lambda_tail", "risk_aversion"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.mode not in {"mean_variance", "cvar"}:
            raise ValueError("optimizer mode must be mean_variance or cvar")
        if not self.solver.strip():
            raise ValueError("optimizer solver must be non-empty")
        return self


class ValidationConfig(StrictConfigModel):
    scheme: str = "expanding"
    train_bars: int = 252
    val_bars: int = 63
    test_bars: int = 63
    embargo_bars: int | None = None
    n_groups: int = 6
    test_groups: int = 2
    n_optuna_trials: int = 20
    seed: int = 42

    @model_validator(mode="after")
    def valid_validation_windows(self) -> ValidationConfig:
        if self.scheme not in {"expanding", "rolling"}:
            raise ValueError("validation scheme must be expanding or rolling")
        for name in (
            "train_bars",
            "val_bars",
            "test_bars",
            "n_groups",
            "test_groups",
            "n_optuna_trials",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.test_groups >= self.n_groups:
            raise ValueError("test_groups must be less than n_groups")
        if self.embargo_bars is not None and self.embargo_bars < 0:
            raise ValueError("embargo_bars must be non-negative")
        return self


class FeatureConfig(StrictConfigModel):
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

    @model_validator(mode="after")
    def valid_feature_settings(self) -> FeatureConfig:
        if not np.isfinite(self.winsor_p) or not 0 <= self.winsor_p < 0.5:
            raise ValueError("winsor_p must be finite and in [0, 0.5)")
        if not np.isfinite(self.ewma_lambda) or not 0 <= self.ewma_lambda < 1:
            raise ValueError("ewma_lambda must be finite and in [0, 1)")
        if self.amihud_lookback < 1 or self.adv_lookback < 1:
            raise ValueError("feature lookbacks must be positive")
        if not self.families or any(not str(name).strip() for name in self.families):
            raise ValueError("feature families must contain non-empty names")
        if len(set(self.families)) != len(self.families):
            raise ValueError("feature families must be unique")
        return self


class MissingConfig(StrictConfigModel):
    policy: MissingPolicy = MissingPolicy.NATIVE


class DataConfig(StrictConfigModel):
    root: Path = Path("data")
    source: str = "synthetic"
    parquet_path: Path | None = None
    synthetic_n_assets: int = 12
    synthetic_n_days: int = 400
    synthetic_seed: int = 7
    synthetic_oracle_beta: float = 0.015
    synthetic_oracle_phi: float = 0.70
    benchmark_id: str = "SEC_MKT"

    @field_validator("source")
    @classmethod
    def supported_source(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"synthetic", "file", "parquet"}:
            raise ValueError("data source must be one of: synthetic, file, parquet")
        return normalized

    @model_validator(mode="after")
    def valid_data_settings(self) -> DataConfig:
        if not self.source.strip():
            raise ValueError("data source must be non-empty")
        if self.synthetic_n_assets < 1 or self.synthetic_n_days < 1:
            raise ValueError("synthetic asset/day counts must be positive")
        if not np.isfinite(self.synthetic_oracle_beta):
            raise ValueError("synthetic_oracle_beta must be finite")
        if not np.isfinite(self.synthetic_oracle_phi) or not -1.0 < self.synthetic_oracle_phi < 1.0:
            raise ValueError("synthetic_oracle_phi must be finite and strictly between -1 and 1")
        if not self.benchmark_id.strip():
            raise ValueError("benchmark_id must be non-empty")
        return self


class TrainConfig(StrictConfigModel):
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
    garch_dist: GarchDist = GarchDist.NORMAL
    garch_vol: GarchVolSpec = GarchVolSpec.GARCH
    har_log: bool = True
    qlike_floor: float = 1e-12
    psd_eigen_tol: float = 1e-10

    @model_validator(mode="after")
    def valid_training_settings(self) -> TrainConfig:
        for name in (
            "ranking_target",
            "distribution_target",
            "volatility_target",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        for name in (
            "xgb_n_estimators",
            "xgb_max_depth",
            "lgbm_n_estimators",
            "lgbm_num_leaves",
            "n_hmm_states",
            "garch_p",
            "garch_q",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("ridge_alpha", "elasticnet_l1", "qlike_floor", "psd_eigen_tol"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0 <= self.elasticnet_l1 <= 1:
            raise ValueError("elasticnet_l1 must be in [0, 1]")
        if self.qlike_floor <= 0 or self.psd_eigen_tol <= 0:
            raise ValueError("qlike_floor and psd_eigen_tol must be positive")
        if self.garch_vol == GarchVolSpec.FIGARCH and (
            self.garch_p not in (0, 1) or self.garch_q not in (0, 1)
        ):
            raise ValueError("FIGARCH garch_p and garch_q must be 0 or 1")
        return self


class RiskGateConfig(StrictConfigModel):
    max_order_notional: float = 1_000_000.0
    max_gross: float = 2.0
    max_net: float = 0.2
    max_name: float = 0.03
    max_participation: float = 0.1
    max_predicted_vol: float = 0.4
    stale_price_bars: int = 3
    stale_model_hours: float = 24.0

    @model_validator(mode="after")
    def valid_risk_limits(self) -> RiskGateConfig:
        for name in (
            "max_order_notional",
            "max_gross",
            "max_net",
            "max_name",
            "max_predicted_vol",
            "stale_model_hours",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not np.isfinite(self.max_participation) or not 0 < self.max_participation <= 1:
            raise ValueError("max_participation must be finite and in (0, 1]")
        if self.stale_price_bars < 0:
            raise ValueError("stale_price_bars must be non-negative")
        return self


class PromotionConfig(StrictConfigModel):
    min_mean_ic: float = 0.0
    min_cost_adjusted_spread: float = 0.0
    max_turnover: float = 5.0
    require_leakage_pass: bool = True
    min_folds: int = 2
    min_fold_ic_stability: float = 0.0  # min fraction of folds with IC > min_mean_ic

    @model_validator(mode="after")
    def valid_promotion_thresholds(self) -> PromotionConfig:
        for name in ("min_mean_ic", "min_cost_adjusted_spread", "max_turnover"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.max_turnover < 0:
            raise ValueError("max_turnover must be non-negative")
        if self.min_folds < 1:
            raise ValueError("min_folds must be positive")
        if not np.isfinite(self.min_fold_ic_stability) or not 0 <= self.min_fold_ic_stability <= 1:
            raise ValueError("min_fold_ic_stability must be finite and in [0, 1]")
        return self


class FusionConfig(StrictConfigModel):
    alpha_weight: float = 1.0
    confidence_weight: float = 1.0
    regime_weight: float = 1.0
    risk_weight: float = 1.0
    tail_penalty: float = 1.0
    liquidity_penalty: float = 1.0
    min_executable_alpha: float = 0.0
    alpha_scale: float = 0.005
    # When True, optimize_asof sizes on fused diagnostics rather than raw alpha.
    use_fused_for_optimize: bool = True
    # When True and conformal intervals exist, clip |w| via interval_risk caps.
    apply_interval_caps: bool = True
    # Research-only: skip conformal_sets_asof in forecast_asof (weight-only smoke).
    # Does not affect live claim semantics; intervals/caps simply absent.
    skip_intervals: bool = False

    @model_validator(mode="after")
    def valid_fusion_weights(self) -> FusionConfig:
        non_negative = (
            "alpha_weight",
            "confidence_weight",
            "regime_weight",
            "tail_penalty",
            "liquidity_penalty",
            "min_executable_alpha",
            "alpha_scale",
        )
        for name in non_negative:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not np.isfinite(self.risk_weight) or self.risk_weight <= 0:
            raise ValueError("risk_weight must be finite and positive")
        return self


class MonitoringConfig(StrictConfigModel):
    psi_alert: float = 0.25
    reference_bars: int = 63
    live_bars: int = 21

    @model_validator(mode="after")
    def valid_monitoring_windows(self) -> MonitoringConfig:
        if not np.isfinite(self.psi_alert) or self.psi_alert < 0:
            raise ValueError("psi_alert must be finite and non-negative")
        if self.reference_bars < 1 or self.live_bars < 1:
            raise ValueError("monitoring windows must be positive")
        return self


class PaperConfig(StrictConfigModel):
    """Phase 17 paper / shadow loop settings."""

    initial_nav: float = 1_000_000.0
    max_steps: int | None = None
    use_wall_clock: bool = False
    enable_shadow: bool = True
    shadow_scale: float = 1.0
    # Optional extra scaled challengers (no capital). Primary shadow stays separate
    # for ledger / promotion_dry_run compatibility. Empty = CLI builds none.
    challenger_scales: list[float] = Field(default_factory=list)
    ledger_subdir: str = "paper"
    # Multi-day resume: continue from an existing run_id ledger when set.
    resume_run_id: str | None = None
    # Shadow→champion promotion dry-run threshold on mean L1 divergence.
    promote_max_mean_l1: float = 0.25
    promote_min_steps: int = 5

    @field_validator("challenger_scales")
    @classmethod
    def finite_challenger_scales(cls, v: list[float]) -> list[float]:
        out: list[float] = []
        for x in v:
            fx = float(x)
            if not np.isfinite(fx) or fx < 0:
                raise ValueError("challenger_scales must be finite and non-negative")
            out.append(fx)
        return out

    @model_validator(mode="after")
    def valid_paper_settings(self) -> PaperConfig:
        if not np.isfinite(self.initial_nav) or self.initial_nav <= 0:
            raise ValueError("initial_nav must be finite and positive")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive when supplied")
        if not np.isfinite(self.shadow_scale) or self.shadow_scale < 0:
            raise ValueError("shadow_scale must be finite and non-negative")
        if not np.isfinite(self.promote_max_mean_l1) or self.promote_max_mean_l1 < 0:
            raise ValueError("promote_max_mean_l1 must be finite and non-negative")
        if self.promote_min_steps < 1:
            raise ValueError("promote_min_steps must be positive")
        ledger_path = Path(self.ledger_subdir)
        if (
            not self.ledger_subdir.strip()
            or ledger_path.is_absolute()
            or any(part in {"", ".", ".."} for part in ledger_path.parts)
        ):
            raise ValueError("ledger_subdir must be a safe relative path")
        return self


class KillSwitchConfig(StrictConfigModel):
    state: str = "ENABLED"
    allow_auto_flatten: bool = False


class NorthsetConfig(StrictConfigModel):
    """Order-book + candlestick research slice (ADR-021)."""

    n_book_levels: int = 5
    n_session_candles: int = 8
    base_spread_bps: float = 4.0
    min_names: int = 5
    sweep_lookback: int = 20
    sweep_horizons: list[int] = Field(default_factory=lambda: [1, 5, 20])
    sweep_vol_lookback: int = 20
    sweep_n_boot: int = 500
    sweep_n_permutations: int = 200
    sweep_n_folds: int = 4
    sweep_min_events: int = 30
    sweep_min_dates: int = 20
    sweep_min_fold_positive_fraction: float = 0.75
    sweep_cooldown_bars: int = 5
    sweep_sensitivity_lookbacks: list[int] = Field(default_factory=lambda: [10, 20, 40])
    require_adjusted_ohlc: bool = True
    book_max_age_seconds: int = 86_400
    book_panel_path: str | None = None  # optional vendor-shaped parquet panel
    book_join_coverage_floor: float = 0.5  # fail-closed when external panel join coverage low
    include_kyle_ofi: bool = False  # nest bench_kyle_ofi_fused under receipt["kyle_ofi"]
    depth_shape_finite_floor: float | None = None  # optional fail-closed depth-shape rate
    concentration_top_finite_floor: float | None = None  # optional fail-closed concentration rate
    queue_priority_finite_floor: float | None = None  # optional fail-closed queue-priority rate
    side_notional_finite_floor: float | None = None  # optional fail-closed side-notional rate
    tob_size_share_finite_floor: float | None = None  # optional fail-closed tob size-share rate
    use_session_l2: bool = True  # multi-snapshot session books → daily path features
    session_l2_identity_floor: float = 0.99  # fail-closed min rate when use_session_l2

    @model_validator(mode="after")
    def valid_northset_settings(self) -> NorthsetConfig:
        if self.n_book_levels < 1:
            raise ValueError("n_book_levels must be positive")
        if self.n_session_candles < 2:
            raise ValueError("n_session_candles must be >= 2")
        if not np.isfinite(self.base_spread_bps) or self.base_spread_bps <= 0:
            raise ValueError("base_spread_bps must be finite and positive")
        if self.min_names < 2:
            raise ValueError("min_names must be >= 2")
        if not np.isfinite(self.session_l2_identity_floor) or not (
            0.0 <= float(self.session_l2_identity_floor) <= 1.0
        ):
            raise ValueError("session_l2_identity_floor must be in [0, 1]")
        if self.sweep_lookback < 2:
            raise ValueError("sweep_lookback must be >= 2")
        if not self.sweep_horizons or any(int(h) < 1 for h in self.sweep_horizons):
            raise ValueError("sweep_horizons must contain positive integers")
        if len(set(self.sweep_horizons)) != len(self.sweep_horizons):
            raise ValueError("sweep_horizons must be unique")
        if self.sweep_vol_lookback < 3:
            raise ValueError("sweep_vol_lookback must be >= 3")
        if self.sweep_n_boot < 50:
            raise ValueError("sweep_n_boot must be >= 50")
        if self.sweep_n_permutations < 50:
            raise ValueError("sweep_n_permutations must be >= 50")
        if self.sweep_n_folds < 2:
            raise ValueError("sweep_n_folds must be >= 2")
        if self.sweep_min_events < 10:
            raise ValueError("sweep_min_events must be >= 10")
        if self.sweep_min_dates < 5:
            raise ValueError("sweep_min_dates must be >= 5")
        if not 0.0 <= self.sweep_min_fold_positive_fraction <= 1.0:
            raise ValueError("sweep_min_fold_positive_fraction must be in [0, 1]")
        if self.sweep_cooldown_bars < 0:
            raise ValueError("sweep_cooldown_bars must be non-negative")
        if any(int(v) < 2 for v in self.sweep_sensitivity_lookbacks):
            raise ValueError("sweep_sensitivity_lookbacks entries must be >= 2")
        if len(set(self.sweep_sensitivity_lookbacks)) != len(self.sweep_sensitivity_lookbacks):
            raise ValueError("sweep_sensitivity_lookbacks must be unique")
        if self.book_max_age_seconds < 1:
            raise ValueError("book_max_age_seconds must be positive")
        if not np.isfinite(self.book_join_coverage_floor) or not (
            0.0 <= float(self.book_join_coverage_floor) <= 1.0
        ):
            raise ValueError("book_join_coverage_floor must be in [0, 1]")
        if self.depth_shape_finite_floor is not None:
            floor = float(self.depth_shape_finite_floor)
            if not np.isfinite(floor) or not (0.0 <= floor <= 1.0):
                raise ValueError("depth_shape_finite_floor must be in [0, 1] or None")
        if self.concentration_top_finite_floor is not None:
            floor = float(self.concentration_top_finite_floor)
            if not np.isfinite(floor) or not (0.0 <= floor <= 1.0):
                raise ValueError("concentration_top_finite_floor must be in [0, 1] or None")
        return self


class RobinhoodPlusConfig(StrictConfigModel):
    """Kronos-derived K-line foundation engine (robinhood+). ADR-023."""

    enabled: bool = True
    backend: RobinhoodPlusBackend = RobinhoodPlusBackend.NUMPY
    decoder: RobinhoodPlusDecoder = RobinhoodPlusDecoder.HIERARCHICAL_MARKOV
    variant: RobinhoodPlusVariant = RobinhoodPlusVariant.MINI
    lookback: int = 64
    pred_len: int = 20
    sample_count: int = 8
    s1_bits: int = 5
    s2_bits: int = 5
    temperature: float = 1.0
    top_p: float = 0.9
    max_context: int = 512
    clip: float = 5.0
    # Default 0: numpy Markov lost the SYNTHETIC ridge champion/challenger card.
    # An engine that cannot beat the baseline must not size the book (ADR-023).
    blend_weight: float = 0.0
    allow_network: bool = False
    tokenizer_path: str | None = None
    model_path: str | None = None

    @model_validator(mode="after")
    def valid_robinhood_plus(self) -> RobinhoodPlusConfig:
        if self.lookback < 2:
            raise ValueError("robinhood_plus.lookback must be at least 2")
        if self.pred_len < 1:
            raise ValueError("robinhood_plus.pred_len must be positive")
        if self.sample_count < 1:
            raise ValueError("robinhood_plus.sample_count must be positive")
        if self.s1_bits < 1 or self.s2_bits < 1:
            raise ValueError("robinhood_plus s1_bits/s2_bits must be positive")
        if self.s1_bits + self.s2_bits > 16:
            raise ValueError("robinhood_plus codebook bits must be <= 16 for the numpy backend")
        if self.max_context < 1:
            raise ValueError("robinhood_plus.max_context must be positive")
        for name in ("temperature", "clip"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"robinhood_plus.{name} must be finite and positive")
        if not np.isfinite(self.top_p) or not 0.0 < self.top_p <= 1.0:
            raise ValueError("robinhood_plus.top_p must be finite and in (0, 1]")
        if not np.isfinite(self.blend_weight) or not 0.0 <= self.blend_weight <= 1.0:
            raise ValueError("robinhood_plus.blend_weight must be finite and in [0, 1]")
        if self.backend is RobinhoodPlusBackend.TORCH and self.allow_network:
            # Hub downloads are explicit; CI and default research stay offline.
            pass
        return self


class AppConfig(StrictConfigModel):
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
    paper: PaperConfig = Field(default_factory=PaperConfig)
    northset: NorthsetConfig = Field(default_factory=NorthsetConfig)
    robinhood_plus: RobinhoodPlusConfig = Field(default_factory=RobinhoodPlusConfig)
    inherit: str | None = None

    def embargo_bars(self) -> int:
        if self.validation.embargo_bars is not None:
            return self.validation.embargo_bars
        return max(self.horizons.bars)

    def dump(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

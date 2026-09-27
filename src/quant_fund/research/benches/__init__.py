"""Scientific family benches. Scores are proper rules, not Sharpe."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import comb
from typing import Any, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.config.models import AppConfig
from quant_fund.execution.almgren_chriss import (
    almgren_chriss_trajectory,
    expected_shortfall_ac,
    slice_trades,
    twap_trajectory,
)
from quant_fund.metrics.conformal import (
    assign_terciles,
    conditional_coverage,
    covered,
    set_metrics,
    worst_slice_coverage,
)
from quant_fund.metrics.cross_section import _date_keys as _date_keys
from quant_fund.metrics.cross_section import date_ic_series, decile_portfolios
from quant_fund.metrics.evalues import bench_e_coverage, e_process, e_process_dm
from quant_fund.metrics.inference import (
    diebold_mariano,
    overlap_aware_hac_lags,
    pairwise_diebold_mariano,
)
from quant_fund.metrics.probability import (
    acerbi_szekely_z1,
    acerbi_szekely_z2,
    brier_score,
    christoffersen_cc,
    christoffersen_independence,
    expected_calibration_error,
    kupiec_pof,
    log_loss,
    pit_ks,
)
from quant_fund.metrics.scoring import (
    coverage,
    crps_from_quantiles,
    date_level_equal_weight,
    mean_crps_gaussian,
    mean_crps_student_t,
    mean_fissler_ziegel,
    mean_pinball,
    nonoverlapping_origin_mask,
    pearson_ic,
    pinball_loss,
    pit_values,
    qlike,
    quantile_crossing_rate,
)
from quant_fund.models.alpha import HistoricalMeanAlpha, RidgeAlpha
from quant_fund.models.conformal import (
    AdaptiveConformal,
    MondrianACI,
    MondrianCQR,
    SplitCQR,
    SplitOneSided,
)
from quant_fund.models.crc import ConformalRiskControl, bench_crc_var, loss_hit
from quant_fund.models.cv_plus import CVPlus, cv_plus_coverage_level
from quant_fund.models.distribution import (
    EmpiricalDistribution,
    GaussianDistribution,
    LinearQuantileDistribution,
    ScaledEmpiricalDistribution,
    ScaledGaussianDistribution,
    ScaledStudentTDistribution,
    fit_operational_wrappee,
    select_scaled_wrappee,
)
from quant_fund.models.jackknife_plus import JackknifePlus, jackknife_plus_coverage_level
from quant_fund.models.quantile_bandit import QuantileThompson
from quant_fund.models.ranking import (
    ORACLE_FEATURES,
    PUBLIC_FEATURES,
    RidgeRanker,
    available_features,
)
from quant_fund.models.regime import GaussianHMMRegime, SingleStateRegime, VolThresholdRegime
from quant_fund.models.rl import run_linucb_panel
from quant_fund.models.tail import DrawdownClassifier, HistoricalTail, ScaledHistoricalTail
from quant_fund.models.weighted_conformal import WeightedSplitCQR
from quant_fund.pipeline.dataset import design_matrix
from quant_fund.pipeline.train import _fit_ranker as _fit_ranker
from quant_fund.pipeline.train import _label_horizon as _label_horizon
from quant_fund.pipeline.train import _make_ranker as _make_ranker
from quant_fund.pipeline.train import _predict_ranker as _predict_ranker
from quant_fund.portfolio.interval_risk import (
    bench_interval_caps,
    cap_from_interval,
    equal_weight_per_date,
    interval_refs,
)
from quant_fund.validation.cpcv import combinatorial_purged_cv
from quant_fund.validation.walk_forward import timestamp_ns, walk_forward

from .common import (
    _BANDIT_MAX_DATES as _BANDIT_MAX_DATES,
)
from .common import (
    _EV_MAX_DATES as _EV_MAX_DATES,
)
from .common import (
    _JP_MAX_CAL as _JP_MAX_CAL,
)
from .common import (
    _JP_MAX_TEST as _JP_MAX_TEST,
)
from .common import (
    _abs_y_labels as _abs_y_labels,
)
from .common import (
    _aligned_col as _aligned_col,
)
from .common import (
    _crps_from_quantiles_obs as _crps_from_quantiles_obs,
)
from .common import (
    _even_take as _even_take,
)
from .common import (
    _gaussian_interval_split as _gaussian_interval_split,
)
from .common import (
    _holdout as _holdout,
)
from .common import (
    _interval_label as _interval_label,
)
from .common import (
    _public_features_in as _public_features_in,
)
from .common import (
    _scaled_fill as _scaled_fill,
)
from .common import (
    _tail_date_mask as _tail_date_mask,
)
from .common import (
    _triple_split as _triple_split,
)
from .common import (
    _vol_or_width as _vol_or_width,
)
from .families import (
    _distribution_horizon_scores as _distribution_horizon_scores,
)
from .families import (
    bench_distribution,
    bench_drawdown,
    bench_liquidity,
    bench_regime,
    bench_rl,
    bench_tail,
    bench_volatility,
)
from .intervals import (
    _bench_cv_plus_panel as _bench_cv_plus_panel,
)
from .intervals import (
    bench_conformal,
    bench_conformal_topk_from_panel,
    bench_cpcv_audit,
    bench_crc,
    bench_cv_plus,
    bench_evalues,
    bench_interval_risk,
    bench_jackknife_plus,
    bench_localized_from_panel,
    bench_online_crc_from_panel,
    bench_portfolio_from_panel,
    bench_quantile_bandit,
    bench_weighted_conformal,
)
from .ranking import (
    _policy_bandit_metrics as _policy_bandit_metrics,
)
from .ranking import (
    _policy_paired_rewards as _policy_paired_rewards,
)
from .ranking import (
    _policy_planted_oracle as _policy_planted_oracle,
)
from .ranking import (
    _policy_public_design as _policy_public_design,
)
from .ranking import (
    _policy_ridge_topk_rewards as _policy_ridge_topk_rewards,
)
from .ranking import (
    _policy_topk_mean as _policy_topk_mean,
)
from .ranking import (
    _public_leak_diagnostic as _public_leak_diagnostic,
)
from .ranking import (
    bench_alpha,
    bench_ranking,
    oos_rank_scores,
)

__all__ = [
    "AdaptiveConformal",
    "Any",
    "AppConfig",
    "CVPlus",
    "ConformalRiskControl",
    "DrawdownClassifier",
    "EmpiricalDistribution",
    "GaussianDistribution",
    "GaussianHMMRegime",
    "HistoricalMeanAlpha",
    "HistoricalTail",
    "JackknifePlus",
    "LinearQuantileDistribution",
    "MondrianACI",
    "MondrianCQR",
    "NDArray",
    "ORACLE_FEATURES",
    "PUBLIC_FEATURES",
    "QuantileThompson",
    "RidgeAlpha",
    "RidgeRanker",
    "ScaledEmpiricalDistribution",
    "ScaledGaussianDistribution",
    "ScaledHistoricalTail",
    "ScaledStudentTDistribution",
    "SingleStateRegime",
    "SplitCQR",
    "SplitOneSided",
    "UTC",
    "VolThresholdRegime",
    "WeightedSplitCQR",
    "acerbi_szekely_z1",
    "acerbi_szekely_z2",
    "almgren_chriss_trajectory",
    "assign_terciles",
    "available_features",
    "bench_alpha",
    "bench_conformal",
    "bench_conformal_topk_from_panel",
    "bench_cpcv_audit",
    "bench_crc",
    "bench_crc_var",
    "bench_cv_plus",
    "bench_distribution",
    "bench_drawdown",
    "bench_e_coverage",
    "bench_evalues",
    "bench_interval_caps",
    "bench_interval_risk",
    "bench_jackknife_plus",
    "bench_liquidity",
    "bench_localized_from_panel",
    "bench_online_crc_from_panel",
    "bench_portfolio_from_panel",
    "bench_quantile_bandit",
    "bench_ranking",
    "bench_regime",
    "bench_rl",
    "bench_tail",
    "bench_volatility",
    "bench_weighted_conformal",
    "brier_score",
    "cap_from_interval",
    "cast",
    "christoffersen_cc",
    "christoffersen_independence",
    "comb",
    "combinatorial_purged_cv",
    "conditional_coverage",
    "coverage",
    "covered",
    "crps_from_quantiles",
    "cv_plus_coverage_level",
    "date_ic_series",
    "date_level_equal_weight",
    "datetime",
    "decile_portfolios",
    "design_matrix",
    "diebold_mariano",
    "e_process",
    "e_process_dm",
    "equal_weight_per_date",
    "expected_calibration_error",
    "expected_shortfall_ac",
    "fit_operational_wrappee",
    "interval_refs",
    "jackknife_plus_coverage_level",
    "kupiec_pof",
    "log_loss",
    "loss_hit",
    "mean_crps_gaussian",
    "mean_crps_student_t",
    "mean_fissler_ziegel",
    "mean_pinball",
    "nonoverlapping_origin_mask",
    "np",
    "oos_rank_scores",
    "overlap_aware_hac_lags",
    "pairwise_diebold_mariano",
    "pearson_ic",
    "pinball_loss",
    "pit_ks",
    "pit_values",
    "pl",
    "qlike",
    "quantile_crossing_rate",
    "run_linucb_panel",
    "select_scaled_wrappee",
    "set_metrics",
    "slice_trades",
    "timedelta",
    "timestamp_ns",
    "twap_trajectory",
    "walk_forward",
    "worst_slice_coverage",
]

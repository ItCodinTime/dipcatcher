"""Train forecast families on gold panels. Holdout is not used for Optuna."""

from __future__ import annotations

import re
import types
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.lightspeed.ranker import NauticaRanker
from quant_fund.metrics.cross_section import date_ic_series
from quant_fund.metrics.probability import brier_score
from quant_fund.metrics.risk import losses_from_returns
from quant_fund.metrics.scoring import (
    GARCH_ONE_STEP_CRPS_TAUS,
    crps_from_quantiles,
    crps_gaussian,
    icir,
    mean_pinball,
    name_level_one_step_density_summary,
    name_level_qlike,
    one_step_density_summary,
    overlap_aware_qlike,
    qlike,
    quantile_crossing_rate,
)
from quant_fund.models.alpha import HistoricalMeanAlpha
from quant_fund.models.asset_pricing import IPCARanker, RandomFourierRanker, SDFRidgeRanker
from quant_fund.models.base import (
    JoblibMixin,
    artifact_identity,
    load_joblib_artifact,
    save_joblib_artifact,
)
from quant_fund.models.calibration import ProbabilityCalibrator
from quant_fund.models.cs_papers import (
    DATED_FIT_RANKERS,
    DATED_PREDICT_RANKERS,
    ID_FIT_RANKERS,
    ID_PREDICT_RANKERS,
    PAPER_RANKER_NAMES,
    AdaptiveLassoRanker,
    ClassicRanker,
    ClassicShortRanker,
    CombinationRanker,
    DoubleSelectionRanker,
    FamaMacBethRanker,
    FamaMacBethRidgeRanker,
    FNWRanker,
    GBRTRanker,
    GXThreePassRanker,
    ICWeightedCombinationRanker,
    KraussRanker,
    MSFECombinationRanker,
    PCRRanker,
    PLSRanker,
    PrincipalPortfolioRanker,
    ReversalRanker,
    RPPCARanker,
    SDFElasticNetRanker,
    ThreePassFilterRanker,
    TSMOMRanker,
    VMERanker,
    make_combo_ic_st,
    make_fm_st,
    make_ridge_neut,
    make_ridge_st,
)
from quant_fund.models.deep_rl import PolicyGradientRanker, run_policy_gradient_panel
from quant_fund.models.distribution import (
    EmpiricalDistribution,
    GaussianDistribution,
    GMMDistribution,
    IsotonicPitDistribution,
    LinearQuantileDistribution,
    SkewTDistribution,
    StackedDistribution,
    TreeQuantileDistribution,
)
from quant_fund.models.quantile_bandit import QuantileThompson
from quant_fund.models.ranking import (
    CompositeRanker,
    ElasticNetRanker,
    EnsembleRanker,
    LGBMLambdaRanker,
    LGBMRegRanker,
    NeuralRanker,
    RidgeRanker,
    XGBRegRanker,
    group_sizes,
)
from quant_fund.models.realized_garch import (
    REALIZED_GARCH_MEASURE,
    RealizedGARCHVol,
    parkinson_daily_variance,
)
from quant_fund.models.regime import GaussianHMMRegime, SingleStateRegime, VolThresholdRegime
from quant_fund.models.rl import (
    LinearThompsonRanker,
    LinUCBRanker,
    run_linucb_panel,
    run_thompson_panel,
)
from quant_fund.models.tail import DrawdownClassifier, GaussianTail, HistoricalTail
from quant_fund.models.volatility import (
    GARCH_DATE_LEVEL_SCOPE,
    GARCH_SECURITY_LEVEL_SCOPE,
    EWMAVol,
    GARCHVol,
    HARVol,
    RollingVol,
    TreeVol,
)
from quant_fund.pipeline.dataset import design_frame, design_matrix, panel
from quant_fund.registry.mlflow_store import attach_artifact_identity, configure_tracking, log_run
from quant_fund.reporting.report import write_evidence_report
from quant_fund.schemas.errors import PointInTimeError
from quant_fund.utils.hashing import canonical_frame_fingerprint, canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision, git_worktree_sha256
from quant_fund.utils.seeds import set_global_seed
from quant_fund.validation.purging import purge_mask
from quant_fund.validation.walk_forward import Fold, timestamp_ns, walk_forward

from . import distribution as _mod_distribution
from . import families as _mod_families
from . import garch_data as _mod_garch_data
from . import ranking as _mod_ranking
from . import splits as _mod_splits
from . import volatility as _mod_volatility
from .distribution import (
    train_distribution,
    train_distribution_auto,
)
from .families import (
    train_alpha,
    train_family,
    train_policy_gradient,
    train_quantile_bandit,
    train_regime,
    train_reinforcement,
    train_reinforcement_auto,
    train_robinhood_plus,
    train_tail,
)
from .garch_data import (
    _garch_name_return_history as _garch_name_return_history,
)
from .garch_data import (
    _garch_return_history as _garch_return_history,
)
from .garch_data import (
    _realized_garch_history as _realized_garch_history,
)
from .garch_data import (
    _realized_garch_ohlc_columns as _realized_garch_ohlc_columns,
)
from .garch_data import (
    _require_garch_security_keys as _require_garch_security_keys,
)
from .ranking import (
    RANKING_MODEL_NAMES,
    train_calibration,
    train_calibration_auto,
    train_ranking,
    train_ranking_auto,
)
from .ranking import (
    _fit_ranker as _fit_ranker,
)
from .ranking import (
    _make_ranker as _make_ranker,
)
from .ranking import (
    _predict_ranker as _predict_ranker,
)
from .splits import (
    _aligned_label_end_times as _aligned_label_end_times,
)
from .splits import (
    _available_stamp_is_missing as _available_stamp_is_missing,
)
from .splits import (
    _chronological_split as _chronological_split,
)
from .splits import (
    _label_horizon as _label_horizon,
)
from .splits import (
    _require_model as _require_model,
)
from .splits import (
    _split_fold as _split_fold,
)
from .splits import (
    _stamp_at_or_before as _stamp_at_or_before,
)
from .splits import (
    _stamp_strictly_before as _stamp_strictly_before,
)
from .splits import (
    _walk_forward_splits as _walk_forward_splits,
)
from .volatility import (
    _empty_garch_density_metrics as _empty_garch_density_metrics,
)
from .volatility import (
    _garch_name_oos_predictions as _garch_name_oos_predictions,
)
from .volatility import (
    _garch_name_origin_return_lookup as _garch_name_origin_return_lookup,
)
from .volatility import (
    _garch_name_row_id as _garch_name_row_id,
)
from .volatility import (
    _garch_one_step_sigma as _garch_one_step_sigma,
)
from .volatility import (
    _garch_oos_predictions as _garch_oos_predictions,
)
from .volatility import (
    _garch_origin_density_record as _garch_origin_density_record,
)
from .volatility import (
    _realized_garch_oos_predictions as _realized_garch_oos_predictions,
)
from .volatility import (
    _requested_garch_security_ids as _requested_garch_security_ids,
)
from .volatility import (
    garch_name_walk_forward,
    train_volatility,
    train_volatility_auto,
)

_FACADE_MODULES = (
    _mod_ranking,
    _mod_splits,
    _mod_garch_data,
    _mod_distribution,
    _mod_volatility,
    _mod_families,
)

_MISSING = object()


def _publish_split_modules() -> None:
    """Copy split-module globals here so rebound functions resolve patches."""
    facade = globals()
    for module in _FACADE_MODULES:
        for key, value in vars(module).items():
            if key.startswith("__"):
                continue
            current = facade.get(key, _MISSING)
            if current is not _MISSING and current is not value:
                raise RuntimeError(
                    f"facade global conflict on {key!r} while publishing {module.__name__}"
                )
            facade[key] = value


def _rebind_to_facade(fn: types.FunctionType) -> types.FunctionType:
    rebound = types.FunctionType(
        fn.__code__,
        globals(),
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )
    rebound.__kwdefaults__ = fn.__kwdefaults__
    rebound.__annotations__ = dict(getattr(fn, "__annotations__", {}))
    rebound.__dict__.update(fn.__dict__)
    rebound.__module__ = __name__
    rebound.__qualname__ = fn.__qualname__
    return rebound


def _rebind_split_functions() -> None:
    facade = globals()
    prefix = __name__ + "."
    for key, value in list(facade.items()):
        module_name = getattr(value, "__module__", "") or ""
        if isinstance(value, types.FunctionType) and module_name.startswith(prefix):
            facade[key] = _rebind_to_facade(value)


_publish_split_modules()
_rebind_split_functions()

__all__ = [
    "AdaptiveLassoRanker",
    "Any",
    "AppConfig",
    "Callable",
    "ClassicRanker",
    "ClassicShortRanker",
    "CombinationRanker",
    "CompositeRanker",
    "DATED_FIT_RANKERS",
    "DATED_PREDICT_RANKERS",
    "DoubleSelectionRanker",
    "DrawdownClassifier",
    "EWMAVol",
    "ElasticNetRanker",
    "EmpiricalDistribution",
    "EnsembleRanker",
    "FNWRanker",
    "FamaMacBethRanker",
    "FamaMacBethRidgeRanker",
    "Fold",
    "GARCHVol",
    "GARCH_DATE_LEVEL_SCOPE",
    "GARCH_ONE_STEP_CRPS_TAUS",
    "GARCH_SECURITY_LEVEL_SCOPE",
    "GBRTRanker",
    "GMMDistribution",
    "GXThreePassRanker",
    "GaussianDistribution",
    "GaussianHMMRegime",
    "GaussianTail",
    "HARVol",
    "HistoricalMeanAlpha",
    "HistoricalTail",
    "ICWeightedCombinationRanker",
    "ID_FIT_RANKERS",
    "ID_PREDICT_RANKERS",
    "IPCARanker",
    "IsotonicPitDistribution",
    "JoblibMixin",
    "KraussRanker",
    "LGBMLambdaRanker",
    "LGBMRegRanker",
    "LinUCBRanker",
    "LinearQuantileDistribution",
    "LinearThompsonRanker",
    "MSFECombinationRanker",
    "NauticaRanker",
    "NeuralRanker",
    "PAPER_RANKER_NAMES",
    "PCRRanker",
    "PLSRanker",
    "Path",
    "PointInTimeError",
    "PolicyGradientRanker",
    "PrincipalPortfolioRanker",
    "ProbabilityCalibrator",
    "QuantileThompson",
    "RANKING_MODEL_NAMES",
    "REALIZED_GARCH_MEASURE",
    "RPPCARanker",
    "RandomFourierRanker",
    "RealizedGARCHVol",
    "ReversalRanker",
    "RidgeRanker",
    "RollingVol",
    "SDFElasticNetRanker",
    "SDFRidgeRanker",
    "SingleStateRegime",
    "SkewTDistribution",
    "StackedDistribution",
    "TSMOMRanker",
    "ThreePassFilterRanker",
    "TreeQuantileDistribution",
    "TreeVol",
    "VMERanker",
    "VolThresholdRegime",
    "XGBRegRanker",
    "artifact_identity",
    "attach_artifact_identity",
    "brier_score",
    "canonical_frame_fingerprint",
    "canonical_json_bytes",
    "configure_tracking",
    "crps_from_quantiles",
    "crps_gaussian",
    "date_ic_series",
    "datetime",
    "design_frame",
    "design_matrix",
    "garch_name_walk_forward",
    "git_revision",
    "git_worktree_sha256",
    "group_sizes",
    "hash_bytes",
    "icir",
    "load_joblib_artifact",
    "log_run",
    "losses_from_returns",
    "make_combo_ic_st",
    "make_fm_st",
    "make_ridge_neut",
    "make_ridge_st",
    "mean_pinball",
    "name_level_one_step_density_summary",
    "name_level_qlike",
    "np",
    "one_step_density_summary",
    "overlap_aware_qlike",
    "panel",
    "parkinson_daily_variance",
    "pl",
    "purge_mask",
    "qlike",
    "quantile_crossing_rate",
    "re",
    "run_linucb_panel",
    "run_policy_gradient_panel",
    "run_thompson_panel",
    "save_joblib_artifact",
    "set_global_seed",
    "timestamp_ns",
    "train_alpha",
    "train_calibration",
    "train_calibration_auto",
    "train_distribution",
    "train_distribution_auto",
    "train_family",
    "train_policy_gradient",
    "train_quantile_bandit",
    "train_ranking",
    "train_ranking_auto",
    "train_regime",
    "train_reinforcement",
    "train_reinforcement_auto",
    "train_robinhood_plus",
    "train_tail",
    "train_volatility",
    "train_volatility_auto",
    "walk_forward",
    "write_evidence_report",
]

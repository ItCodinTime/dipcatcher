from datetime import UTC, datetime
from pathlib import Path

import pytest

from quant_fund.config import load_config
from quant_fund.config.models import (
    AppConfig,
    CalendarConfig,
    CostConfig,
    DataConfig,
    ExecutionConfig,
    FeatureConfig,
    FusionConfig,
    HorizonConfig,
    MonitoringConfig,
    OptimizerConfig,
    PaperConfig,
    PortfolioConstraints,
    PromotionConfig,
    RiskGateConfig,
    RuntimeMode,
    TrainConfig,
    UniverseConfig,
    ValidationConfig,
)
from quant_fund.schemas.errors import PointInTimeError
from quant_fund.schemas.pit import assert_pit_safe


def test_load_research_yaml() -> None:
    cfg = load_config("configs/research.yaml")
    assert cfg.runtime.mode is RuntimeMode.RESEARCH
    assert cfg.runtime.allow_live is False
    assert cfg.embargo_bars() >= 1
    assert cfg.northset.n_book_levels == 5
    assert cfg.northset.n_session_candles == 8
    assert cfg.robinhood_plus.enabled is True
    assert cfg.robinhood_plus.backend.value == "numpy"
    assert cfg.robinhood_plus.blend_weight == 0.0


def test_config_inheritance_cannot_escape_root(tmp_path: Path) -> None:
    config = tmp_path / "research.yaml"
    config.write_text("inherit: ../outside.yaml\n")
    (tmp_path.parent / "outside.yaml").write_text("runtime: {}\n")
    with pytest.raises(ValueError, match="escapes config root"):
        load_config(config)


def test_live_requires_flag() -> None:
    try:
        AppConfig.model_validate({"runtime": {"mode": "live", "allow_live": False}})
        raise AssertionError("should have failed")
    except Exception:
        pass


def test_pit_invariant() -> None:
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    t1 = datetime(2020, 1, 3, tzinfo=UTC)
    assert_pit_safe(t0, t1)
    try:
        assert_pit_safe(t1, t0)
        raise AssertionError("lookahead should fail")
    except PointInTimeError:
        pass


def test_cost_config_rejects_invalid_assumptions() -> None:
    for payload in ({"commission_bps": -1.0}, {"participation_limit": 0.0}):
        try:
            CostConfig.model_validate(payload)
            raise AssertionError("invalid cost configuration should fail")
        except ValueError:
            pass


def test_cost_config_rejects_nonfinite_costs() -> None:
    for name in (
        "commission_bps",
        "half_spread_bps",
        "impact_y",
        "bps_per_turnover",
        "borrow_bps_per_year",
        "financing_bps_per_year",
    ):
        with pytest.raises(ValueError, match="finite"):
            CostConfig.model_validate({name: float("nan")})
        with pytest.raises(ValueError, match="finite"):
            CostConfig.model_validate({name: float("inf")})


def test_portfolio_constraints_reject_invalid_bounds() -> None:
    for payload in (
        {"name_min": 0.1},
        {"name_min": float("nan")},
        {"name_max": float("nan")},
        {"max_adv_participation": float("nan")},
        {"cvar_alpha": 1.0},
        {"max_positions": 0},
    ):
        try:
            PortfolioConstraints.model_validate(payload)
            raise AssertionError("invalid portfolio constraint should fail")
        except ValueError:
            pass


def test_execution_config_rejects_invalid_bounds() -> None:
    for payload in ({"n_slices": 0}, {"participation_rate": 1.1}, {"temporary_impact": -1.0}):
        try:
            ExecutionConfig.model_validate(payload)
            raise AssertionError("invalid execution config should fail")
        except ValueError:
            pass


def test_risk_gate_config_rejects_invalid_limits() -> None:
    for payload in (
        {"max_order_notional": float("nan")},
        {"max_gross": -1.0},
        {"max_participation": 0.0},
        {"max_participation": float("inf")},
        {"stale_price_bars": -1},
    ):
        with pytest.raises(ValueError):
            RiskGateConfig.model_validate(payload)


def test_promotion_config_rejects_invalid_thresholds() -> None:
    for payload in (
        {"min_mean_ic": float("nan")},
        {"min_cost_adjusted_spread": float("inf")},
        {"max_turnover": -1.0},
        {"min_folds": 0},
        {"min_fold_ic_stability": 1.1},
    ):
        with pytest.raises(ValueError):
            PromotionConfig.model_validate(payload)


def test_fusion_config_rejects_invalid_weights() -> None:
    for payload in (
        {"alpha_weight": float("nan")},
        {"tail_penalty": -1.0},
        {"risk_weight": 0.0},
        {"risk_weight": float("inf")},
        {"alpha_scale": -0.1},
    ):
        with pytest.raises(ValueError):
            FusionConfig.model_validate(payload)


def test_monitoring_config_rejects_invalid_windows() -> None:
    for payload in (
        {"psi_alert": float("nan")},
        {"psi_alert": -0.1},
        {"reference_bars": 0},
        {"live_bars": -1},
    ):
        with pytest.raises(ValueError):
            MonitoringConfig.model_validate(payload)


def test_train_config_rejects_invalid_hyperparameters() -> None:
    for payload in (
        {"xgb_n_estimators": 0},
        {"n_hmm_states": -1},
        {"ranking_target": " "},
        {"ridge_alpha": float("nan")},
        {"elasticnet_l1": 1.1},
        {"qlike_floor": 0.0},
        {"rff_n_features": 5},
        {"ipca_n_factors": 0},
        {"tprf_n_factors": 0},
        {"gbrt_learning_rate": 0.0},
    ):
        with pytest.raises(ValueError):
            TrainConfig.model_validate(payload)


def test_paper_config_rejects_invalid_simulation_settings() -> None:
    for payload in (
        {"initial_nav": 0.0},
        {"max_steps": 0},
        {"shadow_scale": float("nan")},
        {"challenger_scales": [-0.1]},
        {"promote_max_mean_l1": -1.0},
        {"promote_min_steps": 0},
        {"ledger_subdir": " "},
        {"ledger_subdir": "../escape"},
        {
            "ledger_subdir": str(
                Path("C:/tmp/paper") if Path("C:/tmp/paper").is_absolute() else Path("/tmp/paper")
            )
        },
    ):
        with pytest.raises(ValueError):
            PaperConfig.model_validate(payload)


def test_feature_config_rejects_invalid_transforms() -> None:
    for payload in (
        {"winsor_p": 0.5},
        {"ewma_lambda": 1.0},
        {"amihud_lookback": 0},
        {"families": ["returns", "returns"]},
    ):
        with pytest.raises(ValueError):
            FeatureConfig.model_validate(payload)


def test_calendar_config_rejects_invalid_weekend() -> None:
    for payload in (
        {"name": " "},
        {"weekend": (5, 5)},
        {"weekend": (-1, 6)},
        {"weekend": (5, 7)},
    ):
        with pytest.raises(ValueError):
            CalendarConfig.model_validate(payload)


def test_quantile_config_rejects_nonfinite_and_duplicate_levels() -> None:
    from quant_fund.config.models import QuantileConfig

    with pytest.raises(ValueError, match="quantiles must be in"):
        QuantileConfig.model_validate({"levels": [0.1, float("nan")]})
    with pytest.raises(ValueError, match="strictly increasing"):
        QuantileConfig.model_validate({"levels": [0.1, 0.1]})


def test_optimizer_config_rejects_invalid_objectives() -> None:
    for payload in ({"lambda_risk": -1.0}, {"mode": "unknown"}, {"solver": " "}):
        try:
            OptimizerConfig.model_validate(payload)
            raise AssertionError("invalid optimizer config should fail")
        except ValueError:
            pass


def test_optimizer_config_named_covariance_paths() -> None:
    assert OptimizerConfig().covariance == "ledoit_wolf"
    assert (
        OptimizerConfig.model_validate({"covariance": "dcc_gaussian"}).covariance == "dcc_gaussian"
    )
    assert (
        OptimizerConfig.model_validate({"covariance": "dcc_student_t"}).covariance
        == "dcc_student_t"
    )
    assert OptimizerConfig.model_validate({"covariance": "adcc"}).covariance == "adcc"
    assert OptimizerConfig.model_validate({"covariance": "ccc"}).covariance == "ccc"
    assert OptimizerConfig.model_validate({"covariance": "bollerslev_1990_ccc"}).covariance == "ccc"
    assert OptimizerConfig.model_validate({"covariance": "agdcc"}).covariance == "agdcc"
    assert OptimizerConfig.model_validate({"covariance": "ag_dcc"}).covariance == "agdcc"
    assert OptimizerConfig.model_validate({"covariance": "diagonal_agdcc"}).covariance == "agdcc"
    assert OptimizerConfig.model_validate({"covariance": "agdcc_full"}).covariance == "agdcc_full"
    assert OptimizerConfig.model_validate({"covariance": "full_agdcc"}).covariance == "agdcc_full"
    assert (
        OptimizerConfig.model_validate(
            {"covariance": "cappiello_engle_sheppard_2006_full_agdcc"}
        ).covariance
        == "agdcc_full"
    )
    assert OptimizerConfig.model_validate({"covariance": "ewma"}).covariance == "ewma"
    assert OptimizerConfig.model_validate({"covariance": "oas"}).covariance == "oas"
    assert (
        OptimizerConfig.model_validate({"covariance": "chen_wiesel_eldar_hero_2010"}).covariance
        == "oas"
    )
    assert (
        OptimizerConfig.model_validate({"covariance": "ledoit_wolf_nonlinear"}).covariance
        == "ledoit_wolf_nonlinear"
    )
    assert (
        OptimizerConfig.model_validate({"covariance": "nlshrink"}).covariance
        == "ledoit_wolf_nonlinear"
    )
    assert (
        OptimizerConfig.model_validate({"covariance": "ledoit_wolf_2020_analytical"}).covariance
        == "ledoit_wolf_nonlinear"
    )
    assert OptimizerConfig.model_validate({"covariance": "sample"}).covariance == "sample"
    assert OptimizerConfig.model_validate({"covariance": "unbiased_sample"}).covariance == "sample"
    with pytest.raises(ValueError, match="unknown_optimizer_covariance"):
        OptimizerConfig.model_validate({"covariance": "t"})
    with pytest.raises(ValueError, match="unwired_optimizer_covariance:factor"):
        OptimizerConfig.model_validate({"covariance": "factor"})
    with pytest.raises(ValueError, match="unknown_dcc_spec:dcc"):
        OptimizerConfig.model_validate({"covariance": "dcc"})
    with pytest.raises(ValueError, match="unknown_dcc_spec:shrinkage"):
        OptimizerConfig.model_validate({"covariance": "shrinkage"})
    with pytest.raises(ValueError, match="analytical 2020, not QuEST 2017"):
        OptimizerConfig.model_validate({"covariance": "ledoit_wolf_2017"})


def test_validation_config_rejects_invalid_protocol() -> None:
    for payload in ({"scheme": "random"}, {"test_groups": 6, "n_groups": 6}, {"embargo_bars": -1}):
        try:
            ValidationConfig.model_validate(payload)
            raise AssertionError("invalid validation config should fail")
        except ValueError:
            pass


def test_universe_and_horizon_configs_reject_invalid_filters() -> None:
    with pytest.raises(ValueError):
        UniverseConfig.model_validate({"min_price": 0})
    with pytest.raises(ValueError):
        HorizonConfig.model_validate({"bars": [1, 1], "names": ["1d", "dup"]})


def test_data_config_rejects_invalid_source_dimensions() -> None:
    with pytest.raises(ValueError):
        DataConfig.model_validate({"source": " "})
    with pytest.raises(ValueError):
        DataConfig.model_validate({"source": "unsupported_vendor"})
    assert DataConfig.model_validate({"source": " PARQUET "}).source == "parquet"
    with pytest.raises(ValueError):
        DataConfig.model_validate({"synthetic_n_days": 0})
    with pytest.raises(ValueError):
        DataConfig.model_validate({"synthetic_oracle_beta": float("nan")})
    with pytest.raises(ValueError):
        DataConfig.model_validate({"synthetic_oracle_phi": 1.0})


def test_allow_live_without_live_mode_fail_closed() -> None:
    """allow_live=true in research/paper/etc must fail closed (no live broker)."""
    for mode in ("research", "backtest", "paper", "shadow"):
        with pytest.raises(ValueError, match="allow_live requires runtime.mode: live"):
            AppConfig.model_validate({"runtime": {"mode": mode, "allow_live": True}})


def test_allow_live_with_live_mode_fail_closed_no_broker() -> None:
    """Even mode=live + allow_live=true fails: no live broker adapter in repo."""
    with pytest.raises(ValueError, match="no live broker adapter"):
        AppConfig.model_validate({"runtime": {"mode": "live", "allow_live": True}})


def test_config_rejects_unknown_keys_at_all_nesting_levels() -> None:
    with pytest.raises(ValueError, match="extra_forbidden"):
        AppConfig.model_validate({"runtime": {"allow_livee": False}})
    with pytest.raises(ValueError, match="extra_forbidden"):
        AppConfig.model_validate({"risk_gate": {"max_gross_typo": 1.0}})


def test_load_production_yaml_stays_research_not_live() -> None:
    cfg = load_config("configs/production.yaml")
    assert cfg.runtime.mode is RuntimeMode.RESEARCH
    assert cfg.runtime.allow_live is False

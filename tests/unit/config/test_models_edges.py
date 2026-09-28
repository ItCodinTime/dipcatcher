"""Config-model validator matrices: every model-level guard rejects bad inputs."""

from __future__ import annotations

import pytest

from quant_fund.config.models import (
    CalendarConfig,
    DataConfig,
    FeatureConfig,
    FusionConfig,
    GarchVolSpec,
    HorizonConfig,
    NorthsetConfig,
    OptimizerConfig,
    PerpConfig,
    PortfolioConstraints,
    PromotionConfig,
    QuantileConfig,
    RiskGateConfig,
    TrainConfig,
    UniverseConfig,
    ValidationConfig,
)

pytestmark = pytest.mark.synthetic


class TestUniverseConfig:
    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"min_price": 0.0}, "min_price"),
            ({"min_price": float("nan")}, "min_price"),
            ({"min_adv": -1.0}, "min_adv"),
            ({"min_adv": float("nan")}, "min_adv"),
            ({"min_history_bars": 0}, "min_history_bars"),
            ({"top_n_adv": 0}, "top_n_adv"),
            ({"exchanges": []}, "exchanges"),
            ({"exchanges": [" "]}, "exchanges"),
            ({"security_types": []}, "security_types"),
            ({"security_types": [" "]}, "security_types"),
        ],
    )
    def test_guards(self, overrides, match) -> None:
        with pytest.raises(ValueError, match=match):
            UniverseConfig(**overrides)

    def test_defaults_ok(self) -> None:
        assert UniverseConfig().top_n_adv == 50


class TestCalendarConfig:
    @pytest.mark.parametrize(
        ("weekend", "match"),
        [
            ((5,), "weekend"),
            ((5, 5), "weekend"),
            ((5, 7), "weekend"),
            ((-1, 5), "weekend"),
        ],
    )
    def test_weekend(self, weekend, match) -> None:
        with pytest.raises(ValueError, match=match):
            CalendarConfig(weekend=weekend)

    def test_empty_name(self) -> None:
        with pytest.raises(ValueError, match="calendar name"):
            CalendarConfig(name=" ")


class TestHorizonConfig:
    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"bars": [1, 5], "names": ["1d"]}, "equal length"),
            ({"bars": [], "names": []}, "positive and unique"),
            ({"bars": [0, 5], "names": ["a", "b"]}, "positive and unique"),
            ({"bars": [5, 5], "names": ["a", "b"]}, "positive and unique"),
            ({"names": ["1d", " ", "x"]}, "non-empty and unique"),
            ({"names": ["a", "a", "c"]}, "non-empty and unique"),
        ],
    )
    def test_guards(self, overrides, match) -> None:
        with pytest.raises(ValueError, match=match):
            HorizonConfig(**overrides)


class TestQuantileConfig:
    @pytest.mark.parametrize(
        "levels",
        [
            [0.0, 0.5],
            [0.5, 1.0],
            [0.5, 0.5],
            [0.9, 0.1],
            [float("nan"), 0.5],
        ],
    )
    def test_levels(self, levels) -> None:
        with pytest.raises(ValueError):
            QuantileConfig(levels=levels)


class TestPortfolioConstraints:
    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"gross_leverage": -1.0}, "non-negative"),
            ({"net_exposure": float("nan")}, "non-negative"),
            ({"name_max": -0.01, "name_min": -0.5}, "bracket zero"),
            ({"name_min": 0.01, "name_max": 0.5}, "bracket zero"),
            ({"cash_buffer": 1.5}, "cash_buffer"),
            ({"cvar_alpha": 1.0}, "cvar_alpha"),
            ({"cvar_alpha": 0.0}, "cvar_alpha"),
            ({"max_positions": 0}, "max_positions"),
            ({"cvar_limit": -1.0}, "cvar_limit"),
            ({"predicted_vol_max": -1.0}, "predicted_vol_max"),
            ({"max_adv_participation": -0.5}, "max_adv_participation"),
            ({"max_adv_participation": 1.5}, "max_adv_participation must be <= 1"),
        ],
    )
    def test_guards(self, overrides, match) -> None:
        with pytest.raises(ValueError, match=match):
            PortfolioConstraints(**overrides)


class TestOptimizerConfig:
    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"covariance": ""}, "non-empty string"),
            ({"covariance": "made_up"}, "covariance"),
            ({"lambda_risk": -1.0}, "non-negative"),
            ({"mode": "nonsense"}, "mean_variance or cvar"),
            ({"solver": " "}, "non-empty"),
        ],
    )
    def test_guards(self, overrides, match) -> None:
        with pytest.raises(ValueError, match=match):
            OptimizerConfig(**overrides)

    def test_defaults(self) -> None:
        assert OptimizerConfig().covariance == "ledoit_wolf"


class TestValidationConfig:
    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"scheme": "bogus"}, "expanding or rolling"),
            ({"train_bars": 0}, "positive"),
            ({"val_bars": 0}, "positive"),
            ({"test_bars": 0}, "positive"),
            ({"n_optuna_trials": 0}, "positive"),
            ({"n_groups": 0}, "positive"),
            ({"n_groups": 2, "test_groups": 2}, "less than n_groups"),
            ({"embargo_bars": -1}, "embargo_bars"),
        ],
    )
    def test_guards(self, overrides, match) -> None:
        with pytest.raises(ValueError, match=match):
            ValidationConfig(**overrides)


class TestFeatureConfig:
    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"winsor_p": 0.5}, "winsor_p"),
            ({"winsor_p": -0.01}, "winsor_p"),
            ({"ewma_lambda": 1.0}, "ewma_lambda"),
            ({"ewma_lambda": -0.1}, "ewma_lambda"),
            ({"amihud_lookback": 0}, "lookbacks"),
            ({"adv_lookback": 0}, "lookbacks"),
            ({"families": []}, "non-empty"),
            ({"families": ["x", " "]}, "non-empty"),
            ({"families": ["a", "a"]}, "unique"),
        ],
    )
    def test_guards(self, overrides, match) -> None:
        with pytest.raises(ValueError, match=match):
            FeatureConfig(**overrides)


class TestDataConfig:
    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"source": " "}, "unsupported data source"),
            ({"synthetic_n_assets": 0}, "synthetic"),
            ({"synthetic_n_days": 0}, "synthetic"),
            ({"synthetic_oracle_beta": float("nan")}, "synthetic_oracle_beta"),
            ({"synthetic_oracle_phi": 1.0}, "synthetic_oracle_phi"),
            ({"synthetic_oracle_phi": -1.0}, "synthetic_oracle_phi"),
            ({"benchmark_id": " "}, "benchmark_id"),
            ({"source_symbol": " "}, "source_symbol"),
            ({"source_interval": " "}, "source_interval"),
            ({"source_limit": 0}, "source_limit"),
            ({"source_limit": 1001}, "source_limit"),
        ],
    )
    def test_guards(self, overrides, match) -> None:
        with pytest.raises(ValueError, match=match):
            DataConfig(**overrides)


class TestPerpConfig:
    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"periods_per_year_override": -1.0}, "periods_per_year_override"),
            ({"bar_seconds_hint": 0.0}, "bar_seconds_hint"),
            ({"funding_spike_multiplier": -1.0}, "non-negative"),
            ({"max_leverage": 0.0}, "max_leverage"),
            ({"maint_margin_ratio": 1.0}, "maint_margin_ratio"),
            ({"maint_margin_ratio": 0.0}, "maint_margin_ratio"),
            ({"fill_delay_bars": -1}, "fill_delay_bars"),
        ],
    )
    def test_guards(self, overrides, match) -> None:
        with pytest.raises(ValueError, match=match):
            PerpConfig(**overrides)


class TestNorthsetConfig:
    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"n_book_levels": 0}, "n_book_levels"),
            ({"n_session_candles": 1}, "n_session_candles"),
            ({"base_spread_bps": -1.0}, "base_spread_bps"),
            ({"min_names": 1}, "min_names"),
            ({"session_l2_identity_floor": 1.5}, "identity_floor"),
            ({"sweep_lookback": 1}, "sweep_lookback"),
            ({"sweep_horizons": [0, 5]}, "sweep_horizons"),
            ({"sweep_horizons": [5, 5]}, "sweep_horizons"),
            ({"sweep_vol_lookback": 2}, "sweep_vol_lookback"),
            ({"sweep_n_boot": 10}, "sweep_n_boot"),
            ({"sweep_n_permutations": 10}, "sweep_n_permutations"),
            ({"sweep_n_folds": 1}, "sweep_n_folds"),
            ({"sweep_min_events": 5}, "sweep_min_events"),
            ({"sweep_min_dates": 2}, "sweep_min_dates"),
            ({"sweep_min_fold_positive_fraction": 1.5}, "positive_fraction"),
            ({"sweep_cooldown_bars": -1}, "cooldown"),
            ({"sweep_sensitivity_lookbacks": [1]}, "sensitivity"),
            ({"sweep_sensitivity_lookbacks": [3, 3]}, "sensitivity"),
            ({"book_max_age_seconds": 0.0}, "book_max_age"),
            ({"book_join_coverage_floor": -0.1}, "coverage_floor"),
        ],
    )
    def test_guards(self, overrides, match) -> None:
        with pytest.raises(ValueError, match=match):
            NorthsetConfig(**overrides)


class TestTrainConfig:
    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"ranking_target": " "}, "non-empty"),
            ({"distribution_target": " "}, "non-empty"),
            ({"volatility_target": " "}, "non-empty"),
            ({"xgb_n_estimators": 0}, "must be positive"),
            ({"xgb_max_depth": 0}, "must be positive"),
            ({"lgbm_n_estimators": 0}, "must be positive"),
            ({"n_hmm_states": 0}, "must be positive"),
            ({"garch_p": -1}, "must be positive"),
            ({"ipca_n_factors": 0}, "must be positive"),
            ({"gbrt_n_estimators": 0}, "must be positive"),
            ({"ridge_alpha": -1.0}, "non-negative"),
            ({"elasticnet_l1": 1.5}, "elasticnet_l1"),
            ({"elasticnet_l1": -0.5}, "elasticnet_l1"),
            ({"qlike_floor": 0.0}, "qlike_floor"),
            ({"psd_eigen_tol": 0.0}, "qlike_floor"),
            ({"auto_min_oos_rows": 0}, "auto_min_oos_rows"),
            ({"rp_pca_n_factors": 0}, "must be positive"),
            ({"fnw_n_intervals": 1}, "fnw_n_intervals"),
            ({"rp_pca_gamma": -2.0}, "rp_pca_gamma"),
            ({"gbrt_learning_rate": 0.0}, "gbrt_learning_rate"),
            (
                {
                    "garch_vol": GarchVolSpec.FIGARCH,
                    "garch_p": 2,
                    "garch_q": 1,
                },
                "FIGARCH",
            ),
        ],
    )
    def test_guards(self, overrides, match) -> None:
        with pytest.raises(ValueError, match=match):
            TrainConfig(**overrides)


class TestRiskGateConfig:
    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"max_order_notional": -1.0}, "max_order_notional"),
            ({"max_gross": -1.0}, "max_gross"),
            ({"max_predicted_vol": float("nan")}, "max_predicted_vol"),
            ({"max_participation": 0.0}, "max_participation"),
            ({"max_participation": 1.5}, "max_participation"),
            ({"stale_price_bars": -1}, "stale_price_bars"),
        ],
    )
    def test_guards(self, overrides, match) -> None:
        with pytest.raises(ValueError, match=match):
            RiskGateConfig(**overrides)


class TestPromotionConfig:
    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"min_mean_ic": float("nan")}, "min_mean_ic"),
            ({"max_turnover": -1.0}, "max_turnover"),
            ({"min_folds": 0}, "min_folds"),
            ({"min_fold_ic_stability": 1.5}, "min_fold_ic_stability"),
            ({"min_fold_ic_stability": -0.1}, "min_fold_ic_stability"),
        ],
    )
    def test_guards(self, overrides, match) -> None:
        with pytest.raises(ValueError, match=match):
            PromotionConfig(**overrides)


class TestFusionConfig:
    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"alpha_weight": -1.0}, "non-negative"),
            ({"confidence_weight": float("nan")}, "non-negative"),
            ({"risk_weight": 0.0}, "risk_weight"),
            ({"probability_calibrator": "bogus"}, "isotonic, or platt"),
            ({"probability_calibration_max_age_days": -1}, "max_age_days"),
        ],
    )
    def test_guards(self, overrides, match) -> None:
        with pytest.raises(ValueError, match=match):
            FusionConfig(**overrides)

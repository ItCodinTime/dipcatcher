import pytest

from quant_fund.config.loader import load_config
from quant_fund.config.models import AppConfig, PortfolioConstraints, RuntimeMode


def test_default_sleeve_matches_gross() -> None:
    c = PortfolioConstraints()
    assert c.long_max + c.short_max >= c.gross_leverage - 1e-12
    assert c.gross_redundant is False
    assert c.effective_gross_cap == c.gross_leverage


def test_base_yaml_sleeve_not_dead() -> None:
    cfg = load_config("configs/base.yaml")
    assert cfg.constraints.long_max + cfg.constraints.short_max >= cfg.constraints.gross_leverage
    assert cfg.constraints.gross_redundant is False


def test_production_yaml_is_research_not_live() -> None:
    cfg = load_config("configs/production.yaml")
    assert cfg.runtime.allow_live is False
    assert cfg.runtime.mode.value == "research"


def test_appconfig_defaults_consistent() -> None:
    cfg = AppConfig()
    assert cfg.constraints.long_max + cfg.constraints.short_max >= cfg.constraints.gross_leverage


def test_allow_live_true_fail_closed_across_modes() -> None:
    """Property: allow_live cannot be enabled without a live broker (none ships)."""
    for mode in RuntimeMode:
        payload = {"runtime": {"mode": mode.value, "allow_live": True}}
        with pytest.raises(ValueError):
            AppConfig.model_validate(payload)


def test_shipped_yamls_never_enable_allow_live() -> None:
    for name in ("base.yaml", "research.yaml", "backtest.yaml", "paper.yaml", "production.yaml"):
        cfg = load_config(f"configs/{name}")
        assert cfg.runtime.allow_live is False
        assert cfg.runtime.mode.value != "live"

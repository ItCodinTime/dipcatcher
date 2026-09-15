from datetime import UTC, datetime

from quant_fund.config import load_config
from quant_fund.config.models import AppConfig, RuntimeMode
from quant_fund.schemas.errors import PointInTimeError
from quant_fund.schemas.pit import assert_pit_safe


def test_load_research_yaml() -> None:
    cfg = load_config("configs/research.yaml")
    assert cfg.runtime.mode is RuntimeMode.RESEARCH
    assert cfg.runtime.allow_live is False
    assert cfg.embargo_bars() >= 1


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

"""Report files stay research-only and the smoke command fails closed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from quant_fund.config.models import AppConfig
from quant_fund.research.catalog.constants import FORBIDDEN_RESEARCH_METRIC_KEYS

from quant_fund.parity.__main__ import main, smoke
from quant_fund.parity.reference import run_backtest_session
from quant_fund.parity.replay import ReplayOptions
from quant_fund.parity.report import assert_clean_report, build_report, write_report
from quant_fund.parity.session import MarketSession
from quant_fund.parity.shadow import run_shadow_session
from quant_fund.parity.strategy import FixedWeightStrategy


def _walk_keys(value: object, found: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            found.add(str(key))
            _walk_keys(item, found)
    elif isinstance(value, list):
        for item in value:
            _walk_keys(item, found)


def test_written_report_has_no_forbidden_headline_keys(
    session: MarketSession, config: AppConfig, tmp_path: Path
) -> None:
    strategy = FixedWeightStrategy({"A": 0.05, "B": -0.02})
    backtest = run_backtest_session(session, strategy, config)
    shadow = run_shadow_session(session, strategy, config)
    report = build_report(backtest, shadow, backtest_strategy=strategy, shadow_strategy=strategy)
    keys: set[str] = set()
    _walk_keys(report, keys)
    assert keys.isdisjoint(FORBIDDEN_RESEARCH_METRIC_KEYS)
    paths = write_report(tmp_path, report)
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    assert summary["live_pnl_claim"] is False
    assert summary["research_only"] is True
    assert summary["would_promote_live"] is False
    assert summary["data_source"] == "SYNTHETIC"
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "live_pnl_claim" in markdown
    assert "delay" in markdown
    assert "code path" in markdown.lower()
    assert_clean_report(report)


def test_smoke_command_writes_a_clean_report(tmp_path: Path) -> None:
    smoke(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["decisions_match"] is True
    assert summary["n_divergent"] == 0
    assert summary["live_pnl_claim"] is False
    main(["smoke", "--out", str(tmp_path / "again")])
    assert (tmp_path / "again" / "report.md").is_file()


def test_assert_clean_report_rejects_a_divergence(
    session: MarketSession, config: AppConfig
) -> None:
    strategy = FixedWeightStrategy({"A": 0.05, "B": -0.02})
    backtest = run_backtest_session(session, strategy, config)
    shadow = run_shadow_session(session, strategy, config, options=ReplayOptions(lot_size=0.03))
    report = build_report(backtest, shadow, backtest_strategy=strategy, shadow_strategy=strategy)
    with pytest.raises(RuntimeError, match="diverged"):
        assert_clean_report(report)

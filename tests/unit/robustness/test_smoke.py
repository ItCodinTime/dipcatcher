"""Coverage: robustness smoke CI check (SYNTHETIC, no receipts)."""

from __future__ import annotations

import json

import pytest

from quant_fund.robustness import smoke


def test_run_smoke_report_shape() -> None:
    report = smoke.run_smoke()
    assert report["evidence_class"] == "SYNTHETIC"
    assert report["research_only"] is True
    assert report["live_trading_claim"] is False
    assert report["linear"]["certified_status"] == "proven"
    assert report["linear"]["worst_case_mean"] == pytest.approx(0.3, abs=1e-12)
    assert report["schedule_jitter_bars"] == 1.0
    # report is JSON-ready
    json.dumps(report)


def test_main_prints_report(capsys) -> None:
    assert smoke.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["evidence_class"] == "SYNTHETIC"

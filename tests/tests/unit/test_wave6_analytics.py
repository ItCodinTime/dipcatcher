"""Wave 6: underwater / drawdown duration + stress_report + schema export."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.backtest.engine import export_backtest_metrics_json, run_backtest
from quant_fund.config.loader import load_config
from quant_fund.metrics.analytics import (
    ANALYTICS_SCHEMA_KEYS,
    book_diagnostics,
    drawdown_duration_stats,
    equity_curve_analytics,
    export_analytics_dict,
    stress_report,
    underwater_periods,
)


def test_underwater_periods_known_path() -> None:
    # Peak at 0, then drop, recover, drop again
    # wealth via drawdowns directly: 0, -0.1, -0.2, -0.05, 0, -0.15, -0.15
    dd = np.array([0.0, -0.1, -0.2, -0.05, 0.0, -0.15, -0.15])
    eps = underwater_periods(dd)
    assert len(eps) == 2
    assert eps[0]["start_idx"] == 1
    assert eps[0]["end_idx"] == 3
    assert eps[0]["duration"] == 3
    assert eps[0]["trough"] == pytest.approx(-0.2)
    assert eps[0]["recovered"] is True
    assert eps[1]["duration"] == 2
    assert eps[1]["recovered"] is False  # open at end


def test_drawdown_duration_stats_from_nav() -> None:
    # NAV: 100, 110, 90, 95, 120 → underwater between peak 110 and recovery to new peak
    nav = np.array([100.0, 110.0, 90.0, 95.0, 120.0])
    out = drawdown_duration_stats(nav=nav)
    assert out["n_episodes"] >= 1
    assert out["max_underwater_duration"] >= 1
    assert out["role"] == 1.0
    assert out["live_pnl_claim"] is False
    eq = equity_curve_analytics(nav)
    assert eq["n_underwater_episodes"] == out["n_episodes"]
    assert eq["max_underwater_duration"] == out["max_underwater_duration"]


def test_stress_report_contributions_and_ranking() -> None:
    w = np.array([0.1, -0.05, 0.05])
    cov = np.eye(3) * 0.0004
    report = stress_report(w, cov, shock_sigma=1.0, liquidity_haircut=0.1)
    assert report["role"] == "stress_report"
    assert report["live_pnl_claim"] is False
    assert report["research_only"] is True
    assert "scenarios" in report
    assert len(report["contributions"]) == 6
    assert report["worst_scenario"] is not None
    assert report["worst_pnl"] is not None
    assert report["worst_pnl"] <= 0 or report["worst_pnl"] != report["worst_pnl"]
    # ranked ascending (most adverse first)
    ranked = report["ranked_adverse"]
    assert ranked
    pnls = [float(c["pnl"]) for c in ranked]
    assert pnls == sorted(pnls)
    assert report["adverse_pnl_sum"] <= 0.0 + 1e-12


def test_book_diagnostics_includes_stress_report_and_dd_duration() -> None:
    rng = np.random.default_rng(3)
    r = rng.normal(0.0, 0.01, size=80)
    # induce a drawdown stretch
    r[20:35] = -0.02
    diag = book_diagnostics(
        r,
        weights=np.array([0.05, -0.03, 0.02]),
        cov=np.eye(3) * 0.0004,
        data_source="SYNTHETIC",
    )
    assert "stress_report" in diag
    assert "drawdown_duration" in diag
    assert diag["stress_report"]["n_scenarios"] == 6
    assert diag["drawdown_duration"]["n_bars"] == 80
    exported = export_analytics_dict(diag)
    for key in ANALYTICS_SCHEMA_KEYS:
        assert key in exported, key
    assert exported["live_pnl_claim"] is False
    assert exported["schema_role"] == "paper_backtest_aligned"


def test_backtest_metrics_export_json_aligned(tmp_path: Path) -> None:
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = True
    cfg.risk_gate.max_name = 0.5
    cfg.risk_gate.max_net = 0.5
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    rows = []
    wrows = []
    for d in range(8):
        t = datetime(2024, 1, 1 + d, tzinfo=UTC)
        for sid, px in [("A", 100.0 + d), ("B", 50.0 + 0.5 * d)]:
            rows.append(
                {
                    "security_id": sid,
                    "event_time": t,
                    "open": px,
                    "close": px,
                    "close_total_return": px,
                    "volume": 1e6,
                    "adv": 1e8,
                    "source": "synthetic",
                }
            )
        wrows.append({"event_time": t, "security_id": "A", "target_weight": 0.1})
        wrows.append({"event_time": t, "security_id": "B", "target_weight": -0.05})
    bars = pl.DataFrame(rows)
    weights = pl.DataFrame(wrows)
    result = run_backtest(bars, weights, cfg, initial_nav=100_000.0)
    assert "analytics" in result.metrics
    assert "analytics_export" in result.metrics
    ae = result.metrics["analytics_export"]
    assert isinstance(ae, dict)
    for key in ("exposure", "execution", "equity", "stress", "stress_report", "drawdown_duration"):
        assert key in ae
    assert ae["live_pnl_claim"] is False
    out = tmp_path / "bt_metrics.json"
    path = export_backtest_metrics_json(result, out)
    assert path.is_file()
    import json

    blob = json.loads(path.read_text())
    assert blob["live_pnl_claim"] is False
    assert blob["research_only"] is True
    assert "aligned_schema_keys" in blob
    # Shared keys present
    shared = set(ANALYTICS_SCHEMA_KEYS) & set(blob.keys())
    assert "stress_report" in shared
    assert "drawdown_duration" in shared

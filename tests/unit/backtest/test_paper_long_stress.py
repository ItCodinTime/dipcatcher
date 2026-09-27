"""Paper loop 100+ step stress + mid-run resume integrity + equity/DD analytics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from quant_fund.config.loader import load_config
from quant_fund.metrics.analytics import equity_curve_analytics
from quant_fund.paper.ledger import load_broker_state
from quant_fund.paper.loop import run_paper_loop


def _bars(n_days: int = 130):
    rows = []
    start = datetime(2024, 1, 1, tzinfo=UTC)
    for d in range(n_days):
        t = start + timedelta(days=d)
        # Mild drift so equity path is non-flat (still SYNTHETIC)
        for sid, px0 in [("A", 100.0 + 0.05 * d), ("B", 50.0 + 0.02 * d)]:
            rows.append(
                {
                    "security_id": sid,
                    "event_time": t,
                    "open": px0,
                    "close": px0,
                    "close_total_return": px0,
                    "volume": 1_000_000.0,
                    "adv": 100_000_000.0,
                    "vol_20": 0.02,
                    "source": "synthetic",
                }
            )
    return pl.DataFrame(rows)


def _weights(n_days: int = 120, scale: float = 1.0):
    rows = []
    start = datetime(2024, 1, 1, tzinfo=UTC)
    for d in range(n_days):
        t = start + timedelta(days=d)
        # Alternate slight tilt so turnover occurs
        tilt = 0.02 if (d % 5 == 0) else 0.0
        rows.append({"event_time": t, "security_id": "A", "target_weight": (0.10 + tilt) * scale})
        rows.append(
            {"event_time": t, "security_id": "B", "target_weight": (-0.05 - tilt / 2) * scale}
        )
    return pl.DataFrame(rows)


def _cfg(tmp_path: Path):
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 0.5
    cfg.risk_gate.max_net = 0.5
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    cfg.costs.frictionless = True
    cfg.paper.promote_min_steps = 10
    cfg.paper.promote_max_mean_l1 = 2.0
    return cfg


def test_paper_100_step_stress_and_resume(tmp_path):
    cfg = _cfg(tmp_path)
    bars = _bars(130)
    w = _weights(120)
    run_id = "paper-wave4-stress"

    first = run_paper_loop(
        bars,
        cfg,
        champion_weights=w,
        shadow_weights=_weights(120, 0.5),
        initial_nav=1_000_000.0,
        max_steps=55,
        run_id=run_id,
        prefer_latest=False,
    )
    assert first.metrics["n_steps"] == 55
    assert first.metrics["live_pnl_claim"] is False
    assert first.metrics["research_only"] is True
    assert "equity_analytics" in first.metrics
    ea = first.metrics["equity_analytics"]
    assert ea["n_points"] == 55
    assert ea["nav_end"] > 0
    assert ea["role"] == 1.0
    assert ea["max_drawdown_nav"] <= 0.0 + 1e-12

    state1 = load_broker_state(tmp_path, run_id)
    assert state1 is not None
    cash1 = float(state1["champion"]["cash"])
    step1 = int(state1["step"])

    second = run_paper_loop(
        bars,
        cfg,
        champion_weights=w,
        shadow_weights=_weights(120, 0.5),
        initial_nav=1_000_000.0,
        max_steps=60,
        resume=True,
        resume_run_id=run_id,
    )
    assert second.metrics["resumed"] is True
    assert second.metrics["n_steps"] >= 100
    assert second.metrics["n_steps_this_run"] == 60
    assert second.metrics["live_pnl_claim"] is False

    ea2 = second.metrics["equity_analytics"]
    assert ea2["n_points"] >= 100
    assert ea2["nav_start"] > 0
    assert ea2["nav_end"] > 0
    assert ea2["max_drawdown_nav"] <= 0.0 + 1e-12
    # Direct helper on ledger equity agrees
    nav = second.champion_equity["nav"].to_numpy()
    direct = equity_curve_analytics(nav)
    assert direct["n_points"] == ea2["n_points"]
    assert abs(direct["nav_end"] - ea2["nav_end"]) < 1e-6

    state2 = load_broker_state(tmp_path, run_id)
    assert state2 is not None
    assert int(state2["step"]) == step1 + 60
    # Cash must not reset to initial_nav on resume
    assert abs(float(state2["champion"]["cash"]) - 1_000_000.0) > 1.0 or cash1 != 1_000_000.0
    assert second.promotion is not None
    assert second.promotion["would_promote_live"] is False


def test_equity_curve_analytics_known_dd():
    import pytest

    # NAV: 100 → 110 → 99 → 105 → peak drawdown from 110 to 99
    nav = [100.0, 110.0, 99.0, 105.0]
    out = equity_curve_analytics(nav)
    assert out["n_points"] == 4
    assert out["nav_start"] == 100.0
    assert out["nav_end"] == 105.0
    assert out["nav_peak"] == 110.0
    assert out["max_drawdown_nav"] == pytest.approx(99.0 / 110.0 - 1.0)
    assert out["role"] == 1.0


def test_paper_200_step_resume_analytics_export(tmp_path):
    """Wave 13: ≥200 steps across resume; analytics_export.json present + valid."""
    import json

    from quant_fund.metrics.analytics import validate_analytics_export

    cfg = _cfg(tmp_path)
    bars = _bars(230)
    w = _weights(220)
    run_id = "paper-wave13-200"

    first = run_paper_loop(
        bars,
        cfg,
        champion_weights=w,
        shadow_weights=_weights(220, 0.5),
        initial_nav=1_000_000.0,
        max_steps=110,
        run_id=run_id,
        prefer_latest=False,
    )
    assert first.metrics["n_steps"] == 110
    assert first.metrics["live_pnl_claim"] is False

    second = run_paper_loop(
        bars,
        cfg,
        champion_weights=w,
        shadow_weights=_weights(220, 0.5),
        initial_nav=1_000_000.0,
        max_steps=100,
        resume=True,
        resume_run_id=run_id,
    )
    assert second.metrics["resumed"] is True
    assert second.metrics["n_steps"] >= 200
    assert second.metrics["n_steps_this_run"] == 100
    assert second.metrics["live_pnl_claim"] is False
    assert second.promotion is not None
    assert second.promotion["would_promote_live"] is False

    ae_path = Path(second.paths["analytics_export"])
    assert ae_path.is_file()
    blob = json.loads(ae_path.read_text())
    report = validate_analytics_export(blob)
    assert report["ok"] is True
    assert report["live_pnl_claim"] is False
    assert second.metrics.get("analytics_export_ok") is True
    ea = second.metrics["equity_analytics"]
    assert ea["n_points"] >= 200
    assert ea["role"] == 1.0

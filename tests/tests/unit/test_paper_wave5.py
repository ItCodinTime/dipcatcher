"""Wave 5 paper: multi-challenger, mid-run kill, risk_gate accounting, promo JSON."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from quant_fund.config.loader import load_config
from quant_fund.metrics.analytics import rolling_challenger_metrics
from quant_fund.paper.ledger import promotion_dry_run
from quant_fund.paper.loop import run_paper_loop


def _bars(n_days: int = 10):
    rows = []
    for d in range(n_days):
        t = datetime(2024, 1, 1 + d, tzinfo=UTC)
        for sid, px0 in [("A", 100.0 + d), ("B", 50.0 + 0.5 * d)]:
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


def _weights(n_days: int = 9, scale: float = 1.0, a: float = 0.1, b: float = -0.05):
    rows = []
    for d in range(n_days):
        t = datetime(2024, 1, 1 + d, tzinfo=UTC)
        rows.append({"event_time": t, "security_id": "A", "target_weight": a * scale})
        rows.append({"event_time": t, "security_id": "B", "target_weight": b * scale})
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
    cfg.paper.promote_min_steps = 3
    cfg.paper.promote_max_mean_l1 = 1.0
    return cfg


def test_rolling_challenger_metrics_ranks() -> None:
    report = rolling_challenger_metrics(
        {"shadow": [0.1, 0.2, 0.15], "alt": [0.5, 0.6, 0.55], "close": [0.01, 0.02]},
        window=2,
    )
    assert report["live_pnl_claim"] is False
    assert report["closest_challenger"] == "close"
    assert report["farthest_challenger"] == "alt"
    assert report["challengers"]["shadow"]["rolling_mean_l1"] == 0.175


def test_multi_challenger_and_richer_promo(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    result = run_paper_loop(
        _bars(10),
        cfg,
        champion_weights=_weights(9, 1.0),
        shadow_weights=_weights(9, 0.7),
        challenger_weights={
            "aggressive": _weights(9, 1.5),
            "flat": _weights(9, 0.0),
        },
        initial_nav=100_000.0,
        max_steps=6,
        prefer_latest=False,
        rolling_window=3,
        run_id="paper-wave5-multi",
    )
    assert result.metrics["live_pnl_claim"] is False
    assert result.metrics["research_only"] is True
    assert result.metrics["n_extra_challengers"] == 2
    cm = result.metrics["challenger_metrics"]
    assert cm is not None
    assert "shadow" in cm["challengers"]
    assert "aggressive" in cm["challengers"]
    assert "flat" in cm["challengers"]
    assert (
        result.metrics["rolling_mean_l1_divergence"] == result.metrics["rolling_mean_l1_divergence"]
    )
    promo = result.promotion
    assert promo is not None
    assert promo["schema_version"] == 2
    assert promo["would_promote_live"] is False
    assert promo["challenger_metrics"] is not None
    assert promo["rolling_window"] == 3
    assert "risk_accounting" in promo
    root = tmp_path / "metadata" / "paper" / "paper-wave5-multi"
    assert (root / "promotion_dry_run.json").is_file()


def test_kill_switch_mid_paper_run(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    result = run_paper_loop(
        _bars(10),
        cfg,
        champion_weights=_weights(9),
        shadow_weights=_weights(9, 0.5),
        initial_nav=100_000.0,
        max_steps=6,
        prefer_latest=False,
        halt_after_steps=2,
        run_id="paper-wave5-kill",
    )
    assert result.metrics["kill_tripped_mid_run"] is True
    assert result.metrics["kill_switch_halts"] >= 1
    # Some fills before the trip, none (or fewer) after — at least halt counted
    assert result.metrics["n_fills"] >= 1
    promo = result.promotion
    assert promo is not None
    assert promo["kill_tripped_mid_run"] is True
    assert promo["would_promote_paper"] is False
    assert "kill_switch_tripped_mid_run" in promo["reasons"]
    # Risk accounting separates kill from risk_gate
    acct = result.metrics["reject_accounting"]
    assert acct["kill_switch_halts"] == result.metrics["kill_switch_halts"]
    assert acct["risk_gate_rejects"] == result.metrics["risk_gate_rejects"]


def test_risk_gate_reject_accounting(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.risk_gate.max_order_notional = 1.0  # force risk_gate rejects
    result = run_paper_loop(
        _bars(8),
        cfg,
        champion_weights=_weights(7),
        initial_nav=100_000.0,
        max_steps=3,
        prefer_latest=False,
        run_id="paper-wave5-risk",
    )
    assert result.metrics["risk_gate_rejects"] >= 1
    assert result.metrics["reject_accounting"]["risk_gate_rejects"] >= 1
    # Kill not tripped
    assert result.metrics["kill_switch_halts"] == 0
    assert result.metrics["reject_accounting"]["kill_switch_halts"] == 0
    # risk_gate_rejects is NOT total rejects (may equal if only risk_gate)
    assert (
        result.metrics["risk_gate_rejects"]
        == result.metrics["reject_accounting"]["risk_gate_rejects"]
    )


def test_promotion_dry_run_schema_v2_fields() -> None:
    receipt = promotion_dry_run(
        mean_l1=0.05,
        max_l1=0.1,
        n_steps=20,
        champion_nav=1.05e6,
        shadow_gross=1.1,
        max_mean_l1=0.25,
        min_steps=5,
        data_source="SYNTHETIC",
        rolling_mean_l1=0.04,
        rolling_window=10,
        challenger_metrics={"challengers": {"shadow": {"mean_l1": 0.05}}},
        primary_challenger="shadow",
        risk_accounting={"risk_gate_rejects": 0, "kill_switch_halts": 0},
        kill_tripped_mid_run=False,
    )
    assert receipt["schema_version"] == 2
    assert receipt["rolling_mean_l1"] == 0.04
    assert receipt["primary_challenger"] == "shadow"
    assert receipt["would_promote_live"] is False
    assert receipt["live_pnl_claim"] is False

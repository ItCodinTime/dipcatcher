"""DayWave9 thin smoke: paper/shadow capital honesty + no live PnL claim.

SYNTHETIC only — research-diagnostic. Does not duplicate Phase 17 e2e;
asserts shadow allow_capital=False cash stays 0 and export honesty flags.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from quant_fund.config.loader import load_config
from quant_fund.metrics.analytics import validate_analytics_export
from quant_fund.paper.ledger import load_broker_state
from quant_fund.paper.loop import run_paper_loop
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent


def _bars(n_days: int = 5) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for d in range(n_days):
        t = datetime(2024, 1, 1 + d, tzinfo=UTC)
        for sid, px0 in (("A", 100.0 + d), ("B", 50.0 + 0.5 * d)):
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


def _weights(n_days: int = 4, scale: float = 1.0) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for d in range(n_days):
        t = datetime(2024, 1, 1 + d, tzinfo=UTC)
        rows.append({"event_time": t, "security_id": "A", "target_weight": 0.1 * scale})
        rows.append({"event_time": t, "security_id": "B", "target_weight": -0.05 * scale})
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
    return cfg


def test_day_paper_shadow_honesty_smoke(tmp_path: Path) -> None:
    """Minimal run_paper_loop: shadow never moves capital; no live PnL claim."""
    cfg = _cfg(tmp_path)
    run_id = "daywave9-paper-shadow-honesty"
    result = run_paper_loop(
        _bars(5),
        cfg,
        champion_weights=_weights(4, 1.0),
        shadow_weights=_weights(4, 0.7),
        initial_nav=100_000.0,
        max_steps=3,
        prefer_latest=False,
        run_id=run_id,
    )

    assert result.source_note == "SYNTHETIC"
    assert result.metrics["has_shadow"] is True
    assert result.metrics["live_pnl_claim"] is False
    assert result.metrics["research_only"] is True
    assert result.metrics.get("live_pnl_claim") is not True
    assert result.metrics["n_fills"] >= 1  # champion may fill

    # Shadow slot: capital-free semantics via persisted broker_state
    state = load_broker_state(tmp_path, run_id)
    assert state is not None
    shadow = state["shadow"]
    assert shadow["slot"] == "shadow"
    assert shadow["allow_capital"] is False
    assert float(shadow["cash"]) == 0.0
    assert int(shadow.get("n_fills", 0)) == 0
    shares = shadow.get("shares") or {}
    # Shadow tracks target weights as notional shares but never cash fills
    assert all(float(v) == float(v) for v in shares.values())  # finite

    ae = result.metrics["analytics_export"]
    assert ae.get("live_pnl_claim") is False
    assert ae.get("live_pnl_claim") is not True
    ae_report = validate_analytics_export(ae)
    assert ae_report["live_pnl_claim"] is False
    assert "live_pnl_claim_must_be_false" not in ae_report.get("errors", [])

    # On-disk export matches in-memory honesty flag
    ae_path = Path(result.paths["analytics_export"])
    disk_ae = json.loads(ae_path.read_text())
    assert disk_ae.get("live_pnl_claim") is False

    promo = result.promotion
    assert promo is not None
    assert promo["live_pnl_claim"] is False
    assert promo["would_promote_live"] is False

    # Research L1 challenger slice: no sharpe/sortino/calmar/nav headline keys.
    # Strip live_pnl_claim itself (token "pnl" is forbidden by design).
    cm = result.metrics["challenger_metrics"]
    assert cm is not None
    assert cm.get("live_pnl_claim") is False
    slim = {k: v for k, v in cm.items() if k != "live_pnl_claim"}
    assert family_blob_forbidden_metrics_absent(slim) is True
    assert "shadow" in cm["challengers"]

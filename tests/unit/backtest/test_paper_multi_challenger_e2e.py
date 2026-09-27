"""Wave 18: multi-challenger E2E via challenger_scales → rolling metrics + promo snapshot."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from quant_fund.config.loader import load_config
from quant_fund.paper.ledger import validate_promotion_dry_run_receipt
from quant_fund.paper.loop import build_scaled_challenger_weights, run_paper_loop


def _weights(n_days: int = 8, a: float = 0.1, b: float = -0.05) -> pl.DataFrame:
    rows = []
    for d in range(n_days):
        t = datetime(2024, 1, 1 + d, tzinfo=UTC)
        rows.append({"event_time": t, "security_id": "A", "target_weight": a})
        rows.append({"event_time": t, "security_id": "B", "target_weight": b})
    return pl.DataFrame(rows)


def _bars(n_days: int = 9) -> pl.DataFrame:
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


def test_challenger_scales_e2e_named_rolling_and_promo_snapshot(tmp_path: Path) -> None:
    """scales → named panels → rolling challenger_metrics + promotion_dry_run snapshot."""
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 0.5
    cfg.risk_gate.max_net = 0.5
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    cfg.costs.frictionless = True
    cfg.paper.promote_min_steps = 2
    cfg.paper.promote_max_mean_l1 = 1.0
    scales = list(cfg.paper.challenger_scales)  # [1.0, 0.5] from paper.yaml
    assert scales == [1.0, 0.5]

    champ = _weights(8)
    shadow = champ.with_columns((pl.col("target_weight") * 0.7).alias("target_weight"))
    extra = build_scaled_challenger_weights(champ, scales)
    assert set(extra) == {"scale_1", "scale_0.5"}

    result = run_paper_loop(
        _bars(9),
        cfg,
        champion_weights=champ,
        shadow_weights=shadow,
        challenger_weights=extra,
        max_steps=6,
        prefer_latest=False,
        rolling_window=3,
        run_id="paper-wave18-scales-e2e",
    )

    assert result.metrics.get("live_pnl_claim") is False
    assert result.metrics.get("research_only") is True
    assert result.metrics["n_extra_challengers"] == 2
    assert result.metrics["has_shadow"] is True

    cm = result.metrics["challenger_metrics"]
    assert cm is not None
    assert cm.get("live_pnl_claim") is False
    names = set(cm["challengers"])
    assert {"shadow", "scale_1", "scale_0.5"} <= names
    # scale_1 matches champion → L1≈0 vs champion; closest among scaled should be scale_1
    assert (
        cm["challengers"]["scale_1"]["mean_l1"] == 0.0
        or cm["challengers"]["scale_1"]["mean_l1"] < cm["challengers"]["scale_0.5"]["mean_l1"]
    )

    promo = result.metrics["promotion_dry_run"]
    assert promo["run_id"] == "paper-wave18-scales-e2e"
    assert promo["would_promote_live"] is False
    assert promo["live_pnl_claim"] is False
    assert promo["research_only"] is True
    assert promo["schema_version"] == 2
    assert promo["primary_challenger"] == "shadow"
    assert promo["challenger_metrics"] is not None
    assert "scale_1" in promo["challenger_metrics"]["challengers"]
    assert "scale_0.5" in promo["challenger_metrics"]["challengers"]
    assert validate_promotion_dry_run_receipt(promo) == []

    root = tmp_path / "metadata" / "paper" / "paper-wave18-scales-e2e"
    promo_path = root / "promotion_dry_run.json"
    assert promo_path.is_file()
    on_disk = json.loads(promo_path.read_text())
    assert on_disk["would_promote_live"] is False
    assert on_disk["live_pnl_claim"] is False
    assert "scale_1" in on_disk["challenger_metrics"]["challengers"]
    assert "scale_0.5" in on_disk["challenger_metrics"]["challengers"]

"""Wave 17: paper.challenger_scales config + CLI helper wiring."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from quant_fund.config.loader import load_config
from quant_fund.config.models import PaperConfig
from quant_fund.paper.loop import build_scaled_challenger_weights, run_paper_loop


def _weights(n_days: int = 6, a: float = 0.1, b: float = -0.05) -> pl.DataFrame:
    rows = []
    for d in range(n_days):
        t = datetime(2024, 1, 1 + d, tzinfo=UTC)
        rows.append({"event_time": t, "security_id": "A", "target_weight": a})
        rows.append({"event_time": t, "security_id": "B", "target_weight": b})
    return pl.DataFrame(rows)


def _bars(n_days: int = 7) -> pl.DataFrame:
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


def test_paper_config_challenger_scales_default_empty() -> None:
    cfg = PaperConfig()
    assert cfg.challenger_scales == []


def test_paper_ledger_subdir_is_used_for_persistence(tmp_path: Path) -> None:
    import json

    from quant_fund.paper.ledger import PaperLedger, load_broker_state

    ledger = PaperLedger(tmp_path, "custom-run", "custom/paper")
    assert ledger.root == tmp_path / "metadata" / "custom" / "paper" / "custom-run"
    (ledger.root / "broker_state.json").write_text(
        json.dumps({"schema_version": 2, "champion": {}, "shadow": None})
    )
    assert load_broker_state(tmp_path, "custom-run", "custom/paper") is not None


def test_paper_ledger_rejects_run_id_path_traversal(tmp_path: Path) -> None:
    from quant_fund.paper.ledger import PaperLedger, latest_run_id, load_broker_state

    with pytest.raises(ValueError, match="path-safe"):
        PaperLedger(tmp_path, "../escape")
    with pytest.raises(ValueError, match="path-safe"):
        load_broker_state(tmp_path, "nested/run")
    latest = tmp_path / "metadata" / "paper"
    latest.mkdir(parents=True, exist_ok=True)
    (latest / "latest_run.json").write_text('{"run_id":"../escape"}')
    with pytest.raises(ValueError, match="path-safe"):
        latest_run_id(tmp_path)


def test_paper_config_rejects_nonfinite_scales() -> None:
    with pytest.raises(ValidationError):
        PaperConfig(challenger_scales=[1.0, float("nan")])
    with pytest.raises(ValidationError):
        PaperConfig(challenger_scales=[float("inf")])


def test_build_scaled_challenger_weights_names_and_scales() -> None:
    champ = _weights(4)
    empty = build_scaled_challenger_weights(champ, [])
    assert empty == {}
    panels = build_scaled_challenger_weights(champ, [1.0, 0.5, 1.0])
    assert set(panels) == {"scale_1", "scale_0.5", "scale_1_2"}
    # scale_1 matches champion weights
    assert panels["scale_1"]["target_weight"].to_list() == champ["target_weight"].to_list()
    # 0.5 is half
    half = panels["scale_0.5"]["target_weight"].to_list()
    expect = [0.5 * x for x in champ["target_weight"].to_list()]
    assert half == pytest.approx(expect)


def test_paper_yaml_loads_challenger_scales() -> None:
    cfg = load_config("configs/paper.yaml")
    assert cfg.paper.challenger_scales == [1.0, 0.5]


def test_multi_challenger_from_scales_in_loop(tmp_path: Path) -> None:
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
    champ = _weights(6)
    shadow = champ.with_columns((pl.col("target_weight") * 0.7).alias("target_weight"))
    extra = build_scaled_challenger_weights(champ, [1.0, 0.5])
    result = run_paper_loop(
        _bars(7),
        cfg,
        champion_weights=champ,
        shadow_weights=shadow,
        challenger_weights=extra,
        max_steps=4,
    )
    assert result.metrics.get("live_pnl_claim") is False
    assert result.metrics["n_extra_challengers"] == 2
    cm = result.metrics["challenger_metrics"]["challengers"]
    assert "shadow" in cm
    assert "scale_1" in cm
    assert "scale_0.5" in cm
    promo = result.metrics["promotion_dry_run"]
    assert promo["would_promote_live"] is False
    assert promo["run_id"] == result.run_id
    assert promo["live_pnl_claim"] is False


def test_paper_promotion_receipt_run_id_mismatch_fails_closed(tmp_path: Path) -> None:
    from quant_fund.paper.ledger import validate_ledger_schema

    run_dir = tmp_path / "metadata" / "paper" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        '{"run_id":"run-a","schema_version":2,"label":"PAPER_SIMULATED",'
        '"disclaimer":"research only"}'
    )
    (run_dir / "promotion_dry_run.json").write_text(
        '{"run_id":"run-b","would_promote_paper":false,'
        '"would_promote_live":false,"live_pnl_claim":false,"research_only":true}'
    )
    report = validate_ledger_schema(run_dir)
    assert report["ok"] is False
    assert "promotion_run_id_mismatch" in report["errors"]


def test_ledger_rejects_invalid_resume_fingerprint(tmp_path: Path) -> None:
    from quant_fund.paper.ledger import validate_ledger_schema

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(
        '{"run_id":"run","schema_version":2,"label":"PAPER_SIMULATED","disclaimer":"research only"}'
    )
    (run_dir / "broker_state.json").write_text(
        '{"run_id":"run","schema_version":2,"step":1,'
        '"resume_fingerprint":"not-a-sha","champion":{"slot":"champion",'
        '"cash":1,"shares":{},"allow_capital":true}}'
    )
    report = validate_ledger_schema(run_dir)
    assert "broker_state_resume_fingerprint_invalid" in report["errors"]


def test_paper_promotion_receipt_rejects_nonfinite_divergence() -> None:
    from quant_fund.paper.ledger import validate_promotion_dry_run_receipt

    errors = validate_promotion_dry_run_receipt(
        {
            "would_promote_paper": True,
            "would_promote_live": False,
            "live_pnl_claim": False,
            "research_only": True,
            "mean_l1": float("nan"),
            "n_steps": -1,
        }
    )
    assert "promotion_mean_l1_must_be_finite_nonnegative" in errors
    assert "promotion_n_steps_must_be_nonnegative_int" in errors


def test_paper_promotion_receipt_rejects_semantic_forgery() -> None:
    from quant_fund.paper.ledger import validate_promotion_dry_run_receipt

    errors = validate_promotion_dry_run_receipt(
        {
            "would_promote_paper": True,
            "would_promote_live": False,
            "live_pnl_claim": False,
            "research_only": True,
            "mean_l1": 0.5,
            "max_l1": 0.2,
            "max_mean_l1": 0.25,
            "n_steps": 2,
            "min_steps": 5,
            "kill_tripped_mid_run": True,
        }
    )
    assert "promotion_max_l1_below_mean_l1" in errors
    assert "promotion_claimed_without_min_steps" in errors
    assert "promotion_claimed_above_mean_l1_threshold" in errors
    assert "promotion_claimed_after_kill_switch" in errors


def test_paper_promotion_receipt_digest_detects_tampering() -> None:
    from quant_fund.paper.ledger import (
        promotion_dry_run,
        validate_promotion_dry_run_receipt,
    )

    receipt = promotion_dry_run(
        run_id="digest-test",
        mean_l1=0.01,
        max_l1=0.02,
        n_steps=5,
        champion_nav=100.0,
        shadow_gross=0.1,
        max_mean_l1=0.25,
        min_steps=5,
        data_source="FILE",
    )
    assert validate_promotion_dry_run_receipt(receipt) == []
    receipt["mean_l1"] = 0.02
    assert "promotion_receipt_sha256_mismatch" in validate_promotion_dry_run_receipt(receipt)


def test_missing_divergence_nan_receipt_is_resumable_diagnostic() -> None:
    from quant_fund.paper.ledger import promotion_dry_run, validate_promotion_dry_run_receipt

    receipt = promotion_dry_run(
        run_id="no-shadow",
        mean_l1=float("nan"),
        max_l1=float("nan"),
        n_steps=2,
        champion_nav=100.0,
        shadow_gross=None,
        max_mean_l1=0.25,
        min_steps=5,
        data_source="SYNTHETIC",
        allow_missing_divergence=True,
    )
    assert "missing_divergence" in receipt["reasons"]
    assert validate_promotion_dry_run_receipt(receipt) == []


def test_paper_promotion_v2_requires_receipt_sha256() -> None:
    """schema_version>=2 must carry receipt_sha256; unsigned v1 legacy still ok."""
    from quant_fund.paper.ledger import promotion_dry_run, validate_promotion_dry_run_receipt

    sealed = promotion_dry_run(
        run_id="seal-required",
        mean_l1=0.01,
        max_l1=0.02,
        n_steps=50,
        champion_nav=100.0,
        shadow_gross=0.1,
        max_mean_l1=0.25,
        min_steps=5,
        data_source="SYNTHETIC",
    )
    assert sealed.get("schema_version") == 2
    assert validate_promotion_dry_run_receipt(sealed) == []

    stripped = dict(sealed)
    stripped.pop("receipt_sha256", None)
    errs = validate_promotion_dry_run_receipt(stripped)
    assert "promotion_receipt_sha256_missing" in errs

    # Unsigned legacy (no schema_version / v1) remains compatible.
    legacy = {
        "would_promote_paper": False,
        "would_promote_live": False,
        "live_pnl_claim": False,
        "research_only": True,
        "mean_l1": 0.01,
        "max_l1": 0.02,
        "n_steps": 10,
        "schema_version": 1,
    }
    assert validate_promotion_dry_run_receipt(legacy) == []

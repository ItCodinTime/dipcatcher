"""Multi-day paper resume + champion/shadow promotion dry-run."""

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from quant_fund.config.loader import load_config
from quant_fund.monitoring.kill_switch import HALT_NEW_ORDERS
from quant_fund.paper.ledger import load_broker_state, promotion_dry_run
from quant_fund.paper.loop import (
    _merge_divergence_summary,
    _paper_resume_fingerprint,
    run_paper_loop,
)


def _bars(n_days: int = 8):
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


def _weights(n_days: int = 7, scale: float = 1.0):
    rows = []
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
    cfg.paper.promote_min_steps = 3
    cfg.paper.promote_max_mean_l1 = 1.0  # loose for dry-run would_promote_paper
    return cfg


def test_paper_resume_preserves_cash(tmp_path):
    cfg = _cfg(tmp_path)
    first = run_paper_loop(
        _bars(8),
        cfg,
        champion_weights=_weights(7),
        shadow_weights=_weights(7, 0.5),
        initial_nav=100_000.0,
        max_steps=3,
        run_id="paper-resume-test",
        prefer_latest=False,
    )
    assert first.metrics["n_steps"] == 3
    state = load_broker_state(tmp_path, "paper-resume-test")
    assert state is not None
    cash_after_first = float(state["champion"]["cash"])
    second = run_paper_loop(
        _bars(8),
        cfg,
        champion_weights=_weights(7),
        shadow_weights=_weights(7, 0.5),
        initial_nav=100_000.0,
        max_steps=2,
        resume=True,
        resume_run_id="paper-resume-test",
    )
    assert second.metrics["resumed"] is True
    assert second.metrics["n_steps"] >= 5
    assert second.metrics["n_steps_this_run"] == 2
    # Cash should not reset to initial_nav
    state2 = load_broker_state(tmp_path, "paper-resume-test")
    assert state2 is not None
    assert abs(float(state2["champion"]["cash"]) - 100_000.0) > 1.0 or cash_after_first != 100_000.0
    # Richer ledger artifacts
    root = tmp_path / "metadata" / "paper" / "paper-resume-test"
    assert (root / "broker_state.json").is_file()
    assert (root / "promotion_dry_run.json").is_file()
    assert (root / "meta.json").is_file()


def test_paper_resume_preserves_tripped_kill_switch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    first = run_paper_loop(
        _bars(8),
        cfg,
        champion_weights=_weights(7),
        initial_nav=100_000.0,
        max_steps=2,
        prefer_latest=False,
        halt_after_steps=1,
        run_id="paper-resume-kill",
    )
    assert first.metrics["kill_tripped_mid_run"] is True
    first_fills = first.metrics["n_fills"]

    resumed = run_paper_loop(
        _bars(8),
        cfg,
        champion_weights=_weights(7),
        initial_nav=100_000.0,
        max_steps=1,
        resume=True,
        resume_run_id="paper-resume-kill",
    )

    state = load_broker_state(tmp_path, "paper-resume-kill")
    assert state is not None
    assert state["champion"]["kill_state"] == HALT_NEW_ORDERS
    assert resumed.metrics["kill_switch_halts"] >= 1
    assert resumed.metrics["kill_tripped_mid_run"] is True
    assert resumed.promotion is not None
    assert resumed.promotion["kill_tripped_mid_run"] is True
    assert resumed.promotion["would_promote_paper"] is False
    assert resumed.metrics["n_fills"] == first_fills


def test_paper_resume_rejects_corrupt_promotion_receipt(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    run_paper_loop(
        _bars(8),
        cfg,
        champion_weights=_weights(7),
        initial_nav=100_000.0,
        max_steps=1,
        run_id="corrupt-promo",
        prefer_latest=False,
    )
    promo = tmp_path / "metadata" / "paper" / "corrupt-promo" / "promotion_dry_run.json"
    promo.write_text("not-json")
    with pytest.raises(ValueError, match="invalid promotion receipt"):
        run_paper_loop(
            _bars(8),
            cfg,
            champion_weights=_weights(7),
            initial_nav=100_000.0,
            resume=True,
            resume_run_id="corrupt-promo",
        )


def test_paper_resume_rejects_corrupt_broker_state(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    run_paper_loop(
        _bars(8),
        cfg,
        champion_weights=_weights(7),
        initial_nav=100_000.0,
        max_steps=1,
        run_id="corrupt-state",
        prefer_latest=False,
    )
    state_path = tmp_path / "metadata" / "paper" / "corrupt-state" / "broker_state.json"
    state = json.loads(state_path.read_text())
    state["champion"] = {"cash": "bad"}
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="invalid champion broker state"):
        run_paper_loop(
            _bars(8),
            cfg,
            champion_weights=_weights(7),
            initial_nav=100_000.0,
            resume=True,
            resume_run_id="corrupt-state",
        )


def test_paper_resume_accepts_legacy_null_mark_ages(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    run_paper_loop(
        _bars(8),
        cfg,
        champion_weights=_weights(7),
        initial_nav=100_000.0,
        max_steps=1,
        run_id="null-mark-ages",
        prefer_latest=False,
    )
    state_path = tmp_path / "metadata" / "paper" / "null-mark-ages" / "broker_state.json"
    state = json.loads(state_path.read_text())
    state["mark_ages"] = None
    state_path.write_text(json.dumps(state))

    result = run_paper_loop(
        _bars(8),
        cfg,
        champion_weights=_weights(7),
        initial_nav=100_000.0,
        resume=True,
        resume_run_id="null-mark-ages",
    )
    assert result.metrics["resumed"] is True


def test_paper_resume_rejects_mismatched_broker_state_identity(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    run_paper_loop(
        _bars(8),
        cfg,
        champion_weights=_weights(7),
        initial_nav=100_000.0,
        max_steps=1,
        run_id="identity-state",
        prefer_latest=False,
    )
    state_path = tmp_path / "metadata" / "paper" / "identity-state" / "broker_state.json"
    state = json.loads(state_path.read_text())
    state["run_id"] = "another-run"
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="mismatched run_id"):
        run_paper_loop(
            _bars(8),
            cfg,
            champion_weights=_weights(7),
            initial_nav=100_000.0,
            resume=True,
            resume_run_id="identity-state",
        )


@pytest.mark.parametrize("field", ["step", "last_decision", "last_exec"])
def test_paper_resume_rejects_invalid_cursor_metadata(tmp_path: Path, field: str) -> None:
    cfg = _cfg(tmp_path)
    run_paper_loop(
        _bars(8),
        cfg,
        champion_weights=_weights(7),
        initial_nav=100_000.0,
        max_steps=1,
        run_id="cursor-state",
        prefer_latest=False,
    )
    state_path = tmp_path / "metadata" / "paper" / "cursor-state" / "broker_state.json"
    state = json.loads(state_path.read_text())
    state[field] = True if field == "step" else "not-a-timestamp"
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match=f"invalid {field}"):
        run_paper_loop(
            _bars(8),
            cfg,
            champion_weights=_weights(7),
            initial_nav=100_000.0,
            resume=True,
            resume_run_id="cursor-state",
        )


def test_paper_resume_rejects_invalid_resume_fingerprint(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    run_paper_loop(
        _bars(8),
        cfg,
        champion_weights=_weights(7),
        initial_nav=100_000.0,
        max_steps=1,
        run_id="fingerprint-state",
        prefer_latest=False,
    )
    state_path = tmp_path / "metadata" / "paper" / "fingerprint-state" / "broker_state.json"
    state = json.loads(state_path.read_text())
    state["resume_fingerprint"] = "not-a-digest"
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="invalid resume_fingerprint"):
        run_paper_loop(
            _bars(8),
            cfg,
            champion_weights=_weights(7),
            initial_nav=100_000.0,
            resume=True,
            resume_run_id="fingerprint-state",
        )


def test_resume_divergence_summary_is_cumulative() -> None:
    mean, maximum = _merge_divergence_summary(
        {"n_steps": 4, "mean_l1": 0.2, "max_l1": 0.4},
        [0.8, 0.0],
        total_steps=6,
    )
    # prior mean covers 6 - 2 = 4 samples, not 4 - 2 = 2.
    assert mean == pytest.approx((0.2 * 4 + 0.8 + 0.0) / 6)
    assert maximum == pytest.approx(0.8)


def test_resume_fingerprint_changes_when_processed_bars_change(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    bars = _bars(8)
    cutoff = bars["event_time"][2]
    changed = bars.with_columns(
        pl.when(pl.col("event_time") == cutoff)
        .then(pl.col("close") + 1.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    assert _paper_resume_fingerprint(bars, cfg, cutoff) != _paper_resume_fingerprint(
        changed, cfg, cutoff
    )
    assert _paper_resume_fingerprint(bars, cfg, cutoff) == _paper_resume_fingerprint(
        bars.reverse(), cfg, cutoff
    )


def test_resume_rejects_changed_processed_history(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    bars = _bars(8)
    weights = _weights(7)
    run_paper_loop(
        bars,
        cfg,
        champion_weights=weights,
        shadow_weights=_weights(7, 0.5),
        max_steps=3,
        run_id="fingerprint-reject",
        prefer_latest=False,
    )
    changed = bars.with_columns(
        pl.when(pl.col("event_time") == bars["event_time"][1])
        .then(pl.col("close") + 7.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run_paper_loop(
            changed,
            cfg,
            champion_weights=weights,
            shadow_weights=_weights(7, 0.5),
            max_steps=1,
            resume=True,
            resume_run_id="fingerprint-reject",
        )


def test_champion_only_resume_keeps_promotion_diagnostic(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    bars = _bars(8)
    weights = _weights(7)
    first = run_paper_loop(
        bars,
        cfg,
        champion_weights=weights,
        max_steps=2,
        run_id="no-shadow-resume",
        prefer_latest=False,
    )
    assert first.promotion is not None
    assert first.promotion["would_promote_paper"] is False
    second = run_paper_loop(
        bars,
        cfg,
        champion_weights=weights,
        max_steps=1,
        resume=True,
        resume_run_id="no-shadow-resume",
    )
    assert second.metrics["resumed"] is True
    assert second.promotion["n_steps"] >= first.promotion["n_steps"]


def test_split_resume_matches_uninterrupted_final_broker_state(tmp_path) -> None:
    bars = _bars(8)
    weights = _weights(7)
    cfg_full = _cfg(tmp_path / "full")
    full = run_paper_loop(
        bars,
        cfg_full,
        champion_weights=weights,
        shadow_weights=_weights(7, 0.5),
        max_steps=5,
        run_id="full",
        prefer_latest=False,
    )
    cfg_split = _cfg(tmp_path / "split")
    run_paper_loop(
        bars,
        cfg_split,
        champion_weights=weights,
        shadow_weights=_weights(7, 0.5),
        max_steps=3,
        run_id="split",
        prefer_latest=False,
    )
    resumed = run_paper_loop(
        bars,
        cfg_split,
        champion_weights=weights,
        shadow_weights=_weights(7, 0.5),
        max_steps=2,
        resume=True,
        resume_run_id="split",
    )
    full_state = load_broker_state(tmp_path / "full", "full")
    split_state = load_broker_state(tmp_path / "split", "split")
    assert full_state is not None and split_state is not None
    assert split_state["step"] == full_state["step"] == 5
    assert split_state["champion"]["cash"] == pytest.approx(full_state["champion"]["cash"])
    assert split_state["champion"]["shares"] == full_state["champion"]["shares"]
    assert resumed.metrics["n_steps_this_run"] == 2
    assert full.metrics["n_steps"] == 5


def test_promotion_dry_run_never_live():
    receipt = promotion_dry_run(
        mean_l1=0.01,
        max_l1=0.02,
        n_steps=20,
        champion_nav=1.1e6,
        shadow_gross=1.2,
        max_mean_l1=0.25,
        min_steps=5,
        data_source="SYNTHETIC",
    )
    assert receipt["would_promote_paper"] is True
    assert receipt["would_promote_live"] is False
    assert receipt["live_pnl_claim"] is False
    assert "synthetic_evidence_not_live_promotable" in receipt["reasons"]


def test_promotion_dry_run_rejects_high_divergence():
    receipt = promotion_dry_run(
        mean_l1=0.9,
        max_l1=1.2,
        n_steps=20,
        champion_nav=1e6,
        shadow_gross=1.0,
        max_mean_l1=0.25,
        min_steps=5,
        data_source="SYNTHETIC",
    )
    assert receipt["would_promote_paper"] is False

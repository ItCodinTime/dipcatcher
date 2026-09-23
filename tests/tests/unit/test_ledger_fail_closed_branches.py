"""Paper ledger validation branches: corrupt artifacts fail closed, never crash.

Complements test_ledger_schema / test_paper_* by exercising the validation and
recovery branches that previously had no coverage (unsafe ids, malformed JSON,
promotion receipt checks, broker-state history integrity, unreadable artifacts).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from quant_fund.execution.simulated_broker import BrokerSnapshot, SimulatedBroker
from quant_fund.paper.ledger import (
    PaperLedger,
    latest_run_id,
    load_broker_state,
    paper_root,
    validate_ledger_schema,
    validate_promotion_dry_run_receipt,
)


@pytest.mark.parametrize("subdir", ["", "  ", "/abs/path", "../escape", "a/../b"])
def test_paper_root_rejects_unsafe_subdir(tmp_path: Path, subdir: str) -> None:
    with pytest.raises(ValueError, match="paper ledger subdir must be a safe relative path"):
        paper_root(tmp_path, subdir)


class _StubBroker:
    def __init__(self, shares: dict[str, float]) -> None:
        self._shares = shares

    def snapshot(self, *, asof: datetime | None = None) -> BrokerSnapshot:
        return BrokerSnapshot(
            cash=1_000.0,
            shares=dict(self._shares),
            nav=1_000.0,
            gross=sum(abs(v) for v in self._shares.values()),
            net=sum(self._shares.values()),
            asof=asof,
            slot="champion",
        )


def test_record_broker_skips_zero_positions(tmp_path: Path) -> None:
    ledger = PaperLedger(tmp_path, "run_zero", load_existing=False)
    ledger.record_broker(_StubBroker({"A": 1.0, "B": 0.0}), asof=datetime(2026, 1, 1, tzinfo=UTC))
    rows = ledger._positions
    assert [row["security_id"] for row in rows] == ["A"]
    assert ledger._equity and ledger._equity[0]["nav"] == 1_000.0


def test_load_equity_missing_file_returns_empty(tmp_path: Path) -> None:
    ledger = PaperLedger(tmp_path, "run_none", load_existing=False)
    assert ledger.load_equity().height == 0
    assert ledger.load_shadow_equity().height == 0


def test_load_broker_state_missing_and_corrupt(tmp_path: Path) -> None:
    assert load_broker_state(tmp_path, "nope") is None
    root = paper_root(tmp_path) / "run_bad"
    root.mkdir(parents=True)
    (root / "broker_state.json").write_text("{not json")
    assert load_broker_state(tmp_path, "run_bad") is None
    (root / "broker_state.json").write_text(json.dumps(["not", "an", "object"]))
    assert load_broker_state(tmp_path, "run_bad") is None


def test_latest_run_id_missing_and_corrupt(tmp_path: Path) -> None:
    assert latest_run_id(tmp_path) is None
    latest = paper_root(tmp_path) / "latest_run.json"
    latest.write_text("{not json")
    assert latest_run_id(tmp_path) is None
    latest.write_text(json.dumps({"run_id": 7}))
    assert latest_run_id(tmp_path) is None
    latest.write_text(json.dumps({"run_id": ""}))
    assert latest_run_id(tmp_path) is None
    latest.write_text(json.dumps({"run_id": "run_ok"}))
    assert latest_run_id(tmp_path) == "run_ok"


def test_promotion_receipt_rejects_non_object() -> None:
    assert validate_promotion_dry_run_receipt(["not", "a", "dict"]) == [
        "promotion_receipt_not_object"
    ]


def _promo() -> dict:
    from quant_fund.paper.ledger import promotion_dry_run

    return promotion_dry_run(
        run_id="run_ok",
        mean_l1=0.05,
        max_l1=0.2,
        n_steps=10,
        champion_nav=1.01,
        shadow_gross=0.0,
        max_mean_l1=0.25,
        min_steps=5,
        data_source="SYNTHETIC",
        rolling_mean_l1=0.05,
        rolling_window=5,
    )


def test_promotion_receipt_flag_and_field_branches() -> None:
    base = _promo()
    assert validate_promotion_dry_run_receipt(base) == []

    bad_paper_flag = {**base, "would_promote_paper": "yes"}
    assert "promotion_would_promote_paper_must_be_bool" in validate_promotion_dry_run_receipt(
        bad_paper_flag
    )

    live_claim = {**base, "live_pnl_claim": True}
    assert "promotion_live_pnl_claim_must_be_false" in validate_promotion_dry_run_receipt(
        live_claim
    )

    zero_window = {**base, "rolling_window": 0}
    assert "promotion_rolling_window_must_be_positive" in validate_promotion_dry_run_receipt(
        zero_window
    )

    unsafe_id = {**base, "run_id": "../escape"}
    assert "promotion_run_id_invalid" in validate_promotion_dry_run_receipt(unsafe_id)

    bad_digest = {**base, "receipt_sha256": "not-a-digest"}
    assert "promotion_receipt_sha256_invalid" in validate_promotion_dry_run_receipt(bad_digest)

    rolling_above_max = {**base, "rolling_mean_l1": 5.0, "max_l1": 1.0}
    assert "promotion_rolling_mean_l1_above_max_l1" in validate_promotion_dry_run_receipt(
        rolling_above_max
    )

    claimed_after_kill = {
        **base,
        "would_promote_paper": True,
        "kill_tripped_mid_run": True,
    }
    assert "promotion_claimed_after_kill_switch" in validate_promotion_dry_run_receipt(
        claimed_after_kill
    )


def test_promotion_dry_run_flags_missing_divergence_and_kill() -> None:
    from quant_fund.paper.ledger import promotion_dry_run

    nan_receipt = promotion_dry_run(
        run_id="run_ok",
        mean_l1=float("nan"),
        max_l1=float("nan"),
        n_steps=10,
        champion_nav=1.0,
        shadow_gross=0.0,
        max_mean_l1=0.25,
        min_steps=5,
        data_source="SYNTHETIC",
        allow_missing_divergence=True,
    )
    assert nan_receipt["would_promote_paper"] is False
    assert "missing_divergence" in nan_receipt["reasons"]
    assert validate_promotion_dry_run_receipt(nan_receipt) == []

    killed = promotion_dry_run(
        run_id="run_ok",
        mean_l1=0.01,
        max_l1=0.05,
        n_steps=10,
        champion_nav=1.0,
        shadow_gross=0.0,
        max_mean_l1=0.25,
        min_steps=5,
        data_source="SYNTHETIC",
        kill_tripped_mid_run=True,
    )
    assert "kill_switch_tripped_mid_run" in killed["reasons"]


def test_promotion_receipt_numeric_and_claim_branches() -> None:
    base = _promo()
    nan_l1 = {**base, "mean_l1": float("nan")}
    assert "promotion_mean_l1_must_be_finite_nonnegative" in validate_promotion_dry_run_receipt(
        nan_l1
    )

    negative = {**base, "max_l1": -0.5}
    assert "promotion_max_l1_must_be_finite_nonnegative" in validate_promotion_dry_run_receipt(
        negative
    )

    fractional_steps = {**base, "n_steps": 2.5}
    assert "promotion_n_steps_must_be_nonnegative_int" in validate_promotion_dry_run_receipt(
        fractional_steps
    )

    sealed_without_digest = {**base, "schema_version": 2}
    sealed_without_digest.pop("receipt_sha256", None)
    assert "promotion_receipt_sha256_missing" in validate_promotion_dry_run_receipt(
        sealed_without_digest
    )

    max_below_mean = {**base, "mean_l1": 0.2, "max_l1": 0.1}
    assert "promotion_max_l1_below_mean_l1" in validate_promotion_dry_run_receipt(max_below_mean)

    under_min_steps = {
        **base,
        "would_promote_paper": True,
        "n_steps": 3,
        "min_steps": 5,
    }
    assert "promotion_claimed_without_min_steps" in validate_promotion_dry_run_receipt(
        under_min_steps
    )

    above_threshold = {
        **base,
        "would_promote_paper": True,
        "mean_l1": 0.5,
        "max_mean_l1": 0.25,
    }
    assert "promotion_claimed_above_mean_l1_threshold" in validate_promotion_dry_run_receipt(
        above_threshold
    )


def test_ledger_promotion_run_id_mismatch(tmp_path: Path) -> None:
    root = _ledger_dir(tmp_path)
    promo = {**_promo(), "run_id": "different_run"}
    (root / "promotion_dry_run.json").write_text(json.dumps(promo))
    report = validate_ledger_schema(root)
    assert "promotion_run_id_mismatch" in report["errors"]


def _ledger_dir(tmp_path: Path, run_id: str = "run1") -> Path:
    root = tmp_path / "metadata" / "paper" / run_id
    root.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id,
        "schema_version": 2,
        "label": "PAPER_SIMULATED",
        "disclaimer": "research only",
    }
    (root / "meta.json").write_text(json.dumps(meta))
    return root


def test_ledger_schema_version_mismatch_is_warning(tmp_path: Path) -> None:
    root = _ledger_dir(tmp_path)
    report = validate_ledger_schema(root, expect_version=99)
    assert any(
        w.startswith("schema_version_mismatch:got=2:expected=99") for w in report["warnings"]
    )
    assert report["schema_version"] == 2


def test_broker_state_fingerprint_and_champion_branches(tmp_path: Path) -> None:
    root = _ledger_dir(tmp_path)
    state = {
        "run_id": "run1",
        "schema_version": 2,
        "step": 1,
        "resume_fingerprint": "zz",
        "champion": {"slot": "champion", "allow_capital": False},
    }
    (root / "broker_state.json").write_text(json.dumps(state))
    report = validate_ledger_schema(root)
    assert "broker_state_resume_fingerprint_invalid" in report["errors"]
    assert "broker_state_champion_missing:cash" in report["errors"]
    assert "broker_state_champion_missing:shares" in report["errors"]

    state["champion"] = {
        "slot": "champion",
        "cash": 1.0,
        "shares": {"A": 1.0},
        "allow_capital": False,
        "history": {"not": "a list"},
    }
    (root / "broker_state.json").write_text(json.dumps(state))
    report = validate_ledger_schema(root)
    assert "broker_state_champion_history_not_list" in report["errors"]


def test_broker_state_history_integrity_branches(tmp_path: Path) -> None:
    root = _ledger_dir(tmp_path)
    state = {
        "run_id": "run1",
        "schema_version": 2,
        "step": 2,
        "champion": {
            "slot": "champion",
            "cash": 1.0,
            "shares": {"A": 1.0},
            "allow_capital": False,
            "n_orders": 3,
            "n_fills": 1,
            "history": [
                "not-a-record",
                {"order": {"order_id": "o1"}},  # missing keys
            ],
        },
    }
    (root / "broker_state.json").write_text(json.dumps(state))
    report = validate_ledger_schema(root)
    assert any(
        e.startswith("broker_state_champion_history_record_invalid:") for e in report["errors"]
    )
    assert any(e.startswith("broker_state_champion_history_missing:1:") for e in report["errors"])
    assert "broker_state_champion_history_order_count_mismatch" in report["errors"]
    assert "broker_state_champion_history_fill_count_mismatch" in report["errors"]


def test_equity_column_and_unreadable_branches(tmp_path: Path) -> None:
    root = _ledger_dir(tmp_path)
    (root / "equity.parquet").write_bytes(b"not parquet at all")
    report = validate_ledger_schema(root)
    assert any(e.startswith("equity_unreadable:") for e in report["errors"])

    pl.DataFrame({"asof": [datetime(2026, 1, 1, tzinfo=UTC)], "cash": [1.0]}).write_parquet(
        root / "equity.parquet"
    )
    report = validate_ledger_schema(root)
    assert "equity_missing:nav" in report["errors"]

    pl.DataFrame({"asof": [datetime(2026, 1, 1, tzinfo=UTC)], "nav": [1.0]}).write_parquet(
        root / "equity.parquet"
    )
    report = validate_ledger_schema(root)
    assert "equity_missing:cash" in report["errors"]

    pl.DataFrame({"nav": [1.0], "cash": [1.0]}).write_parquet(root / "equity.parquet")
    report = validate_ledger_schema(root)
    assert "equity_missing:asof_or_event_time" in report["errors"]


def test_orders_and_cash_unreadable_and_missing_columns(tmp_path: Path) -> None:
    root = _ledger_dir(tmp_path)
    (root / "orders.parquet").write_bytes(b"garbage")
    (root / "cash_ledger.parquet").write_bytes(b"garbage")
    report = validate_ledger_schema(root)
    assert any(e.startswith("orders_unreadable:") for e in report["errors"])
    assert any(e.startswith("cash_ledger_unreadable:") for e in report["errors"])

    pl.DataFrame({"security_id": ["A"], "side": ["buy"], "status": ["filled"]}).write_parquet(
        root / "orders.parquet"
    )
    report = validate_ledger_schema(root)
    assert "orders_missing:order_id" in report["errors"]


def test_analytics_export_sha256_invalid_branch(tmp_path: Path) -> None:
    root = _ledger_dir(tmp_path)
    payload = {
        "label": "PAPER_SIMULATED",
        "data_source": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "analytics_export_sha256": "not-a-digest",
    }
    (root / "analytics_export.json").write_text(json.dumps(payload))
    report = validate_ledger_schema(root)
    assert "analytics_export_sha256_invalid" in report["errors"]


def test_validate_ledger_schema_full_flow_ok(tmp_path: Path) -> None:
    """A minimal coherent ledger validates with no errors (warnings allowed)."""
    ledger = PaperLedger(tmp_path, "run_flow", load_existing=False)
    ledger.set_meta(
        run_id="run_flow",
        schema_version=PaperLedger.SCHEMA_VERSION,
        label="PAPER_SIMULATED",
        disclaimer="research only",
    )
    broker = SimulatedBroker(config=_paper_cfg(tmp_path), initial_cash=100_000.0)
    ledger.record_broker(broker, asof=datetime(2026, 1, 1, tzinfo=UTC))
    ledger.save_broker_state(
        broker,
        None,
        last_decision=None,
        last_exec=None,
        step=1,
        resume_fingerprint=None,
        mark_ages={"A": 2},
    )
    saved_state = load_broker_state(tmp_path, "run_flow")
    assert saved_state is not None
    assert saved_state["mark_ages"] == {"A": 2}
    ledger.flush()
    report = validate_ledger_schema(ledger.root)
    assert report["ok"] is True, report["errors"]
    assert report["schema_version"] == PaperLedger.SCHEMA_VERSION
    assert report["live_pnl_claim"] is False


def test_positive_cursor_requires_matching_equity_rows(tmp_path: Path) -> None:
    root = _ledger_dir(tmp_path, "cursor-equity")
    state = {
        "run_id": "cursor-equity",
        "schema_version": 2,
        "step": 2,
        "champion": {
            "slot": "champion",
            "cash": 1.0,
            "shares": {},
            "allow_capital": False,
            "n_orders": 0,
            "n_fills": 0,
            "history": [],
        },
    }
    (root / "broker_state.json").write_text(json.dumps(state))

    missing = validate_ledger_schema(root)
    assert "broker_state_step_requires_equity" in missing["errors"]

    pl.DataFrame({"event_time": [], "nav": []}).write_parquet(root / "equity.parquet")
    empty = validate_ledger_schema(root)
    assert "broker_state_step_equity_empty" in empty["errors"]

    pl.DataFrame({"event_time": [datetime(2026, 1, 1, tzinfo=UTC)], "nav": [1.0]}).write_parquet(
        root / "equity.parquet"
    )
    mismatch = validate_ledger_schema(root)
    assert "broker_state_step_equity_count_mismatch:step=2:rows=1" in mismatch["errors"]

    pl.DataFrame(
        {
            "event_time": [datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)],
            "nav": [1.0, 1.01],
        }
    ).write_parquet(root / "equity.parquet")
    coherent = validate_ledger_schema(root)
    assert not any(error.startswith("broker_state_step_") for error in coherent["errors"])


def test_broker_state_rejects_malformed_resume_cursor_fields(tmp_path: Path) -> None:
    root = _ledger_dir(tmp_path, "malformed-cursors")
    state = {
        "run_id": "malformed-cursors",
        "schema_version": 2,
        "step": "2",
        "last_decision": "not-a-date",
        "last_exec": 123,
        "mark_ages": {"A": -1},
        "champion": {
            "slot": "champion",
            "cash": 1.0,
            "shares": {},
            "allow_capital": False,
            "n_orders": 0,
            "n_fills": 0,
            "history": [],
        },
    }
    (root / "broker_state.json").write_text(json.dumps(state))

    report = validate_ledger_schema(root)

    assert report["ok"] is False
    assert {
        "broker_state_step_invalid",
        "broker_state_last_decision_invalid",
        "broker_state_last_exec_invalid",
        "broker_state_mark_ages_invalid",
    } <= set(report["errors"])


def test_broker_state_rejects_nonfinite_champion_numbers(tmp_path: Path) -> None:
    root = _ledger_dir(tmp_path, "malformed-champion")
    state = {
        "run_id": "malformed-champion",
        "schema_version": 2,
        "step": 0,
        "champion": {
            "slot": "",
            "cash": "nan",
            "shares": {"A": "inf"},
            "allow_capital": 1,
            "n_orders": 0,
            "n_fills": 0,
            "history": [],
        },
    }
    (root / "broker_state.json").write_text(json.dumps(state))

    report = validate_ledger_schema(root)

    assert report["ok"] is False
    assert {
        "broker_state_champion_slot_invalid",
        "broker_state_champion_cash_invalid",
        "broker_state_champion_shares_invalid",
        "broker_state_champion_allow_capital_invalid",
    } <= set(report["errors"])


def _paper_cfg(tmp_path: Path):  # noqa: ANN202
    from quant_fund.config.loader import load_config

    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    return cfg

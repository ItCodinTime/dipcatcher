"""The checker attributes a crafted one-field difference to one cause."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from quant_fund.parity.checker import CAUSE_PRECEDENCE, attribute_pair, check_parity


def _row(**overrides: object) -> dict:
    stamp = datetime(2024, 1, 2, tzinfo=UTC)
    base: dict = {
        "event_time": stamp,
        "decision_time": stamp,
        "security_id": "A",
        "source": "synthetic",
        "revision_id": "r0",
        "available_time": stamp,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1_000_000.0,
        "exec_open": 101.0,
        "exec_close": 102.0,
        "exec_source": "synthetic",
        "exec_revision": "r0",
        "raw_weight": 0.05,
        "rounded_weight": 0.05,
        "lot_size": 0.0,
        "commission_bps": 1.0,
        "half_spread_bps": 5.0,
        "impact_y": 0.1,
        "bps_per_turnover": 0.0,
        "state_digest": "state-0",
        "code_path_digest": "code-0",
        "restarted": False,
        "restart_generation": 0,
        "fill_signed_qty": 10.0,
        "fill_price": 101.0,
        "fill_fee": 1.0,
        "fill_spread": 2.0,
        "fill_impact": 3.0,
        "explicit_cost": 6.0,
    }
    base.update(overrides)
    return base


class _Run:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.synthetic = True


def _cause(**shadow_overrides: object) -> tuple[str | None, str]:
    return attribute_pair(_row(), _row(**shadow_overrides))


def test_identical_rows_are_not_a_divergence() -> None:
    cause, detail = attribute_pair(_row(), _row())
    assert cause is None
    assert detail == ""


def test_each_injected_cause() -> None:
    stamp = datetime(2024, 1, 2, tzinfo=UTC)
    cases = {
        "data": {"revision_id": "r1"},
        "timing": {"decision_time": stamp + timedelta(seconds=1)},
        "code_path": {"code_path_digest": "other"},
        "state_drift": {
            "state_digest": "state-1",
            "restart_generation": 1,
            "restarted": True,
        },
        "rounding": {"lot_size": 0.03, "rounded_weight": 0.06},
        "costs": {"commission_bps": 8.0, "fill_fee": 4.0, "explicit_cost": 9.0},
        "fills": {"fill_price": 103.0},
    }
    assert set(cases) == set(CAUSE_PRECEDENCE)
    for expected, overrides in cases.items():
        cause, _detail = _cause(**overrides)
        assert cause == expected, overrides


def test_data_wins_when_the_tape_and_the_fill_both_differ() -> None:
    cause, detail = _cause(revision_id="vendor-b", fill_price=999.0, close=80.0, low=80.0)
    assert cause == "data"
    assert "revision_id" in detail
    assert "fill_price" not in detail


def test_cost_parameters_win_over_a_cash_state_change_without_restart() -> None:
    cause, detail = _cause(state_digest="cash-moved", commission_bps=4.0)
    assert cause == "costs"
    assert detail == "commission_bps"


def test_state_digest_without_restart_is_still_state_drift() -> None:
    cause, detail = _cause(state_digest="drifted")
    assert cause == "state_drift"
    assert detail == "state_digest"


def test_output_mismatch_with_identical_inputs_is_code_path() -> None:
    cause, detail = _cause(raw_weight=0.2, rounded_weight=0.2)
    assert cause == "code_path"
    assert detail == "output_mismatch"


def test_missing_bar_is_data_and_a_shifted_clock_is_timing() -> None:
    stamp = datetime(2024, 1, 2, tzinfo=UTC)
    backtest = [_row()]
    missing = check_parity(_Run(backtest), _Run([]))
    assert missing["ledger"][0]["cause"] == "data"
    assert missing["ledger"][0]["detail"] == "missing_bar"

    shifted = _row(event_time=stamp + timedelta(days=1))
    report = check_parity(_Run(backtest), _Run([shifted]))
    assert {row["cause"] for row in report["ledger"]} == {"timing"}
    assert {row["detail"] for row in report["ledger"]} == {"shifted_bar"}


def test_summary_counts_and_honesty_flags() -> None:
    report = check_parity(_Run([_row(), _row(security_id="B")]), _Run([_row(revision_id="r9")]))
    assert report["n_bars"] == 2
    assert report["n_divergent"] == 2
    assert report["n_matched"] == 0
    assert report["by_cause"]["data"] == 2
    assert report["match"] is False
    assert report["live_pnl_claim"] is False
    assert report["research_only"] is True
    assert report["would_promote_live"] is False
    assert abs(report["divergence_rate"] - 1.0) < 1e-12


def test_tolerance_ignores_a_dust_weight() -> None:
    cause, _detail = attribute_pair(
        _row(), _row(raw_weight=0.05 + 1e-12, rounded_weight=0.05 + 1e-12)
    )
    assert cause is None

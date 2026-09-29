"""Lane-receipt contract checks: re-derivation of claims inside sealed payloads."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from scipy.stats import t as t_dist

from quant_fund.research.lane_contracts import lane_contract_errors

RECEIPTS = Path(__file__).resolve().parents[3] / "receipts"


def _load(name: str) -> dict:
    return json.loads((RECEIPTS / name).read_text(encoding="utf-8"))


@pytest.fixture()
def capacity() -> dict:
    return _load("capacity_eval_cd0854242ed8a9ec.json")


@pytest.fixture()
def rankic() -> dict:
    return _load("rankic_eval_9ebdad7da83e7348.json")


@pytest.fixture()
def p42() -> dict:
    return _load("fast_replay_p42_conformance_20260927.json")


class TestCapacityContract:
    def test_committed_receipt_is_clean(self, capacity: dict) -> None:
        assert lane_contract_errors(capacity) == []

    def test_forged_days_to_trade_fails(self, capacity: dict) -> None:
        capacity["results"][0]["days_to_trade"] *= 2.0
        assert any("days_to_trade" in e for e in lane_contract_errors(capacity))

    def test_forged_feasible_fails(self, capacity: dict) -> None:
        row = next(r for r in capacity["results"] if r["feasible"] == 0)
        row["feasible"] = 1  # claim feasible beyond the cap
        assert any("feasible" in e for e in lane_contract_errors(capacity))

    def test_n_rows_mismatch_fails(self, capacity: dict) -> None:
        capacity["n_rows"] += 1
        assert "n_rows" in lane_contract_errors(capacity)

    def test_non_monotone_impact_fails(self, capacity: dict) -> None:
        book_rows = [r for r in capacity["results"] if r["book"] == "uniform"]
        book_rows.sort(key=lambda r: r["aum"])
        book_rows[1]["impact_bps"] = book_rows[0]["impact_bps"] / 2
        assert any("impact_not_monotone" in e for e in lane_contract_errors(capacity))

    def test_non_finite_row_fails(self, capacity: dict) -> None:
        capacity["results"][0]["max_participation"] = float("nan")
        assert any("non_finite" in e for e in lane_contract_errors(capacity))


class TestRankicContract:
    def test_committed_receipt_is_clean(self, rankic: dict) -> None:
        assert lane_contract_errors(rankic) == []

    def test_forged_p_value_fails(self, rankic: dict) -> None:
        row = next(r for r in rankic["results"] if r["status"] == "ok")
        row["p_spearman"] = 1e-300  # fabricate significance
        assert any("p_spearman" in e for e in lane_contract_errors(rankic))

    def test_p_rederivation_is_exact(self, rankic: dict) -> None:
        # The contract re-derives p from t under the writer's df = n - 1.
        row = next(r for r in rankic["results"] if r["status"] == "ok")
        expected = 2.0 * float(t_dist.sf(abs(row["t_spearman"]), df=row["n_dates"] - 1))
        assert math.isclose(row["p_spearman"], expected, rel_tol=1e-6)

    def test_n_rows_mismatch_fails(self, rankic: dict) -> None:
        rankic["n_rows"] = 0
        assert "n_rows" in lane_contract_errors(rankic)

    def test_error_row_without_message_fails(self, rankic: dict) -> None:
        rankic["results"][0] = {"status": "error", "error": ""}
        errors = lane_contract_errors(rankic)
        assert any("error_empty" in e for e in errors)
        assert "n_error_rows" in errors

    def test_out_of_range_ic_fails(self, rankic: dict) -> None:
        rankic["results"][0]["mean_spearman"] = 1.7
        assert any("mean_spearman" in e for e in lane_contract_errors(rankic))


class TestP42Contract:
    def test_committed_receipt_is_clean(self, p42: dict) -> None:
        assert lane_contract_errors(p42) == []

    def test_live_claim_fails(self, p42: dict) -> None:
        p42["live_pnl_claim"] = True
        assert "live_pnl_claim" in lane_contract_errors(p42)

    def test_digested_file_must_exist(self, p42: dict) -> None:
        p42["code_sha256"]["src/quant_fund/does_not_exist.py"] = "0" * 64
        assert any("missing_file" in e for e in lane_contract_errors(p42))

    def test_malformed_digest_fails(self, p42: dict) -> None:
        key = next(iter(p42["code_sha256"]))
        p42["code_sha256"][key] = "notahexdigest"
        assert any("digest" in e for e in lane_contract_errors(p42))

    def test_missing_disclaimer_fails(self, p42: dict) -> None:
        p42["disclaimer"] = "no tape label"
        assert "disclaimer_synthetic" in lane_contract_errors(p42)


def test_unknown_schema_returns_empty() -> None:
    assert lane_contract_errors({"schema": "something_else.v9"}) == []
    assert lane_contract_errors({}) == []

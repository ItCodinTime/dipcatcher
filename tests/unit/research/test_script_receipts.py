"""Contract tests for script-emitted receipt schemas.

Each test pins a re-derivation: the verifier recomputes a headline claim from
the receipt's own embedded detail, so a forged aggregate fails verification
rather than passing on seal consistency alone.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from quant_fund.research.receipt_v2 import verify_receipt_payload
from quant_fund.research.script_receipts import (
    SCRIPT_RECEIPT_CONTRACTS,
    script_receipt_contract_errors,
)

RECEIPTS = Path(__file__).resolve().parents[3] / "receipts"


def _load(name: str) -> dict:
    path = RECEIPTS / name
    if not path.is_file():
        pytest.skip(f"{name} receipt not in this checkout")
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def _errors(name: str) -> list[str]:
    return script_receipt_contract_errors(_load(name)["schema"], _load(name))


def test_committed_script_receipts_satisfy_their_contracts() -> None:
    """Real artifacts must pass their own re-derivation contracts."""
    for name in (
        "adaptive_mix_20asset_1d_20260922.json",
        "adaptive_mix_band_search_20asset_1d_20260922.json",
        "basis_pair_candidate_20asset_1d_20260922.json",
        "basis_reversion_screen_20asset_1d_20260922.json",
        "incumbent_bench_qlib.json",
        "dip_bench_crypto_1d_20260925.json",
    ):
        assert _errors(name) == [], f"{name}: {_errors(name)}"


def test_band_search_forged_selection_caught() -> None:
    payload = _load("adaptive_mix_band_search_20asset_1d_20260922.json")
    # The committed run selected nothing (no candidate was eligible); claim one.
    forged = copy.deepcopy(payload)
    forged["selected_band"] = 0.0
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert "selected_band_not_argmax_dev_sharpe" in errors


def test_band_search_forged_eligibility_caught() -> None:
    payload = _load("adaptive_mix_band_search_20asset_1d_20260922.json")
    forged = copy.deepcopy(payload)
    forged["candidates"][0]["eligible"] = True  # cagr/sharpe are negative
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert any("eligible_mismatch" in e for e in errors)


def test_adaptive_mix_allocation_must_be_a_simplex() -> None:
    payload = _load("adaptive_mix_20asset_1d_20260922.json")
    forged = copy.deepcopy(payload)
    seg = next(iter(forged["mean_adaptive_allocation"]))
    sleeve = next(iter(forged["mean_adaptive_allocation"][seg]))
    forged["mean_adaptive_allocation"][seg][sleeve] = 2.5
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert any("weight_out_of_bounds" in e for e in errors)


def test_adaptive_mix_unknown_sleeve_caught() -> None:
    payload = _load("adaptive_mix_20asset_1d_20260922.json")
    forged = copy.deepcopy(payload)
    seg = next(iter(forged["mean_adaptive_allocation"]))
    weights = forged["mean_adaptive_allocation"][seg]
    weights["phantom_sleeve"] = 0.01
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert any("unknown_sleeve" in e for e in errors)


def test_incumbent_bench_forged_median_caught() -> None:
    payload = _load("incumbent_bench_qlib.json")
    forged = copy.deepcopy(payload)
    forged["latency"]["dipcatcher_ms_median"] = 1.0  # real runs are ~100ms
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert "dipcatcher_ms_median_rederive_mismatch" in errors


def test_dip_bench_forged_event_count_caught() -> None:
    payload = _load("dip_bench_crypto_1d_20260925.json")
    forged = copy.deepcopy(payload)
    forged["n_events"] += 1
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert "n_events_not_sum_of_per_asset" in errors


def test_dip_bench_forged_recovery_caught() -> None:
    payload = _load("dip_bench_crypto_1d_20260925.json")
    forged = copy.deepcopy(payload)
    forged["baseline_recovery"]["1m"] = 1.7
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert "baseline_recovery_not_fractions" in errors


def test_basis_screen_fraction_out_of_bounds_caught() -> None:
    payload = _load("basis_reversion_screen_20asset_1d_20260922.json")
    forged = copy.deepcopy(payload)
    forged["positive_day_fraction"] = 1.4
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert "positive_day_fraction_out_of_bounds" in errors


def test_basis_pair_candidate_non_bool_eligible_caught() -> None:
    payload = _load("basis_pair_candidate_20asset_1d_20260922.json")
    forged = copy.deepcopy(payload)
    forged["development_eligible"] = "true"
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert "development_eligible_not_bool" in errors


def test_unknown_schema_passes_through() -> None:
    assert script_receipt_contract_errors("not_a_real_schema", {}) == []
    assert script_receipt_contract_errors(None, {}) == []


def test_malformed_body_fails_closed() -> None:
    """A payload that breaks the checker must report an error, not pass."""

    class ExplodingMapping(dict):
        def get(self, key: object, default: object = None) -> object:
            raise TypeError("boom")

    assert script_receipt_contract_errors("adaptive_mix_band_search.v1", ExplodingMapping()) == [
        "contract_check_error"
    ]


def test_dispatch_reaches_verify_receipt_payload() -> None:
    """The v1 envelope path must actually invoke the script contracts."""
    payload = _load("dip_bench_crypto_1d_20260925.json")
    forged = copy.deepcopy(payload)
    forged["n_events"] += 1
    result = verify_receipt_payload(forged, "forged.json")
    errors = result["errors"] if isinstance(result, dict) else result.errors
    assert "n_events_not_sum_of_per_asset" in errors


def test_every_committed_schema_has_a_contract() -> None:
    """A new scripts/ receipt schema must not ship unverified."""
    for name in (
        "adaptive_mix_20asset_1d_20260922.json",
        "adaptive_mix_band_search_20asset_1d_20260922.json",
        "basis_pair_candidate_20asset_1d_20260922.json",
        "basis_reversion_screen_20asset_1d_20260922.json",
        "incumbent_bench_qlib.json",
        "dip_bench_crypto_1d_20260925.json",
    ):
        path = RECEIPTS / name
        if not path.is_file():
            continue
        schema = json.loads(path.read_text())["schema"]
        assert schema in SCRIPT_RECEIPT_CONTRACTS, f"{name}: no contract for {schema}"


@pytest.mark.parametrize(
    "value,expected",
    [(3, True), (1.0, False), (0, False), (True, False), (math.nan, False), ("1", False)],
)
def test_positive_int_strictness(value: object, expected: bool) -> None:
    from quant_fund.research.script_receipts import _positive_int

    assert _positive_int(value) is expected

"""Verifier tamper matrix — every mutation class must fail closed.

Self-sealed receipts are tamper-evident, not authenticated: anyone can
recompute ``receipt_sha256``. The verifier's job is to catch what a reseal
cannot hide — schema violations, dishonesty markers, envelope/payload
disagreement, and bound digests that no longer re-derive. Mutations that
change nothing checkable (e.g. a forged score inside an honest reseal over
a generic payload) are a documented residual, anchored externally via the
audit ledger / sigstore path, not by this verifier.

All data here is SYNTHETIC — correctness evidence, never market evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_fund.research.fleet_eval import (
    fleet_head_factories,
    run_distribution_fleet,
    write_fleet_receipt,
)
from quant_fund.research.receipt_v2 import (
    build_receipt_v2,
    seal_receipt,
    verify_receipt_file,
    verify_receipt_payload,
)

TAUS = (0.1, 0.5, 0.9)


def _generic_v2(seed: int = 5) -> dict[str, Any]:
    return seal_receipt(
        build_receipt_v2(
            kind="tamper_matrix_lane",
            data_label="SYNTHETIC",
            dataset={"rows": 4, "content_sha256": "a" * 64},
            params={"seed": seed, "taus": list(TAUS)},
            code_files=(Path(__file__),),
            verdict="pass",
            payload={"schema": "tamper_test.v1", "rows": [1, 2, 3]},
        )
    )


def _fleet_v2(tmp_path: Path) -> dict[str, Any]:
    _, receipt = run_distribution_fleet(
        fleet_head_factories(TAUS, 0, ["empirical", "gaussian"]),
        shards=["iid_gaussian"],
        n_train=64,
        n_eval=32,
        seed=0,
        taus=TAUS,
    )
    path = write_fleet_receipt(receipt, tmp_path, receipt_version=2)
    return json.loads(path.read_text())


def _fleet_v1(tmp_path: Path) -> dict[str, Any]:
    _, receipt = run_distribution_fleet(
        fleet_head_factories(TAUS, 0, ["empirical"]),
        shards=["iid_gaussian"],
        n_train=64,
        n_eval=32,
        seed=0,
        taus=TAUS,
    )
    path = write_fleet_receipt(receipt, tmp_path, receipt_version=1)
    return json.loads(path.read_text())


def _invalid_with(receipt: dict[str, Any], needle: str) -> None:
    result = verify_receipt_payload(receipt)
    assert result["valid"] is False, f"expected invalid, errors={result['errors']}"
    assert any(needle in error for error in result["errors"]), result["errors"]


# --- file layer: malformed input must never crash or pass -------------------


def test_file_with_nan_literal_is_malformed(tmp_path: Path) -> None:
    """json.loads accepts bare NaN by default; the file layer must refuse it
    rather than crash in the strict digest or pass a smuggled metric."""
    path = tmp_path / "nan.json"
    path.write_text('{"kind": "x", "score": NaN, "receipt_sha256": "' + "0" * 64 + '"}')
    result = verify_receipt_file(path)
    assert result["valid"] is False
    assert any(e.startswith("receipt_unreadable") for e in result["errors"])


def test_file_with_infinity_literal_is_malformed(tmp_path: Path) -> None:
    path = tmp_path / "inf.json"
    body = seal_receipt({"kind": "x"})
    path.write_text(json.dumps(body).replace('"kind"', '"score": Infinity, "kind"'))
    result = verify_receipt_file(path)
    assert result["valid"] is False
    assert any(e.startswith("receipt_unreadable") for e in result["errors"])


def test_truncated_and_scalar_files_fail_closed(tmp_path: Path) -> None:
    truncated = tmp_path / "t.json"
    truncated.write_text('{"kind": "x", "receipt_sha256": "abc')
    assert verify_receipt_file(truncated)["valid"] is False
    scalar = tmp_path / "s.json"
    scalar.write_text("[1, 2, 3]")
    result = verify_receipt_file(scalar)
    assert result["valid"] is False
    assert "receipt_not_object" in result["errors"]


# --- seal layer --------------------------------------------------------------


def test_v2_field_tamper_without_reseal_breaks_seal() -> None:
    receipt = _generic_v2()
    receipt["payload"]["rows"] = [9, 9, 9]
    _invalid_with(receipt, "receipt_sha256_mismatch")


def test_v2_seal_stripped_entirely() -> None:
    receipt = _generic_v2()
    del receipt["receipt_sha256"]
    _invalid_with(receipt, "receipt_sha256_missing_or_invalid")


def test_programmatic_nan_body_never_crashes() -> None:
    """A NaN value reaching the programmatic verifier used to raise from the
    strict digest; now it fails closed through the seal check."""
    result = verify_receipt_payload(
        {"kind": "x", "score": float("nan"), "receipt_sha256": "0" * 64}
    )
    assert result["valid"] is False
    assert any(e.startswith("receipt_sha256_") for e in result["errors"])


# --- semantic layer: mutations that survive an honest reseal ------------------


def test_v2_verdict_outside_enum_fails_even_resealed() -> None:
    receipt = seal_receipt({**_generic_v2(), "verdict": "verified"})
    _invalid_with(receipt, "receipt_v2_schema")


def test_v2_dropped_schema_still_fails_closed() -> None:
    """schema_version=2 alone routes to v2; a missing ``schema`` fails the model."""
    receipt = {k: v for k, v in _generic_v2().items() if k != "schema"}
    _invalid_with(seal_receipt(receipt), "receipt_v2_schema")


def test_v2_downgrade_to_v1_dispatch_is_flagged() -> None:
    """Stripping both schema markers and resealing must not launder a v2
    envelope through the weaker v1 contract."""
    downgraded = {k: v for k, v in _generic_v2().items() if k not in ("schema", "schema_version")}
    downgraded["schema_version"] = 1
    result = verify_receipt_payload(seal_receipt(downgraded))
    assert result["valid"] is False
    assert "possible_v2_downgrade" in result["errors"]


def test_v2_downgrade_guard_does_not_fire_on_bare_v1_blobs() -> None:
    """Residual boundary: stripping every v2 evidence field leaves a generic
    sealed dict — honestly a sealed dict, flagged by nothing else."""
    bare = {"kind": "tamper_matrix_lane", "note": "plain v1-style blob"}
    result = verify_receipt_payload(seal_receipt(bare))
    assert result["valid"] is True
    assert "possible_v2_downgrade" not in result["errors"]


def test_v1_receipts_with_shared_field_names_are_not_flagged() -> None:
    """``code_sha256``/``environment`` alone appear on legit v1 receipts; the
    downgrade guard fires only on the joint envelope signature."""
    for extra in (
        {"code_sha256": "b" * 64},
        {"environment": {"platform": "x", "python": "y"}},
        {"dataset_hash": "c" * 64, "params_hash": "d" * 64},
    ):
        blob = seal_receipt({"kind": "legacy_lane", **extra})
        result = verify_receipt_payload(blob)
        assert "possible_v2_downgrade" not in result["errors"], extra


def test_v2_environment_tamper_resealed() -> None:
    receipt = _generic_v2()
    env = dict(receipt["environment"])
    env["packages"] = {**env["packages"], "numpy": "0.0.0"}
    _invalid_with(seal_receipt({**receipt, "environment": env}), "environment_fingerprint_mismatch")


def test_v2_code_files_tamper_resealed() -> None:
    receipt = _generic_v2()
    code_files = {**receipt["code_files"], "planted.py": "0" * 64}
    _invalid_with(seal_receipt({**receipt, "code_files": code_files}), "code_sha256_mismatch")


def test_v2_payload_forbidden_metric_resealed() -> None:
    receipt = _generic_v2()
    nested = {**receipt["payload"], "detail": {"flag_high_sharpe": 4.2}}
    _invalid_with(
        seal_receipt({**receipt, "payload": nested}),
        "payload_forbidden_metrics",
    )


def test_v2_inner_data_label_contradiction_resealed() -> None:
    receipt = _generic_v2()
    _invalid_with(
        seal_receipt({**receipt, "payload": {**receipt["payload"], "data_label": "REAL"}}),
        "payload_data_label_mismatch",
    )


# --- fleet lane: bound digests re-derive from the payload ---------------------


def test_fleet_v2_param_tamper_resealed(tmp_path: Path) -> None:
    receipt = _fleet_v2(tmp_path)
    payload = {**receipt["payload"], "seed": 999}
    _invalid_with(seal_receipt({**receipt, "payload": payload}), "params_hash_mismatch")


def test_fleet_v2_shard_digest_tamper_resealed(tmp_path: Path) -> None:
    receipt = _fleet_v2(tmp_path)
    payload = dict(receipt["payload"])
    shards = {k: dict(v) for k, v in payload["shards"].items()}
    shards["iid_gaussian"]["x_sha256"] = "f" * 64
    payload["shards"] = shards
    _invalid_with(seal_receipt({**receipt, "payload": payload}), "dataset_hash_mismatch")


def test_fleet_v2_inner_error_count_forgery_resealed(tmp_path: Path) -> None:
    """n_error_rows must re-derive from the embedded rows, not the claim."""
    receipt = _fleet_v2(tmp_path)
    payload = dict(receipt["payload"])
    payload["n_error_rows"] = payload["n_error_rows"] + 1
    _invalid_with(seal_receipt({**receipt, "payload": payload}), "payload_n_error_rows_mismatch")


def test_fleet_v2_payload_dropped_fails_schema(tmp_path: Path) -> None:
    receipt = _fleet_v2(tmp_path)
    receipt["payload"] = {}
    _invalid_with(seal_receipt(receipt), "receipt_v2_schema")


# --- fleet v1 contract --------------------------------------------------------


def test_fleet_v1_error_count_forgery_resealed(tmp_path: Path) -> None:
    receipt = _fleet_v1(tmp_path)
    _invalid_with(seal_receipt({**receipt, "n_error_rows": 7}), "n_error_rows_mismatch")


def test_fleet_v1_row_count_forgery_resealed(tmp_path: Path) -> None:
    receipt = _fleet_v1(tmp_path)
    _invalid_with(seal_receipt({**receipt, "n_rows": 999}), "n_rows_mismatch")


def test_fleet_v1_data_label_forgery_resealed(tmp_path: Path) -> None:
    receipt = _fleet_v1(tmp_path)
    _invalid_with(seal_receipt({**receipt, "data_label": "REAL"}), "data_label_not_synthetic")


def test_fleet_v1_results_dropped(tmp_path: Path) -> None:
    receipt = _fleet_v1(tmp_path)
    del receipt["results"]
    _invalid_with(seal_receipt(receipt), "results_missing_or_empty")


def test_fleet_v1_forbidden_metric_resealed(tmp_path: Path) -> None:
    receipt = _fleet_v1(tmp_path)
    _invalid_with(seal_receipt({**receipt, "headline_sharpe": 9.9}), "forbidden_metric_keys")

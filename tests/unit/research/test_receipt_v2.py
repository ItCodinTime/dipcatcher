"""receipt.v2: unified envelope schema, environment fingerprint, verifier.

All data here is SYNTHETIC — correctness evidence, never market evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.research.fleet_eval import (
    fleet_head_factories,
    run_distribution_fleet,
    write_fleet_receipt,
)
from quant_fund.research.receipt_v2 import (
    RECEIPT_V2_SCHEMA_VERSION,
    RECEIPT_V2_VERDICTS,
    ReceiptV2,
    build_receipt_v2,
    code_fingerprint,
    environment_fingerprint,
    receipt_v2_json_schema,
    seal_receipt,
    verify_receipt_file,
    verify_receipt_payload,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

TAUS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def _small_fleet_receipt(seed: int = 5) -> dict[str, Any]:
    _, receipt = run_distribution_fleet(
        fleet_head_factories(TAUS, seed, ["empirical", "gaussian"]),
        shards=["iid_gaussian", "heavy_tail"],
        n_train=96,
        n_eval=48,
        seed=seed,
        taus=TAUS,
    )
    return receipt


def _v2_receipt(seed: int = 5) -> dict[str, Any]:
    return seal_receipt(
        build_receipt_v2(
            kind="unit_test_lane",
            data_label="SYNTHETIC",
            dataset={"rows": 4, "content_sha256": "a" * 64},
            params={"seed": seed, "taus": list(TAUS)},
            code_files=(Path(__file__),),
            verdict="pass",
            payload={"schema": "unit_test.v1", "rows": [1, 2, 3]},
        )
    )


def test_environment_fingerprint_blocks_numeric_stack() -> None:
    env = environment_fingerprint()
    for key in (
        "python",
        "implementation",
        "platform",
        "machine",
        "byteorder",
        "packages",
        "blas",
        "lapack",
        "threadpools",
        "fingerprint_sha256",
    ):
        assert key in env, key
    for package in ("numpy", "polars", "scipy"):
        assert env["packages"][package] not in ("", "UNAVAILABLE")
    assert env["blas"]["name"] and env["lapack"]["name"]
    assert env["byteorder"] == "little"
    digest = env["fingerprint_sha256"]
    body = {k: v for k, v in env.items() if k != "fingerprint_sha256"}
    assert digest == hash_bytes(canonical_json_bytes(body))
    # Deterministic within a process: two captures fingerprint identically.
    assert environment_fingerprint()["fingerprint_sha256"] == digest


def test_environment_threadpools_record_loaded_blas() -> None:
    for pool in environment_fingerprint()["threadpools"]:
        assert pool["user_api"] == "blas"
        assert pool["num_threads"] >= 1
        assert pool["internal_api"]


def test_code_fingerprint_binds_file_map() -> None:
    code = code_fingerprint([Path(__file__)])
    assert set(code["files"]) == {"test_receipt_v2.py"}
    assert code["code_sha256"] == hash_bytes(canonical_json_bytes(code["files"]))
    with pytest.raises(ValueError):
        code_fingerprint([])


def test_build_receipt_v2_validates_itself() -> None:
    receipt = _v2_receipt()
    ReceiptV2.model_validate(receipt)
    assert receipt["schema"] == "receipt.v2"
    assert receipt["schema_version"] == RECEIPT_V2_SCHEMA_VERSION
    assert receipt["live_pnl_claim"] is False
    assert receipt["verdict"] in RECEIPT_V2_VERDICTS
    assert receipt["receipt_sha256"] == hash_bytes(
        canonical_json_bytes({k: v for k, v in receipt.items() if k != "receipt_sha256"})
    )


def test_build_receipt_v2_rejects_bad_verdict() -> None:
    with pytest.raises(Exception, match="verdict"):
        build_receipt_v2(
            kind="unit_test_lane",
            data_label="SYNTHETIC",
            dataset={},
            params={},
            code_files=(Path(__file__),),
            verdict="moonshot",
            payload={"x": 1},
        )


def test_verify_v2_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(_v2_receipt(), indent=2, sort_keys=True))
    result = verify_receipt_file(path)
    assert result["valid"] is True, result["errors"]
    assert result["schema"] == "receipt.v2"
    assert result["digest_convention"] == "canonical_json"
    assert result["verdict"] == "pass"


def test_verify_v2_detects_tamper(tmp_path: Path) -> None:
    receipt = _v2_receipt()
    receipt["payload"]["rows"] = [9, 9, 9]
    result = verify_receipt_payload(receipt)
    assert result["valid"] is False
    assert "receipt_sha256_mismatch" in result["errors"]


def test_verify_v2_detects_resealed_field_forgery() -> None:
    """Re-sealing a tampered body cannot hide envelope/payload disagreement."""
    receipt = _v2_receipt()
    receipt["payload"]["data_label"] = "REAL"
    receipt = seal_receipt(receipt)  # honest re-seal of dishonest body
    result = verify_receipt_payload(receipt)
    assert result["valid"] is False
    assert "payload_data_label_mismatch" in result["errors"]

    receipt = _v2_receipt()
    receipt["payload"]["live_pnl_claim"] = True
    receipt = seal_receipt(receipt)
    result = verify_receipt_payload(receipt)
    assert result["valid"] is False
    assert "payload_live_pnl_claim_not_false" in result["errors"]


def test_verify_v2_rejects_live_pnl_claim() -> None:
    receipt = _v2_receipt()
    receipt["live_pnl_claim"] = True
    result = verify_receipt_payload(receipt)
    assert result["valid"] is False
    assert any("live_pnl_claim" in err for err in result["errors"])


def test_verify_v2_rejects_forbidden_metric_key() -> None:
    receipt = _v2_receipt()
    body = dict(receipt["payload"])
    body["paper_sharpe"] = 9.9
    receipt = seal_receipt({**receipt, "payload": body})
    result = verify_receipt_payload(receipt)
    assert result["valid"] is False
    assert "payload_forbidden_metrics" in result["errors"]


def test_verify_v2_detects_environment_drift() -> None:
    receipt = _v2_receipt()
    env = dict(receipt["environment"])
    env["packages"] = {**env["packages"], "numpy": "0.0.0"}
    receipt = seal_receipt({**receipt, "environment": env})
    result = verify_receipt_payload(receipt)
    assert result["valid"] is False
    assert "environment_fingerprint_mismatch" in result["errors"]


def test_verify_v2_detects_code_map_drift() -> None:
    receipt = _v2_receipt()
    code_files = {**receipt["code_files"], "extra.py": "0" * 64}
    receipt = seal_receipt({**receipt, "code_files": code_files})
    result = verify_receipt_payload(receipt)
    assert result["valid"] is False
    assert "code_sha256_mismatch" in result["errors"]


def test_fleet_receipt_v2_round_trip(tmp_path: Path) -> None:
    receipt = _small_fleet_receipt()
    path = write_fleet_receipt(receipt, tmp_path, receipt_version=2)
    assert path.name.startswith("fleet_eval_")
    payload = json.loads(path.read_text())
    assert payload["schema"] == "receipt.v2"
    assert payload["schema_version"] == 2
    assert payload["kind"] == "distribution_fleet_eval"
    assert payload["data_label"] == "SYNTHETIC"
    assert payload["live_pnl_claim"] is False
    assert payload["verdict"] == "pass"
    assert payload["payload"]["schema"] == "fleet_eval.v1"
    assert payload["payload"]["results"]
    result = verify_receipt_file(path)
    assert result["valid"] is True, result["errors"]


def test_fleet_receipt_v2_verdict_fails_on_error_row(tmp_path: Path) -> None:
    def bad_factory() -> Any:
        class _Bad:
            def fit(self, x: np.ndarray, y: np.ndarray) -> Any:
                raise ValueError("planted fit failure")

            def predict(self, x: np.ndarray) -> np.ndarray:
                raise RuntimeError("unreachable")

            def metadata(self) -> Any:
                return None

        return _Bad()

    _, receipt = run_distribution_fleet(
        {**fleet_head_factories(TAUS, 0, ["gaussian"]), "broken": bad_factory},
        shards=["iid_gaussian"],
        n_train=64,
        n_eval=32,
        taus=TAUS,
    )
    path = write_fleet_receipt(receipt, tmp_path, receipt_version=2)
    payload = json.loads(path.read_text())
    assert payload["verdict"] == "fail"
    assert verify_receipt_file(path)["valid"] is True
    # Forging a pass verdict over the same error rows is caught even after
    # an honest re-seal: the verifier re-derives the verdict from the payload.
    forged = seal_receipt({**payload, "verdict": "pass"})
    result = verify_receipt_payload(forged)
    assert result["valid"] is False
    assert "verdict_mismatch" in result["errors"]


def test_fleet_receipt_v2_detects_rebound_dataset(tmp_path: Path) -> None:
    """Swapping the embedded payload re-seals fine but the bound digests fail."""
    receipt = _small_fleet_receipt(seed=5)
    path = write_fleet_receipt(receipt, tmp_path, receipt_version=2)
    envelope = json.loads(path.read_text())
    other = _small_fleet_receipt(seed=6)
    forged = seal_receipt({**envelope, "payload": other})
    result = verify_receipt_payload(forged)
    assert result["valid"] is False
    assert "dataset_hash_mismatch" in result["errors"]
    assert "params_hash_mismatch" in result["errors"]


def test_verify_receipt_accepts_fleet_v1(tmp_path: Path) -> None:
    path = write_fleet_receipt(_small_fleet_receipt(), tmp_path)
    result = verify_receipt_file(path)
    assert result["valid"] is True, result["errors"]
    assert result["schema"] == "fleet_eval.v1"
    assert result["digest_convention"] == "canonical_json"


def test_verify_receipt_accepts_strict_json_v1(tmp_path: Path) -> None:
    """real_benchmark-style seal: sorted compact JSON, ASCII-escaped.

    The non-ASCII note is load-bearing: ``canonical_json_bytes`` emits UTF-8
    (``ensure_ascii=False``) while the strict digest escapes it, so only the
    strict convention can match this seal.
    """
    body = {"kind": "unit_test", "schema_version": 1, "note": "café"}
    sealed = {
        **body,
        "receipt_sha256": hash_bytes(
            json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ),
    }
    path = tmp_path / "strict.json"
    path.write_text(json.dumps(sealed))
    result = verify_receipt_file(path)
    assert result["valid"] is True, result["errors"]
    assert result["digest_convention"] == "strict_json"


def test_verify_receipt_rejects_unsealed_and_unreadable(tmp_path: Path) -> None:
    unsealed = tmp_path / "unsealed.json"
    unsealed.write_text(json.dumps({"kind": "x"}))
    result = verify_receipt_file(unsealed)
    assert result["valid"] is False
    assert "receipt_sha256_missing_or_invalid" in result["errors"]
    missing = verify_receipt_file(tmp_path / "nope.json")
    assert missing["valid"] is False
    assert any(e.startswith("receipt_unreadable") for e in missing["errors"])
    assert verify_receipt_payload([1, 2, 3])["valid"] is False


def test_fleet_v1_contract_violations_still_refused(tmp_path: Path) -> None:
    receipt = _small_fleet_receipt()
    with pytest.raises(ValueError, match="synthetic research contract"):
        write_fleet_receipt({**receipt, "data_label": "REAL"}, tmp_path)
    with pytest.raises(ValueError, match="synthetic research contract"):
        write_fleet_receipt({**receipt, "data_label": "REAL"}, tmp_path, receipt_version=2)
    with pytest.raises(ValueError, match="receipt_version"):
        write_fleet_receipt(receipt, tmp_path, receipt_version=3)


def test_schema_file_mirrors_pydantic_model() -> None:
    """The published JSON-schema file and the model must not drift."""
    schema = receipt_v2_json_schema()
    model_schema = ReceiptV2.model_json_schema()
    required_file = set(schema["required"])
    required_model = set(model_schema["required"])
    # The file's required set is the unsealed envelope; receipt_sha256 stays
    # optional in both (it is stamped by the seal step).
    assert required_file == required_model - {"receipt_sha256"}
    assert set(schema["properties"]) == set(model_schema["properties"])
    assert schema["properties"]["schema"]["const"] == "receipt.v2"
    assert schema["properties"]["schema_version"]["const"] == 2
    assert schema["properties"]["live_pnl_claim"]["const"] is False
    assert schema["properties"]["verdict"]["enum"] == list(RECEIPT_V2_VERDICTS)


def test_cli_fleet_receipt_version_two(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "fleet",
            "--shards",
            "iid_gaussian",
            "--models",
            "empirical,gaussian",
            "--n-train",
            "64",
            "--n-eval",
            "32",
            "--seed",
            "11",
            "--out-dir",
            str(tmp_path),
            "--receipt-version",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    written = list(tmp_path.glob("fleet_eval_*.json"))
    assert len(written) == 1
    payload = json.loads(written[0].read_text())
    assert payload["schema"] == "receipt.v2"
    verify = CliRunner().invoke(app, ["verify-receipt", str(written[0])])
    assert verify.exit_code == 0, verify.output
    assert '"valid": true' in verify.output


def test_cli_fleet_rejects_bad_receipt_version(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "fleet",
            "--shards",
            "iid_gaussian",
            "--models",
            "empirical",
            "--n-train",
            "64",
            "--n-eval",
            "32",
            "--out-dir",
            str(tmp_path),
            "--receipt-version",
            "7",
        ],
    )
    assert result.exit_code != 0


def test_cli_verify_receipt_fails_closed(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"kind": "forged"}))
    result = CliRunner().invoke(app, ["verify-receipt", str(bad)])
    assert result.exit_code == 1
    assert '"valid": false' in result.output

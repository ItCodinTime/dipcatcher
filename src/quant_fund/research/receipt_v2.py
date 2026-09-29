"""Unified ``receipt.v2`` envelope for sealed research evidence (P7.2 + P7.4).

Every research lane (eval, incumbent, carry, paper) seals the same envelope:
dataset/code/params digests, an environment fingerprint (Python, NumPy,
Polars, SciPy plus the BLAS/LAPACK build actually loaded), a mandatory
``live_pnl_claim`` flag fixed ``false``, a verdict, and a canonical SHA-256
seal. ``receipt_v2.schema.json`` beside this module is the published
contract; the pydantic models here are its executable form and stay in sync
with it (test_receipt_v2 checks the drift).

``verify_receipt_file`` validates either generation structurally and checks
hash consistency. v1 is verified under both digests conventions in the repo:
the canonical ``canonical_json_bytes`` form and the strict ``json.dumps``
form used by ``real_benchmark``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.utils.hashing import SHA256_HEX_LENGTH, canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

RECEIPT_V2_SCHEMA = "receipt.v2"
RECEIPT_V2_SCHEMA_VERSION: Literal[2] = 2
RECEIPT_V2_VERDICTS = ("pass", "fail", "blocked")
RECEIPT_V2_SCHEMA_FILE = "receipt_v2.schema.json"

# The numeric stack a determinism sweep (P7.4) compares across machines.
REQUIRED_ENV_PACKAGES = frozenset({"numpy", "polars", "scipy"})

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


def _is_sha256(value: object) -> bool:
    """Accept only lowercase hexadecimal SHA-256 digests."""
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _timestamp_valid(value: object) -> bool:
    """Require an ISO-8601 timestamp carrying an explicit timezone."""
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


class BlasDependency(BaseModel):
    """One BLAS/LAPACK build dependency as recorded by ``np.__config__.CONFIG``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    found: bool
    version: str | None = None


class ThreadpoolInfo(BaseModel):
    """One loaded BLAS threadpool (``threadpoolctl.threadpool_info`` row)."""

    model_config = ConfigDict(extra="forbid")

    prefix: str = Field(min_length=1)
    user_api: str = Field(min_length=1)
    internal_api: str = Field(min_length=1)
    version: str | None = None
    num_threads: int = Field(ge=1)
    threading_layer: str | None = None
    architecture: str | None = None


class ReceiptEnvironment(BaseModel):
    """Per-receipt environment block: interpreter, numeric stack, BLAS/LAPACK."""

    model_config = ConfigDict(extra="forbid")

    python: str = Field(min_length=1)
    implementation: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    machine: str = Field(min_length=1)
    byteorder: Literal["little", "big"]
    packages: dict[str, str]
    blas: BlasDependency
    lapack: BlasDependency
    threadpools: list[ThreadpoolInfo]
    fingerprint_sha256: Sha256Hex

    @model_validator(mode="after")
    def packages_cover_numeric_stack(self) -> ReceiptEnvironment:
        missing = sorted(REQUIRED_ENV_PACKAGES - self.packages.keys())
        if missing:
            raise ValueError(f"environment.packages missing: {','.join(missing)}")
        if any(not name.strip() for name in self.packages):
            raise ValueError("environment.packages names must be nonempty")
        if any(not value.strip() for value in self.packages.values()):
            raise ValueError("environment.packages versions must be nonempty")
        return self


class ReceiptV2(BaseModel):
    """The unified sealed-receipt envelope.

    ``payload`` holds the lane-specific body (e.g. the ``fleet_eval.v1``
    receipt verbatim). ``receipt_sha256`` is added by the writer's seal step
    and verified, not produced, by this model. The JSON key is ``schema``;
    the attribute is ``schema_id`` because ``schema`` shadows a BaseModel
    member.
    """

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["receipt.v2"] = Field(alias="schema")
    schema_version: Literal[2]
    kind: str = Field(min_length=1)
    data_label: str = Field(min_length=1)
    generated_at: str
    git_revision: str = Field(min_length=1)
    dataset_hash: Sha256Hex
    params_hash: Sha256Hex
    code_sha256: Sha256Hex
    code_files: dict[str, Sha256Hex] = Field(min_length=1)
    environment: ReceiptEnvironment
    live_pnl_claim: Literal[False]
    verdict: Literal["pass", "fail", "blocked"]
    payload: dict[str, Any]
    receipt_sha256: Sha256Hex | None = None

    @model_validator(mode="after")
    def generated_at_has_timezone(self) -> ReceiptV2:
        if not _timestamp_valid(self.generated_at):
            raise ValueError("generated_at must be an ISO-8601 timestamp with timezone")
        if not self.payload:
            raise ValueError("payload must be a nonempty object")
        return self


def _package_versions(names: tuple[str, ...]) -> dict[str, str]:
    packages: dict[str, str] = {}
    for name in names:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "UNAVAILABLE"
    return packages


def _blas_dependency(kind: Literal["blas", "lapack"]) -> dict[str, Any]:
    """Normalize ``np.__config__.CONFIG['Build Dependencies'][kind]``."""
    config = getattr(np.__config__, "CONFIG", None)
    raw: Mapping[str, Any] = {}
    if isinstance(config, Mapping):
        deps = config.get("Build Dependencies")
        if isinstance(deps, Mapping):
            candidate = deps.get(kind)
            if isinstance(candidate, Mapping):
                raw = candidate
    raw_version = raw.get("version")
    return {
        "name": str(raw.get("name") or "unknown"),
        "found": bool(raw.get("found", False)),
        "version": None if raw_version in (None, "unknown") else str(raw_version),
    }


def _blas_threadpools() -> list[dict[str, Any]]:
    """Loaded BLAS pools via threadpoolctl when it is installed (sklearn dep).

    The fingerprint records the library actually mapped into the process and
    its thread count — the two facts a cross-machine determinism sweep needs.
    """
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return []
    pools: list[dict[str, Any]] = []
    for info in threadpool_info():
        if not isinstance(info, Mapping) or info.get("user_api") != "blas":
            continue
        num_threads = info.get("num_threads")
        if not isinstance(num_threads, int) or isinstance(num_threads, bool) or num_threads < 1:
            continue
        row: dict[str, Any] = {
            "prefix": str(info.get("prefix") or "unknown"),
            "user_api": "blas",
            "internal_api": str(info.get("internal_api") or "unknown"),
            "num_threads": num_threads,
        }
        for key in ("version", "threading_layer", "architecture"):
            value = info.get(key)
            if isinstance(value, str) and value:
                row[key] = value
        pools.append(row)
    pools.sort(key=lambda row: (str(row["prefix"]), int(row["num_threads"])))
    return pools


def environment_fingerprint() -> dict[str, Any]:
    """Capture the runtime numeric stack; embed its own canonical digest.

    ``fingerprint_sha256`` covers every other field in the block, so a
    determinism sweep compares one digest instead of diffing nested config.
    """
    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "packages": _package_versions(("numpy", "polars", "scipy")),
        "blas": _blas_dependency("blas"),
        "lapack": _blas_dependency("lapack"),
        "threadpools": _blas_threadpools(),
    }
    environment["fingerprint_sha256"] = hash_bytes(canonical_json_bytes(environment))
    return environment


def code_fingerprint(paths: tuple[Path, ...] | list[Path]) -> dict[str, Any]:
    """Map each writer source file to its SHA-256 and digest the map.

    Returns ``{"files": {name: sha256}, "code_sha256": digest-of-map}`` so a
    receipt can carry both the audit trail and one bound digest.
    """
    files: dict[str, str] = {}
    for raw in paths:
        path = Path(raw)
        if path.name in files:
            raise ValueError(f"duplicate code file name: {path.name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not files:
        raise ValueError("code_fingerprint requires at least one file")
    return {"files": files, "code_sha256": hash_bytes(canonical_json_bytes(files))}


def build_receipt_v2(
    *,
    kind: str,
    data_label: str,
    dataset: Mapping[str, Any],
    params: Mapping[str, Any],
    code_files: tuple[Path, ...] | list[Path],
    verdict: str,
    payload: Mapping[str, Any],
    generated_at: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Build the unsealed ``receipt.v2`` envelope and validate it fail-closed.

    ``dataset``/``params`` are canonicalized to digests — callers pass the
    identity of what was evaluated (e.g. per-shard content hashes) and the
    run parameters, not the raw arrays.
    """
    code = code_fingerprint(list(code_files))
    stamped = generated_at if generated_at is not None else datetime.now(UTC).isoformat()
    receipt: dict[str, Any] = {
        "schema": RECEIPT_V2_SCHEMA,
        "schema_version": RECEIPT_V2_SCHEMA_VERSION,
        "kind": str(kind),
        "data_label": str(data_label),
        "generated_at": stamped,
        "git_revision": revision if revision is not None else git_revision(),
        "dataset_hash": hash_bytes(canonical_json_bytes(dataset)),
        "params_hash": hash_bytes(canonical_json_bytes(params)),
        "code_sha256": code["code_sha256"],
        "code_files": code["files"],
        "environment": environment_fingerprint(),
        "live_pnl_claim": False,
        "verdict": verdict,
        "payload": json.loads(canonical_json_bytes(dict(payload))),
    }
    ReceiptV2.model_validate(receipt)
    return receipt


def seal_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize a receipt body and stamp ``receipt_sha256`` over it."""
    canonical = json.loads(canonical_json_bytes(dict(receipt)))
    canonical.pop("receipt_sha256", None)
    return {**canonical, "receipt_sha256": hash_bytes(canonical_json_bytes(canonical))}


def _canonical_digest(body: Mapping[str, Any]) -> str:
    return hash_bytes(canonical_json_bytes(dict(body)))


def _strict_digest(body: Mapping[str, Any]) -> str:
    """The ``real_benchmark`` seal: sorted compact JSON, ASCII-escaped."""
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class ReceiptVerification(TypedDict):
    """``verify-receipt`` result. ``kind`` is untrusted until errors is empty."""

    valid: bool
    path: str
    schema: str
    kind: object
    verdict: object
    digest_convention: str | None
    errors: list[str]


def _result(
    path: Path, payload: object, digest_convention: str | None, errors: list[str]
) -> ReceiptVerification:
    body = payload if isinstance(payload, dict) else {}
    schema_field = body.get("schema", body.get("schema_version", "unknown"))
    return {
        "valid": not errors,
        "path": str(path),
        "schema": str(schema_field),
        "kind": body.get("kind"),
        "verdict": body.get("verdict"),
        "digest_convention": digest_convention,
        "errors": errors,
    }


def _seal_errors(payload: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    """Check ``receipt_sha256``; report which digest convention matched."""
    seal = payload.get("receipt_sha256")
    if not _is_sha256(seal):
        return None, ["receipt_sha256_missing_or_invalid"]
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if _canonical_digest(body) == seal:
        return "canonical_json", []
    if _strict_digest(body) == seal:
        return "strict_json", []
    return None, ["receipt_sha256_mismatch"]


def _env_fingerprint_errors(environment: object) -> list[str]:
    if not isinstance(environment, dict):
        return ["environment_invalid"]
    stamp = environment.get("fingerprint_sha256")
    if not _is_sha256(stamp):
        return ["environment_fingerprint_missing"]
    body = {key: value for key, value in environment.items() if key != "fingerprint_sha256"}
    if _canonical_digest(body) != stamp:
        return ["environment_fingerprint_mismatch"]
    return []


def _code_consistency_errors(payload: Mapping[str, Any]) -> list[str]:
    files = payload.get("code_files")
    if not isinstance(files, dict) or not files:
        return ["code_files_invalid"]
    if _canonical_digest(files) != payload.get("code_sha256"):
        return ["code_sha256_mismatch"]
    return []


def _looks_like_fleet_eval(body: object) -> bool:
    """Structural fingerprint of a fleet_eval receipt — schema/kind agnostic.

    Dispatch on content, not the claimed ``schema``/``kind``: those fields are
    attacker-controlled, and a reseal costs nothing (the digest is a public
    sha256), so renaming them must not evade the deep contract checks.
    """
    if not (
        isinstance(body, Mapping)
        and isinstance(body.get("results"), list)
        and isinstance(body.get("models"), list)
        and isinstance(body.get("shards"), dict)
        and "n_eval" in body
    ):
        return False
    # Adjacent eval lanes (calibration_eval.v1, coherence_eval.v1) share the
    # results/models/shards envelope — the discriminating signature is the
    # scored grid itself: fleet rows carry per-tau pinball cells and crps.
    return any(
        isinstance(row, Mapping)
        and row.get("status") is not None
        and ("crps" in row or any(key.startswith("pinball_") for key in row))
        for row in body["results"]
    )


def _looks_like_v2_envelope(payload: Mapping[str, Any]) -> bool:
    """Structural fingerprint of the receipt.v2 envelope."""
    return (
        isinstance(payload.get("payload"), Mapping)
        and isinstance(payload.get("environment"), Mapping)
        and isinstance(payload.get("code_files"), dict)
        and isinstance(payload.get("dataset_hash"), str)
        and isinstance(payload.get("params_hash"), str)
    )


def _kind_consistency_errors(payload: Mapping[str, Any]) -> list[str]:
    """Lane-specific re-derivation of the bound digests, where defined."""
    inner = payload.get("payload")
    looks_fleet = _looks_like_fleet_eval(inner)
    if payload.get("kind") == "distribution_fleet_eval" or looks_fleet:
        from quant_fund.research.fleet_eval import fleet_v2_consistency_errors

        errors = fleet_v2_consistency_errors(payload)
        if looks_fleet and payload.get("kind") != "distribution_fleet_eval":
            errors = [*errors, "kind_fingerprint_mismatch"]
        return errors
    return []


def _verify_v2(path: Path, payload: Mapping[str, Any]) -> ReceiptVerification:
    errors: list[str] = []
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    try:
        ReceiptV2.model_validate(payload)
    except ValidationError as exc:
        for issue in exc.errors():
            location = ".".join(str(part) for part in issue["loc"]) or "envelope"
            errors.append(f"receipt_v2_schema:{location}:{issue['msg']}")
        return _result(path, payload, None, errors)
    convention, seal_errors = _seal_errors(payload)
    errors.extend(seal_errors)
    errors.extend(_env_fingerprint_errors(body.get("environment")))
    errors.extend(_code_consistency_errors(body))
    payload_body = body.get("payload")
    if isinstance(payload_body, Mapping):
        # Same exemption as the writers: the envelope honesty flag carries a
        # forbidden token but is required, so it is excluded from the scan.
        scanned = {key: value for key, value in payload_body.items() if key != "live_pnl_claim"}
        if not family_blob_forbidden_metrics_absent(scanned):
            errors.append("payload_forbidden_metrics")
        # Envelope/payload agreement: a payload that echoes either honesty
        # field must not contradict the sealed envelope under a fresh seal.
        inner_label = payload_body.get("data_label")
        if inner_label is not None and inner_label != body.get("data_label"):
            errors.append("payload_data_label_mismatch")
        inner_claim = payload_body.get("live_pnl_claim")
        if inner_claim is not None and inner_claim is not False:
            errors.append("payload_live_pnl_claim_not_false")
    errors.extend(_kind_consistency_errors(body))
    return _result(path, payload, convention, errors)


def _verify_v1(path: Path, payload: Mapping[str, Any]) -> ReceiptVerification:
    errors: list[str] = []
    convention, seal_errors = _seal_errors(payload)
    errors.extend(seal_errors)
    claim = payload.get("live_pnl_claim")
    if claim is not None and claim is not False:
        errors.append("live_pnl_claim_not_false")
    if _looks_like_fleet_eval(payload):
        from quant_fund.research.fleet_eval import fleet_v1_contract_errors

        errors.extend(fleet_v1_contract_errors(payload))
    return _result(path, payload, convention, errors)


def verify_receipt_payload(
    payload: object, path: Path | str = Path("<memory>")
) -> ReceiptVerification:
    """Verify one parsed receipt object: structure plus hash consistency.

    ``receipt.v2`` envelopes are validated against ``ReceiptV2`` and their
    sealed digest, environment fingerprint, code-map digest, and (for known
    kinds) dataset/params digests are re-derived. Older receipts are checked
    for a consistent ``receipt_sha256`` seal under either repo convention;
    ``fleet_eval.v1`` payloads additionally get their writer's contract.
    """
    path = Path(path)
    if not isinstance(payload, dict):
        return _result(path, payload, None, ["receipt_not_object"])
    try:
        if (
            payload.get("schema") == RECEIPT_V2_SCHEMA
            or payload.get("schema_version") == 2
            or _looks_like_v2_envelope(payload)
        ):
            return _verify_v2(path, payload)
        return _verify_v1(path, payload)
    except (TypeError, ValueError, RecursionError) as exc:
        # Non-finite floats and pathological nesting make the canonical digest
        # raise; a malformed receipt must fail closed, not crash a sweep.
        return _result(path, payload, None, [f"receipt_undigestable:{exc.__class__.__name__}"])


def verify_receipt_file(path: Path | str) -> ReceiptVerification:
    """Read a receipt JSON file and verify it. Fails closed on unreadable input."""
    file_path = Path(path)
    try:
        payload: object = json.loads(file_path.read_text())
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        # ValueError covers JSONDecodeError and the >4300-digit integer limit;
        # RecursionError covers pathological nesting.
        return _result(file_path, {}, None, [f"receipt_unreadable:{exc.__class__.__name__}"])
    return verify_receipt_payload(payload, file_path)


def receipt_v2_json_schema() -> dict[str, Any]:
    """Load the published JSON-schema contract shipped beside this module."""
    from importlib.resources import files

    resource = files("quant_fund.research").joinpath(RECEIPT_V2_SCHEMA_FILE)
    return json.loads(resource.read_text(encoding="utf-8"))  # type: ignore[no-any-return]

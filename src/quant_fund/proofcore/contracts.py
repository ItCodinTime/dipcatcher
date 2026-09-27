"""PROOFCORE shared contracts.

Single source of truth for cross-workstream schemas, hashing helpers, and
error types. IMPORTS ONLY stdlib + pydantic — never import quant_fund here
(layer 1 of the PROOFCORE layering contract, DESIGN.md §1.3).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

SCHEMA_VERSION: str = "proofcore/1"
GENESIS_HASH: str = "0" * 64
HASH_HEX_LEN: int = 64
_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
_DIGEST_FIELD_NAMES = {
    "best_trial_id",
    "bundle_hash",
    "bundle_id",
    "merkle_root",
    "trial_id",
}


class _FrozenDict(dict[str, object]):
    """A JSON-serializable mapping that cannot mutate a signed payload."""

    def _immutable(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("proof payload is immutable")

    __setitem__ = _immutable  # type: ignore[assignment]
    __delitem__ = _immutable  # type: ignore[assignment]
    clear = _immutable  # type: ignore[assignment]
    pop = _immutable  # type: ignore[assignment]
    popitem = _immutable  # type: ignore[assignment]
    setdefault = _immutable  # type: ignore[assignment]
    update = _immutable  # type: ignore[assignment]
    __ior__ = _immutable  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Errors (self-contained; adapters in pit/leakage map these onto
# quant_fund.schemas.errors subclasses at their own package boundary).
# ---------------------------------------------------------------------------


class ProofcoreError(Exception):
    """Base error for all PROOFCORE packages."""


class VaultError(ProofcoreError):
    """PIT vault contract violation (write-once, manifest, schema)."""


class VaultUnavailableError(VaultError):
    """asof(t) requested before any version with known_at <= t exists."""


class ManifestError(VaultError):
    """manifest.json missing, malformed, or hash-mismatched."""


class ProofError(ProofcoreError):
    """Proof bundle construction failure."""


class ProofVerificationError(ProofError):
    """Bundle failed verification. Carries a machine-readable reason list."""

    def __init__(self, message: str, reasons: list[str] | None = None) -> None:
        super().__init__(message)
        self.reasons: list[str] = list(reasons or [])


class ProofBundleError(ProofError):
    """Bundle mint failed closed (inputs would produce an unverifiable bundle).

    Raised at MINT time, never at verify time — e.g. degenerate runs whose
    recomputed headline metrics are NaN (canonical JSON encodes NaN as null,
    which the bundle schema then rejects; ADVERSARIAL §2-H).
    """


class SignatureUnavailableError(ProofError):
    """No signing key configured; signature cannot be produced or checked."""


class LeakageError(ProofcoreError):
    """Leakage scan finding at error severity, or watchdog trip."""


class RealityFilterError(ProofcoreError):
    """Reality-filter computation failed fail-closed."""


class ProvenanceError(ProofcoreError):
    """Provenance DB write/read/chain violation."""


# ---------------------------------------------------------------------------
# Canonical hashing helpers — EVERY hash in PROOFCORE goes through these.
# ---------------------------------------------------------------------------


def canonical_json_bytes(obj: object) -> bytes:
    """Deterministic JSON: sorted keys, tight separators, UTF-8, LF-free.

    Only pydantic-model `.model_dump(mode="json")` output, primitives, and
    nested dict/list structures are legal input. Floats must be pre-rounded
    by the caller to the determinism policy precision (DESIGN.md §8.2).
    """

    def json_safe(value: object) -> object:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    return json.dumps(
        json_safe(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_hex_json(obj: object) -> str:
    return sha256_hex_bytes(canonical_json_bytes(obj))


def merkle_root_hex(leaf_hashes: list[str]) -> str:
    """Merkle root over hex digests. Empty list hashes the empty string.

    Leaves are sorted before pairing (order-independent commitment).
    Odd levels duplicate the last node. Domain-separated via b"PC:leaf:"/
    b"PC:node:" prefixes so leaf and node hashes can never collide by design.
    """
    if not leaf_hashes:
        return sha256_hex_bytes(b"")
    for h in leaf_hashes:
        if not isinstance(h, str) or _SHA256_HEX.fullmatch(h) is None:
            raise ProofError(f"merkle leaf is not a sha256 hex digest: {h!r}")
    level = [sha256_hex_bytes(b"PC:leaf:" + bytes.fromhex(h)) for h in sorted(leaf_hashes)]
    while len(level) > 1:
        nxt: list[str] = []
        for i in range(0, len(level), 2):
            left = bytes.fromhex(level[i])
            right = bytes.fromhex(level[min(i + 1, len(level) - 1)])
            nxt.append(sha256_hex_bytes(b"PC:node:" + left + right))
        level = nxt
    return level[0]


# ---------------------------------------------------------------------------
# Proof bundle v1 sub-schemas
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("*", mode="after")
    @classmethod
    def validate_digest_fields(cls, value: object, info: ValidationInfo) -> object:
        name = info.field_name
        if (
            name is not None
            and (
                name == "sha256"
                or name.endswith("_sha256")
                or name.endswith("_hash")
                or name in _DIGEST_FIELD_NAMES
            )
            and (not isinstance(value, str) or _SHA256_HEX.fullmatch(value) is None)
        ):
            raise ValueError(f"{name} must be a lowercase sha256 hex digest")
        return value


class CodeFingerprint(_Strict):
    git_revision: str = Field(min_length=7, max_length=64)
    worktree_sha256: str = Field(min_length=HASH_HEX_LEN, max_length=HASH_HEX_LEN)
    dirty: bool = Field(description="True iff worktree had uncommitted tracked changes")


class EnvFingerprint(_Strict):
    python_version: str
    python_implementation: str
    platform: str
    machine: str
    byteorder: Literal["little", "big"]
    packages: dict[str, str] = Field(
        description="name -> exact version for dipcatcher, numpy, polars, scipy, pydantic"
    )

    @model_validator(mode="after")
    def freeze_packages(self) -> Self:
        object.__setattr__(self, "packages", _FrozenDict(self.packages))
        return self


class DataAccessRecord(_Strict):
    """One recorded PIT vault asof() read."""

    dataset: str = Field(description="vault-relative dataset name, e.g. 'silver/bars'")
    asof_utc: str = Field(description="ISO-8601 UTC decision timestamp supplied by caller")
    params: dict[str, str] = Field(
        default_factory=dict, description="extra read params, str-coerced"
    )
    rows: int = Field(ge=0)
    content_sha256: str = Field(
        min_length=HASH_HEX_LEN,
        max_length=HASH_HEX_LEN,
        description="sha256 of canonical arrow/parquet payload bytes actually returned",
    )

    @field_validator("asof_utc")
    @classmethod
    def normalize_utc(cls, value: str) -> str:
        try:
            timestamp = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("asof_utc must be an ISO-8601 UTC timestamp") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() != UTC.utcoffset(timestamp):
            raise ValueError("asof_utc must include a UTC offset")
        return timestamp.astimezone(UTC).isoformat()

    @model_validator(mode="after")
    def freeze_params(self) -> Self:
        object.__setattr__(self, "params", _FrozenDict(self.params))
        return self


class DataManifestSummary(_Strict):
    reads: tuple[DataAccessRecord, ...]
    merkle_root: str = Field(min_length=HASH_HEX_LEN, max_length=HASH_HEX_LEN)
    n_reads: int = Field(ge=0)

    @model_validator(mode="after")
    def check_read_count(self) -> Self:
        if self.n_reads != len(self.reads):
            raise ValueError("n_reads does not match recorded reads")
        return self


class SignatureBlock(_Strict):
    scheme: Literal["hmac-sha256", "none"]
    key_id: str = Field(description="sha256(key)[:16] for hmac-sha256; 'unsigned' for none")
    value: str = Field(description="hex HMAC over canonical bundle bytes minus signature field")

    @model_validator(mode="after")
    def validate_signature(self) -> Self:
        if self.scheme == "none":
            if self.key_id != "unsigned" or self.value != "":
                raise ValueError("unsigned signature must use unsigned key_id and empty value")
        elif (
            re.fullmatch(r"[0-9a-f]{16}", self.key_id) is None
            or _SHA256_HEX.fullmatch(self.value) is None
        ):
            raise ValueError("hmac-sha256 signature requires hex key_id and value")
        return self


class ProofBundleV1(_Strict):
    """Signed, hash-chained proof of one backtest/research run."""

    schema_version: Literal["proofcore/1"] = SCHEMA_VERSION  # type: ignore[assignment]
    bundle_id: str = Field(min_length=HASH_HEX_LEN, max_length=HASH_HEX_LEN)
    created_utc: str
    run_kind: Literal["backtest", "research", "paper_shadow"]
    code: CodeFingerprint
    data_manifest: DataManifestSummary
    config_sha256: str = Field(min_length=HASH_HEX_LEN, max_length=HASH_HEX_LEN)
    seed: int = Field(ge=0)
    env: EnvFingerprint
    signal_log_sha256: str = Field(min_length=HASH_HEX_LEN, max_length=HASH_HEX_LEN)
    trade_log_sha256: str = Field(min_length=HASH_HEX_LEN, max_length=HASH_HEX_LEN)
    metrics_sha256: str = Field(min_length=HASH_HEX_LEN, max_length=HASH_HEX_LEN)
    metrics_recompute: dict[str, float] = Field(
        description="headline metrics independently recomputed from the trade log at mint time"
    )
    prev_bundle_hash: str = Field(min_length=HASH_HEX_LEN, max_length=HASH_HEX_LEN)
    signature: SignatureBlock

    @model_validator(mode="after")
    def freeze_metrics(self) -> Self:
        object.__setattr__(self, "metrics_recompute", _FrozenDict(self.metrics_recompute))
        return self


# ---------------------------------------------------------------------------
# PIT vault manifest schema
# ---------------------------------------------------------------------------


class PitManifestFile(_Strict):
    path: str = Field(description="vault-relative parquet path")
    sha256: str = Field(min_length=HASH_HEX_LEN, max_length=HASH_HEX_LEN)
    rows: int = Field(ge=0)
    min_known_at: str
    max_known_at: str
    min_event_time: str
    max_event_time: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not value
            or value.startswith("/")
            or "\\" in value
            or ":" in path.parts[0]
            or any(part in (".", "..") for part in value.split("/"))
            or len(path.parts) < 3
            or path.parts[-2] != "parts"
            or re.fullmatch(r"r[0-9]{7}\.parquet", path.name) is None
        ):
            raise ValueError("path must be a vault-relative dataset parts/rNNNNNNN.parquet")
        return value


class PitManifest(_Strict):
    schema_version: Literal["proofcore/1"] = SCHEMA_VERSION  # type: ignore[assignment]
    dataset: str
    created_utc: str
    revision: int = Field(ge=0, description="monotonic per-dataset append counter")
    prev_manifest_sha256: str = Field(min_length=HASH_HEX_LEN, max_length=HASH_HEX_LEN)
    files: tuple[PitManifestFile, ...]


# ---------------------------------------------------------------------------
# Leakage scan report schema
# ---------------------------------------------------------------------------


class LeakageFinding(_Strict):
    rule_id: str = Field(pattern=r"^LH0(0[1-9]|1[0-4])$")
    severity: Literal["error", "warning"]
    path: str
    line: int = Field(ge=1)
    col: int = Field(ge=0)
    message: str
    snippet: str = Field(default="")


class LeakageReport(_Strict):
    schema_version: Literal["proofcore/1"] = SCHEMA_VERSION  # type: ignore[assignment]
    created_utc: str
    scanned_files: int = Field(ge=0)
    findings: list[LeakageFinding]
    errors: int = Field(ge=0)
    warnings: int = Field(ge=0)
    report_sha256: str = Field(min_length=HASH_HEX_LEN, max_length=HASH_HEX_LEN)


# ---------------------------------------------------------------------------
# Reality filter: trial ledger row + report schemas
# ---------------------------------------------------------------------------


class TrialLedgerRow(_Strict):
    """One research trial in the deflation ledger. Append-only."""

    trial_id: str = Field(min_length=HASH_HEX_LEN, max_length=HASH_HEX_LEN)
    bundle_hash: str = Field(
        min_length=HASH_HEX_LEN,
        max_length=HASH_HEX_LEN,
        description="proof bundle that produced this trial's returns",
    )
    family: Literal["calibration", "discovery", "bound"]
    strategy: str
    cluster_id: str = Field(description="effective-trials clustering key (§W4.3)")
    created_utc: str
    n_obs: int = Field(ge=0)
    periods_per_year: float = Field(gt=0)
    sharpe_periodic: float = Field(description="per-period SR — NEVER annualized (A1 F1)")
    skew: float
    kurtosis_raw: float = Field(description="raw 4th moment / sigma^4 (Lo 2002 convention)")
    returns_sha256: str = Field(min_length=HASH_HEX_LEN, max_length=HASH_HEX_LEN)


class RealityReport(_Strict):
    schema_version: Literal["proofcore/1"] = SCHEMA_VERSION  # type: ignore[assignment]
    created_utc: str
    n_trials: int = Field(ge=0)
    n_effective_trials: float = Field(gt=0)
    best_trial_id: str
    psr: float = Field(description="unit-safe PSR vs 0 of best trial")
    min_trl_periods: float = Field(description="MinTRL in PERIODS, not years")
    dsr: float = Field(description="deflated SR with effective trials")
    pbo: float = Field(description="CSCV probability of backtest overfitting, NaN if n/a")
    pbo_logit: list[float] = Field(default_factory=list)
    spa_pvalue: float = Field(description="SPA p-value, NaN if n/a")
    bh_fdr_rejects: list[str] = Field(default_factory=list, description="trial_ids rejected at q")
    fdr_q: float = Field(gt=0, lt=1)
    verdict: Literal["pass", "deflated", "insufficient_evidence"]
    report_sha256: str = Field(min_length=HASH_HEX_LEN, max_length=HASH_HEX_LEN)

    @field_validator("bh_fdr_rejects")
    @classmethod
    def validate_rejected_trials(cls, value: list[str]) -> list[str]:
        if any(_SHA256_HEX.fullmatch(trial_id) is None for trial_id in value):
            raise ValueError("bh_fdr_rejects must contain lowercase sha256 trial ids")
        return value

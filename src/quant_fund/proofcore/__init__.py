"""PROOFCORE: proof-carrying, leakage-proof, statistically honest runtime."""

from __future__ import annotations

from quant_fund.proofcore.contracts import (
    SCHEMA_VERSION,
    CodeFingerprint,
    DataAccessRecord,
    DataManifestSummary,
    EnvFingerprint,
    LeakageFinding,
    LeakageReport,
    PitManifest,
    PitManifestFile,
    ProofBundleV1,
    ProofcoreError,
    RealityReport,
    SignatureBlock,
    TrialLedgerRow,
    canonical_json_bytes,
    merkle_root_hex,
    sha256_hex_bytes,
)

__all__ = [
    "SCHEMA_VERSION",
    "CodeFingerprint",
    "DataAccessRecord",
    "DataManifestSummary",
    "EnvFingerprint",
    "LeakageFinding",
    "LeakageReport",
    "PitManifest",
    "PitManifestFile",
    "ProofBundleV1",
    "ProofcoreError",
    "RealityReport",
    "SignatureBlock",
    "TrialLedgerRow",
    "canonical_json_bytes",
    "merkle_root_hex",
    "sha256_hex_bytes",
]

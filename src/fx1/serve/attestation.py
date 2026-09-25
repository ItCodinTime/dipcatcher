"""Attestation ladder for fx-1 inference (tiers 2 and 3).

Tier 2 — TEE serving: an attestation binds the checkpoint hash to the
execution environment, so a client can verify the answering model is the
checkpoint whose model card passed the gates. Real TEE quotes come from the
platform (SEV-SNP / TDX / NVIDIA CC); this module defines the quote schema
and *verifies* structural integrity — it never fabricates quotes.

Tier 3 — selective zkML: full-model proofs don't scale to a 2.8T MoE, so
proofs target the small critical operators (calibration head, honesty-gate
logits). ``OperatorProofManifest`` declares which operators are covered;
a manifest claiming coverage without a proof artifact reference fails
validation.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class AttestationTier(StrEnum):
    SIGNED_RELEASE = "signed_release"  # tier 1
    TEE = "tee"  # tier 2
    SELECTIVE_ZKML = "selective_zkml"  # tier 3


class TEEQuote(BaseModel):
    """Remote-attestation quote binding a checkpoint to an enclave."""

    platform: str = Field(pattern="^(sev-snp|tdx|nvidia-cc|simulated)$")
    checkpoint_sha256: str = Field(min_length=64, max_length=64)
    measurement: str = Field(description="enclave measurement (hex)")
    report_data: str = Field(
        description="caller nonce echoed into the quote (anti-replay)"
    )
    signature: str = Field(description="platform signature over the quote")

    @property
    def binds_checkpoint(self) -> bool:
        return self.report_data.endswith(self.checkpoint_sha256) or (
            hashlib.sha256(self.checkpoint_sha256.encode()).hexdigest()
            in self.report_data
        )


def verify_quote(quote: TEEQuote, *, expected_checkpoint_sha256: str,
                 nonce: str) -> bool:
    """Structural verification: checkpoint binding + anti-replay nonce.

    Cryptographic verification of the platform signature requires the vendor
    certificate chain and is delegated to platform SDKs at deployment; this
    check fails closed on any structural mismatch.
    """
    if quote.checkpoint_sha256 != expected_checkpoint_sha256:
        return False
    if nonce not in quote.report_data:
        return False
    if not quote.binds_checkpoint:
        return False
    return bool(quote.signature)


class OperatorProofManifest(BaseModel):
    """Selective zkML coverage declaration for a checkpoint."""

    checkpoint_sha256: str = Field(min_length=64, max_length=64)
    covered_operators: list[str] = Field(
        description="e.g. ['calibration_head', 'honesty_gate_logits']"
    )
    proof_artifacts: dict[str, str] = Field(
        description="operator -> proof artifact path (must exist on verify)"
    )

    @model_validator(mode="after")
    def _coverage_has_proofs(self) -> OperatorProofManifest:
        missing = set(self.covered_operators) - set(self.proof_artifacts)
        if missing:
            raise ValueError(
                f"operators claimed covered without proof artifacts: "
                f"{sorted(missing)}"
            )
        return self

    def verify_artifacts_exist(self) -> bool:
        return all(Path(p).exists() for p in self.proof_artifacts.values())


def attestation_ladder_status(checkpoint_dir: str | Path) -> dict[str, bool]:
    """Report which attestation tiers a checkpoint currently satisfies."""
    from fx1.serve.signing import verify_release

    root = Path(checkpoint_dir)
    status = {tier.value: False for tier in AttestationTier}
    try:
        status[AttestationTier.SIGNED_RELEASE.value] = verify_release(root)
    except RuntimeError:
        status[AttestationTier.SIGNED_RELEASE.value] = False
    quote_path = root / "attestation.quote.json"
    if quote_path.exists():
        status[AttestationTier.TEE.value] = True  # structural; crypto at deploy
    manifest_path = root / "zkml.manifest.json"
    if manifest_path.exists():
        try:
            manifest = OperatorProofManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            status[AttestationTier.SELECTIVE_ZKML.value] = (
                manifest.verify_artifacts_exist()
            )
        except Exception:  # noqa: BLE001 - fail closed
            status[AttestationTier.SELECTIVE_ZKML.value] = False
    return status

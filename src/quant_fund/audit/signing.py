"""Signed tree heads: Ed25519 offline, or Sigstore when a real token exists.

Ed25519 uses ``cryptography``, which is already installed with the harness.
Signatures are deterministic. The private key never enters a ledger line.

Sigstore uses keyless Fulcio signing and Rekor inclusion via the optional
``sigstore`` package. If the package or the OIDC token is missing, signing
raises ``SignatureUnavailableError`` and does not emit a placeholder bundle.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from quant_fund.audit.errors import AuditError, SignatureUnavailableError
from quant_fund.utils.atomicio import atomic_write_text

_PUB_LEN = 32
_SIG_LEN = 64


@dataclass(frozen=True)
class Signature:
    """One signature over the canonical signed-tree-head bytes."""

    value: str
    public_key_hex: str | None = None
    sigstore_bundle_json: str | None = None
    sigstore_identity: str | None = None
    sigstore_issuer: str | None = None


def key_id_for(public_key: bytes) -> str:
    """Short id of a raw Ed25519 public key. Not a secret."""
    return hashlib.sha256(public_key).hexdigest()[:16]


class Ed25519Signer:
    """Offline signer for periodic Merkle roots."""

    scheme = "ed25519"

    def __init__(self, private_key: object) -> None:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        if not isinstance(private_key, Ed25519PrivateKey):
            raise AuditError("Ed25519Signer requires an Ed25519 private key")
        self._private = private_key
        public = private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        if len(public) != _PUB_LEN:
            raise AuditError("Ed25519 public key was not 32 bytes")
        self.public_key: bytes = public
        self.public_key_hex: str = public.hex()
        self.key_id: str = key_id_for(public)

    @classmethod
    def generate(cls) -> Ed25519Signer:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def from_private_bytes(cls, raw: bytes) -> Ed25519Signer:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        if len(raw) != _PUB_LEN:
            raise AuditError("Ed25519 private key must be 32 bytes")
        return cls(Ed25519PrivateKey.from_private_bytes(raw))

    @classmethod
    def from_path(cls, path: Path) -> Ed25519Signer:
        text = Path(path).read_text(encoding="utf-8").strip()
        try:
            raw = bytes.fromhex(text)
        except ValueError as exc:
            raise AuditError(f"signing key {path} is not hex") from exc
        return cls.from_private_bytes(raw)

    def write(self, path: Path) -> None:
        """Store the raw private key as hex, mode 0600, plus a sibling ``.pub``."""
        from cryptography.hazmat.primitives import serialization

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = self._private.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, (raw.hex() + "\n").encode("ascii"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(target, 0o600)
        pub = target.with_name(target.name + ".pub")
        atomic_write_text(pub, self.public_key_hex + "\n")

    def sign(self, payload: bytes) -> Signature:
        signature = self._private.sign(payload)
        if len(signature) != _SIG_LEN:
            raise AuditError("Ed25519 signature was not 64 bytes")
        return Signature(value=signature.hex(), public_key_hex=self.public_key_hex)


def verify_ed25519(payload: bytes, signature_hex: str, public_key: bytes) -> bool:
    """Return False for a bad signature. Do not raise on a mismatch."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    if len(public_key) != _PUB_LEN or not isinstance(signature_hex, str):
        return False
    try:
        signature = bytes.fromhex(signature_hex)
    except ValueError:
        return False
    if len(signature) != _SIG_LEN:
        return False
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, payload)
    except InvalidSignature:
        return False
    return True


def load_public_key_hex(path: Path) -> bytes:
    text = Path(path).read_text(encoding="utf-8").strip()
    try:
        raw = bytes.fromhex(text)
    except ValueError as exc:
        raise AuditError(f"public key {path} is not hex") from exc
    if len(raw) != _PUB_LEN:
        raise AuditError("Ed25519 public key must be 32 bytes")
    return raw


def sign_with_sigstore(
    payload: bytes,
    *,
    identity_token: str | None = None,
    instance: str | None = None,
) -> tuple[str, str, str]:
    """Sign ``payload`` with Sigstore and return ``(bundle_json, identity, issuer)``.

    Refuses to run without ``DIPCATCHER_SIGSTORE_ID_TOKEN`` (or an explicit
    token) and without the ``sigstore`` package. A bogus token does not produce
    a bundle.
    """
    token = identity_token
    if token is None:
        token = os.environ.get("DIPCATCHER_SIGSTORE_ID_TOKEN", "")
    if not token.strip():
        raise SignatureUnavailableError(
            "DIPCATCHER_SIGSTORE_ID_TOKEN is not set; refusing to emit a Sigstore bundle"
        )
    if instance is None:
        chosen = os.environ.get("DIPCATCHER_SIGSTORE_INSTANCE", "production")
    else:
        chosen = instance
    try:
        from sigstore.models import Bundle, ClientTrustConfig
        from sigstore.oidc import IdentityToken
        from sigstore.sign import SigningContext
    except ImportError as exc:
        raise SignatureUnavailableError(
            "sigstore package is not installed; refusing to emit a Sigstore bundle"
        ) from exc
    try:
        identity = IdentityToken(token)
        if chosen == "staging":
            trust = ClientTrustConfig.staging()
        elif chosen == "production":
            trust = ClientTrustConfig.production()
        else:
            raise SignatureUnavailableError(
                "DIPCATCHER_SIGSTORE_INSTANCE must be production or staging"
            )
        context = SigningContext.from_trust_config(trust)
        with context.signer(identity) as signer:
            bundle: Bundle = signer.sign_artifact(payload)
    except SignatureUnavailableError:
        raise
    except Exception as exc:
        raise SignatureUnavailableError(f"Sigstore signing failed: {exc}") from exc
    bundle_json = bundle.to_json()
    if not bundle_json:
        raise SignatureUnavailableError("Sigstore returned an empty bundle")
    return bundle_json, identity.identity, identity.issuer


def verify_sigstore_bundle(
    payload: bytes,
    bundle_json: str,
    *,
    identity: str,
    issuer: str | None,
    offline: bool = False,
) -> None:
    """Verify a Sigstore bundle. Raises ``AuditError`` when verification fails."""
    if not identity:
        raise SignatureUnavailableError("Sigstore verification requires a pinned identity")
    try:
        from sigstore.models import Bundle
        from sigstore.verify.policy import Identity
        from sigstore.verify.verifier import Verifier
    except ImportError as exc:
        raise SignatureUnavailableError(
            "sigstore package is not installed; cannot verify a Sigstore bundle"
        ) from exc
    try:
        bundle = Bundle.from_json(bundle_json)
        verifier = Verifier.production(offline=offline)
        verifier.verify_artifact(
            payload,
            bundle,
            Identity(identity=identity, issuer=issuer),
        )
    except SignatureUnavailableError:
        raise
    except Exception as exc:
        raise AuditError(f"Sigstore bundle verification failed: {exc}") from exc


class SigstoreSigner:
    """Keyless signer. ``sign`` performs a real Fulcio/Rekor round trip."""

    scheme = "sigstore"

    def __init__(self, *, identity_token: str | None = None, instance: str = "production") -> None:
        self._token = identity_token
        self._instance = instance
        self.key_id = "sigstore"
        self.public_key_hex: str | None = None

    def sign(self, payload: bytes) -> Signature:
        bundle_json, identity, issuer = sign_with_sigstore(
            payload,
            identity_token=self._token,
            instance=self._instance,
        )
        digest = hashlib.sha256(bundle_json.encode("utf-8")).hexdigest()
        return Signature(
            value=digest,
            public_key_hex=None,
            sigstore_bundle_json=bundle_json,
            sigstore_identity=identity,
            sigstore_issuer=issuer,
        )

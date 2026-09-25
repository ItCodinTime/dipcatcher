"""Release signing for fx-1 checkpoints (attestation ladder, tier 1).

Signed release chain: every checkpoint directory carries ``release.sig`` — an
HMAC signature over the manifest of artifact hashes (modelcard, weights,
config). The signing key comes from the environment only
(``FX1_SIGNING_KEY``); the interface is cosign/Sigstore-compatible in shape
(detached signature over a digest manifest) so keyless OIDC signing can
replace the HMAC backend without touching callers.

Serving enforcement lives in ``LocalFx1Backend``: unsigned or
signature-mismatched checkpoints refuse to serve.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path

from pydantic import BaseModel, Field

SIGNING_KEY_ENV = "FX1_SIGNING_KEY"
SIGNATURE_FILENAME = "release.sig"
MANIFEST_FILENAME = "release.manifest.json"


class ReleaseManifest(BaseModel):
    """Digest manifest of everything a release consists of."""

    checkpoint_dir: str
    artifacts: dict[str, str] = Field(
        description="relative path -> sha256"
    )


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(checkpoint_dir: str | Path) -> ReleaseManifest:
    root = Path(checkpoint_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint dir not found: {root}")
    artifacts: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {
            SIGNATURE_FILENAME, MANIFEST_FILENAME
        }:
            artifacts[str(path.relative_to(root))] = _hash_file(path)
    if not artifacts:
        raise ValueError(f"checkpoint dir {root} contains no artifacts")
    return ReleaseManifest(checkpoint_dir=str(root), artifacts=artifacts)


def _key() -> bytes:
    key = os.environ.get(SIGNING_KEY_ENV, "")
    if not key:
        raise RuntimeError(
            f"{SIGNING_KEY_ENV} is not set; fx-1 never hardcodes signing keys"
        )
    return key.encode()


def sign_release(checkpoint_dir: str | Path) -> Path:
    """Write manifest + detached HMAC signature into the checkpoint dir."""
    root = Path(checkpoint_dir)
    manifest = build_manifest(root)
    manifest_path = root / MANIFEST_FILENAME
    manifest_bytes = manifest.model_dump_json().encode()
    manifest_path.write_bytes(manifest_bytes)
    signature = hmac.new(_key(), manifest_bytes, hashlib.sha256).hexdigest()
    sig_path = root / SIGNATURE_FILENAME
    sig_path.write_text(signature, encoding="utf-8")
    return sig_path


def verify_release(checkpoint_dir: str | Path) -> bool:
    """Fail-closed verification: manifest intact, signature valid, hashes match."""
    root = Path(checkpoint_dir)
    manifest_path = root / MANIFEST_FILENAME
    sig_path = root / SIGNATURE_FILENAME
    if not manifest_path.exists() or not sig_path.exists():
        return False
    manifest_bytes = manifest_path.read_bytes()
    expected = hmac.new(_key(), manifest_bytes, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig_path.read_text().strip()):
        return False
    manifest = ReleaseManifest.model_validate_json(manifest_bytes.decode())
    for rel, digest in manifest.artifacts.items():
        path = root / rel
        if not path.exists() or _hash_file(path) != digest:
            return False
    return True

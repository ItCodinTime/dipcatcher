"""Ed25519 tree heads, and Sigstore's refusal to invent a bundle."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from quant_fund.audit.errors import AuditError, SignatureUnavailableError
from quant_fund.audit.signing import (
    Ed25519Signer,
    load_public_key_hex,
    sign_with_sigstore,
    verify_ed25519,
    verify_sigstore_bundle,
)


def test_ed25519_roundtrip_is_deterministic(tmp_path: Path) -> None:
    signer = Ed25519Signer.generate()
    path = tmp_path / "ledger.key"
    signer.write(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    loaded = Ed25519Signer.from_path(path)
    assert loaded.public_key == signer.public_key
    assert (tmp_path / "ledger.key.pub").read_text(
        encoding="ascii"
    ).strip() == signer.public_key_hex
    first = signer.sign(b"tree-head")
    second = loaded.sign(b"tree-head")
    assert first.value == second.value
    assert verify_ed25519(b"tree-head", first.value, signer.public_key)
    assert not verify_ed25519(b"other", first.value, signer.public_key)
    assert not verify_ed25519(b"tree-head", "00" * 64, signer.public_key)
    assert load_public_key_hex(tmp_path / "ledger.key.pub") == signer.public_key


def test_rejects_bad_key_material(tmp_path: Path) -> None:
    with pytest.raises(AuditError):
        Ed25519Signer.from_private_bytes(b"\x00" * 31)
    path = tmp_path / "bad"
    path.write_text("zzzz\n", encoding="ascii")
    with pytest.raises(AuditError):
        Ed25519Signer.from_path(path)
    with pytest.raises(AuditError):
        load_public_key_hex(path)


def test_sigstore_refuses_without_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIPCATCHER_SIGSTORE_ID_TOKEN", raising=False)
    with pytest.raises(SignatureUnavailableError, match="refusing"):
        sign_with_sigstore(b"tree-head")


def test_bogus_sigstore_token_does_not_produce_a_bundle() -> None:
    with pytest.raises(SignatureUnavailableError):
        sign_with_sigstore(b"tree-head", identity_token="not-a-jwt")


def test_fake_sigstore_bundle_does_not_verify() -> None:
    with pytest.raises(AuditError):
        verify_sigstore_bundle(
            b"tree-head",
            "{}",
            identity="researcher@example.com",
            issuer="https://example.invalid",
        )

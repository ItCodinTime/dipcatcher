"""Edge coverage for audit.signing — Ed25519 round trips + sigstore refusals."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from quant_fund.audit.errors import AuditError, SignatureUnavailableError
from quant_fund.audit.signing import (
    Ed25519Signer,
    SigstoreSigner,
    key_id_for,
    load_public_key_hex,
    sign_with_sigstore,
    verify_ed25519,
    verify_sigstore_bundle,
)


class TestEd25519:
    def test_generate_sign_verify_roundtrip(self) -> None:
        signer = Ed25519Signer.generate()
        assert len(signer.public_key) == 32
        assert signer.key_id == key_id_for(signer.public_key)
        signature = signer.sign(b"payload")
        assert signature.public_key_hex == signer.public_key_hex
        assert len(signature.value) == 128
        assert verify_ed25519(b"payload", signature.value, signer.public_key) is True

    def test_from_private_bytes_roundtrip(self) -> None:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        raw = Ed25519PrivateKey.generate().private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        signer = Ed25519Signer.from_private_bytes(raw)
        assert signer.sign(b"x").value

    def test_from_private_bytes_wrong_length(self) -> None:
        with pytest.raises(AuditError, match="32 bytes"):
            Ed25519Signer.from_private_bytes(b"short")

    def test_constructor_rejects_non_ed25519(self) -> None:
        with pytest.raises(AuditError, match="Ed25519 private key"):
            Ed25519Signer(private_key=object())

    def test_from_path_hex_and_errors(self, tmp_path: Path) -> None:
        signer = Ed25519Signer.generate()
        key_path = tmp_path / "key.hex"
        signer.write(key_path)
        loaded = Ed25519Signer.from_path(key_path)
        assert loaded.public_key == signer.public_key
        bad = tmp_path / "bad.hex"
        bad.write_text("not-hex\n", encoding="utf-8")
        with pytest.raises(AuditError, match="not hex"):
            Ed25519Signer.from_path(bad)

    def test_write_permissions_and_pub_sibling(self, tmp_path: Path) -> None:
        signer = Ed25519Signer.generate()
        path = tmp_path / "nested" / "key.hex"
        signer.write(path)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        pub = path.with_name("key.hex.pub")
        assert pub.read_text().strip() == signer.public_key_hex
        assert Ed25519Signer.from_path(path).public_key == signer.public_key

    def test_verify_ed25519_failure_matrix(self) -> None:
        signer = Ed25519Signer.generate()
        sig = signer.sign(b"payload").value
        assert verify_ed25519(b"other", sig, signer.public_key) is False
        assert verify_ed25519(b"payload", sig, b"short") is False
        assert verify_ed25519(b"payload", "zz" * 40, signer.public_key) is False
        assert verify_ed25519(b"payload", sig[:32], signer.public_key) is False
        other = Ed25519Signer.generate()
        assert verify_ed25519(b"payload", sig, other.public_key) is False

    def test_load_public_key_hex(self, tmp_path: Path) -> None:
        signer = Ed25519Signer.generate()
        path = tmp_path / "key.pub"
        path.write_text(signer.public_key_hex + "\n", encoding="utf-8")
        assert load_public_key_hex(path) == signer.public_key
        bad = tmp_path / "bad.pub"
        bad.write_text("xyz\n", encoding="utf-8")
        with pytest.raises(AuditError, match="not hex"):
            load_public_key_hex(bad)
        short = tmp_path / "short.pub"
        short.write_text("ab" * 16, encoding="utf-8")
        with pytest.raises(AuditError, match="32 bytes"):
            load_public_key_hex(short)


class TestSigstoreRefusals:
    """sigstore is not installed in this environment: every path fails closed."""

    def test_sign_without_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DIPCATCHER_SIGSTORE_ID_TOKEN", raising=False)
        with pytest.raises(SignatureUnavailableError, match="ID_TOKEN"):
            sign_with_sigstore(b"payload")

    def test_sign_with_token_but_no_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DIPCATCHER_SIGSTORE_ID_TOKEN", "fake-token")
        with pytest.raises(SignatureUnavailableError):
            sign_with_sigstore(b"payload")

    def test_sign_explicit_token_no_package(self) -> None:
        with pytest.raises(SignatureUnavailableError):
            sign_with_sigstore(b"payload", identity_token="fake-token")

    def test_verify_bundle_requires_identity(self) -> None:
        with pytest.raises(SignatureUnavailableError, match="identity"):
            verify_sigstore_bundle(b"payload", "{}", identity="", issuer=None)

    def test_verify_bundle_no_package(self) -> None:
        with pytest.raises(SignatureUnavailableError):
            verify_sigstore_bundle(b"payload", "{}", identity="a@b.c", issuer=None)

    def test_sigstore_signer_propagates_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DIPCATCHER_SIGSTORE_ID_TOKEN", raising=False)
        signer = SigstoreSigner()
        assert signer.scheme == "sigstore"
        assert signer.key_id == "sigstore"
        with pytest.raises(SignatureUnavailableError):
            signer.sign(b"payload")

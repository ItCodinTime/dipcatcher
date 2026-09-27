"""Unit tests for proof.sign (DESIGN.md §5.4)."""

from __future__ import annotations

import hashlib

import pytest

from quant_fund.proof.sign import (
    SIGNING_KEY_ENV,
    HmacSha256Signer,
    NullSigner,
    Signer,
)
from quant_fund.proofcore.contracts import SignatureUnavailableError


def test_hmac_signer_roundtrip() -> None:
    signer = HmacSha256Signer(b"test-key-material")
    payload = b'{"a":1}'
    signature = signer.sign(payload)
    assert signer.scheme == "hmac-sha256"
    assert signer.key_id == hashlib.sha256(b"test-key-material").hexdigest()[:16]
    assert signer.verify(payload, signature)
    assert not signer.verify(payload + b"x", signature)
    assert not signer.verify(payload, "0" * 64)


def test_hmac_signer_deterministic() -> None:
    assert HmacSha256Signer(b"k").sign(b"p") == HmacSha256Signer(b"k").sign(b"p")


def test_hmac_signer_empty_key_rejected() -> None:
    with pytest.raises(SignatureUnavailableError):
        HmacSha256Signer(b"")


def test_from_env_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SIGNING_KEY_ENV, raising=False)
    with pytest.raises(SignatureUnavailableError):
        HmacSha256Signer.from_env()


def test_from_env_hex_and_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SIGNING_KEY_ENV, "deadbeef" * 8)
    hex_signer = HmacSha256Signer.from_env()
    assert hex_signer.sign(b"x") == HmacSha256Signer(bytes.fromhex("deadbeef" * 8)).sign(b"x")
    monkeypatch.setenv(SIGNING_KEY_ENV, "not-hex-key!")
    raw_signer = HmacSha256Signer.from_env()
    assert raw_signer.sign(b"x") == HmacSha256Signer(b"not-hex-key!").sign(b"x")


def test_null_signer() -> None:
    signer = NullSigner()
    assert signer.scheme == "none"
    assert signer.key_id == "unsigned"
    assert signer.sign(b"anything") == ""


def test_protocol_conformance() -> None:
    assert isinstance(HmacSha256Signer(b"k"), Signer)
    assert isinstance(NullSigner(), Signer)

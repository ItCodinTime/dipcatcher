"""Release verification must fail on files added after signing.

``verify_release`` previously checked only manifest-listed artifacts, so an
attacker with write access could drop extra files (a ``sitecustomize`` shim,
a config override, an extra weight file) into a still-"verified" checkpoint
directory. The on-disk file set must equal the signed manifest exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fx1.serve.signing import (
    MANIFEST_FILENAME,
    SIGNATURE_FILENAME,
    sign_release,
    verify_release,
)


def _checkpoint(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "modelcard.json").write_text("{}", encoding="utf-8")
    (root / "weights.bin").write_bytes(b"weights")
    return root


@pytest.fixture()
def signed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("FX1_SIGNING_KEY", "test-signing-key")
    root = _checkpoint(tmp_path / "ckpt")
    sign_release(root)
    return root


def test_verify_clean_release(signed: Path) -> None:
    assert verify_release(signed)


def test_verify_fails_on_added_file(signed: Path) -> None:
    (signed / "evil.py").write_text("import os\n", encoding="utf-8")
    assert not verify_release(signed)


def test_verify_fails_on_added_nested_file(signed: Path) -> None:
    nested = signed / "subdir"
    nested.mkdir()
    (nested / "extra.bin").write_bytes(b"extra")
    assert not verify_release(signed)


def test_verify_fails_on_modified_and_missing(signed: Path) -> None:
    (signed / "weights.bin").write_bytes(b"tampered")
    assert not verify_release(signed)
    (signed / "weights.bin").unlink()
    assert not verify_release(signed)


def test_verify_fails_without_signature(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_SIGNING_KEY", "test-signing-key")
    root = _checkpoint(tmp_path / "ckpt")
    sign_release(root)
    (root / SIGNATURE_FILENAME).unlink()
    assert not verify_release(root)
    (root / MANIFEST_FILENAME).unlink()
    assert not verify_release(root)


def test_manifest_parses_and_covers_all_files(signed: Path) -> None:
    manifest = json.loads((signed / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert set(manifest["artifacts"]) == {"modelcard.json", "weights.bin"}

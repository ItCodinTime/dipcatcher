"""Checksum verification for release artifacts (tamper fails closed)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.verify_release_artifacts import (
    ReleaseVerificationError,
    verify_checksums,
    verify_release_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "verify_release_artifacts.py"


def _write_fake_dist(root: Path) -> Path:
    dist = root / "dist"
    dist.mkdir()
    wheel = dist / "fx_1-0.0.0-py3-none-any.whl"
    sdist = dist / "fx_1-0.0.0.tar.gz"
    sbom = dist / "sbom.cdx.json"
    wheel.write_bytes(b"wheel-bytes-v1")
    sdist.write_bytes(b"sdist-bytes-v1")
    sbom.write_text(
        json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.5",
                "components": [{"type": "library", "name": "example", "version": "1.0.0"}],
            }
        ),
        encoding="utf-8",
    )
    lines = []
    for path in (wheel, sdist, sbom):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    (dist / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dist


def test_verify_checksums_accepts_intact_dist(tmp_path: Path) -> None:
    dist = _write_fake_dist(tmp_path)
    entries = verify_checksums(dist)
    assert "fx_1-0.0.0-py3-none-any.whl" in entries
    assert "sbom.cdx.json" in entries


def test_verify_rejects_tampered_wheel(tmp_path: Path) -> None:
    dist = _write_fake_dist(tmp_path)
    verify_release_artifacts(dist)
    (dist / "fx_1-0.0.0-py3-none-any.whl").write_bytes(b"tampered-payload")
    with pytest.raises(ReleaseVerificationError, match="checksum mismatch"):
        verify_release_artifacts(dist)


def test_verify_rejects_missing_sums(tmp_path: Path) -> None:
    dist = _write_fake_dist(tmp_path)
    (dist / "SHA256SUMS").unlink()
    with pytest.raises(ReleaseVerificationError, match="missing checksum file"):
        verify_checksums(dist)


def test_cli_tampered_artifact_exits_nonzero(tmp_path: Path) -> None:
    dist = _write_fake_dist(tmp_path)
    ok = subprocess.run(
        [sys.executable, str(SCRIPT), str(dist)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stderr
    (dist / "fx_1-0.0.0-py3-none-any.whl").write_bytes(b"tampered-again")
    bad = subprocess.run(
        [sys.executable, str(SCRIPT), str(dist)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 1
    assert "checksum mismatch" in bad.stderr

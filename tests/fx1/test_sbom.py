"""SBOM generation tests."""

from pathlib import Path

import pytest

from fx1.sbom import generate_sbom


def test_sbom_from_real_uv_lock():
    lock = Path(__file__).parents[2] / "uv.lock"
    if not lock.exists():  # standalone overlay: synthesize a minimal lock
        lock = Path(__file__).parents[2] / "tests" / "fx1" / "_test_uv.lock"
        lock.write_text(
            '[[package]]\nname = "pydantic"\nversion = "2.11.4"\n'
            + "\n".join(
                f'[[package]]\nname = "pkg{i}"\nversion = "1.0.{i}"'
                for i in range(60)
            ),
            encoding="utf-8",
        )
    sbom = generate_sbom(lock)
    assert len(sbom.entries) > 50
    assert len(sbom.lockfile_sha256) == 64
    names = [e.name for e in sbom.entries]
    assert "pydantic" in names
    assert names == sorted(names)


def test_sbom_fail_closed_missing_lock(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        generate_sbom(tmp_path / "nope.lock")
    bad = tmp_path / "bad.lock"
    bad.write_text("not a lockfile", encoding="utf-8")
    with pytest.raises(ValueError, match="no packages"):
        generate_sbom(bad)

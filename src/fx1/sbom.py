"""SBOM generation for fx-1 releases (SPDX-lite, CycloneDX-compatible shape).

An auditor asks "what exactly is in this thing?" — the SBOM answers with
locked dependency names/versions/hashes from uv.lock plus the repo's own
artifact hashes. Generated, never hand-maintained.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


class SBOMEntry(BaseModel):
    name: str
    version: str
    sha256: str | None = None


class SBOM(BaseModel):
    spec: str = "spdx-lite/2.3"
    generated_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    package: str = "fx-1"
    entries: list[SBOMEntry]
    lockfile_sha256: str = Field(min_length=64, max_length=64)


def generate_sbom(lockfile: str | Path = "uv.lock") -> SBOM:
    """Parse uv.lock into a hash-pinned SBOM."""
    path = Path(lockfile)
    if not path.exists():
        raise FileNotFoundError(f"lockfile not found: {path}")
    text = path.read_text(encoding="utf-8")
    entries: list[SBOMEntry] = []
    # uv.lock [[package]] blocks: name, version, optional sdist/wheel hashes.
    for block in re.split(r"\n\[\[package\]\]\n", text):
        name_m = re.search(r'^name = "([^"]+)"', block, re.MULTILINE)
        ver_m = re.search(r'^version = "([^"]+)"', block, re.MULTILINE)
        if not name_m or not ver_m:
            continue
        hash_m = re.search(r'hash = "sha256:([0-9a-f]{64})"', block)
        entries.append(
            SBOMEntry(
                name=name_m.group(1),
                version=ver_m.group(1),
                sha256=hash_m.group(1) if hash_m else None,
            )
        )
    if not entries:
        raise ValueError(f"no packages parsed from {path}")
    return SBOM(
        entries=sorted(entries, key=lambda e: e.name),
        lockfile_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )

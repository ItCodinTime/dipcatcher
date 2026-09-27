"""PIT vault manifest: read/write/verify of manifest.json + sha256 sidecar.

The manifest is a ``PitManifest`` (proofcore contracts schema) serialized as
canonical JSON. Every successful append atomically rewrites ``manifest.json``
(NamedTemporaryFile + flush + fsync + ``os.replace`` — the
``models/base.py:225-245`` pattern) and writes the **previous** manifest's
sha256 into ``manifest.sha256``, chaining manifests (fx1 ``CorpusLedger``
pattern). Per-file sha256 is computed over raw part bytes, streaming 1 MiB
chunks (DESIGN.md §4.1).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import ValidationError

from quant_fund.proofcore.contracts import (
    GENESIS_HASH,
    ManifestError,
    PitManifest,
    canonical_json_bytes,
    sha256_hex_bytes,
)

_CHUNK_BYTES = 1024 * 1024  # 1 MiB streaming chunks (DESIGN.md §4.1)

MANIFEST_NAME = "manifest.json"
SIDECAR_NAME = "manifest.sha256"
DATASET_META_NAME = "dataset.json"
PARTS_DIR = "parts"


def sha256_file(path: Path) -> str:
    """sha256 hex of file bytes, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_dir(root: Path, dataset: str) -> Path:
    return Path(root) / dataset


def manifest_path(root: Path, dataset: str) -> Path:
    return dataset_dir(root, dataset) / MANIFEST_NAME


def sidecar_path(root: Path, dataset: str) -> Path:
    return dataset_dir(root, dataset) / SIDECAR_NAME


def manifest_json_bytes(manifest: PitManifest) -> bytes:
    """Canonical serialization — the exact bytes written to manifest.json."""
    return canonical_json_bytes(manifest.model_dump(mode="json"))


def _atomic_write(path: Path, payload: bytes) -> None:
    """NamedTemporaryFile + flush + fsync + os.replace (models/base.py pattern)."""
    temporary_path: Path | None = None
    try:
        # Keep the temporary file beside the destination so os.replace is
        # atomic even when the vault directory is on a separate mount.
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_manifest(
    root: Path,
    dataset: str,
    manifest: PitManifest,
    *,
    prev_manifest_sha256: str,
) -> str:
    """Atomically write manifest.json and the sidecar chain anchor.

    ``prev_manifest_sha256`` is the sha256 of the PREVIOUS manifest's bytes
    (``GENESIS_HASH`` at dataset creation); it lands both in the manifest's
    ``prev_manifest_sha256`` field and in the ``manifest.sha256`` sidecar.
    Returns the sha256 of the manifest bytes written.
    """
    if manifest.prev_manifest_sha256 != prev_manifest_sha256:
        raise ManifestError(
            f"{dataset}: manifest prev_manifest_sha256 disagrees with sidecar anchor"
        )
    directory = dataset_dir(root, dataset)
    directory.mkdir(parents=True, exist_ok=True)
    payload = manifest_json_bytes(manifest)
    _atomic_write(manifest_path(root, dataset), payload)
    _atomic_write(sidecar_path(root, dataset), (prev_manifest_sha256 + "\n").encode("ascii"))
    return sha256_hex_bytes(payload)


def read_manifest(root: Path, dataset: str) -> PitManifest:
    """Read + validate manifest.json and its chain sidecar. Fail-closed."""
    path = manifest_path(root, dataset)
    if not path.exists():
        raise ManifestError(f"{dataset}: manifest.json missing at {path}")
    try:
        manifest = PitManifest.model_validate(json.loads(path.read_bytes()))
    except (ValidationError, ValueError) as exc:
        raise ManifestError(f"{dataset}: manifest.json malformed: {exc}") from exc
    if manifest.dataset != dataset:
        raise ManifestError(
            f"{dataset}: manifest dataset field {manifest.dataset!r} disagrees with path"
        )
    sidecar = sidecar_path(root, dataset)
    if not sidecar.exists():
        raise ManifestError(f"{dataset}: manifest.sha256 sidecar missing (chain anchor)")
    anchor = sidecar.read_text(encoding="ascii").strip()
    if anchor != manifest.prev_manifest_sha256:
        raise ManifestError(
            f"{dataset}: manifest chain broken — sidecar {anchor[:16]}… != "
            f"prev_manifest_sha256 {manifest.prev_manifest_sha256[:16]}…"
        )
    return manifest


def genesis_sidecar_ok(root: Path, dataset: str) -> bool:
    """True iff the sidecar is the genesis anchor (freshly created dataset)."""
    sidecar = sidecar_path(root, dataset)
    return sidecar.exists() and sidecar.read_text(encoding="ascii").strip() == GENESIS_HASH


def verify_part_hashes(root: Path, manifest: PitManifest) -> list[str]:
    """Re-hash every manifest part over raw bytes; return violations ([] = ok)."""
    violations: list[str] = []
    listed = {entry.path for entry in manifest.files}
    for entry in manifest.files:
        path = Path(root) / entry.path
        if not path.exists():
            violations.append(f"{entry.path}: part file missing")
            continue
        actual = sha256_file(path)
        if actual != entry.sha256:
            violations.append(
                f"{entry.path}: sha256 mismatch — manifest {entry.sha256[:16]}… "
                f"!= disk {actual[:16]}…"
            )
    parts = dataset_dir(root, manifest.dataset) / PARTS_DIR
    if parts.is_dir():
        for on_disk in sorted(parts.glob("r*.parquet")):
            rel = on_disk.relative_to(root).as_posix()
            if rel not in listed:
                violations.append(f"{rel}: part on disk not listed in manifest")
    return violations

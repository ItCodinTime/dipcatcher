"""PIT vault manifest: retained revisions, current pointer, and hash chain.

Each canonical revision is retained in ``manifests/rNNNNNNN.json``. The
``manifest.json`` pointer must match the last retained revision byte for byte;
each retained revision commits to the previous one's SHA-256. Per-file hashes
are computed over raw part bytes in streaming 1 MiB chunks.
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
MANIFEST_HISTORY_DIR = "manifests"
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


def versioned_manifest_path(root: Path, dataset: str, revision: int) -> Path:
    return dataset_dir(root, dataset) / MANIFEST_HISTORY_DIR / f"r{revision:07d}.json"


def part_path(root: Path, manifest: PitManifest, entry_path: str) -> Path:
    """Resolve a manifest entry only within this dataset's parts directory."""
    expected = Path(manifest.dataset) / PARTS_DIR
    relative = Path(entry_path)
    if relative.parent != expected:
        raise ManifestError(f"{manifest.dataset}: part path outside dataset: {entry_path}")
    parts = (Path(root) / expected).resolve()
    if not parts.is_relative_to(Path(root).resolve()):
        raise ManifestError(f"{manifest.dataset}: dataset escapes vault root")
    candidate = (Path(root) / relative).resolve()
    if not candidate.is_relative_to(parts):
        raise ManifestError(f"{manifest.dataset}: part path escapes dataset: {entry_path}")
    return candidate


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
    history = versioned_manifest_path(root, dataset, manifest.revision)
    history.parent.mkdir(parents=True, exist_ok=True)
    if history.exists():
        raise ManifestError(f"{dataset}: write-once violation: manifest revision exists")
    payload = manifest_json_bytes(manifest)
    _atomic_write(history, payload)
    _atomic_write(manifest_path(root, dataset), payload)
    _atomic_write(sidecar_path(root, dataset), (prev_manifest_sha256 + "\n").encode("ascii"))
    return sha256_hex_bytes(payload)


def _check_manifest_shape(manifest: PitManifest) -> None:
    """Structural invariants every vault-written manifest satisfies.

    ``revision`` counts appends and each append adds exactly one part named
    ``r{revision:07d}.parquet``, so the revision equals the file count and the
    i-th entry is always the i-th part. Forged manifests that violate these
    invariants are rejected before any hash verification runs.
    """
    if manifest.revision != len(manifest.files):
        raise ManifestError(
            f"{manifest.dataset}: revision {manifest.revision} lists {len(manifest.files)} files"
        )
    for index, entry in enumerate(manifest.files, start=1):
        expected = f"{manifest.dataset}/{PARTS_DIR}/r{index:07d}.parquet"
        if entry.path != expected:
            raise ManifestError(
                f"{manifest.dataset}: file {index} path {entry.path!r} != expected {expected!r}"
            )


def _validated_history(root: Path, dataset: str, current: PitManifest) -> list[bytes]:
    """Validate every retained link and return the canonical revision bytes."""
    previous_sha = GENESIS_HASH
    meta_anchor: str | None = None
    revisions: list[bytes] = []
    for revision in range(current.revision + 1):
        revision_path = versioned_manifest_path(root, dataset, revision)
        if not revision_path.is_file():
            raise ManifestError(f"{dataset}: manifest revision {revision} missing")
        revision_bytes = revision_path.read_bytes()
        try:
            prior = PitManifest.model_validate(json.loads(revision_bytes))
        except (ValidationError, ValueError) as exc:
            raise ManifestError(
                f"{dataset}: manifest revision {revision} malformed: {exc}"
            ) from exc
        if prior.dataset != dataset or prior.revision != revision:
            raise ManifestError(f"{dataset}: manifest revision {revision} identity mismatch")
        if prior.prev_manifest_sha256 != previous_sha:
            raise ManifestError(f"{dataset}: manifest chain broken at revision {revision}")
        _check_manifest_shape(prior)
        if meta_anchor is None:
            meta_anchor = prior.dataset_meta_sha256
        elif prior.dataset_meta_sha256 != meta_anchor:
            raise ManifestError(f"{dataset}: manifest revision {revision} metadata anchor drift")
        revisions.append(revision_bytes)
        previous_sha = sha256_hex_bytes(revision_bytes)
    if revisions[-1] != manifest_path(root, dataset).read_bytes():
        raise ManifestError(f"{dataset}: current manifest differs from retained revision")
    return revisions


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
    revisions = _validated_history(root, dataset, manifest)
    if manifest.revision > 0 and anchor != sha256_hex_bytes(revisions[-2]):
        raise ManifestError(f"{dataset}: manifest chain broken at current anchor")
    if manifest.dataset_meta_sha256 != GENESIS_HASH:
        # dataset.json is create-time immutable; the manifest chain anchors its
        # bytes so a tampered/swapped metadata file fails closed like any other
        # committed file. Legacy manifests (GENESIS_HASH) are unanchored.
        meta_file = dataset_dir(root, dataset) / DATASET_META_NAME
        if not meta_file.is_file():
            raise ManifestError(f"{dataset}: dataset.json missing but manifest anchors it")
        try:
            meta_sha = sha256_file(meta_file)
        except OSError as exc:
            raise ManifestError(f"{dataset}: dataset.json unreadable: {exc}") from exc
        if meta_sha != manifest.dataset_meta_sha256:
            raise ManifestError(f"{dataset}: dataset.json sha256 disagrees with anchored manifest")
    extra = versioned_manifest_path(root, dataset, manifest.revision + 1)
    if extra.exists():
        raise ManifestError(f"{dataset}: uncommitted later manifest revision exists")
    return manifest


def recover_interrupted_manifest(root: Path, dataset: str) -> bool:
    """Promote a complete pending revision after an interrupted pointer update.

    The caller must hold the dataset append lock. No bytes are invented: the
    pending snapshot must extend the verified current chain and every listed
    part must match its committed digest before this repairs the pointer.
    """
    path = manifest_path(root, dataset)
    try:
        current = PitManifest.model_validate(json.loads(path.read_bytes()))
    except (OSError, ValidationError, ValueError) as exc:
        raise ManifestError(f"{dataset}: current manifest malformed: {exc}") from exc
    if current.dataset != dataset:
        raise ManifestError(f"{dataset}: current manifest identity mismatch")
    _validated_history(root, dataset, current)
    pending_path = versioned_manifest_path(root, dataset, current.revision + 1)
    if not pending_path.exists():
        sidecar = sidecar_path(root, dataset)
        if (
            not sidecar.exists()
            or sidecar.read_text(encoding="ascii").strip() != current.prev_manifest_sha256
        ):
            _atomic_write(sidecar, (current.prev_manifest_sha256 + "\n").encode("ascii"))
            return True
        return False
    pending_bytes = pending_path.read_bytes()
    try:
        pending = PitManifest.model_validate(json.loads(pending_bytes))
    except (ValidationError, ValueError) as exc:
        raise ManifestError(f"{dataset}: pending manifest malformed: {exc}") from exc
    current_sha = sha256_hex_bytes(path.read_bytes())
    if (
        pending.dataset != dataset
        or pending.revision != current.revision + 1
        or pending.prev_manifest_sha256 != current_sha
        or pending.files[: len(current.files)] != current.files
        or len(pending.files) != len(current.files) + 1
        or pending.dataset_meta_sha256 != current.dataset_meta_sha256
    ):
        raise ManifestError(f"{dataset}: pending manifest does not extend current chain")
    _check_manifest_shape(pending)
    violations = verify_part_hashes(root, pending)
    if violations:
        raise ManifestError(f"{dataset}: pending manifest part invalid: {violations[0]}")
    _atomic_write(path, pending_bytes)
    _atomic_write(sidecar_path(root, dataset), (current_sha + "\n").encode("ascii"))
    return True


def genesis_sidecar_ok(root: Path, dataset: str) -> bool:
    """True iff the sidecar is the genesis anchor (freshly created dataset)."""
    sidecar = sidecar_path(root, dataset)
    return sidecar.exists() and sidecar.read_text(encoding="ascii").strip() == GENESIS_HASH


def verify_part_hashes(root: Path, manifest: PitManifest) -> list[str]:
    """Re-hash every manifest part over raw bytes; return violations ([] = ok)."""
    violations: list[str] = []
    listed = {entry.path for entry in manifest.files}
    for entry in manifest.files:
        try:
            path = part_path(root, manifest, entry.path)
        except ManifestError as exc:
            violations.append(str(exc))
            continue
        if not path.exists():
            violations.append(f"{entry.path}: part file missing")
            continue
        try:
            actual = sha256_file(path)
        except OSError as exc:
            violations.append(f"{entry.path}: unreadable part: {exc}")
            continue
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

"""Audit-coverage ratchet — 'audited' is a CI-enforced claim, not a vibe.

``quality/audit_coverage.json`` maps every ``src/quant_fund`` module to a
status (``audited`` / ``partial`` / ``pending`` / ``waived``);
``quality/audit_coverage_fx1.json`` does the same for ``src/fx1``. The
ratchet:

- every module must resolve to a status (module override wins, else its
  top-level directory entry);
- ``audited`` directories pin ``n_modules``: adding or removing a file under
  an audited directory fails until the manifest is updated — a new file can
  never silently inherit an audit it never received;
- ``waived`` entries require a non-empty ``reason``;
- ``doc`` references must point at real files;
- stale manifest keys (renamed/deleted modules or directories) fail;
- the audited+waived module floor only ratchets up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALID_STATUSES = {"audited", "partial", "pending", "waived"}


@dataclass(frozen=True)
class AuditRoot:
    src: Path
    manifest_path: Path
    audited_floor: int


#: One manifest per source tree; each floor may only increase.
ROOTS = (
    AuditRoot(ROOT / "src" / "quant_fund", ROOT / "quality" / "audit_coverage.json", 446),
    AuditRoot(ROOT / "src" / "fx1", ROOT / "quality" / "audit_coverage_fx1.json", 65),
)


def _src_modules(src: Path) -> list[str]:
    return sorted(
        p.relative_to(src).as_posix() for p in src.rglob("*.py") if "__pycache__" not in p.parts
    )


def _manifest(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return loaded


def _entries(manifest: dict[str, object], key: str) -> dict[str, dict[str, object]]:
    raw = manifest[key]
    if not isinstance(raw, dict):
        raise ValueError(f"audit_coverage manifest[{key!r}] must be an object")
    return raw  # type: ignore[return-value]


def _module_status(manifest: dict[str, object], mod: str) -> str:
    modules = _entries(manifest, "modules")
    if mod in modules:
        return str(modules[mod]["status"])
    top = mod.split("/")[0] if "/" in mod else mod
    directories = _entries(manifest, "directories")
    return str(directories[top]["status"])


def test_manifest_schema_and_references() -> None:
    for root in ROOTS:
        manifest = _manifest(root.manifest_path)
        assert manifest["schema_version"] == "audit-coverage/v1"
        statuses = manifest["statuses"]
        assert isinstance(statuses, dict)
        assert set(statuses) == VALID_STATUSES

        src_modules = _src_modules(root.src)
        real_dirs = {m.split("/")[0] for m in src_modules if "/" in m}
        real_files = set(src_modules)

        directories = _entries(manifest, "directories")
        modules = _entries(manifest, "modules")

        # No stale keys: every manifest entry must name a live dir or file.
        for key, entry in directories.items():
            assert key in real_dirs, f"{root.manifest_path.name}: stale directory entry: {key}"
            assert entry["status"] in VALID_STATUSES
            if entry["status"] == "waived":
                assert str(entry.get("reason", "")).strip(), key
            doc = entry.get("doc")
            if doc is not None:
                assert (ROOT / str(doc)).is_file(), f"{key}: missing doc {doc}"
        for key, entry in modules.items():
            assert key in real_files, f"{root.manifest_path.name}: stale module entry: {key}"
            assert entry["status"] in VALID_STATUSES
            if entry["status"] == "waived":
                assert str(entry.get("reason", "")).strip(), key
            doc = entry.get("doc")
            if doc is not None:
                assert (ROOT / str(doc)).is_file(), f"{key}: missing doc {doc}"


def test_every_module_has_a_status() -> None:
    for root in ROOTS:
        manifest = _manifest(root.manifest_path)
        directories = _entries(manifest, "directories")
        modules = _entries(manifest, "modules")
        uncovered = []
        for mod in _src_modules(root.src):
            if mod in modules:
                continue
            top = mod.split("/")[0] if "/" in mod else mod
            if top not in directories:
                uncovered.append(mod)
        assert not uncovered, (
            f"{root.manifest_path.name}: modules with no audit-coverage entry: "
            + ", ".join(uncovered)
            + f" — add a directory entry to {root.manifest_path.relative_to(ROOT)}"
        )


def test_audited_dirs_pin_file_count() -> None:
    """A file landing in an audited dir can't silently inherit 'audited'."""
    for root in ROOTS:
        manifest = _manifest(root.manifest_path)
        directories = _entries(manifest, "directories")
        modules = _entries(manifest, "modules")
        src_modules = _src_modules(root.src)
        drift = []
        for top, entry in directories.items():
            if entry["status"] != "audited":
                continue
            pinned = entry.get("n_modules")
            assert isinstance(pinned, int), f"{top}: audited dir must pin n_modules"
            actual = sum(
                1
                for m in src_modules
                if (m.split("/")[0] if "/" in m else m) == top
                and modules.get(m, {}).get("status") != "waived"
            )
            if actual != pinned:
                drift.append((top, pinned, actual))
        assert not drift, (
            f"{root.manifest_path.name}: audited directories gained/lost modules "
            "without a manifest update: "
            + ", ".join(f"{d} pinned={p} actual={a}" for d, p, a in drift)
            + " — audit the new module or downgrade the directory status"
        )


def test_audited_floor_only_grows() -> None:
    for root in ROOTS:
        manifest = _manifest(root.manifest_path)
        n_audited = sum(
            1 for mod in _src_modules(root.src) if _module_status(manifest, mod) == "audited"
        )
        assert n_audited >= root.audited_floor, (
            f"{root.manifest_path.name}: audited module count fell to {n_audited} "
            f"(< {root.audited_floor}); audit coverage only ratchets up"
        )

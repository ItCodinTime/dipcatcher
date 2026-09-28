"""Audit-coverage ratchet — 'audited' is a CI-enforced claim, not a vibe.

``quality/audit_coverage.json`` maps every ``src/quant_fund`` module to a
status (``audited`` / ``partial`` / ``pending`` / ``waived``). The ratchet:

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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "quality" / "audit_coverage.json"
SRC = ROOT / "src" / "quant_fund"
VALID_STATUSES = {"audited", "partial", "pending", "waived"}

#: Floor for modules carrying 'audited' status. May only increase.
AUDITED_FLOOR = 338


def _src_modules() -> list[str]:
    return sorted(
        p.relative_to(SRC).as_posix() for p in SRC.rglob("*.py") if "__pycache__" not in p.parts
    )


def _manifest() -> dict[str, object]:
    loaded = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("audit_coverage.json must be a JSON object")
    return loaded


def test_manifest_schema_and_references() -> None:
    manifest = _manifest()
    assert manifest["schema_version"] == "audit-coverage/v1"
    statuses = manifest["statuses"]
    assert isinstance(statuses, dict)
    assert set(statuses) == VALID_STATUSES

    real_dirs = {m.split("/")[0] for m in _src_modules() if "/" in m}
    real_files = set(_src_modules())

    directories = _entries(manifest, "directories")
    modules = _entries(manifest, "modules")

    # No stale keys: every manifest entry must name a live dir or file.
    for key, entry in directories.items():
        assert key in real_dirs, f"stale directory entry: {key}"
        assert entry["status"] in VALID_STATUSES
        if entry["status"] == "waived":
            assert str(entry.get("reason", "")).strip(), key
        doc = entry.get("doc")
        if doc is not None:
            assert (ROOT / str(doc)).is_file(), f"{key}: missing doc {doc}"
    for key, entry in modules.items():
        assert key in real_files, f"stale module entry: {key}"
        assert entry["status"] in VALID_STATUSES
        if entry["status"] == "waived":
            assert str(entry.get("reason", "")).strip(), key
        doc = entry.get("doc")
        if doc is not None:
            assert (ROOT / str(doc)).is_file(), f"{key}: missing doc {doc}"


def _entries(manifest: dict[str, object], key: str) -> dict[str, dict[str, object]]:
    raw = manifest[key]
    if not isinstance(raw, dict):
        raise ValueError(f"audit_coverage.json[{key!r}] must be an object")
    return raw  # type: ignore[return-value]


def _module_status(manifest: dict[str, object], mod: str) -> str:
    modules = _entries(manifest, "modules")
    if mod in modules:
        return str(modules[mod]["status"])
    top = mod.split("/")[0] if "/" in mod else mod
    directories = _entries(manifest, "directories")
    return str(directories[top]["status"])


def test_every_module_has_a_status() -> None:
    manifest = _manifest()
    directories = _entries(manifest, "directories")
    modules = _entries(manifest, "modules")
    uncovered = []
    for mod in _src_modules():
        if mod in modules:
            continue
        top = mod.split("/")[0] if "/" in mod else mod
        if top not in directories:
            uncovered.append(mod)
    assert not uncovered, (
        "modules with no audit-coverage entry: "
        + ", ".join(uncovered)
        + " — add a directory entry to quality/audit_coverage.json"
    )


def test_audited_dirs_pin_file_count() -> None:
    """A file landing in an audited dir can't silently inherit 'audited'."""
    manifest = _manifest()
    directories = _entries(manifest, "directories")
    modules = _entries(manifest, "modules")
    drift = []
    for top, entry in directories.items():
        if entry["status"] != "audited":
            continue
        pinned = entry.get("n_modules")
        assert isinstance(pinned, int), f"{top}: audited dir must pin n_modules"
        actual = sum(
            1
            for m in _src_modules()
            if (m.split("/")[0] if "/" in m else m) == top
            and modules.get(m, {}).get("status") != "waived"
        )
        if actual != pinned:
            drift.append((top, pinned, actual))
    assert not drift, (
        "audited directories gained/lost modules without a manifest update: "
        + ", ".join(f"{d} pinned={p} actual={a}" for d, p, a in drift)
        + " — audit the new module or downgrade the directory status"
    )


def test_audited_floor_only_grows() -> None:
    manifest = _manifest()
    n_audited = sum(1 for mod in _src_modules() if _module_status(manifest, mod) == "audited")
    assert n_audited >= AUDITED_FLOOR, (
        f"audited module count fell to {n_audited} (< {AUDITED_FLOOR}); "
        "audit coverage only ratchets up"
    )

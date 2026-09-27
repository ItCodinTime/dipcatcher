"""PitVault: write-once, content-addressed, bitemporal data store (§4).

One physical choke point for all time-series data reads. Every record carries
``(event_time, known_at)``; the ONLY legal read path is ``asof(t)``, which
cannot return rows with ``known_at > t`` because the filter is applied inside
the vault before any caller code sees the frame (DESIGN.md §4).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, cast

import polars as pl

from quant_fund.pit import manifest as manifest_mod
from quant_fund.pit.corrections import (
    EVENT_TIME_COL,
    KNOWN_AT_COL,
    SECURITY_ID_COL,
    RestatementPolicy,
    frame_span,
    key_columns,
    normalize_pit_frame,
    prepare_correction,
    require_pit_frame,
    select_asof,
)
from quant_fund.pit.frame import PitFrame, VaultUnavailableError
from quant_fund.proofcore.contracts import (
    GENESIS_HASH,
    DataAccessRecord,
    ManifestError,
    PitManifest,
    PitManifestFile,
    VaultError,
    sha256_hex_bytes,
)


class DataAccessRecorder(Protocol):
    """Recorder hook (W2 ``proof/recorder.py`` implements this structurally)."""

    def record(self, read: DataAccessRecord) -> None: ...


class WatchdogProtocol(Protocol):
    """Watchdog hook (W3 ``leakage/watchdog.py`` implements this structurally)."""

    def observe(self, read: DataAccessRecord, decision_time: datetime) -> None: ...


def _require_aware(t: datetime, *, what: str) -> None:
    if t.tzinfo is None or t.tzinfo.utcoffset(t) is None:
        raise VaultError(f"{what} must be timezone-aware (UTC)")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PitVault:
    """Write-once content-addressed bitemporal store rooted at ``root``."""

    def __init__(
        self,
        root: Path,
        *,
        recorder: DataAccessRecorder | None = None,
        watchdog: WatchdogProtocol | None = None,
    ) -> None:
        self.root = Path(root)
        self.recorder = recorder
        self.watchdog = watchdog

    # -- dataset management -------------------------------------------------

    def _validate_name(self, name: str) -> None:
        parts = Path(name).parts
        if (
            not name
            or Path(name).is_absolute()
            or "\\" in name
            or any(part in ("", ".", "..") for part in parts)
        ):
            raise VaultError(f"illegal dataset name: {name!r}")

    def _dataset_dir(self, name: str) -> Path:
        self._validate_name(name)
        return manifest_mod.dataset_dir(self.root, name)

    def _security_level(self, name: str) -> bool:
        directory = self._dataset_dir(name)
        meta_path = directory / manifest_mod.DATASET_META_NAME
        if not manifest_mod.manifest_path(self.root, name).exists() or not meta_path.exists():
            raise VaultError(f"unknown dataset: {name!r}")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise VaultError(f"{name}: dataset.json malformed: {exc}") from exc
        return bool(meta.get("security_level", True))

    def create_dataset(self, name: str, *, security_level: bool = True) -> None:
        """Create an empty dataset (revision 0 manifest chained to GENESIS)."""
        directory = self._dataset_dir(name)
        if manifest_mod.manifest_path(self.root, name).exists():
            raise VaultError(f"dataset already exists: {name!r}")
        (directory / manifest_mod.PARTS_DIR).mkdir(parents=True, exist_ok=True)
        meta = {"dataset": name, "security_level": bool(security_level)}
        manifest_mod._atomic_write(
            directory / manifest_mod.DATASET_META_NAME,
            json.dumps(meta, sort_keys=True).encode("utf-8"),
        )
        manifest = PitManifest(
            dataset=name,
            created_utc=_now_iso(),
            revision=0,
            prev_manifest_sha256=GENESIS_HASH,
            files=[],
        )
        manifest_mod.write_manifest(
            self.root, name, manifest, prev_manifest_sha256=GENESIS_HASH
        )

    def list_datasets(self) -> list[str]:
        """All datasets under root (dirs carrying a manifest.json), sorted."""
        if not self.root.is_dir():
            return []
        found = [
            path.parent.relative_to(self.root).as_posix()
            for path in self.root.rglob(manifest_mod.MANIFEST_NAME)
        ]
        return sorted(found)

    # -- writes ---------------------------------------------------------------

    def append(self, name: str, frame: pl.DataFrame) -> PitManifest:
        """Validate PIT cols, write r{rev}.parquet, update manifest chain."""
        security_level = self._security_level(name)
        require_pit_frame(frame, security_level=security_level)
        frame = normalize_pit_frame(frame)
        current = manifest_mod.read_manifest(self.root, name)
        current_bytes = manifest_mod.manifest_path(self.root, name).read_bytes()
        current_sha = sha256_hex_bytes(current_bytes)

        revision = current.revision + 1
        part_rel = f"{name}/{manifest_mod.PARTS_DIR}/r{revision:07d}.parquet"
        part_path = self.root / part_rel
        if part_path.exists():
            # Write-once: a part whose name exists is never overwritten (§4.1).
            raise VaultError(f"write-once violation: part already exists: {part_rel}")
        frame.write_parquet(part_path)

        min_ka, max_ka, min_et, max_et = frame_span(frame)
        entry = PitManifestFile(
            path=part_rel,
            sha256=manifest_mod.sha256_file(part_path),
            rows=frame.height,
            min_known_at=min_ka,
            max_known_at=max_ka,
            min_event_time=min_et,
            max_event_time=max_et,
        )
        updated = PitManifest(
            dataset=name,
            created_utc=_now_iso(),
            revision=revision,
            prev_manifest_sha256=current_sha,
            files=[*current.files, entry],
        )
        manifest_mod.write_manifest(self.root, name, updated, prev_manifest_sha256=current_sha)
        return updated

    def restate(self, name: str, corrected: pl.DataFrame, *, known_at: datetime) -> PitManifest:
        """Append corrections with explicit known_at; old parts untouched."""
        return self.append(name, prepare_correction(corrected, known_at=known_at))

    # -- THE ONLY READ PATH -----------------------------------------------------

    def asof(
        self,
        name: str,
        t: datetime,
        *,
        columns: list[str] | None = None,
        policy: RestatementPolicy = RestatementPolicy.LATEST_KNOWN,
    ) -> PitFrame:
        """THE ONLY READ PATH. Fails closed:

        - VaultUnavailableError if no version with known_at <= t exists
        - VaultError if t naive, dataset missing, or manifest corrupt
        - records the read into recorder + watchdog if attached
        """
        _require_aware(t, what="asof timestamp")
        t = t.astimezone(timezone.utc)
        security_level = self._security_level(name)  # VaultError if dataset missing
        current = manifest_mod.read_manifest(self.root, name)  # ManifestError if corrupt
        if not current.files:
            raise VaultUnavailableError(
                f"{name}: no versions at all — nothing observable as of {t.isoformat()}"
            )
        parts_glob = str(self._dataset_dir(name) / manifest_mod.PARTS_DIR / "r*.parquet")
        scan = pl.scan_parquet(parts_glob)
        frame = select_asof(
            scan,
            t,
            key_cols=key_columns(security_level=security_level),
            policy=policy,
            columns=columns,
        )
        if frame.height == 0:
            raise VaultUnavailableError(
                f"{name}: no version with known_at <= {t.isoformat()} exists"
            )
        pit_frame = PitFrame.build(frame, dataset=name, asof=t)
        pit_frame.validate(t)  # defense in depth: re-check before caller sees rows
        self._observe(pit_frame, t, columns=columns, policy=policy)
        return pit_frame

    def _observe(
        self,
        pit_frame: PitFrame,
        t: datetime,
        *,
        columns: list[str] | None,
        policy: RestatementPolicy,
    ) -> None:
        if self.recorder is None and self.watchdog is None:
            return
        params = {"policy": policy.value}
        if columns is not None:
            params["columns"] = ",".join(str(c) for c in columns)
        read = DataAccessRecord(
            dataset=pit_frame.dataset,
            asof_utc=t.isoformat(),
            params=params,
            rows=pit_frame.rows,
            content_sha256=pit_frame.content_sha256,
        )
        # The proof commits to exactly the bytes the strategy consumed (§5.1).
        if self.recorder is not None:
            self.recorder.record(read)
        if self.watchdog is not None:
            self.watchdog.observe(read, t)

    # -- audit ------------------------------------------------------------------

    def verify(self, name: str) -> list[str]:
        """Re-hash all parts vs manifest; return list of violations ([] = ok)."""
        try:
            current = manifest_mod.read_manifest(self.root, name)
        except ManifestError as exc:
            return [str(exc)]
        violations = manifest_mod.verify_part_hashes(self.root, current)
        for entry in current.files:
            path = self.root / entry.path
            if not path.exists():
                continue  # already flagged by verify_part_hashes
            stats = pl.scan_parquet(path).select(
                pl.len(),
                pl.col(KNOWN_AT_COL).min().alias("min_known_at"),
                pl.col(KNOWN_AT_COL).max().alias("max_known_at"),
                pl.col(EVENT_TIME_COL).min().alias("min_event_time"),
                pl.col(EVENT_TIME_COL).max().alias("max_event_time"),
            )
            try:
                row = stats.collect().row(0)
            except Exception as exc:  # unreadable part: report, never raise
                violations.append(f"{entry.path}: unreadable part: {exc}")
                continue
            if row[0] != entry.rows:
                violations.append(f"{entry.path}: rows {row[0]} != manifest {entry.rows}")
            for idx, field in enumerate(
                ("min_known_at", "max_known_at", "min_event_time", "max_event_time"), start=1
            ):
                actual = cast(datetime, row[idx]).isoformat()
                if actual != getattr(entry, field):
                    violations.append(
                        f"{entry.path}: {field} {actual} != manifest {getattr(entry, field)}"
                    )
        return violations

    def history(self, name: str, key: tuple[str, datetime]) -> pl.DataFrame:
        """All versions of one (security_id, event_time) — audit/debug only.

        MUST NOT be callable from strategy code paths (LH009 grep gate).
        """
        security_id, event_time = key
        _require_aware(event_time, what="history event_time")
        security_level = self._security_level(name)
        current = manifest_mod.read_manifest(self.root, name)
        if not current.files:
            return pl.DataFrame()
        parts_glob = str(self._dataset_dir(name) / manifest_mod.PARTS_DIR / "r*.parquet")
        scan = pl.scan_parquet(parts_glob).filter(
            pl.col(EVENT_TIME_COL) == pl.lit(event_time.astimezone(timezone.utc))
        )
        if security_level:
            scan = scan.filter(pl.col(SECURITY_ID_COL) == pl.lit(security_id))
        return scan.sort(KNOWN_AT_COL).collect()

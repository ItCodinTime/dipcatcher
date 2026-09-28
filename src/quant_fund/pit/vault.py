"""PitVault: write-once, content-addressed, bitemporal data store (§4).

One physical choke point for all time-series data reads. Every record carries
``(event_time, known_at)``; the ONLY legal read path is ``asof(t)``, which
cannot return rows with ``known_at > t`` because the filter is applied inside
the vault before any caller code sees the frame (DESIGN.md §4).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
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
from quant_fund.proofcore import run_context
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


_LOG = logging.getLogger(__name__)


def _require_aware(t: datetime, *, what: str) -> None:
    if t.tzinfo is None or t.tzinfo.utcoffset(t) is None:
        raise VaultError(f"{what} must be timezone-aware (UTC)")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@contextmanager
def _dataset_write_lock(directory: Path):
    """Serialize appends across processes; SQLite releases the lock on crash."""
    connection = sqlite3.connect(directory / ".append-lock.sqlite3", timeout=30)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        yield
    finally:
        connection.rollback()
        connection.close()


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
        self._warned_auto_attach = False

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
        directory = manifest_mod.dataset_dir(self.root, name)
        if not directory.resolve().is_relative_to(self.root.resolve()):
            raise VaultError(f"dataset path escapes vault root: {name!r}")
        return directory

    def _dataset_meta(self, name: str) -> dict[str, object]:
        directory = self._dataset_dir(name)
        meta_path = directory / manifest_mod.DATASET_META_NAME
        if not manifest_mod.manifest_path(self.root, name).exists() or not meta_path.exists():
            raise VaultError(f"unknown dataset: {name!r}")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict) or not isinstance(meta.get("security_level"), bool):
                raise ValueError("dataset metadata must declare a boolean security_level")
            if not isinstance(meta.get("monotonic_known_at", False), bool):
                raise ValueError("monotonic_known_at must be boolean")
            return meta
        except ValueError as exc:
            raise VaultError(f"{name}: dataset.json malformed: {exc}") from exc

    def _security_level(self, name: str) -> bool:
        return bool(self._dataset_meta(name).get("security_level", True))

    def create_dataset(
        self,
        name: str,
        *,
        security_level: bool = True,
        monotonic_known_at: bool = False,
    ) -> None:
        """Create an empty dataset (revision 0 manifest chained to GENESIS).

        ``monotonic_known_at`` (ADVERSARIAL §1b-W4, opt-in): enforce that each
        append's min known_at is >= the dataset's existing max known_at, so
        writers cannot launder future rows behind backdated knowledge times.
        Default False = warn-only (a backdated append logs a warning).
        """
        directory = self._dataset_dir(name)
        if manifest_mod.manifest_path(self.root, name).exists():
            raise VaultError(f"dataset already exists: {name!r}")
        (directory / manifest_mod.PARTS_DIR).mkdir(parents=True, exist_ok=True)
        meta = {
            "dataset": name,
            "security_level": bool(security_level),
            "monotonic_known_at": bool(monotonic_known_at),
        }
        manifest_mod._atomic_write(
            directory / manifest_mod.DATASET_META_NAME,
            json.dumps(meta, sort_keys=True).encode("utf-8"),
        )
        manifest = PitManifest(
            dataset=name,
            created_utc=_now_iso(),
            revision=0,
            prev_manifest_sha256=GENESIS_HASH,
            files=(),
        )
        manifest_mod.write_manifest(self.root, name, manifest, prev_manifest_sha256=GENESIS_HASH)

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
        """Validate PIT cols, write r{rev}.parquet, update manifest chain.

        Backdated-known_at policy (ADVERSARIAL §1b-W4): if the frame's minimum
        ``known_at`` predates the dataset's existing maximum ``known_at``, the
        append is backdating knowledge. Datasets created with
        ``monotonic_known_at=True`` reject it (VaultError); otherwise it is
        logged as a warning (default warn-only posture).
        """
        meta = self._dataset_meta(name)
        security_level = bool(meta.get("security_level", True))
        require_pit_frame(frame, security_level=security_level)
        frame = normalize_pit_frame(frame)
        directory = self._dataset_dir(name)
        with _dataset_write_lock(directory):
            for stale in (directory / manifest_mod.PARTS_DIR).glob(".r*.tmp"):
                stale.unlink()
            current = manifest_mod.read_manifest(self.root, name)
            if current.files:
                # ADVERSARIAL §1b-W4: monotonic known_at watermark check.
                append_min_ka = frame_span(frame)[0]
                existing_max_ka = max(f.max_known_at for f in current.files)
                if append_min_ka < existing_max_ka:
                    message = (
                        f"{name}: backdated append — min known_at {append_min_ka} < "
                        f"existing max known_at {existing_max_ka} (writer-stamped "
                        "known_at can launder future rows into PIT reads)"
                    )
                    if bool(meta.get("monotonic_known_at", False)):
                        raise VaultError(message)
                    _LOG.warning("%s (warn-only; set monotonic_known_at=True to reject)", message)
            current_bytes = manifest_mod.manifest_path(self.root, name).read_bytes()
            current_sha = sha256_hex_bytes(current_bytes)
            revision = current.revision + 1
            part_rel = f"{name}/{manifest_mod.PARTS_DIR}/r{revision:07d}.parquet"
            part_path = self.root / part_rel
            if part_path.exists():
                # An uncommitted orphan must be inspected or explicitly recovered.
                raise VaultError(f"write-once violation: part already exists: {part_rel}")

            existing_parts = self._verified_parts(current)
            if existing_parts:
                previous_schema = pl.scan_parquet(existing_parts[0]).collect_schema()
                if dict(frame.schema) != dict(previous_schema):
                    raise VaultError(
                        f"{name}: incompatible part schema: {frame.schema} != {previous_schema}"
                    )
                frame = frame.select(previous_schema.names())

            temporary_path: Path | None = None
            try:
                with NamedTemporaryFile(
                    dir=part_path.parent,
                    prefix=f".r{revision:07d}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                frame.write_parquet(temporary_path)
                # r+b: fsync needs a write-capable descriptor on Windows;
                # an rb handle raises OSError(EBADF) there.
                with temporary_path.open("r+b") as handle:
                    os.fsync(handle.fileno())
                # link() is an atomic exclusive publish: it cannot replace a part
                # created by another writer or a crashed earlier append.
                try:
                    os.link(temporary_path, part_path)
                except FileExistsError as exc:
                    raise VaultError(
                        f"write-once violation: part already exists: {part_rel}"
                    ) from exc
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

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
                files=(*current.files, entry),
            )
            manifest_mod.write_manifest(self.root, name, updated, prev_manifest_sha256=current_sha)
            return updated

    def recover_uncommitted_part(self, name: str) -> bool:
        """Remove only the next uncommitted part after operator inspection."""
        directory = self._dataset_dir(name)
        with _dataset_write_lock(directory):
            current = manifest_mod.read_manifest(self.root, name)
            revision = current.revision + 1
            orphan = directory / manifest_mod.PARTS_DIR / f"r{revision:07d}.parquet"
            if manifest_mod.versioned_manifest_path(self.root, name, revision).exists():
                raise VaultError(f"{name}: revision {revision} has a retained manifest")
            if not orphan.exists():
                return False
            if orphan.is_symlink() or not orphan.is_file():
                raise VaultError(f"{name}: unsafe uncommitted part: {orphan}")
            orphan.unlink()
            return True

    def recover_interrupted_manifest(self, name: str) -> bool:
        """Commit a complete pending snapshot after an interrupted append."""
        directory = self._dataset_dir(name)
        with _dataset_write_lock(directory):
            return manifest_mod.recover_interrupted_manifest(self.root, name)

    def _verified_parts(self, manifest: PitManifest) -> list[Path]:
        """Return only manifest-listed, hash-verified part paths."""
        paths: list[Path] = []
        for entry in manifest.files:
            path = manifest_mod.part_path(self.root, manifest, entry.path)
            if not path.is_file():
                raise ManifestError(f"{entry.path}: committed part missing")
            if manifest_mod.sha256_file(path) != entry.sha256:
                raise ManifestError(f"{entry.path}: committed part sha256 mismatch")
            paths.append(path)
        return paths

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
        decision_time: datetime | None = None,
    ) -> PitFrame:
        """THE ONLY READ PATH. Fails closed:

        - VaultUnavailableError if no version with known_at <= t exists
        - VaultError if t naive, dataset missing, or manifest corrupt
        - records the read into recorder + watchdog if attached

        ``t`` is the READ WATERMARK (only rows with known_at <= t are
        returned). ``decision_time`` is the time the strategy is deciding at;
        the watchdog asserts max(known_at of returned rows) <= decision_time.
        If omitted, the active proven-run decision window supplies it
        (proofcore.run_context); if neither exists the decision time falls
        back to ``t`` itself (legacy watermark-only behavior outside proven
        runs. An active proven context without a decision window is rejected).
        """
        _require_aware(t, what="asof timestamp")
        t = t.astimezone(UTC)
        security_level = self._security_level(name)  # VaultError if dataset missing
        current = manifest_mod.read_manifest(self.root, name)  # ManifestError if corrupt
        if not current.files:
            raise VaultUnavailableError(
                f"{name}: no versions at all — nothing observable as of {t.isoformat()}"
            )
        scan = pl.scan_parquet(self._verified_parts(current))
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
        self._observe(pit_frame, t, columns=columns, policy=policy, decision_time=decision_time)
        return pit_frame

    def _observe(
        self,
        pit_frame: PitFrame,
        t: datetime,
        *,
        columns: list[str] | None,
        policy: RestatementPolicy,
        decision_time: datetime | None,
    ) -> None:
        active_recorder = run_context.active_recorder()
        active_watchdog = run_context.active_watchdog()
        clock = run_context.current_decision_time()
        if active_recorder is not None and clock is None:
            raise VaultError("active proven run requires an explicit decision window")
        if decision_time is not None:
            _require_aware(decision_time, what="decision time")
        # The active context is authoritative even if the vault has private hooks.
        recorder = active_recorder if active_recorder is not None else self.recorder
        watchdog = active_watchdog if active_watchdog is not None else self.watchdog
        attached = active_recorder is not None and active_recorder is not self.recorder
        # ADVERSARIAL §1b-W2: re-resolve against the active proven-run context
        # so vaults created before the runner entered its run context are
        # still recorded. ADVERSARIAL R2 §1-W6: active_recorder/watchdog fall
        # back to the run's thread-visible registry, so reads from worker
        # threads carrying no proven-run context attach too (fail-loud).
        if recorder is None:
            recorder = run_context.active_recorder()
            attached = recorder is not None
        if watchdog is None:
            context_watchdog = run_context.active_watchdog()
            if context_watchdog is not None:
                watchdog = context_watchdog
                attached = True
        if attached and not self._warned_auto_attach:
            self._warned_auto_attach = True
            if run_context.context_is_proven():
                _LOG.warning(
                    "PitVault(%s) has no recorder/watchdog of its own; auto-attached "
                    "to the active proven run's hooks so this read is proven",
                    self.root,
                )
            else:
                _LOG.warning(
                    "PitVault(%s) read from a thread with no proven-run context "
                    "while a proven run is active; cross-thread auto-attached to "
                    "the run's recorder/watchdog so this read is proven (use "
                    "proofcore.run_context.proven_thread to propagate the context)",
                    self.root,
                )
        if recorder is None and watchdog is None:
            return
        params = {"policy": policy.value}
        if columns is not None:
            params["columns"] = ",".join(str(c) for c in columns)
        # W1<->W3 seam (adjudicated): the strict LeakageWatchdog fail-closes
        # without a max_known_at watermark on every observed read.
        params["max_known_at"] = pit_frame.max_known_at.isoformat()
        read = DataAccessRecord(
            dataset=pit_frame.dataset,
            asof_utc=t.isoformat(),
            params=params,
            rows=pit_frame.rows,
            content_sha256=pit_frame.content_sha256,
        )
        # The proof commits to exactly the bytes the strategy consumed (§5.1).
        if recorder is not None:
            recorder.record(read)
        if watchdog is not None:
            # ADVERSARIAL §1b-W1 fix: the watchdog compares against the
            # DECISION time (explicit argument, else the active decision
            # window), never blindly against the read watermark — passing the
            # asof argument as the decision time made the check tautological.
            effective_decision = clock or decision_time or t
            if clock is not None and decision_time is not None:
                effective_decision = min(clock, decision_time)
            watchdog.observe(read, effective_decision)

    # -- audit ------------------------------------------------------------------

    def verify(self, name: str) -> list[str]:
        """Re-hash all parts vs manifest; return list of violations ([] = ok)."""
        try:
            current = manifest_mod.read_manifest(self.root, name)
        except ManifestError as exc:
            return [str(exc)]
        violations = manifest_mod.verify_part_hashes(self.root, current)
        for entry in current.files:
            try:
                path = manifest_mod.part_path(self.root, current, entry.path)
            except ManifestError:
                continue  # already flagged by verify_part_hashes
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
        scan = pl.scan_parquet(self._verified_parts(current)).filter(
            pl.col(EVENT_TIME_COL) == pl.lit(event_time.astimezone(UTC))
        )
        if security_level:
            scan = scan.filter(pl.col(SECURITY_ID_COL) == pl.lit(security_id))
        return scan.sort(KNOWN_AT_COL).collect()

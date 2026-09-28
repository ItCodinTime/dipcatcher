"""Append-only, hash-chained audit ledger.

Entries are research runs, paper decisions, simulated orders, fills, and risk
decisions. Each line is the canonical JSON of one entry. The file is only
opened for append. A periodic Merkle root over the ordered entries is signed
and stored in a second append-only checkpoint file.

This is not the paper parquet ledger in ``quant_fund.paper.ledger``. That
store holds simulated broker rows. This store holds tamper-evident commitments
to those rows and to research receipts.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_fund.audit.canonical import canonical_json_bytes, json_safe, sha256_hex
from quant_fund.audit.errors import AuditError
from quant_fund.audit.merkle import merkle_root
from quant_fund.audit.signing import Signature

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

SCHEMA_VERSION = 1
GENESIS_HASH = "0" * 64
KINDS = (
    "research_run",
    "paper_decision",
    "simulated_order",
    "fill",
    "risk_decision",
)
_KIND_SET = frozenset(KINDS)
_HEX = frozenset("0123456789abcdef")


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hex64(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in _HEX for char in value)


def _strict_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


@dataclass(frozen=True)
class LedgerEntry:
    """One committed log entry. ``preimage`` is the Merkle leaf input."""

    index: int
    kind: str
    recorded_at: str
    payload: dict[str, Any]
    prev_hash: str
    entry_hash: str

    def body(self) -> dict[str, Any]:
        return {
            "v": SCHEMA_VERSION,
            "index": self.index,
            "kind": self.kind,
            "recorded_at": self.recorded_at,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
        }

    def preimage(self) -> bytes:
        return canonical_json_bytes(self.body())

    def to_obj(self) -> dict[str, Any]:
        full = self.body()
        full["entry_hash"] = self.entry_hash
        return full


def entry_from_obj(obj: dict[str, Any]) -> tuple[LedgerEntry | None, str | None]:
    """Parse one JSON object. Returns ``(entry, error)``."""
    if set(obj) != {"v", "index", "kind", "recorded_at", "payload", "prev_hash", "entry_hash"}:
        return None, "unexpected_entry_fields"
    index = _strict_int(obj.get("index"))
    version = _strict_int(obj.get("v"))
    kind = obj.get("kind")
    recorded_at = obj.get("recorded_at")
    payload = obj.get("payload")
    prev_hash = obj.get("prev_hash")
    entry_hash = obj.get("entry_hash")
    if version != SCHEMA_VERSION:
        return None, "bad_version"
    if index is None or index < 0:
        return None, "bad_index"
    if not isinstance(kind, str) or kind not in _KIND_SET:
        return None, "unknown_kind"
    if not isinstance(recorded_at, str) or not recorded_at:
        return None, "bad_timestamp"
    if not isinstance(payload, dict):
        return None, "bad_payload"
    if not _hex64(prev_hash) or not _hex64(entry_hash):
        return None, "bad_hash"
    entry = LedgerEntry(
        index=index,
        kind=kind,
        recorded_at=recorded_at,
        payload=payload,
        prev_hash=str(prev_hash),
        entry_hash=str(entry_hash),
    )
    expected = sha256_hex(entry.preimage())
    if expected != entry.entry_hash:
        return entry, "entry_hash_mismatch"
    return entry, None


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load an append-only JSONL file.

    A missing file is an empty log. A missing trailing newline, a blank line,
    non-UTF-8 bytes, or a line that is not canonical JSON is an error. Parsed
    objects are still returned for the lines that parsed, so callers can report
    the first corruption rather than hiding it.
    """
    try:
        if path.is_symlink():
            return [], ["unsafe_symlink"]
        if not path.exists():
            return [], []
        if not path.is_file():
            return [], ["not_regular_file"]
        raw = path.read_bytes()
    except OSError as exc:
        return [], [f"unreadable:{exc.__class__.__name__}"]
    errors: list[str] = []
    if not raw:
        return [], []
    if not raw.endswith(b"\n"):
        errors.append("truncated_record")
    if b"\r" in raw:
        errors.append("carriage_return")
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    objects: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if line == b"":
            errors.append(f"blank_line:{index}")
            continue
        try:
            text = line.decode("utf-8")
            parsed = json.loads(text)
        except (UnicodeError, json.JSONDecodeError):
            errors.append(f"malformed_line:{index}")
            continue
        if not isinstance(parsed, dict):
            errors.append(f"malformed_line:{index}")
            continue
        try:
            canonical = canonical_json_bytes(parsed)
        except AuditError:
            errors.append(f"invalid_json_value:{index}")
            continue
        if line != canonical:
            errors.append(f"noncanonical_line:{index}")
        objects.append(parsed)
    return objects, errors


def _forbidden_research_keys(payload: dict[str, Any]) -> set[str]:
    from quant_fund.research.catalog import FORBIDDEN_RESEARCH_METRIC_KEYS

    return {key for key in payload if key.lower() in FORBIDDEN_RESEARCH_METRIC_KEYS}


class AuditLedger:
    """Directory containing ``entries.jsonl`` and ``checkpoints.jsonl``."""

    def __init__(
        self,
        root: Path | str,
        *,
        signer: Any | None = None,
        sign_every: int = 1,
        sync: bool = True,
        clock: Callable[[], str] | None = None,
    ) -> None:
        if isinstance(sign_every, bool) or not isinstance(sign_every, int) or sign_every < 1:
            raise AuditError("sign_every must be a positive integer")
        self.root = Path(root)
        self.signer = signer
        self.sign_every = sign_every
        self.sync = sync
        self.clock = clock or _now
        self.entries_path = self.root / "entries.jsonl"
        self.checkpoints_path = self.root / "checkpoints.jsonl"

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            self.root / ".lock", os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0), 0o600
        )
        locked = False
        try:
            if sys.platform == "win32":
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"0")
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_EX)
            locked = True
            yield
        finally:
            try:
                if locked:
                    if sys.platform == "win32":
                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def append(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        recorded_at: str | None = None,
    ) -> LedgerEntry:
        """Append one entry and, when due, a signed tree head.

        Earlier bytes of ``entries.jsonl`` are not rewritten. A failed signature
        leaves the new entry on disk (the log cannot roll back) and raises.
        """
        if kind not in _KIND_SET:
            raise AuditError(f"unknown ledger kind {kind!r}")
        if not isinstance(payload, dict):
            raise AuditError("payload must be a JSON object")
        safe_payload = json_safe(payload)
        if not isinstance(safe_payload, dict):
            raise AuditError("payload must be a JSON object")
        if kind == "research_run":
            forbidden = _forbidden_research_keys(safe_payload)
            if forbidden:
                raise AuditError(
                    "research_run payload must not carry research-forbidden headline keys: "
                    + ", ".join(sorted(forbidden))
                )
        stamp = recorded_at if recorded_at is not None else self.clock()
        if not isinstance(stamp, str) or not stamp:
            raise AuditError("recorded_at must be a non-empty string")
        with self._lock():
            existing, errors = self._entries_unlocked()
            if errors:
                raise AuditError("refusing to append to a corrupt ledger", errors=errors)
            index = len(existing)
            prev = existing[-1].entry_hash if existing else GENESIS_HASH
            entry = LedgerEntry(
                index=index,
                kind=kind,
                recorded_at=stamp,
                payload=safe_payload,
                prev_hash=prev,
                entry_hash="",
            )
            digest = sha256_hex(entry.preimage())
            committed = LedgerEntry(
                index=index,
                kind=kind,
                recorded_at=stamp,
                payload=safe_payload,
                prev_hash=prev,
                entry_hash=digest,
            )
            self._append_line(self.entries_path, canonical_json_bytes(committed.to_obj()))
            if self.signer is not None and (index + 1) % self.sign_every == 0:
                self._checkpoint_unlocked()
            return committed

    def checkpoint(self) -> dict[str, Any]:
        """Sign the current full tree even if ``sign_every`` has not elapsed."""
        if self.signer is None:
            raise AuditError("cannot sign a tree head without a signer")
        with self._lock():
            return self._checkpoint_unlocked()

    def _checkpoint_unlocked(self) -> dict[str, Any]:
        entries, errors = self._entries_unlocked()
        if errors:
            raise AuditError("refusing to sign a corrupt ledger", errors=errors)
        if not entries:
            raise AuditError("refusing to sign an empty tree")
        signer = self.signer
        if signer is None:
            raise AuditError("cannot sign a tree head without a signer")
        root = merkle_root([entry.preimage() for entry in entries]).hex()
        signed = {
            "v": SCHEMA_VERSION,
            "scheme": str(signer.scheme),
            "key_id": str(signer.key_id),
            "tree_size": len(entries),
            "merkle_root": root,
            "timestamp_utc": self.clock(),
        }
        signature: Signature = signer.sign(canonical_json_bytes(signed))
        record: dict[str, Any] = {
            **signed,
            "signature": signature.value,
            "public_key": signature.public_key_hex,
            "sigstore_bundle_json": signature.sigstore_bundle_json,
            "sigstore_identity": signature.sigstore_identity,
            "sigstore_issuer": signature.sigstore_issuer,
        }
        self._append_line(self.checkpoints_path, canonical_json_bytes(record))
        if signer.scheme == "ed25519" and signature.public_key_hex:
            pub = self.root / "ed25519.pub"
            pub.write_bytes((signature.public_key_hex + "\n").encode("ascii"))
        return record

    def _append_line(self, path: Path, body: bytes) -> None:
        line = body + b"\n"
        fd = os.open(
            path, os.O_CREAT | os.O_APPEND | os.O_WRONLY | getattr(os, "O_BINARY", 0), 0o644
        )
        try:
            view = line
            while view:
                wrote = os.write(fd, view)
                view = view[wrote:]
            if self.sync:
                os.fsync(fd)
        finally:
            os.close(fd)

    def _entries_unlocked(self) -> tuple[list[LedgerEntry], list[str]]:
        return read_entries(self.entries_path)

    def entries(self) -> list[LedgerEntry]:
        with self._lock():
            entries, errors = self._entries_unlocked()
        if errors:
            raise AuditError("ledger failed verification while reading", errors=errors)
        return entries

    def leaf_preimages(self) -> list[bytes]:
        return [entry.preimage() for entry in self.entries()]

    def inclusion_proof(self, index: int) -> list[bytes]:
        from quant_fund.audit.merkle import inclusion_proof

        return inclusion_proof(self.leaf_preimages(), index)

    def consistency_proof(self, old_size: int) -> list[bytes]:
        from quant_fund.audit.merkle import consistency_proof

        return consistency_proof(self.leaf_preimages(), old_size)


def read_entries(path: Path) -> tuple[list[LedgerEntry], list[str]]:
    """Read entries and check the hash chain. Does not check signatures."""
    objects, errors = load_jsonl(path)
    entries: list[LedgerEntry] = []
    for position, obj in enumerate(objects):
        entry, error = entry_from_obj(obj)
        if entry is None:
            errors.append(f"{error}:{position}")
            continue
        if error:
            errors.append(f"{error}:{entry.index}")
        if entry.index != position:
            errors.append(f"index_gap:{position}")
        expected_prev = entries[-1].entry_hash if entries else GENESIS_HASH
        if entry.prev_hash != expected_prev:
            errors.append(f"prev_hash_mismatch:{entry.index}")
        entries.append(entry)
    return entries, errors

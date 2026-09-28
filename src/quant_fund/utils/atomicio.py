"""Crash-safe durable writes: tmp file + fsync + atomic rename.

Every durable artifact (receipts, parquet stores, ledgers) must go through
these helpers so a crash or interrupt mid-write can never leave a truncated
file at the canonical path for a later reader to consume.

Two contracts:

- :func:`atomic_write_*` — replace semantics: the newest complete write wins.
- :func:`publish_text_once` — immutable-publish semantics for sealed evidence:
  an existing path with identical content is a no-op; different content raises.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import polars as pl


def _fsync_directory(path: Path) -> None:
    """Make an atomic replacement visible after a host crash when supported."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path | str, data: bytes) -> Path:
    """Write ``data`` to ``path`` atomically, replacing any existing file."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{dest.name}.", dir=dest.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, dest)
        _fsync_directory(dest.parent)
    finally:
        temporary_path.unlink(missing_ok=True)
    return dest


def atomic_write_text(path: Path | str, text: str) -> Path:
    """UTF-8 text counterpart of :func:`atomic_write_bytes`."""
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_parquet(frame: pl.DataFrame, path: Path | str) -> Path:
    """Write a parquet artifact atomically — readers never see a torn file."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".parquet", dir=dest.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            frame.write_parquet(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, dest)
        _fsync_directory(dest.parent)
    finally:
        temporary_path.unlink(missing_ok=True)
    return dest


def publish_text_once(path: Path, content: str) -> None:
    """Publish a complete immutable text artifact without replacing an existing one.

    Re-publishing identical content is a no-op; differing content at an
    existing path (or a symlink) raises ``FileExistsError`` — sealed evidence
    must never be silently swapped.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FileExistsError(f"receipt path is a symlink: {path}")
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"receipt already exists with different content: {path}")
        return
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.is_symlink() or path.read_text(encoding="utf-8") != content:
                raise FileExistsError(
                    f"receipt already exists with different content: {path}"
                ) from None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

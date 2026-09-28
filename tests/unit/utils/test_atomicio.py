"""Crash-safe durable writes: atomic replace, publish-once, no tmp litter."""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest

from quant_fund.utils.atomicio import (
    atomic_write_bytes,
    atomic_write_parquet,
    atomic_write_text,
    publish_text_once,
)


def test_atomic_write_text_replaces_and_leaves_no_litter(tmp_path: Path) -> None:
    dest = tmp_path / "receipt.json"
    atomic_write_text(dest, '{"a": 1}\n')
    assert dest.read_text() == '{"a": 1}\n'
    atomic_write_text(dest, '{"a": 2}\n')
    assert dest.read_text() == '{"a": 2}\n'
    assert list(tmp_path.iterdir()) == [dest]


def test_atomic_write_bytes_roundtrip(tmp_path: Path) -> None:
    dest = tmp_path / "nested" / "blob.bin"
    payload = b"\x00\xff" * 512
    atomic_write_bytes(dest, payload)
    assert dest.read_bytes() == payload


def test_atomic_write_parquet_roundtrip(tmp_path: Path) -> None:
    dest = tmp_path / "frame.parquet"
    frame = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    atomic_write_parquet(frame, dest)
    assert pl.read_parquet(dest).equals(frame)


def test_atomic_write_failure_leaves_prior_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write that dies mid-flight must leave the prior artifact untouched."""
    dest = tmp_path / "kept.bin"
    dest.write_bytes(b"prior-good")

    def boom(_fd: int, _flags: int = 0) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "fdopen", boom)
    with pytest.raises(KeyboardInterrupt):
        atomic_write_bytes(dest, b"new")
    assert dest.read_bytes() == b"prior-good"
    assert list(tmp_path.iterdir()) == [dest]


def test_publish_text_once_idempotent_and_fail_closed(tmp_path: Path) -> None:
    dest = tmp_path / "sealed.json"
    publish_text_once(dest, '{"sealed": true}\n')
    publish_text_once(dest, '{"sealed": true}\n')  # identical re-publish is a no-op
    with pytest.raises(FileExistsError, match="different content"):
        publish_text_once(dest, '{"sealed": false}\n')
    assert dest.read_text() == '{"sealed": true}\n'
    assert list(tmp_path.iterdir()) == [dest]


def test_publish_text_once_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "real.json"
    target.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(FileExistsError, match="symlink"):
        publish_text_once(link, "{}")


def test_atomic_write_creates_parents(tmp_path: Path) -> None:
    dest = tmp_path / "deep" / "nested" / "out.txt"
    atomic_write_text(dest, "x")
    assert dest.read_text() == "x"

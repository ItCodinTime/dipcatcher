"""Parquet lake I/O."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import polars as pl

from quant_fund.config.models import AppConfig


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Lake:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        for part in ("raw", "bronze", "silver", "gold", "metadata"):
            (self.root / part).mkdir(parents=True, exist_ok=True)

    def _resolve(self, rel: str) -> Path:
        # Lake-relative paths only: no absolute paths, no traversal — a bad
        # caller must not be able to write outside the lake root.
        parts = Path(rel).parts
        if (
            not rel
            or Path(rel).is_absolute()
            or "\\" in rel
            or any(part in (".", "..") for part in parts)
        ):
            raise ValueError(f"lake path must be a clean relative path, got {rel!r}")
        return self.root / rel

    def write_parquet(self, frame: pl.DataFrame, rel: str) -> Path:
        # Unique tmp + fsync + os.replace + dir fsync: a crash mid-write can
        # never leave a truncated or unflushed parquet at the canonical path,
        # and concurrent writers to the same path cannot share a tmp file.
        path = self._resolve(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            frame.write_parquet(tmp)
            fd = os.open(tmp, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, path)
            _fsync_dir(path.parent)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return path

    def read_parquet(self, rel: str, columns: list[str] | None = None) -> pl.DataFrame:
        """Memory-map a lake parquet file.

        ``columns`` projects before the frame is materialized. Omit it to read
        every stored column, which is what existing callers do.
        """
        return pl.read_parquet(self._resolve(rel), columns=columns, memory_map=True)

    def exists(self, rel: str) -> bool:
        return self._resolve(rel).exists()


def lake_from_config(config: AppConfig) -> Lake:
    return Lake(Path(config.data.root))

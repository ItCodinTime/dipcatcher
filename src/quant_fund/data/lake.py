"""Parquet lake I/O."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from quant_fund.config.models import AppConfig


class Lake:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        for part in ("raw", "bronze", "silver", "gold", "metadata"):
            (self.root / part).mkdir(parents=True, exist_ok=True)

    def write_parquet(self, frame: pl.DataFrame, rel: str) -> Path:
        # Write-then-rename: a crash mid-write must not leave a truncated
        # parquet at the canonical path, so readers never see a torn file.
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        try:
            frame.write_parquet(tmp)
            tmp.replace(path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return path

    def read_parquet(self, rel: str, columns: list[str] | None = None) -> pl.DataFrame:
        """Memory-map a lake parquet file.

        ``columns`` projects before the frame is materialized. Omit it to read
        every stored column, which is what existing callers do.
        """
        return pl.read_parquet(self.root / rel, columns=columns, memory_map=True)

    def exists(self, rel: str) -> bool:
        return (self.root / rel).exists()


def lake_from_config(config: AppConfig) -> Lake:
    return Lake(Path(config.data.root))

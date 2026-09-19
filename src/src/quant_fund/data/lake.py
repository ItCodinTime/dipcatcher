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
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
        return path

    def read_parquet(self, rel: str) -> pl.DataFrame:
        return pl.read_parquet(self.root / rel)

    def exists(self, rel: str) -> bool:
        return (self.root / rel).exists()


def lake_from_config(config: AppConfig) -> Lake:
    return Lake(Path(config.data.root))

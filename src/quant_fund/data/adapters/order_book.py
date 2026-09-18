"""Order-book providers for Northset research.

Network/vendor pulls stay out of unit tests. Parquet fixtures and SYNTHETIC
panels share one interchange schema (ADR-021).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import polars as pl

from quant_fund.microstructure.book_panel import load_book_panel, validate_book_panel
from quant_fund.microstructure.synthetic_lob import synthesize_l2_from_bars


class OrderBookProvider(Protocol):
    """Research L2 panel provider. Returns flat book metrics, not nested levels."""

    def get_book_panel(self, bars: pl.DataFrame | None = None) -> pl.DataFrame: ...


class SyntheticOrderBookProvider:
    """Build SYNTHETIC L2 from OHLCV bars (plumbing substrate)."""

    def __init__(
        self,
        *,
        depth: int = 5,
        seed: int = 7,
        base_spread_bps: float = 4.0,
    ) -> None:
        self.depth = int(depth)
        self.seed = int(seed)
        self.base_spread_bps = float(base_spread_bps)

    def get_book_panel(self, bars: pl.DataFrame | None = None) -> pl.DataFrame:
        if bars is None or bars.height == 0:
            raise ValueError("SyntheticOrderBookProvider requires non-empty bars")
        return validate_book_panel(
            synthesize_l2_from_bars(
                bars,
                depth=self.depth,
                seed=self.seed,
                base_spread_bps=self.base_spread_bps,
            )
        )


class ParquetOrderBookProvider:
    """Load a vendor-shaped book panel from parquet (offline fixture / lake)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def get_book_panel(self, bars: pl.DataFrame | None = None) -> pl.DataFrame:
        del bars  # join happens upstream against bars
        if not self.path.is_file():
            raise FileNotFoundError(f"order-book panel parquet not found: {self.path}")
        return load_book_panel(self.path)

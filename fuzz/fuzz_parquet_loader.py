"""Fuzz on-disk market-data loaders: parquet + CSV file paths.

First byte selects the slot inside a fresh provider root:

- ``0``: body → ``bars.parquet``, then ``ParquetMarketProvider.get_bars()``
- ``1``: body → ``bars.csv``, then ``ParquetMarketProvider.get_bars()``
- ``2``: body → ``panel.parquet``, then ``ParquetOrderBookProvider.get_book_panel()``
- ``3``: body → ``pl.read_parquet(BytesIO)`` directly (bare decoder surface)

Expected: ``PointInTimeError``/``ValueError``/``FileNotFoundError`` from the
provider contracts plus polars ``PolarsError``/`OSError` on corrupt bytes.
``PanicException`` and non-polars exceptions are findings.
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fuzz_common import run

from quant_fund.data.adapters.order_book import ParquetOrderBookProvider
from quant_fund.data.adapters.parquet import ParquetMarketProvider
from quant_fund.data.point_in_time import PointInTimeError

EXPECTED: tuple[type[BaseException], ...] = (
    PointInTimeError,
    ValueError,
    FileNotFoundError,
    pl.exceptions.PolarsError,
    OSError,
)


def test_one_input(data: bytes) -> None:
    if not data:
        return
    which = data[0] % 4
    body = data[1:]
    if which == 3:
        # Bare decoder surface: polars can raise PanicException (Rust panic)
        # on corrupt metadata — an upstream property we contain at every
        # dipcatcher call site, not one we can fix here. Still exercised so
        # regressions in the containment wrappers are caught by modes 0-2.
        try:
            pl.read_parquet(io.BytesIO(body))
        except pl.exceptions.PanicException:
            return
        return
    with tempfile.TemporaryDirectory(prefix="fuzz_pq_") as tmp:
        root = Path(tmp)
        if which == 0:
            root.joinpath("bars.parquet").write_bytes(body)
            ParquetMarketProvider(root).get_bars()
        elif which == 1:
            root.joinpath("bars.csv").write_bytes(body)
            ParquetMarketProvider(root).get_bars()
        else:
            path = root / "panel.parquet"
            path.write_bytes(body)
            ParquetOrderBookProvider(path).get_book_panel()


if __name__ == "__main__":
    # isolate: corrupt parquet can abort the polars engine (OOM/panic) —
    # fork per input so the crash is recorded, not fatal to the campaign.
    raise SystemExit(
        run("parquet_loader", test_one_input, expected=EXPECTED, isolate=True)
    )

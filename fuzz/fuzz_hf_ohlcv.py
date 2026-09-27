"""Fuzz the HF OHLCV-1m adapter: arg parsers + cached-month load path.

First byte selects the sub-target:

- ``0``: ``parse_symbols(text)``
- ``1``: ``normalize_interval(text)``
- ``2``: ``parse_bound(text, role=...)``
- ``3``: ``month_parquet_url(year, month)`` derived from body bytes
- ``4``: ``read_ohlcv_1m`` with an injected ``fetcher`` that writes the fuzz
  bytes as the requested month parquet — exercises ``_require_vendor_parquet``,
  ``_scan_vendor``, ``normalize_vendor_frame`` end to end.

Expected: ``ValueError``/``OhlcvQualityError`` (+ polars errors surfaced
through them). Anything else is a finding.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fuzz_common import run

from quant_fund.data.adapters.hf_ohlcv_1m import (
    OhlcvQualityError,
    month_parquet_url,
    normalize_interval,
    parse_bound,
    parse_symbols,
    read_ohlcv_1m,
)

EXPECTED: tuple[type[BaseException], ...] = (
    ValueError,
    OhlcvQualityError,
    pl.exceptions.PolarsError,
    OSError,
)


def test_one_input(data: bytes) -> None:
    if not data:
        return
    which = data[0] % 5
    body = data[1:]
    if which == 0:
        parse_symbols(body.decode("utf-8", errors="replace"))
    elif which == 1:
        normalize_interval(body.decode("utf-8", errors="replace"))
    elif which == 2:
        parse_bound(
            body.decode("utf-8", errors="replace"),
            role="start" if len(body) % 2 else "end",
        )
    elif which == 3:
        year = 1990 + (body[0] % 60) if body else 2024
        month = (body[1] % 16) if len(body) > 1 else 1
        month_parquet_url(year, month)
    else:
        with tempfile.TemporaryDirectory(prefix="fuzz_hf_") as tmp:
            cache = Path(tmp)

            def fetcher(_url: str, dest: Path) -> None:
                dest.write_bytes(body)

            read_ohlcv_1m(
                symbols="AAA",
                cache_dir=cache,
                start="2024-01-02",
                end="2024-01-05",
                allow_download=True,
                fetcher=fetcher,
                max_months=2,
            )


if __name__ == "__main__":
    # isolate: corrupt parquet can abort the polars engine (OOM/panic) —
    # fork per input so the crash is recorded, not fatal to the campaign.
    raise SystemExit(run("hf_ohlcv", test_one_input, expected=EXPECTED, isolate=True))

"""Fuzz ``parse_stooq_csv``: Stooq ``Date,Open,High,Low,Close,Volume`` payloads.

The parser is the tolerant path of the daily file tape (bad rows are skipped,
never fatal), so *any* exception escaping it is a finding — including
``csv.Error`` on oversized fields and polars schema surprises.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fuzz_common import run

from quant_fund.data.adapters.stooq import parse_stooq_csv

EXPECTED: tuple[type[BaseException], ...] = ()


def test_one_input(data: bytes) -> None:
    text = data.decode("utf-8", errors="replace")
    parse_stooq_csv(text, security_id="SPY", stooq_symbol="spy.us")


if __name__ == "__main__":
    raise SystemExit(run("stooq_csv", test_one_input, expected=EXPECTED))

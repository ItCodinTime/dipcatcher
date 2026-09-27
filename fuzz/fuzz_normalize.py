"""Fuzz the source-normalization helpers: ``csv_rows``, ``normalize_ohlcv``,
``normalize_observations``, ``parse_time``, ``pit_frame``.

First byte selects the sub-target:

- ``0``: ``csv_rows(text)`` — documented channel is ``SourceError``; a bare
  ``csv.Error`` (oversized field) escaping is a finding.
- ``1``: ``normalize_ohlcv(json_rows)`` — ``SourceError`` is the contract;
  anything else (``TypeError``, ``OverflowError`` from ``parse_time``,
  polars errors) is a finding.
- ``2``: ``parse_time(value)`` — fed ``str``/``int``/``float``/``null`` JSON
  scalars; ``ValueError`` is tolerated (ISO parse), ``OverflowError`` and
  friends are findings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fuzz_common import run

from quant_fund.data.sources.base import SourceError, parse_time
from quant_fund.data.sources.normalize import csv_rows, normalize_ohlcv


def _decode(data: bytes) -> object | None:
    try:
        return json.loads(data.decode("utf-8", errors="strict"))
    except ValueError:
        return None


def test_one_input(data: bytes) -> None:
    if not data:
        return
    which = data[0] % 3
    rest = data[1:]
    if which == 0:
        text = rest.decode("utf-8", errors="replace")
        try:
            csv_rows(text)
        except SourceError:
            return
        return
    payload = _decode(rest)
    if payload is None:
        return
    if which == 1:
        rows = payload if isinstance(payload, list) else [payload]
        if not all(isinstance(row, dict) for row in rows):
            return
        try:
            normalize_ohlcv(rows, source="fuzz", revision_id="FUZZ")
        except SourceError:
            return
    elif which == 2:
        if isinstance(payload, dict | list):
            return
        try:
            parse_time(payload)
        except ValueError:
            return


if __name__ == "__main__":
    raise SystemExit(run("normalize", test_one_input, expected=()))

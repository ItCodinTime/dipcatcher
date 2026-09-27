"""Fuzz ``parse_yahoo_chart``: Yahoo v8 chart JSON payloads.

``payload`` comes straight from ``json.loads`` of a vendor HTTP body, so the
harness decodes fuzz bytes as JSON first (decode failures are not findings).
Any exception escaping ``parse_yahoo_chart`` — including ``TypeError``/
``KeyError`` on non-conforming ``quote``/``timestamp`` members and
``OverflowError``/``ValueError`` from ``fromtimestamp`` — is a crash.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fuzz_common import run

from quant_fund.data.adapters.yahoo_eod import parse_yahoo_chart

EXPECTED: tuple[type[BaseException], ...] = ()


def test_one_input(data: bytes) -> None:
    try:
        payload = json.loads(data.decode("utf-8", errors="strict"))
    except ValueError:
        return
    parse_yahoo_chart(payload, security_id="SPY", yahoo_symbol="SPY")


if __name__ == "__main__":
    raise SystemExit(run("yahoo_chart", test_one_input, expected=EXPECTED))

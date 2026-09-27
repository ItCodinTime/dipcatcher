"""Fuzz research receipt readers: ``verify_research_artifact`` +
``validate_analytics_export``.

First byte selects the target:

- ``0``: write body to ``receipt.json`` in a temp dir and run
  ``verify_research_artifact`` — the public ``verify-research`` entry. It must
  return a report dict for any content; an exception is a finding.
  (``research/verify.py`` is covered by the Phase-1 seal: findings here are
  documented, not fixed, per the swarm contract.)
- ``1``: decode JSON and feed ``validate_analytics_export`` — the
  paper/backtest export validator. Designed to never raise.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fuzz_common import run

from quant_fund.metrics.analytics import validate_analytics_export
from quant_fund.research.verify import verify_research_artifact

EXPECTED: tuple[type[BaseException], ...] = ()


def test_one_input(data: bytes) -> None:
    if not data:
        return
    which = data[0] % 2
    body = data[1:]
    if which == 0:
        with tempfile.TemporaryDirectory(prefix="fuzz_rcpt_") as tmp:
            path = Path(tmp) / "receipt.json"
            path.write_bytes(body)
            verify_research_artifact(path)
    else:
        try:
            blob = json.loads(body.decode("utf-8", errors="strict"))
        except ValueError:
            return
        validate_analytics_export(blob if isinstance(blob, dict) else {"x": blob})


if __name__ == "__main__":
    raise SystemExit(run("receipt", test_one_input, expected=EXPECTED))

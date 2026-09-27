"""Standalone receipt verification CLI.

    uv run --no-sync python -m quant_fund.research.replication.verify \
        receipts/paper_replication_<hash>.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from quant_fund.research.replication.receipt import verify_replication_receipt


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m quant_fund.research.replication.verify <receipt.json>")
        return 2
    result = verify_replication_receipt(Path(argv[0]))
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())

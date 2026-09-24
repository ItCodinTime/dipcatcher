"""Verify a preserved external-data receipt without making a performance claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_fund.research.external_receipt import verify_receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--base-dir", type=Path, default=None)
    args = parser.parse_args()
    print(
        json.dumps(verify_receipt(args.receipt, base_dir=args.base_dir), indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()

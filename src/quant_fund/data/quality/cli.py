"""Minimal CLI: ``python -m quant_fund.data.quality.cli bars.parquet [--out r.json]``.

Read-only research tooling; exits 1 when the dataset fails any check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from quant_fund.data.quality import report_json, score_bars


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="data-quality")
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = score_bars(pl.read_parquet(args.parquet), dataset=str(args.parquet))
    text = report_json(report)
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

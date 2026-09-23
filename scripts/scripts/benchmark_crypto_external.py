#!/usr/bin/env python3
"""Verify a preserved Binance candidate receipt without claiming benchmark proof."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from quant_fund.data.adapters.binance import BINANCE_SYMBOLS
from quant_fund.research.binance_receipt import verify_binance_receipt


def _build_report(receipt_path: Path, base_dir: Path | None) -> dict[str, Any]:
    """Build a deterministic receipt report after independent verification."""
    receipt_bytes = receipt_path.read_bytes()
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    verified = verify_binance_receipt(receipt_path, base_dir=base_dir)
    symbols = sorted(str(symbol) for symbol in verified.get("symbols", []))
    if (
        verified.get("archives_verified") != len(BINANCE_SYMBOLS)
        or len(symbols) != len(BINANCE_SYMBOLS)
        or len(set(symbols)) != len(BINANCE_SYMBOLS)
        or set(symbols) != set(BINANCE_SYMBOLS)
    ):
        raise ValueError("Binance receipt must verify exactly the five canonical symbols")

    payload = json.loads(receipt_bytes.decode("utf-8"))
    archives = payload.get("archives")
    if not isinstance(archives, list) or len(archives) != len(BINANCE_SYMBOLS):
        raise ValueError("Binance receipt must contain exactly five archives")
    dates = sorted({str(archive.get("date")) for archive in archives if isinstance(archive, dict)})
    if len(dates) != 1:
        raise ValueError("Binance receipt archives must share one UTC date")

    return {
        "ok": True,
        "source": verified["source"],
        "receipt": str(receipt_path),
        "receipt_sha256": receipt_sha256,
        "candidate_only": True,
        "proof_status": "not_proof",
        "proof_eligible": False,
        "sota_proven": False,
        "archives_verified": len(BINANCE_SYMBOLS),
        "symbols": symbols,
        "rows": verified["rows"],
        "protocol": {
            "input_id": receipt_path.stem,
            "input_sha256": receipt_sha256,
            "symbols": symbols,
            "interval": "1d",
            "timezone": "UTC",
            "availability_policy": "next-day UTC",
            "horizon": 5,
            "n_dates": 1,
            "candidate_only": True,
            "proof_status": "not_proof",
            "proof_eligible": False,
            "sota_proven": False,
        },
        "benchmark_status": "not_run",
        "limitations": [
            "the preserved receipt contains one common-date archive per symbol",
            "a single-date receipt is insufficient for a multi-origin external benchmark",
            "Binance current archives do not preserve historical release vintages",
            "this offline verification report is candidate-only and not industry-grade or SOTA proof",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipt_path = args.receipt.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("refusing to overwrite an existing external benchmark report")

    report = _build_report(receipt_path, args.base_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(rendered)
        stream.write("\n")
    print(rendered)


if __name__ == "__main__":
    main()

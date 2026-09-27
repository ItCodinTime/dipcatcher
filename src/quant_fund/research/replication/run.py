"""Run the paper-replication suite and write a sealed receipt.

Usage:

    uv run --no-sync python -m quant_fund.research.replication \
        --data /path/to/silver/bars.parquet --out-dir receipts

    # fully synthetic labeled fixture (no parquet needed):
    uv run --no-sync python -m quant_fund.research.replication --synthetic

Headline evidence is proper scores + rank-IC statistics only; any descriptive
performance statistic is labeled ``descriptive_*`` inside the receipt.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_fund.research.replication.panel import load_silver_bars, market_returns
from quant_fund.research.replication.receipt import REPLICATION_SCHEMA, write_receipt
from quant_fund.research.replication.strategies import (
    CITATIONS,
    STRATEGY_SPECS,
    evaluate_strategy,
)
from quant_fund.research.replication.synthetic import synthetic_panel
from quant_fund.utils.hashing import (
    canonical_frame_fingerprint,
    canonical_json_bytes,
    hash_bytes,
    hash_file,
)
from quant_fund.utils.reproducibility import git_revision

DISCLAIMER = (
    "Research replication of published equity anomalies on the labeled "
    "SYNTHETIC panel (or synthetic fixtures). Results are proper scores and "
    "rank-IC diagnostics only; descriptive stats are secondary and labeled. "
    "Not live performance, not investment advice, authorizes nothing."
)

DATA_LABEL = "SYNTHETIC"


def _data_provenance(panel: pl.DataFrame, source: dict[str, Any]) -> dict[str, Any]:
    times = panel["event_time"]
    return {
        **source,
        "rows": int(panel.height),
        "assets": int(panel["security_id"].n_unique()),
        "start": str(times.min()),
        "end": str(times.max()),
    }


def run_replication(
    *,
    data: Path | None = None,
    use_synthetic: bool = False,
    synthetic_seed: int = 11,
    synthetic_kwargs: dict[str, Any] | None = None,
    out_dir: Path | str | None = Path("receipts"),
) -> tuple[dict[str, Any], Path | None]:
    """Execute all strategy specs; return (receipt_payload, written_path)."""
    if use_synthetic or data is None:
        kwargs = {"n_assets": 35, "n_days": 504, "seed": synthetic_seed}
        kwargs.update(synthetic_kwargs or {})
        panel = synthetic_panel(**kwargs)
        source = {
            "kind": "generated_synthetic_fixture",
            "generator": "replication.synthetic.synthetic_panel",
            "generator_params": kwargs,
            "sha256": canonical_frame_fingerprint(panel),
        }
    else:
        data = Path(data)
        panel = load_silver_bars(data)
        source = {
            "kind": "lab_derived_synthetic_panel",
            "path": str(data),
            "sha256": hash_file(data),
            "note": (
                "Lab-derived derived bars. Symbols S####/MKT + planted_signal "
                "column indicate the dataset is itself synthetic; all results "
                "are SYNTHETIC research evidence, not real-market evidence."
            ),
        }
    market = market_returns(panel)
    strategies: dict[str, Any] = {}
    for spec in STRATEGY_SPECS:
        strategies[spec.key] = evaluate_strategy(panel, market, spec)
    inputs_sha256 = hash_bytes(
        canonical_json_bytes(
            {
                "source": source,
                "specs": {
                    spec.key: {
                        "signal_fn": spec.signal_fn,
                        "params": spec.params,
                        "outcome_col": spec.outcome_col,
                        "primary_horizon": spec.primary_horizon,
                    }
                    for spec in STRATEGY_SPECS
                },
            }
        )
    )
    receipt: dict[str, Any] = {
        "schema": REPLICATION_SCHEMA,
        "kind": "equity_anomaly_replication",
        "data_label": DATA_LABEL,
        "synthetic": True,
        "live_pnl_claim": False,
        "research_only": True,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        "data": _data_provenance(panel, source),
        "inputs_sha256": inputs_sha256,
        "strategies": strategies,
        "citations": sorted(CITATIONS.values()),
        "limitations": [
            "35 synthetic assets x ~504 daily bars: small cross-section and "
            "short history vs the original papers (CRSP decades).",
            "No transaction costs, short constraints, or market impact "
            "modeled; decile spreads are descriptive only.",
            "Synthetic DGP has planted_signal unrelated to these anomalies; "
            "null/contradicted verdicts are expected and reported honestly.",
            "Multiple-testing: five strategies each report one primary "
            "statistic; treat borderline |t|~2 as weak evidence.",
        ],
        "disclaimer": DISCLAIMER,
    }
    path = write_receipt(receipt, out_dir) if out_dir is not None else None
    return receipt, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="replication", description=__doc__)
    parser.add_argument("--data", type=Path, default=None, help="silver bars parquet")
    parser.add_argument(
        "--synthetic", action="store_true", help="use the labeled synthetic fixture"
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out-dir", type=Path, default=Path("receipts"))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    receipt, path = run_replication(
        data=args.data,
        use_synthetic=args.synthetic,
        synthetic_seed=args.seed,
        out_dir=None if args.no_write else args.out_dir,
    )
    summary = {
        spec: {
            "verdict": block["verdict"]["verdict"],
            "primary_statistic_value": block["verdict"]["primary_statistic_value"],
            "direction_brier": block["scores"].get("direction_brier"),
            "direction_brier_climatology": block["scores"].get("direction_brier_climatology"),
        }
        for spec, block in receipt["strategies"].items()
    }
    print(
        json.dumps(
            {
                "receipt_path": str(path),
                "data_kind": receipt["data"]["kind"],
                "strategies": summary,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

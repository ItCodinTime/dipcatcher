"""Proof runner entry point, closed until decision-time PIT reads exist.

A whole-panel read at a far-future as-of timestamp can select revisions that
were unavailable to historical decisions. Bundle construction and independent
hash/metric verification remain available separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from quant_fund.config.models import AppConfig
from quant_fund.proof.recorder import InMemoryRecorder
from quant_fund.proof.sign import Signer
from quant_fund.proofcore.contracts import ProofBundleV1, ProofError

__all__ = ["BARS_DATASET", "WEIGHTS_DATASET", "run_backtest_proven"]

BARS_DATASET = "silver/bars"
WEIGHTS_DATASET = "gold/weights"


def run_backtest_proven(
    config: AppConfig,
    *,
    seed: int,
    pit_root: Path,
    bundle_dir: Path,
    replay_engine: Literal["reference", "fast"] = "reference",
    signer: Signer | None = None,
    vault: object | None = None,
    recorder: InMemoryRecorder | None = None,
) -> ProofBundleV1:
    """Reject runs until explicit decision times drive each vault as-of read."""
    raise ProofError(
        "proven run unavailable: explicit per-decision as-of vault reads are not implemented"
    )

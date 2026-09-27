"""Proven backtest orchestrator (DESIGN.md §5.2).

``run_backtest_proven`` wraps the existing ``backtest/engine.py`` API — the
engine signature is untouched; this is a separate orchestrator that the CLI
calls. Data enters only through the PIT vault ``asof()`` choke point; every
read is recorded into an ``InMemoryRecorder`` and lands in the bundle's
``data_manifest``.

Determinism note (A3 #6): the panel reads use ``ASOF_SENTINEL`` ("latest
known"), never wall-clock time, so the recorded ``asof_utc`` and the Merkle
root are stable across identical re-runs. Fine-grained per-decision asof
discipline is the vault/watchdog layer's job (W1/W3).

Test seam: ``vault``/``recorder`` keyword-only parameters let tests inject an
in-memory fake implementing the §4.3 ``PitVault.asof`` API while W1 lands in
parallel. Production callers leave both None and get the real vault.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.proof.bundle import build_bundle
from quant_fund.proof.recorder import InMemoryRecorder
from quant_fund.proof.sign import Signer
from quant_fund.proofcore.contracts import ProofBundleV1, ProofError

__all__ = [
    "ASOF_SENTINEL",
    "BARS_DATASET",
    "WEIGHTS_DATASET",
    "PitFrameLike",
    "PitVaultLike",
    "run_backtest_proven",
]

#: Vault-relative dataset names consumed by the proven backtest runner.
BARS_DATASET = "silver/bars"
WEIGHTS_DATASET = "gold/weights"

#: Deterministic "latest known" as-of for panel reads (never wall clock).
ASOF_SENTINEL = datetime(9999, 12, 31, 23, 59, 59, tzinfo=UTC)


class PitFrameLike(Protocol):
    """The §4.3 PitFrame surface the runner consumes."""

    frame: pl.DataFrame
    dataset: str
    asof: datetime
    max_known_at: datetime
    rows: int
    content_sha256: str


class PitVaultLike(Protocol):
    """The §4.3 PitVault read surface the runner consumes."""

    def asof(self, name: str, t: datetime, **kwargs: Any) -> PitFrameLike: ...


def _open_vault(pit_root: Path, recorder: InMemoryRecorder) -> PitVaultLike:
    """Open the real W1 vault with recorder (+ W3 watchdog when available)."""
    from quant_fund.pit import PitVault  # lazy: W1 lands in parallel

    watchdog: Any = None
    try:
        from quant_fund.leakage import LeakageWatchdog  # lazy: W3 lands in parallel

        watchdog = LeakageWatchdog()
    except ImportError:
        watchdog = None
    return PitVault(Path(pit_root), recorder=recorder, watchdog=watchdog)


def _engine_fn(replay_engine: Literal["reference", "fast"]) -> Any:
    if replay_engine == "reference":
        from quant_fund.backtest.engine import run_backtest

        return run_backtest
    if replay_engine == "fast":
        from quant_fund.backtest.fast_replay import run_backtest_fast

        return run_backtest_fast
    raise ProofError(f"unknown replay_engine: {replay_engine!r}")


def _signal_log_frame(weights: pl.DataFrame) -> pl.DataFrame:
    """Per-decision rows (decision_time, security_id, score, weight) — §5.3."""
    if weights.height == 0:
        return pl.DataFrame(
            schema={
                "decision_time": pl.Datetime("us", "UTC"),
                "security_id": pl.Utf8,
                "score": pl.Float64,
                "weight": pl.Float64,
            }
        )
    score_col = "score" if "score" in weights.columns else "target_weight"
    return weights.select(
        pl.col("event_time").alias("decision_time"),
        pl.col("security_id"),
        pl.col(score_col).cast(pl.Float64).alias("score"),
        pl.col("target_weight").cast(pl.Float64).alias("weight"),
    ).sort("decision_time", "security_id")


def _trade_log_frame(fills: pl.DataFrame, equity: pl.DataFrame) -> pl.DataFrame:
    """Fills frame + ``nav`` = last mark-to-market NAV at or before the fill."""
    base_cols = {
        "fill_time": pl.Datetime("us", "UTC"),
        "signal_time": pl.Datetime("us", "UTC"),
        "security_id": pl.Utf8,
        "quantity": pl.Float64,
        "price": pl.Float64,
        "fee": pl.Float64,
        "spread_cost": pl.Float64,
        "impact_cost": pl.Float64,
        "decision_price": pl.Float64,
        "nav": pl.Float64,
    }
    if fills.height == 0:
        return pl.DataFrame(schema=base_cols)
    trades = fills
    if equity.height >= 1 and {"event_time", "nav"} <= set(equity.columns):
        marks = equity.select(
            pl.col("event_time").cast(pl.Datetime("us", "UTC")),
            pl.col("nav").cast(pl.Float64),
        ).sort("event_time")
        trades = trades.with_columns(pl.col("fill_time").cast(pl.Datetime("us", "UTC"))).sort(
            "fill_time"
        )
        trades = trades.join_asof(marks, left_on="fill_time", right_on="event_time").drop(
            "event_time"
        )
    else:
        trades = trades.with_columns(pl.lit(None, dtype=pl.Float64).alias("nav"))
    return trades


def run_backtest_proven(
    config: AppConfig,
    *,
    seed: int,
    pit_root: Path,
    bundle_dir: Path,
    replay_engine: Literal["reference", "fast"] = "reference",
    signer: Signer | None = None,
    vault: PitVaultLike | None = None,
    recorder: InMemoryRecorder | None = None,
) -> ProofBundleV1:
    """Run one proven backtest and mint its signed, hash-chained proof bundle.

    1. set_global_seed(seed)  (utils/seeds.py)
    2. open PitVault(pit_root, recorder=InMemoryRecorder(), watchdog=LeakageWatchdog())
    3. run backtest/engine.py run_backtest via the existing API on vault frames
    4. recompute headline metrics from the trade log (metrics/returns.py)
    5. build ProofBundleV1, chain to the prev bundle head in bundle_dir
    6. sign (HMAC) if signer given, else scheme='none'
    7. append bundle to bundle_dir/bundles.jsonl + bundles/<id>.json
    8. (provenance DB insert is W5's optional wiring, not this function)
    """
    from quant_fund.utils.seeds import set_global_seed

    if seed < 0:
        raise ProofError(f"seed must be >= 0, got {seed}")
    set_global_seed(seed)

    if recorder is None and vault is not None:
        # An injected vault (e.g. the §4.3 fake) may already carry a recorder;
        # reuse it so the bundle's data_manifest reflects the actual reads.
        attached = getattr(vault, "recorder", None)
        if isinstance(attached, InMemoryRecorder):
            recorder = attached
    recorder = recorder if recorder is not None else InMemoryRecorder()
    if vault is None:
        vault = _open_vault(Path(pit_root), recorder)

    bars_read = vault.asof(BARS_DATASET, ASOF_SENTINEL)
    weights_read = vault.asof(WEIGHTS_DATASET, ASOF_SENTINEL)

    result = _engine_fn(replay_engine)(bars_read.frame, weights_read.frame, config)

    return build_bundle(
        run_kind="backtest",
        data_manifest=recorder.manifest_summary(),
        config_dump=config.dump(),
        seed=seed,
        signal_log=_signal_log_frame(weights_read.frame),
        trade_log=_trade_log_frame(result.fills, result.equity),
        engine_metrics=dict(result.metrics),
        bundle_dir=Path(bundle_dir),
        signer=signer,
    )

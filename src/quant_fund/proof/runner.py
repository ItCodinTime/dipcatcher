"""Proven backtest orchestrator (DESIGN.md §5.2).

``run_backtest_proven`` wraps the existing ``backtest/engine.py`` API — the
engine signature is untouched; this is a separate orchestrator that the CLI
calls. Data enters only through the PIT vault ``asof()`` choke point; every
read is recorded into an ``InMemoryRecorder`` and lands in the bundle's
``data_manifest``.

Decision-clock note (ADVERSARIAL §1b-W1 fix): the runner derives the decision
clock FROM THE DATA — the distinct ``known_at`` times of the decision grid
(``gold/weights``) — and steps through it, reading each panel ``asof(t)`` per
decision time ``t`` inside a ``run_context.decision_window(t)``. The watchdog
therefore compares every read's ``max_known_at`` against the DECISION time,
not against the read's own asof argument (the old wiring passed the asof as
decision_time, making the leak check tautological; compounded by the old
``ASOF_SENTINEL`` panel reads). The sentinel is retained only for the single
grid-discovery weights probe. Determinism (A3 #6) is preserved: every
recorded ``asof_utc`` is a data-derived decision time, never wall clock.

The monolithic engine consumes one frame pair per run; it is fed the PIT
state knowable at the LAST decision time (the final per-decision reads).
Per-decision engine stepping requires an engine API change and is tracked as
residual work — the watchdog decision-window wiring is what protects strategy
code that reads the vault directly mid-run.

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
from quant_fund.proofcore import run_context
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

#: Deterministic "latest known" as-of for the single decision-grid discovery
#: probe (never wall clock). All other reads use data-derived decision times.
ASOF_SENTINEL = datetime(9999, 12, 31, 23, 59, 59, tzinfo=UTC)


def _decision_clock(weights: pl.DataFrame) -> list[datetime]:
    """The run's decision clock: distinct sorted ``known_at`` times of the
    decision grid (ADVERSARIAL §1b-W1).

    Decision times are keyed on ``known_at`` (when the grid row became
    observable), not ``event_time``: a decision for grid row ``d`` cannot be
    made before its weights were published, so ``known_at(d)`` is the
    earliest honest decision time and ``asof(known_at(d))`` is exactly the
    PIT state the decision may consume.
    """
    if "known_at" not in weights.columns:
        raise ProofError("decision grid (gold/weights) lacks a known_at column")
    times = (
        weights.select(pl.col("known_at").unique().sort())
        .get_column("known_at")
        .to_list()
    )
    clock = [t for t in times if t is not None]
    for t in clock:
        if not isinstance(t, datetime) or t.tzinfo is None or t.tzinfo.utcoffset(t) is None:
            raise ProofError(f"decision clock entry is not a tz-aware datetime: {t!r}")
    return clock


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
    3. derive the decision clock from the weights grid's known_at times and
       read both panels asof(t) per decision time t inside decision_window(t)
       (ADVERSARIAL §1b-W1); then run backtest/engine.py run_backtest via the
       existing API on the last decision's PIT frames
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
    watchdog = getattr(vault, "watchdog", None)

    # Decision-grid discovery probe: the ONLY sentinel read. The weights
    # panel carries the strategy's decision schedule; its known_at grid is
    # the decision clock.
    grid_read = vault.asof(WEIGHTS_DATASET, ASOF_SENTINEL)
    clock = _decision_clock(grid_read.frame)
    if not clock:
        raise ProofError(
            "decision clock is empty: gold/weights has no known_at rows — "
            "refusing to prove a run with no decisions"
        )

    bars_read: PitFrameLike | None = None
    weights_read: PitFrameLike | None = None
    # ADVERSARIAL §1b-W1/W2: the proven-run context auto-attaches any OTHER
    # vault strategy code may construct mid-run, and decision_window(t) makes
    # the watchdog assert max_known_at <= t (the DECISION time) per read.
    with run_context.proven_run(recorder, watchdog):
        for decision_time in clock:
            with run_context.decision_window(decision_time):
                bars_read = vault.asof(BARS_DATASET, decision_time)
                weights_read = vault.asof(WEIGHTS_DATASET, decision_time)
        assert bars_read is not None and weights_read is not None  # clock non-empty
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

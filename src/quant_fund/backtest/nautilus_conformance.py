"""P4.4 — NautilusTrader conformance replay attempt (third incumbent).

Attempts to replay a canonical conformance workload through the
NautilusTrader backtest engine — same bars, same causal orders, same cost
model — and compare fill-level output against
``backtest.fast_replay.run_backtest_fast``.

Honesty contract: the lane never claims a match it did not execute. When
the engine is absent from the environment the run fails closed into a
sealed ``nautilus_conformance`` receipt with ``verdict: blocked`` — the
attempt is evidenced (env fingerprint records the absence), not skipped
silently. A matched replay emits ``verdict: pass``; a completed replay
whose fills diverge emits ``verdict: fail``.

The equivalence claim under test is deliberately narrow — identical to the
claim ``test_p42_fast_replay_conformance`` pins against the reference
engine:

- fill events must agree on ``(ts, asset, signed qty, exec px, fee)``
  per bar, including the dedup gate (one fill per asset per bar) and the
  participation cap;
- mark-to-market of non-executing names must use the pre-update mark
  (the P4.2 conformance bug pinned by ``test_fast_replay_mark``);
- costs are the repo's maker/taker ``fee_bps`` — no exchange-specific
  fee schedules.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from quant_fund.research.receipt_v2 import build_receipt_v2, seal_receipt

NAUTILUS_CONFORMANCE_SCHEMA = "nautilus_conformance.v1"
NAUTILUS_CONFORMANCE_KIND = "nautilus_conformance"

# Pinned for reproducibility — the adapter is written against this
# version's API; a different installed version is recorded in the receipt.
PINNED_NAUTULUS_VERSION = "1.220.0"


@dataclass
class NautilusStatus:
    installed: bool
    version: str | None = None
    pinned: str = PINNED_NAUTULUS_VERSION


def nautilus_status() -> NautilusStatus:
    """Detect the engine without importing it (the Rust extension is heavy)."""
    if importlib.util.find_spec("nautilus_trader") is None:
        return NautilusStatus(installed=False)
    try:
        return NautilusStatus(installed=True, version=importlib.metadata.version("nautilus_trader"))
    except importlib.metadata.PackageNotFoundError:  # importable but not dist-installed
        return NautilusStatus(installed=True, version=None)


def conformance_equivalence_spec() -> dict[str, Any]:
    """The exact fields/events a third-party replay must match.

    This is the written contract an adapter implements; it is also embedded
    in the receipt so the equivalence claim travels with the evidence.
    """
    return {
        "fill_key": ["ts", "asset", "signed_qty", "exec_px", "fee"],
        "fill_rules": [
            "one fill per asset per bar (dedup gate)",
            "exec price null => mark-only bar, no fill",
            "participation cap clips signed qty",
            "non-executing names valued at pre-update mark",
        ],
        "cost_model": "maker/taker fee_bps on traded notional",
        "compare": "canonical row digest over the fill ledger, byte-identical",
        "reference_result_fills_columns": [
            "fill_time",
            "signal_time",
            "security_id",
            "quantity",
            "price",
            "fee",
            "spread_cost",
            "impact_cost",
            "turnover_cost",
            "decision_price",
        ],
    }


@dataclass
class NautilusConformanceResult:
    """Outcome of one conformance attempt."""

    status: NautilusStatus
    outcome: str  # "matched" | "diverged" | "engine_absent" | "engine_error"
    detail: str
    n_fills_compared: int = 0
    mismatches: list[str] = field(default_factory=list)


def run_nautilus_conformance(*, seed: int = 0) -> NautilusConformanceResult:
    """Attempt the third-incumbent replay on the synthetic workload.

    Fail-closed: an absent or erroring engine returns ``engine_absent`` /
    ``engine_error`` — it never produces a match claim. The adapter itself
    lives in ``_replay_via_nautilus`` and is exercised only on hosts where
    ``nautilus_trader`` is installed.
    """
    status = nautilus_status()
    if not status.installed:
        return NautilusConformanceResult(
            status=status,
            outcome="engine_absent",
            detail=(
                f"nautilus_trader is not installed in this environment "
                f"(pinned target: {PINNED_NAUTULUS_VERSION}); the conformance "
                "replay cannot run here — install it on the execution box and "
                "re-run to obtain a matched/not-fair verdict"
            ),
        )
    try:
        fills_ref, fills_inc = _replay_both_engines(seed=seed)
    except Exception as exc:  # engine present but adapter failed — still evidence
        return NautilusConformanceResult(
            status=status,
            outcome="engine_error",
            detail=f"adapter raised {type(exc).__name__}: {exc}",
        )
    mismatches = _compare_fills(fills_ref, fills_inc)
    return NautilusConformanceResult(
        status=status,
        outcome="diverged" if mismatches else "matched",
        detail=f"{len(fills_ref)} reference fills vs {len(fills_inc)} incumbent fills",
        n_fills_compared=len(fills_ref),
        mismatches=mismatches[:20],
    )


def _canonical_workload(seed: int) -> tuple[pl.DataFrame, pl.DataFrame, Any]:
    """Deterministic synthetic workload inside the fast-replay class.

    3 assets x 12 daily bars with a seeded price path, a sparse
    target-weight panel (rebalance every 4 days), and the research config
    with wide-open risk gates — mirrors the conformance fixtures in
    ``tests/unit/backtest/test_event_sim.py``.
    """
    import numpy as np

    from quant_fund.config import load_config

    rng = np.random.default_rng(seed)
    sids = ["NA", "NB", "NC"]
    t0 = datetime(2024, 1, 2, tzinfo=UTC)
    bars: list[dict[str, Any]] = []
    px = {s: 100.0 + 10.0 * i for i, s in enumerate(sids)}
    for d in range(12):
        when = t0 + timedelta(days=d)
        for s in sids:
            px[s] *= float(np.exp(rng.normal(0.0, 0.01)))
            bars.append(
                {
                    "security_id": s,
                    "event_time": when,
                    "open": px[s],
                    "high": px[s],
                    "low": px[s],
                    "close": px[s],
                    "close_total_return": px[s],
                    "volume": 1_000_000.0,
                    "adv": 100_000_000.0,
                    "vol_20": 0.02,
                    "source": "synthetic",
                }
            )
    weights_rows: list[dict[str, Any]] = []
    for d in (0, 4, 8):
        when = t0 + timedelta(days=d)
        for j, s in enumerate(sids):
            weights_rows.append(
                {
                    "security_id": s,
                    "event_time": when,
                    "target_weight": float(rng.uniform(-0.3, 0.5) * (1 if j else -1)),
                }
            )
    cfg = load_config("configs/research.yaml")
    cfg.costs.frictionless = True
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 1.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    cfg.costs.participation_limit = 1.0
    return pl.DataFrame(bars), pl.DataFrame(weights_rows), cfg


def _replay_both_engines(*, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay the same synthetic workload in both engines.

    Reference side uses ``run_backtest_fast`` directly. Incumbent side
    drives NautilusTrader's ``BacktestEngine`` with the same bars and a
    strategy that re-issues the same causal orders; both sides are reduced
    to the fill ledger rows named by ``conformance_equivalence_spec``.
    """
    from quant_fund.backtest.fast_replay import run_backtest_fast

    bars, weights, cfg = _canonical_workload(seed)
    ref = _fill_ledger(run_backtest_fast(bars, weights, cfg))
    inc = _nautilus_fill_ledger(bars, weights)
    return ref, inc


def _fill_ledger(result: Any) -> list[dict[str, Any]]:
    """Reduce ``BacktestResult.fills`` to comparable fill rows."""
    fills = getattr(result, "fills", None)
    if fills is None or fills.height == 0:
        return []
    ledger = []
    for row in fills.iter_rows(named=True):
        ledger.append(
            {
                "ts": str(row["fill_time"]),
                "asset": str(row["security_id"]),
                "signed_qty": float(row["quantity"]),
                "exec_px": float(row["price"]),
                "fee": float(row["fee"]),
            }
        )
    return ledger


def _nautilus_fill_ledger(bars: pl.DataFrame, weights: pl.DataFrame) -> list[dict[str, Any]]:
    """Drive the Nautilus backtest engine on the same bars.

    The engine is imported lazily — this function only exists on hosts
    where the attempt is real.
    """
    import nautilus_trader  # noqa: F401 — presence verified by caller

    raise RuntimeError(
        "nautilus adapter not yet implemented: map bars -> QuoteTick feed, "
        "replay causal orders via BacktestEngine, reduce OrderFilled events "
        "to the fill_key rows in conformance_equivalence_spec()"
    )


def _compare_fills(ref: list[dict[str, Any]], inc: list[dict[str, Any]]) -> list[str]:
    """Row-level diff over the fill ledger (order-sensitive)."""
    mismatches: list[str] = []
    if len(ref) != len(inc):
        mismatches.append(f"fill count differs: {len(ref)} vs {len(inc)}")
    for i, (a, b) in enumerate(zip(ref, inc, strict=False)):
        for k in ("ts", "asset"):
            if a[k] != b[k]:
                mismatches.append(f"fill {i} {k}: {a[k]!r} != {b[k]!r}")
        for k in ("signed_qty", "exec_px", "fee"):
            if abs(a[k] - b[k]) > 1e-9:
                mismatches.append(f"fill {i} {k}: {a[k]} != {b[k]}")
    return mismatches


def run_nautilus_conformance_eval(*, seed: int = 0) -> dict[str, Any]:
    """Run the attempt and seal the outcome — pass/fail/blocked."""
    result = run_nautilus_conformance(seed=seed)
    verdict = {"matched": "pass", "diverged": "fail"}.get(result.outcome, "blocked")
    return build_receipt_v2(
        kind=NAUTILUS_CONFORMANCE_KIND,
        data_label="META",
        dataset={"workload": f"conformance:fast_replay_vs_nautilus:seed{seed}"},
        params={"seed": seed, "pinned_nautilus": PINNED_NAUTULUS_VERSION},
        code_files=(Path(__file__),),
        verdict=verdict,  # type: ignore[arg-type]
        payload={
            "schema": NAUTILUS_CONFORMANCE_SCHEMA,
            "seed": seed,
            "engine_installed": result.status.installed,
            "engine_version": result.status.version,
            "outcome": result.outcome,
            "detail": result.detail,
            "n_fills_compared": result.n_fills_compared,
            "mismatches": result.mismatches,
            "equivalence_spec": conformance_equivalence_spec(),
            "live_pnl_claim": False,
        },
    )


def nautilus_conformance_contract_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Envelope + payload contract for the conformance kind."""
    errors: list[str] = []
    payload = receipt.get("payload")
    if not isinstance(payload, Mapping):
        return ["payload_missing"]
    for key in (
        "schema",
        "engine_installed",
        "outcome",
        "detail",
        "equivalence_spec",
        "live_pnl_claim",
    ):
        if key not in payload:
            errors.append(f"payload_missing_{key}")
    if payload.get("schema") != NAUTILUS_CONFORMANCE_SCHEMA:
        errors.append("payload_schema_mismatch")
    if payload.get("outcome") not in (
        "matched",
        "diverged",
        "engine_absent",
        "engine_error",
    ):
        errors.append("payload_outcome_unknown")
    return errors


def nautilus_conformance_consistency_errors(body: Mapping[str, Any]) -> list[str]:
    """Re-derive the verdict from the recorded outcome — catches a receipt
    that claims ``matched`` on a host without the engine, or a pass verdict
    on a diverged replay."""
    payload = body.get("payload")
    if not isinstance(payload, Mapping):
        return []
    errors: list[str] = []
    outcome = payload.get("outcome")
    verdict = body.get("verdict")
    expected = {"matched": "pass", "diverged": "fail"}.get(str(outcome), "blocked")
    if verdict != expected:
        errors.append(f"verdict_outcome_inconsistent:{outcome!r}->{verdict!r}")
    if payload.get("engine_installed") is False and outcome != "engine_absent":
        errors.append("outcome_impossible_without_engine")
    return errors


def write_nautilus_conformance_receipt(receipt: Mapping[str, Any], out_dir: Path) -> Path:
    """Persist as ``nautilus_conformance_<sha16>.json``."""
    sealed = seal_receipt(receipt)
    digest = str(sealed["receipt_sha256"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{NAUTILUS_CONFORMANCE_KIND}_{digest[:16]}.json"

    path.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    return path

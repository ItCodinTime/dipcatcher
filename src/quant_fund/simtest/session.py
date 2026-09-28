"""One deterministic research-to-paper session.

The session builds a labeled synthetic tape, writes a checksummed signal
cache, calls a simulated research API, then runs the real paper loop and a
real backtest on the bars that survived the gates. Broker faults are applied
only through ``prepare_broker`` on :class:`~quant_fund.execution.simulated_broker.SimulatedBroker`.
Nothing here submits to a live broker.

Crash and broker-timeout faults raise before the simulated fill is applied.
The paper loop checkpoints only after a step finishes, so the in-flight step
is discarded and the next attempt resumes from the last durable cursor.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_fund.backtest.engine import run_backtest
from quant_fund.config.loader import load_config
from quant_fund.config.models import RuntimeMode
from quant_fund.execution.simulated_broker import OrderRecord, SimulatedBroker
from quant_fund.paper.ledger import (
    paper_root,
    validate_ledger_schema,
    validate_promotion_dry_run_receipt,
)
from quant_fund.paper.loop import StaleValuationError, run_paper_loop
from quant_fund.schemas.orders import Order, OrderStatus
from quant_fund.simtest.clock import SimClock
from quant_fund.simtest.faults import Fault, FaultSchedule, schedule_from_seed
from quant_fund.simtest.feed import NAMES, Bar, SimDataFeed, normalize_bars
from quant_fund.simtest.io import DiskFull, SimDisk, SimNetwork
from quant_fund.simtest.runtime import DeterministicRuntime
from quant_fund.utils.hashing import canonical_frame_fingerprint, canonical_json_bytes

_INITIAL_NAV = 1_000_000.0
_BROKER_KINDS = ("crash", "broker_timeout", "broker_reject", "partial_fill")


class CrashInjected(Exception):
    """Process died inside a simulated submit, before the fill was applied."""

    def __init__(self, order_id: str) -> None:
        super().__init__(order_id)
        self.order_id = order_id


class BrokerTimeout(Exception):
    """Simulated broker did not acknowledge. No fill was applied."""

    def __init__(self, order_id: str) -> None:
        super().__init__(order_id)
        self.order_id = order_id


@dataclass
class SessionResult:
    """Replayable session. Hashes cover every committed intermediate state."""

    seed: int
    n_days: int
    state_hashes: list[str]
    log_bytes: bytes
    outcomes: list[str]
    clock_repeats: int
    idempotent_resume: bool
    receipt_errors: list[str]
    ledger_ok: bool
    conservation_errors: list[str]
    cash: float | None
    shares: dict[str, float]
    order_ids: list[str]
    fill_ids: list[str]
    fills: list[dict[str, object]]
    reject_reasons: list[str]
    research_only: bool = True
    live_pnl_claim: bool = False
    data_source: str = "SYNTHETIC"
    backtest: str = "not_run"
    schedule: FaultSchedule = field(default_factory=lambda: FaultSchedule(()))


class _SubmitGate:
    """Idempotent submit wrapper. Default ``SimulatedBroker.submit`` is untouched."""

    def __init__(self, runtime: DeterministicRuntime, schedule: FaultSchedule) -> None:
        self.runtime = runtime
        self.schedule = schedule
        self.attempts: dict[int, int] = {}
        self.step = 0

    def install(self, broker: SimulatedBroker) -> None:
        broker.id_factory = self.runtime.mint
        original = broker.submit

        def _submit(order: Order, **kwargs: Any) -> OrderRecord:
            return self._guard(broker, original, order, kwargs)

        broker.submit = _submit  # type: ignore[method-assign]

    def _guard(
        self,
        broker: SimulatedBroker,
        original: Any,
        order: Order,
        kwargs: dict[str, Any],
    ) -> OrderRecord:
        for existing in broker.history:
            if existing.order.order_id == order.order_id and (
                existing.fill is not None or existing.reject_reason is not None
            ):
                return existing
        fault = _first_broker_fault(self.schedule.at(self.step))
        attempt = self.attempts.get(self.step, 1)
        if fault is not None and attempt == 1 and fault.kind == "crash":
            raise CrashInjected(order.order_id)
        if fault is not None and attempt == 1 and fault.kind == "broker_timeout":
            raise BrokerTimeout(order.order_id)
        if fault is not None and fault.kind == "broker_reject" and broker.allow_capital:
            rejected = order.model_copy(update={"status": OrderStatus.REJECTED})
            record = OrderRecord(
                order=rejected,
                reject_reason="sim_broker_reject",
                slot=broker.slot,
            )
            # ``submit`` is what normally records the reject. Returning early
            # still has to leave the same record on the book the ledger saw.
            broker.history.append(record)
            broker.reject_count += 1
            return record
        if fault is not None and fault.kind == "partial_fill" and broker.allow_capital:
            fraction = min(max(float(fault.magnitude), 0.05), 0.95)
            order = order.model_copy(
                update={"quantity": max(float(order.quantity) * fraction, 1e-4)}
            )
        filled: OrderRecord = original(order, **kwargs)
        return filled


def _first_broker_fault(faults: tuple[Fault, ...]) -> Fault | None:
    rank = {kind: index for index, kind in enumerate(_BROKER_KINDS)}
    chosen = [fault for fault in faults if fault.kind in rank]
    if not chosen:
        return None
    chosen.sort(key=lambda fault: rank[fault.kind])
    return chosen[0]


def _signal_weights(tape: list[Bar], decision_time: datetime) -> dict[str, float]:
    """Causal sign of the last close-to-close move. Day one is a fixed AAA long."""
    history: dict[str, list[float]] = {name: [] for name in NAMES}
    for bar in tape:
        if bar.event_time <= decision_time:
            history[bar.security_id].append(bar.close)
    weights: dict[str, float] = {}
    for name in NAMES:
        closes = history[name]
        if len(closes) < 2:
            weights[name] = 0.04 if name == "AAA" else 0.0
            continue
        if closes[-1] > closes[-2]:
            weights[name] = 0.04
        elif closes[-1] < closes[-2]:
            weights[name] = -0.04
        else:
            weights[name] = 0.0
    return weights


def _bars_frame(bars: list[Bar]) -> pl.DataFrame:
    frame = pl.DataFrame([bar.as_row() for bar in bars])
    return frame.with_columns(pl.col("event_time").cast(pl.Datetime(time_zone="UTC")))


def _weight_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    frame = pl.DataFrame(rows)
    return frame.with_columns(pl.col("event_time").cast(pl.Datetime(time_zone="UTC")))


def _ledger_dir(root: Path, run_id: str) -> Path:
    return paper_root(root, "simtest-paper") / run_id


def _load_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    loaded: Any = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        return None
    return loaded


def _broker_view(path: Path) -> dict[str, Any]:
    state = _load_state(path)
    if state is None:
        return {"present": False}
    champ = state["champion"]
    history: list[dict[str, Any]] = []
    for item in champ.get("history") or []:
        order = item["order"]
        fill = item.get("fill")
        history.append(
            {
                "order_id": str(order["order_id"]),
                "status": str(order["status"]),
                "side": str(order["side"]),
                "security_id": str(order["security_id"]),
                "qty": float(order["quantity"]),
                "reject_reason": item.get("reject_reason"),
                "fill_id": None if fill is None else str(fill["fill_id"]),
                "fill_qty": None if fill is None else float(fill["quantity"]),
                "fill_price": None if fill is None else float(fill["price"]),
            }
        )
    shares = {str(key): float(value) for key, value in dict(champ.get("shares") or {}).items()}
    marks = {str(key): float(value) for key, value in dict(champ.get("last_marks") or {}).items()}
    return {
        "present": True,
        "step": int(state["step"]),
        "cash": float(champ["cash"]),
        "shares": {key: shares[key] for key in sorted(shares)},
        "marks": {key: marks[key] for key in sorted(marks)},
        "history": history,
    }


def _paper_config(root: Path) -> Any:
    config = load_config("configs/paper.yaml")
    config.data.root = root
    config.data.source = "synthetic"
    config.runtime.mode = RuntimeMode.PAPER
    config.paper.ledger_subdir = "simtest-paper"
    config.paper.promote_min_steps = 5
    config.risk_gate.stale_price_bars = 8
    config.risk_gate.max_name = 0.5
    config.risk_gate.max_net = 1.0
    config.risk_gate.max_gross = 2.0
    config.risk_gate.max_order_notional = 1e12
    config.risk_gate.max_participation = 1.0
    config.costs.participation_limit = 1.0
    return config


def run_session(
    seed: int,
    *,
    n_days: int = 6,
    schedule: FaultSchedule | None = None,
    replay_log: bytes | None = None,
    max_faults: int = 3,
) -> SessionResult:
    """Run one multi-day paper session. Pass ``replay_log`` to replay a trace."""
    if n_days < 1:
        raise ValueError("n_days must be positive")
    resolved = (
        schedule
        if schedule is not None
        else schedule_from_seed(seed, n_days, max_faults=max_faults)
    )
    with tempfile.TemporaryDirectory(prefix="simtest-") as temporary:
        return _run_in(Path(temporary), int(seed), n_days, resolved, replay_log)


def _run_in(
    root: Path,
    seed: int,
    n_days: int,
    schedule: FaultSchedule,
    replay_log: bytes | None,
) -> SessionResult:
    runtime = DeterministicRuntime(seed, replay_log=replay_log)
    feed = SimDataFeed(runtime, n_days, schedule)
    clock = SimClock(feed.calendar[0])
    disk = SimDisk(runtime)
    network = SimNetwork(runtime)
    gate = _SubmitGate(runtime, schedule)
    config = _paper_config(root)
    run_id = f"sim{seed:08x}"
    state_path = _ledger_dir(root, run_id) / "broker_state.json"

    tape: list[Bar] = list(feed.opening_bars())
    weight_rows: list[dict[str, object]] = []
    shadow_rows: list[dict[str, object]] = []
    outcomes: list[str] = []
    repeats = 0
    checkpoint = False

    def commit(phase: str, step: int, outcome: str, extra: dict[str, Any]) -> None:
        outcomes.append(outcome)
        payload: dict[str, Any] = {
            "step": step,
            "phase": phase,
            "outcome": outcome,
            "broker": _broker_view(state_path),
        }
        payload.update(extra)
        runtime.commit_state(f"{phase}-{step}-{len(outcomes)}", payload)

    for step, decision_time in enumerate(feed.calendar[:-1]):
        gate.step = step
        reading = clock.advance_to(decision_time, schedule.at(step))
        runtime.effect("clock", {"step": step}, reading.as_dict())
        if reading.repeat:
            repeats += 1
            commit(
                "clock-repeat", step, "recovered", {"now": reading.now, "leap": reading.leap_second}
            )

        delivered = feed.execution_bars(step)
        normalized, feed_status = normalize_bars(delivered)
        if feed_status == "conflict":
            commit("feed", step, "fail_closed", {"reason": "conflicting_duplicate"})
            break

        weights = _signal_weights(tape, decision_time)
        cache_body = canonical_json_bytes(
            {"decision": decision_time.isoformat(), "weights": weights, "source": "SYNTHETIC"}
        )
        cache_path = f"cache/step-{step}.json"
        kinds = schedule.kinds(step)
        if "disk_full" in kinds:
            try:
                disk.write(cache_path, cache_body, force_full=True)
            except DiskFull:
                commit("disk", step, "fail_closed", {"reason": "disk_full"})
                break
        else:
            disk.write(cache_path, cache_body)

        corrupt = "corrupt_cache" in kinds
        cached = disk.read(cache_path, corrupt=corrupt)
        outcome = "ok"
        if hashlib.sha256(cached).digest() != hashlib.sha256(cache_body).digest():
            try:
                disk.write(cache_path, cache_body)
            except DiskFull:
                commit("disk", step, "fail_closed", {"reason": "disk_full_rewrite"})
                break
            outcome = "recovered"

        status = 503 if "api_5xx" in kinds else 200
        response = network.request(
            "POST",
            "sim://research/signal",
            cache_body,
            status=status,
            slow="api_slow" in kinds,
        )
        if response.status >= 500:
            commit("api", step, "fail_closed", {"status": response.status})
            break
        if reading.causal_block:
            commit("clock", step, "fail_closed", {"reason": "clock_behind_exchange"})
            break

        step_weights = [
            {
                "event_time": decision_time,
                "security_id": name,
                "target_weight": float(weights[name]),
            }
            for name in NAMES
        ]
        step_shadow = [
            {
                "event_time": decision_time,
                "security_id": name,
                "target_weight": float(weights[name]) * 0.5,
            }
            for name in NAMES
        ]
        proposed = tape + normalized
        proposed_weights = weight_rows + step_weights
        proposed_shadow = shadow_rows + step_shadow
        gate.attempts[step] = 1
        try:
            _trade(
                proposed,
                proposed_weights,
                proposed_shadow,
                config=config,
                run_id=run_id,
                resume=checkpoint,
                gate=gate,
            )
        except (CrashInjected, BrokerTimeout):
            gate.attempts[step] = 2
            try:
                _trade(
                    proposed,
                    proposed_weights,
                    proposed_shadow,
                    config=config,
                    run_id=run_id,
                    resume=checkpoint,
                    gate=gate,
                )
            except StaleValuationError:
                commit("paper", step, "fail_closed", {"reason": "stale_valuation"})
                break
            outcome = "recovered"
        except StaleValuationError:
            commit("paper", step, "fail_closed", {"reason": "stale_valuation"})
            break
        tape = proposed
        weight_rows = proposed_weights
        shadow_rows = proposed_shadow
        checkpoint = True
        commit("paper", step, outcome, {"decision": decision_time})

    idempotent = True
    receipt_errors: list[str] = []
    ledger_ok = True
    if checkpoint:
        before = _broker_view(state_path)
        _trade(tape, weight_rows, shadow_rows, config=config, run_id=run_id, resume=True, gate=gate)
        after = _broker_view(state_path)
        idempotent = _same_book(before, after)
        promo_path = _ledger_dir(root, run_id) / "promotion_dry_run.json"
        if promo_path.is_file():
            promo = json.loads(promo_path.read_text())
            receipt_errors = validate_promotion_dry_run_receipt(promo)
        else:
            receipt_errors = ["promotion_receipt_missing"]
        ledger_report = validate_ledger_schema(_ledger_dir(root, run_id))
        ledger_ok = bool(ledger_report.get("ok"))
        if not ledger_ok:
            receipt_errors = list(receipt_errors) + [
                str(item) for item in ledger_report.get("errors", [])
            ]
        commit("resume", n_days, "ok", {"idempotent": idempotent})

    backtest = "not_run"
    if weight_rows:
        backtest = _run_backtest(tape, weight_rows, config, runtime)

    final = _broker_view(state_path)
    cash = float(final["cash"]) if final.get("present") else None
    shares = dict(final["shares"]) if final.get("present") else {}
    history = list(final.get("history") or [])
    fills = [
        {
            "security_id": item["security_id"],
            "side": item["side"],
            "fill_qty": item["fill_qty"],
            "fill_price": item["fill_price"],
            "fill_id": item["fill_id"],
        }
        for item in history
        if item.get("fill_id")
    ]
    conservation = (
        _conservation_errors(state_path, _ledger_dir(root, run_id)) if final.get("present") else []
    )
    return SessionResult(
        seed=seed,
        n_days=n_days,
        state_hashes=list(runtime.hashes),
        log_bytes=runtime.to_bytes(),
        outcomes=outcomes,
        clock_repeats=repeats,
        idempotent_resume=idempotent,
        receipt_errors=receipt_errors,
        ledger_ok=ledger_ok,
        conservation_errors=conservation,
        cash=cash,
        shares=shares,
        order_ids=[str(item["order_id"]) for item in history],
        fill_ids=[str(item["fill_id"]) for item in history if item.get("fill_id")],
        fills=fills,
        reject_reasons=[
            str(item["reject_reason"]) for item in history if item.get("reject_reason")
        ],
        backtest=backtest,
        schedule=schedule,
    )


def _trade(
    bars: list[Bar],
    weights: list[dict[str, object]],
    shadow: list[dict[str, object]],
    *,
    config: Any,
    run_id: str,
    resume: bool,
    gate: _SubmitGate,
) -> None:
    run_paper_loop(
        _bars_frame(bars),
        config,
        champion_weights=_weight_frame(weights),
        shadow_weights=_weight_frame(shadow),
        initial_nav=_INITIAL_NAV,
        run_id=run_id,
        max_steps=1,
        resume=resume,
        resume_run_id=run_id if resume else None,
        prefer_latest=False,
        prepare_broker=gate.install,
    )


def _same_book(before: dict[str, Any], after: dict[str, Any]) -> bool:
    if before.get("present") is not True or after.get("present") is not True:
        return False
    return bool(
        before["cash"] == after["cash"]
        and before["shares"] == after["shares"]
        and [item["order_id"] for item in before["history"]]
        == [item["order_id"] for item in after["history"]]
        and [item["fill_id"] for item in before["history"]]
        == [item["fill_id"] for item in after["history"]]
    )


def _run_backtest(
    tape: list[Bar],
    weights: list[dict[str, object]],
    config: Any,
    runtime: DeterministicRuntime,
) -> str:
    try:
        result = run_backtest(
            _bars_frame(tape),
            _weight_frame(weights),
            config,
            initial_nav=_INITIAL_NAV,
        )
    except StaleValuationError:
        runtime.commit_state("backtest", {"status": "fail_closed"})
        return "fail_closed"
    fingerprint = canonical_frame_fingerprint(result.fills) if result.fills.height else "empty"
    nav = None
    if result.equity.height and "nav" in result.equity.columns:
        nav = float(result.equity["nav"][-1])
    runtime.commit_state("backtest", {"status": "ok", "fills": fingerprint, "final_nav": nav})
    return "ok"


def _conservation_errors(state_path: Path, ledger_dir: Path) -> list[str]:
    state = _load_state(state_path)
    if state is None:
        return ["broker_state_missing"]
    champ = state["champion"]
    cash = float(champ["initial_cash"])
    shares: dict[str, float] = {}
    errors: list[str] = []
    seen_fills: set[str] = set()
    seen_orders: set[str] = set()
    for item in champ.get("history") or []:
        order = item["order"]
        fill = item.get("fill")
        if fill is None:
            continue
        fill_id = str(fill["fill_id"])
        order_id = str(order["order_id"])
        if fill_id in seen_fills:
            errors.append(f"duplicate_fill_id:{fill_id}")
        seen_fills.add(fill_id)
        if order_id in seen_orders:
            errors.append(f"duplicate_filled_order_id:{order_id}")
        seen_orders.add(order_id)
        quantity = float(fill["quantity"])
        price = float(fill["price"])
        signed = quantity if str(order["side"]) == "buy" else -quantity
        costs = float(fill["fee"]) + float(fill["spread_cost"]) + float(fill["impact_cost"])
        cash -= signed * price + costs
        security_id = str(fill["security_id"])
        shares[security_id] = shares.get(security_id, 0.0) + signed
    borrow = 0.0
    cash_path = ledger_dir / "cash_ledger.parquet"
    if cash_path.is_file():
        for row in pl.read_parquet(cash_path).to_dicts():
            if str(row.get("slot")) == "champion" and str(row.get("side")) == "borrow":
                borrow += float(row["cash_delta"])
    cash += borrow
    if abs(cash - float(champ["cash"])) > 1e-4:
        errors.append(f"cash_mismatch:{cash}:{champ['cash']}")
    stored = {str(key): float(value) for key, value in dict(champ.get("shares") or {}).items()}
    names = set(shares) | set(stored)
    for name in sorted(names):
        if abs(shares.get(name, 0.0) - stored.get(name, 0.0)) > 1e-6:
            errors.append(f"position_mismatch:{name}")
    shadow = state.get("shadow")
    if isinstance(shadow, dict) and abs(float(shadow.get("cash", 0.0))) > 1e-9:
        errors.append("shadow_cash_nonzero")
    return errors

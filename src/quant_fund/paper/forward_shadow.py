"""Prospective, paired, simulated-only paper adapter for the frozen Phase-1 slate.

The event journal is the source of truth. A close packet records intended orders;
only a *later* open packet may fill them. No data download or live broker exists.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import subprocess
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig, CostConfig, RiskGateConfig
from quant_fund.execution.costs import total_cost
from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.paper.xnys_calendar import (
    XnysSchedule,
    materialize_xnys_schedule,
    verify_xnys_schedule,
)
from quant_fund.research.net_replay import ReplayConfig, Strategy, _universe, _weights, market_panel
from quant_fund.research.real_benchmark import _load_bars, _read_receipt, _seal, protocol_for_run
from quant_fund.schemas.orders import Order, OrderSide

BOOKS = ("momentum_20", "equal_weight")
SCENARIOS = ("configured", "double_impact")
_EMPTY = "0" * 64
_ROOT = Path(__file__).resolve().parents[3]
_PUBLISHED_INDEX_SHA256 = "801ffecdd04fede3a832fc862a4321cd6b6124b75c97e110903b8da3582528e7"
_CALENDAR_FIRST = date(2026, 9, 18)
_CALENDAR_LAST = date(2034, 12, 31)


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _packet_hash(packet: dict[str, Any]) -> str:
    return _hash(
        json.dumps(packet, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def _dt(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("a timezone-aware ISO timestamp is required")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("timestamps must have an explicit UTC offset")
    return dt.astimezone(UTC)


def _load(path: Path) -> dict[str, Any]:
    raw = gzip.open(path, "rb").read() if path.suffix == ".gz" else path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: expected JSON object")
    return value


def _sealed(path: Path) -> dict[str, Any]:
    data = _load(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: expected JSON object")
    digest = data.pop("receipt_sha256", None)
    if _seal(data)["receipt_sha256"] != digest:
        raise ValueError(f"{path.name}: receipt SHA-256 mismatch")
    return {**data, "receipt_sha256": digest}


def _write_new(path: Path, content: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(content, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def _source_hashes() -> dict[str, str]:
    from quant_fund.config import models
    from quant_fund.execution import costs, simulated_broker
    from quant_fund.monitoring import kill_switch
    from quant_fund.paper import xnys_calendar
    from quant_fund.portfolio import risk_gate
    from quant_fund.research import net_replay, phase1_verify, real_benchmark
    from quant_fund.schemas import orders
    from quant_fund.utils import reproducibility

    def module_hash(module: Any) -> tuple[str, str]:
        filename = module.__file__
        if filename is None:
            raise ValueError("an execution dependency has no local source file")
        path = Path(filename)
        return path.name, _hash(path.read_bytes())

    return dict(
        map(
            module_hash,
            (
                net_replay,
                real_benchmark,
                phase1_verify,
                costs,
                simulated_broker,
                models,
                kill_switch,
                risk_gate,
                orders,
                reproducibility,
                xnys_calendar,
            ),
        )
    ) | {
        Path(__file__).name: _hash(Path(__file__).read_bytes()),
        "cli_main.py": _hash((Path(__file__).parents[1] / "cli" / "main.py").read_bytes()),
    }


def _checkout_state() -> tuple[str, str]:
    """Identify the exact source checkout; only committed, clean trees may freeze."""
    try:
        revision = subprocess.run(
            ["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "-C", str(_ROOT), "status", "--porcelain", "-z", "--untracked-files=all"],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("cannot identify the frozen Git checkout") from exc
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("frozen git_revision must identify an exact commit")
    if changes:
        raise ValueError("commit the forward protocol and use a clean worktree before freezing")
    return revision, _hash(b"")


def _warmup_cutoff(warmup: list[dict[str, Any]]) -> datetime:
    """Every historical availability and ingestion must precede the freeze."""
    return max(
        _dt(row[field])
        for row in warmup
        for field in ("event_time", "available_time", "ingested_time")
    )


@lru_cache(maxsize=4)
def _verified_schedule(raw: str) -> XnysSchedule:
    return verify_xnys_schedule(json.loads(raw))


def _calendar(manifest: dict[str, Any]) -> XnysSchedule:
    return _verified_schedule(
        json.dumps(manifest["exchange_schedule"], sort_keys=True, separators=(",", ":"))
    )


def _commitment(
    *,
    spec_sha256: str,
    benchmark_sha256: str,
    validation_sha256: str,
    code_sha256: dict[str, str],
    warmup: list[dict[str, Any]],
    power_sha256: str,
    git_revision: str,
    git_worktree_sha256: str,
    schedule_sha256: str,
    historical_index_sha256: str,
) -> str:
    return _hash(
        json.dumps(
            {
                "spec_sha256": spec_sha256,
                "benchmark_sha256": benchmark_sha256,
                "validation_sha256": validation_sha256,
                "code_sha256": code_sha256,
                "warmup_sha256": _hash(json.dumps(warmup, sort_keys=True).encode()),
                "power_sha256": power_sha256,
                "git_revision": git_revision,
                "git_worktree_sha256": git_worktree_sha256,
                "schedule_sha256": schedule_sha256,
                "historical_index_sha256": historical_index_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )


def _check_manifest(manifest: dict[str, Any]) -> None:
    revision, worktree_sha256 = _checkout_state()
    if manifest.get("kind") != "forward_shadow_manifest" or manifest.get("schema_version") != 1:
        raise ValueError("forward shadow manifest kind/schema mismatch")
    if (
        manifest.get("research_only") is not True
        or manifest.get("live_pnl_claim") is not False
        or manifest.get("external_attestation_verified") is not False
    ):
        raise ValueError("forward shadow manifest has unsupported honesty flags")
    if manifest.get("strategy") != "momentum_20" or manifest.get("benchmark") != "equal_weight":
        raise ValueError("frozen strategy/comparator changed")
    if manifest.get("min_paired_sessions") != 1400 or manifest.get("source") != "yahoo":
        raise ValueError("forward evidence horizon or source differs from the power plan")
    calendar = _calendar(manifest)
    if calendar.first_date != _CALENDAR_FIRST or calendar.last_date != _CALENDAR_LAST:
        raise ValueError("frozen XNYS schedule bounds changed")
    if manifest.get("historical_index_receipt_sha256") != _PUBLISHED_INDEX_SHA256 or (
        _sealed(_ROOT / "data/metadata/research/phase1_evidence_index.json")["receipt_sha256"]
        != _PUBLISHED_INDEX_SHA256
    ):
        raise ValueError("published historical evidence index changed")
    if manifest.get("code_sha256") != _source_hashes():
        raise ValueError("forward-shadow source code changed since freeze")
    if (
        manifest.get("git_revision") != revision
        or manifest.get("git_worktree_sha256") != worktree_sha256
    ):
        raise ValueError("frozen Git commit/clean worktree differs from this checkout")
    if manifest.get("spec_sha256") != _hash((_ROOT / "configs/net_tournament.json").read_bytes()):
        raise ValueError("frozen spec hash differs from the published slate")
    if manifest["tournament_spec"] != json.loads(
        (_ROOT / "configs/net_tournament.json").read_text()
    ):
        raise ValueError("frozen tournament spec content differs")
    if manifest.get("power_plan_sha256") != _hash(
        (_ROOT / "docs/FORWARD_SHADOW_POWER.md").read_bytes()
    ):
        raise ValueError("the pre-collection power analysis changed")
    benchmark = _sealed(_ROOT / "data/metadata/real_benchmark/us_wide_20260925/manifest.json")
    validation = _sealed(_ROOT / "data/metadata/net_tournament/us_wide_20260925/validation.json.gz")
    if (
        manifest.get("benchmark_receipt_sha256") != benchmark["receipt_sha256"]
        or manifest.get("validation_receipt_sha256") != validation["receipt_sha256"]
        or manifest.get("dataset_sha256") != benchmark["protocol"]["dataset_sha256"]
        or validation.get("selected") != "momentum_20"
    ):
        raise ValueError("frozen source/validation receipts are inconsistent")
    warmup = manifest["warmup"]
    if not isinstance(warmup, list) or not warmup:
        raise ValueError("historical warmup is missing")
    latest = max(_dt(row["event_time"]) for row in warmup)
    cutoff = _warmup_cutoff(warmup)
    ids = sorted({row["security_id"] for row in warmup if _dt(row["event_time"]) == latest})
    if ids != manifest.get("expected_security_ids") or len(ids) < 2:
        raise ValueError("frozen universe differs from historical warmup")
    proof = manifest["freeze"]
    if (
        not isinstance(proof, dict)
        or set(proof) != {"recorded_at", "reference", "issuer", "commitment_sha256"}
        or not proof["reference"]
        or not proof["issuer"]
        or not cutoff < _dt(proof["recorded_at"]) <= _dt(manifest["created_at"])
    ):
        raise ValueError("invalid external freeze timestamp/reference")
    if sum(session.day > _dt(proof["recorded_at"]).date() for session in calendar.sessions) < 1400:
        raise ValueError("frozen XNYS schedule cannot cover the planned paired horizon")
    expected = _commitment(
        spec_sha256=manifest["spec_sha256"],
        benchmark_sha256=manifest["benchmark_receipt_sha256"],
        validation_sha256=manifest["validation_receipt_sha256"],
        code_sha256=manifest["code_sha256"],
        warmup=warmup,
        power_sha256=manifest["power_plan_sha256"],
        git_revision=manifest["git_revision"],
        git_worktree_sha256=manifest["git_worktree_sha256"],
        schedule_sha256=calendar.schedule_sha256,
        historical_index_sha256=manifest["historical_index_receipt_sha256"],
    )
    if (
        proof["commitment_sha256"] != expected
        or manifest.get("protocol_commitment_sha256") != expected
    ):
        raise ValueError("external freeze commitment differs from the sealed protocol")


def _configured_books(spec: dict[str, Any], scenario: str) -> AppConfig:
    execution = ReplayConfig(**spec["execution"])
    execution.validate()
    multiplier = 2.0 if scenario == "double_impact" else 1.0
    return AppConfig(
        costs=CostConfig(
            commission_bps=execution.commission_bps,
            half_spread_bps=execution.half_spread_bps,
            impact_y=execution.impact_y * multiplier,
            participation_limit=execution.participation_limit,
            borrow_bps_per_year=execution.borrow_apr * 1e4,
            financing_bps_per_year=execution.funding_apr * 1e4,
        ),
        risk_gate=RiskGateConfig(
            max_order_notional=execution.initial_nav * 2,
            max_gross=execution.gross_limit,
            max_net=execution.gross_limit,
            max_name=execution.max_name_weight,
            max_participation=execution.participation_limit,
            max_predicted_vol=10.0,
        ),
    )


def _initial(manifest: dict[str, Any]) -> dict[str, Any]:
    spec = manifest["tournament_spec"]
    capital = float(spec["execution"]["initial_nav"])
    books = {}
    for scenario in SCENARIOS:
        cfg = _configured_books(spec, scenario)
        for name in BOOKS:
            key = f"{name}:{scenario}"
            books[key] = _compact(SimulatedBroker(cfg, initial_cash=capital, slot=key))
    return {
        "cursor": 0,
        "phase": "close",
        "last_event_sha256": manifest["receipt_sha256"],
        "last_close": None,
        "last_open": None,
        "last_observed_at": manifest["freeze"]["recorded_at"],
        "history": manifest["warmup"],
        "pending": None,
        "books": books,
        "paired_sessions": 0,
    }


def _compact(broker: SimulatedBroker) -> dict[str, Any]:
    state = broker.to_dict()
    state.pop("history")
    return state


def _restore(manifest: dict[str, Any], state: dict[str, Any], key: str) -> SimulatedBroker:
    scenario = key.split(":", 1)[1]
    broker = SimulatedBroker.from_state(
        _configured_books(manifest["tournament_spec"], scenario), state
    )
    if broker.slot != key or not broker.allow_capital:
        raise ValueError("broker slot/capital mismatch")
    return broker


def _event_path(run: Path, seq: int) -> Path:
    return run / "events" / f"{seq:08d}.json"


def _summary(state: dict[str, Any]) -> dict[str, Any]:
    return {
        **{key: val for key, val in state.items() if key != "history"},
        "history_sha256": _hash(
            json.dumps(
                state["history"], sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ),
    }


def _state(run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _sealed(run / "manifest.json")
    _check_manifest(manifest)
    previous = _initial(manifest)
    for seq, path in enumerate(sorted((run / "events").iterdir()), start=1):
        if path != _event_path(run, seq):
            raise ValueError("event sequence has a gap or an unexpected filename")
        item = _sealed(path)
        if (
            item.get("manifest_sha256") != manifest["receipt_sha256"]
            or item.get("prior_receipt_sha256") != previous["last_event_sha256"]
            or item.get("seq") != seq
            or item.get("before") != _summary(previous)
        ):
            raise ValueError(f"event {seq}: cursor, receipt chain or prior state mismatch")
        after = item.get("after")
        if not isinstance(after, dict) or after.get("cursor") != seq:
            raise ValueError(f"event {seq}: missing next cursor")
        if after.get("last_event_sha256") is not None:
            raise ValueError(f"event {seq}: next receipt link must be self-excluding")
        _check_event(manifest, previous, item)
        history = previous["history"]
        if item["stage"] == "close":
            history = [*history, *item["packet"]["bars"]]
            keep = max(Strategy(**item["strategy"]).required_history, 20) + 1
            dates = set(sorted({_dt(row["event_time"]) for row in history})[-keep:])
            history = [row for row in history if _dt(row["event_time"]) in dates]
        previous = {**after, "history": history, "last_event_sha256": item["receipt_sha256"]}
        if _summary(previous)["history_sha256"] != after["history_sha256"]:
            raise ValueError(f"event {seq}: rolling history digest mismatch")
    return manifest, previous


def _append(
    run: Path,
    manifest: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
    body: dict[str, Any],
) -> dict[str, Any]:
    seq = before["cursor"] + 1
    after = {**after, "cursor": seq, "last_event_sha256": None}
    event = {
        "seq": seq,
        "manifest_sha256": manifest["receipt_sha256"],
        "prior_receipt_sha256": before["last_event_sha256"],
        "before": _summary(before),
        "after": _summary(after),
        **body,
    }
    # The digest becomes the *next* event's prior link, never a field in itself.
    receipt = _seal(event)
    _write_new(_event_path(run, seq), receipt)
    return receipt


def _close_packet(
    packet: dict[str, Any], manifest: dict[str, Any], before: dict[str, Any], observed_at: datetime
) -> tuple[list[dict[str, Any]], datetime]:
    if packet.get("kind") != "forward_shadow_close":
        raise ValueError("expected forward_shadow_close packet")
    bars = packet.get("bars")
    if not isinstance(bars, list) or len(bars) < 2:
        raise ValueError("close packet requires at least two names")
    times = {_dt(row["event_time"]) for row in bars}
    if len(times) != 1 or len({row["security_id"] for row in bars}) != len(bars):
        raise ValueError("close packet must have one session and unique names")
    event = times.pop()
    calendar = _calendar(manifest)
    calendar.validate_close(event)
    freeze = _dt(manifest["freeze"]["recorded_at"])
    if event.date() <= freeze.date() or (
        before["last_close"] and event <= _dt(before["last_close"])
    ):
        raise ValueError("backfilled or repeated close cannot count as prospective")
    expected_ids = set(manifest["expected_security_ids"])
    observed_ids = {row["security_id"] for row in bars}
    if observed_ids != expected_ids:
        raise ValueError(
            f"close packet missing/extra frozen-universe names: "
            f"missing={sorted(expected_ids - observed_ids)} "
            f"extra={sorted(observed_ids - expected_ids)}"
        )
    declared = packet.get("missed_sessions", [])
    if not isinstance(declared, list) or any(
        not isinstance(row, dict) or row.get("reason") != "market_closed" for row in declared
    ):
        raise ValueError("no_feed/downtime must interrupt; only market_closed may be declared")
    if before["last_close"]:
        calendar.validate_gap(
            _dt(before["last_close"]).date(),
            event.date(),
            [row["date"] for row in declared],
        )
    else:
        first = next((s for s in calendar.sessions if s.day > freeze.date()), None)
        if first is None or event.date() != first.day or declared:
            raise ValueError(
                "first decision must use the first scheduled XNYS session after freeze"
            )
    _packet_attestation(packet, observed_at, event, manifest["source"])
    for row in bars:
        if (
            not {"security_id", "event_time", "available_time", "ingested_time", "close", "volume"}
            <= row.keys()
        ):
            raise ValueError("close packet missing price, volume, or causal timestamps")
        available, ingested = _dt(row["available_time"]), _dt(row["ingested_time"])
        if not event <= available <= ingested <= observed_at:
            raise ValueError("close packet violates event <= available <= ingested <= observed")
        # The frozen tournament's decision delay is zero. Later availability
        # cannot be retroactively treated as a close-time decision.
        if available > event:
            raise ValueError("close price was not available at the frozen decision cutoff")
        for key in ("close", "volume"):
            val = row[key]
            if isinstance(val, bool) or not math.isfinite(float(val)) or float(val) <= 0:
                raise ValueError("close and volume must be finite and positive")
    return bars, event


def _packet_attestation(
    packet: dict[str, Any], observed_at: datetime, event: datetime, source: str
) -> None:
    proof = packet.get("external_attestation")
    if not isinstance(proof, dict) or set(proof) != {"issuer", "reference", "recorded_at"}:
        raise ValueError("externally timestamped packet reference required")
    if packet.get("source") != source or not proof["issuer"] or not proof["reference"]:
        raise ValueError("packet source or external attestation reference differs")
    if not event <= _dt(proof["recorded_at"]) <= observed_at:
        raise ValueError("external packet timestamp is outside event/observation bounds")


def _open_packet(
    packet: dict[str, Any], manifest: dict[str, Any], before: dict[str, Any], observed_at: datetime
) -> tuple[dict[str, float], datetime]:
    if packet.get("kind") != "forward_shadow_open":
        raise ValueError("expected forward_shadow_open packet")
    bars = packet.get("bars")
    if not isinstance(bars, list) or not bars:
        raise ValueError("open packet requires bars")
    dates = {_dt(row["event_time"]) for row in bars}
    if len(dates) != 1 or len({row["security_id"] for row in bars}) != len(bars):
        raise ValueError("open packet must have one session and unique names")
    if {row["security_id"] for row in bars} != set(manifest["expected_security_ids"]):
        raise ValueError("open packet missing/extra frozen-universe names")
    event_time = dates.pop()
    calendar = _calendar(manifest)
    calendar.validate_next_open(_dt(before["last_close"]), event_time)
    declared = packet.get("market_closures", [])
    if not isinstance(declared, list) or any(
        not isinstance(row, dict) or row.get("reason") != "market_closed" for row in declared
    ):
        raise ValueError("next-open gap needs scheduled market closure records")
    calendar.validate_gap(
        _dt(before["last_close"]).date(),
        event_time.date(),
        [row["date"] for row in declared],
    )
    if event_time <= _dt(before["last_close"]) or (
        before["last_open"] and event_time <= _dt(before["last_open"])
    ):
        raise ValueError("execution must follow the pending close and prior open")
    if event_time <= _dt(before["last_observed_at"]) or observed_at <= _dt(
        before["last_observed_at"]
    ):
        raise ValueError("execution open must follow the observed close decision")
    _packet_attestation(packet, observed_at, event_time, manifest["source"])
    prices: dict[str, float] = {}
    for row in bars:
        if (
            not {"security_id", "event_time", "available_time", "ingested_time", "open"}
            <= row.keys()
        ):
            raise ValueError("open packet lacks a price or causal timestamp")
        available, ingested = _dt(row["available_time"]), _dt(row["ingested_time"])
        if not event_time <= available <= ingested <= observed_at:
            raise ValueError("open packet was not available at the observed execution time")
        price = float(row["open"])
        if not math.isfinite(price) or price <= 0:
            raise ValueError("open price must be finite and positive")
        prices[row["security_id"]] = price
    return prices, event_time


def prepare(
    spec_path: Path,
    benchmark_run: Path,
    tournament_run: Path,
    freeze_attestation: Path | None,
    output: Path | None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Freeze validation-selected strategy and pre-existing warmup; no decisions."""
    from quant_fund.research.phase1_verify import verify_phase1_index

    now = now or datetime.now(UTC)
    if (
        spec_path.resolve() != (_ROOT / "configs/net_tournament.json").resolve()
        or benchmark_run.resolve()
        != (_ROOT / "data/metadata/real_benchmark/us_wide_20260925").resolve()
        or tournament_run.resolve()
        != (_ROOT / "data/metadata/net_tournament/us_wide_20260925").resolve()
    ):
        raise ValueError("this adapter freezes the published canonical Phase-1 slate only")
    revision, worktree_sha256 = _checkout_state()
    index_path = _ROOT / "data/metadata/research/phase1_evidence_index.json"
    if (
        _sealed(index_path)["receipt_sha256"] != _PUBLISHED_INDEX_SHA256
        or not verify_phase1_index(index_path)["valid"]
    ):
        raise ValueError("the published historical receipt index must verify")
    benchmark = _read_receipt(benchmark_run / "manifest.json")
    tournament = _read_receipt(tournament_run / "manifest.json")
    validation = _sealed(tournament_run / "validation.json.gz")
    if validation.get("selected") != "momentum_20" or not validation.get("complete"):
        raise ValueError("the frozen validation selection must be momentum_20")
    if tournament["benchmark_manifest"] != benchmark:
        raise ValueError("benchmark does not match the selected tournament")
    spec_raw = spec_path.read_bytes()
    spec = json.loads(spec_raw)
    if spec != tournament["spec"] or spec["benchmark"]["name"] != "equal_weight":
        raise ValueError("the forward spec must exactly match the frozen tournament")
    selected = next((s for s in spec["trials"] if s["name"] == "momentum_20"), None)
    if selected is None or selected.get("family") != "momentum":
        raise ValueError("momentum_20 is not in the frozen trial slate")
    if any(s.get("long_short") or s.get("allocation") for s in [selected, spec["benchmark"]]):
        raise ValueError("this bounded paired adapter supports only frozen long-only rank books")
    protocol = protocol_for_run(benchmark, benchmark_run)
    if protocol.horizon_sessions != 1 or protocol.decision_delay_seconds != 0:
        raise ValueError("the selected tournament is not daily next-open/close-time")
    frame = _load_bars(protocol)
    history = max(
        Strategy(**selected).required_history, Strategy(**spec["benchmark"]).required_history
    )
    dates = frame["event_time"].unique().sort().to_list()[-(history + 1) :]
    if len(dates) < history + 1:
        raise ValueError("warmup lacks required observations")
    warmup = [
        {
            "security_id": row["security_id"],
            "event_time": row["event_time"].isoformat(),
            "available_time": row["available_time"].isoformat(),
            "ingested_time": row["ingested_time"].isoformat(),
            "close": row["price"],
            "volume": row[spec["volume_column"]],
        }
        for row in frame.filter(pl.col("event_time").is_in(dates)).iter_rows(named=True)
    ]
    expected_ids = sorted(
        {row["security_id"] for row in warmup if row["event_time"] == dates[-1].isoformat()}
    )
    if len(expected_ids) < 2:
        raise ValueError("latest warmup session has an insufficient frozen universe")
    calendar = materialize_xnys_schedule(_CALENDAR_FIRST, _CALENDAR_LAST)
    commitment = _commitment(
        spec_sha256=_hash(spec_raw),
        benchmark_sha256=benchmark["receipt_sha256"],
        validation_sha256=validation["receipt_sha256"],
        code_sha256=_source_hashes(),
        warmup=warmup,
        power_sha256=_hash((_ROOT / "docs/FORWARD_SHADOW_POWER.md").read_bytes()),
        git_revision=revision,
        git_worktree_sha256=worktree_sha256,
        schedule_sha256=calendar.schedule_sha256,
        historical_index_sha256=_PUBLISHED_INDEX_SHA256,
    )
    if freeze_attestation is None:
        return {
            "protocol_commitment_sha256": commitment,
            "git_revision": revision,
            "git_worktree_sha256": worktree_sha256,
            "exchange_schedule_sha256": calendar.schedule_sha256,
            "last_historical_warmup_session": dates[-1].isoformat(),
            "research_only": True,
            "live_pnl_claim": False,
        }
    if output is None:
        raise ValueError("freezing requires an output directory")
    proof = json.loads(freeze_attestation.read_text())
    if set(proof) != {"recorded_at", "reference", "issuer", "commitment_sha256"} or (
        proof["commitment_sha256"] != commitment or not proof["reference"] or not proof["issuer"]
    ):
        raise ValueError(f"freeze attestation must bind the protocol commitment {commitment}")
    frozen_at = _dt(proof["recorded_at"])
    if not _warmup_cutoff(warmup) < frozen_at <= now:
        raise ValueError("the externally recorded freeze must follow the inspected warmup")
    if sum(session.day > frozen_at.date() for session in calendar.sessions) < 1400:
        raise ValueError("frozen XNYS schedule cannot cover the planned paired horizon")
    if output.exists():
        raise FileExistsError("a frozen run cannot be overwritten")
    receipt = _seal(
        {
            "kind": "forward_shadow_manifest",
            "schema_version": 1,
            "created_at": now.isoformat(),
            "freeze": proof,
            "protocol_commitment_sha256": commitment,
            "strategy": "momentum_20",
            "benchmark": "equal_weight",
            "source": "yahoo",
            "expected_security_ids": expected_ids,
            "tournament_spec": spec,
            "spec_sha256": _hash(spec_raw),
            "benchmark_receipt_sha256": benchmark["receipt_sha256"],
            "validation_receipt_sha256": validation["receipt_sha256"],
            "power_plan_sha256": _hash((_ROOT / "docs/FORWARD_SHADOW_POWER.md").read_bytes()),
            "historical_index_receipt_sha256": _PUBLISHED_INDEX_SHA256,
            "exchange_schedule": calendar.as_dict(),
            "min_paired_sessions": 1400,
            "warmup": warmup,
            "warmup_status": "historical_only_excluded_from_evidence",
            "dataset_sha256": protocol.dataset_sha256,
            "code_sha256": _source_hashes(),
            "git_revision": revision,
            "git_worktree_sha256": worktree_sha256,
            "live_pnl_claim": False,
            "research_only": True,
            "external_attestation_verified": False,
            "limitations": [
                "Attestation timestamps/references are supplied by the data provider and require independent verification.",
                "Simulated open fills and costs are modeled, not broker executions.",
                "Historical warmup is excluded from prospective decisions and outcomes.",
            ],
        }
    )
    output.mkdir(parents=True)
    (output / "events").mkdir()
    _write_new(output / "manifest.json", receipt)
    return receipt


def _read_packet(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    packet = json.loads(raw)
    if not isinstance(packet, dict):
        raise ValueError("packet must be a JSON object")
    if any("synthetic" in str(packet.get(name, "")).lower() for name in ("source", "kind")):
        raise ValueError("synthetic packet cannot be counted as prospective evidence")
    return packet


def _panel(history: list[dict[str, Any]]) -> Any:
    rows = [
        {
            "security_id": row["security_id"],
            "event_time": _dt(row["event_time"]),
            "price": float(row["close"]),
            "open": float(row["close"]),
            "volume": float(row["volume"]),
            "known": True,
        }
        for row in history
    ]
    panel = market_panel(pl.DataFrame(rows), open_column="open", volume_column="volume")
    panel.close = np.asarray(panel.close, dtype=float)
    panel.volume = np.asarray(panel.volume, dtype=float)
    panel.known = np.asarray(panel.known, dtype=bool)
    return panel


def _plan_close(
    manifest: dict[str, Any],
    before: dict[str, Any],
    bars: list[dict[str, Any]],
    event_time: datetime,
    observed_at: datetime,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Derive a close decision from the prior state and the observed packet bars."""
    history = [*before["history"], *bars]
    panel = _panel(history)
    index = len(panel.dates) - 1
    spec = manifest["tournament_spec"]
    execution = ReplayConfig(**spec["execution"])
    strategies = {
        "momentum_20": Strategy(**next(s for s in spec["trials"] if s["name"] == "momentum_20")),
        "equal_weight": Strategy(**spec["benchmark"]),
    }
    required = max(s.required_history for s in strategies.values())
    if index < required:
        raise ValueError("the rolling history lacks sufficient completed sessions")
    eligible, adv, sigma = _universe(panel, index, required, execution)
    after = {
        **before,
        "history": history,
        "last_close": event_time.isoformat(),
        "last_observed_at": observed_at.isoformat(),
        "phase": "open",
    }
    days = set(sorted({_dt(row["event_time"]) for row in history})[-(required + 1) :])
    after["history"] = [row for row in history if _dt(row["event_time"]) in days]
    plans: dict[str, list[dict[str, Any]]] = {}
    decisions: dict[str, Any] = {}
    updated_books = dict(before["books"])
    for scenario in SCENARIOS:
        for name in BOOKS:
            key = f"{name}:{scenario}"
            broker = _restore(manifest, before["books"][key], key)
            close_marks = {
                sid: float(panel.close[index, j])
                for j, sid in enumerate(panel.names)
                if np.isfinite(panel.close[index, j]) and panel.close[index, j] > 0
            }
            if any(abs(q) > 1e-12 and sid not in close_marks for sid, q in broker.shares.items()):
                raise ValueError("a held name lacks a close-time mark; preserve the failed attempt")
            broker.mark(close_marks)
            nav = broker.nav(close_marks)
            if nav <= 0:
                raise ValueError("nonpositive signal NAV")
            target = _weights(panel, index, eligible, strategies[name], execution)
            intended = []
            for j, sid in enumerate(panel.names):
                held = broker.shares.get(sid, 0.0)
                if abs(target[j]) < 1e-12 and abs(held) < 1e-12:
                    continue
                close = panel.close[index, j]
                if not np.isfinite(close) or close <= 0:
                    raise ValueError("a held/target name lacks a causal close mark")
                desired = float(target[j] * nav / close)
                delta = desired - held
                if abs(delta) <= 1e-12:
                    continue
                intended.append(
                    {
                        "security_id": sid,
                        "requested_quantity": float(delta),
                        "target_weight": float(target[j]),
                        "signal_nav": float(nav),
                        "decision_price": float(close),
                        "known_adv": float(adv[j]) if np.isfinite(adv[j]) else None,
                        "known_volatility": float(sigma[j]) if np.isfinite(sigma[j]) else None,
                        "risk_reducing": abs(desired) < abs(held),
                    }
                )
            intended.sort(key=lambda row: (not row["risk_reducing"], row["security_id"]))
            for order_index, item in enumerate(intended):
                item["order_id"] = f"{name}-{scenario}-{before['cursor'] + 1}-{order_index}"
            plans[key] = intended
            decisions[key] = {
                "signal_nav": nav,
                "eligible_names": int(eligible.sum()),
                "target_weights": {
                    panel.names[j]: float(w) for j, w in enumerate(target) if w != 0
                },
                "cash": broker.cash,
                "positions": dict(broker.shares),
            }
            updated_books[key] = {
                **_compact(broker),
                "last_nav": before["books"][key].get("last_nav", execution.initial_nav),
            }
    after["books"] = updated_books
    after["pending"] = plans
    strategy = next(s for s in spec["trials"] if s["name"] == "momentum_20")
    return after, decisions, plans, strategy


def decide(run: Path, packet_path: Path, *, now: datetime | None = None) -> dict[str, Any]:
    """Record decisions using close-time information only; no fill is possible."""
    manifest, before = _state(run)
    if before["phase"] != "close":
        raise ValueError("next-open execution must reconcile before another decision")
    packet = _read_packet(packet_path)
    observed_at = _dt((now or datetime.now(UTC)).isoformat())
    bars, event_time = _close_packet(packet, manifest, before, observed_at)
    if observed_at <= _dt(before["last_observed_at"]):
        raise ValueError("close observation cannot precede the journal observation cursor")
    after, decisions, plans, strategy = _plan_close(manifest, before, bars, event_time, observed_at)
    receipt = _append(
        run,
        manifest,
        before,
        after,
        {
            "stage": "close",
            "session": event_time.isoformat(),
            "observed_at": observed_at.isoformat(),
            "packet": packet,
            "packet_sha256": _packet_hash(packet),
            "strategy": strategy,
            "decisions": decisions,
            "intended_orders": plans,
            "live_pnl_claim": False,
            "research_only": True,
            "external_attestation_verified": False,
        },
    )
    return receipt


def execute(run: Path, packet_path: Path, *, now: datetime | None = None) -> dict[str, Any]:
    """At the *next* session open, simulate independent paired fills and costs."""
    manifest, before = _state(run)
    if before["phase"] != "open" or before["pending"] is None:
        raise ValueError("a prior close decision must be recorded before execution")
    packet = _read_packet(packet_path)
    observed_at = now or datetime.now(UTC)
    prices, event_time = _open_packet(packet, manifest, before, observed_at)
    records: dict[str, Any] = {}
    fills: dict[str, Any] = {}
    rejects: dict[str, Any] = {}
    updated_books = dict(before["books"])
    costs: dict[str, Any] = {}
    cash_events: dict[str, Any] = {}
    positions: dict[str, Any] = {}
    daily: dict[str, Any] = {}
    execution = ReplayConfig(**manifest["tournament_spec"]["execution"])
    for key in sorted(before["books"]):
        broker = _restore(manifest, before["books"][key], key)
        if any(abs(q) > 1e-12 and sid not in prices for sid, q in broker.shares.items()):
            raise ValueError("held name lacks an open mark; no partial session was committed")
        broker.mark(prices)
        previous = before["books"][key]
        years = (
            (event_time - _dt(before["last_open"])).total_seconds() / (365 * 86400)
            if before["last_open"]
            else 0.0
        )
        if years < 0:
            raise ValueError("open dates are reversed")
        shorts = sum(-q * previous["last_marks"][sid] for sid, q in broker.shares.items() if q < 0)
        borrow = shorts * execution.borrow_apr * years
        financing = (
            -broker.cash * execution.funding_apr
            if broker.cash < 0
            else -broker.cash * execution.cash_apr
        ) * years
        broker.cash -= borrow + financing
        cash_events[key] = [
            {
                "type": "borrow_and_financing",
                "delta": -borrow - financing,
                "borrow": borrow,
                "financing": financing,
            }
        ]
        values = {
            "commission": 0.0,
            "spread": 0.0,
            "impact": 0.0,
            "borrow": borrow,
            "financing": financing,
        }
        order_rows = []
        for item in before["pending"][key]:
            sid = item["security_id"]
            q = float(item["requested_quantity"])
            order = Order(
                order_id=item["order_id"],
                security_id=sid,
                symbol=sid,
                side=OrderSide.BUY if q > 0 else OrderSide.SELL,
                quantity=abs(q),
                signal_time=_dt(before["last_close"]),
                decision_time=_dt(before["last_observed_at"]),
                order_time=event_time,
            )
            price = prices.get(sid, 0.0)
            adv, sigma = item["known_adv"], item["known_volatility"]
            record = broker.submit(
                order,
                price=price,
                adv_dollars=adv if adv is not None else 0.0,
                sigma=sigma if sigma is not None else float("nan"),
                decision_price=item["decision_price"],
            )
            fill = record.fill
            row = {
                "order_id": order.order_id,
                "security_id": sid,
                "signal_time": order.signal_time.isoformat(),
                "decision_time": order.decision_time.isoformat(),
                "order_time": order.order_time.isoformat(),
                "requested_quantity": q,
                "status": record.order.status.value,
                "reject_reason": record.reject_reason,
                "fill_quantity": (math.copysign(fill.quantity, q) if fill else 0.0),
                "unfilled_quantity": (q - math.copysign(fill.quantity, q) if fill else q),
                "fill_price": fill.price if fill else None,
                "fill_id": fill.fill_id if fill else None,
                "fill_time": fill.fill_time.isoformat() if fill else None,
                "is_partial": fill.is_partial if fill else False,
                "commission": fill.fee if fill else 0.0,
                "spread": fill.spread_cost if fill else 0.0,
                "impact": fill.impact_cost if fill else 0.0,
                "known_adv": adv,
                "known_volatility": sigma,
                "decision_price": item["decision_price"],
            }
            if fill is not None:
                for name in ("commission", "spread", "impact"):
                    values[name] += row[name]
                cash_events[key].append(
                    {
                        "type": "fill",
                        "order_id": order.order_id,
                        "delta": -row["fill_quantity"] * fill.price
                        - fill.fee
                        - fill.spread_cost
                        - fill.impact_cost,
                    }
                )
            order_rows.append(row)
        nav = broker.nav(prices)
        if not math.isfinite(nav) or nav <= 0:
            raise ValueError("nonpositive ending NAV")
        old_nav = float(previous.get("last_nav", execution.initial_nav))
        state = _compact(broker)
        state["last_nav"] = nav
        updated_books[key] = state
        records[key] = order_rows
        fills[key] = [row for row in order_rows if row["fill_quantity"]]
        rejects[key] = [row for row in order_rows if row["reject_reason"]]
        costs[key] = values
        positions[key] = dict(broker.shares)
        daily[key] = {
            "cash": broker.cash,
            "nav": nav,
            "net_return": nav / old_nav - 1,
            "position_market_value": nav - broker.cash,
            "rejects": broker.reject_count - int(previous["reject_count"]),
        }
    after = {
        **before,
        "phase": "close",
        "pending": None,
        "last_observed_at": observed_at.isoformat(),
        "last_open": event_time.isoformat(),
        "books": updated_books,
        "paired_sessions": before["paired_sessions"] + 1,
    }
    return _append(
        run,
        manifest,
        before,
        after,
        {
            "stage": "open",
            "session": event_time.isoformat(),
            "observed_at": observed_at.isoformat(),
            "packet": packet,
            "packet_sha256": _packet_hash(packet),
            "orders": records,
            "fills": fills,
            "rejects": rejects,
            "cash_ledger": cash_events,
            "positions": positions,
            "costs": costs,
            "daily": daily,
            "net_difference": {
                scenario: daily[f"momentum_20:{scenario}"]["net_return"]
                - daily[f"equal_weight:{scenario}"]["net_return"]
                for scenario in SCENARIOS
            },
            "live_pnl_claim": False,
            "research_only": True,
            "external_attestation_verified": False,
        },
    )


def interrupt(
    run: Path,
    reason: str,
    *,
    now: datetime | None = None,
    attempted_stage: str | None = None,
    attempted_packet: Path | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Preserve downtime or a broken feed and close this protocol version."""
    if reason not in {"no_feed", "downtime", "missing_name", "bad_timestamp", "other"}:
        raise ValueError(
            "interruption reason must be no_feed/downtime/missing_name/bad_timestamp/other"
        )
    manifest, before = _state(run)
    if before["phase"] == "blocked":
        raise ValueError("forward paper run is already interrupted")
    observed_at = now or datetime.now(UTC)
    if observed_at <= _dt(before["last_observed_at"]):
        raise ValueError("interruption observation must advance the timestamp cursor")
    after = {
        **before,
        "phase": "blocked",
        "last_observed_at": observed_at.isoformat(),
        "interruption_reason": reason,
    }
    packet: dict[str, Any] = {
        "kind": "forward_shadow_interruption",
        "reason": reason,
        "observed_at": observed_at.isoformat(),
    }
    if attempted_packet is not None and attempted_packet.exists():
        raw = attempted_packet.read_bytes()
        packet["attempted_file_sha256"] = _hash(raw)
        try:
            attempted = json.loads(raw)
            if isinstance(attempted, dict):
                packet["attempted_packet"] = attempted
                packet["attempted_packet_sha256"] = _packet_hash(attempted)
        except (UnicodeDecodeError, ValueError):
            packet["attempted_packet"] = None
    packet["attempted_stage"] = attempted_stage
    packet["error"] = error
    return _append(
        run,
        manifest,
        before,
        after,
        {
            "stage": "interruption",
            "session": observed_at.isoformat(),
            "observed_at": observed_at.isoformat(),
            "packet": packet,
            "packet_sha256": _packet_hash(packet),
            "reason": reason,
            "research_only": True,
            "live_pnl_claim": False,
            "external_attestation_verified": False,
        },
    )


def _check_event(manifest: dict[str, Any], previous: dict[str, Any], item: dict[str, Any]) -> None:
    if item.get("research_only") is not True or item.get("live_pnl_claim") is not False:
        raise ValueError("paper event may not assert live performance")
    if item.get("external_attestation_verified") is not False:
        raise ValueError("external attestation requires independent verification")
    if item.get("packet_sha256") is None or not isinstance(item.get("packet"), dict):
        raise ValueError("paper event must bind its input packet")
    if item["packet_sha256"] != _packet_hash(item["packet"]):
        raise ValueError("paper event input packet hash differs")
    observed = _dt(item["observed_at"])
    if (
        observed <= _dt(previous["last_observed_at"])
        or item["after"]["last_observed_at"] != item["observed_at"]
    ):
        raise ValueError("event observations are not strictly monotone")
    if item.get("stage") == "close":
        bars, date = _close_packet(item["packet"], manifest, previous, observed)
        if date.isoformat() != item["session"] or previous["phase"] != "close":
            raise ValueError("close event has wrong session or phase")
        expected_after, decisions, orders, strategy = _plan_close(
            manifest, previous, bars, date, observed
        )
        if item.get("strategy") != strategy:
            raise ValueError("close event strategy differs from the frozen selection")
        if item.get("decisions") != decisions:
            raise ValueError("close event weights or decisions differ from replay")
        if item.get("intended_orders") != orders:
            raise ValueError("close event intended orders differ from replay")
        expected_summary = _summary(
            {**expected_after, "cursor": previous["cursor"] + 1, "last_event_sha256": None}
        )
        if item["after"] != expected_summary:
            raise ValueError("close event broker or state differs from replay")
    elif item.get("stage") == "open":
        if previous["phase"] != "open" or previous["pending"] is None:
            raise ValueError("open event lacks prior decision")
        prices, event = _open_packet(item["packet"], manifest, previous, observed)
        if item["session"] != event.isoformat() or item["after"]["last_open"] != item["session"]:
            raise ValueError("open event date differs from its packet/state")
        if item["after"]["last_close"] != previous["last_close"]:
            raise ValueError("open event altered prior close")
        if _dt(item["session"]) <= _dt(previous["last_observed_at"]):
            raise ValueError("open was already historical when the decision was observed")
        if item["after"]["phase"] != "close" or item["after"]["pending"] is not None:
            raise ValueError("open event did not clear its pending decision")
        if item["after"]["paired_sessions"] != previous["paired_sessions"] + 1:
            raise ValueError("open event session counter mismatch")
        for key in previous["books"]:
            if item["fills"][key] != [r for r in item["orders"][key] if r["fill_quantity"]]:
                raise ValueError("open event fills ledger is incomplete")
            if item["rejects"][key] != [r for r in item["orders"][key] if r["reject_reason"]]:
                raise ValueError("open event rejects ledger is incomplete")
            if len(item["orders"][key]) != len(previous["pending"][key]):
                raise ValueError("open event omitted orders or rejects")
            if [row["order_id"] for row in item["orders"][key]] != [
                row["order_id"] for row in previous["pending"][key]
            ]:
                raise ValueError("open event order sequence differs from the frozen decision")
            before_cash = previous["books"][key]["cash"]
            cash_after = before_cash + sum(row["delta"] for row in item["cash_ledger"][key])
            book = item["after"]["books"][key]
            if not math.isclose(cash_after, book["cash"], abs_tol=1e-7):
                raise ValueError("open event cash/fill ledger does not reconcile")
            linear = item["cash_ledger"][key]
            if (
                len(linear) != len(item["fills"][key]) + 1
                or linear[0]["type"] != "borrow_and_financing"
            ):
                raise ValueError("open event cash event count is inconsistent")
            scenario = key.split(":", 1)[1]
            config = _configured_books(manifest["tournament_spec"], scenario)
            expected_costs = {name: 0.0 for name in ("commission", "spread", "impact")}
            previous_book = previous["books"][key]
            years = (
                (event - _dt(previous["last_open"])).total_seconds() / (365 * 86400)
                if previous["last_open"]
                else 0.0
            )
            execution = ReplayConfig(**manifest["tournament_spec"]["execution"])
            borrow = (
                sum(
                    -q * previous_book["last_marks"][sid]
                    for sid, q in previous_book["shares"].items()
                    if q < 0
                )
                * execution.borrow_apr
                * years
            )
            financing = (
                (-previous_book["cash"] * execution.funding_apr)
                if previous_book["cash"] < 0
                else (-previous_book["cash"] * execution.cash_apr)
            ) * years
            if (
                not math.isclose(linear[0]["delta"], -borrow - financing, abs_tol=1e-8)
                or not math.isclose(linear[0]["borrow"], borrow, abs_tol=1e-8)
                or not math.isclose(linear[0]["financing"], financing, abs_tol=1e-8)
            ):
                raise ValueError("open event borrow/financing does not reconcile")
            for intended, row in zip(previous["pending"][key], item["orders"][key], strict=True):
                if (
                    row["security_id"] != intended["security_id"]
                    or row["requested_quantity"] != intended["requested_quantity"]
                    or row["signal_time"] != previous["last_close"]
                    or row["decision_time"] != previous["last_observed_at"]
                    or row["order_time"] != item["session"]
                    or row["decision_price"] != intended["decision_price"]
                    or row["known_adv"] != intended["known_adv"]
                    or row["known_volatility"] != intended["known_volatility"]
                ):
                    raise ValueError("open event order differs from its frozen decision")
                if row["fill_quantity"] == 0 and not row["reject_reason"]:
                    raise ValueError("unfilled order must record a reject reason")
                if row["fill_quantity"] and row["reject_reason"]:
                    raise ValueError("filled order cannot be marked rejected")
                qty = float(row["fill_quantity"])
                if (
                    abs(qty) > abs(row["requested_quantity"]) + 1e-9
                    or qty * row["requested_quantity"] < 0
                    or not math.isclose(
                        row["unfilled_quantity"], row["requested_quantity"] - qty, abs_tol=1e-9
                    )
                ):
                    raise ValueError("fill participation or residual quantity is inconsistent")
                if qty:
                    px = prices.get(row["security_id"])
                    if (
                        px is None
                        or row["fill_price"] != px
                        or row["fill_time"] != item["session"]
                        or not row["fill_id"]
                    ):
                        raise ValueError("open fill price/time/id differs from observed packet")
                    costs = total_cost(
                        qty, px, row["known_adv"], row["known_volatility"], config.costs
                    )
                    if abs(qty) * px > config.costs.participation_limit * row["known_adv"] + 1e-7:
                        raise ValueError("open fill exceeds frozen participation")
                    for name, field in (
                        ("commission", "commission"),
                        ("spread", "spread"),
                        ("impact", "impact"),
                    ):
                        if not math.isclose(row[field], float(costs[name]), abs_tol=1e-7):
                            raise ValueError("open fill cost differs from frozen model")
                        expected_costs[field] += row[field]
                elif (
                    row["fill_price"] is not None
                    or row["fill_id"] is not None
                    or any(row[field] != 0 for field in expected_costs)
                ):
                    raise ValueError("rejected order carried a fill or modeled cost")
            for ledger_row, fill_row in zip(linear[1:], item["fills"][key], strict=True):
                expected_delta = -fill_row["fill_quantity"] * fill_row["fill_price"] - sum(
                    fill_row[k] for k in expected_costs
                )
                if (
                    ledger_row["type"] != "fill"
                    or ledger_row["order_id"] != fill_row["order_id"]
                    or not math.isclose(ledger_row["delta"], expected_delta, abs_tol=1e-7)
                ):
                    raise ValueError("cash event differs from fill notional/cost")
            for name, amount in {
                **expected_costs,
                "borrow": borrow,
                "financing": financing,
            }.items():
                if not math.isclose(item["costs"][key][name], amount, abs_tol=1e-7):
                    raise ValueError("daily costs differ from fill ledger")
            for sid in set(previous["books"][key]["shares"]) | set(item["positions"][key]):
                expected = previous["books"][key]["shares"].get(sid, 0.0) + sum(
                    row["fill_quantity"] for row in item["orders"][key] if row["security_id"] == sid
                )
                if not math.isclose(expected, item["positions"][key].get(sid, 0.0), abs_tol=1e-9):
                    raise ValueError("open event shares do not reconcile with fills")
            if book["shares"] != item["positions"][key]:
                raise ValueError("open event position snapshot differs from broker state")
            if not math.isclose(item["daily"][key]["cash"], book["cash"], abs_tol=1e-9):
                raise ValueError("open event daily cash differs from broker state")
            nav = book["cash"] + sum(
                q * prices[sid] for sid, q in book["shares"].items() if abs(q) > 1e-12
            )
            old_nav = previous_book.get("last_nav", execution.initial_nav)
            daily = item["daily"][key]
            if (
                not math.isclose(daily["nav"], nav, abs_tol=1e-7)
                or not math.isclose(book["last_nav"], nav, abs_tol=1e-7)
                or not math.isclose(daily["net_return"], nav / old_nav - 1, abs_tol=1e-9)
                or not math.isclose(
                    daily["position_market_value"], nav - book["cash"], abs_tol=1e-7
                )
                or daily["rejects"] != book["reject_count"] - previous_book["reject_count"]
            ):
                raise ValueError("daily NAV/net return or reject count does not reconcile")
        for scenario in SCENARIOS:
            actual = (
                item["daily"][f"momentum_20:{scenario}"]["net_return"]
                - item["daily"][f"equal_weight:{scenario}"]["net_return"]
            )
            if not math.isclose(item["net_difference"][scenario], actual, abs_tol=1e-12):
                raise ValueError("paired net difference does not reconcile")
    elif item.get("stage") == "interruption":
        attempted = item["packet"].get("attempted_packet")
        if isinstance(attempted, dict) and item["packet"].get(
            "attempted_packet_sha256"
        ) != _packet_hash(attempted):
            raise ValueError("interruption attempted packet hash changed")
        if (
            previous["phase"] == "blocked"
            or item["after"]["phase"] != "blocked"
            or item["after"]["paired_sessions"] != previous["paired_sessions"]
            or item["after"]["books"] != previous["books"]
            or item["after"]["pending"] != previous["pending"]
            or item["after"]["last_close"] != previous["last_close"]
            or item["after"]["last_open"] != previous["last_open"]
            or item["after"]["history_sha256"] != _summary(previous)["history_sha256"]
            or item["reason"] != item["packet"].get("reason")
            or item["after"]["interruption_reason"] != item["reason"]
            or item["reason"]
            not in {"no_feed", "downtime", "missing_name", "bad_timestamp", "other"}
        ):
            raise ValueError("interruption event does not preserve a closed protocol cursor")
    else:
        raise ValueError("unknown forward-shadow event stage")


def verify(run: Path) -> dict[str, Any]:
    """Validate the immutable receipt chain and every restart state transition."""
    try:
        manifest, state = _state(run)
        if (
            manifest.get("kind") != "forward_shadow_manifest"
            or manifest.get("research_only") is not True
        ):
            raise ValueError("forward shadow manifest is not research only")
        if manifest.get("live_pnl_claim") is not False:
            raise ValueError("forward shadow manifest claims live performance")
        if manifest["code_sha256"] != _source_hashes():
            raise ValueError("forward-shadow source code changed since freeze")
        return {
            "valid": True,
            "kind": "forward_shadow",
            "state": state["phase"],
            "interruption_reason": state.get("interruption_reason"),
            "paired_sessions": state["paired_sessions"],
            "minimum": manifest["min_paired_sessions"],
            "external_attestation_verified": False,
            "deterministic_close_replay": True,
            "independent_strategy_replay": False,
            "forward_evidence_accepted": False,
            "research_only": True,
            "live_pnl_claim": False,
            "errors": [],
        }
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return {"valid": False, "kind": "forward_shadow", "errors": [str(exc)]}

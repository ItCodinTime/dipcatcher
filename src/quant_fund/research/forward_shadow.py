"""Prospective, locally timestamped shadow record; no live broker connectivity."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sqlite3
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from scipy.stats import norm

from quant_fund.research import cost_allocation, forward_evidence, net_replay, shadow_journal
from quant_fund.research.cost_allocation import AllocationConfig
from quant_fund.research.forward_evidence import evidence_plan, long_run_variance
from quant_fund.research.net_replay import (
    MarketPanel,
    ReplayConfig,
    Strategy,
    _cost_weights,
    _universe,
    _weights,
    execute_orders,
)
from quant_fund.research.shadow_journal import canonical, digest


def _now() -> datetime:
    return datetime.now(UTC)


def _time(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return result.astimezone(UTC)


def _stamp(value: str) -> str:
    return _time(value).isoformat()


def _identity() -> dict[str, Any]:
    return {
        "code": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(net_replay.__file__),
                Path(cost_allocation.__file__),
                Path(shadow_journal.__file__),
                Path(forward_evidence.__file__),
            )
        },
        "runtime": {
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
            **{
                n: importlib.metadata.version(n)
                for n in ("numpy", "polars", "cvxpy", "clarabel", "scipy")
            },
        },
    }


def _strategy(raw: dict[str, Any]) -> Strategy:
    fields = dict(raw)
    if fields.get("allocation") is not None:
        fields["allocation"] = AllocationConfig(**fields["allocation"])
    strategy = Strategy(**fields)
    strategy.validate()
    return strategy


def _vector(value: Any, n: int, *, positive: bool, nullable: bool) -> list[float | None]:
    if not isinstance(value, list) or len(value) != n:
        raise ValueError("market vectors must match the frozen asset order")
    result: list[float | None] = []
    for v in value:
        if v is None and nullable:
            result.append(None)
        elif (
            isinstance(v, bool)
            or not isinstance(v, (float, int))
            or not np.isfinite(v)
            or (v <= 0 if positive else v < 0)
        ):
            raise ValueError("invalid price or volume")
        else:
            result.append(float(v))
    return result


def _bar(raw: dict[str, Any], n: int, now: datetime) -> dict[str, Any]:
    if set(raw) != {"event_time", "available_time", "close", "volume"}:
        raise ValueError("close snapshot requires event_time, available_time, close and volume")
    event, available = _time(raw["event_time"]), _time(raw["available_time"])
    if not event <= available <= now:
        raise ValueError("close snapshot is future-dated or has inconsistent availability")
    return {
        "event_time": event.isoformat(),
        "available_time": available.isoformat(),
        "close": _vector(raw["close"], n, positive=True, nullable=True),
        "volume": _vector(raw["volume"], n, positive=False, nullable=True),
    }


def freeze(spec: dict[str, Any], bootstrap: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    now = _now()
    if set(spec) != {
        "strategy",
        "benchmark",
        "execution",
        "assets",
        "sessions",
        "source_url",
        "price_basis",
        "selection_basis",
        "lead_seconds",
        "calibration",
        "evidence",
    }:
        raise ValueError("forward specification has missing or unknown keys")
    candidate, benchmark = _strategy(spec["strategy"]), _strategy(spec["benchmark"])
    if benchmark.family != "equal_weight" or benchmark.name == candidate.name:
        raise ValueError("a separate equal-weight benchmark is required")
    config = ReplayConfig(**spec["execution"])
    config.validate()
    if candidate.allocation and config.funding_apr < config.cash_apr:
        raise ValueError("allocation requires funding_apr >= cash_apr")
    assets = spec["assets"]
    if (
        not isinstance(assets, list)
        or not 2 <= len(assets) <= 500
        or any(not isinstance(a, str) or not a.strip() for a in assets)
        or assets != sorted(set(assets))
    ):
        raise ValueError("assets must be 2..500 unique sorted security identifiers")
    if (
        not isinstance(spec["source_url"], str)
        or not spec["source_url"].startswith("https://")
        or not isinstance(spec["selection_basis"], str)
        or not spec["selection_basis"].strip()
        or spec["price_basis"] not in {"raw_price_return", "consistently_adjusted_price_return"}
    ):
        raise ValueError("declare source, selection basis and consistent price-return units")
    if type(spec["lead_seconds"]) is not int or not 1 <= spec["lead_seconds"] <= 3600:
        raise ValueError("lead_seconds must be an integer in [1, 3600]")
    sessions = spec["sessions"]
    if not isinstance(sessions, list) or not 2 <= len(sessions) <= 10000:
        raise ValueError("freeze 2..10000 ordered session windows")
    prior = now
    schedule = []
    for session in sessions:
        if set(session) != {"signal_time", "execution_time"}:
            raise ValueError("each session requires signal_time and execution_time")
        signal, execution = _time(session["signal_time"]), _time(session["execution_time"])
        if (
            not prior < signal < execution - timedelta(seconds=spec["lead_seconds"])
            or execution.date() <= signal.date()
        ):
            raise ValueError(
                "session windows must be future, nonoverlapping and allow decision lead time"
            )
        schedule.append(
            {"signal_time": signal.isoformat(), "execution_time": execution.isoformat()}
        )
        prior = execution
    history = max(candidate.required_history, benchmark.required_history)
    if not isinstance(bootstrap, list) or not history <= len(bootstrap) <= 1000:
        raise ValueError("bootstrap must contain sufficient bounded close history")
    bars = [_bar(b, len(assets), now) for b in bootstrap]
    if any(
        _time(a["event_time"]).date() >= _time(b["event_time"]).date()
        for a, b in zip(bars, bars[1:], strict=False)
    ):
        raise ValueError(
            "bootstrap must contain at most one close per UTC date, in increasing order"
        )
    if bars[-1]["event_time"] >= schedule[0]["signal_time"]:
        raise ValueError("bootstrap overlaps the forward schedule")
    calibration = spec["calibration"]
    if set(calibration) != {"end_time", "source", "net_differences"}:
        raise ValueError("calibration requires provenance, end_time and matched net differences")
    if (
        _time(calibration["end_time"]) >= now
        or not isinstance(calibration["source"], str)
        or not calibration["source"].strip()
    ):
        raise ValueError("calibration must precede freeze and declare provenance")
    plan = evidence_plan(calibration["net_differences"], **spec["evidence"])
    manifest = {
        "schema": 1,
        "spec": {**spec, "sessions": schedule},
        "bootstrap": bars,
        "strategy": asdict(candidate),
        "benchmark": asdict(benchmark),
        "execution": asdict(config),
        "history": history,
        "evidence_plan": plan,
        "identity": _identity(),
        "created_at": now.isoformat(),
        "external_timestamp_verified": False,
        "live_pnl_claim": False,
    }
    shadow_journal.create(path, manifest, _initial(manifest), now.isoformat())
    events, _ = shadow_journal.reconcile(path, rebuild)
    return {
        "head_sha256": events[-1]["hash"],
        "manifest_sha256": digest(manifest),
        "evidence_plan": plan,
        "schedule_covers_plan": len(schedule) >= plan["required_sessions"],
        "classification": "local_forward_simulated",
        "external_timestamp_verified": False,
    }


def _initial(manifest: dict[str, Any]) -> dict[str, Any]:
    n = len(manifest["spec"]["assets"])
    capital = manifest["execution"]["initial_nav"]
    return {
        "cursor": 0,
        "pending": None,
        "bars": manifest["bootstrap"],
        "missed": 0,
        "failures": 0,
        "books": {
            s["name"]: {
                "cash": capital,
                "shares": [0.0] * n,
                "last_nav": capital,
                "previous_time": None,
                "previous_prices": [0.0] * n,
            }
            for s in (manifest["strategy"], manifest["benchmark"])
        },
    }


def _plans(
    manifest: dict[str, Any], state: dict[str, Any], bar: dict[str, Any], missed: bool
) -> dict[str, Any]:
    assets = manifest["spec"]["assets"]
    config = ReplayConfig(**manifest["execution"])
    bars = [*state["bars"], bar]
    closes = np.array([b["close"] for b in bars], dtype=float)
    volumes = np.array([b["volume"] for b in bars], dtype=float)
    panel = MarketPanel(
        [_time(b["event_time"]) for b in bars],
        assets,
        closes,
        np.full_like(closes, np.nan),
        volumes,
        np.isfinite(closes),
    )
    i = len(bars) - 1
    eligible, adv, sigma = _universe(panel, i, manifest["history"], config)
    terminal = state["cursor"] == len(manifest["spec"]["sessions"]) - 1
    plans = {}
    for raw in (manifest["strategy"], manifest["benchmark"]):
        strategy = _strategy(raw)
        book = state["books"][strategy.name]
        shares = np.array(book["shares"])
        held = shares != 0
        diagnostic: dict[str, Any] = {}
        if missed:
            desired = shares.copy()  # Never manufacture a late order, including terminal exits.
        else:
            if not (np.isfinite(closes[i, held]) & (closes[i, held] > 0)).all():
                raise ValueError("held asset lacks an available signal-close valuation")
            nav = book["cash"] + float(shares[held] @ closes[i, held])
            if not np.isfinite(nav) or nav <= 0:
                raise ValueError("nonpositive signal NAV")
            target = (
                np.zeros(len(assets))
                if terminal
                else _weights(panel, i, eligible, strategy, config)
            )
            if strategy.allocation is not None and not terminal:
                previous = np.zeros(len(assets))
                previous[held] = shares[held] * closes[i, held] / nav
                target, diagnostic = _cost_weights(
                    panel, i, target, previous, adv, sigma, strategy, config, nav
                )
            desired = np.zeros(len(assets))
            active = target != 0
            desired[active] = target[active] * nav / closes[i, active]
        # JSON null preserves unavailable liquidity; the execution helper rejects affected orders.
        plans[strategy.name] = {
            "desired": desired.tolist(),
            "adv": _nullable(adv),
            "sigma": _nullable(sigma),
            "eligible_count": int(eligible.sum()),
            "terminal": terminal,
            "allocation": diagnostic,
            "orders": [
                {
                    "order_id": f"{state['cursor']}:{strategy.name}:{assets[j]}",
                    "security_id": assets[j],
                    "quantity": float(desired[j] - shares[j]),
                }
                for j in np.flatnonzero(np.abs(desired - shares) > 1e-12)
            ],
        }
    return plans


def _nullable(value: np.ndarray) -> list[float | None]:
    return [float(v) if np.isfinite(v) else None for v in value]


def _settle(
    manifest: dict[str, Any], state: dict[str, Any], prices: list[float | None]
) -> dict[str, Any]:
    pending = state["pending"]
    session = manifest["spec"]["sessions"][state["cursor"]]
    outputs = {}
    for name, book in state["books"].items():
        plan = pending["plans"][name]
        cash, shares, daily, fills, rejects = execute_orders(
            names=manifest["spec"]["assets"],
            desired=np.array(plan["desired"]),
            shares=np.array(book["shares"]),
            cash=book["cash"],
            prices=np.array(prices, dtype=float),
            adv=np.array(plan["adv"], dtype=float),
            sigma=np.array(plan["sigma"], dtype=float),
            signal_time=_time(session["signal_time"]),
            execution_time=_time(session["execution_time"]),
            previous_time=_time(book["previous_time"]) if book["previous_time"] else None,
            previous_prices=np.array(book["previous_prices"], dtype=float),
            last_nav=book["last_nav"],
            config=ReplayConfig(**manifest["execution"]),
            eligible_count=plan["eligible_count"],
            terminal=plan["terminal"],
        )
        for row in [*fills, *rejects]:
            row["order_id"] = f"{state['cursor']}:{name}:{row['security_id']}"
        outputs[name] = {
            "daily": daily,
            "fills": fills,
            "rejections": rejects,
            "state": {
                "cash": cash,
                "shares": shares.tolist(),
                "last_nav": daily["nav"],
                "previous_time": session["execution_time"],
                "previous_prices": prices,
            },
        }
    return outputs


def rebuild(events: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = events[0]["payload"]
    if manifest["identity"] != _identity():
        raise ValueError("frozen shadow code/runtime changed; restore the original environment")
    state = _initial(manifest)
    last_recorded = events[0]["recorded_at"]
    for event in events[1:]:
        if event["recorded_at"] < last_recorded:
            raise ValueError("journal recording times moved backwards")
        last_recorded = event["recorded_at"]
        payload, kind = event["payload"], event["kind"]
        if kind == "failure":
            state["failures"] += 1
            continue
        if payload["session"] != state["cursor"]:
            raise ValueError("shadow session sequence mismatch")
        if kind in {"decision", "missed"}:
            if state["pending"] is not None:
                raise ValueError("previous decision is not settled")
            state["pending"] = payload
            state["bars"] = [*state["bars"], payload["bar"]][-max(manifest["history"] + 1, 2) :]
            state["missed"] += int(kind == "missed")
        elif kind == "settlement":
            if state["pending"] is None:
                raise ValueError("settlement has no previously committed decision")
            actual = _settle(manifest, state, payload["prices"])
            if canonical(actual) != canonical(payload["outcomes"]):
                raise ValueError("cash/position/fill reconciliation mismatch")
            state["books"] = {name: value["state"] for name, value in actual.items()}
            state["pending"] = None
            state["cursor"] += 1
        else:
            raise ValueError("unknown shadow event")
    return state


def record(path: Path, operation: str, request: dict[str, Any]) -> dict[str, Any]:
    if operation not in {"decide", "miss", "settle"} or type(request.get("session")) is not int:
        raise ValueError("valid operation and integer session are required")
    session_id = request["session"]
    key = f"{'settlement' if operation == 'settle' else 'decision'}:{session_id}"

    def build(
        events: list[dict[str, Any]], state: dict[str, Any]
    ) -> tuple[str, dict[str, Any], str]:
        manifest = events[0]["payload"]
        now = _now()
        if session_id != state["cursor"] or not 0 <= session_id < len(manifest["spec"]["sessions"]):
            raise ValueError(
                "process the next frozen session; completed sessions cannot be skipped"
            )
        session = manifest["spec"]["sessions"][session_id]
        signal, execution = _time(session["signal_time"]), _time(session["execution_time"])
        n = len(manifest["spec"]["assets"])
        if operation == "settle":
            if set(request) != {"session", "event_time", "available_time", "prices"}:
                raise ValueError(
                    "settlement requires session, event_time, available_time and prices"
                )
            if state["pending"] is None:
                raise ValueError(
                    "settlement requires a committed decision or explicit missed session"
                )
            if (
                _time(request["event_time"]) != execution
                or not execution <= _time(request["available_time"]) <= now
            ):
                raise ValueError("execution outcome is future-dated or belongs to another session")
            prices = _vector(request["prices"], n, positive=True, nullable=True)
            outcomes = _settle(manifest, state, prices)
            return (
                "settlement",
                {
                    "session": session_id,
                    "prices": prices,
                    "outcomes": outcomes,
                    "available_time": _stamp(request["available_time"]),
                },
                _now().isoformat(),
            )
        expected = {"session", "bar"} | ({"reason"} if operation == "miss" else set())
        if set(request) != expected or state["pending"] is not None:
            raise ValueError("invalid decision request or pending unsettled orders")
        bar = _bar(request["bar"], n, now)
        if _time(bar["event_time"]) != signal:
            raise ValueError("close snapshot does not match the scheduled signal session")
        if operation == "miss":
            if (
                now < execution
                or not isinstance(request["reason"], str)
                or not request["reason"].strip()
            ):
                raise ValueError("missed decisions require elapsed execution time and a reason")
        elif not signal <= now < execution - timedelta(seconds=manifest["spec"]["lead_seconds"]):
            raise ValueError("decision window is closed; record an explicit missed session")
        plans = _plans(manifest, state, bar, operation == "miss")
        finished = _now()
        if operation == "decide" and finished >= execution - timedelta(
            seconds=manifest["spec"]["lead_seconds"]
        ):
            raise ValueError("decision computation exceeded its deadline")
        return (
            ("missed" if operation == "miss" else "decision"),
            {
                "session": session_id,
                "bar": bar,
                "plans": plans,
                "reason": request.get("reason"),
                "missed": operation == "miss",
                "decision_deadline": (
                    execution - timedelta(seconds=manifest["spec"]["lead_seconds"])
                ).isoformat(),
            },
            finished.isoformat(),
        )

    def commit_guard(event: dict[str, Any]) -> None:
        now = _now()
        if now < _time(event["recorded_at"]):
            raise ValueError("local clock moved backwards before commit")
        if event["kind"] == "decision" and now >= _time(event["payload"]["decision_deadline"]):
            raise ValueError("decision exceeded its deadline before commit")

    wrapped = {"operation": operation, "input": request}
    try:
        return shadow_journal.transact(path, key, wrapped, rebuild, build, commit_guard)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        # Failed work has no portfolio effect, but keep the attempt visible when the journal is healthy.
        error = str(exc)
        with suppress(ValueError, sqlite3.Error):
            shadow_journal.transact(
                path,
                f"failure:{uuid4().hex}",
                wrapped,
                rebuild,
                lambda events, state: (
                    "failure",
                    {
                        "operation": operation,
                        "session": session_id,
                        "error": error,
                        "request": request,
                    },
                    _now().isoformat(),
                ),
            )
        raise


def report(path: Path, *, repair: bool = False, expected_head: str | None = None) -> dict[str, Any]:
    events, state = shadow_journal.reconcile(
        path, rebuild, repair=repair, expected_head=expected_head
    )
    manifest = events[0]["payload"]
    candidate, benchmark = manifest["strategy"]["name"], manifest["benchmark"]["name"]
    settled = [e["payload"]["outcomes"] for e in events if e["kind"] == "settlement"]
    differences = np.array(
        [o[candidate]["daily"]["net_return"] - o[benchmark]["daily"]["net_return"] for o in settled]
    )
    plan = manifest["evidence_plan"]
    complete = state["cursor"] == len(manifest["spec"]["sessions"])
    flat = all(np.max(np.abs(b["shares"])) < 1e-10 for b in state["books"].values())
    eligible = (
        complete and flat and state["missed"] == 0 and len(differences) >= plan["required_sessions"]
    )
    inference: dict[str, Any] = {"status": "not_final_or_insufficient_or_missed", "p_value": None}
    if eligible:
        try:
            variance = long_run_variance(differences, plan["lag"])
            z = float(differences.mean() / np.sqrt(variance / len(differences)))
            inference = {
                "status": "fixed_horizon_approximation",
                "p_value": float(norm.sf(z)),
                "long_run_variance": variance,
            }
        except ValueError:
            inference = {"status": "degenerate", "p_value": None}
    overdue = sum(
        _time(s["execution_time"]) <= _now()
        for s in manifest["spec"]["sessions"][state["cursor"] :]
    )
    return {
        "head_sha256": events[-1]["hash"],
        "manifest_sha256": digest(manifest),
        "events": events,
        "state": state,
        "reconciled": True,
        "completed_sessions": len(settled),
        "overdue_unsettled_sessions": overdue,
        "schedule_complete": complete,
        "terminal_liquidation_complete": flat and complete,
        "evidence_plan": plan,
        "mean_daily_net_difference": float(differences.mean()) if len(differences) else None,
        "inference": inference,
        "classification": "local_forward_simulated",
        "external_timestamp_verified": False,
        "source_availability_verified": False,
        "live_pnl_claim": False,
        "promote": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("freeze")
    prepare.add_argument("--spec", type=Path, required=True)
    prepare.add_argument("--bootstrap", type=Path, required=True)
    prepare.add_argument("--run", type=Path, required=True)
    for command in ("decide", "miss", "settle"):
        sub = commands.add_parser(command)
        sub.add_argument("--run", type=Path, required=True)
        sub.add_argument("--input", type=Path, required=True)
    check = commands.add_parser("reconcile")
    check.add_argument("--run", type=Path, required=True)
    check.add_argument("--repair", action="store_true")
    check.add_argument("--expected-head")
    check.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            result = freeze(
                json.loads(args.spec.read_text()), json.loads(args.bootstrap.read_text()), args.run
            )
        elif args.command == "reconcile":
            result = report(args.run, repair=args.repair, expected_head=args.expected_head)
            if args.output:
                with args.output.open("x") as handle:
                    handle.write(json.dumps(result, indent=2, allow_nan=False))
            result = {k: v for k, v in result.items() if k not in {"events", "state"}}
        else:
            event = record(args.run, args.command, json.loads(args.input.read_text()))
            result = {k: event[k] for k in ("seq", "hash", "kind", "recorded_at")}
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, TypeError, OSError, sqlite3.Error) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

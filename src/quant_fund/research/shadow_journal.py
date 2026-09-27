"""Atomic, hash-chained local journal with a rebuildable state projection.

A local hash chain detects accidental edits against a retained head. It is not
an external timestamp, signature or defense against a privileged full rewrite.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True, timeout=30)
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def read_events(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT seq, event_key, body, hash FROM events ORDER BY seq"
    ).fetchall()
    previous = "0" * 64
    events = []
    for expected, (seq, event_key, body, checksum) in enumerate(rows, 1):
        value = json.loads(body)
        if (
            value["key"] != event_key
            or seq != expected
            or value["seq"] != seq
            or value["previous"] != previous
            or digest(value) != checksum
        ):
            raise ValueError("shadow journal hash/sequence mismatch")
        events.append({**value, "hash": checksum})
        previous = checksum
    if not events or events[0]["kind"] != "freeze":
        raise ValueError("shadow journal has no frozen manifest")
    return events


def _append(
    connection: sqlite3.Connection,
    events: list[dict[str, Any]],
    key: str,
    kind: str,
    payload: dict[str, Any],
    recorded_at: str,
) -> dict[str, Any]:
    value = {
        "seq": len(events) + 1,
        "previous": events[-1]["hash"] if events else "0" * 64,
        "key": key,
        "kind": kind,
        "recorded_at": recorded_at,
        "payload": payload,
    }
    checksum = digest(value)
    connection.execute(
        "INSERT INTO events(seq, event_key, body, hash) VALUES (?, ?, ?, ?)",
        (value["seq"], key, canonical(value), checksum),
    )
    return {**value, "hash": checksum}


def _save_state(connection: sqlite3.Connection, head: str, state: dict[str, Any]) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO projection(id, head, state) VALUES (1, ?, ?)",
        (head, canonical(state)),
    )


def create(path: Path, manifest: dict[str, Any], state: dict[str, Any], timestamp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation: never overwrite another experiment, even after a failed freeze.
    with path.open("xb"):
        pass
    with closing(connect(path)) as connection, connection:
        connection.executescript("""
            CREATE TABLE events(seq INTEGER PRIMARY KEY, event_key TEXT UNIQUE NOT NULL,
                                body TEXT NOT NULL, hash TEXT NOT NULL);
            CREATE TABLE projection(id INTEGER PRIMARY KEY CHECK(id=1), head TEXT, state TEXT);
            CREATE TRIGGER no_event_update BEFORE UPDATE ON events BEGIN
                SELECT RAISE(ABORT, 'events are immutable'); END;
            CREATE TRIGGER no_event_delete BEFORE DELETE ON events BEGIN
                SELECT RAISE(ABORT, 'events are immutable'); END;
        """)
        event = _append(connection, [], "freeze", "freeze", manifest, timestamp)
        _save_state(connection, event["hash"], state)


Reducer = Callable[[list[dict[str, Any]]], dict[str, Any]]
Builder = Callable[[list[dict[str, Any]], dict[str, Any]], tuple[str, dict[str, Any], str]]


def transact(
    path: Path,
    key: str,
    request: dict[str, Any],
    reducer: Reducer,
    builder: Builder,
    commit_guard: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Serialize both books and their cursor in one durable transaction.

    Identical retries return the original result. Conflicting retries fail.
    Builder failures leave the transaction unchanged; the caller records an
    explicit failed-attempt event separately when appropriate.
    """
    with closing(connect(path)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        events = read_events(connection)
        state = reducer(events)
        projection = connection.execute("SELECT head, state FROM projection WHERE id=1").fetchone()
        if projection != (events[-1]["hash"], canonical(state)):
            raise ValueError("shadow projection mismatch; run reconcile --repair")
        old = next((e for e in events if e["key"] == key), None)
        if old is not None:
            if old["payload"].get("request_sha256") != digest(request):
                raise ValueError("conflicting retry for committed event")
            return old
        kind, payload, timestamp = builder(events, state)
        if timestamp < events[-1]["recorded_at"]:
            raise ValueError("local clock moved backwards")
        payload = {**payload, "request_sha256": digest(request)}
        event = _append(connection, events, key, kind, payload, timestamp)
        rebuilt = reducer([*events, event])
        _save_state(connection, event["hash"], rebuilt)
        if commit_guard is not None:
            commit_guard(event)
        return event


def reconcile(
    path: Path, reducer: Reducer, *, repair: bool = False, expected_head: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with closing(connect(path)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE" if repair else "BEGIN")
        events = read_events(connection)
        if expected_head is not None and events[-1]["hash"] != expected_head:
            raise ValueError("journal differs from retained external head")
        state = reducer(events)
        projection = connection.execute("SELECT head, state FROM projection WHERE id=1").fetchone()
        if projection != (events[-1]["hash"], canonical(state)):
            if not repair:
                raise ValueError("shadow projection mismatch; run reconcile --repair")
            _save_state(connection, events[-1]["hash"], state)
        return events, state

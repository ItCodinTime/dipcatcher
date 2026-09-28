"""Point-in-time index membership and price-coverage reporting.

Membership files follow the ``coiltrade/point-in-time-sp500`` shape vendored
at ``research/reality/survivorship/membership.json``: a ``current`` ticker
list plus a newest-first ``changes`` log. Coverage is the fraction of
members on a session that have at least one real price bar that day.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl


def load_membership(path: Path | str) -> dict[str, Any]:
    """Load a membership snapshot and refuse a file that is not newest-first."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("membership file must be a JSON object")
    changes = payload.get("changes")
    current = payload.get("current")
    if not isinstance(changes, list) or not isinstance(current, list):
        raise ValueError("membership file needs current and changes")
    dates = [str(event["date"]) for event in changes]
    if dates != sorted(dates, reverse=True):
        raise ValueError("membership changes must be newest-first")
    return payload


def members_asof(data: dict[str, Any], day: str) -> set[str]:
    """Index members at the close of ``day``. Changes dated ``day`` are in effect.

    Walks the change log newest-first and stops at the first event on or
    before ``day``. Newer events are undone. This does not look at prices.
    """
    date.fromisoformat(day)
    members = {str(symbol) for symbol in data["current"]}
    for event in data["changes"]:
        if str(event["date"]) <= day:
            break
        added = str(event.get("added") or "")
        removed = str(event.get("removed") or "")
        if added:
            members.discard(added)
        if removed:
            members.add(removed)
    return members


def _ny_session_dates(frame: pl.DataFrame) -> pl.DataFrame:
    if "event_time" not in frame.columns:
        raise ValueError("bars need an event_time column")
    return frame.with_columns(
        pl.col("event_time").dt.convert_time_zone("America/New_York").dt.date().alias("ny_date")
    )


def session_on_or_after(sessions: list[date], anchor: date) -> date | None:
    return next((day for day in sessions if day >= anchor), None)


def membership_price_coverage(
    bars: pl.DataFrame,
    membership: dict[str, Any],
    anchors: list[str],
    *,
    security_col: str = "security_id",
) -> list[dict[str, Any]]:
    """Fraction of index members with a real bar on the first session ≥ each anchor.

    ``bars`` may combine multiple sources (Yahoo + DoltHub). Duplicate
    ``(security_id, session)`` rows count once. Empty membership on a session
    yields coverage 0.0.
    """
    if security_col not in bars.columns:
        raise ValueError(f"bars need a {security_col!r} column")
    if bars.is_empty():
        sessions: list[date] = []
        stamped = bars
    else:
        stamped = _ny_session_dates(bars)
        sessions = sorted(
            day for day in stamped["ny_date"].unique().to_list() if isinstance(day, date)
        )
    reports: list[dict[str, Any]] = []
    for anchor in anchors:
        start = date.fromisoformat(anchor)
        session = session_on_or_after(sessions, start)
        if session is None:
            # No panel session yet — still report membership size with zero bars.
            members = members_asof(membership, anchor)
            reports.append(
                {
                    "anchor": anchor,
                    "session": None,
                    "members": len(members),
                    "with_bar": 0,
                    "coverage": 0.0,
                    "n_missing": len(members),
                    "missing": sorted(members),
                }
            )
            continue
        members = members_asof(membership, session.isoformat())
        have = {
            str(symbol)
            for symbol in stamped.filter(pl.col("ny_date") == session)[security_col]
            .cast(pl.String)
            .to_list()
        }
        covered = members & have
        missing = sorted(members - have)
        reports.append(
            {
                "anchor": anchor,
                "session": session.isoformat(),
                "members": len(members),
                "with_bar": len(covered),
                "coverage": (len(covered) / len(members)) if members else 0.0,
                "n_missing": len(missing),
                "missing": missing,
            }
        )
    return reports


def merge_bar_panels(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Union bar panels; last source wins on duplicate ``(security_id, event_time)``."""
    nonempty = [frame for frame in frames if frame is not None and not frame.is_empty()]
    if not nonempty:
        return pl.DataFrame()
    return (
        pl.concat(nonempty, how="diagonal_relaxed")
        .unique(subset=["security_id", "event_time"], keep="last")
        .sort(["security_id", "event_time"])
    )


def coverage_report_dict(
    rows: list[dict[str, Any]],
    *,
    sources: list[str],
    membership_path: str | None = None,
    membership_sha256: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """JSON-serializable coverage receipt (missing lists included)."""
    public_rows = []
    for row in rows:
        public_rows.append(
            {
                "anchor": row["anchor"],
                "session": row["session"],
                "members": row["members"],
                "with_bar": row["with_bar"],
                "coverage": row["coverage"],
                "n_missing": row["n_missing"],
                "missing": row.get("missing", []),
            }
        )
    payload: dict[str, Any] = {
        "schema": "dipcatcher.membership_price_coverage.v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "sources": sources,
        "membership_path": membership_path,
        "membership_sha256": membership_sha256,
        "coverage": public_rows,
        "mean_coverage": (
            sum(float(r["coverage"]) for r in public_rows) / len(public_rows)
            if public_rows
            else 0.0
        ),
    }
    if extra:
        payload.update(extra)
    return payload

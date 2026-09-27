"""Frozen XNYS sessions are an infrastructure check, not market-data evidence."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from quant_fund.paper.xnys_calendar import (
    XnysSchedule,
    materialize_xnys_schedule,
    verify_xnys_schedule,
)


@pytest.fixture(scope="module")
def schedule() -> XnysSchedule:
    return materialize_xnys_schedule("2026-10-29", "2026-12-28")


def test_official_2026_hours_dst_and_early_closes(schedule: XnysSchedule) -> None:
    # NYSE regular session 09:30-16:00 ET, including November DST transition.
    assert schedule.session("2026-10-30").open_utc.isoformat() == "2026-10-30T13:30:00+00:00"
    assert schedule.session("2026-10-30").close_utc.isoformat() == "2026-10-30T20:00:00+00:00"
    assert schedule.session("2026-11-02").open_utc.isoformat() == "2026-11-02T14:30:00+00:00"
    assert schedule.session("2026-11-02").close_utc.isoformat() == "2026-11-02T21:00:00+00:00"
    assert schedule.session("2026-11-27").close_utc.isoformat() == "2026-11-27T18:00:00+00:00"
    assert schedule.session("2026-12-24").close_utc.isoformat() == "2026-12-24T18:00:00+00:00"
    with pytest.raises(ValueError, match="not a scheduled"):
        schedule.session("2026-11-26")  # Thanksgiving
    with pytest.raises(ValueError, match="not a scheduled"):
        schedule.session("2026-12-25")  # Christmas


def test_exact_packet_event_times_reject_shifted_self_consistent_bars(
    schedule: XnysSchedule,
) -> None:
    assert schedule.validate_close("2026-10-30T20:00:00Z").day.isoformat() == "2026-10-30"
    assert schedule.validate_open("2026-11-02T14:30:00Z").day.isoformat() == "2026-11-02"
    assert (
        schedule.validate_next_open("2026-10-30T20:00:00Z", "2026-11-02T14:30:00Z").day.isoformat()
        == "2026-11-02"
    )
    with pytest.raises(ValueError, match="scheduled XNYS close"):
        schedule.validate_close("2026-10-30T21:00:00Z")
    with pytest.raises(ValueError, match="scheduled XNYS open"):
        schedule.validate_open("2026-11-02T13:30:00Z")
    with pytest.raises(ValueError, match="explicit UTC offset"):
        schedule.validate_close("2026-10-30T20:00:00")
    with pytest.raises(ValueError, match="immediately next"):
        schedule.validate_next_open("2026-10-30T20:00:00Z", "2026-11-03T14:30:00Z")


def test_missing_sessions_and_caller_closures(schedule: XnysSchedule) -> None:
    assert schedule.missing_sessions("2026-11-25", "2026-11-27") == ()
    schedule.validate_gap("2026-11-25", "2026-11-27", ["2026-11-26"])
    assert schedule.missing_sessions("2026-11-25", "2026-11-30") == ("2026-11-27",)
    with pytest.raises(ValueError, match="missing scheduled XNYS sessions: 2026-11-27"):
        schedule.validate_gap("2026-11-25", "2026-11-30", ["2026-11-26", "2026-11-27"])
    with pytest.raises(ValueError, match="market_closed claims"):
        schedule.validate_gap("2026-11-25", "2026-11-27", [])
    with pytest.raises(ValueError, match="market_closed claims"):
        schedule.validate_gap("2026-11-25", "2026-11-27", ["2026-11-26", "2026-11-27"])
    schedule.validate_gap("2026-10-30", "2026-11-02", [])


def test_schedule_receipt_is_deterministic_and_not_external_attestation(
    schedule: XnysSchedule,
) -> None:
    receipt = schedule.as_dict()
    assert receipt["package_version"] == "4.13.2"
    assert receipt["official_comparison_verified"] is False
    assert receipt["forward_evidence_accepted"] is False
    assert verify_xnys_schedule(receipt).as_dict() == receipt
    assert (
        materialize_xnys_schedule("2026-10-29", "2026-12-28").schedule_sha256
        == receipt["schedule_sha256"]
    )
    altered = deepcopy(receipt)
    altered["sessions"][1]["close_utc"] = "2026-10-30T21:00:00+00:00"
    core = {key: value for key, value in altered.items() if key != "schedule_sha256"}
    altered["schedule_sha256"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="differs from pinned package"):
        verify_xnys_schedule(altered)
    altered = deepcopy(receipt)
    altered["official_comparison_verified"] = True
    with pytest.raises(ValueError, match="honesty status"):
        verify_xnys_schedule(altered)


def test_invalid_bounds_fail_closed() -> None:
    with pytest.raises(ValueError, match="first_date < last_date"):
        materialize_xnys_schedule("2026-11-30", "2026-11-01")

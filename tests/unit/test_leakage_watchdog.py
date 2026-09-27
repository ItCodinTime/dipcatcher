"""Runtime watchdog tests (DESIGN.md §6.3): future known_at read MUST trip."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_fund.leakage import LeakageError, LeakageWatchdog
from quant_fund.leakage.watchdog import MAX_KNOWN_AT_PARAM
from quant_fund.proofcore.contracts import DataAccessRecord
from quant_fund.proofcore.contracts import LeakageError as ContractsLeakageError
from quant_fund.schemas.errors import LeakageError as SchemasLeakageError
from quant_fund.schemas.errors import PointInTimeError

T0 = datetime(2026, 1, 10, 12, 0, tzinfo=UTC)


def _read(max_known_at: datetime | None, *, dataset: str = "silver/bars") -> DataAccessRecord:
    params = {} if max_known_at is None else {MAX_KNOWN_AT_PARAM: max_known_at.isoformat()}
    return DataAccessRecord(
        dataset=dataset,
        asof_utc=T0.isoformat(),
        params=params,
        rows=10,
        content_sha256="ab" * 32,
    )


def test_future_known_at_read_raises() -> None:
    """The core guarantee: a read whose max known_at exceeds the decision time trips."""
    watchdog = LeakageWatchdog()
    future = datetime(2026, 1, 11, 12, 0, tzinfo=UTC)
    with pytest.raises(LeakageError) as excinfo:
        watchdog.observe(_read(future), T0)
    msg = str(excinfo.value)
    assert "silver/bars" in msg and future.isoformat() in msg and T0.isoformat() in msg


def test_clean_read_passes_and_watermark_tracks() -> None:
    watchdog = LeakageWatchdog()
    past = datetime(2026, 1, 9, 0, 0, tzinfo=UTC)
    watchdog.observe(_read(past), T0)
    assert watchdog.watermark() == past
    earlier = datetime(2026, 1, 8, 0, 0, tzinfo=UTC)
    watchdog.observe(_read(earlier), T0)
    assert watchdog.watermark() == past  # watermark is monotone
    assert watchdog.n_observed == 2


def test_boundary_known_at_equals_decision_time_passes() -> None:
    watchdog = LeakageWatchdog()
    watchdog.observe(_read(T0), T0)
    assert watchdog.watermark() == T0


def test_naive_decision_time_fails_closed() -> None:
    watchdog = LeakageWatchdog()
    with pytest.raises(LeakageError, match="tz-aware"):
        watchdog.observe(_read(T0), datetime(2026, 1, 10, 12, 0))


def test_missing_watermark_param_strict_raises() -> None:
    watchdog = LeakageWatchdog(strict=True)
    with pytest.raises(LeakageError, match="unverifiable read"):
        watchdog.observe(_read(None), T0)


def test_missing_watermark_param_non_strict_skips() -> None:
    watchdog = LeakageWatchdog(strict=False)
    watchdog.observe(_read(None), T0)  # no trip, no watermark
    assert watchdog.watermark() is None
    assert watchdog.n_observed == 0


def test_malformed_watermark_param_raises() -> None:
    watchdog = LeakageWatchdog()
    read = DataAccessRecord(
        dataset="silver/bars",
        asof_utc=T0.isoformat(),
        params={MAX_KNOWN_AT_PARAM: "not-a-timestamp"},
        rows=1,
        content_sha256="cd" * 32,
    )
    with pytest.raises(LeakageError, match="malformed"):
        watchdog.observe(read, T0)


def test_naive_watermark_param_fails_closed() -> None:
    watchdog = LeakageWatchdog()
    read = DataAccessRecord(
        dataset="silver/bars",
        asof_utc=T0.isoformat(),
        params={MAX_KNOWN_AT_PARAM: "2026-01-09 00:00:00"},
        rows=1,
        content_sha256="ef" * 32,
    )
    with pytest.raises(LeakageError, match="tz-aware"):
        watchdog.observe(read, T0)


def test_error_taxonomy_bridges_both_roots() -> None:
    """§8.3: leakage.LeakageError catches via contracts AND schemas roots."""
    assert issubclass(LeakageError, ContractsLeakageError)
    assert issubclass(LeakageError, SchemasLeakageError)
    assert issubclass(LeakageError, PointInTimeError)


def test_watermark_none_before_any_read() -> None:
    assert LeakageWatchdog().watermark() is None

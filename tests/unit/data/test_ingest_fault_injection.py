"""Fault-injection tests for the bronze-ingest validation layer.

Vendor data is untrusted: every malformed OHLCV batch must fail CLOSED through
``normalize_ohlcv`` / ``pit_frame`` (``SourceError``) or the PIT gates
(``PointInTimeError``) — never pass silently into a "valid" frame. Each test
below targets a documented rejection branch in
``src/quant_fund/data/sources/normalize.py`` and ``data/sources/base.py``.

Known accepted-by-design inputs (not faults): numeric strings coerce via
``float()``, naive timestamps are assumed UTC by ``parse_time``, and row order
is normalized by the final sort. Absurd magnitudes above
``_MAX_OHLCV_MAGNITUDE`` and empty payloads raise ``SourceError`` (both
chaos-tested below).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

from quant_fund.data.point_in_time import (
    PIT_COLS,
    require_pit_columns,
    validate_feature_frame,
)
from quant_fund.data.sources.base import SourceError
from quant_fund.data.sources.normalize import csv_rows, normalize_observations, normalize_ohlcv
from quant_fund.schemas.errors import PointInTimeError

PRICE_FIELDS = ("open", "high", "low", "close")
BASE_TIME = datetime(2018, 1, 1, tzinfo=UTC)

_POSITIVE_FLOAT = st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False)
_NONNEG_FLOAT = st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False)
_NONFINITE = st.sampled_from([float("nan"), float("inf"), float("-inf")])


def _envelope_prices(draw) -> dict[str, float]:
    low = draw(_POSITIVE_FLOAT)
    high = draw(st.floats(min_value=low, max_value=1e6, allow_nan=False, allow_infinity=False))
    open_ = draw(st.floats(min_value=low, max_value=high, allow_nan=False, allow_infinity=False))
    close = draw(st.floats(min_value=low, max_value=high, allow_nan=False, allow_infinity=False))
    return {"open": open_, "high": high, "low": low, "close": close}


def _valid_row(draw, index: int) -> dict[str, object]:
    prices = _envelope_prices(draw)
    return {
        "security_id": draw(st.sampled_from(["AAA", "BBB", "CCC"])),
        "event_time": BASE_TIME + timedelta(minutes=index),
        **prices,
        "volume": draw(_NONNEG_FLOAT),
    }


@composite
def valid_batch(draw, min_size: int = 1, max_size: int = 20) -> list[dict[str, object]]:
    """Envelope-consistent rows with distinct (security_id, event_time) keys."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    return [_valid_row(draw, i) for i in range(n)]


@composite
def _nonpositive(draw) -> float:
    return draw(st.sampled_from([0.0, -0.0001, -1.0, -1e6]))


def _is_finite_float_text(text: str) -> bool:
    try:
        return math.isfinite(float(text))
    except (TypeError, ValueError):
        return False


@settings(max_examples=60, deadline=None)
@given(batch=valid_batch(), field=st.sampled_from((*PRICE_FIELDS, "volume")), bad=_NONFINITE)
def test_non_finite_price_or_volume_fails_closed(
    batch: list[dict[str, object]], field: str, bad: float
) -> None:
    """NaN/inf/-inf in any OHLCV numeric field raises SourceError, never coerces."""
    batch[0][field] = bad
    with pytest.raises(SourceError, match="not finite"):
        normalize_ohlcv(batch, source="fault")


@settings(max_examples=60, deadline=None)
@given(
    batch=valid_batch(),
    field=st.sampled_from(PRICE_FIELDS),
    bad=_nonpositive(),
)
def test_non_positive_prices_fail_closed(
    batch: list[dict[str, object]], field: str, bad: float
) -> None:
    """Zero or negative OHLC prices raise SourceError."""
    batch[0][field] = bad
    with pytest.raises(SourceError, match="positive"):
        normalize_ohlcv(batch, source="fault")


@settings(max_examples=60, deadline=None)
@given(
    batch=valid_batch(),
    # "low" is excluded: an inflated low trips the high/low envelope check
    # first (still SourceError, different branch).
    field=st.sampled_from(("open", "high", "close", "volume")),
    magnitude=st.sampled_from([1.000001e12, 1e15, 1e300]),
)
def test_absurd_magnitudes_fail_closed(
    batch: list[dict[str, object]], field: str, magnitude: float
) -> None:
    """Values above _MAX_OHLCV_MAGNITUDE (unit-scaling/fat-finger) raise."""
    batch[0][field] = magnitude
    with pytest.raises(SourceError, match="magnitude exceeds plausible bound"):
        normalize_ohlcv(batch, source="fault")


@settings(max_examples=60, deadline=None)
@given(
    batch=valid_batch(),
    fault=st.sampled_from(
        [
            "high_below_low",
            "open_above_high",
            "open_below_low",
            "close_above_high",
            "close_below_low",
            "negative_volume",
        ]
    ),
)
def test_envelope_violations_fail_closed(batch: list[dict[str, object]], fault: str) -> None:
    """high < low, open/close outside [low, high], and volume < 0 all raise."""
    row = batch[0]
    if fault == "high_below_low":
        row["high"] = float(row["low"]) / 2.0
    elif fault == "open_above_high":
        row["open"] = float(row["high"]) + abs(float(row["high"])) + 1e-3
    elif fault == "open_below_low":
        row["open"] = float(row["low"]) / 2.0  # positive, but below low
    elif fault == "close_above_high":
        row["close"] = float(row["high"]) + abs(float(row["high"])) + 1e-3
    elif fault == "close_below_low":
        row["close"] = float(row["low"]) / 2.0
    else:
        row["volume"] = -1.0
    with pytest.raises(SourceError, match="envelope|within low/high"):
        normalize_ohlcv(batch, source="fault")


@settings(max_examples=60, deadline=None)
@given(batch=valid_batch(min_size=2, max_size=20), perturb_prices=st.booleans())
def test_duplicate_security_time_keys_fail_closed(
    batch: list[dict[str, object]], perturb_prices: bool
) -> None:
    """A repeated (event_time, security_id) key raises even if prices differ."""
    dupe = dict(batch[0])
    if perturb_prices:
        dupe["close"] = float(dupe["close"]) * 1.5 + 1e-3
    batch.append(dupe)
    with pytest.raises(SourceError, match="duplicate OHLCV key"):
        normalize_ohlcv(batch, source="fault")


_NONNUMERIC_TEXT = st.text(min_size=1).filter(lambda s: not _is_finite_float_text(s))


@settings(max_examples=60, deadline=None)
@given(
    batch=valid_batch(),
    field=st.sampled_from((*PRICE_FIELDS, "volume")),
    bad=st.one_of(_NONNUMERIC_TEXT, st.none(), st.binary(min_size=1)),
)
def test_non_numeric_fields_fail_closed(
    batch: list[dict[str, object]], field: str, bad: object
) -> None:
    """Non-numeric vendor values (garbage strings, None, bytes) raise SourceError."""
    batch[0][field] = bad
    with pytest.raises(SourceError, match="not numeric"):
        normalize_ohlcv(batch, source="fault")


@settings(max_examples=60, deadline=None)
@given(
    batch=valid_batch(),
    dropped=st.sampled_from(["security_id", "event_time", *PRICE_FIELDS, "volume"]),
)
def test_missing_required_fields_fail_closed(batch: list[dict[str, object]], dropped: str) -> None:
    """Rows missing security_id/event_time/a price column raise SourceError."""
    del batch[0][dropped]
    with pytest.raises(SourceError, match="no security_id|no event_time|not numeric"):
        normalize_ohlcv(batch, source="fault")


@settings(max_examples=40, deadline=None)
@given(
    batch=valid_batch(max_size=5),
    fault=st.sampled_from(["available_before_event", "available_in_future"]),
)
def test_impossible_pit_chain_fails_closed(batch: list[dict[str, object]], fault: str) -> None:
    """available_time before event_time or after ingest time raises SourceError."""
    if fault == "available_before_event":
        batch[0]["available_time"] = BASE_TIME - timedelta(days=3650)
    else:
        batch[0]["available_time"] = datetime.now(UTC) + timedelta(days=3650)
    with pytest.raises(SourceError, match="impossible event_time"):
        normalize_ohlcv(batch, source="fault")


@settings(max_examples=40, deadline=None)
@given(batch=valid_batch())
def test_valid_batches_pass_and_output_is_sorted(batch: list[dict[str, object]]) -> None:
    """Control: envelope-consistent batches pass, carry PIT columns, and are sorted."""
    frame = normalize_ohlcv(batch, source="control")
    assert frame.height == len(batch)
    for col in ("event_time", "available_time", "ingested_time", "source", "security_id"):
        assert col in frame.columns
    keys = list(zip(frame["security_id"].to_list(), frame["event_time"].to_list(), strict=True))
    assert keys == sorted(keys)


@settings(max_examples=40, deadline=None)
@given(
    low=_POSITIVE_FLOAT,
    spread=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
def test_naive_timestamps_assumed_utc(low: float, spread: float) -> None:
    """Documented handling: naive vendor timestamps coerce to aware UTC."""
    row = {
        "security_id": "AAA",
        "event_time": datetime(2018, 1, 1),
        "open": low,
        "high": low + spread,
        "low": low,
        "close": low + spread / 2.0,
        "volume": 1.0,
    }
    frame = normalize_ohlcv([row], source="control")
    assert frame["event_time"][0].tzinfo is not None
    assert frame["event_time"][0] == datetime(2018, 1, 1, tzinfo=UTC)


@settings(max_examples=40, deadline=None)
@given(
    present=st.lists(st.sampled_from(PIT_COLS), unique=True).filter(
        lambda cols: len(cols) < len(PIT_COLS)
    )
)
def test_frames_missing_pit_columns_fail_closed(present: list[str]) -> None:
    """A bronze frame missing any PIT column is rejected with PointInTimeError."""
    frame = pl.DataFrame({col: [1] for col in present})
    with pytest.raises(PointInTimeError, match="missing PIT columns"):
        require_pit_columns(frame)


@settings(max_examples=40, deadline=None)
@given(
    n_valid=st.integers(min_value=0, max_value=5),
    fault=st.sampled_from(["null", "future"]),
)
def test_unobservable_available_time_fails_closed(n_valid: int, fault: str) -> None:
    """Null or future available_time relative to decision_time fails closed."""
    decision = BASE_TIME + timedelta(days=30)
    good = [BASE_TIME + timedelta(days=i) for i in range(n_valid)]
    if fault == "null":
        availabilities: list[datetime | None] = good + [None]
    else:
        availabilities = good + [decision + timedelta(days=1)]
    frame = pl.DataFrame({"available_time": availabilities})
    with pytest.raises(PointInTimeError, match="null or future"):
        validate_feature_frame(frame, decision)


def test_poison_row_among_999_good_rows_rejects_batch() -> None:
    """One bad row anywhere in a 1000-row batch rejects the whole batch."""
    good = []
    for i in range(999):
        t = BASE_TIME + timedelta(minutes=i)
        good.append(
            {
                "security_id": "AAA" if i % 2 == 0 else "BBB",
                "event_time": t,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000.0,
            }
        )
    poison = dict(good[500])
    poison["security_id"] = "CCC"  # unique key; only the price is corrupt
    poison["close"] = 5.0  # far below the low of 99.0
    with pytest.raises(SourceError, match="within low/high"):
        normalize_ohlcv(good + [poison], source="poison")


def test_poison_row_duplicate_timestamp_rejects_batch() -> None:
    """A duplicated (security_id, event_time) key anywhere rejects the batch."""
    good = [
        {
            "security_id": "AAA",
            "event_time": BASE_TIME + timedelta(minutes=i),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
        }
        for i in range(999)
    ]
    poison = dict(good[999 - 1])
    with pytest.raises(SourceError, match="duplicate OHLCV key"):
        normalize_ohlcv(good + [poison], source="poison")


def test_thousand_row_clean_batch_passes() -> None:
    """Control: a 1000-row clean batch ingests intact."""
    rows = [
        {
            "security_id": "AAA" if i % 2 == 0 else "BBB",
            "event_time": BASE_TIME + timedelta(minutes=i),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
        }
        for i in range(1000)
    ]
    frame = normalize_ohlcv(rows, source="control")
    assert frame.height == 1000


def test_empty_payload_fails_closed() -> None:
    """Empty payloads raise SourceError for both normalizers — never a frame."""
    with pytest.raises(SourceError, match="OHLCV payload is empty"):
        normalize_ohlcv([], source="fault")
    with pytest.raises(SourceError, match="observation payload is empty"):
        normalize_observations([], source="fault")


def test_headerless_csv_fails_closed() -> None:
    with pytest.raises(SourceError, match="no header"):
        csv_rows("")

"""sota_eval_native must validate bars with the canonical contract.

The script previously caught ImportError and fell back to an inlined
``validate_bars`` copy that silently dropped the canonical checks
(datetime dtype, nonnegative volume, OHLC consistency). Silent
substitution is forbidden: the module must re-export the canonical
validator or fail to import.
"""

from __future__ import annotations


def test_native_eval_uses_canonical_validate_bars() -> None:
    import scripts.sota_eval_native as native

    from quant_fund.research import sota_evidence

    assert native.validate_bars is sota_evidence.validate_bars


def test_canonical_validate_bars_rejects_non_datetime_and_bad_ohlc() -> None:
    """The checks the dropped fallback omitted must actually fire."""
    import polars as pl
    import pytest

    from quant_fund.research.sota_evidence import validate_bars

    rows = {
        "security_id": ["btcusdt"] * 3,
        "event_time": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "available_time": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "open": [1.0, 1.0, 1.0],
        "high": [1.0, 1.0, 1.0],
        "low": [1.0, 1.0, 1.0],
        "close": [1.0, 1.0, 1.0],
        "volume": [1.0, 1.0, 1.0],
    }
    frame = pl.DataFrame(rows)  # string timestamps must be rejected
    with pytest.raises(ValueError, match="datetime"):
        validate_bars(frame)

    frame = pl.DataFrame(rows).with_columns(
        pl.col("event_time").str.to_datetime(),
        pl.col("available_time").str.to_datetime(),
    )
    validate_bars(frame)  # sane frame passes

    negative_volume = frame.with_columns(pl.lit(-1.0).alias("volume"))
    with pytest.raises(ValueError, match="volume"):
        validate_bars(negative_volume)

    bad_ohlc = frame.with_columns(pl.lit(2.0).alias("low"))
    with pytest.raises(ValueError, match="inconsistent OHLC"):
        validate_bars(bad_ohlc)

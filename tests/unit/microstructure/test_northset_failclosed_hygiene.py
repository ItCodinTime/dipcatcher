"""Fail-closed hygiene: degenerate bars emit null, not 1e-12-scaled junk."""

from __future__ import annotations

import polars as pl

from quant_fund.northset.candles import candle_geometry
from quant_fund.northset.estimators import volume_over_range


def _bars(rows: list[tuple[float, float, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": ["A"] * len(rows),
            "event_time": list(range(len(rows))),
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [1000.0] * len(rows),
        }
    )


class TestVolumeOverRange:
    def test_flat_bar_is_null_not_inflated(self):
        out = volume_over_range(_bars([(10.0, 11.0, 9.0, 10.5), (10.5, 10.5, 10.5, 10.5)]))
        vor = out["volume_over_range"].to_list()
        assert vor[0] == 1000.0 / 2.0
        assert vor[1] is None  # flat bar: undefined, not 1e12

    def test_inverted_bar_null(self):
        out = volume_over_range(_bars([(10.0, 9.0, 11.0, 10.0)]))  # high < low
        assert out["volume_over_range"][0] is None


class TestCandleGeometry:
    def test_flat_bar_geometry_null(self):
        # Corrupt flat bar with close outside the (zero) range must not
        # flag marubozu on a 1e-12 denominator.
        out = candle_geometry(_bars([(10.0, 10.0, 10.0, 10.6)]))
        assert out["candle_marubozu"][0] is None
        assert out["close_location_value"][0] is None
        assert out["candle_body_frac"][0] is None

    def test_negative_range_null(self):
        out = candle_geometry(_bars([(10.0, 9.0, 11.0, 10.0)]))
        assert out["candle_body_frac"][0] is None
        assert out["candle_doji"][0] is None

    def test_valid_bar_flags(self):
        # Strong bullish marubozu-ish bar followed by engulfing setup.
        out = candle_geometry(_bars([(10.0, 10.9, 9.95, 10.9), (11.0, 11.5, 10.9, 11.4)]))
        assert out["candle_marubozu"][0] == 1.0
        assert out["candle_direction"][0] == 1.0
        # First bar has no previous bar: engulfing is undefined.
        assert out["candle_engulfing"][0] is None
        # Second bar: bullish body engulfing nothing (prev was also bull) → 0.
        assert out["candle_engulfing"][1] == 0.0

    def test_gap_null_when_prev_close_nonpositive(self):
        out = candle_geometry(_bars([(10.0, 11.0, 9.0, 10.5), (11.0, 12.0, 10.5, 11.5)]))
        assert out["candle_gap"][0] is None
        assert out["candle_gap"][1] is not None


class TestVolumeClockVpin:
    def test_remainder_carried_across_bucket(self):
        # buy=[60,140], sell=[60,0], bucket_volume=100:
        # row 1: 120 volume -> bucket tox |60-60|/100 = 0, carry 10/10
        # row 2: acc 150/10 -> bucket tox |90-10|/100 = 0.8, carry 60/0
        # Rolling mean of [0.0, 0.8] = 0.4. The old code discarded the
        # overflow, yielding a single inflated bucket (mean 1.0 at row 2).
        import numpy as np

        from quant_fund.northset.estimators import _volume_clock_vpin

        out = _volume_clock_vpin(
            np.array([60.0, 140.0]),
            np.array([60.0, 0.0]),
            bucket_volume=100.0,
            window=50,
        )
        assert out[0] == 0.0
        assert abs(out[1] - 0.4) < 1e-9

    def test_single_giant_row_fills_multiple_exact_buckets(self):
        # 300 volume in one row at bucket 100 -> three exact buckets, each
        # at the row's toxicity |b-s|/(b+s); nothing is dropped.
        import numpy as np

        from quant_fund.northset.estimators import _volume_clock_vpin

        out = _volume_clock_vpin(
            np.array([0.0, 250.0]),
            np.array([0.0, 50.0]),
            bucket_volume=100.0,
            window=50,
        )
        assert abs(out[1] - (200.0 / 300.0)) < 1e-9

    def test_nan_row_skipped_not_accumulated(self):
        import math

        import numpy as np

        from quant_fund.northset.estimators import _volume_clock_vpin

        out = _volume_clock_vpin(
            np.array([50.0, np.nan, 50.0]),
            np.array([0.0, 0.0, 0.0]),
            bucket_volume=100.0,
            window=50,
        )
        assert math.isnan(out[0])
        assert out[2] == 1.0

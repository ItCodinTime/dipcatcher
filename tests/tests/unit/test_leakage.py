"""Leakage detectors — fail-closed patterns (not a duplicate of the PIT sprawl suite).

These are intentional smoke/edge checks: future fundamentals, full-sample CS
normalization, boundary equality, same-bar feature=label, and unpurged label
overlap into the test window. Full PIT / purge / optuna coverage lives elsewhere.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from quant_fund.schemas.errors import PointInTimeError
from quant_fund.schemas.pit import assert_pit_safe
from quant_fund.validation.purging import overlaps, purge_mask


def test_future_fundamental_injection() -> None:
    decision = datetime(2020, 6, 1, tzinfo=UTC)
    filing_available = datetime(2020, 8, 1, tzinfo=UTC)
    with pytest.raises(PointInTimeError):
        assert_pit_safe(filing_available, decision)


def test_available_equal_decision_is_safe() -> None:
    """Boundary: available_time == decision_time is observable (not lookahead)."""
    t = datetime(2020, 6, 1, 16, 0, tzinfo=UTC)
    assert_pit_safe(t, t)  # must not raise


def test_future_cs_normalization_would_use_full_sample() -> None:
    """Full-sample z-score differs from per-date z-score — detector for leakage pattern."""
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    t1 = datetime(2020, 1, 3, tzinfo=UTC)
    df = pl.DataFrame({"event_time": [t0, t0, t1, t1], "x": [0.0, 1.0, 100.0, 101.0]})
    global_mean = df["x"].mean()
    per_date = df.group_by("event_time").agg(pl.col("x").mean().alias("m"))
    t0_mean = per_date.filter(pl.col("event_time") == t0)["m"][0]
    assert global_mean != t0_mean


def test_same_bar_feature_equals_label_is_leakage_pattern() -> None:
    """Using the label return as a feature yields perfect |corr| — classic leak detector."""
    rng = np.random.default_rng(42)
    label = rng.normal(0.0, 0.01, size=64)
    feature = label.copy()  # same-bar / target-in-features
    corr = float(np.corrcoef(feature, label)[0, 1])
    assert abs(corr) > 0.999


def test_unpurged_label_overlap_into_test_is_leakage() -> None:
    """Label horizon reaching into the test window must be purged (fail-closed).

    A naive keep-all train mask retains the overlapping pre-test decision; purge_mask
    drops it. overlaps() is True for that geometry.
    """
    bar = timedelta(days=1)
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    times = [t0 + i * bar for i in range(8)]
    # Decision at times[3] with horizon=2 reaches times[5] which is inside [test_start, test_end]
    test_start, test_end = times[5], times[6]
    assert overlaps(times[3], test_start, horizon_bars=2, bar_delta=bar) is True

    naive_keep = [True] * len(times)  # leaky: keeps overlapping pre-test rows
    purged = purge_mask(times, test_start, test_end, horizon_bars=2)
    assert naive_keep[3] is True
    assert purged[3] is False  # fail-closed: overlapping label row removed from train
    # Pure post-test / far-pre-test rows stay
    assert purged[0] is True
    assert purged[7] is True


def test_purge_single_session_test_window() -> None:
    bar = timedelta(days=1)
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    times = [t0 + i * bar for i in range(6)]
    # A one-session test window has equal lower/upper indices.
    mask = purge_mask(times, times[4], times[4], horizon_bars=1)
    assert mask[3] is False
    assert mask[2] is True

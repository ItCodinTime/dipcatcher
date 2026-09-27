"""Day Wave 104: overlap-aware multi-horizon QLIKE / HAC scoring.

Research-diagnostic identities only. live_pnl_claim=false; not a live edge.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.metrics.inference import overlap_aware_hac_lags
from quant_fund.metrics.scoring import (
    date_level_equal_weight,
    nonoverlapping_origin_mask,
    overlap_aware_qlike,
    qlike,
)
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False


def test_date_level_equal_weight_is_cross_sectional_mean() -> None:
    dates = np.array(["d0", "d0", "d1", "d1"], dtype=object)
    values = np.array([0.10, 0.30, 0.04, 0.08])
    keys, means = date_level_equal_weight(dates, values)
    assert keys.tolist() == ["d0", "d1"]
    assert means.tolist() == pytest.approx([0.20, 0.06])


def test_date_level_equal_weight_drops_nonfinite_and_empty_dates() -> None:
    dates = np.array(["d0", "d0", "d1"], dtype=object)
    values = np.array([0.10, np.nan, np.inf])
    keys, means = date_level_equal_weight(dates, values)
    assert keys.tolist() == ["d0"]
    assert means.tolist() == pytest.approx([0.10])


def test_date_level_equal_weight_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        date_level_equal_weight(["d0"], np.array([0.1, 0.2]))


def test_nonoverlapping_origin_mask_stride_equals_horizon() -> None:
    keep = nonoverlapping_origin_mask(np.arange(10, dtype=int), horizon_bars=5)
    assert keep.tolist() == [True, False, False, False, False, True, False, False, False, False]


def test_nonoverlapping_origin_mask_uses_session_gaps() -> None:
    # Origins at sessions 0,1,2,7 with h=5 keep 0 and 7 (7 >= 0+5).
    keep = nonoverlapping_origin_mask(np.array([0, 1, 2, 7], dtype=int), horizon_bars=5)
    assert keep.tolist() == [True, False, False, True]


def test_nonoverlapping_origin_mask_rejects_bool_and_unsorted() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        nonoverlapping_origin_mask(np.arange(3), True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="strictly increasing"):
        nonoverlapping_origin_mask(np.array([0, 2, 1], dtype=int), horizon_bars=2)


def test_overlap_aware_hac_lags_at_least_horizon_minus_one() -> None:
    assert overlap_aware_hac_lags(100, 20) == 19
    assert overlap_aware_hac_lags(27, 5) == 4
    with pytest.raises(ValueError, match="positive integer"):
        overlap_aware_hac_lags(10, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        overlap_aware_hac_lags(-1, 5)


def test_overlap_aware_qlike_is_date_level_not_name_broadcast() -> None:
    dates = np.array(["d0", "d0", "d1", "d1", "d2", "d2"], dtype=object)
    forecast = np.array([0.20, 0.20, 0.20, 0.20, 0.20, 0.20])
    realized = np.array([0.10, 0.30, 0.10, 0.30, 0.10, 0.30])
    scored = overlap_aware_qlike(dates, forecast, realized, horizon_bars=2)
    date_level = qlike(np.array([0.20, 0.20, 0.20]), np.array([0.20, 0.20, 0.20]))
    name_broadcast = qlike(realized, forecast)
    assert scored["scoring_scope"] == "date_level_equal_weight"
    assert scored["n_origins_overlapping"] == 3
    assert scored["n_origins_nonoverlapping"] == 2
    assert scored["qlike"] == pytest.approx(0.0)
    assert scored["qlike_overlapping"] == pytest.approx(date_level)
    assert name_broadcast > 0.0
    assert not math.isclose(float(name_broadcast), float(scored["qlike_overlapping"]))
    assert family_blob_forbidden_metrics_absent(scored) is True
    assert "live_pnl_claim" not in scored


def test_overlap_aware_qlike_session_index_skips_gapped_overlap() -> None:
    dates = np.array(["t0", "t1", "t5"], dtype=object)
    forecast = np.array([0.04, 0.04, 0.04])
    realized = np.array([0.04, 0.09, 0.04])
    session_index = {"t0": 0, "t1": 1, "t5": 5}
    scored = overlap_aware_qlike(
        dates,
        forecast,
        realized,
        horizon_bars=5,
        session_index=session_index,
    )
    # t1 overlaps t0; t5 is five sessions later and is kept.
    assert scored["n_origins_nonoverlapping"] == 2
    assert scored["qlike"] == pytest.approx(0.0)


def test_overlap_aware_qlike_rejects_within_date_forecast_mismatch() -> None:
    dates = np.array(["d0", "d0"], dtype=object)
    with pytest.raises(ValueError, match="unique within each date"):
        overlap_aware_qlike(dates, np.array([0.1, 0.2]), np.array([0.1, 0.1]), horizon_bars=1)


def test_overlap_aware_qlike_empty_and_missing_session_index_fail_closed() -> None:
    with pytest.raises(ValueError, match="at least one observation"):
        overlap_aware_qlike([], np.array([]), np.array([]), horizon_bars=1)
    with pytest.raises(ValueError, match="session_index missing"):
        overlap_aware_qlike(
            np.array(["d0"], dtype=object),
            np.array([0.1]),
            np.array([0.1]),
            horizon_bars=1,
            session_index={},
        )

"""Wave 38/46: scoring empty/NaN/length-mismatch edges — fail-closed or honest NaN.

Research metrics only — live_pnl_claim=false; not a live edge.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.scoring import (
    coverage,
    crps_from_quantiles,
    icir,
    interval_width,
    mean_pinball,
    pearson_ic,
    pinball_loss,
    pit_values,
    qlike,
    quantile_crossing_rate,
    rank_ic,
)

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False


def test_pinball_empty_and_bad_tau() -> None:
    empty = np.array([])
    assert pinball_loss(empty, empty, 0.5).size == 0
    with pytest.raises(ValueError, match="tau"):
        pinball_loss(np.array([1.0]), np.array([1.0]), 0.0)
    with pytest.raises(ValueError, match="tau"):
        pinball_loss(np.array([1.0]), np.array([1.0]), 1.0)


def test_pinball_length_mismatch_fail_closed() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        pinball_loss(np.array([1.0, 2.0]), np.array([1.0]), 0.5)


def test_coverage_empty_nan_and_mismatch() -> None:
    assert np.isnan(coverage(np.array([]), np.array([]), np.array([])))
    assert np.isclose(
        coverage(np.array([0.0, 1.0]), np.array([-1.0, 0.0]), np.array([0.5, 2.0])),
        1.0,
    )
    # One miss
    assert np.isclose(
        coverage(np.array([0.0, 5.0]), np.array([-1.0, 0.0]), np.array([0.5, 2.0])),
        0.5,
    )
    with pytest.raises(ValueError, match="length mismatch"):
        coverage(np.array([1.0, 2.0]), np.array([0.0]), np.array([2.0, 3.0]))


def test_interval_width_empty_and_mismatch() -> None:
    assert np.isnan(interval_width(np.array([]), np.array([])))
    assert np.isclose(interval_width(np.array([0.0, 1.0]), np.array([2.0, 4.0])), 2.5)
    with pytest.raises(ValueError, match="length mismatch"):
        interval_width(np.array([0.0]), np.array([1.0, 2.0]))
    assert np.isclose(
        interval_width(np.array([0.0, 2.0]), np.array([1.0, 1.0])),
        1.0,
    )


def test_quantile_crossing_rate_does_not_count_nan_as_non_crossing() -> None:
    taus = np.array([0.1, 0.5, 0.9])
    q = np.array([[0.0, 1.0, 2.0], [np.nan, np.nan, np.nan]])
    assert quantile_crossing_rate(q, taus) == 0.0
    assert np.isnan(quantile_crossing_rate(np.full((1, 3), np.nan), taus))
    with pytest.raises(ValueError, match="taus must be"):
        quantile_crossing_rate(np.zeros((1, 3)), np.array([0.5, 0.1, 0.9]))


def test_crps_empty_mismatch_and_nan_y() -> None:
    taus = np.array([0.1, 0.5, 0.9])
    assert np.isnan(crps_from_quantiles(np.array([]), np.zeros((0, 3)), taus))
    with pytest.raises(ValueError, match="length mismatch"):
        crps_from_quantiles(
            np.array([0.0]),
            np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0]]),
            taus,
        )
    with pytest.raises(ValueError, match="matching taus"):
        crps_from_quantiles(np.array([0.0]), np.array([[0.0, 1.0]]), taus)
    # NaN y propagates into mean (honest, not silently zero)
    out = crps_from_quantiles(
        np.array([np.nan]),
        np.array([[-0.2, 0.0, 0.2]]),
        taus,
    )
    assert np.isnan(out)
    with pytest.raises(ValueError, match="strictly inside"):
        crps_from_quantiles(np.array([0.0]), np.zeros((1, 3)), np.array([0.1, 0.9, 0.5]))


def test_qlike_empty_mismatch_and_floor() -> None:
    assert np.isnan(qlike(np.array([]), np.array([])))
    with pytest.raises(ValueError, match="length mismatch"):
        qlike(np.array([0.1, 0.2]), np.array([0.1]))
    # Zero forecast variance is malformed input under the fail-closed variance
    # contract (finite y, y >= 0, yhat > 0) — masked to NaN, never floor-clipped
    # into a fabricated finite score.
    assert np.isnan(qlike(np.array([0.0]), np.array([0.0])))
    assert np.isnan(qlike(np.array([0.1]), np.array([0.0])))
    assert np.isnan(qlike(np.array([-0.1]), np.array([0.1])))
    # Mixed valid + invalid: only the valid pair scores; invalid entries are
    # masked out, never floor-clipped into the mean.
    mixed = qlike(np.array([0.04, 0.0, -0.1]), np.array([0.09, 0.0, 0.1]))
    assert np.isfinite(mixed)
    assert mixed == pytest.approx(0.04 / 0.09 - np.log(0.04 / 0.09) - 1.0)


def test_pearson_rank_ic_short_nan_const_mismatch() -> None:
    assert np.isnan(pearson_ic(np.array([]), np.array([])))
    assert np.isnan(pearson_ic(np.array([1.0, 2.0]), np.array([1.0, 2.0])))
    assert np.isnan(rank_ic(np.array([1.0, np.nan]), np.array([1.0, 2.0])))
    # Constant series → NaN (near-zero std)
    assert np.isnan(pearson_ic(np.ones(5), np.linspace(0, 1, 5)))
    assert np.isnan(pearson_ic(np.linspace(0, 1, 5), np.ones(5)))
    # All-nan mask leaves < 3
    assert np.isnan(pearson_ic(np.array([np.nan, 1.0, 2.0]), np.array([0.0, np.nan, 2.0])))
    with pytest.raises(ValueError, match="length mismatch"):
        pearson_ic(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="length mismatch"):
        rank_ic(np.array([1.0, 2.0, 3.0]), np.array([1.0]))


def test_icir_empty_short_const_near_zero_std() -> None:
    assert np.isnan(icir(np.array([])))
    assert np.isnan(icir(np.array([0.1])))
    assert np.isnan(icir(np.array([np.nan, np.nan])))
    # Exact and near-constant (float noise on 0.1*3) → NaN fail-closed
    assert np.isnan(icir(np.array([0.1, 0.1])))
    assert np.isnan(icir(np.array([0.1, 0.1, 0.1])))
    ics = np.array([0.05, -0.02, 0.03, 0.01])
    assert np.isfinite(icir(ics))
    assert np.isfinite(icir(ics, annualize=True, periods_per_year=252.0))


def test_crossing_rate_empty_and_shape() -> None:
    taus = np.array([0.1, 0.5, 0.9])
    assert np.isnan(quantile_crossing_rate(np.zeros((0, 3)), taus))
    with pytest.raises(ValueError, match="quantiles must be"):
        quantile_crossing_rate(np.array([0.1, 0.2, 0.3]), taus)


def test_mean_pinball_empty_and_mismatch() -> None:
    assert np.isnan(mean_pinball(np.array([]), np.array([]), 0.5))
    with pytest.raises(ValueError, match="length mismatch"):
        mean_pinball(np.array([1.0, 2.0]), np.array([1.0]), 0.5)
    assert np.isclose(mean_pinball(np.array([1.0]), np.array([1.0]), 0.5), 0.0)


def test_pit_values_empty_mismatch_and_shape() -> None:
    taus = np.array([0.1, 0.5, 0.9])
    out = pit_values(np.array([]), np.zeros((0, 3)), taus)
    assert out.size == 0
    with pytest.raises(ValueError, match="length mismatch"):
        pit_values(np.array([0.0]), np.zeros((2, 3)), taus)
    with pytest.raises(ValueError, match="matching taus"):
        pit_values(np.array([0.0]), np.zeros((1, 2)), taus)
    with pytest.raises(ValueError, match="quantiles must be 2d"):
        pit_values(np.array([0.0]), np.array([0.1, 0.5, 0.9]), taus)
    # Mid quantile → ~0.5 PIT
    pits = pit_values(np.array([0.0]), np.array([[-1.0, 0.0, 1.0]]), taus)
    assert np.isclose(pits[0], 0.5)
    dirty = pit_values(
        np.array([0.0, 0.0, np.nan]),
        np.array([[-1.0, 0.0, 1.0], [1.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]),
        taus,
    )
    assert np.isclose(dirty[0], 0.5)
    assert np.isnan(dirty[1]) and np.isnan(dirty[2])
    with pytest.raises(ValueError, match="taus must be"):
        pit_values(np.array([0.0]), np.array([[-1.0, 0.0, 1.0]]), np.array([0.1, np.nan, 0.9]))
    with pytest.raises(ValueError, match="strictly inside"):
        pit_values(np.array([0.0]), np.array([[-1.0, 0.0, 1.0]]), np.array([0.0, 0.5, 0.9]))

import numpy as np
import pytest

from quant_fund.portfolio.portfolio_conformal import (
    SplitPortfolioCQR,
    bench_portfolio_cqr,
    gaussian_band,
    portfolio_return,
    sets_by_date,
)


def test_portfolio_return_ignores_nan_names() -> None:
    w = np.array([0.5, np.nan, 0.5])
    r = np.array([0.10, 0.99, 0.20])
    assert portfolio_return(w, r) == pytest.approx(0.15)
    assert portfolio_return(np.array([0.5, 0.5]), np.array([0.10, np.nan])) == pytest.approx(0.05)
    assert np.isnan(portfolio_return(np.array([np.nan]), np.array([0.1])))


def test_portfolio_return_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        portfolio_return(np.array([0.5, 0.5]), np.array([0.1]))


def test_exchangeable_port_returns_cover_at_alpha_0_10() -> None:
    rng = np.random.default_rng(0)
    n_dates, n_names = 400, 8
    weights = np.full((n_dates, n_names), 1.0 / n_names)
    returns = rng.normal(0.0, 0.02, size=(n_dates, n_names))
    sets = sets_by_date(weights, returns, alpha=0.10)
    y = sets.r_p[sets.test_mask]
    lo = sets.lower[sets.test_mask]
    hi = sets.upper[sets.test_mask]
    cov = float(np.mean((y >= lo) & (y <= hi)))
    assert cov >= 0.85
    assert sets.qhat >= 0.0
    assert int(sets.test_mask.sum()) == y.size


def test_one_date_is_one_score() -> None:
    rng = np.random.default_rng(1)
    n_dates, n_names = 80, 12
    weights = np.full((n_dates, n_names), 1.0 / n_names)
    returns = rng.normal(0.0, 0.01, size=(n_dates, n_names))
    r_p = np.array([portfolio_return(weights[i], returns[i]) for i in range(n_dates)])
    lo = np.full(n_dates, -0.01)
    hi = np.full(n_dates, 0.01)
    cal = slice(0, 40)
    cqr = SplitPortfolioCQR(0.10).calibrate(r_p[cal], lo[cal], hi[cal])
    assert cqr.scores_ is not None
    assert cqr.scores_.size == 40
    assert cqr.scores_.size != n_dates * n_names
    sets = sets_by_date(weights, returns, alpha=0.10)
    assert sets.r_p.size == n_dates
    assert sets.lower.size == n_dates
    assert sets.scores is not None
    assert sets.scores.size == sets.n_cal
    assert sets.n_cal < n_dates * n_names
    assert sets.n_cal == int(np.isfinite(sets.scores).sum())


def test_sets_by_date_walks_dict_keys() -> None:
    weights = {2: np.array([0.6, 0.4]), 1: np.array([1.0, 0.0]), 3: np.array([0.5, 0.5])}
    returns = {1: np.array([0.02, 0.00]), 2: np.array([0.01, -0.01]), 3: np.array([0.00, 0.04])}
    sets = sets_by_date(weights, returns, alpha=0.10)
    assert sets.dates == ["1", "2", "3"]
    assert sets.r_p.size == 3
    assert np.isclose(sets.r_p[0], 0.02)


def test_normalized_vol_widens_high_vol_dates() -> None:
    rng = np.random.default_rng(2)
    n_dates, n_names = 200, 6
    vol = np.where(np.arange(n_dates) < 100, 0.01, 0.04)
    weights = np.full((n_dates, n_names), 1.0 / n_names)
    returns = rng.normal(0.0, 1.0, size=(n_dates, n_names)) * vol[:, None]
    sets = sets_by_date(weights, returns, alpha=0.10, vol=vol)
    width = sets.upper - sets.lower
    assert float(np.mean(width[100:])) > float(np.mean(width[:100]))


def test_split_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        SplitPortfolioCQR(0.0)
    with pytest.raises(ValueError):
        gaussian_band(0.0, 0.01, 1.0)


def test_bench_portfolio_cqr_coverage_width_no_sharpe() -> None:
    row = bench_portfolio_cqr(seed=7)
    assert {"coverage", "mean_width", "n_dates"} <= set(row)
    assert row["coverage"] >= 0.85
    assert row["mean_width"] > 0.0
    assert row["n_dates"] > 0
    forbidden = ("sharpe", "pnl", "information_ratio", "sortino")
    assert not any(any(tok in k.lower() for tok in forbidden) for k in row)


def test_bench_does_not_stack_names_as_dates() -> None:
    rng = np.random.default_rng(3)
    n_dates, n_names = 120, 20
    weights = np.full((n_dates, n_names), 1.0 / n_names)
    returns = rng.normal(size=(n_dates, n_names))
    row = bench_portfolio_cqr(weights, returns, alpha=0.10)
    assert row["n_dates"] < n_dates
    assert row["n_dates"] < n_dates * n_names
    assert row["n_cal"] < n_dates * n_names

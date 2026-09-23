import numpy as np

from quant_fund.metrics.cross_section import date_ic_series, decile_portfolios


def test_date_ic_recovers_perfect_within_date() -> None:
    dates = np.repeat(np.arange(30), 8)
    scores = np.linspace(-1, 1, dates.size)
    # scramble globally but keep within-date rank via grouping
    y = scores + 0.01 * np.sin(np.arange(dates.size))
    out = date_ic_series(scores, y, dates, min_names=5)
    assert out.n_dates >= 20
    assert out.mean_pearson > 0.9
    assert out.p_pearson < 1e-6


def test_deciles_monotone_when_aligned() -> None:
    rng = np.random.default_rng(4)
    n_dates, n_names = 40, 20
    dates = np.repeat(np.arange(n_dates), n_names)
    scores = rng.normal(size=n_dates * n_names)
    y = scores + 0.2 * rng.normal(size=scores.size)
    dec = decile_portfolios(scores, y, dates, n_buckets=5, min_names=8)
    assert dec.monotonicity > 0.6
    assert dec.mean_ls > 0
    assert dec.p_ls < 0.05

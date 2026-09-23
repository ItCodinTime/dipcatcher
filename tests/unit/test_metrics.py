import numpy as np

from quant_fund.metrics.overfitting import deflated_sharpe, probabilistic_sharpe
from quant_fund.metrics.returns import (
    downside_deviation,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    wealth_index,
)
from quant_fund.metrics.risk import historical_es, historical_var, losses_from_returns
from quant_fund.metrics.scoring import (
    crps_from_quantiles,
    crps_gaussian,
    crps_gaussian_mixture,
    gaussian_mixture_quantiles,
    pearson_ic,
    pinball_loss,
    qlike,
    quantile_crossing_rate,
    rank_ic,
    rearrange_quantiles,
)


def test_downside_and_sortino_use_per_period_mar() -> None:
    returns = np.array([0.02, -0.01, 0.03, -0.02])
    downside = np.sqrt((0.01**2 + 0.02**2) / 4.0)
    assert np.isclose(downside_deviation(returns, mar=0.0, periods_per_year=1.0), downside)
    assert np.isclose(
        sortino_ratio(returns, mar=0.0, periods_per_year=4.0),
        np.mean(returns) / downside * 2.0,
    )


def test_return_metrics_reject_invalid_annualization() -> None:
    with np.testing.assert_raises(ValueError):
        downside_deviation(np.array([0.01, -0.01]), periods_per_year=0.0)
    with np.testing.assert_raises(ValueError):
        sortino_ratio(np.array([0.01, -0.01]), mar=float("nan"))


def test_pinball_zero_error() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert np.allclose(pinball_loss(y, y, 0.5), 0.0)


def test_pinball_asymmetric() -> None:
    y = np.array([1.0])
    q = np.array([0.0])
    # y > q, tau=0.1 → 0.1 * 1
    assert pinball_loss(y, q, 0.1)[0] == 0.1
    q = np.array([2.0])
    # y < q, tau=0.1 → 0.9 * 1
    assert np.isclose(pinball_loss(y, q, 0.1)[0], 0.9)


def test_qlike_perfect() -> None:
    y = np.array([0.04, 0.09])
    assert qlike(y, y) == 0.0


def test_drawdown_monotone_wealth() -> None:
    r = np.array([0.01, 0.02, 0.0, 0.03])
    assert max_drawdown(r) == 0.0
    w = wealth_index(r)
    assert w[-1] > w[0]


def test_drawdown_hand_example() -> None:
    r = np.array([0.1, -0.5, 0.0])
    # wealth: 1.1, 0.55, 0.55; peak 1.1; mdd = 0.55/1.1 - 1 = -0.5
    assert np.isclose(max_drawdown(r), -0.5)


def test_sharpe_flags_high() -> None:
    rng = np.random.default_rng(0)
    r = 0.05 + 1e-6 * rng.normal(size=500)
    out = sharpe_ratio(r)
    assert out["flag_high_sharpe"] is True


def test_historical_es_uses_fractional_tail_mass() -> None:
    losses = np.array([0.0, 1.0, 2.0, 3.0])

    # alpha=.625 leaves 1.5 observations in the tail: 3 plus half of 2.
    assert np.isclose(historical_es(losses, 0.625), (3.0 + 0.5 * 2.0) / 1.5)


def test_quantile_es_integrates_between_uneven_quantiles() -> None:
    from quant_fund.metrics.risk import var_es_from_return_quantiles

    var, es = var_es_from_return_quantiles({0.01: -4.0, 0.05: -2.0, 0.20: 1.0}, 0.9)
    # At p=.1, linear interpolation gives -1; the lower-tail area is
    # constant -4 on [0,.01], then trapezoidal over [.01,.05] and [.05,.1].
    expected_area = 0.01 * -4.0 + 0.5 * 0.04 * (-4.0 - 2.0) + 0.5 * 0.05 * (-2.0 - 1.0)
    assert np.isclose(var, 1.0)
    assert np.isclose(es, -expected_area / 0.1)


def test_var_es_loss_convention() -> None:
    rets = np.array([0.01, -0.02, -0.05, 0.0, 0.03])
    losses = losses_from_returns(rets)
    var = historical_var(losses, 0.8)
    es = historical_es(losses, 0.8)
    assert es >= var - 1e-12
    assert var >= 0  # 80% loss quantile of mostly small losses; -(-0.05)=0.05 is worst


def test_crossing_and_rearrange() -> None:
    q = np.array([[0.2, 0.1, 0.3]])
    taus = np.array([0.1, 0.5, 0.9])
    assert quantile_crossing_rate(q, taus) == 1.0
    q2 = rearrange_quantiles(q)
    assert quantile_crossing_rate(q2, taus) == 0.0


def test_crps_approx_nonnegative() -> None:
    y = np.array([0.0, 0.1])
    q = np.array([[-0.2, 0.0, 0.2], [-0.1, 0.1, 0.3]])
    taus = np.array([0.1, 0.5, 0.9])
    assert crps_from_quantiles(y, q, taus) >= 0


def test_crps_gaussian_mixture_single_component_matches_gaussian() -> None:
    y = np.array([-0.05, 0.0, 0.12])
    mix = crps_gaussian_mixture(
        y, np.array([1.0]), np.array([0.01]), np.array([0.03])
    )
    ref = crps_gaussian(y, np.full(3, 0.01), np.full(3, 0.03))
    np.testing.assert_allclose(mix, ref, rtol=0, atol=1e-14)


def test_crps_gaussian_mixture_matches_monte_carlo() -> None:
    rng = np.random.default_rng(0)
    w = np.array([0.35, 0.5, 0.15])
    mu = np.array([-0.02, 0.005, 0.05])
    sig = np.array([0.01, 0.025, 0.06])
    y = np.array([0.0, 0.04, -0.03])
    exact = crps_gaussian_mixture(y, w, mu, sig)
    # CRPS = E|X - y| - 0.5 E|X - X'| on independent draws.
    n = 2_000_000
    for k, yi in enumerate(y):
        c1 = rng.choice(3, size=n, p=w)
        c2 = rng.choice(3, size=n, p=w)
        x1 = rng.normal(mu[c1], sig[c1])
        x2 = rng.normal(mu[c2], sig[c2])
        mc = np.abs(x1 - yi).mean() - 0.5 * np.abs(x1 - x2).mean()
        assert abs(exact[k] - mc) < 5e-4
    assert np.all(exact > 0.0)


def test_gaussian_mixture_quantiles_invert_cdf() -> None:
    from scipy.special import erf

    w = np.array([0.4, 0.6])
    mu = np.array([-0.01, 0.03])
    sig = np.array([0.02, 0.05])
    taus = np.array([0.05, 0.5, 0.95])
    q = gaussian_mixture_quantiles(w, mu, sig, taus)
    for qi, tau in zip(q, taus, strict=True):
        z = (qi - mu) / sig
        cdf = float((w * (0.5 * (1.0 + erf(z / np.sqrt(2.0))))).sum())
        assert abs(cdf - tau) < 1e-10
    assert np.all(np.diff(q) > 0.0)


def test_gaussian_mixture_degenerate_fails_closed() -> None:
    y = np.array([0.0])
    assert np.all(np.isnan(crps_gaussian_mixture(y, [0.5, 0.5], [0.0, np.nan], [0.1, 0.1])))
    assert np.all(np.isnan(crps_gaussian_mixture(y, [0.5, 0.5], [0.0, 0.0], [0.1, -0.1])))
    assert np.all(np.isnan(crps_gaussian_mixture(y, [0.5, 0.4], [0.0, 0.0], [0.1, 0.1])))
    assert np.all(np.isnan(gaussian_mixture_quantiles([0.5, 0.4], [0.0, 0.0], [0.1, 0.1], [0.5])))


def test_ic_perfect() -> None:
    x = np.linspace(0, 1, 20)
    assert pearson_ic(x, x) > 0.99
    assert rank_ic(x, x) > 0.99


def test_psr_dsr_bounds() -> None:
    psr = probabilistic_sharpe(0.0, 0.0, 100, 0.0, 3.0)
    assert 0.4 < psr < 0.6
    dsr = deflated_sharpe(2.0, 252, 0.0, 3.0, 100, 0.04)
    assert 0.0 <= dsr <= 1.0

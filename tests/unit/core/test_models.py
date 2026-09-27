import numpy as np

from quant_fund.metrics.scoring import quantile_crossing_rate
from quant_fund.models.distribution import GaussianDistribution
from quant_fund.models.ranking import RidgeRanker, group_sizes
from quant_fund.models.regime import GaussianHMMRegime, SingleStateRegime


def test_ridge_ranker_fits() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 3))
    y = x[:, 0] * 0.2 + rng.normal(size=100) * 0.01
    m = RidgeRanker(1.0).fit(x, y)
    pred = m.predict(x)
    assert pred.shape == (100,)
    assert np.corrcoef(pred, y)[0, 1] > 0.5


def test_group_sizes() -> None:
    dates = np.array([1, 1, 1, 2, 2, 3])
    g = group_sizes(dates)
    assert list(g) == [3, 2, 1]


def test_scaled_historical_tail_hits_heteroskedastic() -> None:
    from quant_fund.metrics.probability import kupiec_pof
    from quant_fund.models.tail import HistoricalTail, ScaledHistoricalTail

    rng = np.random.default_rng(5)
    n = 1200
    scale = rng.choice(np.array([0.5, 2.0]), size=n)
    y = rng.normal(0.0, scale)
    tr, te = slice(0, 600), slice(600, None)
    hist = HistoricalTail(0.95).fit(np.zeros((600, 1)), y[tr])
    var, _ = hist.predict_var_es()
    scaled = ScaledHistoricalTail(0.95).fit(y[tr], scale[tr])
    var_s, _ = scaled.predict_var_es(scale[te])
    high = scale[te] >= 1.0
    loss = -y[te]
    gap_raw = abs(float(np.mean(loss[high] >= var)) - float(np.mean(loss[~high] >= var)))
    gap_scaled = abs(
        float(np.mean(loss[high] >= var_s[high])) - float(np.mean(loss[~high] >= var_s[~high]))
    )
    hits = (loss >= var_s).astype(float)
    rate, _lr, p = kupiec_pof(hits, 0.05)
    assert gap_scaled < gap_raw
    assert 0.02 <= rate <= 0.10
    assert p > 1e-6


def test_scaled_gaussian_covers_heteroskedastic() -> None:
    from quant_fund.models.distribution import ScaledGaussianDistribution

    rng = np.random.default_rng(4)
    n = 800
    scale = rng.choice(np.array([0.5, 2.0]), size=n)
    y = rng.normal(0.0, scale)
    tr, te = slice(0, 400), slice(400, None)
    m = ScaledGaussianDistribution([0.05, 0.95]).fit(y[tr], scale[tr])
    q = m.predict(scale[te])
    cov = float(np.mean((y[te] >= q[:, 0]) & (y[te] <= q[:, 1])))
    assert cov >= 0.85


def test_gaussian_quantiles_ordered() -> None:
    y = np.random.default_rng(0).normal(size=200)
    m = GaussianDistribution([0.1, 0.5, 0.9]).fit(np.zeros((200, 1)), y)
    q = m.predict(np.zeros((5, 1)))
    assert quantile_crossing_rate(q, np.array([0.1, 0.5, 0.9])) == 0.0


def test_hmm_probs_sum_to_one() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(80, 3))
    m = GaussianHMMRegime(n_states=3, seed=1).fit(x)
    p = m.predict_proba(x)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert p.shape == (80, 3)


def test_single_state() -> None:
    x = np.ones((10, 2))
    p = SingleStateRegime().fit(x).predict_proba(x)
    assert np.allclose(p, 1.0)


def test_har_design_uses_only_past_realized_volatility() -> None:
    from quant_fund.models.volatility import HARVol

    rv = np.arange(1.0, 30.0)
    design = HARVol.har_design(rv)

    # The forecast made at t must not contain rv[t]; all HAR aggregates are lagged.
    assert design[22, 1] == rv[21]
    assert np.isclose(design[22, 2], np.mean(rv[17:22]))
    assert np.isclose(design[22, 3], np.mean(rv[0:22]))


def test_scaled_student_t_covers_heteroskedastic() -> None:
    from quant_fund.models.distribution import ScaledStudentTDistribution

    rng = np.random.default_rng(6)
    n = 800
    scale = rng.choice(np.array([0.5, 2.0]), size=n)
    y = rng.normal(0.0, scale)
    tr, te = slice(0, 400), slice(400, None)
    m = ScaledStudentTDistribution([0.05, 0.95]).fit(y[tr], scale[tr])
    q = m.predict(scale[te])
    cov = float(np.mean((y[te] >= q[:, 0]) & (y[te] <= q[:, 1])))
    assert cov >= 0.85
    assert 3.0 <= m.nu <= 30.0

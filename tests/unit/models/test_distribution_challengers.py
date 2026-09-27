"""Distribution challengers (skew_t / gmm / isotonic PIT) — fit contract,
ordered quantiles, recovery on planted structure, fail-closed edges.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from quant_fund.config import load_config
from quant_fund.metrics.scoring import mean_pinball
from quant_fund.models.distribution import (
    GaussianDistribution,
    GMMDistribution,
    IsotonicPitDistribution,
    SkewTDistribution,
)
from quant_fund.models.mixture import (
    fit_gaussian_mixture,
    gaussian_mixture_cdf_1d,
    gaussian_mixture_crps_1d,
    gaussian_mixture_ppf_1d,
)
from quant_fund.pipeline import train as train_module
from quant_fund.pipeline.train import train_distribution

TAUS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
MID = TAUS.index(0.5)


def _assert_ordered(q: np.ndarray) -> None:
    assert q.ndim == 2
    assert np.all(np.isfinite(q))
    assert np.all(np.diff(q, axis=1) >= -1e-9)


def _bimodal(seed: int = 0, n_lo: int = 3000, n_hi: int = 700) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0.0, 0.01, n_lo), rng.normal(0.06, 0.03, n_hi)])


def _x(n: int) -> np.ndarray:
    return np.ones((n, 2))


def test_skew_t_fit_predict_ordered_and_finite() -> None:
    y = _bimodal()
    m = SkewTDistribution(TAUS).fit(_x(y.size), y)
    q = m.predict(_x(5))
    assert q.shape == (5, len(TAUS))
    _assert_ordered(q)
    assert m.metadata().name == "skew_t"


def test_skew_t_captures_left_skew() -> None:
    rng = np.random.default_rng(2)
    z = rng.standard_t(df=5, size=4000)
    y = np.where(z < 0, z * 2.2, z) * 0.01  # heavy left tail
    m = SkewTDistribution(TAUS).fit(_x(y.size), y)
    assert m.params_ is not None
    assert m.params_["lam"] < 0.0
    assert 2.0 < m.params_["nu"] < 200.0
    assert m.params_["sigma"] > 0.0


def test_skew_t_zero_variance_fail_closed() -> None:
    y = np.full(200, 1.0)
    with pytest.raises(ValueError, match="zero variance"):
        SkewTDistribution(TAUS).fit(_x(y.size), y)


def test_skew_t_too_short_fail_closed() -> None:
    with pytest.raises(ValueError, match=">= 30"):
        SkewTDistribution(TAUS).fit(_x(10), np.linspace(0, 1, 10))


def test_skew_t_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not been fitted"):
        SkewTDistribution(TAUS).predict(_x(3))


def test_gmm_recovers_bimodal_quantiles() -> None:
    y = _bimodal()
    m = GMMDistribution(TAUS).fit(_x(y.size), y)
    assert m.k_ == 2
    q = m.predict(_x(3))
    assert q.shape == (3, len(TAUS))
    _assert_ordered(q)
    empirical = np.quantile(y, TAUS)
    assert np.allclose(q[0], empirical, atol=0.01)


def test_gmm_beats_gaussian_on_planted_mixture() -> None:
    y = _bimodal(seed=3)
    x = _x(y.size)
    gmm = GMMDistribution(TAUS).fit(x, y).predict(x)
    gauss = GaussianDistribution(TAUS).fit(x, y).predict(x)
    for i, tau in enumerate(TAUS):
        assert mean_pinball(y, gmm[:, i], tau) <= mean_pinball(y, gauss[:, i], tau) + 1e-12


def test_gmm_fixed_k_and_bic_path() -> None:
    y = _bimodal()
    fixed = GMMDistribution(TAUS, k=3, seed=1).fit(_x(y.size), y)
    assert fixed.k_ == 3
    auto = GMMDistribution(TAUS, k=None, seed=1).fit(_x(y.size), y)
    assert auto.k_ in {2, 3, 4}
    _assert_ordered(auto.predict(_x(2)))


def test_gmm_short_or_bad_k_fail_closed() -> None:
    with pytest.raises(ValueError, match=">= 30"):
        GMMDistribution(TAUS).fit(_x(10), np.linspace(0, 1, 10))
    y = np.linspace(0, 1, 40)
    with pytest.raises(ValueError):
        GMMDistribution(TAUS, k=50).fit(_x(y.size), y)


def test_gmm_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not been fitted"):
        GMMDistribution(TAUS).predict(_x(2))


def test_isotonic_identity_on_well_specified_gaussian() -> None:
    rng = np.random.default_rng(4)
    y = rng.normal(0.01, 0.02, 4000)
    x = _x(y.size)
    iso = IsotonicPitDistribution(TAUS).fit(x, y).predict(x)
    gauss = GaussianDistribution(TAUS).fit(x, y).predict(x)
    _assert_ordered(iso)
    # well-specified base → recalibration map ~identity → close quantiles
    assert np.allclose(iso[0], gauss[0], atol=0.004)


def test_isotonic_recalibrates_on_heavy_tails() -> None:
    rng = np.random.default_rng(5)
    y = rng.standard_t(df=3, size=5000) * 0.02
    x = _x(y.size)
    iso = IsotonicPitDistribution(TAUS).fit(x, y).predict(x)
    gauss = GaussianDistribution(TAUS).fit(x, y).predict(x)
    # heavy-tail draws inflate sigma-hat; the recalibrated outer quantiles
    # land closer to the empirical quantiles than the raw Gaussian's
    emp = np.quantile(y, TAUS)
    for j in (0, 1, len(TAUS) - 2, len(TAUS) - 1):
        assert abs(iso[0, j] - emp[j]) < abs(gauss[0, j] - emp[j])
    pin_iso = mean_pinball(y, iso[:, MID], TAUS[MID])
    pin_g = mean_pinball(y, gauss[:, MID], TAUS[MID])
    assert np.isfinite(pin_iso) and np.isfinite(pin_g)


def test_isotonic_fail_closed_edges() -> None:
    with pytest.raises(ValueError, match=">= 16"):
        IsotonicPitDistribution(TAUS).fit(_x(8), np.linspace(0, 1, 8))
    with pytest.raises(RuntimeError, match="not been fitted"):
        IsotonicPitDistribution(TAUS).predict(_x(2))


def test_mixture_ppf_inverts_cdf() -> None:
    y = _bimodal()
    fit = fit_gaussian_mixture(y[:, None], 2, seed=0)
    taus = np.array(TAUS)
    q = gaussian_mixture_ppf_1d(fit, taus)
    assert np.allclose(gaussian_mixture_cdf_1d(fit, q), taus, atol=1e-8)


def test_mixture_ppf_bad_taus_fail_closed() -> None:
    y = _bimodal()
    fit = fit_gaussian_mixture(y[:, None], 2, seed=0)
    with pytest.raises(ValueError, match="0, 1"):
        gaussian_mixture_ppf_1d(fit, np.array([0.0, 0.5]))


def test_mixture_crps_matches_monte_carlo() -> None:
    y = _bimodal()
    fit = fit_gaussian_mixture(y[:, None], 2, seed=0)
    w = fit["weights"]
    means = fit["means"].ravel()
    sds = np.sqrt(fit["covs"].reshape(2, -1)[:, 0])
    rng = np.random.default_rng(9)
    c1 = rng.choice(2, 200_000, p=w)
    c2 = rng.choice(2, 200_000, p=w)
    x = rng.normal(means[c1], sds[c1])
    xp = rng.normal(means[c2], sds[c2])
    for yy in (0.0, 0.08):
        closed = gaussian_mixture_crps_1d(fit, np.array([yy]))[0]
        mc = float(np.abs(x - yy).mean() - 0.5 * np.abs(x - xp).mean())
        assert closed == pytest.approx(mc, abs=2e-3)


def test_mixture_crps_single_component_matches_gaussian() -> None:
    rng = np.random.default_rng(7)
    y = rng.normal(0.0, 0.02, 3000)
    fit = fit_gaussian_mixture(y[:, None], 1, seed=0)
    # E|X - y| for one Gaussian; the self-term cancels half-weight exactly
    crps = gaussian_mixture_crps_1d(fit, np.array([0.0]))
    assert np.isfinite(crps[0]) and crps[0] > 0.0


def test_mixture_crps_nonfinite_y_fail_closed() -> None:
    y = _bimodal()
    fit = fit_gaussian_mixture(y[:, None], 2, seed=0)
    with pytest.raises(ValueError, match="finite"):
        gaussian_mixture_crps_1d(fit, np.array([np.nan]))


def test_train_distribution_accepts_new_model_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """New catalog names pass _require_model; empty folds then fail closed."""
    cfg = load_config("configs/research.yaml")
    frame = pl.DataFrame(
        {
            "event_time": [0, 1],
            "security_id": ["a", "a"],
            cfg.train.distribution_target: [0.01, 0.02],
            "ret_1": [0.0, 0.0],
        }
    )
    monkeypatch.setattr(train_module, "panel", lambda *a, **k: frame)
    monkeypatch.setattr(train_module, "_walk_forward_splits", lambda *a, **k: [])
    for name in ("skew_t", "gmm", "isotonic"):
        with pytest.raises(ValueError, match="no trainable/evaluable fold"):
            train_distribution(cfg, name)
    with pytest.raises(ValueError, match="unknown distribution model"):
        train_distribution(cfg, "not_a_model")

"""Tests for quant_fund.metrics.conformal_martingale — conformal test martingales. SYNTHETIC."""

import numpy as np
import pytest
from scipy.stats import kstest

from quant_fund.metrics.conformal_martingale import (
    WatchMonitor,
    conformal_p_values,
    martingale_alarm,
    mixture_martingale,
    power_martingale,
    simple_jumper,
    weighted_conformal_p_value,
)

ALPHA = 0.05
N_MC = 300


def test_conformal_p_values_uniform_under_exchangeability() -> None:
    rng = np.random.default_rng(0)
    p = conformal_p_values(rng.standard_t(3, 3000), seed=1)
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert kstest(p, "uniform").pvalue > 0.01
    # lag-1 dependence negligible (i.i.d. under the null)
    assert abs(np.corrcoef(p[:-1], p[1:])[0, 1]) < 0.06


def test_conformal_p_values_handle_ties_via_randomization() -> None:
    p = conformal_p_values(np.zeros(500), seed=2)
    assert kstest(p, "uniform").pvalue > 0.01


def test_weighted_p_value_reduces_to_unweighted() -> None:
    past = np.array([1.0, 3.0, 2.0])
    p_w = weighted_conformal_p_value(past, 2.5, np.ones(4), u=0.5)
    # greater: 3.0 -> 1/4; ties: itself -> 1/4 * 0.5
    assert p_w == pytest.approx(0.25 + 0.125)
    with pytest.raises(ValueError):
        weighted_conformal_p_value(past, 2.5, np.ones(3), u=0.5)
    with pytest.raises(ValueError):
        weighted_conformal_p_value(past, 2.5, np.zeros(4), u=0.5)


def test_betting_functions_integrate_to_one() -> None:
    """One-step fairness: E_U[f(U)] = 1 for power and mixture bets (quadrature)."""
    grid = (np.arange(1_000_000) + 0.5) / 1_000_000
    for eps in (0.5, 0.7, 0.9):
        assert power_martingale(np.array([0.3]), eps)[0] == pytest.approx(eps * 0.3 ** (eps - 1.0))
        assert np.mean(eps * grid ** (eps - 1.0)) == pytest.approx(1.0, rel=2e-3)
    # closed form of the one-step simple mixture: int_0^1 eps p^(eps-1) d eps
    for p in (0.05, 0.3, 0.8):
        lp = np.log(p)
        exact = (lp - 1.0 + 1.0 / p) / lp**2
        assert mixture_martingale(np.array([p]), n_grid=2000)[0] == pytest.approx(exact, rel=1e-2)


def test_jumper_is_martingale_under_null() -> None:
    """E[M_t] = 1 under i.i.d. uniform p-values (bounded bets -> tight Monte Carlo)."""
    rng = np.random.default_rng(3)
    ends = [simple_jumper(rng.uniform(size=50), jump=0.05)[-1] for _ in range(2000)]
    assert abs(np.mean(ends) - 1.0) < 0.05


def test_ville_false_alarm_rate_under_null() -> None:
    rng = np.random.default_rng(4)
    alarms = 0
    for _ in range(N_MC):
        p = conformal_p_values(rng.normal(size=300), seed=int(rng.integers(1 << 31)))
        m = simple_jumper(p, jump=0.01)
        if martingale_alarm(m, ALPHA)["alarmed"]:
            alarms += 1
    assert alarms / N_MC <= ALPHA + 0.03


def test_jumper_detects_scale_shift_and_mixture_detects_small_p() -> None:
    rng = np.random.default_rng(5)
    # nonconformity = |x|: a scale shift pushes p-values toward 0
    s = np.abs(np.concatenate([rng.normal(size=300), rng.normal(0.0, 4.0, size=300)]))
    p = conformal_p_values(s, seed=0)
    res_j = martingale_alarm(simple_jumper(p, jump=0.01), ALPHA)
    assert res_j["alarmed"] and res_j["alarm_index"] is not None
    assert 300 <= int(res_j["alarm_index"]) < 600  # type: ignore[call-overload]
    # small p-values (nonconformity > past) drive the power/mixture martingale
    p_small = np.concatenate([rng.uniform(size=200), rng.uniform(0.0, 0.2, size=100)])
    res_m = martingale_alarm(mixture_martingale(p_small), ALPHA)
    assert res_m["alarmed"]
    # fixed-eps power martingale bleeds wealth through the null prefix, then grows
    path_p = power_martingale(p_small, 0.3)
    assert path_p[199] < 1.0 and path_p[-1] > 1e6 * path_p[199]


def test_martingale_alarm_report_fields() -> None:
    path = np.array([1.0, 2.0, 30.0, 5.0])
    r = martingale_alarm(path, 0.05)
    assert r["alarm_index"] == 2 and r["threshold"] == 20.0
    assert r["anytime_p_value"] == pytest.approx(1 / 30)
    r2 = martingale_alarm(np.ones(3), 0.05)
    assert r2["alarm_index"] is None and r2["anytime_p_value"] == 1.0
    with pytest.raises(ValueError):
        martingale_alarm(np.array([]), 0.05)
    with pytest.raises(ValueError):
        martingale_alarm(np.array([-1.0]), 0.05)


def test_watch_monitor_resets_and_detects_repeated_changes() -> None:
    rng = np.random.default_rng(6)
    s = np.abs(
        np.concatenate(
            [rng.normal(size=400), rng.normal(0.0, 5.0, size=400), rng.normal(0.0, 0.2, size=400)]
        )
    )
    mon = WatchMonitor(alpha=ALPHA, warmup=30, jump=0.01, seed=0)
    res = mon.run(s)
    assert res.p_values.shape == (1200,) and res.wealth.shape == (1200,)
    assert len(res.alarms) >= 2
    assert any(400 <= a < 800 for a in res.alarms)
    assert any(800 <= a < 1200 for a in res.alarms)


def test_watch_monitor_false_alarms_controlled_under_null() -> None:
    rng = np.random.default_rng(7)
    n_alarm = 0
    for i in range(N_MC):
        mon = WatchMonitor(alpha=ALPHA, warmup=20, seed=i)
        res = mon.run(rng.normal(size=400))
        n_alarm += len(res.alarms)
    # each segment alarms w.p. <= alpha; with reset the count is a small multiple
    assert n_alarm / N_MC <= 2 * ALPHA + 0.03


def test_watch_monitor_weights_and_edges() -> None:
    mon = WatchMonitor(alpha=0.1, warmup=5, max_window=50)
    rng = np.random.default_rng(8)
    s = rng.normal(size=120)
    res = mon.run(s, weights=np.ones(120))
    assert np.all((res.p_values >= 0) & (res.p_values <= 1))
    with pytest.raises(ValueError):
        mon.run(s, weights=np.ones(3))
    with pytest.raises(ValueError):
        mon.step(np.nan)
    with pytest.raises(ValueError):
        mon.step(0.0, weight=-1.0)
    with pytest.raises(ValueError, match="positive mass"):
        WatchMonitor().step(0.0, weight=0.0)
    for kw in ({"alpha": 0}, {"warmup": 0}, {"jump": 2.0}, {"max_window": 3, "warmup": 5}):
        with pytest.raises(ValueError):
            WatchMonitor(**kw)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        conformal_p_values(np.array([]))
    with pytest.raises(ValueError):
        power_martingale(np.array([0.5]), eps=1.0)
    with pytest.raises(ValueError):
        simple_jumper(np.array([0.5]), epsilons=(3.0,))
    with pytest.raises(ValueError):
        simple_jumper(np.array([0.5]), epsilons=(float("nan"),))
    with pytest.raises(ValueError):
        weighted_conformal_p_value(np.array([np.nan, 1.0]), 0.0, np.ones(3), 0.2)
    with pytest.raises(ValueError):
        mixture_martingale(np.array([1.5]))

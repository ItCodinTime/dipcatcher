"""Canon tests: threshold-weighted CRPS."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.twcrps import crps_ensemble, twcrps_ensemble, twcrps_tail


def test_uniform_weight_equals_crps() -> None:
    rng = np.random.default_rng(5)
    for _ in range(20):
        ens = rng.normal(0.0, 1.0, 15)
        y = float(rng.normal(0.0, 2.0))
        assert twcrps_ensemble(ens, y) == pytest.approx(crps_ensemble(ens, y), abs=1e-12)


def test_twcrps_tail_known_answer() -> None:
    # ens = [1, 2, 3], y = 2, tau = 2 -> only segment [2, 3) contributes
    # (F = 2/3, ind = 1) -> (1/3)^2 * 1 = 1/9.
    assert twcrps_tail(np.array([1.0, 2.0, 3.0]), 2.0, 2.0) == pytest.approx(1.0 / 9.0)


def test_twcrps_tail_ignores_below_tau() -> None:
    ens = np.array([0.0, 0.1])
    # identical ens; moving y below tau changes body mass but twCRPS counts
    # only z >= tau, where both cases have F=1 and ind=1 -> zero.
    assert twcrps_tail(ens, -5.0, 1.0) == pytest.approx(0.0)
    # error above tau counts
    assert twcrps_tail(ens, 2.0, 1.0) > 0.0


def test_twcrps_tail_bounded_by_crps_support() -> None:
    rng = np.random.default_rng(9)
    ens = rng.normal(0.0, 1.0, 25)
    y = 1.5
    assert 0.0 <= twcrps_tail(ens, y, 0.0) <= crps_ensemble(ens, y) + 1e-9


def test_twcrps_smooth_weight() -> None:
    ens = np.linspace(-2.0, 2.0, 9)
    y = 0.0
    w_tail = twcrps_ensemble(ens, y, weight=lambda z: np.exp(np.clip(z, -5, 5)))
    w_flat = twcrps_ensemble(ens, y)
    assert w_tail > 0.0 and np.isfinite(w_tail)
    assert w_tail != pytest.approx(w_flat)


def test_twcrps_negative_weight_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        twcrps_ensemble(np.array([0.0, 1.0]), 0.5, weight=lambda z: -np.ones_like(z))


@pytest.mark.parametrize("fn", [twcrps_ensemble, twcrps_tail])
def test_validation(fn) -> None:
    good = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        fn(np.array([1.0]), 0.0, *(1.0,) if fn is twcrps_tail else ())
    with pytest.raises(ValueError):
        fn(np.array([0.0, np.nan]), 0.0, *(1.0,) if fn is twcrps_tail else ())
    with pytest.raises(ValueError):
        fn(good, np.inf, *(1.0,) if fn is twcrps_tail else ())


def test_twcrps_tail_tau_validation() -> None:
    with pytest.raises(ValueError):
        twcrps_tail(np.array([0.0, 1.0]), 0.5, np.inf)

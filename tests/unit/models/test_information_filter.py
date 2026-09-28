"""Canon tests: information / Huber / square-root Kalman filters."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from quant_fund.models.information_filter import (
    huber_kalman,
    information_filter,
    sqrt_kalman,
)

Array = NDArray[np.float64]


def _reference_kalman(
    yy: Array, f: Array, h: Array, q: Array, r: Array, x0: Array, p0: Array
) -> tuple[Array, Array, float]:
    """Textbook covariance-form Kalman filter (test oracle)."""
    n = x0.size
    xs = np.empty((yy.shape[0], n))
    ps = np.empty((yy.shape[0], n, n))
    x, p = x0.copy(), p0.copy()
    nll = 0.0
    for t in range(yy.shape[0]):
        if t > 0:
            x = f @ x
            p = f @ p @ f.T + q
        e = yy[t] - h @ x
        s = h @ p @ h.T + r
        sign, logdet = np.linalg.slogdet(s)
        nll += 0.5 * (logdet + e @ np.linalg.solve(s, e) + e.size * np.log(2 * np.pi))
        k = np.linalg.solve(s.T, (p @ h.T).T).T
        x = x + k @ e
        p = p - k @ s @ k.T
        p = (p + p.T) / 2.0
        xs[t] = x
        ps[t] = p
    return xs, ps, nll


def _local_level(t: int = 60, seed: int = 7) -> Array:
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0.0, 0.1, t))
    return x + rng.normal(0.0, 0.5, t)


def _ssm_args(y: Array, q: float = 0.01, r: float = 0.25) -> tuple:
    f = np.eye(1)
    h = np.eye(1)
    return (
        np.atleast_2d(f),
        np.atleast_2d(h),
        np.array([[q]]),
        np.array([[r]]),
        np.array([y[0]]),
        np.array([[4.0]]),
    )


def test_information_filter_matches_standard_kalman() -> None:
    y = _local_level()
    f, h, q, r, x0, p0 = _ssm_args(y)
    ref_x, ref_p, ref_nll = _reference_kalman(y[:, None], f, h, q, r, x0, p0)
    out = information_filter(y, f, h, q, r, x0, p0)
    np.testing.assert_allclose(out["x"], ref_x, atol=1e-10)
    np.testing.assert_allclose(out["P"], ref_p, atol=1e-10)
    assert out["loglik"] == pytest.approx(-ref_nll, abs=1e-8)


def test_information_filter_converges_to_level() -> None:
    y = np.full(50, 3.0)
    f, h, q, r, x0, p0 = _ssm_args(y, q=1e-8, r=0.1)
    out = information_filter(y, f, h, q, r, np.array([0.0]), np.array([[10.0]]))
    assert abs(float(out["x"][-1, 0]) - 3.0) < 0.2


def test_information_filter_tracks_jump() -> None:
    y = np.concatenate([np.zeros(30), np.full(30, 5.0)])
    f, h, q, r, x0, p0 = _ssm_args(y, q=0.05, r=0.1)
    out = information_filter(y, f, h, q, r, x0, p0)
    assert out["x"][29, 0] < 1.0 < out["x"][-1, 0]


def test_sqrt_kalman_matches_reference() -> None:
    y = _local_level(seed=11)
    f, h, q, r, x0, p0 = _ssm_args(y)
    ref_x, ref_p, _ = _reference_kalman(y[:, None], f, h, q, r, x0, p0)
    out = sqrt_kalman(y, f, h, q, r, x0, p0)
    np.testing.assert_allclose(out["x"], ref_x, atol=1e-8)
    np.testing.assert_allclose(out["P"], ref_p, atol=1e-8)


def test_sqrt_kalman_psd_by_construction() -> None:
    y = _local_level(seed=13)
    f, h, q, r, x0, p0 = _ssm_args(y, q=1e-12)
    out = sqrt_kalman(y, f, h, q, r, x0, p0)
    assert np.linalg.eigvalsh(out["P"][-1]).min() >= -1e-12


def test_huber_kalman_resists_outlier() -> None:
    rng = np.random.default_rng(17)
    y = rng.normal(0.0, 0.3, 80)
    y[40] += 30.0
    f, h, q, r, x0, p0 = _ssm_args(y, q=0.01, r=0.09)
    robust = huber_kalman(y, f, h, q, r, x0, p0, c=1.5)
    plain_x, _, _ = _reference_kalman(y[:, None], f, h, q, r, x0, p0)
    assert abs(robust["x"][40, 0]) < abs(plain_x[40, 0])
    assert abs(robust["x"][40, 0]) < 3.0


def test_huber_kalman_clip_none_behaves_gaussian() -> None:
    y = _local_level(seed=23)
    f, h, q, r, x0, p0 = _ssm_args(y)
    big_clip = huber_kalman(y, f, h, q, r, x0, p0, c=1e6)
    ref_x, _, _ = _reference_kalman(y[:, None], f, h, q, r, x0, p0)
    np.testing.assert_allclose(big_clip["x"], ref_x, atol=1e-8)


@pytest.mark.parametrize(
    "fn",
    [information_filter, sqrt_kalman, huber_kalman],
)
def test_validation_matrix(fn) -> None:
    y = _local_level()
    f, h, q, r, x0, p0 = _ssm_args(y)
    with pytest.raises(ValueError):
        fn(y[:1], f, h, q, r, x0, p0)  # too short
    with pytest.raises(ValueError):
        fn(np.full(30, np.nan), f, h, q, r, x0, p0)
    with pytest.raises(ValueError):
        fn(y, np.eye(2), h, q, r, x0, p0)  # F wrong shape
    with pytest.raises(ValueError):
        fn(y, f, np.eye(2), q, r, x0, p0)  # H wrong shape
    with pytest.raises(ValueError):
        fn(y, f, h, np.eye(3), r, x0, p0)  # Q wrong shape


def test_huber_rejects_bad_clip() -> None:
    y = _local_level()
    f, h, q, r, x0, p0 = _ssm_args(y)
    with pytest.raises(ValueError):
        huber_kalman(y, f, h, q, r, x0, p0, c=0.0)
    with pytest.raises(ValueError):
        huber_kalman(y, f, h, q, r, x0, p0, c=np.inf)

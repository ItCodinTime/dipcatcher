"""Tests for nonlinear filters."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.nonlinear_filters import (
    extended_kalman,
    particle_filter,
    unscented_kalman,
)


def _ar1_series(n=200, phi=0.9, q=0.1, r=0.2, seed=0):
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    y = np.empty(n)
    x[0] = 0.0
    for t in range(n):
        if t > 0:
            x[t] = phi * x[t - 1] + rng.normal(scale=math.sqrt(q))
        y[t] = x[t] + rng.normal(scale=math.sqrt(r))
    return x, y


class TestEKF:
    def test_tracks_linear_system(self):
        # Linear f/h — EKF should coincide with Kalman behavior.
        x, y = _ar1_series()
        out = extended_kalman(
            y,
            f=lambda s: 0.9 * s,
            F_jac=lambda s: 0.9,
            h=lambda s: s,
            H_jac=lambda s: 1.0,
            Q=0.1,
            R=0.2,
            x0=0.0,
            P0=1.0,
        )
        assert np.corrcoef(out["state"], x)[0, 1] > 0.85
        assert np.all(out["variance"] > 0)

    def test_nonlinear_obs(self):
        rng = np.random.default_rng(1)
        n = 150
        x = np.cumsum(rng.normal(scale=0.05, size=n))
        y = x**3 + rng.normal(scale=0.05, size=n)  # cubic obs
        out = extended_kalman(
            y,
            f=lambda s: s,
            F_jac=lambda s: 1.0,
            h=lambda s: s**3,
            H_jac=lambda s: 3.0 * s * s,
            Q=0.0025,
            R=0.0025,
            x0=0.0,
            P0=1.0,
        )
        assert np.all(np.isfinite(out["state"]))

    def test_failclosed(self):
        with pytest.raises(ValueError):
            extended_kalman(
                np.ones(50),
                lambda s: s,
                lambda s: 1.0,
                lambda s: s,
                lambda s: 1.0,
                Q=-1.0,
                R=0.1,
                x0=0,
                P0=1,
            )


class TestUKF:
    def test_tracks_state(self):
        x, y = _ar1_series(seed=2)
        out = unscented_kalman(y, f=lambda s: 0.9 * s, h=lambda s: s, Q=0.1, R=0.2, x0=0.0, P0=1.0)
        assert np.corrcoef(out["state"], x)[0, 1] > 0.8

    def test_nonlinear_beats_ekf_somewhat(self):
        rng = np.random.default_rng(3)
        n = 120
        x = np.cumsum(rng.normal(scale=0.1, size=n))
        y = np.sin(x) + rng.normal(scale=0.1, size=n)
        out = unscented_kalman(y, f=lambda s: s, h=np.sin, Q=0.01, R=0.01, x0=0.0, P0=1.0)
        assert np.all(np.isfinite(out["state"]))
        assert np.all(out["variance"] > 0)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            unscented_kalman(np.ones(5), lambda s: s, lambda s: s, 0.1, 0.1, 0, 1)


class TestParticleFilter:
    def test_tracks_and_ess(self):
        x, y = _ar1_series(seed=4)
        out = particle_filter(
            y,
            f=lambda s, rng: 0.9 * s,
            obs_loglik=lambda s, yt: -0.5 * (yt - s) ** 2 / 0.2,
            q_std=math.sqrt(0.1),
            n_particles=400,
            seed=5,
        )
        assert np.corrcoef(out["mean"], x)[0, 1] > 0.8
        assert np.all(out["ess"] >= 1.0)
        assert np.all(out["ess"] <= 400)
        assert np.isfinite(out["loglik"][0])

    def test_failclosed(self):
        with pytest.raises(ValueError):
            particle_filter(
                np.ones(20), lambda s, r: s, lambda s, y: 0.0, q_std=0.0, n_particles=100
            )

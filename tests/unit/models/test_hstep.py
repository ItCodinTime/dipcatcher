"""HStepScaledDistribution (dip_hstep) — vol-scaled h-step challenger.

SYNTHETIC data only: correctness testing, not market evidence.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

from quant_fund.config import load_config
from quant_fund.models.hstep import MIN_OBS_BUFFER, HStepScaledDistribution
from quant_fund.pipeline import train as train_module
from quant_fund.pipeline.train import train_distribution

TAUS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
T = len(TAUS)
HORIZONS = (1, 5, 20)
N_H = len(HORIZONS)


def _x(n: int, k: int = 2) -> np.ndarray:
    return np.ones((n, k))


def _blocks(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a predict() output into (student_t, empirical) per-h sub-blocks."""
    blocks = q.reshape(q.shape[0], N_H, 2, T)
    return blocks[:, :, 0, :], blocks[:, :, 1, :]


def test_hstep_shapes_monotone_finite() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(0.0005, 0.01, 4000)
    m = HStepScaledDistribution(TAUS).fit(_x(y.size), y)
    q = m.predict(_x(7))
    assert q.shape == (7, 2 * T * N_H)
    assert np.all(np.isfinite(q))
    assert np.all(q == q[0])  # unconditional: rows are tiled copies
    q_st, q_emp = _blocks(q)
    for arr in (q_st, q_emp):
        assert np.all(np.diff(arr, axis=2) >= -1e-9)  # monotone within each h-block
    assert m.metadata().name == "hstep"
    assert m.metadata().extra["horizons"] == list(HORIZONS)


def test_hstep_student_recovers_iid_gaussian_scaling() -> None:
    """iid Gaussian: h-step truth is N(h*mu, h*sigma^2); sqrt(h) vol scaling."""
    rng = np.random.default_rng(1)
    mu, sig = 0.0008, 0.012
    y = rng.normal(mu, sig, 8000)
    m = HStepScaledDistribution(TAUS).fit(_x(y.size), y)
    q_st, _ = _blocks(m.predict(_x(1)))
    tt = np.asarray(TAUS)
    for b, h in enumerate(HORIZONS):
        truth = h * mu + np.sqrt(h) * sig * norm.ppf(tt)
        assert np.allclose(q_st[0, b], truth, atol=0.2 * np.sqrt(h) * sig)


def test_hstep_empirical_tracks_student_on_iid_gaussian() -> None:
    """Overlapping-sum empirical quantiles land near the scaled-t construction."""
    rng = np.random.default_rng(2)
    mu, sig = 0.0005, 0.01
    y = rng.normal(mu, sig, 8000)
    m = HStepScaledDistribution(TAUS).fit(_x(y.size), y)
    q_st, q_emp = _blocks(m.predict(_x(1)))
    for b, h in enumerate(HORIZONS):
        assert np.allclose(q_emp[0, b], q_st[0, b], atol=0.6 * np.sqrt(h) * sig)


def test_hstep_h1_empirical_matches_plain_quantiles() -> None:
    rng = np.random.default_rng(3)
    y = rng.standard_t(df=5, size=6000) * 0.01
    m = HStepScaledDistribution(TAUS).fit(_x(y.size), y)
    _, q_emp = _blocks(m.predict(_x(1)))
    assert np.allclose(q_emp[0, 0], np.quantile(y, TAUS), atol=1e-12)


def test_hstep_recovers_planted_heavy_tails() -> None:
    rng = np.random.default_rng(4)
    y = rng.standard_t(df=4, size=6000) * 0.01
    m = HStepScaledDistribution(TAUS).fit(_x(y.size), y)
    assert m.q_student_ is not None
    # capped MLE picks up tails heavier than Gaussian (t4 -> nu << NU_MAX)
    assert 2.0 < m.nu_ < 15.0
    # both honest constructions agree on the heavy-tail h-step quantiles
    q_st, q_emp = _blocks(m.predict(_x(1)))
    sig_h = np.sqrt(np.asarray(HORIZONS, dtype=float)) * y.std(ddof=1)
    assert np.all(np.abs(q_st[0] - q_emp[0]) <= (0.35 * sig_h)[:, None])


def test_hstep_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not been fitted"):
        HStepScaledDistribution(TAUS).predict(_x(3))


def test_hstep_too_few_obs_fail_closed() -> None:
    n = MIN_OBS_BUFFER + max(HORIZONS) - 1
    with pytest.raises(ValueError, match="finite observations"):
        HStepScaledDistribution(TAUS).fit(_x(n), np.linspace(0, 1, n))


def test_hstep_degenerate_input_fail_closed() -> None:
    y = np.full(200, 1.0)
    with pytest.raises(ValueError, match="positive window variance"):
        HStepScaledDistribution(TAUS).fit(_x(y.size), y)


def test_hstep_bad_constructor_args_fail_closed() -> None:
    with pytest.raises(ValueError, match="taus"):
        HStepScaledDistribution([0.5, 0.4])
    with pytest.raises(ValueError, match="horizons"):
        HStepScaledDistribution(TAUS, horizons=(1, 0, 5))
    with pytest.raises(ValueError, match="horizons"):
        HStepScaledDistribution(TAUS, horizons=())
    with pytest.raises(ValueError, match="horizons"):
        HStepScaledDistribution(TAUS, horizons=(1, float("inf")))


def test_generic_distribution_runner_rejects_hstep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one-step evaluator cannot score multi-horizon output or labels."""
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
    with pytest.raises(ValueError, match="unknown distribution model"):
        train_distribution(cfg, "hstep")

import numpy as np
import pytest

from quant_fund.models.crc import (
    ConformalRiskControl,
    bench_crc_var,
    crc_threshold,
    loss_hit,
)


def test_loss_hit_is_strict() -> None:
    hits = loss_hit(np.array([1.0, 2.0, 2.0]), np.array([1.0, 1.5, 2.0]))
    assert list(hits) == [0.0, 1.0, 0.0]


def test_crc_threshold_hand_computed() -> None:
    losses = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cands = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # λ=2: L̂=0.6 → (5*0.6+1)/6 = 2/3 > 0.5
    # λ=3: L̂=0.4 → (5*0.4+1)/6 = 0.5
    assert crc_threshold(losses, cands, alpha=0.5, B=1.0) == 3.0


def test_base_already_safe_lambda_small() -> None:
    rng = np.random.default_rng(0)
    losses = rng.exponential(scale=0.02, size=200)
    base = np.full_like(losses, 1.0)
    assert float(np.mean(loss_hit(losses, base))) == 0.0
    crc = ConformalRiskControl(alpha=0.05).calibrate(losses, base)
    out = crc.predict_bound(base)
    assert crc.lambda_hat == 0.0
    assert np.allclose(out, base)


def test_tight_base_expands_until_crc_inequality() -> None:
    rng = np.random.default_rng(1)
    losses = rng.exponential(scale=0.05, size=200) + 0.02
    base = np.full_like(losses, 0.005)
    assert float(np.mean(loss_hit(losses, base))) == 1.0
    crc = ConformalRiskControl(alpha=0.05).calibrate(losses, base)
    out = crc.predict_bound(base)
    assert crc.lambda_hat > 0.0
    assert np.all(out >= base)
    hits = loss_hit(losses, out)
    n = int(losses.size)
    crc_stat = (n * float(np.mean(hits)) + 1.0) / (n + 1)
    assert crc_stat <= 0.05 + 1e-12


def test_larger_alpha_tighter_threshold() -> None:
    rng = np.random.default_rng(2)
    losses = rng.exponential(scale=0.04, size=250)
    cands = np.sort(np.unique(losses))
    tight = crc_threshold(losses, cands, alpha=0.20, B=1.0)
    wide = crc_threshold(losses, cands, alpha=0.05, B=1.0)
    assert tight <= wide
    base = np.full_like(losses, float(np.quantile(losses, 0.2)))
    lo = ConformalRiskControl(0.20).calibrate(losses, base)
    hi = ConformalRiskControl(0.05).calibrate(losses, base)
    assert lo.lambda_hat <= hi.lambda_hat
    assert np.all(lo.predict_bound(base) <= hi.predict_bound(base) + 1e-12)


def test_bench_crc_var_seeded_risk_vs_alpha() -> None:
    rng = np.random.default_rng(12)
    losses = rng.exponential(scale=0.04, size=400)
    base = np.full_like(losses, 0.01)
    out = bench_crc_var(losses, base, alpha=0.05)
    assert set(out) >= {"risk", "nominal", "n", "lambda_hat"}
    assert "sharpe" not in {k.lower() for k in out}
    assert out["nominal"] == 0.05
    assert out["n"] == 400.0
    assert out["lambda_hat"] > 0.0
    n = int(out["n"])
    crc_stat = (n * out["risk"] + 1.0) / (n + 1)
    assert crc_stat <= out["nominal"] + 1e-12
    assert out["risk"] < out["nominal"]


def test_crc_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        crc_threshold(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 0.0)
    with pytest.raises(ValueError):
        ConformalRiskControl(alpha=1.0)

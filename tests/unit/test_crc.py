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


# --- Wave 13 CRC extremes ---


def test_crc_threshold_empty_losses_raises() -> None:
    with pytest.raises(ValueError, match="at least one finite"):
        crc_threshold(np.array([]), np.array([1.0, 2.0]), alpha=0.1, B=1.0)
    with pytest.raises(ValueError, match="at least one finite"):
        crc_threshold(np.array([np.nan, np.inf]), np.array([1.0]), alpha=0.1, B=1.0)


def test_crc_threshold_empty_candidates_raises() -> None:
    with pytest.raises(ValueError, match="candidates"):
        crc_threshold(np.array([1.0, 2.0]), np.array([]), alpha=0.1, B=1.0)
    with pytest.raises(ValueError, match="candidates"):
        crc_threshold(np.array([1.0, 2.0]), np.array([np.nan]), alpha=0.1, B=1.0)


def test_crc_b_edges_and_alpha_boundaries() -> None:
    losses = np.array([1.0, 2.0, 3.0])
    cands = np.array([1.0, 2.0, 3.0])
    # B <= 0 / non-finite
    for bad_b in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="B must"):
            crc_threshold(losses, cands, alpha=0.05, B=bad_b)
        with pytest.raises(ValueError, match="B must"):
            ConformalRiskControl(alpha=0.05, B=bad_b)
    # alpha must be in (0, B)
    with pytest.raises(ValueError, match="alpha"):
        crc_threshold(losses, cands, alpha=0.0, B=1.0)
    with pytest.raises(ValueError, match="alpha"):
        crc_threshold(losses, cands, alpha=1.0, B=1.0)
    with pytest.raises(ValueError, match="alpha"):
        crc_threshold(losses, cands, alpha=1.5, B=1.0)
    with pytest.raises(ValueError, match="alpha"):
        ConformalRiskControl(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        ConformalRiskControl(alpha=1.0)
    # alpha == B fails (must be strictly < B)
    with pytest.raises(ValueError, match="alpha"):
        crc_threshold(losses, cands, alpha=2.0, B=2.0)
    # Valid: alpha just below B
    thr = crc_threshold(losses, cands, alpha=1.99, B=2.0)
    assert thr in set(cands.tolist())


def test_crc_threshold_monotone_in_candidates() -> None:
    """Larger α → weakly smaller selected λ; when a feasible λ exists, CRC ≤ α."""
    losses = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    cands = np.sort(np.unique(losses))
    prev = np.inf
    for a in (0.05, 0.10, 0.20, 0.40, 0.60):
        thr = crc_threshold(losses, cands, alpha=a, B=1.0)
        assert thr <= prev + 1e-15
        prev = thr
        n = losses.size
        lhat = float(np.mean(losses > thr))
        crc_stat = (n * lhat + 1.0) / (n + 1)
        # If any candidate is feasible, selected thr must satisfy CRC; else λ_max fallback
        any_ok = any(
            (n * float(np.mean(losses > float(c))) + 1.0) / (n + 1) <= a + 1e-12 for c in cands
        )
        if any_ok:
            assert crc_stat <= a + 1e-12
        else:
            assert thr == float(cands[-1])


def test_crc_threshold_returns_lambda_max_when_none_ok() -> None:
    # Very tight alpha with B=1: even λ_max may fail CRC; then return cands[-1]
    losses = np.ones(5) * 10.0
    cands = np.array([1.0, 2.0, 3.0])  # all bounds << losses → L̂=1
    # (5*1+1)/6 = 1.0; need alpha >= 1 which is invalid for B=1, so use alpha just under 1
    # With L̂=1, crc_stat=1.0 for every cand → none ≤ alpha=0.99 → return λ_max=3
    thr = crc_threshold(losses, cands, alpha=0.99, B=1.0)
    assert thr == 3.0


def test_crc_calibrate_empty_paired_sets_lambda_zero() -> None:
    crc = ConformalRiskControl(alpha=0.05).calibrate(
        np.array([np.nan, np.nan]), np.array([1.0, 1.0])
    )
    assert crc.n_cal == 0
    assert crc.lambda_hat == 0.0
    assert np.allclose(crc.predict_bound(np.array([0.1, 0.2])), np.array([0.1, 0.2]))

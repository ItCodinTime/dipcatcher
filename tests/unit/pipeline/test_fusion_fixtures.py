"""Wave 13: fuse_signals / ablation_inputs closed-form + fail-closed fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.config.models import FusionConfig
from quant_fund.fusion.engine import ablation_inputs, fuse_signals


def _cfg(**kwargs: float) -> FusionConfig:
    return FusionConfig(**kwargs)


def test_fuse_signals_unit_weights_identity() -> None:
    n = 5
    alpha = np.arange(1.0, 1.0 + n)
    conf = np.ones(n)
    regime = np.ones(n)
    risk = np.ones(n)
    tail = np.zeros(n)
    liq = np.zeros(n)
    out = fuse_signals(alpha, conf, regime, risk, tail, liq, _cfg())
    # With all weights 1 and penalties 0: fused = alpha * 1 * 1 / 1 = alpha
    assert np.allclose(out, alpha)


def test_fuse_signals_weight_product_closed_form() -> None:
    alpha = np.array([2.0, 4.0])
    conf = np.array([0.5, 0.25])
    regime = np.array([1.0, 2.0])
    risk = np.array([2.0, 1.0])
    tail = np.array([0.1, 0.2])
    liq = np.array([0.05, 0.0])
    cfg = _cfg(
        alpha_weight=2.0,
        confidence_weight=3.0,
        regime_weight=0.5,
        risk_weight=4.0,
        tail_penalty=10.0,
        liquidity_penalty=20.0,
    )
    out = fuse_signals(alpha, conf, regime, risk, tail, liq, cfg)
    raw = (
        cfg.alpha_weight
        * alpha
        * (cfg.confidence_weight * conf)
        * (cfg.regime_weight * regime)
        / (cfg.risk_weight * risk)
    )
    expected = raw - cfg.tail_penalty * tail - cfg.liquidity_penalty * liq
    assert np.allclose(out, expected)


def test_fuse_signals_risk_floor_fail_closed() -> None:
    # Zero / tiny risk must be clipped to 1e-8 (no div-by-zero / inf)
    alpha = np.array([1.0, 1.0, 1.0])
    ones = np.ones(3)
    zeros_risk = np.array([0.0, 1e-20, 1.0])
    out = fuse_signals(alpha, ones, ones, zeros_risk, np.zeros(3), np.zeros(3), _cfg())
    assert np.all(np.isfinite(out))
    # First two use floor 1e-8 → large but finite
    assert out[0] == pytest.approx(1.0 / 1e-8)
    assert out[1] == pytest.approx(1.0 / 1e-8)
    assert out[2] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "risk",
    [np.array([np.nan, 1.0, 1.0]), np.array([-1.0, 1.0, 1.0])],
)
def test_fuse_signals_rejects_invalid_risk(risk: np.ndarray) -> None:
    ones = np.ones(3)
    with pytest.raises(ValueError):
        fuse_signals(ones, ones, ones, risk, np.zeros(3), np.zeros(3), _cfg())


def test_fuse_signals_penalties_subtract() -> None:
    alpha = np.ones(3)
    ones = np.ones(3)
    base = fuse_signals(alpha, ones, ones, ones, np.zeros(3), np.zeros(3), _cfg())
    with_tail = fuse_signals(alpha, ones, ones, ones, np.full(3, 0.2), np.zeros(3), _cfg())
    with_liq = fuse_signals(alpha, ones, ones, ones, np.zeros(3), np.full(3, 0.3), _cfg())
    assert np.allclose(with_tail, base - 0.2)
    assert np.allclose(with_liq, base - 0.3)


def test_ablation_inputs_named_components() -> None:
    arrays = {
        "regime_compat": np.array([0.2, 0.8]),
        "tail_penalty": np.array([1.0, 2.0]),
        "liq_penalty": np.array([0.5, 0.5]),
        "confidence": np.array([0.3, 0.7]),
        "alpha": np.array([1.0, 2.0]),
    }
    # without_regime → ones
    r = ablation_inputs("without_regime", arrays)
    assert np.allclose(r["regime_compat"], 1.0)
    assert np.allclose(r["tail_penalty"], arrays["tail_penalty"])  # untouched
    # without_tail → zeros
    t = ablation_inputs("without_tail", arrays)
    assert np.allclose(t["tail_penalty"], 0.0)
    # without_liquidity → zeros
    liq = ablation_inputs("without_liquidity", arrays)
    assert np.allclose(liq["liq_penalty"], 0.0)
    # without_confidence → ones
    c = ablation_inputs("without_confidence", arrays)
    assert np.allclose(c["confidence"], 1.0)


def test_ablation_inputs_unknown_name_fails_closed() -> None:
    """A typo must not silently return un-ablated inputs (fake ablation result)."""
    arrays = {
        "regime_compat": np.array([0.2]),
        "tail_penalty": np.array([1.0]),
        "liq_penalty": np.array([0.5]),
        "confidence": np.array([0.3]),
    }
    with pytest.raises(ValueError, match="unknown ablation"):
        ablation_inputs("without_something_else", arrays)


def test_ablation_then_fuse_matches_manual() -> None:
    arrays = {
        "alpha": np.array([1.5, 2.5]),
        "confidence": np.array([0.4, 0.6]),
        "regime_compat": np.array([0.5, 1.0]),
        "predicted_risk": np.array([1.0, 2.0]),
        "tail_penalty": np.array([0.1, 0.2]),
        "liq_penalty": np.array([0.05, 0.1]),
    }
    cfg = _cfg()
    ab = ablation_inputs("without_tail", arrays)
    fused = fuse_signals(
        ab["alpha"],
        ab["confidence"],
        ab["regime_compat"],
        ab["predicted_risk"],
        ab["tail_penalty"],
        ab["liq_penalty"],
        cfg,
    )
    manual = fuse_signals(
        arrays["alpha"],
        arrays["confidence"],
        arrays["regime_compat"],
        arrays["predicted_risk"],
        np.zeros(2),
        arrays["liq_penalty"],
        cfg,
    )
    assert np.allclose(fused, manual)


def test_fuse_signals_broadcast_length_mismatch_raises() -> None:
    # Mismatched shapes must not silently produce a wrong scalar
    alpha = np.array([1.0, 2.0, 3.0])
    short = np.array([1.0, 1.0])
    ones3 = np.ones(3)
    with pytest.raises(ValueError):
        _ = fuse_signals(alpha, short, ones3, ones3, ones3, ones3, _cfg())

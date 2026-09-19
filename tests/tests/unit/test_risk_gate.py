from datetime import UTC, datetime

import pytest

from quant_fund.config import load_config
from quant_fund.portfolio.risk_gate import check_order, resolve_gate_predicted_vol
from quant_fund.schemas.errors import RiskGateRejected
from quant_fund.schemas.orders import Order, OrderSide


def _order(qty: float = 1.0) -> Order:
    now = datetime(2024, 1, 1, tzinfo=UTC)
    return Order(
        order_id="o1",
        security_id="A",
        symbol="A",
        side=OrderSide.BUY,
        quantity=qty,
        signal_time=now,
        decision_time=now,
        order_time=now,
    )


def _kwargs() -> dict[str, object]:
    return {
        "nav": 100.0,
        "price": 1.0,
        "current_weight": 0.0,
        "gross_after": 0.01,
        "net_after": 0.01,
        "participation": 0.01,
        "predicted_vol": 0.1,
    }


def test_risk_gate_rejects_non_finite_inputs() -> None:
    cfg = load_config("configs/research.yaml")
    with pytest.raises(RiskGateRejected, match="non-finite"):
        check_order(_order(), **{**_kwargs(), "price": float("nan")}, config=cfg)


def test_risk_gate_rejects_stale_price_and_model() -> None:
    cfg = load_config("configs/research.yaml")
    with pytest.raises(RiskGateRejected, match="price is stale"):
        check_order(
            _order(), **_kwargs(), config=cfg, price_age_bars=cfg.risk_gate.stale_price_bars + 1
        )
    with pytest.raises(RiskGateRejected, match="model is stale"):
        check_order(
            _order(), **_kwargs(), config=cfg, model_age_hours=cfg.risk_gate.stale_model_hours + 1
        )


def test_risk_gate_rejects_max_gross() -> None:
    cfg = load_config("configs/research.yaml")
    over = cfg.risk_gate.max_gross + 0.01
    with pytest.raises(RiskGateRejected, match="gross"):
        check_order(_order(), **{**_kwargs(), "gross_after": over}, config=cfg)
    # At limit is allowed (strict >).
    check_order(_order(), **{**_kwargs(), "gross_after": cfg.risk_gate.max_gross}, config=cfg)


def test_risk_gate_rejects_nonpositive_nav_price_and_zero_qty() -> None:
    cfg = load_config("configs/research.yaml")
    valid = _order()
    with pytest.raises(RiskGateRejected, match="nav and price must be positive"):
        check_order(_order(), **{**_kwargs(), "nav": 0.0}, config=cfg)
    with pytest.raises(RiskGateRejected, match="nav and price must be positive"):
        check_order(_order(), **{**_kwargs(), "price": -1.0}, config=cfg)
    with pytest.raises(RiskGateRejected, match="quantity must be finite and strictly positive"):
        check_order(valid.model_copy(update={"quantity": 0.0}), **_kwargs(), config=cfg)
    with pytest.raises(RiskGateRejected, match="quantity must be finite and strictly positive"):
        check_order(valid.model_copy(update={"quantity": float("inf")}), **_kwargs(), config=cfg)
    with pytest.raises(RiskGateRejected, match="quantity must be finite and strictly positive"):
        check_order(valid.model_copy(update={"quantity": -1.0}), **_kwargs(), config=cfg)


def test_risk_gate_rejects_negative_ages() -> None:
    cfg = load_config("configs/research.yaml")
    with pytest.raises(RiskGateRejected, match="price age cannot be negative"):
        check_order(_order(), **_kwargs(), config=cfg, price_age_bars=-1)
    with pytest.raises(RiskGateRejected, match="model age cannot be negative"):
        check_order(_order(), **_kwargs(), config=cfg, model_age_hours=-0.1)


def test_risk_gate_passes_clean_order() -> None:
    cfg = load_config("configs/research.yaml")
    check_order(_order(), **_kwargs(), config=cfg, price_age_bars=0, model_age_hours=0.0)


def test_risk_gate_rejects_impossible_nonnegative_metrics() -> None:
    cfg = load_config("configs/research.yaml")
    for key, message in (
        ("gross_after", "gross exposure"),
        ("participation", "participation"),
        ("predicted_vol", "predicted volatility"),
    ):
        with pytest.raises(RiskGateRejected, match=message):
            check_order(_order(), **{**_kwargs(), key: -0.01}, config=cfg)


def test_resolve_gate_predicted_vol_prefers_market_overlay() -> None:
    assert resolve_gate_predicted_vol(0.2, None) == 0.2
    assert resolve_gate_predicted_vol(0.2, 0.01) == 0.01


def test_risk_gate_uses_market_overlay_instead_of_name_vol() -> None:
    cfg = load_config("configs/research.yaml")
    cfg.risk_gate.max_predicted_vol = 0.1
    # Name vol would reject; market overlay is inside the limit.
    check_order(
        _order(),
        **{**_kwargs(), "predicted_vol": 0.5},
        config=cfg,
        market_predicted_vol=0.05,
    )
    with pytest.raises(RiskGateRejected, match="predicted vol 0.5"):
        check_order(
            _order(),
            **{**_kwargs(), "predicted_vol": 0.05},
            config=cfg,
            market_predicted_vol=0.5,
        )


def test_risk_gate_name_vol_still_fail_closed_when_overlay_present() -> None:
    cfg = load_config("configs/research.yaml")
    with pytest.raises(RiskGateRejected, match="predicted volatility"):
        check_order(
            _order(),
            **{**_kwargs(), "predicted_vol": -0.01},
            config=cfg,
            market_predicted_vol=0.05,
        )
    with pytest.raises(RiskGateRejected, match="non-finite"):
        check_order(
            _order(),
            **{**_kwargs(), "predicted_vol": 0.05},
            config=cfg,
            market_predicted_vol=float("nan"),
        )

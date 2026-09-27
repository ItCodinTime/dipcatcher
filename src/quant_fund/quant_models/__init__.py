"""davidalmeida90 quant-models engines inside Dipcatcher.

Analytic pricing, HRP, GEX last-hour *decision* (no IBKR), TSMOM,
Krauss linear window, GKX R². Neural nets stay behind ADR-007.
Research only; ``blend_weight`` stays 0.

pandas and sklearn load with the engines that need them, not with this package.
The CLI imports ``quant_models.cli`` without pulling those libraries.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "LastHourDecision",
    "blume_beta",
    "bs_price",
    "cost_of_equity",
    "crr_american",
    "crr_european",
    "crr_parameters",
    "d1",
    "d2",
    "equal_risk_contribution",
    "fit_svi",
    "flip_level",
    "gbm_paths",
    "gex_at",
    "greeks",
    "hcaa_weights",
    "hedge_error_summary",
    "heston_call",
    "heston_char",
    "heston_implied_vol",
    "heston_put",
    "hrp_weights",
    "implied_volatility",
    "inverse_variance_weights",
    "inverse_vol_weights",
    "krauss_hit_rate",
    "krauss_walk_forward_proba",
    "last_hour_decide",
    "long_only_mean_variance",
    "market_beta",
    "nss_discount",
    "nss_forward",
    "nss_yield",
    "one_day_short_call_hedge",
    "price_bounds",
    "put_call_parity_gap",
    "r2_oos",
    "scaled_greeks",
    "strangle_volga",
    "svi_butterfly_ok",
    "svi_total_variance",
    "tsmom_weights",
    "validate_greeks",
]

_EXPORTS: dict[str, str] = {
    "LastHourDecision": "quant_fund.quant_models.gex",
    "blume_beta": "quant_fund.quant_models.beta",
    "bs_price": "quant_fund.quant_models.black_scholes",
    "cost_of_equity": "quant_fund.quant_models.beta",
    "crr_american": "quant_fund.quant_models.binomial",
    "crr_european": "quant_fund.quant_models.binomial",
    "crr_parameters": "quant_fund.quant_models.binomial",
    "d1": "quant_fund.quant_models.black_scholes",
    "d2": "quant_fund.quant_models.black_scholes",
    "equal_risk_contribution": "quant_fund.quant_models.risk_parity",
    "fit_svi": "quant_fund.quant_models.svi",
    "flip_level": "quant_fund.quant_models.gex",
    "gbm_paths": "quant_fund.quant_models.monte_carlo",
    "gex_at": "quant_fund.quant_models.gex",
    "greeks": "quant_fund.quant_models.greeks",
    "hcaa_weights": "quant_fund.quant_models.hrp",
    "hedge_error_summary": "quant_fund.quant_models.monte_carlo",
    "heston_call": "quant_fund.quant_models.heston",
    "heston_char": "quant_fund.quant_models.heston",
    "heston_implied_vol": "quant_fund.quant_models.heston",
    "heston_put": "quant_fund.quant_models.heston",
    "hrp_weights": "quant_fund.quant_models.hrp",
    "implied_volatility": "quant_fund.quant_models.black_scholes",
    "inverse_variance_weights": "quant_fund.quant_models.hrp",
    "inverse_vol_weights": "quant_fund.quant_models.risk_parity",
    "krauss_hit_rate": "quant_fund.quant_models.krauss",
    "krauss_walk_forward_proba": "quant_fund.quant_models.krauss",
    "last_hour_decide": "quant_fund.quant_models.gex",
    "long_only_mean_variance": "quant_fund.quant_models.mvo",
    "market_beta": "quant_fund.quant_models.beta",
    "nss_discount": "quant_fund.quant_models.nss",
    "nss_forward": "quant_fund.quant_models.nss",
    "nss_yield": "quant_fund.quant_models.nss",
    "one_day_short_call_hedge": "quant_fund.quant_models.monte_carlo",
    "price_bounds": "quant_fund.quant_models.black_scholes",
    "put_call_parity_gap": "quant_fund.quant_models.black_scholes",
    "r2_oos": "quant_fund.quant_models.gkx",
    "scaled_greeks": "quant_fund.quant_models.greeks",
    "strangle_volga": "quant_fund.quant_models.greeks",
    "svi_butterfly_ok": "quant_fund.quant_models.svi",
    "svi_total_variance": "quant_fund.quant_models.svi",
    "tsmom_weights": "quant_fund.quant_models.tsmom",
    "validate_greeks": "quant_fund.quant_models.greeks",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))

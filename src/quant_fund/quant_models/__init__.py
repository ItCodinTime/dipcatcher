"""davidalmeida90 quant-models engines inside Dipcatcher.

Analytic pricing, HRP, GEX last-hour *decision* (no IBKR), TSMOM,
Krauss linear window, GKX R². Neural nets stay behind ADR-007.
Research only; ``blend_weight`` stays 0.
"""

from quant_fund.quant_models.beta import blume_beta, cost_of_equity, market_beta
from quant_fund.quant_models.binomial import crr_american, crr_european, crr_parameters
from quant_fund.quant_models.black_scholes import (
    bs_price,
    d1,
    d2,
    implied_volatility,
    price_bounds,
    put_call_parity_gap,
)
from quant_fund.quant_models.gex import LastHourDecision, flip_level, gex_at, last_hour_decide
from quant_fund.quant_models.gkx import r2_oos
from quant_fund.quant_models.greeks import greeks, scaled_greeks, strangle_volga, validate_greeks
from quant_fund.quant_models.heston import heston_call, heston_char, heston_implied_vol, heston_put
from quant_fund.quant_models.hrp import hcaa_weights, hrp_weights, inverse_variance_weights
from quant_fund.quant_models.krauss import krauss_hit_rate, krauss_walk_forward_proba
from quant_fund.quant_models.monte_carlo import (
    gbm_paths,
    hedge_error_summary,
    one_day_short_call_hedge,
)
from quant_fund.quant_models.mvo import long_only_mean_variance
from quant_fund.quant_models.nss import nss_discount, nss_forward, nss_yield
from quant_fund.quant_models.risk_parity import equal_risk_contribution, inverse_vol_weights
from quant_fund.quant_models.svi import fit_svi, svi_butterfly_ok, svi_total_variance
from quant_fund.quant_models.tsmom import tsmom_weights

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

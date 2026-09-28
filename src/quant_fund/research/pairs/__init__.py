"""Pairs-trading / stat-arb research pack (SYNTHETIC evidence only).

Cointegration screening (Engle-Granger residual ADF + correlation
pre-filter + BH/Bonferroni), hedge-ratio estimation (OLS + scalar Kalman),
Ornstein-Uhlenbeck half-life on the spread, z-score band signals — all
strictly point-in-time (trailing windows only). Proper-score framing only;
nothing here makes a trading, sizing, routing, or live-execution claim.

References: Engle, Granger (1987); Vidyamurthy (2004); Elliott, van der
Hoek, Malcolm (2005); Benjamini, Hochberg (1995).
"""

from quant_fund.research.pairs.cointegration import (
    EG_CV_1PCT,
    EG_CV_5PCT,
    EG_CV_10PCT,
    EGResult,
    benjamini_hochberg,
    bonferroni,
    correlation_prefilter,
    engle_granger_adf,
    screen_pairs,
)
from quant_fund.research.pairs.evaluation import (
    PAIRS_SCHEMA,
    format_pairs_table,
    run_pairs_eval,
    write_pairs_receipt,
)
from quant_fund.research.pairs.fixtures import (
    ar1_spread,
    independent_walks_panel,
    planted_pair_panel,
)
from quant_fund.research.pairs.hedge import kalman_hedge_ratio, ols_hedge_ratio
from quant_fund.research.pairs.pit import pit_pair_signals, rolling_hedge_ratio
from quant_fund.research.pairs.spread import (
    ar1_fit,
    bands_position,
    ou_half_life,
    ou_params,
    zscore_trailing,
)

__all__ = [
    "EG_CV_1PCT",
    "EG_CV_5PCT",
    "EG_CV_10PCT",
    "EGResult",
    "PAIRS_SCHEMA",
    "ar1_fit",
    "ar1_spread",
    "bands_position",
    "benjamini_hochberg",
    "bonferroni",
    "correlation_prefilter",
    "engle_granger_adf",
    "format_pairs_table",
    "independent_walks_panel",
    "kalman_hedge_ratio",
    "ols_hedge_ratio",
    "ou_half_life",
    "ou_params",
    "pit_pair_signals",
    "planted_pair_panel",
    "rolling_hedge_ratio",
    "run_pairs_eval",
    "screen_pairs",
    "write_pairs_receipt",
    "zscore_trailing",
]

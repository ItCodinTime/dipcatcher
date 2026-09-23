"""cosmic-hydra/lightspeed engines inside Dipcatcher.

Frozen TQQQ 20/180 rotation, nautica/stock-momentum top-1 63d book, and
AFML metalabel (reduce-only). Research formulas only — no Alpaca ALL-LIVE.
``blend_weight`` stays 0. Champion remains public ridge.
"""

from quant_fund.lightspeed.ema import sma_seeded_ema
from quant_fund.lightspeed.metalabel import MetaLabelResult, meta_label_gate, metalabel_multiplier
from quant_fund.lightspeed.momentum import momentum_scores, momentum_target_weights
from quant_fund.lightspeed.ranker import NauticaRanker
from quant_fund.lightspeed.rotation import tqqq_target_weights
from quant_fund.lightspeed.specs import (
    frozen_families,
    nautica_momentum_v1,
    stock_momentum_v1,
    tqqq_long_full_v1,
)

__all__ = [
    "MetaLabelResult",
    "NauticaRanker",
    "frozen_families",
    "meta_label_gate",
    "metalabel_multiplier",
    "momentum_scores",
    "momentum_target_weights",
    "nautica_momentum_v1",
    "sma_seeded_ema",
    "stock_momentum_v1",
    "tqqq_long_full_v1",
    "tqqq_target_weights",
]

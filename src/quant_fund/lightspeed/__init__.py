"""cosmic-hydra/lightspeed engines inside Dipcatcher.

Frozen TQQQ 20/180 rotation, nautica/stock-momentum top-1 63d book, and
AFML metalabel (reduce-only). Research formulas only — no Alpaca ALL-LIVE.
``blend_weight`` stays 0. Champion remains public ridge.

Exports load on first use so ``lightspeed.cli`` does not import sklearn.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

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

_EXPORTS: dict[str, str] = {
    "MetaLabelResult": "quant_fund.lightspeed.metalabel",
    "NauticaRanker": "quant_fund.lightspeed.ranker",
    "frozen_families": "quant_fund.lightspeed.specs",
    "meta_label_gate": "quant_fund.lightspeed.metalabel",
    "metalabel_multiplier": "quant_fund.lightspeed.metalabel",
    "momentum_scores": "quant_fund.lightspeed.momentum",
    "momentum_target_weights": "quant_fund.lightspeed.momentum",
    "nautica_momentum_v1": "quant_fund.lightspeed.specs",
    "sma_seeded_ema": "quant_fund.lightspeed.ema",
    "stock_momentum_v1": "quant_fund.lightspeed.specs",
    "tqqq_long_full_v1": "quant_fund.lightspeed.specs",
    "tqqq_target_weights": "quant_fund.lightspeed.rotation",
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

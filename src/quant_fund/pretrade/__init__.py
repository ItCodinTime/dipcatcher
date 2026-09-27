"""Standalone pre-trade risk engine. Shadow mode only.

This package does not submit orders and is not imported by the simulated
broker or the paper loop. See ``docs/PRETRADE_RISK.md``.
"""

from quant_fund.pretrade.codes import (
    HARD_LIMIT_MASK,
    decision_allowed,
    reason_names,
)
from quant_fund.pretrade.config import (
    PretradeConfig,
    load_pretrade_config,
    sign_config,
)
from quant_fund.pretrade.engine import (
    Decision,
    OrderView,
    PretradeEngine,
)
from quant_fund.pretrade.shadow import ShadowRiskAdapter, business_session_id

__all__ = [
    "HARD_LIMIT_MASK",
    "Decision",
    "OrderView",
    "PretradeConfig",
    "PretradeEngine",
    "ShadowRiskAdapter",
    "business_session_id",
    "decision_allowed",
    "load_pretrade_config",
    "reason_names",
    "sign_config",
]

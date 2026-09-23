"""Market-risk engines used by the Artificial Hedge paper book.

pyRisk (lprtk) and pyriskmgmt (Oddo) are implemented in-repo under MIT
attribution. They size and diagnose the paper book. They do not claim live P&L
and they do not move ``blend_weight``.
"""

from quant_fund.risk.gates import (
    GateResult,
    GateSpec,
    apply_gate_stack,
    crash_leverage,
    crc_leverage,
    dd_halt,
    kelly_leverage,
    stepm_leverage,
    vol_target,
)
from quant_fund.risk.overlay import BookRiskOverlay
from quant_fund.risk.pyrisk import BackTesting, ExpectedShortfall, ValueAtRisk
from quant_fund.risk.pyriskmgmt import portfolio_var_es, scale_weights_to_es

__all__ = [
    "BackTesting",
    "BookRiskOverlay",
    "ExpectedShortfall",
    "GateResult",
    "GateSpec",
    "ValueAtRisk",
    "apply_gate_stack",
    "crc_leverage",
    "crash_leverage",
    "dd_halt",
    "kelly_leverage",
    "portfolio_var_es",
    "scale_weights_to_es",
    "stepm_leverage",
    "vol_target",
]

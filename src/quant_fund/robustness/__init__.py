"""Robustness certification for trading strategies.

Threat models, empirical attacks, randomized-smoothing certificates, and
Wasserstein distributionally robust bounds. Research diagnostic only: nothing
here routes orders or claims live performance.

Proven statements and empirical searches are labeled separately. See
``docs/ROBUSTNESS.md``.
"""

from quant_fund.robustness.certify import certify
from quant_fund.robustness.schema import (
    ROBUSTNESS_EXTENSION_SCHEMA_VERSION,
    migrate_robustness_view,
    robustness_extension_errors,
    stamp_robustness,
)

__all__ = [
    "ROBUSTNESS_EXTENSION_SCHEMA_VERSION",
    "certify",
    "migrate_robustness_view",
    "robustness_extension_errors",
    "stamp_robustness",
]

"""Domain errors."""

from __future__ import annotations


class QuantFundError(Exception):
    """Base error."""


class PointInTimeError(QuantFundError):
    """Look-ahead or availability violation."""


class LeakageError(PointInTimeError):
    """Detected use of future information."""


class OptimizationInfeasible(QuantFundError):
    """CVXPY problem infeasible; constraints were not relaxed."""


class RiskGateRejected(QuantFundError):
    """Pre-trade risk gate rejected an order."""


class KillSwitchActive(QuantFundError):
    """Trading halted by kill switch."""


class ConfigError(QuantFundError):
    """Invalid configuration."""


class DataContractError(QuantFundError):
    """Schema or lake contract violation."""

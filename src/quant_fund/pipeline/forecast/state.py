"""Process-local forecast caches and shared aliases.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.models.realized_garch import RealizedGARCHVol
from quant_fund.models.volatility import GARCHVol
from quant_fund.utils.logging import get_logger

if TYPE_CHECKING:
    from .history import ForecastIntervals

Array = NDArray[np.float64]
log = get_logger()
INTERVAL_ALPHA = 0.10
# Process-local caches for causal panel / multi-asof hot paths
_RANKER_CACHE: dict[tuple[str, str], object] = {}
_RL_POLICY_CACHE: dict[tuple[str, str], object] = {}
_GARCH_SPEC_CACHE: dict[tuple[str, str], GARCHVol] = {}
_GARCH_ASOF_CACHE: OrderedDict[tuple[object, ...], object] = OrderedDict()
_GARCH_NAME_ASOF_CACHE: OrderedDict[tuple[object, ...], object] = OrderedDict()
_REALIZED_GARCH_SPEC_CACHE: dict[tuple[str, str], RealizedGARCHVol] = {}
_REALIZED_GARCH_ASOF_CACHE: OrderedDict[tuple[object, ...], object] = OrderedDict()
_PANEL_ASOF_CACHE: dict[tuple[str, float, float], pl.DataFrame] = {}
_CONFORMAL_CACHE: dict[tuple[object, ...], ForecastIntervals] = {}
# Wrappee train-fit reuse: key = train-primary fingerprint + selected family name.
# Family is re-selected on current cal each call; fit reused only when family matches.
_WRAPPEE_CACHE: OrderedDict[tuple[object, ...], object] = OrderedDict()

__all__ = [
    "Array",
    "INTERVAL_ALPHA",
    "log",
]

"""Research-compute helpers: process sweeps and content-hash fit caches.

These helpers do not submit orders or touch broker state.
"""

from quant_fund.compute.cache import FitCache
from quant_fund.compute.parallel import derive_seed, process_map

__all__ = ["FitCache", "derive_seed", "process_map"]

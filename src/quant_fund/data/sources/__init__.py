"""Public and open data source adapters."""

from quant_fund.data.sources.adapters import (
    AlfredSource,
    BeaSource,
    BinanceMarketSource,
    BinancePublicDataSource,
    CcxtSource,
    CftcSource,
    CryptofeedSource,
    Fi2010Source,
    FinaSource,
    FredSource,
    GdeltSource,
    ItchSampleSource,
    OpenBBSource,
    SecEdgarSource,
    TreasurySource,
    WorldBankSource,
)
from quant_fund.data.sources.base import HttpClient, SourceAdapter, SourceError
from quant_fund.data.sources.registry import SOURCE_REGISTRY, get_source, source_names
from quant_fund.data.sources.storage import write_source_frame

__all__ = [
    "AlfredSource",
    "BeaSource",
    "BinanceMarketSource",
    "BinancePublicDataSource",
    "CcxtSource",
    "CftcSource",
    "CryptofeedSource",
    "FinaSource",
    "Fi2010Source",
    "FredSource",
    "GdeltSource",
    "HttpClient",
    "ItchSampleSource",
    "OpenBBSource",
    "SecEdgarSource",
    "SourceAdapter",
    "SourceError",
    "TreasurySource",
    "WorldBankSource",
    "SOURCE_REGISTRY",
    "get_source",
    "source_names",
    "write_source_frame",
]

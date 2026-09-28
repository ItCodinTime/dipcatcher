"""Lazy registry for the public source adapters."""

from __future__ import annotations

from quant_fund.data.sources.adapters import (
    AlfredSource,
    BeaSource,
    BinanceFundingRateSource,
    BinanceMarketSource,
    BinancePerpUniverseSource,
    BinancePublicDataSource,
    BinanceUsdtmPerpSource,
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
from quant_fund.data.sources.base import SourceAdapter

SOURCE_REGISTRY: dict[str, type[SourceAdapter]] = {
    cls.name: cls
    for cls in (
        BinancePublicDataSource,
        BinanceMarketSource,
        BinanceUsdtmPerpSource,
        BinanceFundingRateSource,
        BinancePerpUniverseSource,
        CryptofeedSource,
        CcxtSource,
        ItchSampleSource,
        Fi2010Source,
        GdeltSource,
        SecEdgarSource,
        FredSource,
        AlfredSource,
        TreasurySource,
        CftcSource,
        FinaSource,
        WorldBankSource,
        BeaSource,
        OpenBBSource,
    )
}

ALIASES = {
    "binance": "binance_public_data",
    "binance_ws": "binance_market_websocket",
    "binance_perp": "binance_usdtm_perp",
    "binance_funding": "binance_funding_rate",
    "perp_universe": "binance_perp_universe",
    "nasdaq_itch": "nasdaq_itch",
    "fi-2010": "fi_2010",
    "fred/alfred": "fred",
    "treasury": "us_treasury",
    "cftc": "cftc_cot",
    "finra": "finra_short_sale_volume",
    "ohlcv-1m": "hf_ohlcv_1m",
    "hf-ohlcv-1m": "hf_ohlcv_1m",
    "dolthub": "dolthub_stocks",
    "dolt-stocks": "dolthub_stocks",
    "post-no-preference": "dolthub_stocks",
}


def _install_lazy_sources() -> None:
    """Register adapters that live outside sources/adapters.py (avoid cycles)."""
    if "hf_ohlcv_1m" not in SOURCE_REGISTRY:
        from quant_fund.data.adapters.hf_ohlcv_1m import HfOhlcv1mSource

        SOURCE_REGISTRY[HfOhlcv1mSource.name] = HfOhlcv1mSource
    if "dolthub_stocks" not in SOURCE_REGISTRY:
        from quant_fund.data.adapters.dolthub_stocks import DolthubStocksSource

        SOURCE_REGISTRY[DolthubStocksSource.name] = DolthubStocksSource


def source_names() -> tuple[str, ...]:
    _install_lazy_sources()
    return tuple(sorted(SOURCE_REGISTRY))


def get_source(name: str) -> SourceAdapter:
    _install_lazy_sources()
    key = ALIASES.get(name.strip().lower(), name.strip().lower())
    try:
        cls = SOURCE_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown public data source {name!r}; expected one of: {', '.join(source_names())}"
        ) from exc
    return cls()

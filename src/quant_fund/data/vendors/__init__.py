"""Market-data vendor adapters behind one daily-bars interface.

Each adapter wraps a public/free-tier vendor endpoint in
:class:`~quant_fund.data.vendors.client.VendorClient` (token-bucket pacing,
Retry-After-aware retry with jitter) and normalizes to the repo's canonical
PIT OHLCV frame via ``normalize_ohlcv``. API keys come from env vars only —
never hardcoded — and are resolved lazily so fixtures parse with zero keys.

See ``docs/VENDOR_ADAPTERS.md``.
"""

from __future__ import annotations

from typing import Any

from quant_fund.data.vendors.alphavantage import AlphaVantageDailyAdapter
from quant_fund.data.vendors.base import (
    DailyBarsAdapter,
    VendorBarAdapter,
    empty_bars_frame,
)
from quant_fund.data.vendors.binance import BinanceKlinesAdapter
from quant_fund.data.vendors.client import RetryPolicy, VendorClient, parse_retry_after
from quant_fund.data.vendors.errors import (
    VendorAuthError,
    VendorError,
    VendorHTTPError,
    VendorRateLimitError,
    VendorResponseError,
)
from quant_fund.data.vendors.polygon import PolygonDailyAdapter
from quant_fund.data.vendors.ratelimit import TokenBucket
from quant_fund.data.vendors.stooq import StooqDailyAdapter
from quant_fund.data.vendors.tiingo import TiingoDailyAdapter

VENDOR_ADAPTERS: dict[str, type[DailyBarsAdapter]] = {
    cls.name: cls
    for cls in (
        TiingoDailyAdapter,
        PolygonDailyAdapter,
        AlphaVantageDailyAdapter,
        StooqDailyAdapter,
        BinanceKlinesAdapter,
    )
}


def vendor_adapter_names() -> tuple[str, ...]:
    return tuple(sorted(VENDOR_ADAPTERS))


def get_vendor_adapter(name: str, **kwargs: Any) -> DailyBarsAdapter:
    """Construct a vendor adapter by registry name (see ``vendor_adapter_names``)."""
    key = name.strip().lower()
    try:
        cls = VENDOR_ADAPTERS[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown vendor adapter {name!r}; expected one of: "
            f"{', '.join(vendor_adapter_names())}"
        ) from exc
    return cls(**kwargs)


__all__ = [
    "AlphaVantageDailyAdapter",
    "BinanceKlinesAdapter",
    "DailyBarsAdapter",
    "PolygonDailyAdapter",
    "RetryPolicy",
    "StooqDailyAdapter",
    "TiingoDailyAdapter",
    "TokenBucket",
    "VENDOR_ADAPTERS",
    "VendorAuthError",
    "VendorBarAdapter",
    "VendorClient",
    "VendorError",
    "VendorHTTPError",
    "VendorRateLimitError",
    "VendorResponseError",
    "empty_bars_frame",
    "get_vendor_adapter",
    "parse_retry_after",
    "vendor_adapter_names",
]

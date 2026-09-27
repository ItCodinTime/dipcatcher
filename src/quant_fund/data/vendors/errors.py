"""Error types for the market-data vendor adapter layer.

Hierarchy anchors to the repo's shared bases (``QuantFundError`` /
``ConfigError`` / ``DataContractError``) so callers can catch either the
vendor-specific class or the lab-wide category.
"""

from __future__ import annotations

from quant_fund.schemas.errors import ConfigError, DataContractError, QuantFundError


class VendorError(QuantFundError):
    """Base class for vendor adapter failures."""


class VendorAuthError(VendorError, ConfigError):
    """Missing entitlement: the required API key env var is unset/blank.

    Raised lazily at request build time (never at import or construction),
    so adapters remain usable for offline fixture parsing with zero keys
    configured.
    """


class VendorHTTPError(VendorError):
    """A vendor HTTP request failed after all retries (or non-retryable)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class VendorRateLimitError(VendorHTTPError):
    """The vendor throttled the request (HTTP 429 or a vendor throttle page)."""


class VendorResponseError(VendorError, DataContractError):
    """The vendor payload breached the expected wire shape (fail closed)."""

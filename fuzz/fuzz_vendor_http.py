"""Fuzz ``VendorHttpAdapter`` payload normalization via an injected transport.

The adapter's contract is fail-closed: every malformed payload must raise
``VendorDataError`` (or ``VendorEntitlementError`` at construction). The
transport returns the fuzz bytes verbatim as the HTTP body; the first byte
selects which endpoint method is exercised so one corpus covers bars,
corporate actions, and the security master.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fuzz_common import run

from quant_fund.data.adapters.vendor_http import (
    VendorDataError,
    VendorHttpAdapter,
    VendorHttpConfig,
)

EXPECTED: tuple[type[BaseException], ...] = (VendorDataError,)

_ADAPTER: VendorHttpAdapter | None = None


def _adapter() -> VendorHttpAdapter:
    global _ADAPTER
    if _ADAPTER is None:
        _ADAPTER = VendorHttpAdapter(
            VendorHttpConfig(
                vendor="fuzz",
                base_url="https://vendor.invalid",
                license_acknowledged=True,
                environ={"VENDOR_HTTP_API_KEY": "fuzz-key"},
            ),
            transport=lambda _url, _headers: _BODY[0],
            clock=lambda: datetime(2030, 1, 1, tzinfo=UTC),
        )
    return _ADAPTER


_BODY = [b""]


def test_one_input(data: bytes) -> None:
    if not data:
        return
    _BODY[0] = data[1:]
    adapter = _adapter()
    which = data[0] % 3
    if which == 0:
        adapter.get_bars()
    elif which == 1:
        adapter.get_corporate_actions()
    else:
        adapter.get_security_master()


if __name__ == "__main__":
    raise SystemExit(run("vendor_http", test_one_input, expected=EXPECTED))

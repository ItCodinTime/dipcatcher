"""Expanded public file tape for the Artificial Hedge fund lab.

Yahoo v8 EOD is vendor-adjusted session-close PIT, not SIP vintages. More names
than the SOTA 16-name shell so the book has a cross-section; still not a live
claim and still not 100 GiB of payload.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_fund.data.adapters.yahoo_eod import download_yahoo_universe
from quant_fund.hedge_lab.resources import assert_disk_budget, lab_root

# Liquid US cash + sector + rates/commodity ETFs. Cross-section for a paper book.
HEDGE_LAB_US: tuple[tuple[str, str], ...] = (
    ("SPY", "SPY"),
    ("QQQ", "QQQ"),
    ("IWM", "IWM"),
    ("AAPL", "AAPL"),
    ("MSFT", "MSFT"),
    ("GOOGL", "GOOGL"),
    ("AMZN", "AMZN"),
    ("META", "META"),
    ("NVDA", "NVDA"),
    ("TSLA", "TSLA"),
    ("AVGO", "AVGO"),
    ("JPM", "JPM"),
    ("BAC", "BAC"),
    ("GS", "GS"),
    ("WFC", "WFC"),
    ("JNJ", "JNJ"),
    ("UNH", "UNH"),
    ("LLY", "LLY"),
    ("PFE", "PFE"),
    ("XOM", "XOM"),
    ("CVX", "CVX"),
    ("V", "V"),
    ("MA", "MA"),
    ("PG", "PG"),
    ("KO", "KO"),
    ("PEP", "PEP"),
    ("COST", "COST"),
    ("WMT", "WMT"),
    ("HD", "HD"),
    ("ORCL", "ORCL"),
    ("CRM", "CRM"),
    ("AMD", "AMD"),
    ("INTC", "INTC"),
    ("QCOM", "QCOM"),
    ("DIS", "DIS"),
    ("NFLX", "NFLX"),
    ("CAT", "CAT"),
    ("BA", "BA"),
    ("GE", "GE"),
    ("T", "T"),
    ("TLT", "TLT"),
    ("IEF", "IEF"),
    ("HYG", "HYG"),
    ("LQD", "LQD"),
    ("GLD", "GLD"),
    ("SLV", "SLV"),
    ("XLF", "XLF"),
    ("XLK", "XLK"),
    ("XLE", "XLE"),
    ("XLV", "XLV"),
    ("XLI", "XLI"),
    ("XLP", "XLP"),
    ("XLU", "XLU"),
    ("EFA", "EFA"),
    ("EEM", "EEM"),
)


def fetch_hedge_lab_tape(
    root: Path | None = None,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict[str, Any]:
    """Download the lab universe into ``data/file_us/raw``. Disk budget gated."""
    dest = Path(root) if root is not None else lab_root() / "data" / "file_us" / "raw"
    dest.mkdir(parents=True, exist_ok=True)
    assert_disk_budget(extra_bytes=512 * 1024**2)
    tape = download_yahoo_universe(
        dest,
        HEDGE_LAB_US,
        start=start or datetime(2016, 1, 4, tzinfo=UTC),
        end=end or datetime.now(tz=UTC),
        pause_s=0.12,
    )
    tape["lab_universe"] = "HEDGE_LAB_US"
    tape["n_requested"] = int(len(HEDGE_LAB_US))
    tape["champion_alias"] = False
    tape["sip_vintage"] = False
    return tape

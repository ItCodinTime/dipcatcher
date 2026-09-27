"""Static structure of the hedge-lab tape universes (no network)."""

from __future__ import annotations

from quant_fund.hedge_lab.tape import (
    HEDGE_LAB_US,
    HEDGE_LAB_US_WIDE_SECTORS,
    wide_universe,
)


def test_hedge_lab_us_is_unique_pairs_spy_first() -> None:
    assert HEDGE_LAB_US[0][0] == "SPY"
    tickers = [t for t, _ in HEDGE_LAB_US]
    assert len(tickers) == len(set(tickers))
    assert all(isinstance(t, str) and t for t, name in HEDGE_LAB_US)
    assert len(HEDGE_LAB_US) >= 30


def test_wide_sectors_disjoint_and_nonempty() -> None:
    seen: set[str] = set()
    for sector, tickers in HEDGE_LAB_US_WIDE_SECTORS.items():
        assert isinstance(sector, str) and sector
        assert tickers, f"sector {sector} is empty"
        for t in tickers:
            assert t not in seen, f"{t} appears in two sectors"
            seen.add(t)
    # A few known large caps are on the wide tape.
    for t in ("AAPL", "MSFT", "NVDA", "JPM", "XOM"):
        assert t in seen


def test_wide_universe_spy_first_deduped_mapped() -> None:
    names, sectors = wide_universe()
    tickers = [t for t, _ in names]
    assert tickers[0] == "SPY"
    assert len(tickers) == len(set(tickers))
    assert sectors["SPY"] == "Benchmark"
    # Every name has a sector and every sector label comes from the map.
    for t in tickers:
        assert t in sectors
    for t in sectors:
        assert t == "SPY" or t in set(tickers)
    # Wide tape is strictly larger than the narrow tape.
    assert len(tickers) > len(HEDGE_LAB_US)

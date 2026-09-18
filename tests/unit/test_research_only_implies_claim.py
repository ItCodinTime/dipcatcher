"""research_only True ⇒ claim == research_diagnostic_only (candle + northset)."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    candle_order_book_claim_honesty_errors,
    northset_top_level_claim_honesty_errors,
)


def test_research_only_true_requires_claim_present() -> None:
    assert "northset_claim_missing_while_research_only_true" in (
        northset_top_level_claim_honesty_errors({"research_only": True})
    )
    assert "candle_claim_missing_while_research_only_true" in (
        candle_order_book_claim_honesty_errors(
            {"family": "candle_order_book", "research_only": True}
        )
    )


def test_research_only_true_wrong_claim() -> None:
    assert "northset_claim_not_research_diagnostic_only" in (
        northset_top_level_claim_honesty_errors({"research_only": True, "claim": "live"})
    )
    assert "candle_claim_not_research_diagnostic_only" in (
        candle_order_book_claim_honesty_errors(
            {
                "family": "candle_order_book",
                "research_only": True,
                "claim": "live",
            }
        )
    )


def test_synth_northset_and_candle_claim_coupled() -> None:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    nr = bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)
    cr = bench_candle_order_book(SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars())
    assert nr.get("research_only") is True
    assert cr.get("research_only") is True
    assert northset_top_level_claim_honesty_errors(nr) == []
    assert candle_order_book_claim_honesty_errors(cr) == []

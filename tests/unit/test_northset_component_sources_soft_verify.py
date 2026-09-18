"""Always-on component_sources receipt map soft-verify."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_component_sources_honesty_errors,
)


def _cfg(*, session_l2: bool) -> AppConfig:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = session_l2
    return cfg


def test_synth_session_l2_on_component_sources_clean() -> None:
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        _cfg(session_l2=True),
    )
    cs = receipt["component_sources"]
    assert cs["session_candles"] == "synthetic_reconstruction"
    assert cs["session_book"] == "synthetic_reconstruction"
    assert cs["book"] == receipt["book_source"]
    assert northset_component_sources_honesty_errors(receipt) == []


def test_synth_session_l2_off_session_book_disabled() -> None:
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        _cfg(session_l2=False),
    )
    assert receipt["component_sources"]["session_book"] == "disabled"
    assert northset_component_sources_honesty_errors(receipt) == []


def test_not_dict_fail_closed() -> None:
    assert northset_component_sources_honesty_errors({"component_sources": []}) == [
        "component_sources_not_dict"
    ]


def test_missing_key_fail_closed() -> None:
    errs = northset_component_sources_honesty_errors(
        {
            "component_sources": {
                "bars": "synthetic",
                "book": "synthetic_lob",
                "session_candles": "synthetic_reconstruction",
            }
        }
    )
    assert "component_sources_missing_session_book" in errs


def test_session_book_mismatch_use_session_l2_fail_closed() -> None:
    errs = northset_component_sources_honesty_errors(
        {
            "use_session_l2": True,
            "book_source": "synthetic_lob",
            "component_sources": {
                "bars": "synthetic",
                "book": "synthetic_lob",
                "session_candles": "synthetic_reconstruction",
                "session_book": "disabled",
            },
        }
    )
    assert "component_sources_session_book_mismatch_use_session_l2" in errs


def test_helper_registered() -> None:
    assert northset_component_sources_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS

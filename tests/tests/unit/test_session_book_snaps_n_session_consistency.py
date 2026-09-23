"""mean_session_book_snaps vs n_session_book_rows / n_session_candles consistency."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_SESSION_MEANS_HONESTY_HELPERS,
    northset_session_book_snaps_n_session_consistency_errors,
    northset_session_means_honesty_errors,
)


def _cfg(*, session_l2: bool) -> AppConfig:
    cfg = AppConfig()
    cfg.northset.use_session_l2 = session_l2
    cfg.northset.n_session_candles = 8
    return cfg


def test_synth_session_l2_on_snaps_n_session_consistent() -> None:
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=20, seed=7).get_bars(),
        _cfg(session_l2=True),
    )
    assert float(receipt["mean_session_book_snaps"]) > 0.0
    assert int(receipt["n_session_book_rows"]) == int(receipt["n_session_candles"])
    assert northset_session_book_snaps_n_session_consistency_errors(receipt) == []
    assert "n_session_book_rows_ne_n_session_candles" not in northset_session_means_honesty_errors(
        receipt
    )


def test_synth_session_l2_off_skipped() -> None:
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=20, seed=7).get_bars(),
        _cfg(session_l2=False),
    )
    assert northset_session_book_snaps_n_session_consistency_errors(receipt) == []


def test_rows_zero_despite_mean_fail_closed() -> None:
    errs = northset_session_book_snaps_n_session_consistency_errors(
        {"mean_session_book_snaps": 8.0, "n_session_book_rows": 0}
    )
    assert "n_session_book_rows_non_positive_despite_mean_session_book_snaps" in errs


def test_rows_missing_despite_mean_fail_closed() -> None:
    errs = northset_session_book_snaps_n_session_consistency_errors(
        {"mean_session_book_snaps": 8.0}
    )
    assert "n_session_book_rows_missing_despite_mean_session_book_snaps" in errs


def test_rows_ne_candles_fail_closed() -> None:
    errs = northset_session_book_snaps_n_session_consistency_errors(
        {
            "mean_session_book_snaps": 8.0,
            "n_session_book_rows": 100,
            "n_session_candles": 101,
        }
    )
    assert "n_session_book_rows_ne_n_session_candles" in errs


def test_mean_nan_despite_rows_fail_closed() -> None:
    errs = northset_session_book_snaps_n_session_consistency_errors(
        {
            "mean_session_book_snaps": float("nan"),
            "n_session_book_rows": 50,
            "n_session_candles": 50,
        }
    )
    assert "mean_session_book_snaps_non_positive_despite_n_session_book_rows" in errs


def test_helper_registered_in_session_means_dispatcher() -> None:
    assert (
        northset_session_book_snaps_n_session_consistency_errors
        in NORTHSET_SESSION_MEANS_HONESTY_HELPERS
    )

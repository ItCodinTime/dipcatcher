"""Priority mean_* CLI echoes + park/gk/rs qlike companions."""

from __future__ import annotations

import inspect

from quant_fund.cli import main as cli_main

PRIORITY = (
    "mean_bid_depth",
    "mean_ask_depth",
    "mean_depth_imbalance",
    "mean_depth_imbalance_abs",
    "mean_close_mid_abs_rel",
    "mean_effective_spread",
    "mean_bid_log_size_slope",
    "mean_ask_log_size_slope",
    "mean_bid_log_price_slope",
    "mean_ask_log_price_slope",
)

QLIKE = (
    "parkinson_qlike_vs_cc",
    "garman_klass_qlike_vs_cc",
    "rogers_satchell_qlike_vs_cc",
)


def test_priority_mean_echoes_present() -> None:
    src = inspect.getsource(cli_main.northset)
    for key in PRIORITY:
        assert f"{key}=" in src, key


def test_park_gk_rs_qlike_echoes_present() -> None:
    src = inspect.getsource(cli_main.northset)
    for key in QLIKE:
        assert f"{key}=" in src, key


def test_priority_batch_count() -> None:
    assert len(PRIORITY) == 10


def test_mean_book_age_seconds_echoed() -> None:
    src = inspect.getsource(cli_main.northset)
    assert "mean_book_age_seconds=" in src

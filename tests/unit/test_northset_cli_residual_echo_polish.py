"""CLI northset echoes residual honesty companions."""

from __future__ import annotations

import inspect

from quant_fund.cli import main as cli_main


def test_northset_cli_echoes_residual_companions() -> None:
    src = inspect.getsource(cli_main.northset)
    for token in (
        "gap_finite_rate=",
        "queue_imbalance_mean=",
        "mean_bid_log_size_slope=",
        "n_sweep_high=",
        "overnight_share=",
        "session_mean_rv=",
        "yang_zhang_variance=",
        "sweep_reject_fold_positive_fraction=",
        "mean_fwd_ret_after_high_reclaim=",
        "book_hypothesis_eligible=",
    ):
        assert token in src


def test_mwb_echo_has_trailing_space_before_microprice() -> None:
    src = inspect.getsource(cli_main.northset)
    assert "mean_microprice_weight_balance={blob.get('mean_microprice_weight_balance')} " in src

"""northset CLI echoes the stamped receipt cluster owned by this change."""

from __future__ import annotations

import inspect

from quant_fund.cli.main import northset


def test_northset_cli_source_echoes_receipt_cluster() -> None:
    source = inspect.getsource(northset)
    for key in (
        "mean_bid_log_size_slope",
        "mean_ask_log_size_slope",
        "gap_finite_rate",
        "book_uncrossed_rate",
    ):
        assert f"{key}=" in source
        assert f"blob.get('{key}')" in source

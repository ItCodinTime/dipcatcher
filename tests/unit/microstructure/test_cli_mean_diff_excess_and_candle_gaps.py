"""Northset sweep mean_diff/excess echoes + candle_book CLI receipt parity."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from quant_fund.cli import main as cli_main


def test_northset_echoes_sweep_event_and_control_diff_bps() -> None:
    src = inspect.getsource(cli_main.northset)
    for key in (
        "sweep_reject_event_mean_bps",
        "sweep_follow_event_mean_bps",
        "sweep_reject_control_diff_mean_bps",
        "sweep_follow_control_diff_mean_bps",
        "sweep_reject_liq_control_diff_mean_bps",
        "sweep_follow_liq_control_diff_mean_bps",
        "sweep_follow_oot_holdout_mean_bps",
        "sweep_median_event_adv_participation",
        "sweep_n_counted_trials",
        "sweep_primary_test_id",
    ):
        assert f"{key}=" in src


def test_benches_source_fields_mean_excess_and_diff() -> None:
    """Source row fields mean_excess_bps / mean_diff_bps feed the prefixed receipt keys."""
    benches = Path("src/quant_fund/northset/benches.py").read_text()
    assert "mean_excess_bps" in benches
    assert "mean_diff_bps" in benches


def test_candle_book_cli_echoes_bench_mean_keys() -> None:
    src = inspect.getsource(cli_main.candle_book)
    bench = Path("src/quant_fund/microstructure/bench.py").read_text()
    means = set(re.findall(r'"(mean_[a-z0-9_]+)"\s*:', bench))
    missing = [k for k in sorted(means) if f"{k}=" not in src and f"get('{k}')" not in src]
    assert missing == [], missing


def test_candle_book_echoes_n_fused_min_names() -> None:
    src = inspect.getsource(cli_main.candle_book)
    assert "n_fused=" in src
    assert "min_names=" in src


def test_candle_book_cli_echoes_all_non_ic_receipt_keys() -> None:
    """Every non-ic key stamped by bench_candle_order_book must appear in CLI source."""

    from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
    from quant_fund.microstructure import bench_candle_order_book

    bars = SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars()
    receipt = bench_candle_order_book(bars, book=None, depth=5, seed=7, label="SYNTHETIC")
    src = inspect.getsource(cli_main.candle_book)
    missing = []
    for key in sorted(receipt):
        if str(key).startswith("ic_"):
            continue
        if (
            f"{key}=" in src
            or f"get('{key}')" in src
            or f'get("{key}")' in src
            or f"['{key}']" in src
            or f'["{key}"]' in src
        ):
            continue
        missing.append(key)
    assert missing == [], missing

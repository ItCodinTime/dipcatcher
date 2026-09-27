"""dipcatcher northset CLI echoes identity/reconstruct rate stamps."""

from __future__ import annotations

from pathlib import Path

_CLI = Path("src/quant_fund/cli/main.py")


def _northset_echo_block() -> str:
    text = _CLI.read_text(encoding="utf-8")
    start = text.find('typer.echo("Northset — Dipcatcher')
    assert start > 0
    end = text.find("typer.echo(blob)", start)
    assert end > start
    return text[start:end]


def test_northset_cli_echoes_identity_and_reconstruct_rates() -> None:
    block = _northset_echo_block()
    for key in (
        "ohlc_identity_rate",
        "session_ohlc_identity_rate",
        "book_uncrossed_rate",
        "session_chain_rate",
        "session_reconstructs_daily_rate",
        "session_volume_conservation_rate",
        "structure_finite_rate",
        "gap_finite_rate",
        "mean_bid_log_size_slope",
        "mean_ask_log_size_slope",
    ):
        assert f"{key}=" in block, key

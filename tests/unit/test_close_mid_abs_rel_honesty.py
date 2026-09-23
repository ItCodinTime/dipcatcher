"""close_mid_abs_rel honesty: receipt + soft-verify + CLI echo."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import northset_spread_receipt_honesty_errors


def test_soft_verify_ok_when_effective_matches_quoted() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    receipt = bench_northset(bars, cfg)
    assert northset_spread_receipt_honesty_errors(receipt) == []
    assert "mean_close_mid_abs_rel" in receipt


def test_soft_verify_flags_divergence() -> None:
    blob = {
        "mean_quoted_spread": 0.1,
        "mean_effective_spread": 0.5,  # would be old overwrite behavior
        "mean_close_mid_abs_rel": 0.5,
    }
    errs = northset_spread_receipt_honesty_errors(blob)
    assert "mean_effective_spread_diverges_from_mean_quoted_spread" in errs


def test_northset_cli_echoes_both_means(tmp_path: Path) -> None:
    # This test exercises CLI wiring and receipt labels, not the full 36×504
    # research sweep. Keep the fixture bounded so canonical CI cannot spend
    # minutes rebuilding a heavyweight synthetic session book for two strings.
    data_root = tmp_path / "data"
    config = tmp_path / "northset-test.yaml"
    config.write_text(
        "\n".join(
            [
                "runtime:",
                "  mode: research",
                "data:",
                f"  root: {str(data_root).replace(chr(92), '/')}",
                "  source: synthetic",
                "  synthetic_n_assets: 4",
                "  synthetic_n_days: 24",
                "  synthetic_seed: 5",
                "northset:",
                "  require_adjusted_ohlc: false",
                "  min_names: 3",
                "  use_session_l2: false",
                "  sweep_n_boot: 50",
                "  sweep_n_permutations: 50",
                "  sweep_n_folds: 2",
                "  sweep_min_events: 10",
                "  sweep_min_dates: 10",
                "",
            ]
        )
    )
    result = CliRunner().invoke(app, ["northset", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "mean_effective_spread=" in result.output
    assert "mean_close_mid_abs_rel=" in result.output

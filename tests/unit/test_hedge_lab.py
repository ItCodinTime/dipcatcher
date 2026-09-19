"""Artificial Hedge fund lab — paper/backtest economic catalog."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quant_fund.hedge_lab.resources import DISK_BUDGET_BYTES, assert_disk_budget, ram_plan
from quant_fund.hedge_lab.runner import HedgeLabProtocol, rebalance_dates, run_hedge_lab
from quant_fund.hedge_lab.scoreboard import book_economic_scoreboard, moving_block_bootstrap_ci
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent


def test_disk_budget_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="disk budget"):
        assert_disk_budget(extra_bytes=DISK_BUDGET_BYTES + 1, root=tmp_path)


def test_ram_plan_leaves_headroom() -> None:
    plan = ram_plan()
    assert int(plan["workspace_bytes"]) > 0
    assert int(plan["workspace_bytes"]) < int(plan["total_bytes"]) or int(plan["total_bytes"]) == 0
    if int(plan["total_bytes"]) > 0:
        assert int(plan["headroom_bytes"]) > 0
        assert int(plan["workspace_bytes"]) + int(plan["headroom_bytes"]) <= int(plan["total_bytes"])


def test_economic_scoreboard_not_a_research_family_blob() -> None:
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0004, 0.01, size=252)
    bench = rng.normal(0.0003, 0.009, size=252)
    blob = book_economic_scoreboard(rets, data_source="parquet", benchmark_returns=bench)
    assert blob["catalog"] == "hedge_lab_analytics"
    assert blob["execution_claim"] == "paper_backtest"
    assert np.isfinite(blob["sharpe"])
    assert np.isfinite(blob["calmar"])
    assert np.isfinite(blob["sortino"])
    assert np.isfinite(blob["information_ratio"])
    assert blob["synthetic_not_promotable"] is False
    assert family_blob_forbidden_metrics_absent(blob) is False


def test_research_twin_stays_sharpe_free() -> None:
    twin = {
        "family": "hedge_lab",
        "research_only": True,
        "execution_claim": "research_only",
        "blend_weight": 0.0,
        "champion_alias": False,
        "n_rebalance_dates": 12,
    }
    assert family_blob_forbidden_metrics_absent(twin) is True


def test_moving_block_bootstrap_ci_vectorized() -> None:
    rng = np.random.default_rng(3)
    rets = rng.normal(0.0005, 0.012, size=80)
    out = moving_block_bootstrap_ci(rets, n_boot=8_000, block=8, seed=3)
    assert out["status"] == "ok"
    assert int(out["n_boot"]) >= 64
    assert out["sharpe_p05"] <= out["sharpe_p50"] <= out["sharpe_p95"]
    assert out["calmar_p05"] <= out["calmar_p50"] <= out["calmar_p95"]


def test_lab_artifact_dirs_keep_synthetic_off_public_tree(tmp_path: Path) -> None:
    from quant_fund.config import load_config
    from quant_fund.hedge_lab.runner import _lab_artifact_dirs

    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    dirs = _lab_artifact_dirs(cfg, "SYNTHETIC")
    assert dirs == [tmp_path / "artifacts" / "hedge_lab"]


def test_rebalance_dates_steps() -> None:
    from datetime import datetime

    times = [datetime(2020, 1, 1) for _ in range(3)] + [
        datetime(2020, 1, 2 + i) for i in range(20)
    ]
    picked = rebalance_dates(times, lookback=5, every=5)
    assert picked
    assert len(picked) <= 5


@pytest.mark.synthetic
def test_run_hedge_lab_synthetic_smoke(tmp_path: Path) -> None:
    from quant_fund.config import load_config

    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.synthetic_n_assets = 6
    cfg.data.synthetic_n_days = 90
    cfg.universe.min_history_bars = 8
    cfg.fusion.skip_intervals = True
    cfg.validation.train_bars = 30
    cfg.validation.val_bars = 10
    cfg.validation.test_bars = 10
    cfg.risk_gate.stale_price_bars = 63
    proto = HedgeLabProtocol(
        rebalance_every=30,
        lookback_bars=15,
        bootstrap=True,
        n_boot=4_000,
        claim_ram=False,
    )
    receipt = run_hedge_lab(cfg, proto, claim_ram=False)
    assert receipt["blend_weight"] == 0.0
    assert receipt["execution_claim"] == "paper_backtest"
    assert receipt["synthetic_not_promotable"] is True
    assert receipt["champion_alias"] is False
    assert receipt["economic"]["n_returns"] >= 10
    assert family_blob_forbidden_metrics_absent(receipt["research_twin"]) is True
    assert family_blob_forbidden_metrics_absent(receipt["economic"]) is False
    assert receipt["backtest"].get("book_risk_overlay") is not None
    assert (tmp_path / "metadata" / "hedge_lab_receipt.json").is_file()

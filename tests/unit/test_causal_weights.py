"""P0: causal backtest weights — no end-of-sample broadcast / look-ahead."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from quant_fund.config import load_config
from quant_fund.pipeline.dataset import build_gold
from quant_fund.pipeline.forecast import build_causal_weight_panel, optimize_asof


@pytest.mark.synthetic
def test_early_and_final_weights_differ_when_alphas_evolve(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.synthetic_n_assets = 6
    cfg.data.synthetic_n_days = 120
    feats, _labs = build_gold(cfg)
    dates = feats["event_time"].unique().sort().to_list()
    early, late = dates[20], dates[-1]
    w_early = optimize_asof(cfg, early, persist=False)
    w_late = optimize_asof(cfg, late, persist=False)
    assert w_early["event_time"].unique().to_list() == [early]
    assert w_late["event_time"].unique().to_list() == [late]
    # Join on security_id; alphas (hence weights) should move over time on synthetic
    joined = w_early.select(
        pl.col("security_id"),
        pl.col("target_weight").alias("w_early"),
        pl.col("alpha").alias("a_early"),
    ).join(
        w_late.select(
            pl.col("security_id"),
            pl.col("target_weight").alias("w_late"),
            pl.col("alpha").alias("a_late"),
        ),
        on="security_id",
    )
    assert joined.height > 0
    alpha_changed = (joined["a_early"] - joined["a_late"]).abs().max() > 1e-12
    weight_changed = (joined["w_early"] - joined["w_late"]).abs().max() > 1e-12
    assert alpha_changed or weight_changed, "expected early ≠ final when signals evolve"


@pytest.mark.synthetic
def test_causal_panel_not_broadcast(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.synthetic_n_assets = 4
    cfg.data.synthetic_n_days = 80
    cfg.universe.min_history_bars = 10
    feats, _ = build_gold(cfg)
    dates = feats["event_time"].unique().sort().to_list()
    sample = [dates[10], dates[40], dates[-1]]
    panel = build_causal_weight_panel(cfg, sample)
    assert set(panel["event_time"].unique().to_list()) == set(sample)
    # Broadcasting would make every date share identical weight vectors
    by_date = {
        d: panel.filter(pl.col("event_time") == d).sort("security_id")["target_weight"].to_list()
        for d in sample
    }
    assert by_date[sample[0]] != by_date[sample[-1]] or by_date[sample[0]] != by_date[sample[1]]

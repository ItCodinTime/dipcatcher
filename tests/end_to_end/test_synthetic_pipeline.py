"""End-to-end SYNTHETIC market path. Not evidence of live edge."""

import json
from pathlib import Path

import polars as pl
import pytest

from quant_fund.backtest.engine import run_backtest
from quant_fund.config import load_config
from quant_fund.pipeline.dataset import build_gold, ensure_silver
from quant_fund.pipeline.forecast import build_causal_weight_panel, forecast_asof, optimize_asof
from quant_fund.pipeline.train import train_family


@pytest.mark.synthetic
def test_ingest_train_optimize_backtest(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.synthetic_n_assets = 8
    cfg.data.synthetic_n_days = 180
    feats, labs = build_gold(cfg)
    manifest = json.loads((tmp_path / "metadata" / "data_manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["source"] == "synthetic"
    assert len(manifest["artifacts"]["silver"]["sha256"]) == 64
    assert feats.height > 100
    assert "future_excess_return_5" in labs.columns
    rank = train_family(cfg, "ranking", "ridge")
    assert rank["data_source"] == "SYNTHETIC"
    dist = train_family(cfg, "distribution", "empirical")
    assert "metrics" in dist
    vol = train_family(cfg, "volatility", "ewma")
    assert "qlike" in vol["metrics"]
    state = forecast_asof(cfg)
    assert "SYNTHETIC" in state.notes
    assert state.forecasts
    first = state.forecasts[0]
    assert first.interval_alpha == 0.10
    assert first.interval_method in {"mondrian_cqr", "split_cqr"}
    hz = next(iter(first.interval_lo))
    assert first.interval_lo[hz] < first.interval_hi[hz]
    w = optimize_asof(cfg)
    assert "target_weight" in w.columns
    dates = feats["event_time"].unique().sort().to_list()
    # Causal panel (no end-of-sample broadcast). Subsample for runtime; include ends.
    step = max(1, len(dates) // 15)
    sample = list(dates[::step])
    if dates[0] not in sample:
        sample = [dates[0], *sample]
    if dates[-1] not in sample:
        sample = [*sample, dates[-1]]
    weights = build_causal_weight_panel(cfg, sample)
    assert weights["event_time"].n_unique() == len(sample)
    # Gold features are intentionally universe-filtered; execution/mark bars
    # must remain the complete silver panel so held positions never appear
    # stale merely because a name leaves the research universe.
    bars = ensure_silver(cfg)
    # The synthetic provider deliberately has no liquidation settlement.  Make
    # the test portfolio causal by inserting a zero target on the last valid
    # decision session before each delist, so NEXT_OPEN exits on the event
    # session.  The engine must otherwise fail closed on an unvalued hold.
    actions = pl.read_parquet(tmp_path / "bronze" / "corporate_actions.parquet")
    delists = actions.filter(pl.col("action_type") == "delist")
    if not delists.is_empty():
        all_dates = bars["event_time"].unique().sort().to_list()
        terminal_rows = []
        for action in delists.iter_rows(named=True):
            event_time = action["event_time"]
            prior_dates = [date for date in all_dates if date < event_time]
            assert prior_dates
            terminal_rows.append(
                {
                    "event_time": prior_dates[-1],
                    "security_id": str(action["security_id"]),
                    "target_weight": 0.0,
                }
            )
        terminal = pl.DataFrame(terminal_rows)
        weights = pl.concat(
            [weights.select("event_time", "security_id", "target_weight"), terminal]
        ).unique(subset=["event_time", "security_id"], keep="last", maintain_order=True)
    result = run_backtest(bars, weights, cfg)
    assert result.source_note == "SYNTHETIC"
    assert "sharpe" in result.metrics
    assert result.frictionless is False

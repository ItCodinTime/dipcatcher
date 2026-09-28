"""Smoke tests for the offline demo-data bootstrapper.

``scripts/gen_demo_data.py`` (``make demo-data``) writes a small labeled
SYNTHETIC dataset that the real ``file`` and ``hf_ohlcv_1m`` providers plus
the ingest → gold pipeline must load without vendor keys or downloads.
Results on demo data prove pipeline correctness only — never market evidence.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from quant_fund.config import load_config
from quant_fund.data.adapters.hf_ohlcv_1m import HfOhlcv1mProvider
from quant_fund.data.adapters.parquet import ParquetMarketProvider
from quant_fund.data.ingest import ingest
from quant_fund.pipeline.dataset import build_gold

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "gen_demo_data.py"
FIXTURE = REPO / "tests" / "fixtures" / "demo"

_spec = importlib.util.spec_from_file_location("gen_demo_data", SCRIPT)
assert _spec is not None and _spec.loader is not None
gen_demo_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_demo_data)

pytestmark = pytest.mark.synthetic


def _daily_provider(tmp_path: Path) -> ParquetMarketProvider:
    return ParquetMarketProvider(tmp_path / "daily" / "raw")


def test_committed_fixture_loads_through_file_provider() -> None:
    provider = ParquetMarketProvider(FIXTURE)
    bars = provider.get_bars()
    assert bars.height == 6
    assert bars.schema["event_time"] == pl.Datetime("us", "UTC")
    assert set(bars["source"].to_list()) == {"synthetic"}
    assert set(bars["revision_id"].to_list()) == {"SYNTHETIC"}
    master = provider.get_security_master()
    assert master.height == 2
    assert provider.get_corporate_actions().is_empty()


def test_generator_deterministic_and_labeled(tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    gen_demo_data.generate(out_a, seed=13)
    gen_demo_data.generate(out_b, seed=13)
    manifest_a = json.loads((out_a / "manifest.json").read_text())
    manifest_b = json.loads((out_b / "manifest.json").read_text())
    assert manifest_a == manifest_b
    assert manifest_a["data_label"] == "SYNTHETIC"
    for rel in manifest_a["files"]:
        frame_a = pl.read_parquet(out_a / rel)
        frame_b = pl.read_parquet(out_b / rel)
        assert frame_a.equals(frame_b), rel

    provider = _daily_provider(out_a)
    bars = provider.get_bars()
    assert bars.height == manifest_a["rows"]["daily_bars"]
    # 8 symbols (market + 7 names), all sessions present for every name.
    assert bars["security_id"].n_unique() == 8
    assert set(bars["source"].to_list()) == {"synthetic"}
    assert set(bars["revision_id"].to_list()) == {"SYNTHETIC"}
    assert bars["event_time"].n_unique() == manifest_a["n_daily_sessions"]
    actions = provider.get_corporate_actions()
    assert sorted(actions["action_type"].to_list()) == ["cash_dividend", "split"]
    master = provider.get_security_master()
    assert master.height == 8


def test_minute_cache_loads_through_hf_provider(tmp_path: Path) -> None:
    gen_demo_data.generate(tmp_path, seed=13)
    provider = HfOhlcv1mProvider(
        tmp_path / "minute_cache",
        symbols=["S0001", "S0002"],
        allow_download=False,
    )
    bars = provider.get_bars()
    # 2 tickers x 3 sessions x 390 RTH minutes.
    assert bars.height == 2340
    assert bars.schema["event_time"] == pl.Datetime("us", "UTC")
    assert isinstance(bars["event_time"][0], datetime)
    gaps = provider.quality_report
    assert gaps["n_rth_missing"].sum() == 0


def test_ingest_and_gold_on_demo_config(tmp_path: Path) -> None:
    gen_demo_data.generate(tmp_path, seed=13)
    cfg = load_config(REPO / "configs" / "demo.yaml")
    cfg.data.root = tmp_path / "daily"
    paths = ingest(cfg)
    for name in ("bars", "silver", "universe", "manifest"):
        assert Path(paths[name]).is_file(), name
    manifest = json.loads(Path(paths["manifest"]).read_text())
    assert manifest["source"] == "file"
    silver = pl.read_parquet(paths["silver"])
    assert "close_total_return" in silver.columns
    universe = pl.read_parquet(paths["universe"])
    assert universe["security_id"].n_unique() == 8

    feats, labs = build_gold(cfg)
    assert feats.height > 0
    assert "feature_set_version" in feats.columns
    assert "future_excess_return_5" in labs.columns


def test_demo_backtest_labels_synthetic(tmp_path: Path) -> None:
    """The real causal-weights + event-loop path stays SYNTHETIC on demo data."""
    gen_demo_data.generate(tmp_path, seed=13)
    cfg = load_config(REPO / "configs" / "demo.yaml")
    cfg.data.root = tmp_path / "daily"
    feats, _ = build_gold(cfg)

    from quant_fund.backtest.engine import run_backtest
    from quant_fund.pipeline.dataset import ensure_silver
    from quant_fund.pipeline.forecast import build_causal_weight_panel

    dates = feats["event_time"].unique().sort().to_list()[-3:]
    weights = build_causal_weight_panel(cfg, dates)
    bars = ensure_silver(cfg)
    result = run_backtest(bars, weights, cfg)
    assert result.source_note == "SYNTHETIC"
    assert result.frictionless is False
    assert result.equity.height > 0

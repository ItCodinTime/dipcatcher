"""DayWave9: volatility bench wires research-only e-process DM.

SYNTHETIC tiny panel — live_pnl_claim=false; not a live capital claim.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from quant_fund.config import load_config
from quant_fund.pipeline.dataset import build_gold, panel
from quant_fund.research.benches import bench_volatility
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False


@pytest.fixture(scope="module")
def tiny_synth_panel(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Any]:
    root = tmp_path_factory.mktemp("bench_vol_eprocess")
    cfg = load_config("configs/research.yaml")
    cfg.data.root = root
    cfg.data.synthetic_n_assets = 8
    cfg.data.synthetic_n_days = 80
    build_gold(cfg)
    df = panel(cfg)
    return cfg, df


def test_bench_volatility_exposes_eprocess_dm(tiny_synth_panel: tuple[Any, Any]) -> None:
    """ewma vs rolling DM blob includes e_dm_* research keys; forbidden metrics absent."""
    cfg, df = tiny_synth_panel
    out = bench_volatility(df, cfg)
    assert out, "bench_volatility should score SYNTHETIC gold"
    # Existing DM keys retained
    assert "dm_p" in out and "dm_stat" in out and "dm_preferred" in out
    # DayWave9 e-process fields (research-only)
    assert "e_dm_final" in out
    assert "e_dm_reject" in out
    assert "e_dm_n" in out
    assert math.isfinite(float(out["e_dm_final"])) and float(out["e_dm_final"]) > 0.0
    assert isinstance(out["e_dm_reject"], bool)
    assert int(out["e_dm_n"]) >= 1
    assert out.get("research_only") is True
    # live_pnl_claim key must stay absent (pnl token forbidden in research blobs)
    assert out.get("live_pnl_claim") is not True
    assert "live_pnl_claim" not in out
    assert family_blob_forbidden_metrics_absent(out) is True

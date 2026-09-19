"""DayWave18/19: distribution bench e-process DM on CRPS (unscaled + scaled).

SYNTHETIC tiny panel — live_pnl_claim=false; not a live capital claim.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from quant_fund.config import load_config
from quant_fund.pipeline.dataset import build_gold, panel
from quant_fund.research.benches import bench_distribution
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False


@pytest.fixture(scope="module")
def tiny_synth_panel(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Any]:
    root = tmp_path_factory.mktemp("bench_dist_eprocess")
    cfg = load_config("configs/research.yaml")
    cfg.data.root = root
    cfg.data.synthetic_n_assets = 8
    cfg.data.synthetic_n_days = 80
    build_gold(cfg)
    df = panel(cfg)
    return cfg, df


def test_bench_distribution_exposes_eprocess_dm_crps(tiny_synth_panel: tuple[Any, Any]) -> None:
    """gaussian vs empirical CRPS blob includes e_dm_crps_* when dm_crps_* present."""
    cfg, df = tiny_synth_panel
    out = bench_distribution(df, cfg)
    assert out, "bench_distribution should score SYNTHETIC gold"
    # Existing DM keys retained
    assert "dm_crps_p" in out and "dm_crps_stat" in out and "dm_crps_preferred" in out
    # DayWave18 e-process fields (research-only; prefixed to avoid vol e_dm_* clash)
    assert "e_dm_crps_final" in out
    assert "e_dm_crps_reject" in out
    assert "e_dm_crps_n" in out
    assert math.isfinite(float(out["e_dm_crps_final"])) and float(out["e_dm_crps_final"]) > 0.0
    assert isinstance(out["e_dm_crps_reject"], bool)
    assert int(out["e_dm_crps_n"]) >= 1
    assert out.get("research_only") is True
    # Nested by_horizon primary also carries the same keys
    by_h = out.get("by_horizon")
    assert isinstance(by_h, dict) and by_h
    primary = next(iter(by_h.values()))
    assert "e_dm_crps_final" in primary
    assert primary.get("research_only") is True
    # live_pnl_claim key must stay absent (pnl token forbidden in research blobs)
    assert out.get("live_pnl_claim") is not True
    assert "live_pnl_claim" not in out
    assert family_blob_forbidden_metrics_absent(out) is True


def test_bench_distribution_exposes_scaled_eprocess_dm_crps(
    tiny_synth_panel: tuple[Any, Any],
) -> None:
    """When vol_20 path present, scaled gauss vs t CRPS includes dm/e_dm keys."""
    cfg, df = tiny_synth_panel
    assert "vol_20" in df.columns, "SYNTHETIC gold should carry vol_20 for scaled path"
    out = bench_distribution(df, cfg)
    assert out, "bench_distribution should score SYNTHETIC gold"
    # Unscaled DayWave18 keys retained
    assert "e_dm_crps_final" in out
    assert "e_dm_crps_reject" in out
    assert "e_dm_crps_n" in out
    # DayWave19 scaled DM
    assert "dm_crps_scaled_preferred" in out
    assert "dm_crps_scaled_p" in out
    assert "dm_crps_scaled_stat" in out
    assert out["dm_crps_scaled_preferred"] in (
        "scaled_gaussian",
        "scaled_student_t",
        "tie",
        "inconclusive",
    )
    assert math.isfinite(float(out["dm_crps_scaled_p"]))
    assert math.isfinite(float(out["dm_crps_scaled_stat"]))
    # DayWave19 scaled e-process (omit-on-failure; expect present on SYNTHETIC)
    assert "e_dm_crps_scaled_final" in out
    assert "e_dm_crps_scaled_reject" in out
    assert "e_dm_crps_scaled_n" in out
    assert (
        math.isfinite(float(out["e_dm_crps_scaled_final"]))
        and float(out["e_dm_crps_scaled_final"]) > 0.0
    )
    assert isinstance(out["e_dm_crps_scaled_reject"], bool)
    assert int(out["e_dm_crps_scaled_n"]) >= 1
    assert out.get("research_only") is True
    # Wrappee selection unchanged (still present when scaled path runs)
    assert "wrappee" in out
    by_h = out.get("by_horizon")
    assert isinstance(by_h, dict) and by_h
    primary = next(iter(by_h.values()))
    assert "dm_crps_scaled_p" in primary
    assert "e_dm_crps_scaled_final" in primary
    assert primary.get("research_only") is True
    assert out.get("live_pnl_claim") is not True
    assert "live_pnl_claim" not in out
    assert family_blob_forbidden_metrics_absent(out) is True

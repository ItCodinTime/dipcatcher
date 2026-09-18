"""DayWave8: distribution bench wires closed-form Gaussian CRPS + DM.

SYNTHETIC tiny panel — research-diagnostic only (live_pnl_claim=false).
Quantile-approx keys stay; closed-form reported beside them.
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
    root = tmp_path_factory.mktemp("bench_crps_closed")
    cfg = load_config("configs/research.yaml")
    cfg.data.root = root
    cfg.data.synthetic_n_assets = 8
    cfg.data.synthetic_n_days = 80
    build_gold(cfg)
    df = panel(cfg)
    return cfg, df


def test_bench_distribution_closed_form_crps_finite(tiny_synth_panel: tuple[Any, Any]) -> None:
    """Holdout Gaussian closed-form CRPS is finite; quantile approx keys retained."""
    cfg, df = tiny_synth_panel
    out = bench_distribution(df, cfg)
    assert out, "bench_distribution should score SYNTHETIC gold"
    assert "crps_gaussian" in out  # quantile Riemann approx (unchanged)
    assert "crps_empirical" in out
    assert "crps_gaussian_closed" in out
    closed = float(out["crps_gaussian_closed"])
    approx = float(out["crps_gaussian"])
    assert math.isfinite(closed) and closed >= 0.0
    assert math.isfinite(approx) and approx >= 0.0
    # Closed form and quantile approx should be in the same ballpark on dense taus.
    assert abs(closed - approx) / max(closed, approx, 1e-12) < 2.0
    if "crps_scaled_gaussian_closed" in out:
        sc = float(out["crps_scaled_gaussian_closed"])
        assert math.isfinite(sc) and sc >= 0.0
    if "crps_scaled_student_t_closed" in out:
        stc = float(out["crps_scaled_student_t_closed"])
        assert math.isfinite(stc) and stc >= 0.0
        assert "crps_scaled_student_t" in out  # quantile Riemann retained
    # DM pairwise on per-obs quantile-CRPS losses (gaussian vs empirical)
    assert "dm_crps_preferred" in out
    assert out["dm_crps_preferred"] in {"gaussian", "empirical", "tie", "inconclusive"}
    assert "dm_crps_p" in out
    assert math.isfinite(float(out["dm_crps_p"])) or math.isnan(float(out["dm_crps_p"]))
    assert family_blob_forbidden_metrics_absent(out) is True
    assert out.get("live_pnl_claim") is not True

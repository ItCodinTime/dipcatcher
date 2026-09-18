"""DayWave15–16: bench_tail ES + Christoffersen beside Kupiec.

Research-diagnostic only — live_pnl_claim=false; not a live capital claim.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from quant_fund.config import load_config
from quant_fund.pipeline.dataset import build_gold, panel
from quant_fund.research.benches import bench_tail
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False

_ES_KEYS = (
    "acerbi_szekely_z1",
    "acerbi_szekely_z2",
    "fissler_ziegel_mean",
    "es_hit_count",
)
_KUPIEC_KEYS = (
    "kupiec_lr",
    "kupiec_p",
    "hit_rate",
    "nominal_hit_rate",
    "realized_es",
)
# Analytics-aligned names (var_backtest_hooks): ind/cc before lr/p.
_CHRISTOFFERSEN_KEYS = (
    "christoffersen_ind_lr",
    "christoffersen_ind_p",
    "christoffersen_cc_lr",
    "christoffersen_cc_p",
)


@pytest.fixture(scope="module")
def tiny_synth_panel(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Any]:
    root = tmp_path_factory.mktemp("bench_tail_es")
    cfg = load_config("configs/research.yaml")
    cfg.data.root = root
    cfg.data.synthetic_n_assets = 8
    cfg.data.synthetic_n_days = 60
    build_gold(cfg)
    df = panel(cfg)
    return cfg, df


def _finite_or_nan(x: object) -> bool:
    if x is None:
        return False
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isnan(v) or math.isfinite(v)


def test_bench_tail_es_diagnostics_keys_and_honesty(tiny_synth_panel: tuple[Any, Any]) -> None:
    cfg, df = tiny_synth_panel
    out = bench_tail(df, cfg)
    assert isinstance(out, dict) and out, "bench_tail should be non-empty on SYNTHETIC"
    assert family_blob_forbidden_metrics_absent(out) is True
    assert out.get("research_only") is True
    # live_pnl_claim key must stay absent (pnl token forbidden in research blobs)
    assert "live_pnl_claim" not in out
    assert out.get("live_pnl_claim") is not True
    blob_l = " ".join(str(k).lower() for k in out)
    for bad in ("sharpe", "pnl", "nav"):
        assert bad not in blob_l, f"forbidden token {bad!r} in bench_tail keys"

    for key in _KUPIEC_KEYS:
        assert key in out, f"retained Kupiec key missing: {key}"
    for key in _ES_KEYS:
        assert key in out, f"ES diagnostic key missing: {key}"
        assert _finite_or_nan(out[key]), f"{key} must be finite or honest NaN"
    for key in _CHRISTOFFERSEN_KEYS:
        assert key in out, f"Christoffersen key missing: {key}"
        assert _finite_or_nan(out[key]), f"{key} must be finite or honest NaN"

    # Unscaled mirrors always present; primary matches unscaled when no scaled path,
    # or matches scaled when vol_20 scaled path ran.
    for key in (*_ES_KEYS, *_CHRISTOFFERSEN_KEYS):
        assert f"{key}_unscaled" in out
        assert _finite_or_nan(out[f"{key}_unscaled"])
    if "var_95_scaled" in out:
        for key in (*_ES_KEYS, *_CHRISTOFFERSEN_KEYS):
            assert f"{key}_scaled" in out
            assert out[key] == out[f"{key}_scaled"]
    else:
        for key in (*_ES_KEYS, *_CHRISTOFFERSEN_KEYS):
            assert out[key] == out[f"{key}_unscaled"]


def test_bench_tail_empty_frame_returns_empty() -> None:
    """No future_return label → empty blob (honest skip, not silent zeros)."""
    import polars as pl

    cfg = load_config("configs/research.yaml")
    empty = pl.DataFrame({"date": [], "asset_id": []})
    out = bench_tail(empty, cfg)
    assert out == {}

"""Volatility bench must invoke Diebold–Mariano (MATH_SPEC SOTA)."""

from quant_fund.config.loader import load_config
from quant_fund.pipeline.dataset import build_gold, panel
from quant_fund.research.benches import bench_volatility


def test_bench_volatility_exposes_diebold_mariano(tmp_path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.synthetic_n_assets = 8
    cfg.data.synthetic_n_days = 80
    build_gold(cfg)
    df = panel(cfg)
    out = bench_volatility(df, cfg)
    assert out
    assert "dm_p" in out and "dm_stat" in out and "dm_preferred" in out

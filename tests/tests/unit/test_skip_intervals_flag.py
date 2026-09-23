"""Research-only skip_intervals flag for weight-only smoke benches."""

from quant_fund.config.loader import load_config
from quant_fund.pipeline.forecast import forecast_asof, optimize_asof


def test_skip_intervals_omits_conformal_on_forecast() -> None:
    cfg = load_config("configs/research.yaml")
    cfg.fusion.skip_intervals = True
    cfg.fusion.apply_interval_caps = False
    state = forecast_asof(cfg)
    assert all((not f.interval_lo) and (not f.interval_hi) for f in state.forecasts)


def test_skip_intervals_optimize_still_returns_weights() -> None:
    cfg = load_config("configs/research.yaml")
    cfg.fusion.skip_intervals = True
    cfg.fusion.apply_interval_caps = False
    w = optimize_asof(cfg, persist=False)
    assert w.height > 0
    assert "target_weight" in w.columns

import polars as pl

from quant_fund.backtest.engine import cost_sensitivity
from quant_fund.config import load_config


def test_cost_sensitivity_reports_matched_impact_scenarios() -> None:
    cfg = load_config("configs/research.yaml")
    cfg.costs.frictionless = False
    rows = []
    for day in range(4):
        for sid, price in (("A", 100.0 + day), ("B", 50.0 + day)):
            rows.append(
                {
                    "event_time": f"2024-01-0{day + 1}",
                    "security_id": sid,
                    "open": price,
                    "close": price,
                    "close_total_return": price,
                    "volume": 1_000_000.0,
                    "source": "SYNTHETIC",
                }
            )
    bars = pl.DataFrame(rows)
    weights = pl.DataFrame(
        {
            "event_time": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "security_id": ["A", "A", "A"],
            "target_weight": [0.01, 0.02, 0.01],
        }
    )
    report = cost_sensitivity(bars, weights, cfg)
    assert set(report["scenarios"]) == {"impact_1x", "impact_2x"}
    assert report["claim"] == "execution_diagnostic_only"
    scenarios = report["scenarios"]
    assert scenarios["impact_2x"]["impact_cost"] >= scenarios["impact_1x"]["impact_cost"]


def test_cost_sensitivity_empty_multipliers_raises() -> None:
    import pytest

    cfg = load_config("configs/research.yaml")
    bars = pl.DataFrame(
        {
            "event_time": ["2024-01-01"],
            "security_id": ["A"],
            "open": [100.0],
            "close": [100.0],
            "close_total_return": [100.0],
            "volume": [1_000_000.0],
            "source": ["SYNTHETIC"],
        }
    )
    weights = pl.DataFrame(
        {
            "event_time": ["2024-01-01"],
            "security_id": ["A"],
            "target_weight": [0.01],
        }
    )
    with pytest.raises(ValueError, match="impact_multipliers"):
        cost_sensitivity(bars, weights, cfg, impact_multipliers=())
    with pytest.raises(ValueError, match="impact_multipliers"):
        cost_sensitivity(bars, weights, cfg, impact_multipliers=(-1.0,))


def test_cost_sensitivity_empty_bars_returns_empty_scenarios() -> None:
    cfg = load_config("configs/research.yaml")
    bars = pl.DataFrame(
        {
            "event_time": pl.Series([], dtype=pl.Utf8),
            "security_id": pl.Series([], dtype=pl.Utf8),
            "open": pl.Series([], dtype=pl.Float64),
            "close": pl.Series([], dtype=pl.Float64),
            "close_total_return": pl.Series([], dtype=pl.Float64),
            "volume": pl.Series([], dtype=pl.Float64),
            "source": pl.Series([], dtype=pl.Utf8),
        }
    )
    weights = pl.DataFrame(
        {
            "event_time": pl.Series([], dtype=pl.Utf8),
            "security_id": pl.Series([], dtype=pl.Utf8),
            "target_weight": pl.Series([], dtype=pl.Float64),
        }
    )
    report = cost_sensitivity(bars, weights, cfg)
    assert report["scenarios"] == {}
    assert report["empty"] is True
    assert report["claim"] == "execution_diagnostic_only"
    assert report["live_pnl_claim"] is False
    assert report["research_only"] is True
    assert "sharpe" not in report


def test_cost_sensitivity_mismatched_weights_schema_raises() -> None:
    import pytest

    cfg = load_config("configs/research.yaml")
    bars = pl.DataFrame(
        {
            "event_time": ["2024-01-01"],
            "security_id": ["A"],
            "open": [100.0],
            "close": [100.0],
            "close_total_return": [100.0],
            "volume": [1_000_000.0],
            "source": ["SYNTHETIC"],
        }
    )
    # Missing target_weight
    weights = pl.DataFrame({"event_time": ["2024-01-01"], "security_id": ["A"]})
    with pytest.raises(ValueError, match="weights missing"):
        cost_sensitivity(bars, weights, cfg)


def test_cost_sensitivity_no_sharpe_keys_in_output() -> None:
    cfg = load_config("configs/research.yaml")
    cfg.costs.frictionless = False
    rows = []
    for day in range(4):
        for sid, price in (("A", 100.0 + day), ("B", 50.0 + day)):
            rows.append(
                {
                    "event_time": f"2024-01-0{day + 1}",
                    "security_id": sid,
                    "open": price,
                    "close": price,
                    "close_total_return": price,
                    "volume": 1_000_000.0,
                    "source": "SYNTHETIC",
                }
            )
    bars = pl.DataFrame(rows)
    weights = pl.DataFrame(
        {
            "event_time": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "security_id": ["A", "A", "A"],
            "target_weight": [0.01, 0.02, 0.01],
        }
    )
    report = cost_sensitivity(bars, weights, cfg)
    assert report["live_pnl_claim"] is False
    assert report["research_only"] is True
    assert "sharpe" not in report
    assert "flag_high_sharpe" not in report
    for name, sc in report["scenarios"].items():
        assert "sharpe" not in sc, name
        assert "flag_high_sharpe" not in sc, name
        assert sc["claim"] == "execution_diagnostic_only"

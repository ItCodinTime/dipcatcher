"""Regression checks for stress calibration and report honesty."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.stress.report import (
    _assert_no_forbidden_keys,
    _synthetic_block,
    build_stress_report,
)
from quant_fund.stress.risk import garch_filtered_var_es
from quant_fund.stress.strategy import ResearchStrategy, load_return_panel, strategy_from_mapping


def test_filtered_var_es_is_shift_equivariant_and_forecasts_next_sigma() -> None:
    losses = np.random.default_rng(42).normal(0.0, 1.0, size=120)
    base = garch_filtered_var_es(losses)
    shifted = garch_filtered_var_es(losses + 10.0)
    assert shifted["var"] == pytest.approx(base["var"] + 10.0, abs=1e-10)
    assert shifted["es"] == pytest.approx(base["es"] + 10.0, abs=1e-10)
    assert shifted["sigma_forecast"] == pytest.approx(base["sigma_forecast"], abs=1e-10)
    assert base["sigma_forecast"] > 0.0
    assert base["sigma_last"] > 0.0
    shocked = losses.copy()
    shocked[-1] = 20.0
    after_shock = garch_filtered_var_es(shocked)
    assert after_shock["sigma_forecast"] > after_shock["sigma_last"]


@pytest.mark.parametrize("key", ["sim_pnl", "shock-down-pnl", "NAV_diagnostic"])
def test_composite_forbidden_research_keys_fail_closed(key: str) -> None:
    with pytest.raises(ValueError, match="forbidden research metric key"):
        _assert_no_forbidden_keys({"nested": {key: 1.0}})


def test_merton_calibration_uses_log_returns() -> None:
    rng = np.random.default_rng(7)
    portfolio = rng.normal(0.01, 0.03, size=100)
    block = _synthetic_block(portfolio[:, None], portfolio, 32, 5)
    jump = block["jump_diffusion"]
    assert jump["status"] == "ok"
    assert jump["input_return_convention"] == "log1p(simple_portfolio_return)"
    assert jump["analytic_mean"] == pytest.approx(float(np.log1p(portfolio).mean()), abs=1e-10)


def test_merton_rejects_impossible_simple_return() -> None:
    rng = np.random.default_rng(7)
    portfolio = rng.normal(0.01, 0.03, size=100)
    portfolio[3] = -1.0
    block = _synthetic_block(portfolio[:, None], portfolio, 32, 5)
    assert block["jump_diffusion"]["status"] == "unavailable"
    assert "above -1" in block["jump_diffusion"]["reason"]


def test_return_panel_requires_ordered_dates_and_unique_headers(tmp_path) -> None:
    path = tmp_path / "returns.csv"
    rows = [f"2020-01-{day:02d},{day / 1000:.3f}" for day in range(1, 11)]
    path.write_text("date,a\n" + "\n".join(rows) + "\n", encoding="utf-8")
    names, panel = load_return_panel(path)
    assert names == ("a",)
    assert panel.shape == (10, 1)
    rows[5] = rows[4]
    path.write_text("date,a\n" + "\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dates must increase strictly"):
        load_return_panel(path)
    path.write_text(
        "date,a,a\n" + "\n".join(f"2020-01-{d:02d},0.1,0.2" for d in range(1, 11)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        load_return_panel(path)


def test_report_refuses_nonfinite_panel() -> None:
    strategy: ResearchStrategy = strategy_from_mapping(
        {
            "name": "finite",
            "research_only": True,
            "positions": [{"factor": "us_equity", "weight": 1.0}],
        }
    )
    panel = np.zeros((80, 1))
    panel[10, 0] = np.nan
    with pytest.raises(ValueError, match="returns finite"):
        build_stress_report(strategy, names=("a",), panel=panel, n_scenarios=32)

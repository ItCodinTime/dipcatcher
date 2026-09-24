"""Holdout confirmation helpers for pre-declared tsmom vs ridge."""

from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.hedge_lab.gated_race import CONFIRM_ENGINES, slice_ic_window, slice_path_window
from quant_fund.lightspeed.specs import HOLDOUT_START, SELECTION_END


def test_ml_lane_is_the_preregistered_tree_set() -> None:
    from quant_fund.hedge_lab.gated_race import ML_LANE_ENGINES

    assert ML_LANE_ENGINES == ("ridge", "gbrt", "lambdarank", "xgboost")
    assert CONFIRM_ENGINES == ("ridge", "tsmom", "nautica")
    assert HOLDOUT_START == "2025-01-02"
    assert SELECTION_END == "2024-12-31"


def test_slice_ic_window_uses_frozen_cut() -> None:
    card = {
        "name": "tsmom",
        "ic_dates": ["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"],
        "ic_series": [0.1, 0.2, 0.3, 0.4],
    }
    selection = slice_ic_window(card, start=None, end=SELECTION_END)
    holdout = slice_ic_window(card, start=HOLDOUT_START, end=None)
    assert selection["ic_series"] == [0.1, 0.2]
    assert holdout["ic_series"] == [0.3, 0.4]
    assert selection["n_dates"] == 2
    assert holdout["n_dates"] == 2
    assert selection["mean_ic"] == pytest.approx(0.15)
    assert holdout["mean_ic"] == pytest.approx(0.35)


def test_slice_path_window_aligns_with_ic_cut() -> None:
    dates = ["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"]
    rets = np.array([0.01, 0.02, -0.03, 0.04])
    selection = slice_path_window(dates, rets, start=None, end=SELECTION_END)
    holdout = slice_path_window(dates, rets, start=HOLDOUT_START, end=None)
    assert selection.tolist() == [0.01, 0.02]
    assert holdout.tolist() == [-0.03, 0.04]


def test_ls_cli_confirm_help() -> None:
    result = CliRunner().invoke(app, ["ls", "confirm", "--help"])
    assert result.exit_code == 0, result.output
    assert "holdout" in result.output.lower()
    assert "Not a promotion" in result.output or "promotion" in result.output.lower()

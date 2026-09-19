from datetime import UTC, datetime

import polars as pl
import pytest

from quant_fund.backtest.engine import _target_weight_map
from quant_fund.paper.loop import _default_weights_from_panel


def test_backtest_target_weights_reject_duplicate_date_security_rows() -> None:
    weights = pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 2, tzinfo=UTC)] * 2,
            "security_id": ["A", "A"],
            "target_weight": [0.1, 0.2],
        }
    )

    with pytest.raises(ValueError, match="duplicate target weights"):
        _target_weight_map(weights)


def test_paper_target_weights_reject_duplicate_date_security_rows() -> None:
    weights = pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 2, tzinfo=UTC)] * 2,
            "security_id": ["A", "A"],
            "target_weight": [0.1, 0.2],
        }
    )

    with pytest.raises(ValueError, match="duplicate target weights"):
        _default_weights_from_panel(weights)


def test_target_weight_map_normalizes_security_ids_and_is_stable() -> None:
    dt = datetime(2024, 1, 2, tzinfo=UTC)
    weights = pl.DataFrame(
        {
            "event_time": [dt, dt],
            "security_id": [2, 1],
            "target_weight": [0.2, 0.1],
        }
    )

    assert _target_weight_map(weights) == {"1": 0.1, "2": 0.2}

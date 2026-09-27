"""Decision-time comparisons use UTC instants across CLI and pipeline seams."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from quant_fund.cli.forecast_cmds import _parse_asof
from quant_fund.config.models import AppConfig
from quant_fund.models.calibration import ProbabilityCalibrator
from quant_fund.pipeline.forecast.artifacts import _load_probability_calibrator
from quant_fund.pipeline.kronos import _resolve_asof
from quant_fund.pipeline.train.splits import _stamp_at_or_before, _stamp_strictly_before


def test_cli_and_kronos_normalize_naive_and_offset_asof() -> None:
    naive = _parse_asof("2020-01-02T00:30:00")
    assert naive == datetime(2020, 1, 2, 0, 30, tzinfo=UTC)
    assert _resolve_asof(pl.DataFrame(), naive) == naive

    offset = _parse_asof("2020-01-02T00:30:00+05:30")
    assert offset is not None
    assert _resolve_asof(pl.DataFrame(), offset) == datetime(2020, 1, 1, 19, 0, tzinfo=UTC)


def test_mixed_stamp_comparisons_apply_the_offset() -> None:
    naive_utc = datetime(2020, 1, 2, 0, 30)
    later_instant = datetime(2020, 1, 1, 20, 0, tzinfo=timezone(timedelta(hours=-5)))
    assert _stamp_strictly_before(naive_utc, later_instant)
    assert _stamp_at_or_before(naive_utc, later_instant)
    assert not _stamp_strictly_before(later_instant, naive_utc)
    assert not _stamp_at_or_before(later_instant, naive_utc)


def test_calibrator_fit_end_after_decision_is_rejected_across_offsets(tmp_path) -> None:
    calibrator = ProbabilityCalibrator("platt").fit(np.linspace(0, 1, 20), np.tile([0.0, 1.0], 10))
    calibrator.score_feature = "cs_pct_mom_20"
    calibrator.label = "future_excess_return_5"
    calibrator.horizon = "future_excess_return_5"
    calibrator.fit_start = "2020-01-01"
    calibrator.fit_end = "2020-02-01T23:30:00-05:00"  # 2020-02-02 04:30 UTC
    calibrator.oos_start = "2020-02-02"
    calibrator.oos_end = "2020-02-10"
    calibrator.save(tmp_path / "metadata" / "calibrator_auto.joblib")

    cfg = AppConfig()
    cfg.data.root = tmp_path
    cfg.fusion.probability_calibration_max_age_days = 0
    with pytest.raises(ValueError, match="stale"):
        _load_probability_calibrator(cfg, asof=datetime(2020, 2, 1, 23, 0, tzinfo=UTC))

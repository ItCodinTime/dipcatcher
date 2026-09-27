"""Finite proper-score inputs and receipt-ready JSON for synthetic diagnostics."""

from __future__ import annotations

import json

import numpy as np
import pytest
from numpy.typing import NDArray

from quant_fund.research.explainability import (
    attribution_drift,
    brier,
    build_report,
    partial_dependence_1d,
    permutation_attribution,
    pinball,
    write_report,
)


def _point_predict(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return x[:, 0]


@pytest.mark.parametrize("field", ["feature", "label"])
def test_permutation_rejects_nonfinite_eval_data(field: str) -> None:
    x = np.asarray([[1.0], [2.0], [3.0], [4.0]])
    y = x[:, 0].copy()
    if field == "feature":
        x[0, 0] = np.inf
    else:
        y[0] = np.nan
    with pytest.raises(ValueError, match="x and y must be finite"):
        permutation_attribution(_point_predict, x, y)


def test_nonfinite_prediction_cannot_be_scored() -> None:
    x = np.asarray([[1.0], [2.0], [3.0]])
    y = np.asarray([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="predictions must be finite"):
        permutation_attribution(lambda block: np.full(len(block), np.nan), x, y)
    with pytest.raises(ValueError, match="predictions must be finite"):
        partial_dependence_1d(lambda block: np.full(len(block), np.nan), x, 0)


def test_brier_rejects_invalid_probability_and_label() -> None:
    y = np.asarray([0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="probabilities in"):
        brier()(y, np.asarray([0.1, 1.2, 0.3]))
    with pytest.raises(ValueError, match="labels must be binary"):
        brier()(np.asarray([0.0, 0.5, 1.0]), np.asarray([0.1, 0.5, 0.9]))


def test_point_score_rejects_multiple_prediction_columns() -> None:
    with pytest.raises(ValueError, match="one prediction column"):
        pinball()(np.asarray([1.0, 2.0]), np.ones((2, 2)))


def test_drift_rejects_nat_and_invalid_threshold() -> None:
    x = np.arange(8.0).reshape(-1, 1)
    y = x[:, 0]
    times = np.asarray(
        [
            "2026-01-01",
            "NaT",
            "2026-01-03",
            "2026-01-04",
            "2026-01-05",
            "2026-01-06",
            "2026-01-07",
            "2026-01-08",
        ],
        dtype="datetime64[D]",
    )
    with pytest.raises(ValueError, match="NaT"):
        attribution_drift(_point_predict, x, y, times=times, n_blocks=2)
    with pytest.raises(ValueError, match="thresholds must be finite"):
        attribution_drift(_point_predict, x, y, n_blocks=2, js_ceiling=np.inf)


def test_undefined_constant_rank_correlation_serializes_as_null(tmp_path) -> None:
    x = np.ones((8, 2))
    y = np.zeros(8)
    report = build_report(
        None,
        x,
        y,
        predict=lambda block: np.zeros(len(block)),
        feature_names=["a", "b"],
        n_blocks=2,
        n_repeats=2,
        top_k=1,
        grid_points=2,
        synthetic=True,
        generated_at="2026-09-27T00:00:00+00:00",
    )
    assert report.to_dict()["drift"]["min_spearman"] is None
    paths = write_report(report, tmp_path)
    raw = paths["json"].read_text()
    assert "NaN" not in raw
    assert json.loads(raw)["drift"]["min_spearman"] is None

"""metrics/scoring edge paths: shape/taus guards, invalid-mixture fail-closed
branches, overlap-aware subsample guards, and name-level validation."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.scoring import (
    _mean_finite,
    _require_unique_name_dates,
    _security_id_list,
    crps_from_quantiles,
    crps_gaussian_mixture,
    date_level_equal_weight,
    mean_log_score_gaussian,
    name_level_one_step_density_summary,
    name_level_qlike,
    nonoverlapping_origin_mask,
    one_step_density_summary,
    overlap_aware_qlike,
)

pytestmark = pytest.mark.synthetic


class TestCrpsFromQuantiles:
    def test_shape_and_taus_guards(self) -> None:
        y = np.asarray([0.01, 0.02])
        q = np.asarray([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
        taus = np.asarray([0.1, 0.5, 0.9])
        with pytest.raises(ValueError, match="2d"):
            crps_from_quantiles(y, np.asarray([0.1, 0.2]), taus)
        with pytest.raises(ValueError, match="matching taus"):
            crps_from_quantiles(y, q, np.asarray([0.1, 0.5]))
        with pytest.raises(ValueError, match="taus"):
            crps_from_quantiles(y, q, np.asarray([0.0, 0.5, 1.0]))
        with pytest.raises(ValueError, match="taus"):
            crps_from_quantiles(y, q, np.asarray([0.9, 0.5, 0.1]))
        with pytest.raises(ValueError, match="length mismatch"):
            crps_from_quantiles(np.asarray([0.01]), q, taus)

    def test_empty_y(self) -> None:
        out = crps_from_quantiles(np.asarray([]), np.empty((0, 2)), np.asarray([0.3, 0.7]))
        assert not np.isfinite(out)


class TestMixture:
    def test_empty_and_invalid_weights_fail_closed(self) -> None:
        out = crps_gaussian_mixture(
            np.asarray([]), np.asarray([1.0]), np.asarray([0.0]), np.asarray([1.0])
        )
        assert out.size == 0
        out = crps_gaussian_mixture(
            np.asarray([0.0, 0.1]),
            np.asarray([0.3, 0.3]),  # does not sum to 1
            np.asarray([0.0, 0.0]),
            np.asarray([1.0, 1.0]),
        )
        assert not np.isfinite(out).all()
        out = crps_gaussian_mixture(
            np.asarray([0.0, 0.1]),
            np.asarray([0.5, -0.5]),  # negative weight
            np.asarray([0.0, 0.0]),
            np.asarray([1.0, 1.0]),
        )
        assert not np.isfinite(out).all()

    def test_valid_mixture(self) -> None:
        out = crps_gaussian_mixture(
            np.asarray([0.0, 0.5]),
            np.asarray([0.5, 0.5]),
            np.asarray([0.0, 0.2]),
            np.asarray([1.0, 1.5]),
        )
        assert np.isfinite(out).all()

    def test_mean_log_score_edges(self) -> None:
        assert not np.isfinite(
            mean_log_score_gaussian(np.asarray([]), np.asarray([]), np.asarray([]))
        )
        assert not np.isfinite(
            mean_log_score_gaussian(
                np.asarray([float("nan")]), np.asarray([0.0]), np.asarray([1.0])
            )
        )


class TestDateLevel:
    def test_length_mismatch_and_empty(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            date_level_equal_weight(
                np.asarray(["2020-01-01"], dtype=object), np.asarray([1.0, 2.0])
            )
        dates, values = date_level_equal_weight(np.asarray([], dtype=object), np.asarray([]))
        assert dates.size == 0 and values.size == 0
        dates, values = date_level_equal_weight(
            np.asarray(["2020-01-01"], dtype=object), np.asarray([float("nan")])
        )
        assert dates.size == 0 and values.size == 0


class TestNonoverlapMask:
    @pytest.mark.parametrize(
        ("positions", "match"),
        [
            (np.asarray([[1, 2]]), "1d"),
            (np.asarray([0.5, 1.0]), "integer-valued"),
            (np.asarray([1, 1]), "strictly increasing"),
            (np.asarray([3, 1]), "strictly increasing"),
        ],
    )
    def test_guards(self, positions, match) -> None:
        with pytest.raises(ValueError, match=match):
            nonoverlapping_origin_mask(positions, 2)

    def test_empty_and_horizon(self) -> None:
        out = nonoverlapping_origin_mask(np.asarray([], dtype=int), 2)
        assert out.size == 0
        keep = nonoverlapping_origin_mask(np.arange(6), 3)
        assert keep.sum() == 2  # origins 0 and 3


class TestOverlapAwareQlike:
    def _data(self, n: int = 6):
        rng = np.random.default_rng(7)
        dates = np.asarray([f"2020-01-{i + 1:02d}" for i in range(n)], dtype=object)
        return dates, np.abs(rng.normal(0.001, 0.0005, n)), np.abs(rng.normal(0.001, 0.0005, n))

    def test_no_finite_observations(self) -> None:
        dates, yhat, y = self._data()
        with pytest.raises(ValueError, match="no finite date-level"):
            overlap_aware_qlike(dates, np.full_like(yhat, np.nan), y, horizon_bars=1)

    def test_session_index_missing_date(self) -> None:
        dates, yhat, y = self._data()
        with pytest.raises(ValueError, match="session_index missing"):
            overlap_aware_qlike(dates, yhat, y, horizon_bars=1, session_index={"2020-01-01": 0})

    def test_happy_with_session_index(self) -> None:
        dates, yhat, y = self._data()
        session = {d: i for i, d in enumerate(dates.tolist())}
        out = overlap_aware_qlike(dates, yhat, y, horizon_bars=2, session_index=session)
        assert np.isfinite(out["qlike"])


class TestDensitySummary:
    def _inputs(self, n: int = 5):
        rng = np.random.default_rng(2)
        dates = np.asarray([f"2020-02-{i + 1:02d}" for i in range(n)], dtype=object)
        logs = rng.normal(-5, 0.5, n)
        crps = np.abs(rng.normal(0.01, 0.002, n))
        pits = rng.uniform(0, 1, n)
        return dates, logs, crps, pits

    def test_length_guards(self) -> None:
        dates, logs, crps, pits = self._inputs()
        with pytest.raises(ValueError, match="same length"):
            one_step_density_summary(dates, logs[:2], crps, pits, horizon_bars=1)
        with pytest.raises(ValueError, match="same length"):
            one_step_density_summary(dates, logs, crps, pits[:2], horizon_bars=1)

    def test_empty_and_duplicates(self) -> None:
        dates, logs, crps, pits = self._inputs(0)
        with pytest.raises(ValueError, match="at least one origin"):
            one_step_density_summary(dates, logs, crps, pits, horizon_bars=1)
        dates, logs, crps, pits = self._inputs(4)
        dates[1] = dates[0]
        with pytest.raises(ValueError, match="unique dates"):
            one_step_density_summary(dates, logs, crps, pits, horizon_bars=1)

    def test_session_index_missing(self) -> None:
        dates, logs, crps, pits = self._inputs()
        with pytest.raises(ValueError, match="session_index missing"):
            one_step_density_summary(
                dates, logs, crps, pits, horizon_bars=1, session_index={"x": 0}
            )

    def test_happy(self) -> None:
        dates, logs, crps, pits = self._inputs(30)
        out = one_step_density_summary(dates, logs, crps, pits, horizon_bars=2)
        assert "ignorance_one_step" in out


class TestNameLevel:
    def test_security_id_list_guards(self) -> None:
        with pytest.raises(ValueError, match="1d"):
            _security_id_list(np.asarray([["a", "b"]], dtype=object))
        with pytest.raises(ValueError, match="strings"):
            _security_id_list(np.asarray([1, 2], dtype=object))
        with pytest.raises(ValueError, match="non-empty"):
            _security_id_list(np.asarray([""], dtype=object))

    def test_unique_name_dates(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            _require_unique_name_dates(["a", "a"], np.asarray(["d1", "d1"], dtype=object))

    def test_name_level_qlike_guards(self) -> None:
        ids = np.asarray(["a", "b"], dtype=object)
        dates = np.asarray(["2020-01-01", "2020-01-02"], dtype=object)
        yhat = np.asarray([1.0, 2.0])
        y = np.asarray([1.0, 2.0])
        with pytest.raises(ValueError, match="same length"):
            name_level_qlike(ids, dates[:1], yhat, y, horizon_bars=1)
        with pytest.raises(ValueError, match="at least one"):
            name_level_qlike(
                np.asarray([], dtype=object),
                np.asarray([], dtype=object),
                np.asarray([]),
                np.asarray([]),
                horizon_bars=1,
            )

    def test_name_level_qlike_happy(self) -> None:
        ids = np.asarray(["a"] * 6 + ["b"] * 6, dtype=object)
        dates = np.asarray([f"2020-01-0{i + 1}" for i in range(6)] * 2, dtype=object)
        rng = np.random.default_rng(3)
        yhat = np.abs(rng.normal(0.001, 0.0005, 12))
        y = np.abs(rng.normal(0.001, 0.0005, 12))
        out = name_level_qlike(ids, dates, yhat, y, horizon_bars=2)
        assert np.isfinite(out["qlike"])

    def test_name_level_density_guards(self) -> None:
        ids = np.asarray(["a"], dtype=object)
        dates = np.asarray(["2020-01-01"], dtype=object)
        one = np.asarray([1.0])
        with pytest.raises(ValueError, match="same length"):
            name_level_one_step_density_summary(ids, dates[:0], one, one, one, horizon_bars=1)
        with pytest.raises(ValueError, match="same length"):
            name_level_one_step_density_summary(
                ids, dates, np.asarray([1.0, 2.0]), one, one, horizon_bars=1
            )

    def test_mean_finite_empty(self) -> None:
        assert not np.isfinite(_mean_finite(np.asarray([float("nan")])))

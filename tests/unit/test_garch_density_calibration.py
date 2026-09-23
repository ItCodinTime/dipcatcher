"""Day Wave 105: one-step GARCH density scores (CRPS / log-score / PIT).

Research-diagnostic identities only. live_pnl_claim=false; not a live edge.
The one-step date-level return law is not the h-bar realized-variance QLIKE
target and is not a Gaussian approximation to cumulative variance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm
from scipy.stats import t as student_t

from quant_fund.metrics.scoring import (
    crps_gaussian,
    log_score_gaussian,
    mean_log_score_gaussian,
    one_step_density_summary,
)
from quant_fund.models.volatility import GARCHVol
from quant_fund.pipeline.train import _garch_oos_predictions
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False


def _returns(n: int = 240, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, n)


def test_log_score_gaussian_standard_normal_at_median() -> None:
    expected = -0.5 * math.log(2.0 * math.pi)
    out = log_score_gaussian(np.array([0.0]), np.array([0.0]), np.array([1.0]))
    assert out[0] == pytest.approx(expected)
    assert mean_log_score_gaussian(
        np.array([0.0]), np.array([0.0]), np.array([1.0])
    ) == pytest.approx(expected)


def test_log_score_gaussian_jacobian_identity() -> None:
    y, mu, sigma = 0.02, 0.01, 0.03
    z = (y - mu) / sigma
    unit = log_score_gaussian(np.array([z]), np.array([0.0]), np.array([1.0]))[0]
    scaled = log_score_gaussian(np.array([y]), np.array([mu]), np.array([sigma]))[0]
    assert scaled == pytest.approx(unit - math.log(sigma))


def test_log_score_gaussian_empty_mismatch_bad_sigma() -> None:
    assert log_score_gaussian(np.array([]), np.array([]), np.array([])).size == 0
    assert np.isnan(mean_log_score_gaussian(np.array([]), np.array([]), np.array([])))
    with pytest.raises(ValueError, match="length mismatch"):
        log_score_gaussian(np.array([0.0, 1.0]), np.array([0.0]), np.array([1.0]))
    out = log_score_gaussian(
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, -1.0, np.nan]),
    )
    assert np.isfinite(out[0])
    assert np.isnan(out[1]) and np.isnan(out[2]) and np.isnan(out[3])


def test_garch_log_density_matches_closed_form_gaussian() -> None:
    returns = _returns()
    model = GARCHVol(dist="normal", mean="Zero").fit_returns(returns)
    y = np.array([returns[-1]])
    sigma = np.array([float(model.forecast(horizon=1)["sigma"][0])])
    mu = np.array([float(model.forecast(horizon=1)["mean"])])
    assert model.log_density(y, sigma) == pytest.approx(log_score_gaussian(y, mu, sigma))
    assert float(model.pit(y, sigma)[0]) == pytest.approx(
        float(norm.cdf((y[0] - mu[0]) / sigma[0]))
    )


def test_garch_t_log_density_is_standardized_not_textbook_t() -> None:
    returns = _returns()
    model = GARCHVol(dist="t", mean="Zero").fit_returns(returns)
    if model.fit_status != "fitted":
        pytest.skip("Student-t GARCH did not converge on the fixture")
    y = np.array([0.0])
    sigma = np.array([float(model.forecast(horizon=1)["sigma"][0])])
    nu = float(model.result.params["nu"])
    scored = float(model.log_density(y, sigma)[0])
    textbook = float(student_t.logpdf(0.0, nu) - math.log(sigma[0]))
    assert np.isfinite(scored)
    assert not math.isclose(scored, textbook, rel_tol=1e-8, abs_tol=1e-8)


def test_garch_fallback_density_is_explicitly_gaussian() -> None:
    model = GARCHVol(dist="t", min_obs=100).fit_returns(_returns(8))
    assert model.fit_status == "fallback"
    forecast = model.forecast(horizon=1)
    assert forecast["distribution"] == "normal"
    assert forecast["requested_distribution"] == "t"
    y = np.array([0.01])
    sigma = np.array([float(forecast["sigma"][0])])
    mu = np.array([float(forecast["mean"])])
    assert model.log_density(y, sigma) == pytest.approx(log_score_gaussian(y, mu, sigma))
    assert float(model.pit(y, sigma)[0]) == pytest.approx(0.5, abs=0.5)


def test_one_step_density_summary_uses_all_origins_and_qlike_stride() -> None:
    dates = np.array(["d0", "d1", "d2", "d3"], dtype=object)
    log_scores = np.array([-0.5, -1.5, -0.5, -1.5])
    crps = np.array([0.1, 0.3, 0.1, 0.3])
    pits = np.linspace(0.1, 0.9, 4)
    scored = one_step_density_summary(
        dates,
        log_scores,
        crps,
        pits,
        horizon_bars=2,
        session_index={"d0": 0, "d1": 1, "d2": 2, "d3": 3},
    )
    assert scored["density_horizon"] == 1
    assert scored["density_target"] == "date_level_ret_1"
    assert scored["n_density_origins"] == 4
    assert scored["n_density_origins_qlike_stride"] == 2
    assert scored["log_score_one_step"] == pytest.approx(-1.0)
    assert scored["ignorance_one_step"] == pytest.approx(1.0)
    assert scored["crps_one_step"] == pytest.approx(0.2)
    assert scored["log_score_one_step_qlike_origins"] == pytest.approx(-0.5)
    assert scored["crps_one_step_qlike_origins"] == pytest.approx(0.1)
    assert family_blob_forbidden_metrics_absent(scored) is True
    assert "live_pnl_claim" not in scored


def test_one_step_density_summary_fail_closed_on_empty_or_duplicate_dates() -> None:
    with pytest.raises(ValueError, match="at least one origin"):
        one_step_density_summary([], np.array([]), np.array([]), np.array([]), horizon_bars=1)
    with pytest.raises(ValueError, match="unique dates"):
        one_step_density_summary(
            np.array(["d0", "d0"], dtype=object),
            np.array([-1.0, -1.0]),
            np.array([0.1, 0.1]),
            np.array([0.4, 0.6]),
            horizon_bars=1,
        )


class _DensityGarch:
    fit_status = "fitted"
    last_fit_returns: np.ndarray | None = None

    def fit(self, _x, _y, *, returns):  # noqa: ANN001
        self.last_fit_returns = np.asarray(returns, dtype=float).copy()
        self.last_sigma = 0.02
        return self

    def forecast(self, *, horizon, quantiles=None):  # noqa: ANN001
        sigma = np.full(horizon, self.last_sigma)
        out = {
            "cumulative_variance": np.full(horizon, 0.04 + horizon),
            "variance": np.square(sigma),
            "sigma": sigma,
            "mean": 0.0,
            "distribution": "normal",
            "requested_distribution": "normal",
            "fit_status": self.fit_status,
        }
        if quantiles is not None:
            levels = np.asarray(quantiles, dtype=float)
            out["quantiles"] = self.last_sigma * norm.ppf(levels)[None, :]
        return out

    def log_density(self, returns, sigma=None):  # noqa: ANN001
        values = np.asarray(returns, dtype=float)
        mu = np.zeros(values.size)
        scales = (
            np.full(values.size, self.last_sigma)
            if sigma is None
            else np.asarray(sigma, dtype=float)
        )
        return log_score_gaussian(values, mu, scales)

    def pit(self, returns, sigma=None):  # noqa: ANN001
        values = np.asarray(returns, dtype=float).reshape(-1)
        scales = (
            np.full(values.size, self.last_sigma)
            if sigma is None
            else np.asarray(sigma, dtype=float)
        )
        return norm.cdf(values / scales)


def test_garch_oos_density_scores_origin_ret_1_not_in_fit() -> None:
    dates = np.array([0, 1, 2, 3])
    x = np.zeros((4, 1))
    y = np.array([9.0, 9.0, 9.0, 9.0])
    test_mask = dates >= 3
    return_dates = np.arange(4)
    return_values = np.array([0.01, 0.02, 0.03, 0.04])
    models: list[_DensityGarch] = []

    def make_model() -> _DensityGarch:
        model = _DensityGarch()
        models.append(model)
        return model

    predictions, statuses, density = _garch_oos_predictions(
        make_model,
        x,
        y,
        dates,
        test_mask,
        label_horizon=5,
        return_dates=return_dates,
        return_values=return_values,
    )

    assert predictions.tolist() == [5.04]
    assert statuses == ["fitted"]
    assert 3 in density
    record = density[3]
    assert record["y"] == pytest.approx(0.04)
    assert record["y"] != pytest.approx(9.0)
    assert record["density_horizon"] == 1
    assert record["sigma"] == pytest.approx(0.02)
    assert record["sigma"] != pytest.approx(math.sqrt(5.04))
    assert record["crps_method"] == "gaussian_closed"
    assert record["crps"] == pytest.approx(
        float(crps_gaussian(np.array([0.04]), np.array([0.0]), np.array([0.02]))[0])
    )
    fitted = models[0].last_fit_returns
    assert fitted is not None
    assert fitted.tolist() == [0.01, 0.02, 0.03]
    assert 0.04 not in fitted.tolist()


def test_garch_oos_density_fail_closed_when_origin_ret_1_missing() -> None:
    dates = np.array([5])
    x = np.zeros((1, 1))
    y = np.array([0.1])
    with pytest.raises(ValueError, match="missing origin ret_1"):
        _garch_oos_predictions(
            _DensityGarch,
            x,
            y,
            dates,
            np.array([True]),
            label_horizon=1,
            return_dates=np.arange(3),
            return_values=np.array([0.01, 0.02, 0.03]),
        )


def test_garch_log_density_rejects_length_mismatch() -> None:
    model = GARCHVol().fit_returns(_returns())
    with pytest.raises(ValueError, match="same length"):
        model.log_density(np.array([0.01, 0.02]), np.array([0.01]))

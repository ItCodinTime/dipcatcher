"""covariance.py edges: validators, DCC/AGDCC internals, estimator runs."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quant_fund.models import covariance as cov


def _returns(t: int = 90, n: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.01, t)
    out = np.empty((t, n))
    for j in range(n):
        out[:, j] = 0.6 * common + rng.normal(0, 0.008, t) + 0.02 * np.sin(np.arange(t) / (7 + j))
    return out


def _is_psd(m: np.ndarray) -> bool:
    return np.linalg.eigvalsh(m).min() > -1e-8


class TestPsdHelpers:
    def test_repair_indefinite(self) -> None:
        bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalue -1
        fixed, info = cov.repair_psd(bad)
        assert _is_psd(fixed)
        assert info["repaired"] == 1.0
        assert info["eig_min_before"] < 0.0
        assert info["eig_min_after"] > 0.0

    def test_repair_psd_passthrough(self) -> None:
        good = np.eye(3)
        fixed, _ = cov.repair_psd(good)
        assert_allclose(fixed, good)

    def test_symmetry_and_eig(self) -> None:
        assert cov.is_symmetric(np.eye(2)) is True
        assert cov.is_symmetric(np.array([[1.0, 0.5], [0.0, 1.0]])) is False
        assert cov.min_eigenvalue(np.diag([1.0, 2.0])) == pytest.approx(1.0)


class TestCleanAndValidate:
    def test_clean_rejects_1d_and_empty(self) -> None:
        with pytest.raises(ValueError, match="2D array"):
            cov._clean_returns(np.zeros(5))
        with pytest.raises(ValueError, match="2D array"):
            cov._clean_returns(np.zeros((3, 0)))
        with pytest.raises(ValueError, match="finite return rows"):
            cov._clean_returns(np.array([[np.nan], [0.1]]), min_rows=2)

    def test_clean_drops_nan_rows(self) -> None:
        x = np.array([[0.1], [np.nan], [0.2], [0.3]])
        out = cov._clean_returns(x)
        assert out.shape == (3, 1)

    def test_validate_lambda_bounds(self) -> None:
        cov._validate_lambda(0.5)
        for bad in (-0.1, 1.1, np.nan, np.inf):
            with pytest.raises(ValueError, match="lam"):
                cov._validate_lambda(bad)


class TestNamedEstimators:
    def test_sample(self) -> None:
        m, params = cov.sample(_returns())
        assert params["family"] == "sample"
        assert params["ddof"] == 1
        assert _is_psd(m)

    def test_sample_cov_matches_numpy(self) -> None:
        x = _returns()
        assert_allclose(cov.sample_cov(x), np.cov(x, rowvar=False, ddof=1))

    def test_ledoit_wolf(self) -> None:
        m, params = cov.ledoit_wolf(_returns())
        assert params["family"] == "ledoit_wolf"
        assert _is_psd(m)

    def test_oas(self) -> None:
        m, params = cov.oas(_returns())
        assert params["family"] == "oas"
        assert 0.0 <= params["shrinkage"] <= 1.0
        assert _is_psd(m)

    def test_ledoit_wolf_nonlinear(self) -> None:
        m, params = cov.ledoit_wolf_nonlinear(_returns(t=40, n=5))
        assert params["family"] == "ledoit_wolf_nonlinear"
        assert params["concentration"] == pytest.approx(5 / 39)
        assert _is_psd(m)

    def test_nonlinear_shrinkage_guards(self) -> None:
        with pytest.raises(ValueError, match="effective sample size"):
            cov._analytical_nonlinear_shrinkage(np.zeros((4, 2)), 2)
        with pytest.raises(ValueError, match="effective sample size"):
            cov._analytical_nonlinear_shrinkage(np.zeros(4), 10)
        with pytest.raises(ValueError, match="non-empty 2D array"):
            cov._analytical_nonlinear_shrinkage(np.zeros(4), 20)
        # Singular sample (identical columns) fails closed.
        dup = np.tile(np.linspace(-0.1, 0.1, 30), (3, 1)).T
        with pytest.raises(ValueError, match="singular"):
            cov._analytical_nonlinear_shrinkage(dup - dup.mean(axis=0), 29)

    def test_factor_cov(self) -> None:
        betas = np.array([[0.8, 0.1], [1.2, -0.3], [0.5, 0.2]])
        fcov = np.array([[0.04, 0.002], [0.002, 0.01]])
        idio = np.array([0.01, 0.02, 0.015])
        out = cov.factor_cov(betas, fcov, idio)
        assert out.shape == (3, 3)
        assert _is_psd(out)
        assert_allclose(np.diag(out), (betas @ fcov @ betas.T).diagonal() + idio)

    def test_ewma_cov_matches_recursion(self) -> None:
        x = _returns(t=60, n=2)
        out = cov.ewma_cov(x, 0.94)
        ref = np.outer(x[0], x[0])
        for t in range(1, x.shape[0]):
            ref = 0.94 * ref + 0.06 * np.outer(x[t], x[t])
        ref = 0.5 * (ref + ref.T)
        assert_allclose(out, ref, atol=1e-12)
        with pytest.raises(ValueError, match="lam"):
            cov.ewma_cov(x, 1.5)

    def test_ewma_estimator_and_1d(self) -> None:
        m, params = cov.ewma(_returns(t=60))
        assert params["family"] == "ewma"
        assert _is_psd(m)
        v = cov.ewma_variance_1d(np.array([0.01, -0.02, 0.03]), 0.5)
        assert v.shape == (3,)
        # v[t] = lam*v[t-1] + (1-lam)*r[t-1]^2
        assert v[0] == pytest.approx(0.01**2)
        assert v[1] == pytest.approx(0.5 * 0.01**2 + 0.5 * 0.01**2)
        with pytest.raises(ValueError, match="1D"):
            cov.ewma_variance_1d(np.ones((3, 2)))

    def test_condition_number(self) -> None:
        assert cov.condition_number(np.diag([1.0, 4.0])) == pytest.approx(4.0)


class TestTrailingWindow:
    def test_terminal_nan_fails_closed(self) -> None:
        x = _returns()
        x[-1, 0] = np.nan
        with pytest.raises(ValueError, match="incomplete_terminal_row"):
            cov.dcc_trailing_complete_window(x)

    def test_interior_hole_uses_trailing_block(self) -> None:
        x = _returns(t=80)
        x[20, :] = np.nan
        out = cov.dcc_trailing_complete_window(x)
        assert out.shape[0] == 59  # rows 21..79

    def test_too_short_window(self) -> None:
        x = _returns(t=60)
        x[40, :] = np.nan
        with pytest.raises(ValueError, match="insufficient_contiguous_rows"):
            cov.dcc_trailing_complete_window(x, min_rows=50)

    def test_prepare_window_reports_dropped(self) -> None:
        x = _returns(t=80)
        x[10, :] = np.nan
        win, dropped = cov._dcc_prepare_window(x)
        assert dropped == 11.0


class TestSpecValidators:
    @pytest.mark.parametrize(
        "name,family",
        [
            ("dcc_gaussian", "dcc_gaussian"),
            ("gaussian", "dcc_gaussian"),
            ("engle_2002", "dcc_gaussian"),
            ("dcc_student_t", "dcc_student_t"),
            ("adcc", "adcc"),
            ("ces_2006", "adcc"),
            ("ccc", "ccc"),
            ("agdcc", "agdcc"),
            ("agdcc_full", "agdcc_full"),
        ],
    )
    def test_dcc_spec_aliases(self, name: str, family: str) -> None:
        assert cov.require_implemented_dcc_spec(name) == family

    def test_dcc_spec_errors(self) -> None:
        for bad in ("", "  ", "dcc", "unknown_thing", 5, True):
            with pytest.raises(ValueError):
                cov.require_implemented_dcc_spec(bad)  # type: ignore[arg-type]

    def test_optimizer_named_paths(self) -> None:
        assert cov.require_implemented_optimizer_covariance("ledoit_wolf") == "ledoit_wolf"
        assert cov.require_implemented_optimizer_covariance("EWMA") == "ewma"
        assert cov.require_implemented_optimizer_covariance("oas") == "oas"
        assert cov.require_implemented_optimizer_covariance("sample") == "sample"
        assert cov.require_implemented_optimizer_covariance("ccc") == "ccc"
        assert cov.require_implemented_optimizer_covariance("agdcc_full") == "agdcc_full"
        assert (
            cov.require_implemented_optimizer_covariance("ledoit_wolf_nonlinear")
            == "ledoit_wolf_nonlinear"
        )

    @pytest.mark.parametrize(
        "bad",
        ["", "dcc", "gaussian", "t", "student_t", "shrinkage", "factor", "quest", 3, False],
    )
    def test_optimizer_rejects_ambiguous(self, bad) -> None:
        with pytest.raises(ValueError):
            cov.require_implemented_optimizer_covariance(bad)


class TestDccInternals:
    def test_qbar_and_r_roundtrip(self) -> None:
        z = _returns(t=80, n=3)
        qbar = cov._dcc_qbar(z)
        assert_allclose(np.diag(qbar), np.ones(3))
        assert _is_psd(qbar)
        r = cov._dcc_r_from_q(qbar)
        assert_allclose(np.diag(r), np.ones(3))
        assert _is_psd(r)

    def test_qbar_constant_column_fails(self) -> None:
        z = np.column_stack([np.zeros(60), np.ones(60) * 0.01])
        with pytest.raises(ValueError, match="non-finite"):
            cov._dcc_qbar(z)

    def test_step_and_one_step(self) -> None:
        z = _returns(t=60, n=2)
        qbar = cov._dcc_qbar(z)
        q1 = cov._dcc_step_q(qbar, qbar, z[-1], 0.05, 0.9)
        assert q1.shape == (2, 2)
        h = cov._dcc_one_step_h(z, qbar, 0.05, 0.9, np.array([0.01, 0.012]))
        assert h.shape == (2, 2)
        assert _is_psd(h)
        assert np.diag(h)[0] != np.diag(h)[1]

    def test_adcc_helpers(self) -> None:
        z = _returns(t=60, n=2)
        n_shock = cov._adcc_negative_shocks(z)
        assert np.all(n_shock <= 0.0)
        nbar = cov._adcc_nbar(n_shock)
        assert _is_psd(nbar)
        qbar = cov._dcc_qbar(z)
        kappa = cov._adcc_kappa(qbar, nbar)
        assert kappa >= 0.0
        q_next = cov._adcc_step_q(qbar, qbar, nbar, z[-1], n_shock[-1], 0.05, 0.85, 0.03)
        assert q_next.shape == (2, 2)
        h = cov._adcc_one_step_h(z, qbar, nbar, 0.05, 0.85, 0.03, np.array([0.01, 0.01]))
        assert _is_psd(h)

    def test_agdcc_diagonal_helpers(self) -> None:
        z = _returns(t=60, n=2)
        qbar = cov._dcc_qbar(z)
        n_shock = cov._adcc_negative_shocks(z)
        nbar = cov._adcc_nbar(n_shock)
        a = np.array([0.05, 0.04])
        b = np.array([0.85, 0.8])
        g = np.array([0.03, 0.02])
        diag = cov._agdcc_congruence_diag(np.sqrt(a), qbar)
        assert diag.shape == (2, 2)
        news = cov._agdcc_news(g, n_shock[-1])
        assert news.shape == (2, 2)
        intercept = cov._agdcc_intercept(qbar, nbar, a, b, g)
        assert _is_psd(intercept)
        q_next = cov._agdcc_step_q(qbar, intercept, z[-1], n_shock[-1], a, b, g)
        assert q_next.shape == (2, 2)
        h = cov._agdcc_one_step_h(z, qbar, intercept, a, b, g, np.array([0.01, 0.01]))
        assert _is_psd(h)

    def test_agdcc_broadcast_and_start(self) -> None:
        v = cov._agdcc_broadcast_diag(None, 3, 0.1, "a")
        assert_allclose(v, np.full(3, 0.1))
        v2 = cov._agdcc_broadcast_diag(0.2, 2, 0.1, "a")
        assert_allclose(v2, [0.2, 0.2])
        with pytest.raises(ValueError, match="non-negative"):
            cov._agdcc_broadcast_diag(-0.1, 3, 0.1, "a")
        with pytest.raises(ValueError, match="finite"):
            cov._agdcc_broadcast_diag(np.nan, 3, 0.1, "a")
        sa, sb, sg = cov._agdcc_scale_start(
            np.array([0.5, 0.5]),
            np.array([0.9, 0.9]),
            np.array([0.1, 0.1]),
            np.eye(2),
            np.eye(2) * 0.2,
        )
        assert sa.shape == (2,)
        assert np.all(sb <= 0.995)

    def test_agdcc_full_helpers(self) -> None:
        n = 2
        qbar = np.eye(n) * 1.0 + 0.1 * (1 - np.eye(n))
        nbar = np.eye(n) * 0.2
        a = np.eye(n) * 0.1
        b = np.eye(n) * 0.8
        g = np.eye(n) * 0.05
        cong = cov._agdcc_full_congruence(qbar, a)
        assert cong.shape == (n, n)
        news = cov._agdcc_full_news(g, np.array([-0.5, 0.1]))
        assert news.shape == (n, n)
        intercept = cov._agdcc_full_intercept(qbar, nbar, a, b, g)
        assert intercept.shape == (n, n)
        q_next = cov._agdcc_full_step_q(
            qbar, intercept, np.array([0.1, -0.2]), np.array([-0.1, 0.0]), a, b, g
        )
        assert q_next.shape == (n, n)
        radius = cov._agdcc_full_kronecker_radius(a, b, g)
        assert radius < 1.0
        offdiag = cov._agdcc_full_offdiag_maxabs(a)
        assert offdiag == 0.0
        params = np.concatenate([a.ravel(), b.ravel(), g.ravel()])
        ua, ub, ug = cov._agdcc_full_unpack(params, n)
        assert_allclose(ua, a)
        bounds = cov._agdcc_full_bounds(n)
        assert len(bounds) == 3 * n * n
        h = cov._agdcc_full_one_step_h(
            np.array([[0.1, -0.2], [0.05, 0.1]]),
            qbar,
            intercept,
            a,
            b,
            g,
            np.array([0.01, 0.01]),
        )
        assert _is_psd(h)

    def test_corr_nlls(self) -> None:
        z = np.array([0.5, -0.3])
        corr = np.array([[1.0, 0.3], [0.3, 1.0]])
        g_nll = cov._gaussian_corr_nll(z, corr)
        t_nll = cov.student_t_corr_nll(z, corr, 8.0)
        assert np.isfinite(g_nll)
        assert np.isfinite(t_nll)
        # Degenerate correlation -> fail-closed sentinel, never an exception.
        assert cov._gaussian_corr_nll(z, np.ones((2, 2))) == 1e12
        assert cov.student_t_corr_nll(z, np.ones((2, 2)), 8.0) == 1e12
        with pytest.raises(ValueError, match="greater than 2"):
            cov.student_t_corr_nll(z, corr, 2.0)
        with pytest.raises(ValueError, match="shapes disagree"):
            cov.student_t_corr_nll(np.ones(3), corr, 8.0)


class TestEstimatorRuns:
    """End-to-end QML runs cover the big optimizer bodies (1300-1730)."""

    def test_dcc_gaussian(self) -> None:
        h, params = cov.dcc_gaussian(_returns(t=80, n=2))
        assert _is_psd(h)
        assert params["family"] == "dcc_gaussian"
        assert params["asymmetric"] == "false"

    def test_dcc_gaussian_param_guards(self) -> None:
        x = _returns(t=80, n=2)
        with pytest.raises(ValueError, match="a0"):
            cov.dcc_gaussian(x, a0=-0.1)
        with pytest.raises(ValueError, match="b0"):
            cov.dcc_gaussian(x, b0=np.nan)

    def test_dcc_student_t(self) -> None:
        h, params = cov.dcc_student_t(_returns(t=80, n=2))
        assert _is_psd(h)
        assert params["family"] == "dcc_student_t"

    def test_dcc_student_t_nu_guard(self) -> None:
        with pytest.raises(ValueError, match="nu"):
            cov.dcc_student_t(_returns(t=80, n=2), nu=1.5)

    def test_adcc(self) -> None:
        h, params = cov.adcc(_returns(t=80, n=2))
        assert _is_psd(h)
        assert params["family"] == "adcc"

    def test_agdcc(self) -> None:
        h, params = cov.agdcc(_returns(t=80, n=2))
        assert _is_psd(h)
        assert params["family"] == "agdcc"

    def test_agdcc_full(self) -> None:
        h, params = cov.agdcc_full(_returns(t=80, n=2))
        assert _is_psd(h)
        assert params["family"] == "agdcc_full"

    def test_ccc(self) -> None:
        h, params = cov.ccc(_returns(t=80, n=2))
        assert _is_psd(h)
        assert params["family"] == "ccc"
        assert params["dynamic_correlation"] == "false"

    def test_ccc_single_column_fails(self) -> None:
        with pytest.raises(ValueError, match="two return series"):
            cov.ccc(_returns(t=80, n=1))

    def test_estimators_deterministic(self) -> None:
        x = _returns(t=80, n=2, seed=42)
        h1, _ = cov.dcc_gaussian(x)
        h2, _ = cov.dcc_gaussian(x)
        assert_allclose(h1, h2)

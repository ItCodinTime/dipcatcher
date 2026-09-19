"""Wave 9: effective_bets / average_pairwise_corr closed-form + boundaries."""

import numpy as np
import pytest

from quant_fund.portfolio.concentration import average_pairwise_corr, effective_bets


def test_effective_bets_identity_corr_equals_n_equal_weights() -> None:
    """Under I_n, abs-normalized equal weights → N_eff = n."""
    for n in (1, 2, 5, 10):
        w = np.full(n, 1.0 / n)
        assert effective_bets(w, np.eye(n)) == pytest.approx(float(n))


def test_effective_bets_perfect_corr_collapses_to_one() -> None:
    """All-ones corr (valid unit diagonal) → q=1 → N_eff=1 for long-only."""
    n = 4
    corr = np.ones((n, n))
    w = np.full(n, 0.25)
    assert effective_bets(w, corr) == pytest.approx(1.0)


def test_effective_bets_closed_form_two_asset() -> None:
    """wn=[0.5,0.5], ρ → 1 / (0.5 + 0.5ρ) = 2/(1+ρ)."""
    for rho in (0.0, 0.2, 0.5, 0.9):
        corr = np.array([[1.0, rho], [rho, 1.0]])
        w = np.array([0.5, 0.5])
        expected = 2.0 / (1.0 + rho)
        assert effective_bets(w, corr) == pytest.approx(expected)


def test_effective_bets_abs_normalizes_signed_weights() -> None:
    """Signs ignored for concentration; |w| L1-normalized."""
    corr = np.eye(2)
    assert effective_bets(np.array([0.5, -0.5]), corr) == pytest.approx(2.0)
    assert effective_bets(np.array([1.0, -1.0]), corr) == pytest.approx(2.0)
    assert effective_bets(np.array([2.0, 0.0]), corr) == pytest.approx(1.0)


def test_effective_bets_zero_weights() -> None:
    assert effective_bets(np.zeros(3), np.eye(3)) == 0.0


def test_effective_bets_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="aligned"):
        effective_bets(np.ones(2), np.eye(3))
    with pytest.raises(ValueError, match="finite"):
        effective_bets(np.array([1.0, np.nan]), np.eye(2))
    with pytest.raises(ValueError, match="symmetric"):
        effective_bets(np.ones(2), np.array([[1.0, 0.2], [0.1, 1.0]]))
    with pytest.raises(ValueError, match="unit diagonal"):
        effective_bets(np.ones(2), np.array([[2.0, 0.0], [0.0, 1.0]]))


def test_average_pairwise_corr_closed_form() -> None:
    corr = np.array(
        [
            [1.0, 0.2, -0.4],
            [0.2, 1.0, 0.6],
            [-0.4, 0.6, 1.0],
        ]
    )
    # Off-diagonal mean: (0.2 + -0.4 + 0.6) * 2 / 6 = 0.4 / 3
    assert average_pairwise_corr(corr) == pytest.approx(0.4 / 3.0)


def test_average_pairwise_corr_identity_is_zero() -> None:
    assert average_pairwise_corr(np.eye(5)) == pytest.approx(0.0)


def test_average_pairwise_corr_n_lt_2_nan() -> None:
    assert np.isnan(average_pairwise_corr(np.array([[1.0]])))


def test_average_pairwise_corr_rejects_bad() -> None:
    with pytest.raises(ValueError, match="finite"):
        average_pairwise_corr(np.array([[1.0, np.nan], [np.nan, 1.0]]))
    with pytest.raises(ValueError, match="square"):
        average_pairwise_corr(np.ones((2, 3)))

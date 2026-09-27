"""Jurafsky & Martin SLP3 Appendix A — Eisner ice-cream HMM."""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.hmm.discrete import (
    backward,
    baum_welch,
    forward,
    likelihood,
    mle_supervised,
    viterbi,
)
from quant_fund.hmm.eisner import COLD, EISNER, HOT, ice_cream


def test_eisner_forward_matches_figure_a5() -> None:
    o = ice_cream([3, 1, 3])
    alpha, p = forward(EISNER, o)
    assert float(alpha[0, HOT]) == pytest.approx(0.32)
    assert float(alpha[0, COLD]) == pytest.approx(0.02)
    assert float(alpha[1, HOT]) == pytest.approx(0.0404)
    assert float(alpha[1, COLD]) == pytest.approx(0.069)
    assert p == pytest.approx(float(alpha[2].sum()))
    assert p == pytest.approx(0.028562, rel=1e-6)
    beta = backward(EISNER, o)
    assert np.allclose(beta[-1], 1.0)
    p_back = float((EISNER.pi * EISNER.B[:, o[0]] * beta[0]).sum())
    assert p_back == pytest.approx(p, rel=1e-12)


def test_eisner_viterbi_matches_figure_a8() -> None:
    o = ice_cream([3, 1, 3])
    path, path_p = viterbi(EISNER, o)
    assert path.tolist() == [HOT, COLD, HOT]
    assert path_p == pytest.approx(0.0128, rel=1e-6)


def test_baum_welch_likelihood_is_non_decreasing() -> None:
    o = ice_cream([3, 1, 3, 2, 1, 2, 3, 1, 3])
    _, hist = baum_welch(o, n_states=2, n_obs=3, n_iter=12, seed=3)
    for a, b in zip(hist[:-1], hist[1:], strict=True):
        assert b + 1e-12 >= a
    assert hist[-1] > hist[0]


def test_supervised_mle_recovers_counts() -> None:
    # One visible sequence: H H C  with emissions 3, 3, 1
    states = [HOT, HOT, COLD]
    obs = ice_cream([3, 3, 1])
    model = mle_supervised(states, obs, n_states=2, n_obs=3)
    assert float(model.pi[HOT]) == pytest.approx(1.0)
    assert float(model.A[HOT, HOT]) == pytest.approx(0.5)
    assert float(model.A[HOT, COLD]) == pytest.approx(0.5)
    assert float(model.B[HOT, 2]) == pytest.approx(1.0)  # both H days ate 3
    assert float(model.B[COLD, 0]) == pytest.approx(1.0)


def test_likelihood_rejects_bad_obs() -> None:
    with pytest.raises(ValueError, match="out of vocabulary"):
        likelihood(EISNER, [9])


def test_hmm_cli_eisner() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["hmm", "eisner", "--obs", "3,1,3"])
    assert result.exit_code == 0, result.output
    blob = json.loads(result.output)
    assert blob["viterbi_path"] == ["HOT", "COLD", "HOT"]
    assert blob["likelihood"] == pytest.approx(0.028562, rel=1e-6)

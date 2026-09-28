"""``hmm train`` CLI: Baum–Welch run on the Eisner ice-cream toy data."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from quant_fund.hmm.cli import hmm_app

pytestmark = pytest.mark.synthetic

runner = CliRunner()


def test_train_prints_nondecreasing_likelihood_history() -> None:
    result = runner.invoke(hmm_app, ["train", "--obs", "3,1,3,2,1,2,3", "--n-iter", "8"])
    assert result.exit_code == 0, result.output
    blob = json.loads(result.output)
    hist = blob["likelihood_history"]
    assert len(hist) == 9  # initial likelihood plus one per EM iteration
    assert blob["non_decreasing"] is True
    assert all(hist[i] <= hist[i + 1] + 1e-12 for i in range(len(hist) - 1))
    assert blob["research_only"] is True
    for matrix in ("A", "B"):
        for row in blob[matrix]:
            assert abs(sum(row) - 1.0) < 1e-9
    assert abs(sum(blob["pi"]) - 1.0) < 1e-9


def test_train_with_three_states() -> None:
    result = runner.invoke(hmm_app, ["train", "--n-states", "3", "--n-iter", "4"])
    assert result.exit_code == 0, result.output
    blob = json.loads(result.output)
    assert len(blob["A"]) == 3


def test_eisner_command_reports_likelihood_and_path() -> None:
    result = runner.invoke(hmm_app, ["eisner", "--obs", "3,1,3"])
    assert result.exit_code == 0, result.output
    blob = json.loads(result.output)
    assert blob["likelihood"] > 0.0
    assert len(blob["viterbi_path"]) == 3
    assert blob["research_only"] is True

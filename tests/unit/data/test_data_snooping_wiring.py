"""Data-snooping wiring: agent H51 + catalog honesty + verify hooks.

Research-diagnostic only — never a live Sharpe / P&L claim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quant_fund.research.agent import _build_hypotheses, _ranker_data_snooping
from quant_fund.research.catalog import (
    H99_HYPOTHESIS_ID,
    data_snooping_has_finite_spa_p,
    h99_hypothesis_consistency_errors,
    hypotheses_include_h99,
    ranking_data_snooping_blob,
    ranking_data_snooping_honesty_errors,
)

DATES = [f"2024-{m:02d}-{d:02d}" for m in range(1, 7) for d in range(1, 16)]


def _ranker(name: str, bump: float, seed: int) -> dict[str, object]:
    series = (np.random.default_rng(seed).normal(0.0, 0.05, size=len(DATES)) + bump).tolist()
    return {"name": name, "ic_series": series, "ic_dates": list(DATES)}


def _clean_blob() -> dict[str, object]:
    rankers = [_ranker("ridge", 0.001, 1), _ranker("lgbm", 0.0005, 2), _ranker("oracle", 0.008, 3)]
    blob = _ranker_data_snooping(rankers)
    assert blob is not None
    return blob


def test_agent_ranker_data_snooping_shape() -> None:
    blob = _clean_blob()
    assert blob["research_only"] is True
    assert blob["claim"] == "research_diagnostic_only"
    assert blob["n_trials"] == 3 and blob["n_obs"] == len(DATES)
    assert blob["best_trial"] == "oracle"
    assert blob["spa_p_lower"] <= blob["spa_p_upper"]
    assert blob["stepm_n_rejected"] == len(blob["stepm_rejected"])
    assert blob["mcs_n_included"] == len(blob["mcs_included"])
    assert ranking_data_snooping_honesty_errors({"data_snooping": blob}) == []


def test_agent_snooping_fail_closed_without_universe() -> None:
    assert _ranker_data_snooping([]) is None
    assert _ranker_data_snooping([_ranker("ridge", 0.001, 1)]) is None
    # Fewer than 10 aligned dates → no block.
    short = _ranker("short", 0.001, 1)
    short["ic_dates"] = list(DATES[:6])
    short["ic_series"] = list(short["ic_series"][:6])  # type: ignore[index]
    assert _ranker_data_snooping([short, _ranker("ridge", 0.001, 2)]) is None


def test_agent_mints_h99_from_snooping_blob() -> None:
    blob = _clean_blob()
    families = {"ranking": {"data_snooping": blob}}
    hyps = _build_hypotheses(families, [])
    h46 = [h for h in hyps if h.id == H99_HYPOTHESIS_ID]
    assert len(h46) == 1
    assert h46[0].family == "discovery"
    assert h46[0].p_value == blob["spa_p_consistent"]
    # No block → no H51 row.
    assert [h for h in _build_hypotheses({"ranking": {}}, []) if h.id == H99_HYPOTHESIS_ID] == []


def test_catalog_helpers_skip_and_require_h99() -> None:
    blob = _clean_blob()
    assert ranking_data_snooping_blob({"data_snooping": blob}) is not None
    assert ranking_data_snooping_blob({}) is None
    assert ranking_data_snooping_blob(None) is None
    assert data_snooping_has_finite_spa_p(blob) is True
    assert data_snooping_has_finite_spa_p({}) is False
    assert hypotheses_include_h99([{"id": H99_HYPOTHESIS_ID, "family": "discovery"}]) is True
    assert hypotheses_include_h99([{"id": H99_HYPOTHESIS_ID, "family": "bound"}]) is False
    # Finite consistent p without H51 → soft fail-closed.
    assert h99_hypothesis_consistency_errors({"data_snooping": blob}, []) == [
        "hypothesis_h99_missing_despite_finite_spa_p"
    ]
    assert (
        h99_hypothesis_consistency_errors(
            {"data_snooping": blob}, [{"id": H99_HYPOTHESIS_ID, "family": "discovery"}]
        )
        == []
    )
    # Non-finite p / no block → skip.
    nan_blob = dict(blob, spa_p_consistent=float("nan"))
    assert h99_hypothesis_consistency_errors({"data_snooping": nan_blob}, []) == []
    assert h99_hypothesis_consistency_errors({}, []) == []


def test_catalog_honesty_flags_malformed_blocks() -> None:
    blob = _clean_blob()
    bad = dict(blob, reality_check_p=1.5)
    assert "data_snooping_reality_check_p_out_of_unit_interval" in (
        ranking_data_snooping_honesty_errors({"data_snooping": bad})
    )
    non_numeric = dict(blob, spa_p_consistent="0.1")
    assert "data_snooping_spa_p_consistent_non_numeric" in (
        ranking_data_snooping_honesty_errors({"data_snooping": non_numeric})
    )
    mismatch = dict(blob, stepm_n_rejected=int(blob["stepm_n_rejected"]) + 1)
    assert "data_snooping_stepm_n_rejected_mismatch" in (
        ranking_data_snooping_honesty_errors({"data_snooping": mismatch})
    )
    no_claim = dict(blob, research_only=False, claim="live")
    errs = ranking_data_snooping_honesty_errors({"data_snooping": no_claim})
    assert "data_snooping_research_only_missing_or_false" in errs
    assert "data_snooping_claim_not_research_diagnostic_only" in errs
    # Missing block / non-dict → skip.
    assert ranking_data_snooping_honesty_errors({}) == []
    assert ranking_data_snooping_honesty_errors({"data_snooping": "nope"}) == []


def test_verify_wires_data_snooping() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "ranking_data_snooping_honesty_errors" in src
    assert "h99_hypothesis_consistency_errors" in src
    assert 'families.get("ranking")' in src

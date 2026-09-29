"""Evaluation-loop and receipt tests (SYNTHETIC fixtures only).

Realized-vol checks use constructed blocks where the forward returns are
constant, so the RMS realized vol is known exactly — no P&L or headline
metrics anywhere.
"""

from __future__ import annotations

import json
import math
import typing
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from quant_fund.research.allocation import (
    ALLOCATION_RECEIPT_SCHEMA,
    AllocationConstraints,
    build_receipt,
    compare_engines,
    fit_weights,
    run_walk_forward,
    verify_allocation_receipt,
    write_receipt,
)

Array = NDArray[np.float64]


def _returns(seed: int = 1, t: int = 160, n: int = 3) -> Array:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    cov = a @ a.T * 1e-4 + np.diag(rng.uniform(1e-4, 4e-4, n))
    return np.asarray(rng.multivariate_normal(np.zeros(n), cov, t), dtype=float)


# --- walk-forward structure ------------------------------------------------


def test_walk_forward_decision_grid_and_shapes() -> None:
    returns = _returns(t=160, n=3)
    ev = run_walk_forward(returns, "inverse_volatility", window=30, step=10)
    assert ev.decision_index.tolist() == list(range(30, 160, 10))
    assert ev.weights.shape == (13, 3)
    assert ev.predicted_vol.shape == ev.realized_vol.shape == (13,)
    assert ev.constraint_violations == 0
    assert np.all(np.isfinite(ev.predicted_vol))
    assert np.all(np.isfinite(ev.realized_vol))


def test_initial_turnover_is_gross_of_first_book() -> None:
    returns = _returns()
    ev = run_walk_forward(returns, "inverse_volatility", window=30, step=10)
    assert np.isclose(ev.turnover[0], float(np.abs(ev.weights[0]).sum()))


def test_turnover_matches_manual_weight_diff() -> None:
    returns = _returns()
    ev = run_walk_forward(returns, "risk_parity", window=30, step=20)
    expected = float(np.abs(ev.weights[2] - ev.weights[1]).sum())
    assert np.isclose(ev.turnover[2], expected)


def test_weights_match_direct_fit_on_same_window() -> None:
    returns = _returns()
    window, step = 30, 25
    ev = run_walk_forward(returns, "risk_parity", window=window, step=step)
    t = int(ev.decision_index[1])
    direct = fit_weights("risk_parity", returns[t - window : t])
    assert np.allclose(ev.weights[1], direct, atol=0)


def test_realized_vol_known_answer_constant_forward_block() -> None:
    # Window rows vary (so covariance is nondegenerate); the forward block is
    # constant r0 -> RMS vol = |r0 * gross|. Long-only inv-vol gross = 1.
    window, n, horizon = 40, 2, 5
    rng = np.random.default_rng(4)
    cov = np.diag([4e-4, 1e-4])
    wrows = rng.multivariate_normal(np.zeros(n), cov, window)
    r0 = 0.02
    tail = np.tile(np.array([[r0, r0]]), (horizon, 1))
    returns = np.vstack([wrows, tail])
    ev = run_walk_forward(
        returns, "inverse_volatility", window=window, step=horizon, horizon=horizon
    )
    assert np.isclose(ev.realized_vol[0], r0, atol=1e-12)


def test_vol_target_ex_ante_hits_target() -> None:
    returns = _returns(t=140, n=4)
    # Non-binding constraints: the ex-ante target check requires the scale
    # factor to be unconstrained — window 3 needs leverage ~10x, which
    # per-asset max_weight=1.0 would clip on this fixture.
    cons = AllocationConstraints(leverage_cap=1e6, max_weight=1e6)
    ev = run_walk_forward(
        returns, "vol_target", window=40, step=20, target_vol=0.02, constraints=cons
    )
    assert np.allclose(ev.predicted_vol, 0.02, atol=1e-9)
    assert ev.metrics["target_vol_rmse"] >= 0.0


def test_erc_residual_zero_for_risk_parity() -> None:
    returns = _returns(t=140, n=4)
    ev = run_walk_forward(returns, "risk_parity", window=40, step=20)
    assert float(ev.erc_residual.max()) < 1e-6
    assert ev.metrics["erc_residual_max"] < 1e-6


def test_metrics_exclude_forbidden_headline_keys() -> None:
    returns = _returns()
    ev = run_walk_forward(returns, "inverse_volatility", window=30, step=10)
    banned = ("sharpe", "sortino", "calmar", "pnl", "nav")
    for key in ev.metrics:
        tokens = key.lower().split("_")
        assert not any(b in tokens for b in banned), key


def test_compare_engines_runs_all() -> None:
    returns = _returns(t=120, n=3)
    cons = AllocationConstraints(leverage_cap=2.0)
    out = compare_engines(
        returns,
        ("inverse_volatility", "risk_parity", "kelly", "vol_target"),
        window=30,
        step=15,
        constraints=cons,
        target_vol=0.02,
    )
    assert set(out) == {"inverse_volatility", "risk_parity", "kelly", "vol_target"}


def test_bad_inputs_rejected() -> None:
    with pytest.raises(ValueError, match="window"):
        run_walk_forward(_returns(t=60), "inverse_volatility", window=3)
    with pytest.raises(ValueError, match="shorter than the sample"):
        run_walk_forward(_returns(t=60), "inverse_volatility", window=60)
    bad = _returns(t=60)
    bad[10, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        run_walk_forward(bad, "inverse_volatility", window=30)


# --- receipts ---------------------------------------------------------------


def _eval() -> typing.Any:
    returns = _returns(t=140, n=3)
    return run_walk_forward(
        returns,
        "vol_target",
        window=40,
        step=20,
        target_vol=0.02,
        constraints=AllocationConstraints(leverage_cap=2.0),
    )


def test_receipt_build_verify_roundtrip(tmp_path: Path) -> None:
    ev = _eval()
    receipt = build_receipt(ev, synthetic=True, label="unit-test")
    assert receipt["schema"] == ALLOCATION_RECEIPT_SCHEMA
    assert receipt["claim"] == "research_only"
    assert receipt["synthetic"] is True
    assert "payload_sha256" in receipt
    path = write_receipt(receipt, tmp_path / "alloc.json")
    result = verify_allocation_receipt(path)
    assert result == {"valid": True, "errors": []}


def test_receipt_tamper_detected(tmp_path: Path) -> None:
    ev = _eval()
    receipt = build_receipt(ev, synthetic=True)
    path = write_receipt(receipt, tmp_path / "alloc.json")
    payload = json.loads(path.read_text())
    payload["metrics"]["vol_rmse"] = 12345.0
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    result = verify_allocation_receipt(path)
    assert result["valid"] is False
    assert "receipt_payload_hash_mismatch" in result["errors"]


def test_receipt_forbidden_key_fails_closed(tmp_path: Path) -> None:
    receipt = {
        "schema": ALLOCATION_RECEIPT_SCHEMA,
        "claim": "research_only",
        "synthetic": True,
        "metrics": {"sharpe": 1.0},
        "payload_sha256": "deadbeef",
    }
    with pytest.raises(ValueError, match="forbidden"):
        write_receipt(receipt, tmp_path / "alloc.json")


def test_verify_receipt_schema_and_synthetic_errors(tmp_path: Path) -> None:
    path = tmp_path / "alloc.json"
    path.write_text(json.dumps({"schema": "other", "payload_sha256": "x"}))
    result = verify_allocation_receipt(path)
    assert result["valid"] is False
    assert "receipt_schema_mismatch" in result["errors"]


def test_verify_missing_file() -> None:
    result = verify_allocation_receipt("/nonexistent/alloc.json")
    assert result["valid"] is False
    assert result["errors"]
    assert not math.isnan(0.0)  # nothing else to assert — missing is missing

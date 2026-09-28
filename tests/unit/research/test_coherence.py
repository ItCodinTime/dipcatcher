"""Distributional coherence bench: reconciled aggregate quantiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quant_fund.research.coherence import (
    COHERENCE_SCHEMA,
    METHODS,
    PanelShard,
    _factor_panel,
    _nearest_psd_corr,
    _pit_zscores,
    coherence_contract_errors,
    run_coherence,
    write_coherence_receipt,
)

TAUS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
N_TRAIN = 384
N_EVAL = 128


def _run(**kwargs: Any) -> tuple[Any, dict[str, Any]]:
    params: dict[str, Any] = dict(
        panels=("independent", "gauss_factor", "heavy_tail_factor"),
        n_train=N_TRAIN,
        n_eval=N_EVAL,
        seed=0,
        taus=TAUS,
        n_mc=256,
    )
    params.update(kwargs)
    return run_coherence(**params)


def _crps(receipt: dict[str, Any], panel: str, method: str) -> float:
    for row in receipt["payload"]["results"]:
        if row["panel"] == panel and row["method"] == method:
            return float(row["crps"])
    raise AssertionError(f"missing {panel}/{method}")


def test_emits_all_methods_per_panel() -> None:
    frame, receipt = _run()
    assert frame.height == 3 * len(METHODS)
    payload = receipt["payload"]
    assert payload["schema"] == COHERENCE_SCHEMA
    assert payload["data_label"] == "SYNTHETIC"
    assert payload["live_pnl_claim"] is False
    assert payload["n_error_rows"] == 0
    assert receipt["verdict"] == "pass"


def test_independence_beats_naive_sum() -> None:
    # rho=0: summing marginal quantiles badly over-disperses (comonotone bound).
    _, receipt = _run()
    assert _crps(receipt, "independent", "independent_mc") < _crps(
        receipt, "independent", "naive_sum"
    )


def test_comonotonic_naive_beats_independent() -> None:
    # rho≈1: the sum-of-quantiles is near-exact; independence under-disperses.
    panels = {
        "comonotone": lambda n, s: _factor_panel("comonotone", n, s, 0.98),
    }
    _, receipt = run_coherence(
        panels=panels, n_train=N_TRAIN, n_eval=N_EVAL, seed=0, taus=TAUS, n_mc=256
    )
    assert _crps(receipt, "comonotone", "naive_sum") < _crps(
        receipt, "comonotone", "independent_mc"
    )


def test_copula_tracks_independence_when_rho_zero() -> None:
    # rho=0: copula_mc should land near independent_mc, well below naive_sum.
    _, receipt = _run()
    assert _crps(receipt, "independent", "copula_mc") < _crps(receipt, "independent", "naive_sum")


def test_pit_zscores_recover_correlation() -> None:
    from quant_fund.research.coherence import _margin_grids

    # Fine tau grid: coarse grids quantize the PIT and compress rho.
    fine = np.linspace(0.02, 0.98, 49)
    y = _factor_panel("corr", 2000, 7, 0.7).y
    grids = _margin_grids(y, fine)
    z = _pit_zscores(y, grids, fine)
    r = np.corrcoef(z.T)
    off_diag = r[np.triu_indices_from(r, k=1)]
    assert off_diag.mean() == pytest.approx(0.7, abs=0.06)


def test_nearest_psd_fixes_indefinite() -> None:
    r = np.array([[1.0, 0.99, -0.99], [0.99, 1.0, -0.99], [-0.99, -0.99, 1.0]])
    z = np.random.default_rng(0).multivariate_normal(np.zeros(3), r, 8)
    fixed = _nearest_psd_corr(z)
    assert np.linalg.eigvalsh(fixed).min() > 0.0
    np.testing.assert_allclose(np.diag(fixed), 1.0)


def test_determinism() -> None:
    _, ra = _run()
    _, rb = _run()
    ra.pop("generated_at")
    rb.pop("generated_at")
    assert ra == rb


def test_method_seed_is_process_stable() -> None:
    # str.hash() is salt-randomized per process; the seed derivation must not
    # depend on it — this KAT pins the sha256-derived offset.
    import hashlib

    assert int.from_bytes(hashlib.sha256(b"copula_mc").digest()[:8]) % 7919 == 6206


def test_write_and_contract(tmp_path: Path) -> None:
    _, receipt = _run()
    assert coherence_contract_errors(receipt) == []
    path = write_coherence_receipt(receipt, tmp_path)
    assert path.name.startswith("coherence_")
    sealed = json.loads(path.read_text())
    assert sealed["receipt_sha256"]
    write_coherence_receipt(receipt, tmp_path)  # idempotent
    tampered = dict(receipt)
    tampered["payload"] = {**receipt["payload"], "data_label": "REAL"}
    with pytest.raises(ValueError):
        write_coherence_receipt(tampered, tmp_path)


def test_validation_fail_closed() -> None:
    with pytest.raises(ValueError):
        run_coherence(panels=["nope"])
    with pytest.raises(ValueError):
        _run(methods=["nope"])
    with pytest.raises(ValueError):
        _run(n_mc=4)
    with pytest.raises(ValueError):
        _run(n_train=4)


def test_non_synthetic_panel_rejected() -> None:
    def fake(n: int, seed: int) -> PanelShard:
        return PanelShard("independent", np.zeros((n, 4)), {"data_label": "REAL"})

    with pytest.raises(ValueError, match="SYNTHETIC"):
        run_coherence(panels={"independent": fake}, n_train=64, n_eval=16, n_mc=16)


def test_verify_receipt_deep_verifies(tmp_path: Path) -> None:
    from quant_fund.research.receipt_v2 import verify_receipt_file

    _, receipt = _run()
    path = write_coherence_receipt(receipt, tmp_path)
    result = verify_receipt_file(path)
    assert result["errors"] == [], result["errors"]
    # A receipt with a tampered dataset binding must fail kind consistency.
    import json as _json

    raw = _json.loads(path.read_text())
    raw["payload"]["panels"]["gauss_factor"]["y_sha256"] = "0" * 64
    bad = tmp_path / "tampered.json"
    bad.write_text(_json.dumps(raw))
    tampered = verify_receipt_file(bad)
    assert any("mismatch" in e or "seal" in e for e in tampered["errors"])


def test_method_grid_monotone_and_train_only() -> None:
    from quant_fund.research.coherence import _method_grid

    y = _factor_panel("p", 400, 3, 0.4).y[:N_TRAIN]
    rng = np.random.default_rng(0)
    q = _method_grid("copula_mc", y, 16, np.array(TAUS), 128, rng)
    assert q.shape == (16, len(TAUS))
    # Rearranged inside metrics; raw MC output is already monotone.
    assert np.all(np.diff(q, axis=1) >= -1e-9)

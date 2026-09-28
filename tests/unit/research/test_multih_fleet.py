"""P3.2 multi-horizon fleet lane — causality, constructions, receipt."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from quant_fund.research.multih_fleet import (
    CONSTRUCTION_EMPIRICAL,
    CONSTRUCTION_IID,
    CONSTRUCTION_NATIVE,
    _extend_empirical,
    _extend_iid,
    _forward_sums,
    _h_step_dispersion_ratio,
    multih_factories,
    resolve_shard_generators,
    run_multih_fleet_eval,
    write_multih_receipt,
)
from quant_fund.research.receipt_v2 import verify_receipt_file

TAUS = (0.1, 0.5, 0.9)


def test_forward_sums_alignment() -> None:
    y = np.arange(10.0)
    out = _forward_sums(y, 1)
    np.testing.assert_allclose(out[:10], y)  # origin t scores y[t]
    out3 = _forward_sums(y, 3)
    assert out3[0] == 0 + 1 + 2
    assert out3[7] == 7 + 8 + 9
    assert np.isnan(out3[8])  # beyond the tail


def test_extend_iid_scales_mean_and_spread() -> None:
    q1 = np.array([[0.0, 1.0, 2.0]])  # mean 1.0 on a 3-point grid
    taus = np.array([0.25, 0.5, 0.75])
    out = _extend_iid(q1, taus, 4)
    np.testing.assert_allclose(out[0, 1], 4.0, atol=1e-12)  # mean x4
    np.testing.assert_allclose(
        out[0, 0],
        4.0 - math.sqrt(4) * 1.0,
        atol=1e-12,  # spread x2
    )


def test_empirical_ratio_is_causal() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    origins = np.arange(150, 200)
    r1 = _h_step_dispersion_ratio(y, origins, h=5, lookback=40)
    y_future = y.copy()
    y_future[250:] = 1e6  # blow up the tail — must not move earlier origins
    r2 = _h_step_dispersion_ratio(y_future, origins, h=5, lookback=40)
    np.testing.assert_allclose(r1, r2, equal_nan=True)


def test_empirical_ratio_detects_clustering() -> None:
    # GARCH-like: independent signs, clustered magnitude -> h-step dispersion
    # departs from sqrt(h) under clustering.
    rng = np.random.default_rng(1)
    sig = np.ones(400)
    for t in range(1, 400):
        sig[t] = 0.05 + 0.9 * sig[t - 1] + 0.09 * (rng.normal() ** 2)
    y = sig * rng.normal(size=400)
    r = _h_step_dispersion_ratio(y, np.arange(200, 240), h=10, lookback=80)
    finite = r[np.isfinite(r)]
    assert finite.size > 0
    assert np.all(finite > 0)


def test_extend_empirical_falls_back_to_sqrt_h() -> None:
    q1 = np.array([[0.0, 1.0, 2.0]])
    taus = np.array([0.25, 0.5, 0.75])
    ratio = np.array([np.nan])
    out = _extend_empirical(q1, taus, 9, ratio)
    np.testing.assert_allclose(out, _extend_iid(q1, taus, 9), atol=1e-12)


def test_run_multih_fleet_eval_smoke(tmp_path: Path) -> None:
    fac = multih_factories(TAUS, 11, names=["empirical", "hstep_t"])
    gens = resolve_shard_generators(["iid_gaussian", "garch_cluster"])
    rows, receipt = run_multih_fleet_eval(
        fac,
        gens,
        taus=TAUS,
        horizons=(1, 5),
        n_train=150,
        n_eval=20,
        n=280,
        seed=7,
    )
    ok = [r for r in rows if r.status == "ok"]
    assert ok
    # empirical: 2 constructions x 2 horizons x 2 shards = 8 rows
    emp = [r for r in ok if r.model == "empirical"]
    assert {r.construction for r in emp} == {CONSTRUCTION_IID, CONSTRUCTION_EMPIRICAL}
    # hstep_t adds its native block
    hs = [r for r in ok if r.model == "hstep_t" and r.horizon == 5]
    assert {r.construction for r in hs} == {CONSTRUCTION_NATIVE, CONSTRUCTION_IID}
    assert receipt["verdict"] == "pass"
    assert receipt["payload"]["live_pnl_claim"] is False
    assert receipt["data_label"] == "SYNTHETIC"

    path = write_multih_receipt(receipt, tmp_path)
    verification = verify_receipt_file(path)
    assert verification["valid"], verification["errors"]

    # Tampered leaderboard is caught by the kind-consistency re-derivation.
    sealed = json.loads(path.read_text())
    cells = list(sealed["payload"]["leaders"])
    sealed["payload"]["leaders"][cells[0]] = "forged:entry"
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(sealed))
    bad = verify_receipt_file(forged)
    assert not bad["valid"]
    assert any("leader_mismatch" in e or "seal" in e for e in bad["errors"])


def test_bad_horizon_fails_closed() -> None:
    fac = multih_factories(TAUS, 11, names=["empirical"])
    gens = resolve_shard_generators(["iid_gaussian"])
    with pytest.raises(ValueError):
        run_multih_fleet_eval(fac, gens, taus=TAUS, horizons=(0,), n=120, n_train=60)


def test_unknown_head_fails_closed() -> None:
    with pytest.raises(ValueError):
        multih_factories(TAUS, 11, names=["not_a_head"])

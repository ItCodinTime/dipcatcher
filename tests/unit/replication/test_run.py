"""End-to-end replication run + sealed receipt tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.research.replication.receipt import (
    REPLICATION_SCHEMA,
    verify_replication_receipt,
    write_receipt,
)
from quant_fund.research.replication.run import run_replication


def test_run_replication_synthetic_writes_verifiable_receipt(tmp_path: Path) -> None:
    receipt, path = run_replication(
        use_synthetic=True,
        synthetic_seed=5,
        synthetic_kwargs={"n_assets": 12, "n_days": 300, "ar1_rho": -0.15},
        out_dir=tmp_path,
    )
    assert path is not None and path.exists()
    assert receipt["schema"] == REPLICATION_SCHEMA
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["live_pnl_claim"] is False
    assert receipt["research_only"] is True
    assert set(receipt["strategies"]) == {
        "tsmom_mop2012",
        "str_jegadeesh1990",
        "str_lehmann1990",
        "lowvol_bbw2011",
        "bab_fp2014",
        "overnight_lps2019",
        "tug_of_war_lps2019",
        "tom_mx2008",
    }
    result = verify_replication_receipt(path)
    assert result["valid"], result["errors"]


def test_run_replication_deterministic_seed(tmp_path: Path) -> None:
    kwargs = {"use_synthetic": True, "synthetic_seed": 9, "out_dir": tmp_path}
    r1, _ = run_replication(**kwargs)
    r2, _ = run_replication(**kwargs)
    # generated_at differs between runs; compare the sealed content hash inputs
    for key in ("strategies", "inputs_sha256", "data"):
        assert json.dumps(r1[key], sort_keys=True) == json.dumps(r2[key], sort_keys=True)


def test_receipt_rejects_tampering(tmp_path: Path) -> None:
    receipt, path = run_replication(
        use_synthetic=True,
        synthetic_seed=2,
        synthetic_kwargs={"n_assets": 10, "n_days": 200},
        out_dir=tmp_path,
    )
    assert path is not None
    payload = json.loads(path.read_text())
    payload["strategies"]["tsmom_mop2012"]["verdict"]["verdict"] = "replicated"
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload))
    result = verify_replication_receipt(tampered)
    assert not result["valid"]
    assert any("mismatch" in e for e in result["errors"])


def test_write_receipt_refuses_forbidden_metrics(tmp_path: Path) -> None:
    receipt, _ = run_replication(
        use_synthetic=True,
        synthetic_seed=2,
        synthetic_kwargs={"n_assets": 8, "n_days": 120},
        out_dir=None,
    )
    receipt["strategies"]["tsmom_mop2012"]["scores"]["sharpe_like_metric"] = 1.23
    with pytest.raises(ValueError, match="forbidden"):
        write_receipt(receipt, tmp_path)


def test_verdict_consistency_on_null_world() -> None:
    """Verdicts stay machine-consistent on a zero-planted-effect panel.

    (A weak bound: we do not assert "nothing replicates" — a |t|>=2 draw is
    possible by chance at the 5% level; we assert the verdict vocabulary and
    the replicated-implies-flags invariant instead.)
    """
    receipt, _ = run_replication(
        use_synthetic=True,
        synthetic_seed=4,
        synthetic_kwargs={"n_assets": 15, "n_days": 300},
        out_dir=None,
    )
    for _name, block in receipt["strategies"].items():
        v = block["verdict"]
        assert v["verdict"] in {"replicated", "contradicted", "inconclusive", "untestable"}
        if v["verdict"] == "replicated":
            assert v["direction_consistent_with_paper"] is True
            assert v["significant_at_5pct"] is True
        if v["verdict"] == "untestable":
            assert v["untestable_reason"]

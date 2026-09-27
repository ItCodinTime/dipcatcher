"""`make reality-gate` on an exported provenance ledger.

A fresh DB has no trials: the gate must skip explicitly (exit 0) rather than
fail closed inside `trial-report`. A DB that has a row must still be scored
with the unchanged filter.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from quant_fund.proofcore.contracts import (
    GENESIS_HASH,
    CodeFingerprint,
    DataManifestSummary,
    EnvFingerprint,
    ProofBundleV1,
    SignatureBlock,
    TrialLedgerRow,
    sha256_hex_bytes,
)
from quant_fund.proofcore.provenance import ProvenanceDB

REPO_ROOT = Path(__file__).resolve().parents[2]
_HEX = "ab" * 32


def _run_gate(db: Path, ledger: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", "reality-gate", f"PROOFCORE_DB={db}", f"PROOFCORE_LEDGER={ledger}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )


def test_reality_gate_skips_when_export_has_no_rows(tmp_path: Path) -> None:
    db = tmp_path / "proofcore.duckdb"
    ledger = tmp_path / "trials.jsonl"
    proc = _run_gate(db, ledger)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert "exported 0 trial rows" in combined
    assert "REALITY_FILTER_SKIP:" in proc.stdout
    assert "contains no trial rows" in proc.stdout
    assert "verdict=" not in proc.stdout
    assert ledger.read_text(encoding="utf-8").strip() == ""


def test_reality_gate_scores_a_recorded_trial(tmp_path: Path) -> None:
    """One recorded row is scored. n < 10 is insufficient_evidence (exit 1)."""
    db = tmp_path / "proofcore.duckdb"
    ledger = tmp_path / "trials.jsonl"
    bundle_id = sha256_hex_bytes(b"reality-gate-bundle")
    trial_id = sha256_hex_bytes(b"reality-gate-trial")
    bundle = ProofBundleV1(
        bundle_id=bundle_id,
        created_utc="2026-09-26T00:00:00+00:00",
        run_kind="research",
        code=CodeFingerprint(git_revision="a26be34", worktree_sha256=_HEX, dirty=False),
        data_manifest=DataManifestSummary(reads=[], merkle_root=_HEX, n_reads=0),
        config_sha256=_HEX,
        seed=7,
        env=EnvFingerprint(
            python_version="3.12.12",
            python_implementation="CPython",
            platform="linux",
            machine="x86_64",
            byteorder="little",
            packages={"dipcatcher": "0.1.0"},
        ),
        signal_log_sha256=_HEX,
        trade_log_sha256=_HEX,
        metrics_sha256=_HEX,
        metrics_recompute={"sharpe_periodic": 0.0, "n_trades": 0.0},
        prev_bundle_hash=GENESIS_HASH,
        signature=SignatureBlock(scheme="none", key_id="unsigned", value=""),
    )
    trial = TrialLedgerRow(
        trial_id=trial_id,
        bundle_hash=bundle_id,
        family="discovery",
        strategy="recorded-research-trial",
        cluster_id="recorded",
        created_utc="2026-09-26T00:00:00+00:00",
        n_obs=64,
        periods_per_year=252.0,
        sharpe_periodic=0.0,
        skew=0.0,
        kurtosis_raw=3.0,
        returns_sha256=_HEX,
    )
    with ProvenanceDB(db) as prov:
        prov.insert_bundle(bundle, None)
        prov.insert_trial(trial)

    proc = _run_gate(db, ledger)
    combined = proc.stdout + proc.stderr
    # ledger-gate exits 1. GNU make then exits 2 and prints "Error 1".
    assert proc.returncode == 2, combined
    assert "Error 1" in combined
    assert "REALITY_FILTER_READY: n_rows=1" in proc.stdout
    assert "verdict=insufficient_evidence" in proc.stdout
    assert "REALITY_FILTER_SKIP:" not in proc.stdout
    assert trial_id in ledger.read_text(encoding="utf-8")

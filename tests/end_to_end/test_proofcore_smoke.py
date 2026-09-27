"""PROOFCORE end-to-end smoke (DESIGN.md §2.6): pit -> proof -> leakage -> reality.

The cross-stack part activates once W1-W4 merge (merge order §10: W5 last);
until then those packages are absent and the corresponding test skips loudly.
The W5-owned part (provenance DB as the integration sink) always runs.
"""

from __future__ import annotations

import json

import pytest

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

_HEX = "cd" * 32


def _bundle(bundle_id: str) -> ProofBundleV1:
    return ProofBundleV1(
        bundle_id=bundle_id,
        created_utc="2026-09-26T00:00:00+00:00",
        run_kind="backtest",
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


def test_provenance_sink_smoke(tmp_path) -> None:
    """Bundle mint -> provenance insert -> trial row -> ledger export round trip."""
    db_path = tmp_path / "metadata" / "proofcore.duckdb"
    bundle = _bundle(sha256_hex_bytes(b"bundle-1"))
    trial = TrialLedgerRow(
        trial_id=sha256_hex_bytes(b"trial-1"),
        bundle_hash=bundle.bundle_id,
        family="discovery",
        strategy="smoke",
        cluster_id="smoke-cluster",
        created_utc="2026-09-26T00:00:00+00:00",
        n_obs=64,
        periods_per_year=252.0,
        sharpe_periodic=0.0,
        skew=0.0,
        kurtosis_raw=3.0,
        returns_sha256=sha256_hex_bytes(b"returns"),
    )
    with ProvenanceDB(db_path) as db:
        db.insert_bundle(bundle, None)
        db.insert_trial(trial)
        assert db.chain_head() == bundle.bundle_id
        rows = db.trials()
    assert len(rows) == 1
    assert rows[0] == trial
    # the exported JSONL is exactly what `quant reality` consumes (§14.4)
    line = json.dumps(rows[0].model_dump(mode="json"), sort_keys=True)
    assert TrialLedgerRow.model_validate(json.loads(line)) == trial


def test_cross_stack_modules_importable() -> None:
    """pit/proof/leakage/reality mount cleanly once W1-W4 land (§10 order)."""
    for module in (
        "quant_fund.pit",
        "quant_fund.proof",
        "quant_fund.leakage",
        "quant_fund.reality",
    ):
        pytest.importorskip(module, reason=f"{module} lands in a parallel workstream")

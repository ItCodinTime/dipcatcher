"""W5 provenance DB tests (DESIGN.md §9.1, §12 W5 row).

Round-trip, idempotency, tamper-evidence, chain-head, and export-schema tests
over a real duckdb file in tmp_path.
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
    ProvenanceError,
    SignatureBlock,
    TrialLedgerRow,
    sha256_hex_bytes,
)
from quant_fund.proofcore.provenance import ProvenanceDB

_HEX = "ab" * 32  # valid 64-char sha256 hex


def _bundle(bundle_id: str, *, prev: str = GENESIS_HASH, created: str) -> ProofBundleV1:
    return ProofBundleV1(
        bundle_id=bundle_id,
        created_utc=created,
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
            packages={"dipcatcher": "0.1.0", "numpy": "2.0.0"},
        ),
        signal_log_sha256=_HEX,
        trade_log_sha256=_HEX,
        metrics_sha256=_HEX,
        metrics_recompute={"sharpe_periodic": 0.0, "n_trades": 0.0},
        prev_bundle_hash=prev,
        signature=SignatureBlock(scheme="none", key_id="unsigned", value=""),
    )


def _trial(trial_id: str, bundle_hash: str, *, family: str = "discovery") -> TrialLedgerRow:
    return TrialLedgerRow(
        trial_id=trial_id,
        bundle_hash=bundle_hash,
        family=family,
        strategy="dip-bench",
        cluster_id="cluster-0",
        created_utc="2026-09-26T00:00:00+00:00",
        n_obs=252,
        periods_per_year=252.0,
        sharpe_periodic=0.01,
        skew=0.0,
        kurtosis_raw=3.0,
        returns_sha256=_HEX,
    )


def _id(seed_byte: int) -> str:
    return sha256_hex_bytes(bytes([seed_byte]))


class _FakeVerification:
    """Structural VerificationResultLike (W2's shape, DESIGN.md §5.5)."""

    def __init__(self, ok: bool) -> None:
        self.ok = ok
        self.reasons: list[str] = [] if ok else ["metrics_sha256 mismatch"]

    def model_dump(self, *, mode: str = "python") -> dict[str, object]:
        return {"ok": self.ok, "reasons": list(self.reasons)}


def test_bundle_round_trip(tmp_path) -> None:
    db = ProvenanceDB(tmp_path / "prov.duckdb")
    bundle = _bundle(_id(1), created="2026-09-26T00:00:00+00:00")
    db.insert_bundle(bundle, _FakeVerification(ok=True))
    rows = db.bundles()
    assert len(rows) == 1
    row = rows[0]
    assert row["bundle_id"] == bundle.bundle_id
    assert row["run_kind"] == "backtest"
    assert row["git_revision"] == "a26be34"
    assert row["config_sha256"] == bundle.config_sha256
    assert row["seed"] == 7
    assert row["merkle_root"] == bundle.data_manifest.merkle_root
    assert row["prev_bundle_hash"] == GENESIS_HASH
    assert row["signature_scheme"] == "none"
    assert row["verified_ok"] is True
    assert json.loads(row["verification_json"])["ok"] is True
    db.close()


def test_bundle_insert_idempotent(tmp_path) -> None:
    db = ProvenanceDB(tmp_path / "prov.duckdb")
    bundle = _bundle(_id(1), created="2026-09-26T00:00:00+00:00")
    db.insert_bundle(bundle, None)
    db.insert_bundle(bundle, _FakeVerification(ok=False))  # re-verify updates row
    rows = db.bundles()
    assert len(rows) == 1
    assert rows[0]["verified_ok"] is False
    db.close()


def test_chain_head_empty_db_is_genesis(tmp_path) -> None:
    with ProvenanceDB(tmp_path / "prov.duckdb") as db:
        assert db.chain_head() == GENESIS_HASH


def test_chain_head_follows_links(tmp_path) -> None:
    with ProvenanceDB(tmp_path / "prov.duckdb") as db:
        b1 = _bundle(_id(1), created="2026-09-26T00:00:00+00:00")
        b2 = _bundle(_id(2), prev=b1.bundle_id, created="2026-09-26T00:01:00+00:00")
        b3 = _bundle(_id(3), prev=b2.bundle_id, created="2026-09-26T00:02:00+00:00")
        db.insert_bundle(b1, None)
        db.insert_bundle(b2, None)
        db.insert_bundle(b3, None)
        assert db.chain_head() == b3.bundle_id


def test_trial_round_trip_and_family_filter(tmp_path) -> None:
    with ProvenanceDB(tmp_path / "prov.duckdb") as db:
        bundle = _bundle(_id(1), created="2026-09-26T00:00:00+00:00")
        db.insert_bundle(bundle, None)
        t1 = _trial(_id(11), bundle.bundle_id, family="calibration")
        t2 = _trial(_id(12), bundle.bundle_id, family="discovery")
        db.insert_trial(t1)
        db.insert_trial(t2)
        all_rows = db.trials()
        assert [r.trial_id for r in all_rows] == [t1.trial_id, t2.trial_id]
        cal = db.trials(family="calibration")
        assert [r.trial_id for r in cal] == [t1.trial_id]
        # values survive the DOUBLE/BIGINT round trip exactly
        r = all_rows[0]
        assert r.n_obs == 252
        assert r.periods_per_year == 252.0
        assert r.sharpe_periodic == 0.01
        assert r.kurtosis_raw == 3.0


def test_trial_insert_idempotent_but_tamper_refused(tmp_path) -> None:
    with ProvenanceDB(tmp_path / "prov.duckdb") as db:
        bundle = _bundle(_id(1), created="2026-09-26T00:00:00+00:00")
        db.insert_bundle(bundle, None)
        trial = _trial(_id(11), bundle.bundle_id)
        db.insert_trial(trial)
        db.insert_trial(trial)  # identical re-insert is a no-op
        assert len(db.trials()) == 1
        tampered = trial.model_copy(update={"sharpe_periodic": 99.0})
        with pytest.raises(ProvenanceError, match="append-only"):
            db.insert_trial(tampered)
        # ledger unchanged after the refused write
        assert db.trials()[0].sharpe_periodic == 0.01


def test_trial_requires_known_bundle(tmp_path) -> None:
    """trial_ledger.bundle_hash REFERENCES proof_bundles — orphans fail closed."""
    with ProvenanceDB(tmp_path / "prov.duckdb") as db:
        with pytest.raises(ProvenanceError):
            db.insert_trial(_trial(_id(11), _id(99)))


def test_export_schema_matches_trial_ledger_contract(tmp_path) -> None:
    """The --out JSONL export must round-trip through TrialLedgerRow (§14.4)."""
    from quant_fund.proofcore.cli import proofcore_app
    from typer.testing import CliRunner

    db_path = tmp_path / "prov.duckdb"
    out_path = tmp_path / "ledger.jsonl"
    with ProvenanceDB(db_path) as db:
        bundle = _bundle(_id(1), created="2026-09-26T00:00:00+00:00")
        db.insert_bundle(bundle, None)
        db.insert_trial(_trial(_id(11), bundle.bundle_id))

    result = CliRunner().invoke(
        proofcore_app, ["export", "--db", str(db_path), "--out", str(out_path)]
    )
    assert result.exit_code == 0, result.output
    lines = out_path.read_text().splitlines()
    assert len(lines) == 1
    row = TrialLedgerRow.model_validate(json.loads(lines[0]))
    assert row.trial_id == _id(11)
    assert set(json.loads(lines[0]).keys()) == set(TrialLedgerRow.model_fields.keys())


def test_db_persists_across_connections(tmp_path) -> None:
    path = tmp_path / "metadata" / "proofcore.duckdb"  # also exercises mkdir
    with ProvenanceDB(path) as db:
        db.insert_bundle(_bundle(_id(1), created="2026-09-26T00:00:00+00:00"), None)
    with ProvenanceDB(path) as db:
        assert len(db.bundles()) == 1
        assert db.chain_head() == _id(1)

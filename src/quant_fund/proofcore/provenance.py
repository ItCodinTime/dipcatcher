"""PROOFCORE provenance DB (duckdb) — W5.

Append-oriented provenance ledger for proof bundles (W2) and reality-filter
trial rows (W4). Legitimizes the existing duckdb hard dependency (A2 F14).

Layering (DESIGN.md §1.3, layer 3): imports contracts + duckdb + stdlib ONLY.
The standalone W2 verifier exists, but no bound result schema or validation
path connects its verdict to this database. Verification ingestion remains
fail-closed; bundles may be logged as unverified evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from quant_fund.proofcore.contracts import (
    GENESIS_HASH,
    ProofBundleV1,
    ProvenanceError,
    TrialLedgerRow,
    canonical_json_bytes,
)

DEFAULT_DB_PATH: Path = Path("data/metadata/proofcore.duckdb")

_BUNDLE_COLUMNS: tuple[str, ...] = (
    "bundle_id",
    "created_utc",
    "run_kind",
    "git_revision",
    "config_sha256",
    "seed",
    "merkle_root",
    "prev_bundle_hash",
    "signature_scheme",
    "bundle_json",
    "verified_ok",
    "verification_json",
)

_TRIAL_COLUMNS: tuple[str, ...] = (
    "trial_id",
    "bundle_hash",
    "family",
    "strategy",
    "cluster_id",
    "n_obs",
    "periods_per_year",
    "sharpe_periodic",
    "skew",
    "kurtosis_raw",
    "returns_sha256",
    "created_utc",
)

_DDL = """
CREATE TABLE IF NOT EXISTS proof_bundles (
    bundle_id TEXT PRIMARY KEY,
    created_utc TEXT NOT NULL,
    run_kind TEXT NOT NULL,
    git_revision TEXT NOT NULL,
    config_sha256 TEXT NOT NULL,
    seed BIGINT NOT NULL,
    merkle_root TEXT NOT NULL,
    prev_bundle_hash TEXT NOT NULL UNIQUE,
    signature_scheme TEXT NOT NULL,
    bundle_json TEXT NOT NULL,
    verified_ok BOOLEAN,
    verification_json TEXT
);
CREATE TABLE IF NOT EXISTS trial_ledger (
    trial_id TEXT PRIMARY KEY,
    bundle_hash TEXT NOT NULL REFERENCES proof_bundles (bundle_id),
    family TEXT NOT NULL,
    strategy TEXT NOT NULL,
    cluster_id TEXT NOT NULL,
    n_obs BIGINT NOT NULL,
    periods_per_year DOUBLE NOT NULL,
    sharpe_periodic DOUBLE NOT NULL,
    skew DOUBLE NOT NULL,
    kurtosis_raw DOUBLE NOT NULL,
    returns_sha256 TEXT NOT NULL,
    created_utc TEXT NOT NULL
);
"""


class ProvenanceDB:
    """duckdb file at <root>/metadata/proofcore.duckdb. Two tables:

    proof_bundles(bundle_id TEXT PK, created_utc TEXT, run_kind TEXT,
                  git_revision TEXT, config_sha256 TEXT, seed BIGINT,
                  merkle_root TEXT, prev_bundle_hash TEXT, signature_scheme TEXT,
                  bundle_json TEXT, verified_ok BOOLEAN, verification_json TEXT)
    trial_ledger(trial_id TEXT PK, bundle_hash TEXT REFERENCES proof_bundles,
                 family TEXT, strategy TEXT, cluster_id TEXT, n_obs BIGINT,
                 periods_per_year DOUBLE, sharpe_periodic DOUBLE, skew DOUBLE,
                 kurtosis_raw DOUBLE, returns_sha256 TEXT, created_utc TEXT)
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if self.path.parent and str(self.path.parent) not in ("", "."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._con = duckdb.connect(str(self.path))
            # duckdb enforces declared FK constraints natively (no pragma).
            self._con.execute(_DDL)
        except Exception as exc:  # duckdb.IOException and friends
            raise ProvenanceError(f"cannot open provenance DB at {self.path}: {exc}") from exc

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def insert_bundle(self, bundle: ProofBundleV1, verification: object | None) -> None:
        """Append one bundle, preserving exact content and chain order.

        An identical reinsert is a no-op. Verification results cannot be
        accepted until a bound verifier result schema and ingestion path exist.
        """
        if verification is not None:
            raise ProvenanceError(
                "verification ingestion unavailable until a bound result path exists"
            )
        bundle_json = canonical_json_bytes(bundle.model_dump(mode="json")).decode("utf-8")
        try:
            self._con.execute("BEGIN TRANSACTION")
            existing = self._con.execute(
                "SELECT bundle_json FROM proof_bundles WHERE bundle_id = ?",
                [bundle.bundle_id],
            ).fetchone()
            if existing is not None:
                if existing[0] != bundle_json:
                    raise ProvenanceError(
                        f"bundle {bundle.bundle_id} already stored with different contents; "
                        "the bundle ledger is append-only"
                    )
                self._con.execute("COMMIT")
                return
            expected_prev = self.chain_head()
            if bundle.prev_bundle_hash != expected_prev:
                raise ProvenanceError(
                    f"bundle {bundle.bundle_id} predecessor {bundle.prev_bundle_hash} "
                    f"does not match chain head {expected_prev}"
                )
            # Column names come from fixed module tuples; values remain bound parameters.
            self._con.execute(
                f"INSERT INTO proof_bundles ({', '.join(_BUNDLE_COLUMNS)}) "  # nosec B608
                f"VALUES ({', '.join('?' for _ in _BUNDLE_COLUMNS)})",
                [
                    bundle.bundle_id,
                    bundle.created_utc,
                    bundle.run_kind,
                    bundle.code.git_revision,
                    bundle.config_sha256,
                    bundle.seed,
                    bundle.data_manifest.merkle_root,
                    bundle.prev_bundle_hash,
                    bundle.signature.scheme,
                    bundle_json,
                    None,
                    None,
                ],
            )
            self._con.execute("COMMIT")
        except ProvenanceError:
            self._con.execute("ROLLBACK")
            raise
        except Exception as exc:
            self._con.execute("ROLLBACK")
            raise ProvenanceError(f"insert_bundle({bundle.bundle_id}) failed: {exc}") from exc

    def insert_trial(self, row: TrialLedgerRow) -> None:
        """Append one trial-ledger row.

        Idempotent for byte-identical re-inserts; refuses a ``trial_id`` whose
        stored row differs (raise ``ProvenanceError``) — tamper-evidence at
        the DB layer (DESIGN.md §9.1).
        """
        # Column names come from a fixed module tuple; trial_id remains bound.
        existing = self._con.execute(
            f"SELECT {', '.join(_TRIAL_COLUMNS)} FROM trial_ledger WHERE trial_id = ?",  # nosec B608
            [row.trial_id],
        ).fetchone()
        values = self._trial_values(row)
        if existing is not None:
            stored = [self._normalize_cell(v) for v in existing]
            wanted = [self._normalize_cell(v) for v in values]
            if stored != wanted:
                raise ProvenanceError(
                    f"trial {row.trial_id} already stored with different contents; "
                    "the trial ledger is append-only"
                )
            return
        try:
            # Column names come from fixed module tuples; values remain bound parameters.
            self._con.execute(
                f"INSERT INTO trial_ledger ({', '.join(_TRIAL_COLUMNS)}) "  # nosec B608
                f"VALUES ({', '.join('?' for _ in _TRIAL_COLUMNS)})",
                values,
            )
        except Exception as exc:
            raise ProvenanceError(f"insert_trial({row.trial_id}) failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def trials(self, *, family: str | None = None) -> list[TrialLedgerRow]:
        """All trial rows (optionally one family), ordered by (created_utc, trial_id)."""
        sql = f"SELECT {', '.join(_TRIAL_COLUMNS)} FROM trial_ledger"
        params: list[Any] = []
        if family is not None:
            sql += " WHERE family = ?"
            params.append(family)
        sql += " ORDER BY created_utc, trial_id"
        rows = self._con.execute(sql, params).fetchall()
        return [
            TrialLedgerRow(
                trial_id=r[0],
                bundle_hash=r[1],
                family=r[2],
                strategy=r[3],
                cluster_id=r[4],
                n_obs=r[5],
                periods_per_year=r[6],
                sharpe_periodic=r[7],
                skew=r[8],
                kurtosis_raw=r[9],
                returns_sha256=r[10],
                created_utc=r[11],
            )
            for r in rows
        ]

    def bundles(self) -> list[dict[str, Any]]:
        """All proof-bundle rows as plain dicts (audit/export path), in hash-chain
        order — the genesis-linked bundle first, each row followed by the bundle
        that points to it. Wall-clock order is not chain order: skewed
        ``created_utc`` values must not reorder an audit trail.

        Any row the genesis walk cannot reach (dangling prev pointer, hand-edited
        table) raises ``ProvenanceError`` — a broken chain must be loud, not
        silently reordered."""
        sql = f"SELECT {', '.join(_BUNDLE_COLUMNS)} FROM proof_bundles"  # nosec B608
        rows = [
            dict(zip(_BUNDLE_COLUMNS, r, strict=True)) for r in self._con.execute(sql).fetchall()
        ]
        # prev_bundle_hash is UNIQUE and bundle_id is the PK, so the table is a
        # linked list: at most one row claims each predecessor.
        by_prev = {row["prev_bundle_hash"]: row for row in rows}
        ordered: list[dict[str, Any]] = []
        cursor = by_prev.pop(GENESIS_HASH, None)
        while cursor is not None:
            ordered.append(cursor)
            cursor = by_prev.pop(cursor["bundle_id"], None)
        if by_prev:
            raise ProvenanceError(
                "proof_bundles does not form a single chain from genesis: "
                f"{len(by_prev)} row(s) unreachable — possible tamper"
            )
        return ordered

    def chain_head(self) -> str:
        """Current head of the bundle chain: the stored bundle no other bundle
        points to via ``prev_bundle_hash``. ``GENESIS_HASH`` on an empty DB."""
        row = self._con.execute(
            "SELECT bundle_id FROM proof_bundles "
            "WHERE bundle_id NOT IN (SELECT prev_bundle_hash FROM proof_bundles) "
            "ORDER BY created_utc DESC, bundle_id DESC LIMIT 1"
        ).fetchone()
        return row[0] if row is not None else GENESIS_HASH

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> ProvenanceDB:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _trial_values(row: TrialLedgerRow) -> list[Any]:
        return [
            row.trial_id,
            row.bundle_hash,
            row.family,
            row.strategy,
            row.cluster_id,
            row.n_obs,
            row.periods_per_year,
            row.sharpe_periodic,
            row.skew,
            row.kurtosis_raw,
            row.returns_sha256,
            row.created_utc,
        ]

    @staticmethod
    def _normalize_cell(value: Any) -> Any:
        # DuckDB returns Python ints for BIGINT and floats for DOUBLE; do not
        # coerce ints to float, which loses precision above 2**53.
        return value

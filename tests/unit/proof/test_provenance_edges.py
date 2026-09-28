"""ProvenanceDB edge paths: generic write failure rolls back, and a
conflicting trial re-insert stays append-only."""

from __future__ import annotations

from typing import Any

import pytest

from quant_fund.proofcore.contracts import ProvenanceError
from quant_fund.proofcore.provenance import ProvenanceDB
from tests.unit.test_proofcore_provenance import _HEX, _bundle, _trial

_T1 = "cd" * 32
_T2 = "ef" * 32

pytestmark = pytest.mark.synthetic


def test_insert_bundle_generic_failure_rolls_back(tmp_path: Any) -> None:
    """A non-ProvenanceError at INSERT time wraps + rolls back."""
    with ProvenanceDB(tmp_path / "prov.duckdb") as db:
        bundle = _bundle(_HEX, created="2026-09-27T00:00:00+00:00")

        real_execute = db._con.execute

        class _Boom:
            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def execute(self, sql: str, params: Any = None) -> Any:
                if "INSERT" in sql.upper():
                    raise RuntimeError("disk full")
                if params is not None:
                    return real_execute(sql, params)
                return real_execute(sql)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

        db._con = _Boom(db._con)  # type: ignore[assignment]
        with pytest.raises(ProvenanceError, match="insert_bundle"):
            db.insert_bundle(bundle, None)
        # Rolled back: nothing stored.
        assert db.chain_head() != _HEX


def test_insert_trial_conflicting_row_is_append_only(tmp_path: Any) -> None:
    with ProvenanceDB(tmp_path / "prov.duckdb") as db:
        bundle = _bundle(_HEX, created="2026-09-26T00:00:00+00:00")
        db.insert_bundle(bundle, None)
        db.insert_trial(_trial(_T1, _HEX, family="discovery"))
        conflicting = _trial(_T1, _HEX, family="bound")
        with pytest.raises(ProvenanceError, match="append-only"):
            db.insert_trial(conflicting)
        # The stored row is untouched.
        rows = db.trials(family="discovery")
        assert len(rows) == 1 and rows[0].family == "discovery"


def test_insert_trial_duplicate_is_noop(tmp_path: Any) -> None:
    with ProvenanceDB(tmp_path / "prov.duckdb") as db:
        bundle = _bundle(_HEX, created="2026-09-26T00:00:00+00:00")
        db.insert_bundle(bundle, None)
        row = _trial(_T1, _HEX)
        db.insert_trial(row)
        db.insert_trial(row)
        assert len(db.trials()) == 1


def test_insert_trial_write_failure_wraps(tmp_path: Any) -> None:
    with ProvenanceDB(tmp_path / "prov.duckdb") as db:
        bundle = _bundle(_HEX, created="2026-09-26T00:00:00+00:00")
        db.insert_bundle(bundle, None)

        real_execute = db._con.execute

        class _Boom:
            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def execute(self, sql: str, params: Any = None) -> Any:
                if "INSERT" in sql.upper():
                    raise RuntimeError("io error")
                if params is not None:
                    return real_execute(sql, params)
                return real_execute(sql)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

        db._con = _Boom(db._con)  # type: ignore[assignment]
        with pytest.raises(ProvenanceError, match="insert_trial"):
            db.insert_trial(_trial(_T2, _HEX))

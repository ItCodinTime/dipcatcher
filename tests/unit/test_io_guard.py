"""WAVE2 §6 IO guard tests (W8).

Covers:
- enforce mode blocks sneaky ``pl.read_parquet`` / ``pd.read_parquet`` /
  ``open()`` inside a proven decision window — on the driving thread AND from
  a bare ``threading.Thread`` worker (R2 registry coverage);
- redirect mode routes vault-mapped paths through a recording stub vault's
  ``asof(name, decision_time)`` and warns+proceeds on unknown paths;
- audit mode records everything and raises nothing;
- allowlisted runner tempfiles pass in enforce mode;
- outside a proven run (or inside a run but outside a window) behavior is
  untouched — zero interception;
- interposition is fully removed on exit, including on exception, and nested
  installs share one guard.
"""

from __future__ import annotations

import builtins
import io
import threading
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from quant_fund.leakage.guard import (
    IOGuardWarning,
    install_io_guard,
)
from quant_fund.leakage.watchdog import LeakageError
from quant_fund.proofcore.run_context import decision_window, proven_run

T0 = datetime(2024, 1, 1, tzinfo=UTC)


@pytest.fixture()
def parquet_file(tmp_path: Path) -> Path:
    path = tmp_path / "sneaky.parquet"
    pl.DataFrame({"x": [1, 2, 3]}).write_parquet(path)
    return path


class RecordingVault:
    """Stub vault: records asof(name, t) calls and returns a fixed frame."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, datetime]] = []
        self.frame = pl.DataFrame({"x": [10, 20]})

    def asof(self, name: str, t: datetime) -> pl.DataFrame:
        self.calls.append((name, t))
        return self.frame


def test_outside_run_everything_passes_through(parquet_file: Path) -> None:
    with install_io_guard("enforce") as guard:
        assert pl.read_parquet(parquet_file).height == 3
        assert pd.read_parquet(parquet_file).shape == (3, 1)
        with open(parquet_file, "rb") as fh:
            assert fh.read(4) == b"PAR1"
        with io.open(parquet_file, "rb") as fh:  # noqa: UP020 — io.open is an interposition target
            assert fh.read(4) == b"PAR1"
    assert guard.records == ()


def test_inside_run_but_outside_window_passes_through(parquet_file: Path) -> None:
    with install_io_guard("enforce") as guard, proven_run(object(), object()):
        # proven run active but NO decision window -> zero interception.
        assert pl.read_parquet(parquet_file).height == 3
    assert guard.records == ()


def test_enforce_blocks_polars_read_inside_window(parquet_file: Path) -> None:
    with (
        install_io_guard("enforce") as guard,
        proven_run(object(), object()),
        decision_window(T0),
        pytest.raises(LeakageError, match="io_guard"),
    ):
        pl.read_parquet(parquet_file)
    (rec,) = guard.records
    assert rec.function == "polars.read_parquet"
    assert rec.action == "blocked"
    assert rec.mode == "enforce"
    assert rec.decision_time == T0
    assert rec.path == str(parquet_file)


def test_enforce_blocks_pandas_read_inside_window(parquet_file: Path) -> None:
    with (
        install_io_guard("enforce") as guard,
        proven_run(object(), object()),
        decision_window(T0),
        pytest.raises(LeakageError, match="pandas.read_parquet"),
    ):
        pd.read_parquet(parquet_file)
    assert guard.records[-1].function == "pandas.read_parquet"


def test_enforce_blocks_open_read_inside_window(parquet_file: Path) -> None:
    with (
        install_io_guard("enforce") as guard,
        proven_run(object(), object()),
        decision_window(T0),
    ):
        with pytest.raises(LeakageError, match="raw open"):
            open(parquet_file, "rb")
        with pytest.raises(LeakageError):
            io.open(parquet_file)  # noqa: UP020 — io.open is an interposition target
    assert [r.function for r in guard.records] == ["open", "open"]


def test_enforce_blocks_pathless_source_inside_window() -> None:
    """A file-like source has no path -> unknown -> fail-closed in enforce."""
    with (
        install_io_guard("enforce") as guard,
        proven_run(object(), object()),
        decision_window(T0),
        pytest.raises(LeakageError),
    ):
        pl.read_parquet(io.BytesIO(b"not parquet"))
    assert guard.records[-1].path is None
    assert guard.records[-1].action == "blocked"


def test_enforce_blocks_bare_worker_thread(parquet_file: Path) -> None:
    """A bare threading.Thread has no contextvars context; the R2 registry
    fallback must still expose the window, so the worker's read is blocked."""
    errors: list[str] = []

    def worker() -> None:
        try:
            pl.read_parquet(parquet_file)
        except LeakageError:
            errors.append("blocked")

    with install_io_guard("enforce"), proven_run(object(), object()), decision_window(T0):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
    assert errors == ["blocked"]


def test_open_write_mode_is_not_intercepted(tmp_path: Path) -> None:
    target = tmp_path / "out.bin"
    with (
        install_io_guard("enforce") as guard,
        proven_run(object(), object()),
        decision_window(T0),
        open(target, "wb") as fh,
    ):
        fh.write(b"ok")
    assert target.read_bytes() == b"ok"
    assert guard.records == ()


def test_allowlisted_path_passes_in_enforce(parquet_file: Path, tmp_path: Path) -> None:
    other = tmp_path / "other.parquet"
    pl.DataFrame({"y": [1]}).write_parquet(other)
    with (
        install_io_guard("enforce", allowlist=(parquet_file,)) as guard,
        proven_run(object(), object()),
        decision_window(T0),
    ):
        assert pl.read_parquet(parquet_file).height == 3  # allowlisted
        with pytest.raises(LeakageError):  # not allowlisted
            pl.read_parquet(other)
    actions = [r.action for r in guard.records]
    assert actions == ["allowlisted", "blocked"]


def test_allowlisted_directory_prefix(parquet_file: Path, tmp_path: Path) -> None:
    with (
        install_io_guard("enforce", allowlist=(tmp_path,)) as guard,
        proven_run(object(), object()),
        decision_window(T0),
    ):
        assert pl.read_parquet(parquet_file).height == 3
    assert guard.records[-1].action == "allowlisted"


def test_redirect_routes_vault_mapped_path(tmp_path: Path) -> None:
    vault_root = tmp_path / "vault"
    part = vault_root / "main" / "part0.parquet"
    part.parent.mkdir(parents=True)
    pl.DataFrame({"x": [999]}).write_parquet(part)
    vault = RecordingVault()
    with (
        install_io_guard("redirect", vault_map={vault_root: vault}) as guard,
        proven_run(object(), object()),
        decision_window(T0),
    ):
        frame = pl.read_parquet(part)
    assert vault.calls == [("main/part0", T0)]
    assert frame.equals(vault.frame)  # the vault's frame, not the raw file
    (rec,) = guard.records
    assert rec.action == "redirected"
    assert rec.decision_time == T0


def test_redirect_unknown_path_warns_records_and_proceeds(parquet_file: Path) -> None:
    with (
        install_io_guard("redirect") as guard,
        proven_run(object(), object()),
        decision_window(T0),
        pytest.warns(IOGuardWarning, match="not vault-mapped"),
    ):
        frame = pl.read_parquet(parquet_file)
    assert frame.height == 3  # real read proceeded
    (rec,) = guard.records
    assert rec.action == "warned"


def test_redirect_vault_mapped_path_in_enforce_mode_also_redirects(tmp_path: Path) -> None:
    """Enforce never reads a vault-mapped path raw either — it goes through
    vault.asof so the read lands in the recorder/watchdog."""
    vault_root = tmp_path / "vault"
    part = vault_root / "ds.parquet"
    vault_root.mkdir()
    pl.DataFrame({"x": [1]}).write_parquet(part)
    vault = RecordingVault()
    with (
        install_io_guard("enforce", vault_map={vault_root: vault}),
        proven_run(object(), object()),
        decision_window(T0),
    ):
        frame = pl.read_parquet(part)
    assert vault.calls == [("ds", T0)]
    assert frame.equals(vault.frame)


def test_audit_records_everything_and_raises_nothing(parquet_file: Path) -> None:
    with install_io_guard("audit") as guard, proven_run(object(), object()), decision_window(T0):
        assert pl.read_parquet(parquet_file).height == 3
        assert pd.read_parquet(parquet_file).shape == (3, 1)
        with open(parquet_file, "rb") as fh:
            fh.read(1)
    # Every intercepted call is recorded (pandas' pyarrow engine may open()
    # the file internally, which is also recorded) and nothing raised.
    actions = [r.action for r in guard.records]
    assert actions == ["audited"] * len(actions)
    assert len(actions) >= 3
    functions = [r.function for r in guard.records]
    assert functions[0] == "polars.read_parquet"
    assert functions[1] == "pandas.read_parquet"
    assert "open" in functions


def test_audit_records_vault_mapped_without_redirecting(tmp_path: Path) -> None:
    vault_root = tmp_path / "vault"
    part = vault_root / "ds.parquet"
    vault_root.mkdir()
    pl.DataFrame({"x": [5]}).write_parquet(part)
    vault = RecordingVault()
    with (
        install_io_guard("audit", vault_map={vault_root: vault}) as guard,
        proven_run(object(), object()),
        decision_window(T0),
    ):
        frame = pl.read_parquet(part)
    assert vault.calls == []  # audit never redirects
    assert frame["x"].to_list() == [5]  # raw file read
    assert guard.records[-1].action == "audited"


def test_interposition_fully_removed_on_exit(parquet_file: Path) -> None:
    orig_pl, orig_pd = pl.read_parquet, pd.read_parquet
    orig_open, orig_io_open = builtins.open, io.open
    with install_io_guard("enforce"):
        assert pl.read_parquet is not orig_pl
        assert builtins.open is not orig_open
    assert pl.read_parquet is orig_pl
    assert pd.read_parquet is orig_pd
    assert builtins.open is orig_open
    assert io.open is orig_io_open
    # and reads still work post-teardown inside a proven window-less context
    assert pl.read_parquet(parquet_file).height == 3


def test_interposition_removed_on_exception() -> None:
    orig_pl, orig_open = pl.read_parquet, builtins.open
    with pytest.raises(RuntimeError, match="boom"), install_io_guard("enforce"):
        raise RuntimeError("boom")
    assert pl.read_parquet is orig_pl
    assert builtins.open is orig_open
    assert io.open is orig_open


def test_nested_install_shares_one_guard(parquet_file: Path) -> None:
    orig_pl = pl.read_parquet
    with install_io_guard("enforce") as outer:
        with install_io_guard("audit") as inner:  # config ignored: same guard
            assert inner is outer
        # inner exit must NOT remove interposition
        assert pl.read_parquet is not orig_pl
        with proven_run(object(), object()), decision_window(T0), pytest.raises(LeakageError):
            pl.read_parquet(parquet_file)
    assert pl.read_parquet is orig_pl


def test_invalid_mode_rejected() -> None:
    with (
        pytest.raises(ValueError, match="unknown IO guard mode"),
        install_io_guard("paranoid"),  # type: ignore[arg-type]
    ):
        pass


def test_mode_property() -> None:
    with install_io_guard("audit") as guard:
        assert guard.mode == "audit"


def test_captured_wrapper_after_teardown_passes_through(parquet_file: Path) -> None:
    """A wrapper reference captured during install must degrade to a pure
    passthrough once the guard is gone (zero interception, no stale state)."""
    with install_io_guard("enforce"):
        captured = pl.read_parquet
    assert captured(parquet_file).height == 3


def test_resolve_fallback_on_oserror(monkeypatch: pytest.MonkeyPatch, parquet_file: Path) -> None:
    """Path resolution failures must not crash the guard (fail-closed path)."""

    def boom(self: Path, *args: object, **kwargs: object) -> Path:
        raise OSError("no fs")

    monkeypatch.setattr(Path, "resolve", boom)
    with (
        install_io_guard("enforce") as guard,
        proven_run(object(), object()),
        decision_window(T0),
        pytest.raises(LeakageError),
    ):
        pl.read_parquet(parquet_file)
    assert guard.records[-1].action == "blocked"


def test_worker_thread_reads_are_recorded_with_thread_id(parquet_file: Path) -> None:
    done = threading.Event()

    def worker() -> None:
        try:
            pl.read_parquet(parquet_file)
        except LeakageError:
            pass
        finally:
            done.set()

    with install_io_guard("audit") as guard, proven_run(object(), object()), decision_window(T0):
        thread = threading.Thread(target=worker)
        thread.start()
        assert done.wait(timeout=10)
        thread.join()
        assert guard.records[-1].thread_id == thread.ident

"""Fuzz CLI argument handling: ``_collect_param_value`` + the typer app.

Two modes per input:

- first byte even → decode body as UTF-8 and feed ``_collect_param_value``
  (the ``--param key=value`` coercer). It must return ``int|float|str``;
  anything else or any exception is a finding.
- first byte odd → tokenize body on NUL/whitespace and invoke the real
  ``quant`` typer app via ``CliRunner`` inside an isolated filesystem (so a
  stray ``--config`` never resolves a repo YAML and all writes stay in the
  sandbox). Network and retry sleeps are disabled so fetch paths fail fast.
  ``result.exception`` is a finding unless it is a normal exit/usage error.
"""

from __future__ import annotations

import os
import socket
import sys
import tempfile
import time
from pathlib import Path

import yaml
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fuzz_common import run

from quant_fund.cli._app import _collect_param_value
from quant_fund.data.sources.base import SourceError

EXPECTED: tuple[type[BaseException], ...] = (
    OSError,  # missing config files, blocked sockets
    ValueError,  # load_config/AppConfig/typer.BadParameter surface here
    ValidationError,
    yaml.YAMLError,
    SourceError,  # fail-closed fetch path working as designed
)


def _no_network(*_args: object, **_kwargs: object) -> None:
    raise OSError("network disabled inside the fuzz harness")


def _install_guards() -> None:
    socket.create_connection = _no_network  # type: ignore[assignment]
    socket.getaddrinfo = _no_network  # type: ignore[assignment]
    _orig_sleep = time.sleep
    time.sleep = lambda s: _orig_sleep(0)  # keep retry loops fast


def _argv_from(data: bytes) -> list[str]:
    text = data.decode("utf-8", errors="replace")
    return [tok for tok in text.replace("\x00", " ").split() if tok][:24]


def test_one_input(data: bytes) -> None:
    if not data:
        return
    if data[0] % 2 == 0:
        _collect_param_value(data[1:].decode("utf-8", errors="replace"))
        return
    _install_guards()
    from typer.testing import CliRunner

    from quant_fund.cli._app import app

    argv = _argv_from(data[1:])
    if not argv:
        return
    runner = CliRunner()
    with tempfile.TemporaryDirectory(prefix="fuzz_cli_") as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            result = runner.invoke(app, argv, catch_exceptions=True)
        finally:
            os.chdir(cwd)
    exc = result.exception
    if exc is None or isinstance(exc, (SystemExit, *EXPECTED)):
        return
    raise exc


if __name__ == "__main__":
    raise SystemExit(run("cli_args", test_one_input, expected=EXPECTED))

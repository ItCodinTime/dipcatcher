"""Run the examples gallery offline, one subprocess per script."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = (
    "01_receipt_round_trip.py",
    "02_purged_walkforward_conformal.py",
    "03_synthetic_l2_book.py",
    "04_hf_ohlcv_1m.py",
    "05_phase1_evidence.py",
)
TIMEOUT_SECONDS = 120
BANNED_IMPORT_PREFIXES = (
    "quant_fund.paper",
    "quant_fund.execution",
    "quant_fund.api",
    "quant_fund.backtest",
)
_WRAPPER = """
import runpy
import socket

_connect = socket.socket.connect

def _blocked(self, address):
    host = ""
    if isinstance(address, tuple) and address:
        host = str(address[0])
    if host not in {{"127.0.0.1", "::1", "localhost"}}:
        raise OSError("examples CI forbids network access: %r" % (address,))
    return _connect(self, address)

socket.socket.connect = _blocked
runpy.run_path({script!r}, run_name="__main__")
"""


def _script(name: str) -> Path:
    path = REPO / "examples" / name
    if not path.is_file():
        raise AssertionError(f"missing example {path}")
    return path


def _run(name: str) -> subprocess.CompletedProcess[str]:
    script = _script(name)
    env = os.environ.copy()
    env["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    env["HF_OHLCV_1M_ALLOW_DOWNLOAD"] = "0"
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    try:
        return subprocess.run(
            [sys.executable, "-c", _WRAPPER.format(script=str(script))],
            cwd=REPO,
            env=env,
            text=True,
            capture_output=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode()
        if isinstance(stderr, bytes):
            stderr = stderr.decode()
        raise AssertionError(
            f"{name} exceeded {TIMEOUT_SECONDS}s\n{stdout[-2000:]}\n{stderr[-2000:]}"
        ) from exc


def _snapshot_matches_benchmark() -> bool:
    config_path = REPO / "configs" / "real_benchmark_us_wide.json"
    protocol = json.loads(config_path.read_text())
    dataset = (config_path.parent / protocol["dataset_path"]).resolve()
    if not dataset.is_file():
        return False
    digest = hashlib.sha256()
    with dataset.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == protocol["dataset_sha256"]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_runner_blocks_outbound_sockets(tmp_path: Path) -> None:
    probe = tmp_path / "probe_network.py"
    probe.write_text(
        "import socket\nsocket.create_connection(('1.1.1.1', 443), timeout=2)\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", _WRAPPER.format(script=str(probe))],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert "forbids network access" in result.stderr + result.stdout


def test_gallery_files_match_the_runner() -> None:
    present = sorted(path.name for path in (REPO / "examples").glob("*.py"))
    assert present == sorted(EXAMPLES)


@pytest.mark.parametrize("name", EXAMPLES)
def test_examples_do_not_import_trading_routes(name: str) -> None:
    modules = _imported_modules(_script(name))
    banned = [
        module
        for module in modules
        for prefix in BANNED_IMPORT_PREFIXES
        if module == prefix or module.startswith(prefix + ".")
    ]
    assert banned == []


@pytest.mark.parametrize("name", EXAMPLES)
def test_example_runs_offline(name: str) -> None:
    result = _run(name)
    assert result.returncode == 0, (
        f"{name} exited {result.returncode}\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}"
    )
    stdout = result.stdout
    assert "not_investment_advice=true" in stdout
    assert "no_live_trading_claim=true" in stdout
    assert "claim=research_only" in stdout
    if name == "01_receipt_round_trip.py":
        assert "data_label=SYNTHETIC" in stdout
        assert "receipt_valid=true" in stdout
        assert "tamper_bytes_changed=1" in stdout
        assert "tampered_valid=false" in stdout
        assert "tamper_rejected=true" in stdout
    elif name == "02_purged_walkforward_conformal.py":
        assert "data_label=tracked_real_snapshot" in stdout
        if _snapshot_matches_benchmark():
            assert "SKIP:" not in stdout
            assert "purge_embargo_ok=true" in stdout
            assert "pinball_0.5=" in stdout
            assert "crps_gaussian=" in stdout
            assert "conformal_coverage=" in stdout
        else:
            assert "SKIP: tracked real US snapshot" in stdout
    elif name == "03_synthetic_l2_book.py":
        assert "data_label=SYNTHETIC" in stdout
        assert "book_dgp=synthetic_lob" in stdout
        assert "vpin_mean=" in stdout
        assert "queue_imbalance_mean=" in stdout
        assert "join_coverage=" in stdout
    elif name == "04_hf_ohlcv_1m.py":
        assert "data_label=fixture" in stdout
        assert "source=hf_ohlcv_1m" in stdout
        assert "license=undeclared" in stdout
        assert "vendored_dataset=false" in stdout
        assert "allow_download=false" in stdout
        assert "redistribution=refused" in stdout
    elif name == "05_phase1_evidence.py":
        assert "verifier=phase1_evidence_index" in stdout
        assert "index_kind=phase1_evidence_index" in stdout
        assert "seal_errors=0" in stdout
        assert "research_only=true" in stdout
        assert "live_pnl_claim=false" in stdout
        assert "verification_authorizes_live_trading=false" in stdout
        assert "state=blocked" in stdout
        assert "state=complete" in stdout

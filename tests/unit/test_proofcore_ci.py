"""W5 CI-helper tests (DESIGN.md §12 W5 row).

- coverage floor config parses and matches the pyproject single source of truth
- receipts re-verification loop: fail-closed on bit-rot, empty dir, exceptions
- proofcore.yml (staged at docs/proofcore/) is valid YAML, declares the §9.3
  gate matrix, and every `make <target>` it invokes exists in the Makefile
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

from quant_fund.proofcore.ci import (
    REQUIRED_FLOOR_PACKAGES,
    coverage_floors,
    coverage_gate,
    receipt_paths,
    reverify_receipts,
)
from quant_fund.proofcore.contracts import ProofcoreError

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
# Canonical staged copy (always tracked). The activated path
# (.github/workflows/proofcore.yml) is installed by a workflow-scoped token.
WORKFLOW_STAGED = REPO_ROOT / "docs" / "proofcore" / "proofcore.yml"
WORKFLOW_ACTIVATED = REPO_ROOT / ".github" / "workflows" / "proofcore.yml"
MAKEFILE = REPO_ROOT / "Makefile"

# §9.3/§12 (lead-adjudicated): pit/proof/reality/proofcore 90, leakage 85.
EXPECTED_FLOORS = {"pit": 90, "proof": 90, "leakage": 85, "reality": 90, "proofcore": 90}


# ---------------------------------------------------------------------------
# Coverage floor config (A3 #2)
# ---------------------------------------------------------------------------


def test_coverage_floors_parse_from_pyproject() -> None:
    floors = coverage_floors(PYPROJECT)
    assert floors == EXPECTED_FLOORS
    assert set(floors) == set(REQUIRED_FLOOR_PACKAGES)


def test_coverage_floors_fail_closed_on_missing_table(tmp_path: Path) -> None:
    fake = tmp_path / "pyproject.toml"
    fake.write_text("[project]\nname = 'x'\n")
    with pytest.raises(ProofcoreError, match="coverage-floors"):
        coverage_floors(fake)


def test_coverage_floors_reject_floor_below_global_ratchet(tmp_path: Path) -> None:
    fake = tmp_path / "pyproject.toml"
    rows = "\n".join(
        f"{pkg} = {70 if pkg == 'leakage' else floor}" for pkg, floor in EXPECTED_FLOORS.items()
    )
    fake.write_text(f"[tool.proofcore.coverage-floors]\n{rows}\n")
    with pytest.raises(ProofcoreError, match="raise-never-lower"):
        coverage_floors(fake)


def test_coverage_gate_reports_failures_via_runner() -> None:
    """The gate shells out per package and collects every failure."""
    calls: list[list[str]] = []

    import subprocess

    def fake_runner(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        rc = 1 if "leakage" in cmd[4] else 0
        return subprocess.CompletedProcess(cmd, rc, stdout="TOTAL 42%", stderr="")

    failures = coverage_gate(PYPROJECT, runner=fake_runner)
    assert len(calls) == len(EXPECTED_FLOORS)
    assert len(failures) == 1 and "leakage" in failures[0]
    for cmd in calls:
        include = cmd[4]
        floor = int(cmd[5].split("=")[1])
        pkg = include.removeprefix("--include=src/quant_fund/").removesuffix("/*")
        assert floor == EXPECTED_FLOORS[pkg]


# ---------------------------------------------------------------------------
# Receipts re-verification loop (A3 #4)
# ---------------------------------------------------------------------------


def _write_receipt(path: Path, payload: dict[str, object]) -> None:
    body = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha256(body.encode()).hexdigest()
    path.write_text(json.dumps({"payload": payload, "sha256": digest}))


def _checksum_verifier(path: Path) -> bool:
    data = json.loads(path.read_text())
    body = json.dumps(data["payload"], sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest() == data["sha256"]


def test_receipt_paths_sorted(tmp_path: Path) -> None:
    for name in ("b.json", "a.json", "c.json"):
        _write_receipt(tmp_path / name, {"name": name})
    (tmp_path / "notes.txt").write_text("not a receipt")
    assert [p.name for p in receipt_paths(tmp_path)] == ["a.json", "b.json", "c.json"]


def test_reverify_all_clean(tmp_path: Path) -> None:
    for i in range(3):
        _write_receipt(tmp_path / f"r{i}.json", {"i": i})
    assert reverify_receipts(tmp_path, verify=_checksum_verifier) == []


def test_reverify_bit_rot_detected(tmp_path: Path) -> None:
    """§12 W5 adversarial: flip one byte in one committed receipt -> failure."""
    for i in range(3):
        _write_receipt(tmp_path / f"r{i}.json", {"i": i})
    victim = tmp_path / "r1.json"
    data = json.loads(victim.read_text())
    data["sha256"] = "0" * 64  # one corrupted field — the bit-rot simulation
    victim.write_text(json.dumps(data))
    failures = reverify_receipts(tmp_path, verify=_checksum_verifier)
    assert len(failures) == 1
    assert "r1.json" in failures[0]


def test_reverify_fail_closed_on_empty_dir_and_raising_verifier(tmp_path: Path) -> None:
    assert reverify_receipts(tmp_path, verify=_checksum_verifier) != []
    _write_receipt(tmp_path / "r.json", {"i": 0})

    def boom(path: Path) -> bool:
        raise RuntimeError("verifier crashed")

    failures = reverify_receipts(tmp_path, verify=boom)
    assert len(failures) == 1 and "verifier raised" in failures[0]


def test_committed_receipts_dir_nonempty() -> None:
    """The gate only means something while receipts/ carries tracked receipts."""
    assert receipt_paths(REPO_ROOT / "receipts"), "receipts/ must not be empty"


def test_cli_verifier_respects_exit_status(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    from quant_fund.proofcore import ci

    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}")

    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert cmd[-2:] == ["verify-receipt", str(receipt)]
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(ci.subprocess, "run", fake_run)
    assert not ci._cli_verifier(receipt)


def test_gate_main_dispatch_and_failure_reporting(monkeypatch, capsys, tmp_path: Path) -> None:
    from quant_fund.proofcore import ci

    monkeypatch.setattr(ci, "coverage_gate", lambda _path: ["pit below floor"])
    assert ci.main(["coverage-gate", "--pyproject", str(tmp_path / "pyproject.toml")]) == 1
    assert "pit below floor" in capsys.readouterr().err

    monkeypatch.setattr(ci, "reverify_receipts", lambda _path: [])
    assert ci.main(["receipts-reverify", str(tmp_path)]) == 0
    assert "GATE OK" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Workflow YAML validity + make-target cross-reference
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def workflow() -> dict:
    assert WORKFLOW_STAGED.is_file(), "docs/proofcore/proofcore.yml (staged workflow) missing"
    text = WORKFLOW_STAGED.read_text()
    # Strip the 3-line ACTIVATION header comment before parsing.
    body = text.split("\n", 3)[3]
    data = yaml.safe_load(body)
    assert isinstance(data, dict) and "jobs" in data
    return data


def test_staged_workflow_matches_activated_copy() -> None:
    """Once activated, .github/workflows/proofcore.yml must equal the staged copy."""
    if not WORKFLOW_ACTIVATED.is_file():
        pytest.skip("workflow not yet activated (needs workflow-scoped token)")
    staged_body = WORKFLOW_STAGED.read_text().split("\n", 3)[3]
    assert staged_body == WORKFLOW_ACTIVATED.read_text()


def test_workflow_declares_gate_matrix(workflow: dict) -> None:
    """§9.3 CI gate matrix, all present as jobs."""
    expected = {
        "proof-integrity",
        "leakage-scan",
        "reality-filter",
        "layering",
        "coverage-floors",
        "fx1-coverage",
        "proofcore-test",
    }
    assert expected <= set(workflow["jobs"])


def test_workflow_references_existing_make_targets(workflow: dict) -> None:
    makefile = MAKEFILE.read_text()
    targets = set(re.findall(r"^([a-zA-Z0-9_-]+):(?:\s|.*##)", makefile, flags=re.M))
    invoked: set[str] = set()
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            run = step.get("run", "")
            invoked.update(re.findall(r"\bmake\s+([a-z0-9-]+)", run))
    assert invoked, "workflow must invoke make targets"
    missing = invoked - targets
    assert not missing, f"workflow invokes undefined make targets: {sorted(missing)}"


def test_workflow_leakage_and_reality_gates_are_blocking(workflow: dict) -> None:
    jobs = workflow["jobs"]
    # The Make target owns --fail-on error; the workflow must invoke it as a
    # normal blocking step while retaining the report even on failure.
    leakage_runs = "\n".join(step.get("run", "") for step in jobs["leakage-scan"]["steps"])
    assert "make leakage-scan" in leakage_runs
    assert not jobs["leakage-scan"].get("continue-on-error", False)
    assert "blocking on errors" in yaml.safe_dump(jobs["leakage-scan"]).lower()
    leakage_target = MAKEFILE.read_text().split("leakage-scan:", 1)[1].split("\n\n", 1)[0]
    assert "--fail-on error" in leakage_target
    assert "||" not in leakage_target
    # Empty provenance exports skip with a notice. The job itself is blocking:
    # a scored ledger that is not 'pass' fails the build. continue-on-error
    # used to hide the empty-ledger exit 2.
    reality = jobs["reality-filter"]
    assert not reality.get("continue-on-error", False)
    reality_run = "\n".join(step.get("run", "") for step in reality["steps"])
    assert "make reality-gate" in reality_run
    assert "REALITY_FILTER_SKIP:" in reality_run
    assert "::notice title=reality-filter::" in reality_run
    for name in ("proof-integrity", "layering", "coverage-floors", "reality-filter"):
        assert not jobs[name].get("continue-on-error", False), name


def test_workflow_floors_match_pyproject(workflow: dict) -> None:
    """No second copy of the floor numbers may drift into the workflow."""
    floors = coverage_floors(PYPROJECT)
    for value in floors.values():
        assert f"--fail-under={value}" not in yaml.safe_dump(workflow), (
            "floors live in pyproject [tool.proofcore.coverage-floors]; "
            "the workflow must go through `make proofcore-coverage`"
        )

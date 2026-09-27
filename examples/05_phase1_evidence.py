# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Phase-1 evidence receipts
#
# Data label: tracked real snapshot (sealed runs) plus the verifier's checkout comparison.
#
# Not investment advice. No live-trading claim.
# The example calls `verify_phase1_index` and `verify_phase1_run`, the same path as
# `dipcatcher verify-research`. A runtime mismatch against the sealing interpreter
# is reported and does not by itself fail the summary. Any other verifier error
# fails the example. Passing verification does not authorize live trading.

# %%
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

from quant_fund.research.phase1_verify import verify_phase1_index, verify_phase1_run

ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "data" / "metadata" / "research" / "phase1_evidence_index.json"
_RUNTIME_NOTE = "runtime differs from this environment"


def _load_object(path: Path) -> dict[str, object]:
    loaded: object = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} must be a JSON object")
    parsed: dict[str, object] = {}
    for key, value in loaded.items():
        if not isinstance(key, str):
            raise SystemExit(f"{path} has a non-string key")
        parsed[key] = value
    return parsed


def _errors(result: dict[str, object]) -> list[str]:
    raw = result.get("errors")
    if not isinstance(raw, list):
        raise SystemExit("verifier result has no error list")
    return [str(item) for item in raw]


def _flag(value: object) -> str:
    if not isinstance(value, bool):
        return "missing"
    return str(value).lower()


def _receipt_file(path: Path) -> Path | None:
    compressed = path.with_name(path.name + ".gz")
    if path.is_file() and compressed.is_file():
        raise SystemExit(f"ambiguous receipt {path.name}")
    if path.is_file():
        return path
    if compressed.is_file():
        return compressed
    return None


def _open_receipt(path: Path) -> dict[str, object] | None:
    """Read a raw or gzip receipt the way the Phase-1 verifier does."""
    found = _receipt_file(path)
    if found is None:
        return None
    if found.suffix == ".gz":
        with gzip.open(found, "rt", encoding="utf-8") as handle:
            loaded: object = json.loads(handle.read())
    else:
        loaded = json.loads(found.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit(f"{found.name} is not a JSON object")
    parsed: dict[str, object] = {}
    for key, value in loaded.items():
        if not isinstance(key, str):
            raise SystemExit(f"{found.name} has a non-string key")
        parsed[key] = value
    return parsed


def _summarize_run(index_dir: Path, entry: dict[str, object], number: int) -> None:
    kind = entry.get("kind")
    relative = entry.get("path")
    if kind not in {"real_benchmark", "net_tournament"} or not isinstance(relative, str):
        raise SystemExit(f"index run {number} is not a sealed run entry")
    run_dir = (index_dir / relative).resolve()
    verified = verify_phase1_run(run_dir)
    manifest = _load_object(run_dir / "manifest.json")
    test_sha = entry.get("test_sha256")
    print(f"run={number}")
    print(f"kind={verified.get('kind')}")
    print(f"state={verified.get('state')}")
    print(f"run_valid={_flag(verified.get('valid'))}")
    print(f"research_only={_flag(manifest.get('research_only'))}")
    print(f"live_pnl_claim={_flag(manifest.get('live_pnl_claim'))}")
    print(f"test_sha256_sealed={str(isinstance(test_sha, str)).lower()}")
    validation = _open_receipt(run_dir / "validation.json")
    if validation is None:
        print("validation_receipt=absent")
    else:
        print("validation_receipt=sealed")
        print(f"promote={_flag(validation.get('promote'))}")
        print(f"validation_claim={validation.get('claim')}")
        if "selected" in validation:
            selected = validation["selected"]
            if selected is None:
                print("selected=null")
            elif isinstance(selected, str) and selected.strip():
                print(f"selected={selected}")
            else:
                print("selected=present")
        scores = validation.get("scores")
        if isinstance(scores, dict) and scores:
            first = next(iter(scores.values()))
            keys = sorted(str(key) for key in first) if isinstance(first, dict) else []
            print("score_models=" + ",".join(sorted(str(key) for key in scores)))
            print("score_keys=" + ",".join(keys))
    test_on_disk = _receipt_file(run_dir / "test.json") is not None
    if isinstance(test_sha, str) != test_on_disk:
        raise SystemExit(f"index run {number} test seal does not match the files on disk")
    print("test_receipt=sealed" if test_on_disk else "test_receipt=absent")


def main() -> None:
    if not INDEX_PATH.is_file():
        raise SystemExit(f"phase-1 evidence index is absent: {INDEX_PATH}")
    index = _load_object(INDEX_PATH)
    verified = verify_phase1_index(INDEX_PATH)
    errors = _errors(verified)
    seal_errors = [error for error in errors if _RUNTIME_NOTE not in error]
    runtime_errors = [error for error in errors if _RUNTIME_NOTE in error]
    print("verifier=phase1_evidence_index")
    print("data_label=tracked_real_snapshot")
    print("claim=research_only")
    print("not_investment_advice=true")
    print("no_live_trading_claim=true")
    print("verification_authorizes_live_trading=false")
    print(f"index_kind={index.get('kind')}")
    print(f"index_valid={_flag(verified.get('valid'))}")
    print(f"runtime_errors={len(runtime_errors)}")
    print(f"seal_errors={len(seal_errors)}")
    for error in seal_errors:
        print(f"seal_error={error}")
    runs = index.get("runs")
    if not isinstance(runs, list) or not runs:
        raise SystemExit("evidence index has no runs")
    for number, entry in enumerate(runs):
        if not isinstance(entry, dict):
            raise SystemExit(f"index run {number} is not an object")
        _summarize_run(INDEX_PATH.parent, entry, number)
    if seal_errors:
        raise SystemExit("phase-1 verification reported a non-runtime error")
    if index.get("kind") != "phase1_evidence_index":
        raise SystemExit("evidence file is not a phase-1 index")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"example_failed={type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

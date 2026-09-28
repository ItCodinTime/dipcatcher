"""Seal-coverage ratchet for durable JSON artifacts.

Every module under ``src/quant_fund`` that writes JSON to disk must either
self-digest (carry a ``receipt_sha256``-style marker or call the canonical
hashing helpers in the module) or be whitelisted here with a reason.

Adding a new JSON writer without sealing it fails this test. The whitelist
may only shrink — entries for files sealed on an open PR should be removed
once that PR merges.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "quant_fund"

# Markers that prove a module self-digests its JSON artifacts.
SEAL_MARKERS = (
    "receipt_sha256",
    "artifact_sha256",
    "manifest_sha256",
    "report_sha256",
    "seal_receipt",
    "canonical_json_bytes",
)

# Files that write JSON but produce no durable evidence — or are sealed on an
# open PR (remove the entry once that PR merges). Every entry needs a reason.
UNSEALED_WRITERS: dict[str, str] = {
    "config/loader.py": "dump_resolved echoes the tracked config file — snapshot, not evidence",
    "data/ingest.py": "sealed on PR #261 (data_manifest receipt_sha256)",
    "data/sources/storage.py": "sealed on PR #261 (per-parquet provenance sidecar)",
    "hedge_lab/gated_race.py": "sealed on PR #257",
    "hedge_lab/lightspeed_book.py": "sealed on PR #257",
    "hedge_lab/mirror.py": "sealed on PR #257",
    "hedge_lab/runner.py": "sealed on PR #257",
    "hedge_lab/target_hunt.py": "sealed on PR #257",
    "hedge_lab/v2_slate.py": "sealed on PR #257",
    "market_sim/__main__.py": "payload echoes to stdout — CLI report output, not a receipt writer",
    "paper/sim_live.py": "sealed on PR #258",
    "research/research100_cli.py": "catalog/benchmark emitter — stdout/report output, not a receipt writer",
    "research/sota_protocol.py": "sealed on PR #254",
}

_JSON_FUNCS = frozenset({"dump", "dumps"})
_WRITE_FUNCS = frozenset({"write_text", "write_bytes"})
_DIGEST_HELPERS = frozenset({"canonical_json_bytes", "canonical_json_str", "canonical_json"})


def _is_jsonish_call(node: ast.AST) -> bool:
    """True when ``node`` is a call to json.dump/dumps or a canonical helper."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name) and func.value.id == "json":
            return func.attr in _JSON_FUNCS
        return func.attr in _DIGEST_HELPERS
    if isinstance(func, ast.Name):
        return func.id in _DIGEST_HELPERS
    return False


def _expr_mentions_json(node: ast.AST) -> bool:
    return any(_is_jsonish_call(n) for n in ast.walk(node))


def json_writer_lines(tree: ast.AST) -> list[int]:
    """Line numbers of disk-write calls whose payload comes from json.*."""
    # Names bound to a json-producing expression anywhere in the module, so
    # ``data = json.dumps(x); path.write_text(data)`` is still caught.
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _expr_mentions_json(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bound.add(target.id)

    hits: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        func = node.func
        # json.dump(payload, fh)
        if isinstance(func.value, ast.Name) and func.value.id == "json" and func.attr == "dump":
            hits.append(node.lineno)
            continue
        if func.attr not in _WRITE_FUNCS:
            continue
        for arg in node.args:
            if (
                _expr_mentions_json(arg)
                or (isinstance(arg, ast.Name) and arg.id in bound)
                or (
                    # f-string / concatenation / wrapper carrying a bound name
                    any(isinstance(n, ast.Name) and n.id in bound for n in ast.walk(arg))
                )
            ):
                hits.append(node.lineno)
                break
    return hits


def _writer_files() -> dict[str, list[int]]:
    writers: dict[str, list[int]] = {}
    for path in sorted(SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - defensive
            continue
        hits = json_writer_lines(tree)
        if hits:
            writers[str(path.relative_to(SRC))] = hits
    return writers


def test_every_json_writer_seals_or_is_whitelisted() -> None:
    writers = _writer_files()
    uncovered: list[str] = []
    for rel, lines in writers.items():
        if rel in UNSEALED_WRITERS:
            continue
        source = (SRC / rel).read_text(encoding="utf-8")
        if any(marker in source for marker in SEAL_MARKERS):
            continue
        uncovered.append(f"{rel} (writes at lines {lines})")
    assert not uncovered, (
        "JSON-writing modules with no seal marker and no whitelist entry:\n  "
        + "\n  ".join(sorted(uncovered))
        + "\nSeal the artifact (receipt_sha256 over canonical_json_bytes) or add "
        "a justified UNSEALED_WRITERS entry."
    )


def test_whitelist_is_hygiene_only() -> None:
    for rel, reason in UNSEALED_WRITERS.items():
        assert (SRC / rel).is_file(), f"stale whitelist entry: {rel}"
        assert reason.strip(), f"whitelist entry for {rel} needs a justification"

"""Integrity checks for the web explorer's committed fixtures.

The app serves verbatim copies of sealed receipts plus mechanical exports of
committed artifacts (see ``web/scripts/export_fixtures.py``). These tests pin
the fixture contract: copies stay byte-identical to the sealed originals
(receipts are immutable — a mismatch means someone edited a receipt or the
fixtures are stale), the index references resolve, and equity series are
well-formed. No quant_fund imports; runs under plain pytest.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "web" / "public" / "fixtures"

SHA256_LEN = 64


def _load(name: str) -> dict:
    path = FIXTURES / name
    assert path.is_file(), f"missing fixture {name} — re-run export script"
    payload: object = json.loads(path.read_text())
    assert isinstance(payload, dict)
    return payload


def test_index_shape_and_references() -> None:
    index = _load("index.json")
    assert index["honesty"]["research_only"] is True
    assert index["honesty"]["live_pnl_claim"] is False
    assert index["strategies"], "index has no strategies"
    assert index["receipts"], "index has no receipts"
    for strategy in index["strategies"]:
        assert (FIXTURES / strategy["stats_file"]).is_file()
        if strategy["equity_file"] is not None:
            assert (FIXTURES / strategy["equity_file"]).is_file()
    for receipt in index["receipts"]:
        assert (FIXTURES / receipt["file"]).is_file()


def test_receipt_fixtures_are_byte_identical_to_sealed_sources() -> None:
    """Receipts are immutable: fixture copies must equal receipts/ exactly."""
    index = _load("index.json")
    for receipt in index["receipts"]:
        src = REPO_ROOT / receipt["file"]
        assert src.is_file(), f"source receipt missing: {receipt['file']}"
        fixture = FIXTURES / receipt["file"]
        assert fixture.read_bytes() == src.read_bytes(), (
            f"{receipt['file']} drifted — re-run "
            "web/scripts/export_fixtures.py (receipts are immutable)"
        )


def test_receipts_declare_research_only_flags() -> None:
    for path in sorted((FIXTURES / "receipts").glob("*.json")):
        payload = json.loads(path.read_text())
        assert payload.get("research_only") is True, path.name
        assert payload.get("live_pnl_claim") is False, path.name


def test_equity_fixtures_are_finite_sorted_series() -> None:
    index = _load("index.json")
    for strategy in index["strategies"]:
        eq = strategy.get("equity_file")
        if eq is None:
            continue
        payload = _load(eq)
        points = payload["points"]
        assert payload["rows"] == len(points)
        dates = [p[0] for p in points]
        assert dates == sorted(dates), f"{eq}: dates out of order"
        for row in points:
            _, nav, gross, net, turnover = row
            for value in (nav, gross, net, turnover):
                assert math.isfinite(value), eq
        assert payload["source_file"].startswith("artifacts/")


def test_hash_match_table_points_at_committed_files() -> None:
    index = _load("index.json")
    for digest, rel in index["hash_matches"].items():
        assert len(digest) == SHA256_LEN
        int(digest, 16)  # must be hex
        canonical = rel.removesuffix(" (canonical-json)")
        assert (REPO_ROOT / canonical).is_file(), f"match target missing: {rel}"


def test_strategies_keep_honesty_flags() -> None:
    for path in sorted((FIXTURES / "strategies").glob("*.json")):
        payload = json.loads(path.read_text())
        assert payload["research_only"] is True, path.name
        assert payload["live_pnl_claim"] is False, path.name
        assert payload["segments"], path.name


def _exporter():
    import importlib.util

    path = REPO_ROOT / "web/scripts/export_fixtures.py"
    spec = importlib.util.spec_from_file_location("web_fixture_export", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_hashes_ignore_dirty_untracked_and_symlink_bytes(tmp_path, monkeypatch) -> None:
    import hashlib
    import subprocess

    module = _exporter()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "src/example.py"
    source.parent.mkdir()
    committed = b"print('committed')\n"
    source.write_bytes(committed)
    (source.parent / "link.py").symlink_to("example.py")
    subprocess.run(["git", "-C", str(tmp_path), "add", "src"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    source.write_bytes(b"dirty\n")
    (source.parent / "untracked.py").write_bytes(b"untracked\n")
    matches = module._build_hash_index()
    assert matches == {hashlib.sha256(committed).hexdigest(): "src/example.py"}
    files = module._committed_files()
    assert module._committed_bytes(files["src/example.py"]) == committed


def test_export_hash_counts_match_fields_not_distinct_values() -> None:
    module = _exporter()
    values: list[str] = []
    module._collect_hashes(
        {"a_sha256": "a" * 64, "input_hashes": {"x": "a" * 64, "bad": "z" * 64}}, values
    )
    assert values == ["a" * 64, "a" * 64]

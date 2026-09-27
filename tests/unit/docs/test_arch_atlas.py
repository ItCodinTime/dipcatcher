"""Freshness and determinism contracts for ``scripts/gen_arch_diagrams.py``.

The atlas diagrams are generated from ``src/`` imports via stdlib ``ast``;
these tests pin the properties the CI job relies on: deterministic output,
byte-stable committed artifacts, stale detection when imports drift, and
fail-closed anchor verification for the curated diagrams.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "gen_arch_diagrams.py"


def _load():
    spec = importlib.util.spec_from_file_location("gen_arch_diagrams", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load()


def _make_repo(root: Path) -> Path:
    """Minimal fake repo: two first-party packages + atlas doc with markers."""
    for pkg in ("pkg", "other"):
        (root / "src" / "quant_fund" / pkg).mkdir(parents=True)
        (root / "src" / "quant_fund" / pkg / "__init__.py").write_text("")
    (root / "src" / "quant_fund" / "__init__.py").write_text("")
    (root / "src" / "quant_fund" / "pkg" / "a.py").write_text(
        "from quant_fund.pkg import b\nfrom quant_fund.other import c\nX = 1\n"
    )
    (root / "src" / "quant_fund" / "pkg" / "b.py").write_text("")
    (root / "src" / "quant_fund" / "other" / "c.py").write_text("")
    doc = root / "docs" / "ARCHITECTURE_ATLAS.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    blocks = "".join(
        f"<!-- BEGIN GENERATED: {name} -->\n<!-- END GENERATED: {name} -->\n"
        for name in ("module_deps", "data_flow", "paper_loop", "coverage")
    )
    doc.write_text("# atlas\n\n" + blocks)
    return root


def test_committed_artifacts_are_fresh() -> None:
    assert gen.stale_artifacts(REPO) == []


def test_check_cli_exit_zero() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_generation_is_deterministic() -> None:
    first = gen.generated_artifacts(REPO)
    second = gen.generated_artifacts(REPO)
    assert first == second
    manifest = json.loads(first["docs/architecture/manifest.json"])
    assert manifest["modules"] == sorted(manifest["modules"])
    edge_pairs = [(e["src"], e["dst"]) for e in manifest["module_edges"]]
    assert edge_pairs == sorted(edge_pairs)


def test_import_drift_marks_artifacts_stale(tmp_path, monkeypatch) -> None:
    repo = _make_repo(tmp_path)
    monkeypatch.setattr(gen, "ANCHORS", {})
    gen.write_all(repo)
    assert gen.stale_artifacts(repo) == []

    # Drop the cross-package import -> dep graph + manifest must go stale.
    (repo / "src" / "quant_fund" / "pkg" / "a.py").write_text(
        "from quant_fund.pkg import b\nX = 1\n"
    )
    stale = gen.stale_artifacts(repo)
    assert "docs/architecture/module_deps.mmd" in stale
    assert "docs/architecture/manifest.json" in stale


def test_module_addition_marks_manifest_stale(tmp_path, monkeypatch) -> None:
    repo = _make_repo(tmp_path)
    monkeypatch.setattr(gen, "ANCHORS", {})
    gen.write_all(repo)
    (repo / "src" / "quant_fund" / "pkg" / "newmod.py").write_text("Y = 2\n")
    assert "docs/architecture/manifest.json" in gen.stale_artifacts(repo)


def test_missing_anchor_fails_closed(tmp_path, monkeypatch) -> None:
    repo = _make_repo(tmp_path)
    monkeypatch.setattr(gen, "ANCHORS", {"ghost": ("src/quant_fund/pkg/b.py", "does_not_exist")})
    errors = gen.verify_anchors(repo)
    assert errors and "does_not_exist" in errors[0]


def test_atlas_splice_requires_marker_pair(tmp_path, monkeypatch) -> None:
    repo = _make_repo(tmp_path)
    monkeypatch.setattr(gen, "ANCHORS", {})
    gen.write_all(repo)
    doc = repo / "docs" / "ARCHITECTURE_ATLAS.md"
    text = doc.read_text()
    assert "BEGIN GENERATED: module_deps" in text
    assert "quant_fund_pkg" in text  # real generated content spliced in

    doc.write_text("# atlas without markers\n")
    # A doc with no marker pair cannot embed generated content -> stale.
    assert str(gen.ATLAS_DOC) in gen.stale_artifacts(repo)


def test_check_reports_stale_path(tmp_path, monkeypatch, capsys) -> None:
    repo = _make_repo(tmp_path)
    monkeypatch.setattr(gen, "ANCHORS", {})
    rc = gen.main(["--check", "--root", str(repo)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "stale" in out


def test_real_graph_shape_invariants() -> None:
    """Repo-level invariants the atlas documents (guard against regressions)."""
    modules, edges, module_edges = gen.collect_graph(REPO)
    assert len(modules) > 400  # whole-repo scan, not a subset
    assert all(m.split(".")[0] in {"quant_fund", "fx1"} for m in modules)
    # fx1 must never import the harness (ADR-0002 filesystem boundary).
    assert not any(s.startswith("fx1") and d.startswith("quant_fund") for s, d in module_edges)
    # Version is single-sourced: quant_fund -> fx1 is the only cross-root edge.
    cross_root = {(s, d) for s, d in module_edges if s.startswith("quant_fund") and d == "fx1"}
    assert cross_root == {("quant_fund", "fx1")}

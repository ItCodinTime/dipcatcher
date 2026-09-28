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
    # dataclasses consult sys.modules while the class body is executing.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gen = _load()


def _without_curated(monkeypatch) -> None:
    """Synthetic repos do not carry the real diagram anchors."""
    monkeypatch.setattr(gen, "ANCHORS", {})
    monkeypatch.setattr(gen, "DATA_FLOW_GROUPS", ())
    monkeypatch.setattr(gen, "PAPER_LOOP_PARTICIPANTS", ())


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
    _without_curated(monkeypatch)
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
    _without_curated(monkeypatch)
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
    _without_curated(monkeypatch)
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
    _without_curated(monkeypatch)
    rc = gen.main(["--check", "--root", str(repo)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "stale" in out


def test_real_graph_shape_invariants() -> None:
    """Repo-level invariants the atlas documents (guard against regressions)."""
    modules, _edges, module_edges = gen.collect_graph(REPO)
    assert len(modules) > 400  # whole-repo scan, not a subset
    assert all(m.split(".")[0] in {"quant_fund", "fx1"} for m in modules)
    # Pinned cross-root boundary (ADR-0002). Any added or removed edge fails.
    assert gen.cross_root_errors(module_edges) == []


def test_fx1_internal_import_fails_cross_root_pin() -> None:
    """``fx1.data.corpus`` is a cross-root edge even though it is not exactly ``fx1``.

    The previous predicate kept only destinations equal to ``fx1``, so an
    import of a model internal left the pin unchanged and the test passed.
    """
    edges = set(gen.QUANT_FUND_TO_FX1_EDGES) | set(gen.FX1_TO_QUANT_FUND_EDGES)
    edges.add(("quant_fund.pipeline.dataset", "fx1.data.corpus"))
    errors = gen.cross_root_errors(edges)
    assert errors and any("fx1.data.corpus" in err for err in errors)
    legacy = {(src, dst) for src, dst in edges if src.startswith("quant_fund") and dst == "fx1"}
    assert ("quant_fund.pipeline.dataset", "fx1.data.corpus") not in legacy
    assert ("quant_fund.pipeline.dataset", "fx1.data.corpus") in gen.quant_fund_to_fx1_edges(edges)


def test_subgraph_ids_do_not_collide_with_package_nodes() -> None:
    text = gen.render_module_deps(
        ["quant_fund", "quant_fund.cli", "fx1", "fx1.data"],
        gen.Counter({("quant_fund.cli", "fx1"): 1, ("fx1.data", "fx1"): 2}),
    )
    assert 'subgraph cluster_quant_fund["quant_fund (harness)"]' in text
    assert 'subgraph cluster_fx1["fx1 (model project)"]' in text
    assert 'quant_fund["quant_fund"]' in text
    assert 'fx1["fx1"]' in text
    assert "subgraph quant_fund[" not in text
    assert "subgraph fx1[" not in text


def test_rendered_label_drift_fails_even_when_bytes_match(tmp_path, monkeypatch) -> None:
    """A label that does not name the anchored symbol fails closed.

    Byte-comparing a committed file to a renderer that ignores anchors would
    stay green. Binding verification has to reject that renderer on its own.
    """
    repo = _make_repo(tmp_path)
    source = repo / "src" / "quant_fund" / "pkg" / "a.py"
    source.write_text("def ingest():\n    return 1\ndef other():\n    return 2\n")
    monkeypatch.setattr(gen, "ANCHORS", {"ingest": ("src/quant_fund/pkg/a.py", "ingest")})
    monkeypatch.setattr(
        gen,
        "DATA_FLOW_GROUPS",
        (
            gen.FlowGroup(
                "pit",
                "data",
                (gen.AnchoredNode("ingest", ("ingest",), "bronze"),),
            ),
        ),
    )
    monkeypatch.setattr(gen, "PAPER_LOOP_PARTICIPANTS", ())
    assert gen.verify_diagram_bindings(repo) == []
    gen.write_all(repo)
    assert gen.stale_artifacts(repo) == []

    # Retargeting the anchor at another real symbol must stale the picture.
    monkeypatch.setattr(gen, "ANCHORS", {"ingest": ("src/quant_fund/pkg/a.py", "other")})
    assert "docs/architecture/data_flow.mmd" in gen.stale_artifacts(repo)

    monkeypatch.setattr(gen, "ANCHORS", {"ingest": ("src/quant_fund/pkg/a.py", "ingest")})

    def drifted() -> str:
        header = str(gen.GENERATED_HEADER)
        return (
            header
            + "\n"
            + 'flowchart LR\n  subgraph cluster_pit["data"]\n    ingest["ghost()\\nbronze"]\n  end\n'
        )

    drifted_text = drifted()
    (repo / "docs" / "architecture" / "data_flow.mmd").write_text(drifted_text)
    monkeypatch.setattr(gen, "render_data_flow", drifted)
    errors = gen.verify_diagram_bindings(repo)
    assert any("anchored symbol ingest" in err for err in errors)
    try:
        gen.stale_artifacts(repo)
    except SystemExit as exc:
        assert "diagram binding verification failed" in str(exc)
    else:
        raise AssertionError("label drift must fail closed")

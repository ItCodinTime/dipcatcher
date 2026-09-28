"""Unit tests for scripts/check_import_boundaries.py.

The checker is stdlib-only; these tests are likewise dependency-free and run
under both pytest and ``python -m unittest``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "check_import_boundaries.py"
REAL_CONFIG = REPO_ROOT / "configs" / "arch_boundaries.toml"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_import_boundaries", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look the module up here
    spec.loader.exec_module(mod)
    return mod


checker = _load_checker()


def _write_tree(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


CONFIG_TOML = """\
[[layers]]
name = "low"
packages = ["schemas", "data"]

[[layers]]
name = "high"
packages = ["research", "cli"]

[[deny]]
name = "no-cli"
importer = "**"
forbidden = ["quant_fund.cli.**"]
except_importers = ["quant_fund.cli.**"]
note = "library must not import cli"

[[deny]]
name = "research-purity"
importer = "quant_fund.research.**"
forbidden = ["quant_fund.paper.**"]

[[deny]]
name = "no-fx1"
importer = "quant_fund.**"
except_importers = ["quant_fund"]
forbidden = ["fx1.**"]

[[allow_only]]
name = "fx1-surface"
importer = "fx1.**"
target_root = "quant_fund"
allowed = ["quant_fund.schemas.**"]
"""


def _config(**kwargs) -> checker.Config:
    raw_layers = kwargs.get(
        "layers",
        (
            checker.Layer("low", frozenset({"schemas", "data"})),
            checker.Layer("high", frozenset({"research", "cli"})),
        ),
    )
    return checker.Config(
        layers=raw_layers,
        denies=kwargs.get("denies", ()),
        allow_onlys=kwargs.get("allow_onlys", ()),
        baseline=kwargs.get("baseline", ()),
    )


class TestPatternMatch(unittest.TestCase):
    def test_star_star_suffix_matches_package_and_below(self):
        self.assertTrue(checker.match("quant_fund.paper.**", "quant_fund.paper"))
        self.assertTrue(checker.match("quant_fund.paper.**", "quant_fund.paper.loop"))
        self.assertTrue(checker.match("quant_fund.paper.**", "quant_fund.paper.deep.mod"))
        self.assertFalse(checker.match("quant_fund.paper.**", "quant_fund.pap"))
        self.assertFalse(checker.match("quant_fund.paper.**", "quant_fund.paperx"))

    def test_bare_star_star_matches_everything(self):
        self.assertTrue(checker.match("**", "quant_fund.anything"))
        self.assertTrue(checker.match("**", "fx1"))

    def test_exact_and_fnmatch(self):
        self.assertTrue(checker.match("quant_fund.x", "quant_fund.x"))
        self.assertFalse(checker.match("quant_fund.x", "quant_fund.x.y"))
        self.assertTrue(checker.match("quant_fund.*.broker", "quant_fund.exec.broker"))


class TestImportExtraction(unittest.TestCase):
    """AST extraction over a synthetic module tree."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = Path(self.tmp.name) / "src"
        _write_tree(
            self.src,
            {
                "quant_fund/__init__.py": "",
                "quant_fund/data/__init__.py": "",
                "quant_fund/data/ingest.py": "",
                "quant_fund/research/__init__.py": "",
                "quant_fund/cli/__init__.py": "",
                "quant_fund/cli/main.py": "",
            },
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _sites(self, rel: str, content: str):
        path = self.src / rel
        path.write_text(content, encoding="utf-8")
        modules = checker.collect_modules(self.src)
        known = frozenset(modules)
        importer = checker.module_name_for(path, self.src)
        return checker.iter_import_sites(
            path, rel, importer, checker._package_parts(path, importer), known
        )

    def test_absolute_import(self):
        sites = self._sites("quant_fund/data/ingest.py", "import quant_fund.cli.main\nimport os\n")
        self.assertEqual([s.imported for s in sites], ["quant_fund.cli.main"])
        self.assertFalse(sites[0].lazy)

    def test_from_import_expands_known_submodule(self):
        sites = self._sites("quant_fund/data/ingest.py", "from quant_fund.cli import main\n")
        imported = {s.imported for s in sites}
        self.assertIn("quant_fund.cli", imported)
        self.assertIn("quant_fund.cli.main", imported)

    def test_from_import_symbol_is_not_a_module(self):
        sites = self._sites("quant_fund/data/ingest.py", "from quant_fund.research import run\n")
        # ``run`` is not a real module: only the package target is emitted.
        self.assertEqual([s.imported for s in sites], ["quant_fund.research"])

    def test_relative_import_resolves(self):
        sites = self._sites("quant_fund/data/ingest.py", "from ..cli import main\n")
        imported = {s.imported for s in sites}
        self.assertIn("quant_fund.cli", imported)
        self.assertIn("quant_fund.cli.main", imported)

    def test_lazy_and_type_checking_flags(self):
        sites = self._sites(
            "quant_fund/data/ingest.py",
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n"
            "    import quant_fund.cli\n"
            "def f():\n"
            "    import quant_fund.research\n",
        )
        by_name = {s.imported: s for s in sites}
        self.assertTrue(by_name["quant_fund.cli"].type_checking)
        self.assertFalse(by_name["quant_fund.cli"].lazy)
        self.assertTrue(by_name["quant_fund.research"].lazy)


class TestEvaluation(unittest.TestCase):
    def _site(self, importer, imported, lazy=False, tc=False):
        return checker.ImportSite("src/x.py", 1, importer, imported, lazy, tc)

    def test_layer_order_blocks_upward(self):
        cfg = _config()
        v = checker.evaluate([self._site("quant_fund.data.ingest", "quant_fund.research.x")], cfg)
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0].rule, "layer-order")

    def test_downward_and_same_layer_ok(self):
        cfg = _config()
        sites = [
            self._site("quant_fund.research.x", "quant_fund.data.ingest"),
            self._site("quant_fund.research.x", "quant_fund.research.y"),
        ]
        self.assertEqual(checker.evaluate(sites, cfg), [])

    def test_lazy_import_exempt_from_layer_but_not_deny(self):
        cfg = _config(
            denies=(
                checker.DenyRule(
                    "deny", "**", ("quant_fund.research.**",), ("quant_fund.research.safe",)
                ),
            )
        )
        upward_lazy = self._site("quant_fund.data.x", "quant_fund.research.y", lazy=True)
        self.assertEqual(checker.evaluate([upward_lazy], cfg)[0].rule, "deny")
        # lazy + allowed target -> nothing fires
        allowed_lazy = self._site("quant_fund.data.x", "quant_fund.research.safe", lazy=True)
        self.assertEqual(checker.evaluate([allowed_lazy], cfg), [])

    def test_facade_exempt_from_layer_order(self):
        cfg = _config()
        sites = [
            self._site("quant_fund.research.x", "quant_fund"),  # bare root
            self._site("quant_fund", "quant_fund.research.x"),  # __init__ facade
            self._site("quant_fund.public", "quant_fund.cli.x"),  # public facade
        ]
        self.assertEqual(checker.evaluate(sites, cfg), [])

    def test_unclassified_package_flagged(self):
        cfg = _config()
        v = checker.evaluate([self._site("quant_fund.newpkg.x", "quant_fund.data.y")], cfg)
        self.assertEqual(v[0].rule, "unclassified-package")
        v = checker.evaluate([self._site("quant_fund.data.x", "quant_fund.newpkg.y")], cfg)
        self.assertEqual(v[0].rule, "unclassified-package")

    def test_deny_rule_with_except_importers(self):
        cfg = _config(
            denies=(
                checker.DenyRule(
                    "no-cli",
                    "**",
                    ("quant_fund.cli.**",),
                    except_importers=("quant_fund.cli.**",),
                ),
            )
        )
        bad = self._site("quant_fund.data.x", "quant_fund.cli.main")
        ok = self._site("quant_fund.cli.bootstrap", "quant_fund.cli.main")
        rules = {v.rule for v in checker.evaluate([bad], cfg)}
        # the deny fires; the layer-order also fires (data < cli) — both are real
        self.assertIn("no-cli", rules)
        self.assertEqual(checker.evaluate([ok], cfg), [])

    def test_allow_only_rule(self):
        cfg = _config(
            allow_onlys=(
                checker.AllowOnlyRule(
                    "fx1-surface",
                    "fx1.**",
                    "quant_fund",
                    ("quant_fund.schemas.**",),
                ),
            )
        )
        bad = self._site("fx1.forecast.x", "quant_fund.research.catalog")
        ok = self._site("fx1.forecast.x", "quant_fund.schemas.bars")
        outside = self._site("fx1.forecast.x", "polars")  # not under target_root
        self.assertEqual(checker.evaluate([bad], cfg)[0].rule, "fx1-surface")
        self.assertEqual(checker.evaluate([ok, outside], cfg), [])


class TestBaseline(unittest.TestCase):
    def _violation(self, file, module, rule="layer-order"):
        site = checker.ImportSite(file, 5, "quant_fund.data.x", module, False, False)
        return checker.Violation(site, rule, "detail")

    def test_baseline_suppresses_only_matches(self):
        entries = (checker.BaselineEntry("src/a.py", "quant_fund.research.**", "layer-order"),)
        v_match = self._violation("src/a.py", "quant_fund.research.catalog")
        v_other_file = self._violation("src/b.py", "quant_fund.research.catalog")
        v_other_rule = self._violation("src/a.py", "quant_fund.research.x", "deny")
        active, baselined, stale = checker.apply_baseline(
            [v_match, v_other_file, v_other_rule], entries
        )
        self.assertEqual(baselined, [v_match])
        self.assertEqual({v.site.file for v in active}, {"src/b.py", "src/a.py"})
        self.assertEqual(stale, [])

    def test_stale_entry_reported(self):
        entries = (checker.BaselineEntry("src/gone.py", "quant_fund.research", "layer-order"),)
        active, baselined, stale = checker.apply_baseline([], entries)
        self.assertEqual(active, [])
        self.assertEqual(baselined, [])
        self.assertEqual(len(stale), 1)

    def test_rule_wildcard(self):
        entries = (checker.BaselineEntry("src/a.py", "quant_fund.x", "*"),)
        v = self._violation("src/a.py", "quant_fund.x", "anything")
        _, baselined, _ = checker.apply_baseline([v], entries)
        self.assertEqual(len(baselined), 1)


class TestConfigLoading(unittest.TestCase):
    def test_real_config_loads(self):
        cfg = checker.load_config(REAL_CONFIG)
        self.assertGreaterEqual(len(cfg.layers), 5)
        self.assertGreaterEqual(len(cfg.denies), 3)
        self.assertEqual(len(cfg.allow_onlys), 1)
        self.assertGreaterEqual(len(cfg.baseline), 1)

    def test_every_quant_fund_package_is_classified(self):
        """The real config must claim every current top-level package."""
        cfg = checker.load_config(REAL_CONFIG)
        claimed = set().union(*(layer.packages for layer in cfg.layers))
        on_disk = {
            p.name
            for p in (REPO_ROOT / "src" / "quant_fund").iterdir()
            if p.is_dir() and (p / "__init__.py").exists()
        }
        self.assertEqual(on_disk - claimed, set())

    def test_bad_config_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.toml"
            bad.write_text("[[layers]]\npackages = 5\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                checker.load_config(bad)
            with self.assertRaises(ValueError):
                checker.load_config(Path(tmp) / "missing.toml")


class TestEndToEnd(unittest.TestCase):
    """run() against a synthetic tree + generated config."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = self.root / "src"
        _write_tree(
            self.src,
            {
                "quant_fund/__init__.py": "",
                "quant_fund/data/__init__.py": "",
                "quant_fund/data/clean.py": "import quant_fund.schemas\n",
                "quant_fund/data/bad.py": "import quant_fund.research\n",
                "quant_fund/research/__init__.py": "",
                "quant_fund/schemas/__init__.py": "",
            },
        )
        self.config_path = self.root / "cfg.toml"
        self.config_path.write_text(CONFIG_TOML, encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_clean_tree_exit_zero(self):
        (self.src / "quant_fund/data/bad.py").unlink()
        cfg = checker.load_config(self.config_path)
        cwd = os.getcwd()
        os.chdir(self.root)
        try:
            rc = checker.run(Path("src"), cfg, show_baselined=False)
        finally:
            os.chdir(cwd)
        self.assertEqual(rc, 0)

    def test_violation_exit_one(self):
        cfg = checker.load_config(self.config_path)
        cwd = os.getcwd()
        os.chdir(self.root)
        try:
            rc = checker.run(Path("src"), cfg, show_baselined=False)
        finally:
            os.chdir(cwd)
        self.assertEqual(rc, 1)

    def test_baselined_violation_exit_zero(self):
        self.config_path.write_text(
            CONFIG_TOML
            + '\n[[baseline]]\nfile = "src/quant_fund/data/bad.py"\n'
            + 'module = "quant_fund.research"\nrule = "layer-order"\n',
            encoding="utf-8",
        )
        cfg = checker.load_config(self.config_path)
        cwd = os.getcwd()
        os.chdir(self.root)
        try:
            rc = checker.run(Path("src"), cfg, show_baselined=True)
        finally:
            os.chdir(cwd)
        self.assertEqual(rc, 0)

    def test_stale_baseline_exit_one(self):
        (self.src / "quant_fund/data/bad.py").unlink()
        self.config_path.write_text(
            CONFIG_TOML
            + '\n[[baseline]]\nfile = "src/quant_fund/data/bad.py"\n'
            + 'module = "quant_fund.research"\nrule = "layer-order"\n',
            encoding="utf-8",
        )
        cfg = checker.load_config(self.config_path)
        cwd = os.getcwd()
        os.chdir(self.root)
        try:
            rc = checker.run(Path("src"), cfg, show_baselined=False)
        finally:
            os.chdir(cwd)
        self.assertEqual(rc, 1)

    def test_real_tree_passes_with_real_config(self):
        """The shipped config must pass on the actual repo (the CI gate)."""
        cfg = checker.load_config(REAL_CONFIG)
        cwd = os.getcwd()
        os.chdir(REPO_ROOT)
        try:
            rc = checker.run(REPO_ROOT / "src", cfg, show_baselined=False)
        finally:
            os.chdir(cwd)
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()

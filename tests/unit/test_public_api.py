"""Lock the dipcatcher public API and check it against the engines."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import subprocess
import sys
import tomllib
import typing
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

import quant_fund
import quant_fund.public as public
from quant_fund.backtest.engine import StaleValuationError as EngineStaleValuationError
from quant_fund.backtest.engine import run_backtest as engine_run_backtest
from quant_fund.data.ingest import ingest as engine_ingest
from quant_fund.research.agent import HypothesisResult, ResearchNotebook
from quant_fund.research.verify import verify_research_artifact

_SNAPSHOT = Path(__file__).with_name("public_api_snapshot.txt")
_ROOT = Path(__file__).resolve().parents[2]

_STRICT_FLAGS = (
    "check_untyped_defs",
    "disallow_any_generics",
    "disallow_incomplete_defs",
    "disallow_subclassing_any",
    "disallow_untyped_calls",
    "disallow_untyped_decorators",
    "disallow_untyped_defs",
    "extra_checks",
    "strict_equality",
    "warn_return_any",
    "warn_unused_ignores",
)


def _annotation(value: object) -> str:
    if value is inspect.Signature.empty or value is inspect.Parameter.empty:
        return ""
    if isinstance(value, str):
        return value
    forwarded = getattr(value, "__forward_arg__", None)
    if isinstance(forwarded, str):
        return forwarded
    name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    return name if isinstance(name, str) else repr(value)


def _signature(fn: object) -> str:
    signature = inspect.signature(fn)  # type: ignore[arg-type]
    parts: list[str] = []
    saw_keyword_only = False
    for param in signature.parameters.values():
        if param.name == "self":
            continue
        if param.kind is inspect.Parameter.KEYWORD_ONLY and not saw_keyword_only:
            parts.append("*")
            saw_keyword_only = True
        rendered = param.name
        annotation = _annotation(param.annotation)
        if annotation:
            rendered = f"{rendered}: {annotation}"
        if param.default is not inspect.Parameter.empty:
            rendered = f"{rendered} = {param.default!r}"
        parts.append(rendered)
    name = getattr(fn, "__name__", "fn")
    return f"{name}({', '.join(parts)}) -> {_annotation(signature.return_annotation)}"


def _typeddict_lines(cls: type) -> list[str]:
    lines = [f"class {cls.__name__}"]
    for name, annotation in inspect.get_annotations(cls, eval_str=False).items():
        rendered = _annotation(annotation)
        lines.append(f"  {name}: {rendered}")
    return lines


def _dataclass_lines(cls: type) -> list[str]:
    lines = [f"class {cls.__name__}"]
    for field in dataclasses.fields(cls):
        rendered = field.type if isinstance(field.type, str) else _annotation(field.type)
        lines.append(f"  {field.name}: {rendered}")
    for name, member in cls.__dict__.items():
        if name.startswith("_") or not inspect.isfunction(member):
            continue
        lines.append(f"  {_signature(member)}")
    return lines


def render_public_api() -> str:
    """Stable signature listing of the public surface."""
    lines = [
        "# dipcatcher public API snapshot",
        "# Intentional changes update this file in the same commit.",
        "",
        "[quant_fund.__all__]",
        *quant_fund.__all__,
        "",
        "[quant_fund.public.__all__]",
        *public.__all__,
        "",
    ]
    for name in public.__all__:
        obj = getattr(public, name)
        lines.append(f"[{name}]")
        if inspect.isfunction(obj):
            lines.append(_signature(obj))
        elif typing.is_typeddict(obj):
            lines.extend(_typeddict_lines(obj))
        elif isinstance(obj, typing.TypeAliasType):
            lines.append(f"type {name} = {obj.__value__}")
        elif dataclasses.is_dataclass(obj) and inspect.isclass(obj):
            lines.extend(_dataclass_lines(obj))
        elif inspect.isclass(obj) and issubclass(obj, BaseException):
            bases = ", ".join(base.__name__ for base in obj.__bases__)
            lines.append(f"class {name}({bases})")
        elif inspect.isclass(obj) and hasattr(obj, "model_fields"):
            lines.append(f"class {name}")
            lines.extend(f"  {field_name}" for field_name in obj.model_fields)
        elif inspect.isclass(obj):
            lines.append(f"class {name}")
            lines.append(_signature(obj.__init__).replace("__init__", f"{name}.__init__", 1))
        else:
            lines.append(f"{name}: {type(obj).__name__}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def test_public_api_snapshot() -> None:
    actual = render_public_api()
    expected = _SNAPSHOT.read_text(encoding="utf-8")
    assert actual == expected


def test_public_names_match_package_root() -> None:
    assert list(quant_fund.__all__) == ["__firm__", "__version__", *public.__all__]


def test_package_import_does_not_load_the_research_stack() -> None:
    script = (
        "import sys\n"
        "import quant_fund\n"
        "assert 'quant_fund.public' not in sys.modules, sys.modules.keys()\n"
        "assert quant_fund.__firm__ == 'Artificial Hedge'\n"
        "load_config = quant_fund.load_config\n"
        "assert load_config.__name__ == 'load_config'\n"
        "assert 'quant_fund.public' in sys.modules\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=_ROOT,
    )
    assert completed.returncode == 0, completed.stderr


def test_public_module_does_not_spell_any() -> None:
    tree = ast.parse((Path(public.__file__) or Path()).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "Any":
            raise AssertionError("quant_fund.public must not reference Any")
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            imported = [alias.name for alias in node.names]
            assert "Any" not in imported


def test_py_typed_marks_the_package_partial() -> None:
    marker = _ROOT / "src" / "quant_fund" / "py.typed"
    assert marker.read_text(encoding="utf-8") == "partial\n"


def test_mypy_strict_flags_are_per_module() -> None:
    config = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    mypy = config["tool"]["mypy"]
    assert mypy["strict"] is False
    matched = [
        block
        for block in mypy["overrides"]
        if block.get("module") == ["quant_fund", "quant_fund.public"]
    ]
    assert len(matched) == 1
    block = matched[0]
    assert "strict" not in block
    assert block.get("ignore_errors") is not True
    for flag in _STRICT_FLAGS:
        assert block[flag] is True
    assert block["implicit_reexport"] is False


def test_stale_valuation_error_is_the_engine_type() -> None:
    assert public.StaleValuationError is EngineStaleValuationError


def _bars_and_weights() -> tuple[pl.DataFrame, pl.DataFrame]:
    bars = pl.DataFrame(
        {
            "security_id": ["A", "A", "A"],
            "event_time": [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                datetime(2024, 1, 3, tzinfo=UTC),
            ],
            "open": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
            "close_total_return": [100.0, 110.0, 110.0],
            "volume": [1_000_000.0, 1_000_000.0, 1_000_000.0],
            "adv": [100_000_000.0, 100_000_000.0, 100_000_000.0],
            "vol_20": [0.02, 0.02, 0.02],
            "source": ["file", "file", "file"],
        }
    )
    weights = pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 1, tzinfo=UTC)],
            "security_id": ["A"],
            "target_weight": [1.0],
        }
    )
    return bars, weights


def test_run_backtest_matches_engine_frames(tmp_path: Path) -> None:
    cfg = public.load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = True
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 1.0
    cfg.risk_gate.max_order_notional = 1e12
    bars, weights = _bars_and_weights()
    published = public.run_backtest(bars, weights, cfg, initial_nav=100_000.0)
    engine = engine_run_backtest(bars, weights, cfg, initial_nav=100_000.0)
    assert published.equity.equals(engine.equity)
    assert published.fills.equals(engine.fills)
    assert published.frictionless is engine.frictionless
    assert published.source_note == engine.source_note
    assert published.metrics["research_only"] is True
    assert published.metrics["live_pnl_claim"] is False
    assert published.metrics["total_return"] == engine.metrics["total_return"]
    assert published.equity["nav"][0] == 110_000.0


def test_ingest_returns_the_engine_paths(tmp_path: Path) -> None:
    cfg = public.load_config("configs/research.yaml")
    cfg.data.root = tmp_path / "public"
    cfg.data.synthetic_n_assets = 2
    cfg.data.synthetic_n_days = 16
    published = public.ingest(cfg)
    direct_cfg = public.load_config("configs/research.yaml")
    direct_cfg.data.root = tmp_path / "engine"
    direct_cfg.data.synthetic_n_assets = 2
    direct_cfg.data.synthetic_n_days = 16
    direct = engine_ingest(direct_cfg)
    assert set(published) == set(direct)
    for name, path in published.items():
        assert path.is_file()
        assert path.name == direct[name].name


def test_verify_research_matches_the_harness_report(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert public.verify_research(missing) == verify_research_artifact(missing)
    parsed = tmp_path / "empty.json"
    parsed.write_text("{}\n", encoding="utf-8")
    assert public.verify_research(parsed) == verify_research_artifact(parsed)


def test_research_run_receipt_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notebook = ResearchNotebook(
        schema_version=3,
        firm="Artificial Hedge",
        product="Dipcatcher",
        version="test",
        generated_at="2026-01-01T00:00:00+00:00",
        data_source="SYNTHETIC",
        synthetic=True,
        disclaimer="research only",
        ranking_target="future_return_1",
        claim="research_only",
        families={"ranking": {"mean_ic": float("nan")}},
        rankers=[{"name": "ridge", "mean_ic": 0.1}],
        hypotheses=[
            HypothesisResult(
                id="H0",
                statement="synthetic check",
                test="none",
                statistic=float("nan"),
                p_value=float("nan"),
                reject_raw=False,
                reject_fdr=False,
                decision="inconclusive",
                family="discovery",
            )
        ],
        scorecard={"ranking": {"executed": True, "claim": "research_metric_only"}},
        provenance={"run_id": "abc", "point_in_time": True},
        artifacts={"json": "latest.json"},
    )
    monkeypatch.setattr(public, "_run_research", lambda _config: notebook)
    cfg = public.load_config("configs/research.yaml")
    run = public.run_research(cfg)
    assert run.hypotheses == tuple(notebook.hypotheses)
    assert run.families["ranking"]["mean_ic"] is None
    receipt = run.to_receipt()
    assert receipt["hypotheses"][0]["p_value"] is None
    assert receipt["claim"] == "research_only"
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert public.read_research_receipt(path) == receipt


def test_read_research_receipt_rejects_a_non_object(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        public.read_research_receipt(path)

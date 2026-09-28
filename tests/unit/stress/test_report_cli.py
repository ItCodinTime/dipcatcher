"""Stress report, CLI, French parser, and the import boundary."""

from __future__ import annotations

import ast
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from quant_fund.cli._app import app
from quant_fund.research.catalog.constants import FORBIDDEN_RESEARCH_METRIC_KEYS
from quant_fund.stress.report import (
    _assert_no_forbidden_keys,
    build_stress_report,
    render_html,
    render_markdown,
)
from quant_fund.stress.strategy import (
    Position,
    ResearchStrategy,
    load_return_panel,
    portfolio_returns,
)
from quant_fund.stress.vendor_cache import (
    FRENCH_DAILY_URL,
    fetch_french_daily_factors,
    parse_french_daily_factors,
    slice_dates,
)

_FRENCH = """
Header line that is not data

,Mkt-RF,SMB,HML,RF
19260701,    0.10,   -0.24,   -0.28,  0.009
19260702,    0.45,   -0.32,   -0.08,  0.009
19871019,  -17.41,    0.10,    0.20,  0.021
20000310,   -1.20,    0.30,    0.10,  0.021
20080915,   -4.50,   -0.20,    0.80,  0.007
20150824,   -3.90,    0.15,   -0.40,  0.000
20180205,   -4.10,    0.05,   -0.20,  0.005
20200316,   -11.99,   0.50,   -1.10,  0.004
20200317,    -5.00,   0.20,    0.10,  0.004
20221230,    1.75,   -0.20,    0.40,  0.016

Annual Factors
,Mkt-RF,SMB,HML,RF
1927, 10.0, 1.0, 2.0, 3.0
"""


def _keys(payload: object, found: set[str]) -> None:
    if isinstance(payload, dict):
        found.update(str(key).lower() for key in payload)
        for value in payload.values():
            _keys(value, found)
    elif isinstance(payload, list):
        for value in payload:
            _keys(value, found)


def test_catalog_report_escapes_html_and_skips_dfast_in_the_equity_cloud() -> None:
    strategy = ResearchStrategy(
        name="<script>alert(1)</script>",
        positions=(
            Position("us_equity", 0.6, "return"),
            Position("ust_10y", 7.0, "duration"),
        ),
    )
    report = build_stress_report(strategy, n_scenarios=32, n_boot=50, seed=0)
    assert report["research_only"] is True
    assert report["live_trading"] is False
    assert "Research simulation only" in report["disclaimer"]
    assert report["synthetic"]["label"] == "SYNTHETIC"
    assert report["synthetic"]["market_evidence"] is False
    ids = {row["crisis_id"] for row in report["crises"]}
    assert "crash_1987" in ids and "dfast_2025_severely_adverse" in ids
    cloud = report["reverse_historical_equity"]
    assert cloud["status"] == "ok"
    assert cloud["shocks"] == pytest.approx([-0.226, -0.57, -0.10, -0.078, -0.042])
    assert -0.50 not in cloud["shocks"]
    keys: set[str] = set()
    _keys(report, keys)
    assert keys.isdisjoint(FORBIDDEN_RESEARCH_METRIC_KEYS)
    page = render_html(report)
    assert "<script>alert" not in page
    assert "&lt;script&gt;" in page
    text = render_markdown(report)
    assert "crash_1987" in text and "live_trading: False" in text
    with pytest.raises(ValueError):
        _assert_no_forbidden_keys({"sharpe": 1.0})
    with pytest.raises(ValueError):
        build_stress_report(strategy, n_scenarios=8)


def test_panel_report_labels_synthetic_and_ridges_a_singular_covariance(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    calm = rng.normal(0.0002, 0.01, size=30)
    stressed = rng.normal(-0.002, 0.03, size=30)
    panel = np.column_stack([np.concatenate([calm, stressed]), np.concatenate([calm, stressed])])
    names = ("a", "b")
    strategy = ResearchStrategy(
        "panel", (Position("us_equity", 1.0, "return"),), asset_weights=(("a", 0.5), ("b", 0.5))
    )
    report = build_stress_report(
        strategy, names=names, panel=panel, n_scenarios=32, n_boot=50, seed=1
    )
    assert report["synthetic"]["label"] == "SYNTHETIC"
    assert report["synthetic"]["market_evidence"] is False
    assert report["synthetic"]["stationary_bootstrap"]["n_paths"] == 32
    for block in ("garch_t_copula", "hmm", "jump_diffusion"):
        assert report["synthetic"][block]["status"] in {"ok", "unavailable"}
    assert report["risk"]["backtest"]["status"] == "unavailable"
    assert report["reverse_return_panel"]["ridge"] == pytest.approx(1e-6)
    assert report["reverse_return_panel"]["status"] == "ok"

    path = tmp_path / "returns.csv"
    rows = ["date,a,b"]
    for i in range(12):
        rows.append(f"2020-01-{i + 1:02d},{panel[i, 0]},{panel[i, 1]}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    loaded_names, loaded = load_return_panel(path)
    assert loaded_names == ("a", "b")
    assert loaded.shape[0] == 12
    short = build_stress_report(
        strategy, names=loaded_names, panel=loaded, n_scenarios=32, n_boot=50, seed=2
    )
    assert short["risk"]["estimates"]["status"] == "unavailable"
    assert short["synthetic"]["garch_t_copula"]["status"] == "unavailable"


def test_long_panel_backtest_runs_without_forbidden_keys() -> None:
    rng = np.random.default_rng(5)
    panel = rng.normal(0.0, 0.01, size=(100, 2))
    panel[:40, :] *= 0.5
    names = ("x", "y")
    strategy = ResearchStrategy("long", (Position("us_equity", 1.0, "return"),))
    report = build_stress_report(
        strategy, names=names, panel=panel, n_scenarios=32, n_boot=50, seed=3
    )
    backtest = report["risk"]["backtest"]
    assert backtest["kupiec"]["status"] == "ok"
    assert backtest["christoffersen"]["status"] in {"ok", "undefined"}
    assert "acerbi_szekely_z1" in backtest
    assert report["synthetic"]["garch_t_copula"]["status"] == "ok"
    assert report["reverse_return_panel"]["ridge"] == 0.0
    weights = portfolio_returns(names, panel, ())
    assert weights.shape == (100,)
    keys: set[str] = set()
    _keys(report, keys)
    assert keys.isdisjoint(FORBIDDEN_RESEARCH_METRIC_KEYS)


def test_cli_writes_markdown_and_html(tmp_path: Path) -> None:
    strategy = tmp_path / "book.yaml"
    strategy.write_text(
        yaml.safe_dump(
            {
                "name": "cli-book",
                "research_only": True,
                "positions": [{"factor": "us_equity", "weight": 1.0, "mapping": "return"}],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "report.md"
    runner = CliRunner()
    listed = runner.invoke(app, ["stress", "crises"])
    assert listed.exit_code == 0
    assert "crash_1987" in listed.stdout
    assert "hypothetical" in listed.stdout
    result = runner.invoke(
        app,
        [
            "stress",
            "report",
            "--strategy",
            str(strategy),
            "--out",
            str(out),
            "--format",
            "markdown",
            "--n-scenarios",
            "32",
            "--n-boot",
            "50",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "research_only=True" in result.stdout
    assert "live_trading=False" in result.stdout
    text = out.read_text(encoding="utf-8")
    assert "crash_1987" in text
    sidecar = out.with_suffix(".json")
    assert sidecar.is_file()
    from quant_fund.research.receipt_v2 import verify_receipt_file

    assert verify_receipt_file(sidecar)["valid"] is True
    html_path = tmp_path / "report.html"
    html_result = runner.invoke(
        app,
        [
            "stress",
            "report",
            "--strategy",
            str(strategy),
            "--out",
            str(html_path),
            "--format",
            "html",
            "--n-scenarios",
            "32",
            "--n-boot",
            "50",
        ],
    )
    assert html_result.exit_code == 0, html_result.output
    assert "<html" in html_path.read_text(encoding="utf-8")
    bad = runner.invoke(
        app,
        [
            "stress",
            "report",
            "--strategy",
            str(strategy),
            "--out",
            str(tmp_path / "nope.txt"),
            "--format",
            "pdf",
        ],
    )
    assert bad.exit_code != 0


def test_french_parser_stops_before_the_annual_block(tmp_path: Path) -> None:
    dates, values = parse_french_daily_factors(_FRENCH)
    assert dates[2] == "1987-10-19"
    assert values[2, 0] == pytest.approx(-0.1741)
    assert values.shape[1] == 4
    sliced_dates, sliced = slice_dates(dates, values, "2020-03-01", "2020-03-31")
    assert sliced_dates == ("2020-03-16", "2020-03-17")
    assert sliced.shape == (2, 4)
    with pytest.raises(ValueError):
        slice_dates(dates, values, "2020-03-16", "2020-03-16")
    with pytest.raises(ValueError):
        slice_dates(dates, values, "1900-01-01", "1900-01-02")
    with pytest.raises(ValueError):
        parse_french_daily_factors("no header here\n")

    class _Client:
        def get_bytes(self, url: str) -> bytes:
            assert url == FRENCH_DAILY_URL
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as archive:
                archive.writestr("F-F_Research_Data_Factors_daily.CSV", _FRENCH)
            return buf.getvalue()

    class _EmptyZip:
        def get_bytes(self, url: str) -> bytes:
            assert url == FRENCH_DAILY_URL
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as archive:
                archive.writestr("readme.txt", "no csv")
            return buf.getvalue()

    with pytest.raises(ValueError, match="did not contain a CSV"):
        fetch_french_daily_factors(tmp_path / "missing.csv", client=_EmptyZip())  # type: ignore[arg-type]

    dest = tmp_path / "nested" / "french.csv"
    fetch_french_daily_factors(dest, client=_Client())  # type: ignore[arg-type]
    text = dest.read_text(encoding="utf-8")
    assert text.startswith("date,mkt_rf,smb,hml,rf")
    assert "1987-10-19" in text
    assert "1927," not in text


def test_stress_modules_do_not_import_execution_or_paper() -> None:
    root = Path(__file__).resolve().parents[3] / "src" / "quant_fund" / "stress"
    banned = ("quant_fund.execution", "quant_fund.paper", "quant_fund.api")
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                assert not module.startswith(banned), f"{path.name} imports {module}"

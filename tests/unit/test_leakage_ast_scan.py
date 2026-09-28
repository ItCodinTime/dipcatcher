"""Per-rule unit tests: one positive catch + one clean negative each."""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_fund.leakage import scan_paths
from quant_fund.proofcore.contracts import LeakageReport


def _scan(tmp_path: Path, source: str, *, name: str = "mod.py") -> LeakageReport:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return scan_paths([path])


def _rule_ids(report: LeakageReport, severity: str | None = None) -> set[str]:
    return {f.rule_id for f in report.findings if severity is None or f.severity == severity}


LH001_POSITIVE = """
import polars as pl


def f(close: pl.Series) -> pl.Series:
    return close.shift(-1)
"""

LH001_NEGATIVE = """
import polars as pl


def f(close: pl.Series, feats: pl.Series) -> tuple[pl.Series, pl.Series]:
    return close.shift(1), feats.shift(-1)
"""

LH002_POSITIVE = """
def f(x):
    return x.rolling_mean(window_size=5, center=True)
"""

LH002_NEGATIVE = """
def f(x):
    return x.rolling_mean(window_size=5, center=False)
"""

LH003_POSITIVE = """
def train(features):
    scaler = make_scaler()
    scaler.fit(features)
    return scaler
"""

LH003_NEGATIVE = """
def train(features, folds):
    for train_idx, test_idx in folds:
        scaler = make_scaler()
        scaler.fit([features[i] for i in train_idx])
"""

LH003_DISTRIBUTION_NEGATIVE = """
from scipy import stats


def shape(sample):
    # Distribution MLE, not a train/test scaler.
    norm_params = stats.norm.fit(sample)
    log_params = stats.lognorm.fit(sample, floc=0)
    return norm_params, log_params
"""

LH004_POSITIVE = """
def fuse(grid, feats):
    return grid.join_asof(feats, on="event_time", by="security_id")
"""

LH004_NEGATIVE = """
def fuse(grid, feats):
    return grid.join_asof(feats, on="known_at", by="security_id")
"""

LH005_POSITIVE = """
universe = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META"]
"""

LH005_NEGATIVE = """
symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META"]
universe = ["SPY"]
"""

LH006_POSITIVE = """
def add(frame):
    delta_mid = frame["mid"].shift(-1) - frame["mid"]
    return delta_mid
"""

LH006_ALIAS_POSITIVE = """
import polars as pl


def add(frame):
    return frame.with_columns((pl.col("mid").shift(-1) - pl.col("mid")).alias("chg_mid"))
"""

LH006_NEGATIVE = """
def add(frame):
    fwd_delta_mid = frame["mid"].shift(-1) - frame["mid"]
    delta_mid = frame["mid"] - frame["mid"].shift(1)
    return fwd_delta_mid, delta_mid
"""

LH007_POSITIVE = """
def scorebook(returns, n, skew, kurt):
    sr = sharpe_ratio(returns)
    sharpe = float(sr["sharpe"])
    return probabilistic_sharpe(sharpe, 0.0, n, skew, kurt)
"""

LH007_POSITIVE_DIRECT = """
def scorebook(returns, n, skew, kurt):
    return min_track_record_length(sharpe_ratio(returns, periods_per_year=252.0), skew, kurt)
"""

LH007_NEGATIVE = """
def scorebook(returns, n, skew, kurt):
    sr = sharpe_ratio(returns, periods_per_year=1.0)
    psr = probabilistic_sharpe(sr, 0.0, n, skew, kurt)
    other = sharpe_ratio(returns, irregular=True)
    return psr, deflated_sharpe(other, n, skew, kurt, 1, 0.0)
"""

LH008_POSITIVE = """
def render() -> str:
    return "Sharpe ratio came in at 2.1 for the window"
"""

LH008_NEGATIVE_DOCSTRING = '''"""Sharpe 2.1 discussed in a docstring is documentation, not output."""


def render() -> str:
    return "proper scores: pinball 0.03 and crps 0.01"
'''

LH009_POSITIVE = """
import polars as pl


def load(path):
    return pl.read_parquet(path)
"""

LH010_POSITIVE = """
def clean(frame):
    frame = frame.sort("event_time")
    return frame.with_columns(frame["mid"].bfill())
"""

LH010_NEGATIVE = """
def clean(frame):
    return frame.with_columns(frame["mid"].bfill())
"""

LH012_POSITIVE = "def broken(:\n"


@pytest.mark.parametrize(
    "source,rule_id",
    [
        (LH001_POSITIVE, "LH001"),
        (LH002_POSITIVE, "LH002"),
        (LH003_POSITIVE, "LH003"),
        (LH004_POSITIVE, "LH004"),
        (LH005_POSITIVE, "LH005"),
        (LH006_POSITIVE, "LH006"),
        (LH006_ALIAS_POSITIVE, "LH006"),
        (LH007_POSITIVE, "LH007"),
        (LH007_POSITIVE_DIRECT, "LH007"),
        (LH008_POSITIVE, "LH008"),
        (LH009_POSITIVE, "LH009"),
        (LH010_POSITIVE, "LH010"),
    ],
)
def test_rule_positive_catch(tmp_path: Path, source: str, rule_id: str) -> None:
    report = _scan(tmp_path, source)
    assert rule_id in _rule_ids(report), f"{rule_id} did not fire: {report.findings}"


@pytest.mark.parametrize(
    "source,rule_id",
    [
        (LH001_NEGATIVE, "LH001"),
        (LH002_NEGATIVE, "LH002"),
        (LH003_NEGATIVE, "LH003"),
        (LH003_DISTRIBUTION_NEGATIVE, "LH003"),
        (LH004_NEGATIVE, "LH004"),
        (LH005_NEGATIVE, "LH005"),
        (LH006_NEGATIVE, "LH006"),
        (LH007_NEGATIVE, "LH007"),
        (LH008_NEGATIVE_DOCSTRING, "LH008"),
        (LH010_NEGATIVE, "LH010"),
    ],
)
def test_rule_clean_negative(tmp_path: Path, source: str, rule_id: str) -> None:
    report = _scan(tmp_path, source)
    assert rule_id not in _rule_ids(report), f"{rule_id} false positive: {report.findings}"


def test_lh009_is_warning_only_this_wave(tmp_path: Path) -> None:
    """Adjudicated: LH009 ships as WARNING until the call-site migration."""
    report = _scan(tmp_path, LH009_POSITIVE)
    findings = [f for f in report.findings if f.rule_id == "LH009"]
    assert findings and all(f.severity == "warning" for f in findings)
    assert report.errors == 0 and report.warnings == 1


def test_lh009_exempt_inside_data_layer(tmp_path: Path) -> None:
    pkg = tmp_path / "src" / "quant_fund" / "data"
    pkg.mkdir(parents=True)
    (pkg / "reader.py").write_text(LH009_POSITIVE, encoding="utf-8")
    report = scan_paths([pkg])
    assert "LH009" not in _rule_ids(report)


def test_lh012_parse_failure_is_warning_not_crash(tmp_path: Path) -> None:
    report = _scan(tmp_path, LH012_POSITIVE)
    findings = [f for f in report.findings if f.rule_id == "LH012"]
    assert len(findings) == 1 and findings[0].severity == "warning"


def test_lh012_respects_rule_filter(tmp_path: Path) -> None:
    path = tmp_path / "broken.py"
    path.write_text(LH012_POSITIVE, encoding="utf-8")
    assert scan_paths([path], rules={"LH001"}).findings == []


def test_lh001_detects_direct_subscript_shift(tmp_path: Path) -> None:
    report = _scan(tmp_path, 'def f(frame):\n    return frame["close"].shift(-1)\n')
    assert "LH001" in _rule_ids(report, "error")


def test_lh001_function_allowlist_still_flags_other_leaks_in_file(tmp_path: Path) -> None:
    """forward_close_return_labels is exempt; a sibling leak in the same file is not."""
    repo = Path(__file__).resolve().parents[2]
    rel = Path("src/quant_fund/microstructure/candle_book_features.py")
    source = (repo / rel).read_text(encoding="utf-8")
    label_line = next(
        i for i, line in enumerate(source.splitlines(), start=1) if ".shift(-1)" in line
    )
    leaked = source + ("\n\ndef leaked_close_feature(close):\n    return close.shift(-1)\n")
    path = tmp_path / rel
    path.parent.mkdir(parents=True)
    path.write_text(leaked, encoding="utf-8")
    report = scan_paths([path], rules={"LH001"})
    lh001 = [f for f in report.findings if f.rule_id == "LH001"]
    assert len(lh001) == 1
    assert lh001[0].line != label_line
    assert lh001[0].snippet == "return close.shift(-1)"


def test_lh001_function_allowlist_does_not_exempt_same_name_elsewhere(tmp_path: Path) -> None:
    source = "def forward_close_return_labels(close):\n    return close.shift(-1)\n"
    report = _scan(tmp_path, source, name="other_labels.py")
    assert "LH001" in _rule_ids(report, "error")


def test_lh001_function_allowlist_still_flags_nested_helper(tmp_path: Path) -> None:
    """A helper nested inside the label function is a different function and stays scanned."""
    rel = Path("src/quant_fund/microstructure/candle_book_features.py")
    source = (
        "def forward_close_return_labels(close):\n"
        "    def leaked_close_feature(series):\n"
        "        return series.shift(-1)\n"
        "    return leaked_close_feature(close)\n"
    )
    path = tmp_path / rel
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")
    report = scan_paths([path], rules={"LH001"})
    lh001 = [f for f in report.findings if f.rule_id == "LH001"]
    assert len(lh001) == 1
    assert lh001[0].snippet == "return series.shift(-1)"


@pytest.mark.parametrize("headline", ["Sharpe:2.1", "P&L=$4,200"])
def test_lh008_detects_compact_headline(tmp_path: Path, headline: str) -> None:
    report = _scan(tmp_path, f"title = {headline!r}\n")
    assert "LH008" in _rule_ids(report, "error")


def test_lh008_allowlist_does_not_exempt_new_literal(tmp_path: Path) -> None:
    path = tmp_path / "src/quant_fund/risk/gates.py"
    path.parent.mkdir(parents=True)
    path.write_text('note = "Sharpe 3.1 is a headline"\n', encoding="utf-8")
    report = scan_paths([path], rules={"LH008"})
    assert "LH008" in _rule_ids(report, "error")


def test_scan_paths_rejects_invalid_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scan target"):
        scan_paths([tmp_path / "missing.py"])
    text_file = tmp_path / "notes.txt"
    text_file.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="scan target"):
        scan_paths([text_file])


def test_lh011_layering_violation(tmp_path: Path) -> None:
    pkg = tmp_path / "src" / "quant_fund" / "leakage"
    pkg.mkdir(parents=True)
    (pkg / "evil.py").write_text(
        "from quant_fund.backtest.engine import run_backtest\n", encoding="utf-8"
    )
    report = scan_paths([pkg], rules={"LH011"})
    assert "LH011" in _rule_ids(report, "error")


def test_lh011_allows_whitelisted_and_lazy(tmp_path: Path) -> None:
    pkg = tmp_path / "src" / "quant_fund" / "leakage"
    pkg.mkdir(parents=True)
    (pkg / "ok.py").write_text(
        "from quant_fund.proofcore.contracts import LeakageReport\n"
        "from quant_fund.schemas.errors import PointInTimeError\n"
        "from quant_fund.leakage.rules import RULE_REGISTRY\n"
        "\n"
        "def f():\n"
        "    from quant_fund.pit import PitVault  # lazy whitelist\n"
        "    return PitVault\n",
        encoding="utf-8",
    )
    report = scan_paths([pkg], rules={"LH011"})
    assert "LH011" not in _rule_ids(report)


def test_lh011_rejects_non_whitelisted_lazy(tmp_path: Path) -> None:
    pkg = tmp_path / "src" / "quant_fund" / "leakage"
    pkg.mkdir(parents=True)
    (pkg / "evil_lazy.py").write_text(
        "def f():\n    from quant_fund.backtest import engine\n    return engine\n",
        encoding="utf-8",
    )
    report = scan_paths([pkg], rules={"LH011"})
    assert "LH011" in _rule_ids(report, "error")


def test_lh011_rejects_relative_import_into_unlisted_package(tmp_path: Path) -> None:
    pkg = tmp_path / "src" / "quant_fund" / "leakage"
    pkg.mkdir(parents=True)
    (pkg / "evil_relative.py").write_text("from ..backtest import engine\n", encoding="utf-8")
    report = scan_paths([pkg], rules={"LH011"})
    assert "LH011" in _rule_ids(report, "error")


def test_lh011_checks_every_absolute_import(tmp_path: Path) -> None:
    pkg = tmp_path / "src" / "quant_fund" / "leakage"
    pkg.mkdir(parents=True)
    (pkg / "evil_multi.py").write_text(
        "import quant_fund.backtest, quant_fund.proofcore\n", encoding="utf-8"
    )
    report = scan_paths([pkg], rules={"LH011"})
    assert "LH011" in _rule_ids(report, "error")


def test_scan_paths_rule_filter_and_unknown(tmp_path: Path) -> None:
    report = _scan(tmp_path, LH001_POSITIVE)
    assert "LH001" in _rule_ids(report)
    only_lh002 = scan_paths([tmp_path / "mod.py"], rules={"LH002"})
    assert only_lh002.findings == []
    with pytest.raises(ValueError, match="unknown leakage rule"):
        scan_paths([tmp_path / "mod.py"], rules={"LH999"})


def test_report_schema_and_counts(tmp_path: Path) -> None:
    report = _scan(tmp_path, LH001_POSITIVE + LH009_POSITIVE)
    assert report.schema_version == "proofcore/1"
    assert report.scanned_files == 1
    assert report.errors == sum(1 for f in report.findings if f.severity == "error")
    assert report.warnings == sum(1 for f in report.findings if f.severity == "warning")
    assert len(report.report_sha256) == 64
    for f in report.findings:
        assert f.line >= 1 and f.snippet

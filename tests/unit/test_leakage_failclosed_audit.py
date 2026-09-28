"""Fail-closed regression tests from the leakage-scanner audit.

Each test pairs a probe that was a CONFIRMED silent miss (or wrong exit /
wrong error type) at HEAD~ with the fail-closed behavior now enforced.
Inline sources in the `_scan` style of test_leakage_ast_scan.py — no fixture
files, so test_fixture_dir_complete stays pinned.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.leakage import scan_paths
from quant_fund.leakage.cli import leakage_app
from quant_fund.leakage.watchdog import LeakageError, LeakageWatchdog
from quant_fund.proofcore.contracts import DataAccessRecord, LeakageReport


def _scan(tmp_path: Path, source: str, *, name: str = "mod.py") -> LeakageReport:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return scan_paths([path])


def _scan_bytes(tmp_path: Path, data: bytes, *, name: str = "mod.py") -> LeakageReport:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return scan_paths([path])


def _rule_ids(report: LeakageReport, severity: str | None = None) -> set[str]:
    return {f.rule_id for f in report.findings if severity is None or f.severity == severity}


# ---------------------------------------------------------------- LH001 ---
# The shift-period check accepted only int literals and setdefault() kept the
# FIRST binding — every one of these evasions was a silent miss at HEAD.


@pytest.mark.parametrize(
    "body",
    [
        "close.shift(-1.0)",  # float literal period
        "close.shift(0 - 1)",  # constant arithmetic
        "k: int = -1\n    close.shift(k)",  # AnnAssign binding
        "a, k = 0, -1\n    close.shift(k)",  # tuple destructure
        "k = 1\n    k = -1\n    close.shift(k)",  # reassignment: live binding is -1
        "getattr(close, 'shift')(-1)",  # getattr dispatch
        "shift(close, -1)",  # bare functional call
        "close.fill_null(0).shift(-1)",  # chained receiver on price-like base
        "import numpy as np\n    np.log(close).shift(-1)",  # arg of a transform call
    ],
)
def test_lh001_evasion_shapes_flag(tmp_path: Path, body: str) -> None:
    source = f"def f(close):\n    {body}\n"
    assert "LH001" in _rule_ids(_scan(tmp_path, source), "error")


@pytest.mark.parametrize(
    "body",
    [
        "k = -1\n    k = 1\n    close.shift(k)",  # live binding is +1
        "close.shift(1)",
        "feats.shift(-1)",  # non-price receiver
    ],
)
def test_lh001_clean_controls_do_not_flag(tmp_path: Path, body: str) -> None:
    source = f"def f(close, feats=()):\n    {body}\n"
    assert "LH001" not in _rule_ids(_scan(tmp_path, source), "error")


# ---------------------------------------------------------------- LH002 ---


def test_lh002_center_kwarg_resolves_binding(tmp_path: Path) -> None:
    source = "C = True\n\ndef f(x):\n    return x.rolling_mean(5, center=C)\n"
    assert "LH002" in _rule_ids(_scan(tmp_path, source), "error")


def test_lh002_center_one_flags(tmp_path: Path) -> None:
    source = "def f(x):\n    return x.rolling_mean(5, center=1)\n"
    assert "LH002" in _rule_ids(_scan(tmp_path, source), "error")


def test_lh002_center_zero_clean(tmp_path: Path) -> None:
    source = "def f(x):\n    return x.rolling_mean(5, center=0)\n"
    assert "LH002" not in _rule_ids(_scan(tmp_path, source), "error")


# ---------------------------------------------------------------- LH003 ---


@pytest.mark.parametrize(
    "body",
    [
        "StandardScaler().fit(X)",  # anonymous ctor receiver
        "scaler.fit_transform(X)",  # fit_transform is a fit
        "scaler.partial_fit(X)",  # incremental fit is a fit
        "scaler.fit(X=X)",  # kwarg arg, still outside a fold
    ],
)
def test_lh003_unscoped_fits_flag(tmp_path: Path, body: str) -> None:
    source = f"def train(X):\n    {body}\n"
    assert "LH003" in _rule_ids(_scan(tmp_path, source), "error")


def test_lh003_scipy_from_import_stays_clean(tmp_path: Path) -> None:
    source = "from scipy.stats import norm\n\ndef mle(x):\n    return norm.fit(x)\n"
    assert "LH003" not in _rule_ids(_scan(tmp_path, source), "error")


def test_lh003_scaled_distribution_ctor_clean(tmp_path: Path) -> None:
    source = "def fit_dist(y):\n    return ScaledGaussianDistribution(1).fit(y)\n"
    assert "LH003" not in _rule_ids(_scan(tmp_path, source), "error")


def test_lh003_fold_func_kwarg_consumes_param(tmp_path: Path) -> None:
    source = "def per_fold(fold):\n    scaler.fit(X=fold)\n"
    assert "LH003" not in _rule_ids(_scan(tmp_path, source), "error")


def test_lh003_async_for_fold_loop_is_scope(tmp_path: Path) -> None:
    source = "async def train(folds):\n    async for fold in folds:\n        scaler.fit(fold)\n"
    assert "LH003" not in _rule_ids(_scan(tmp_path, source), "error")


# ---------------------------------------------------------------- LH004 ---


@pytest.mark.parametrize(
    "kw",
    [
        'right_on="event_time"',  # right-side key checked too
        'on="Event_Time"',  # case-insensitive
        'on="adjusted_event_time"',  # containment, not exact equality
    ],
)
def test_lh004_key_shapes_flag(tmp_path: Path, kw: str) -> None:
    source = f"def fuse(grid, feats):\n    return grid.join_asof(feats, {kw}, by='sid')\n"
    assert "LH004" in _rule_ids(_scan(tmp_path, source), "error")


def test_lh004_reassigned_keyvar_flags(tmp_path: Path) -> None:
    source = (
        "def fuse(grid, feats):\n"
        '    key = "asof"\n'
        '    key = "event_time"\n'
        "    return grid.join_asof(feats, on=key, by='sid')\n"
    )
    assert "LH004" in _rule_ids(_scan(tmp_path, source), "error")


def test_lh004_known_at_key_clean(tmp_path: Path) -> None:
    source = "def fuse(grid, feats):\n    return grid.join_asof(feats, on='known_at', by='sid')\n"
    assert "LH004" not in _rule_ids(_scan(tmp_path, source), "error")


# ---------------------------------------------------------------- LH005 ---


@pytest.mark.parametrize(
    "rhs",
    [
        '["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD"]',  # list literal
        '("BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD")',  # tuple
        '{"BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD"}',  # set literal
        '{"BTC-USD": 1, "ETH-USD": 1, "SOL-USD": 1, "XRP-USD": 1, "DOGE-USD": 1}',  # dict keys
        'list(["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD"])',  # wrapped
    ],
)
def test_lh005_universe_containers_flag(tmp_path: Path, rhs: str) -> None:
    assert "LH005" in _rule_ids(_scan(tmp_path, f"universe = {rhs}\n"), "error")


def test_lh005_annassign_universe_flags(tmp_path: Path) -> None:
    source = 'universe: list[str] = ["A", "BB", "CC", "DD", "EE"]\n'
    assert "LH005" in _rule_ids(_scan(tmp_path, source), "error")


def test_lh005_short_list_clean(tmp_path: Path) -> None:
    source = 'universe = ["A", "BB", "CC"]\n'
    assert "LH005" not in _rule_ids(_scan(tmp_path, source), "error")


def test_lh005_non_universe_name_clean(tmp_path: Path) -> None:
    source = 'watchlist_names = ["A", "BB", "CC", "DD", "EE"]\n'
    assert "LH005" not in _rule_ids(_scan(tmp_path, source), "error")


# ---------------------------------------------------------------- LH006 ---


def test_lh006_subscript_target_flags(tmp_path: Path) -> None:
    source = "def f(frame, mid):\n    frame['delta_mid'] = mid.shift(-1) - mid\n"
    assert "LH006" in _rule_ids(_scan(tmp_path, source), "error")


def test_lh006_ret_prefix_flags(tmp_path: Path) -> None:
    source = "def f(px):\n    ret_1d = px.shift(-1) - px\n"
    assert "LH006" in _rule_ids(_scan(tmp_path, source), "error")


def test_lh006_var_period_flags(tmp_path: Path) -> None:
    source = "def f(px):\n    k = -1\n    delta_px = px.shift(k) - px\n"
    assert "LH006" in _rule_ids(_scan(tmp_path, source), "error")


def test_lh006_annassign_flags(tmp_path: Path) -> None:
    source = "def f(px):\n    delta_px: float = px.shift(-1) - px\n    return delta_px\n"
    assert "LH006" in _rule_ids(_scan(tmp_path, source), "error")


def test_lh006_fwd_named_clean(tmp_path: Path) -> None:
    source = "def f(px):\n    fwd_delta_px = px.shift(-1) - px\n    return fwd_delta_px\n"
    assert "LH006" not in _rule_ids(_scan(tmp_path, source), "error")


# ---------------------------------------------------------------- LH007 ---


def test_lh007_sr_kwarg_flags(tmp_path: Path) -> None:
    source = (
        "def f(rets):\n"
        "    return probabilistic_sharpe(sr=sharpe_ratio(rets), sr_star=0.0,\n"
        "                                 n_obs=10, skew=0.0, kurtosis_raw=3.0)\n"
    )
    assert "LH007" in _rule_ids(_scan(tmp_path, source), "error")


def test_lh007_resolved_per_period_clean(tmp_path: Path) -> None:
    source = (
        "def f(rets):\n"
        "    ppy = 1.0\n"
        "    sr = sharpe_ratio(rets, periods_per_year=ppy)\n"
        "    return probabilistic_sharpe(sr, 0.0, 10, 0.0, 3.0)\n"
    )
    assert "LH007" not in _rule_ids(_scan(tmp_path, source), "error")


# ---------------------------------------------------------------- LH008 ---


def test_lh008_concat_via_module_assign_flags(tmp_path: Path) -> None:
    source = 'TMPL = "Sharpe: "\nlabel = TMPL + "2.1"\n'
    assert "LH008" in _rule_ids(_scan(tmp_path, source), "error")


def test_lh008_metric_name_key_clean(tmp_path: Path) -> None:
    source = 'stats = {"sharpe_p05": 1.0, "calmar_p50": 0.5}\n'
    assert "LH008" not in _rule_ids(_scan(tmp_path, source))


# ---------------------------------------------------------------- LH009 ---


def test_lh009_aliased_parquet_import_flags(tmp_path: Path) -> None:
    source = "from polars import read_parquet as rp\n\ndef load(p):\n    return rp(p)\n"
    assert "LH009" in _rule_ids(_scan(tmp_path, source))


def test_lh009_execute_sql_string_flags(tmp_path: Path) -> None:
    source = "def load(conn, p):\n    return conn.execute('select * from read_parquet(p)')\n"
    assert "LH009" in _rule_ids(_scan(tmp_path, source))


def test_lh009_eval_string_flags(tmp_path: Path) -> None:
    source = "def load(p):\n    return eval('pl.read_parquet(p)')\n"
    assert "LH009" in _rule_ids(_scan(tmp_path, source))


def test_lh009_unrelated_execute_clean(tmp_path: Path) -> None:
    source = "def run(conn):\n    return conn.execute('select 1')\n"
    assert "LH009" not in _rule_ids(_scan(tmp_path, source))


# ---------------------------------------------------------------- LH010 ---


def test_lh010_strategy_var_flags(tmp_path: Path) -> None:
    source = (
        "def fix(frame):\n"
        '    s = "backward"\n'
        '    return frame.with_columns(frame["event_time"]), frame.fill_null(strategy=s)\n'
    )
    assert "LH010" in _rule_ids(_scan(tmp_path, source))


def test_lh010_fillna_method_flags(tmp_path: Path) -> None:
    source = (
        "def fix(frame):\n"
        '    frame["event_time"] = frame["ts"]\n'
        '    return frame.fillna(method="bfill")\n'
    )
    assert "LH010" in _rule_ids(_scan(tmp_path, source))


def test_lh010_forward_fill_clean(tmp_path: Path) -> None:
    source = (
        "def fix(frame):\n"
        '    frame["event_time"] = frame["ts"]\n'
        '    return frame.fillna(method="ffill")\n'
    )
    assert "LH010" not in _rule_ids(_scan(tmp_path, source))


# ---------------------------------------------------------------- LH011 ---


def test_lh011_from_root_import_flags(tmp_path: Path) -> None:
    source = "from quant_fund import backtest\n"
    report = _scan(tmp_path, source, name="quant_fund/proof/evil.py")
    assert "LH011" in _rule_ids(report, "error")


def test_lh011_import_module_flags(tmp_path: Path) -> None:
    # module-level dynamic import: backtest is only *lazy*-allowed for proof
    source = "import importlib\n\nmod = importlib.import_module('quant_fund.backtest')\n"
    report = _scan(tmp_path, source, name="quant_fund/proof/evil.py")
    assert "LH011" in _rule_ids(report, "error")


def test_lh011_lazy_import_module_still_gated(tmp_path: Path) -> None:
    # inside a function: `reality` is in neither whitelist nor lazy whitelist
    source = (
        "import importlib\n\n"
        "def load():\n"
        "    return importlib.import_module('quant_fund.reality')\n"
    )
    report = _scan(tmp_path, source, name="quant_fund/proof/evil.py")
    assert "LH011" in _rule_ids(report, "error")


def test_lh011_lazy_whitelisted_dynamic_import_clean(tmp_path: Path) -> None:
    # backtest IS lazy-whitelisted for proof -> function-level is allowed
    source = (
        "import importlib\n\n"
        "def load():\n"
        "    return importlib.import_module('quant_fund.backtest')\n"
    )
    report = _scan(tmp_path, source, name="quant_fund/proof/evil.py")
    assert "LH011" not in _rule_ids(report, "error")


def test_lh011_whitelisted_import_clean(tmp_path: Path) -> None:
    source = "from quant_fund import schemas\n"
    report = _scan(tmp_path, source, name="quant_fund/proof/evil.py")
    assert "LH011" not in _rule_ids(report, "error")


def test_lh011_outside_proofcore_clean(tmp_path: Path) -> None:
    source = "from quant_fund import backtest\n"
    report = _scan(tmp_path, source, name="quant_fund/research/mod.py")
    assert "LH011" not in _rule_ids(report)


# ---------------------------------------------------------------- LH012 ---


def test_lh012_bom_file_still_scanned(tmp_path: Path) -> None:
    source = "def f(close):\n    return close.shift(-1)\n"
    report = _scan_bytes(tmp_path, source.encode("utf-8-sig"))
    ids = _rule_ids(report)
    assert "LH012" not in ids  # no longer degrades to unscanned
    assert "LH001" in ids


def test_lh012_unparseable_still_reports(tmp_path: Path) -> None:
    report = _scan(tmp_path, "def broken(:\n")
    assert "LH012" in _rule_ids(report)


# ---------------------------------------------------------------- LH013 ---


def test_lh013_fstring_no_spec_flags(tmp_path: Path) -> None:
    source = 'def report(sr):\n    return f"Sharpe {sr}"\n'
    assert "LH013" in _rule_ids(_scan(tmp_path, source))


def test_lh013_percent_template_flags(tmp_path: Path) -> None:
    source = 'def report(sr):\n    return "Sharpe %s" % sr\n'
    assert "LH013" in _rule_ids(_scan(tmp_path, source))


def test_lh013_str_format_flags(tmp_path: Path) -> None:
    source = 'def report(sr):\n    return "Sharpe {}".format(sr)\n'
    assert "LH013" in _rule_ids(_scan(tmp_path, source))


def test_lh013_concat_template_flags(tmp_path: Path) -> None:
    source = 'def report(sr):\n    return "Sharpe " + str(sr)\n'
    assert "LH013" in _rule_ids(_scan(tmp_path, source))


def test_lh013_plain_token_no_number_clean(tmp_path: Path) -> None:
    # A bare token mention with no number and no interpolation is not a
    # headline — nothing to warn on.
    source = 'def report():\n    return "Sharpe"\n'
    report = _scan(tmp_path, source)
    assert not _rule_ids(report)


# ---------------------------------------------------------------- LH014 ---


def test_lh014_dotted_helper_call_flags(tmp_path: Path) -> None:
    source = (
        "class Engine:\n"
        "    def helper(self, close):\n"
        "        return close.shift(-1)\n"
        "    def run(self, close):\n"
        "        return self.helper(close)\n"
    )
    assert "LH014" in _rule_ids(_scan(tmp_path, source))


# ------------------------------------------------------- cli exit codes ---

runner = CliRunner()


def test_cli_empty_dir_fails_closed(tmp_path: Path) -> None:
    result = runner.invoke(leakage_app, ["scan", "--paths", str(tmp_path)])
    assert result.exit_code == 2


def test_cli_empty_rule_set_fails_closed(tmp_path: Path) -> None:
    src = tmp_path / "m.py"
    src.write_text("x = 1\n", encoding="utf-8")
    result = runner.invoke(
        leakage_app, ["scan", "--paths", str(src), "--rules", ",", "--fail-on", "none"]
    )
    assert result.exit_code == 2


def test_cli_unparseable_file_fails_closed(tmp_path: Path) -> None:
    src = tmp_path / "m.py"
    src.write_text("def broken(:\n", encoding="utf-8")
    # even under --fail-on none, an unscanned file is a scan defect
    result = runner.invoke(leakage_app, ["scan", "--paths", str(src), "--fail-on", "none"])
    assert result.exit_code == 2
    assert "LH012" in result.output


def test_cli_clean_file_still_exit_0(tmp_path: Path) -> None:
    src = tmp_path / "m.py"
    src.write_text("x = 1\n", encoding="utf-8")
    result = runner.invoke(leakage_app, ["scan", "--paths", str(src)])
    assert result.exit_code == 0


# ------------------------------------------------------------- watchdog ---


def test_watchdog_non_string_watermark_fails_closed() -> None:
    # model_construct bypasses str coercion -> raw non-str reaches the parse;
    # must surface as LeakageError, not an uncaught TypeError.
    record = DataAccessRecord.model_construct(
        dataset="d",
        asof_utc="2024-01-01T00:00:00+00:00",
        params={"max_known_at": 123},  # type: ignore[dict-item]
        rows=0,
        content_sha256="a" * 64,
    )
    watchdog = LeakageWatchdog(strict=True)
    with pytest.raises(LeakageError):
        watchdog.observe(record, datetime.now(UTC))

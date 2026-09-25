"""Tests for the unified datasource layer (18 professional sources)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fx1.cli import app
from fx1.data import CorpusLedger
from fx1.data.sources.adapters import (
    AgentGwAdapter,
    CaixinAdapter,
    FinanceFetchAdapter,
    McpAdapter,
    XhcjAdapter,
    build_adapter,
)
from fx1.data.sources.base import (
    FetchRequest,
    FetchResult,
    RunOutcome,
    SourceKind,
    SourceStatus,
)
from fx1.data.sources.ingest import fetch_to_example, record_fetch_in_ledger
from fx1.data.sources.registry import (
    DEFAULT_PLUGIN_ROOT,
    REGISTRY,
    SourceSpec,
    get_spec,
    list_sources,
    routing_candidates,
)
from fx1.data.sources.router import (
    classify_market,
    classify_need,
    fetch_routed,
    route,
)

SYSTEM = "You are fx-1."
EXPECTED_SOURCES = {
    "wind", "ifind", "gildata", "sp_data", "sec_edgar", "yahoo_finance",
    "dongcai", "cls", "caixin", "binance_crypto", "imf", "world_bank",
    "igo_open_data", "xhcj", "finance_research", "tianyancha",
    "finance_fetch", "finenter",
}


@pytest.fixture(autouse=True)
def _stub_plugin_scripts(tmp_path, monkeypatch):
    """Host-independent scripts + credential names for grammar tests.

    The real plugin CLIs live under DEFAULT_PLUGIN_ROOT on the author's
    host; these tests exercise argv grammar and routing with injected
    runners, so materialize each spec's first script candidate and dummy
    credential *names* so the fail-closed preflight passes. Tests that set
    their own FX1_PLUGIN_ROOTS or delete credentials still override this.
    """
    root = tmp_path / "plugins"
    for spec in list_sources():
        if spec.kind is SourceKind.MCP or not spec.script_candidates:
            continue
        script = root / spec.script_candidates[0]
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("# test stub\n", encoding="utf-8")
    monkeypatch.setenv("FX1_PLUGIN_ROOTS", str(root))
    for name in ("KIMI_API_KEY", "AGENT_GW_TOKEN", "DATASOURCE_BASE_URL",
                 "DATASOURCE_API_KEY"):
        monkeypatch.setenv(name, "test-dummy")


def _ok_runner(text: str = "payload"):
    def runner(argv, *, timeout_s, extra_env=None):
        return RunOutcome(returncode=0, stdout=text, stderr="")

    return runner


def _fail_runner(code: int = 1, stderr: str = "boom"):
    def runner(argv, *, timeout_s, extra_env=None):
        return RunOutcome(returncode=code, stdout="", stderr=stderr)

    return runner


class _ArgvRecorder:
    def __init__(self, outcome: RunOutcome) -> None:
        self.outcome = outcome
        self.calls: list[list[str]] = []

    def __call__(self, argv, *, timeout_s, extra_env=None):
        self.calls.append(list(argv))
        return self.outcome


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
def test_registry_covers_all_expected_sources():
    assert set(REGISTRY) == EXPECTED_SOURCES
    assert len(list_sources()) == 18


def test_registry_specs_are_wellformed():
    for spec in list_sources():
        assert spec.owner_plugins, spec.name
        assert spec.markets and spec.assets, spec.name
        if spec.kind is SourceKind.MCP:
            assert spec.name == "finenter"
        else:
            assert spec.script_candidates, spec.name
        for cred in spec.credentials:
            assert cred.isupper()  # env var *names*, never values


def test_registry_scripts_resolve_on_this_host():
    if not Path(DEFAULT_PLUGIN_ROOT).exists():
        pytest.skip("plugin root not present on this host")
    for spec in list_sources():
        if spec.kind is SourceKind.MCP:
            continue
        assert build_adapter(spec).resolve_script() is not None, spec.name


def test_get_spec_unknown_raises():
    with pytest.raises(KeyError):
        get_spec("bloomberg")


# --------------------------------------------------------------------------
# Probe / fail-closed availability
# --------------------------------------------------------------------------
def test_probe_no_script_when_root_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("FX1_PLUGIN_ROOTS", str(tmp_path))
    probe = build_adapter(get_spec("wind")).probe()
    assert probe.status is SourceStatus.NO_SCRIPT
    assert "FX1_PLUGIN_ROOTS" in probe.detail


def test_probe_mcp_source_reports_mcp_required():
    probe = build_adapter(get_spec("finenter")).probe()
    assert probe.status is SourceStatus.MCP_REQUIRED


def test_probe_no_credentials_named_not_valued(monkeypatch, tmp_path):
    monkeypatch.setenv("FX1_PLUGIN_ROOTS", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))  # hide any runtime config file
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_GW_TOKEN", raising=False)
    script = tmp_path / "wind_tool.py"
    script.write_text("# stub", encoding="utf-8")
    spec = get_spec("wind").model_copy(
        update={"script_candidates": ["wind_tool.py"]}
    )
    probe = build_adapter(spec).probe()
    assert probe.status is SourceStatus.NO_CREDENTIALS
    assert "KIMI_API_KEY" in probe.detail


def test_probe_never_leaks_secret_values(monkeypatch, tmp_path):
    secret = "sk-test-SECRET-123"
    monkeypatch.setenv("KIMI_API_KEY", secret)
    monkeypatch.setenv("FX1_PLUGIN_ROOTS", str(tmp_path))
    (tmp_path / "wind_tool.py").write_text("# stub", encoding="utf-8")
    spec = get_spec("wind").model_copy(
        update={"script_candidates": ["wind_tool.py"]}
    )
    probe = build_adapter(spec).probe()
    assert probe.status is SourceStatus.READY
    assert secret not in probe.model_dump_json()


def test_agent_gw_config_file_counts_as_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_GW_TOKEN", raising=False)
    kimi_dir = tmp_path / ".kimi"
    kimi_dir.mkdir()
    (kimi_dir / "agent-gw.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    adapter = build_adapter(get_spec("wind"))
    assert adapter.credentials_present() is True


# --------------------------------------------------------------------------
# Adapter grammars
# --------------------------------------------------------------------------
def test_agent_gw_fetch_argv_and_success():
    recorder = _ArgvRecorder(RunOutcome(returncode=0, stdout="price: 42", stderr=""))
    adapter = AgentGwAdapter(get_spec("yahoo_finance"), runner=recorder)
    result = adapter.fetch(
        FetchRequest(api="get_stock_info", params={"ticker": "AAPL"},
                     as_of="2026-09-24")
    )
    assert result.ok
    assert result.payload_sha256 == FetchResult.hash_payload("price: 42")
    assert result.as_of == "2026-09-24"
    argv = recorder.calls[0]
    assert argv[2] == "call" and "--api-name" in argv and "--params-json" in argv
    params = json.loads(argv[argv.index("--params-json") + 1])
    assert params == {"ticker": "AAPL"}


def test_agent_gw_describe_verb_override_dongcai():
    recorder = _ArgvRecorder(RunOutcome(returncode=0, stdout="docs", stderr=""))
    AgentGwAdapter(get_spec("dongcai"), runner=recorder).describe()
    assert recorder.calls[0][-1] == "desc"


def test_caixin_describe_uses_search():
    recorder = _ArgvRecorder(RunOutcome(returncode=0, stdout="apis", stderr=""))
    adapter = CaixinAdapter(get_spec("caixin"), runner=recorder)
    adapter.describe()
    assert recorder.calls[0][2] == "search"


def test_xhcj_key_value_grammar():
    recorder = _ArgvRecorder(RunOutcome(returncode=0, stdout="news", stderr=""))
    adapter = XhcjAdapter(get_spec("xhcj"), runner=recorder)
    adapter.fetch(FetchRequest(api="get_AStock_News_byStockName",
                               params={"stockName": "京东方A"}))
    argv = recorder.calls[0]
    assert argv[2:] == ["call", "get_AStock_News_byStockName", "stockName=京东方A"]


def test_finance_fetch_envelope_ok_false_is_honest_failure():
    envelope = json.dumps({"ok": False, "error": "paid tier unconfigured"})
    recorder = _ArgvRecorder(RunOutcome(returncode=0, stdout=envelope, stderr=""))
    adapter = FinanceFetchAdapter(get_spec("finance_fetch"), runner=recorder)
    result = adapter.fetch(
        FetchRequest(api="us.income", params={"ticker": "AAPL", "periods": "5Q"})
    )
    assert not result.ok
    assert "honest degradation" in (result.error or "")
    argv = recorder.calls[0]
    assert "us.income" in argv and "AAPL" in argv
    assert "--periods" in argv and "5Q" in argv


def test_finance_fetch_envelope_ok_true_passes():
    envelope = json.dumps({"ok": True, "data": {"revenue": 1}})
    adapter = FinanceFetchAdapter(
        get_spec("finance_fetch"), runner=_ok_runner(envelope)
    )
    assert adapter.fetch(FetchRequest(api="quote", params={"ticker": "NVDA"})).ok


def test_mcp_adapter_fails_closed():
    adapter = McpAdapter(get_spec("finenter"))
    result = adapter.fetch(FetchRequest(api="roadshow_list"))
    assert not result.ok
    assert result.status is SourceStatus.MCP_REQUIRED
    assert "refusing to fabricate" in (result.error or "")


def test_fetch_failure_never_carries_payload():
    result = AgentGwAdapter(get_spec("imf"), runner=_fail_runner()).fetch(
        FetchRequest(api="weo_query", params={"indicator": "NGDP_RPCH"})
    )
    assert not result.ok
    assert result.text == "" and result.payload_sha256 == ""
    assert "boom" in (result.error or "")


def test_fetch_timeout_is_honest():
    def timeout_runner(argv, *, timeout_s, extra_env=None):
        return RunOutcome(returncode=124, stdout="", stderr="timeout",
                          timed_out=True)

    result = AgentGwAdapter(get_spec("imf"), runner=timeout_runner).fetch(
        FetchRequest(api="weo_query")
    )
    assert not result.ok and "timeout" in (result.error or "")


def test_empty_output_is_failure_not_empty_success():
    result = AgentGwAdapter(get_spec("imf"), runner=_ok_runner("   ")).fetch(
        FetchRequest(api="weo_query")
    )
    assert not result.ok
    assert "empty" in (result.error or "")


def test_custom_cli_guard_rejects_unknown_grammar():
    spec = SourceSpec(
        name="mystery", display="mystery", owner_plugins=["x"],
        kind=SourceKind.CUSTOM_CLI, script_candidates=["x.py"],
        markets=["cn"], assets=["news"], latency="news",
    )
    with pytest.raises(ValueError, match="mystery"):
        build_adapter(spec)


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("question", "need", "market"),
    [
        ("贵州茅台最新股价", "quote", "cn"),
        ("AAPL income statement 估值", "fundamentals", "cn"),
        ("BTC 实时价格", "crypto", "crypto"),
        ("中国最新CPI同比 宏观", "macro", "cn"),
        ("财联社最新快讯 新闻", "news", "cn"),
        ("券商研报 晨会观点", "research", "cn"),
        ("港股 .HK 行情", "quote", "hk"),
    ],
)
def test_classification(question, need, market):
    assert classify_need(question) == need
    assert classify_market(question) == market


def test_cn_quote_routes_specific_first():
    plan = route("贵州茅台最新股价", market="cn", need="quote")
    assert [c.source for c in plan.candidates] == [
        "ifind", "wind", "gildata", "dongcai"
    ]


def test_us_fundamentals_routes_sp_first_yahoo_last():
    names = routing_candidates("fundamentals", "us")
    assert names[0] == "sp_data"
    assert names.index("yahoo_finance") > names.index("sec_edgar")


def test_crypto_only_binance():
    assert routing_candidates("crypto", "crypto") == ["binance_crypto"]


def test_unknown_rule_refuses_to_guess():
    plan = route("q", market="global", need="quote")
    assert plan.candidates == []
    assert "refusing to guess" in plan.note


def test_fetch_routed_first_success_wins():
    plan = route("q", market="cn", need="quote")
    result = fetch_routed(
        plan,
        api_by_source={"ifind": "q", "wind": "q", "gildata": "q",
                       "dongcai": "q"},
        runner=_ok_runner("data"),
    )
    assert result.ok and result.text == "data"


def test_fetch_routed_aggregates_failures():
    plan = route("q", market="cn", need="quote")
    result = fetch_routed(
        plan,
        api_by_source={"ifind": "q", "wind": "q", "gildata": "q",
                       "dongcai": "q"},
        runner=_fail_runner(stderr="vendor down"),
    )
    assert not result.ok
    assert "vendor down" in (result.error or "")
    assert "ifind" in (result.error or "")


def test_fetch_routed_all_unavailable_is_described(tmp_path, monkeypatch):
    monkeypatch.setenv("FX1_PLUGIN_ROOTS", str(tmp_path))
    plan = route("q", market="cn", need="quote")
    result = fetch_routed(plan, api_by_source={"ifind": "q"})
    assert not result.ok
    assert "skipped" in (result.error or "")


# --------------------------------------------------------------------------
# Ingest gate
# --------------------------------------------------------------------------
def _good_result(**over) -> FetchResult:
    base = {
        "ok": True, "source": "wind", "api": "get_stock_price_indicators",
        "text": "600519.SH close 1680.0", "as_of": "2026-09-24",
    }
    base.update(over)
    result = FetchResult(**base)
    result.payload_sha256 = FetchResult.hash_payload(result.text)
    return result


def test_ingest_refuses_failed_fetch():
    decision = fetch_to_example(
        FetchResult.failure(source="wind", api="x", error="down",
                            status=SourceStatus.READY),
        SYSTEM,
    )
    assert not decision.accepted and "fetch failed" in decision.reason


def test_ingest_refuses_missing_as_of_for_market_data():
    decision = fetch_to_example(_good_result(as_of=None), SYSTEM)
    assert not decision.accepted
    assert "leakage guard" in decision.reason


def test_ingest_accepts_dated_payload_with_provenance():
    decision = fetch_to_example(_good_result(), SYSTEM)
    assert decision.accepted and not decision.negative
    example = decision.example
    assert example is not None
    assert example.receipt_sha256 == decision.payload_sha256
    assert example.source_path.startswith("datasource://wind/")
    assistant = example.messages[-1]["content"]
    assert "research/backtest evidence, not live performance" in assistant
    assert "as_of=2026-09-24" in assistant


def test_ingest_live_claim_becomes_negative():
    decision = fetch_to_example(
        _good_result(text="strategy returned 42% live_pnl_claim true"), SYSTEM
    )
    assert decision.accepted and decision.negative
    assert decision.example is not None and decision.example.negative


def test_ingest_research_source_does_not_require_as_of():
    result = _good_result(source="finance_research", api="search", as_of=None)
    decision = fetch_to_example(result, SYSTEM)
    assert decision.accepted


def test_ledger_chains_ingest_and_exclusion(tmp_path):
    ledger = CorpusLedger(tmp_path / "ledger.jsonl")
    record_fetch_in_ledger(ledger, fetch_to_example(_good_result(), SYSTEM))
    record_fetch_in_ledger(
        ledger, fetch_to_example(_good_result(as_of=None), SYSTEM)
    )
    export = ledger.audit_export()
    assert export["examples"] == 1
    assert export["exclusions"] == 1
    assert export["exclusion_rules"] == {"datasource_ingest_gate": 1}
    assert export["chain_valid"]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_cli_sources_list():
    result = CliRunner().invoke(app, ["sources", "list"])
    assert result.exit_code == 0
    for name in EXPECTED_SOURCES:
        assert name in result.output


def test_cli_sources_probe_json():
    result = CliRunner().invoke(app, ["sources", "probe", "wind"])
    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["probes"][0]["credentials"] == list(
        ("KIMI_API_KEY", "AGENT_GW_TOKEN")
    )


def test_cli_fetch_unknown_source_fails():
    result = CliRunner().invoke(
        app, ["sources", "fetch", "bloomberg", "--api", "x"]
    )
    assert result.exit_code != 0


def test_cli_route_outputs_plan():
    result = CliRunner().invoke(
        app, ["sources", "route", "--question", "贵州茅台股价", "--market", "cn"]
    )
    assert result.exit_code == 0
    plan = json.loads(result.output)
    assert plan["candidates"][0]["source"] == "ifind"


def test_cli_ingest_refused_without_as_of(tmp_path):
    # finance_fetch envelope failure guarantees an unsuccessful fetch offline
    result = CliRunner().invoke(
        app,
        ["corpus", "ingest-source", "wind", "--api", "get_stock_price_indicators",
         "--params-json", "{}", "--ledger-path", str(tmp_path / "l.jsonl"),
         "--out", str(tmp_path / "c.jsonl")],
    )
    assert result.exit_code == 1
    assert "ingest refused" in result.output


def test_cli_describe_unknown_fails():
    result = CliRunner().invoke(app, ["sources", "describe", "bloomberg"])
    assert result.exit_code != 0

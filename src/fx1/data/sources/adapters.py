"""Concrete adapters over each datasource's bundled CLI script.

Every adapter subclasses :class:`DataSourceAdapter` and inherits the
fail-closed preflight (script presence + credential gating). Adapters differ
only in argument grammar:

- ``AgentGwAdapter`` — the standard ``describe``/``call --api-name …
  --params-json …`` convention (Wind, iFinD, Gildata, S&P, EDGAR, Yahoo,
  Dongcai [``desc``], CLS, Caixin, Binance, IMF, World Bank, IGO, research,
  Tianyancha).
- ``XhcjAdapter`` — ``call <api> key=value`` grammar.
- ``FinanceFetchAdapter`` — positional ``<scenario> [ticker]`` with a unified
  JSON envelope; ``ok=false`` in the envelope is an honest failure.
- ``McpAdapter`` — Finenter/进门投研: no local CLI; probe reports
  ``MCP_REQUIRED`` and fetch fails closed with guidance.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

from fx1.data.sources.base import (
    DataSourceAdapter,
    FetchRequest,
    FetchResult,
    SourceKind,
    SourceStatus,
)
from fx1.data.sources.registry import SourceSpec


class AgentGwAdapter(DataSourceAdapter):
    """Standard agent-gw datasource script (describe/call grammar)."""

    def describe(self) -> FetchResult:
        preflight = self._preflight(self.spec.describe_verb)
        if preflight is not None:
            return preflight
        script = self.resolve_script()
        assert script is not None  # preflight guarantees
        request = FetchRequest(api=self.spec.describe_verb)
        return self._run(
            [sys.executable, str(script), self.spec.describe_verb],
            request=request,
            started=datetime.now(UTC),
        )

    def fetch(self, request: FetchRequest) -> FetchResult:
        preflight = self._preflight(request.api)
        if preflight is not None:
            return preflight
        script = self.resolve_script()
        assert script is not None
        argv = [
            sys.executable,
            str(script),
            "call",
            "--api-name",
            request.api,
            "--params-json",
            json.dumps(request.params, ensure_ascii=False),
        ]
        return self._run(argv, request=request, started=datetime.now(UTC))


class XhcjAdapter(DataSourceAdapter):
    """Xinhua Finance: ``xhcj_query.py call <api> key=value`` / ``desc``."""

    def describe(self) -> FetchResult:
        preflight = self._preflight("desc")
        if preflight is not None:
            return preflight
        script = self.resolve_script()
        assert script is not None
        return self._run(
            [sys.executable, str(script), "desc"],
            request=FetchRequest(api="desc"),
            started=datetime.now(UTC),
        )

    def fetch(self, request: FetchRequest) -> FetchResult:
        preflight = self._preflight(request.api)
        if preflight is not None:
            return preflight
        script = self.resolve_script()
        assert script is not None
        pairs = [f"{k}={v}" for k, v in request.params.items()]
        argv = [sys.executable, str(script), "call", request.api, *pairs]
        return self._run(argv, request=request, started=datetime.now(UTC))


class FinanceFetchAdapter(DataSourceAdapter):
    """institutional-finance-kit scenario router.

    ``finance_fetch.py <scenario> [ticker] [flags]`` prints a unified JSON
    envelope; ``ok=false`` is an *honest* degradation (e.g. paid tier
    unconfigured), surfaced verbatim — never smoothed over.
    """

    def describe(self) -> FetchResult:
        preflight = self._preflight("list-scenarios")
        if preflight is not None:
            return preflight
        script = self.resolve_script()
        assert script is not None
        return self._run(
            [sys.executable, str(script), "--list-scenarios"],
            request=FetchRequest(api="list-scenarios"),
            started=datetime.now(UTC),
        )

    def fetch(self, request: FetchRequest) -> FetchResult:
        preflight = self._preflight(request.api)
        if preflight is not None:
            return preflight
        script = self.resolve_script()
        assert script is not None
        argv = [sys.executable, str(script), request.api]
        ticker = request.params.get("ticker")
        if ticker:
            argv.append(str(ticker))
        for key, value in request.params.items():
            if key == "ticker":
                continue
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    argv.append(flag)
            else:
                argv.extend([flag, str(value)])
        result = self._run(argv, request=request, started=datetime.now(UTC))
        if not result.ok:
            return result
        # Envelope discipline: ok=false is a described failure, not data.
        try:
            envelope = json.loads(result.text)
        except json.JSONDecodeError:
            return result
        if isinstance(envelope, dict) and envelope.get("ok") is False:
            reason = (
                envelope.get("error")
                or envelope.get("message")
                or ("scenario chain returned ok=false")
            )
            return FetchResult.failure(
                source=self.spec.name,
                api=request.api,
                error=f"honest degradation: {reason}",
                status=result.status,
            )
        return result


class McpAdapter(DataSourceAdapter):
    """MCP-only source (Finenter). No offline CLI — fails closed."""

    def describe(self) -> FetchResult:
        return FetchResult.failure(
            source=self.spec.name,
            api="describe",
            error="MCP connector: capabilities are exposed through the host "
            "MCP runtime (comein-agent), not an offline CLI.",
            status=SourceStatus.MCP_REQUIRED,
        )

    def fetch(self, request: FetchRequest) -> FetchResult:
        return FetchResult.failure(
            source=self.spec.name,
            api=request.api,
            error="MCP connector: fetch must run inside the host MCP runtime; "
            "refusing to fabricate an offline result.",
            status=SourceStatus.MCP_REQUIRED,
        )


# Caixin shares the agent-gw grammar (call --api-name …) but its discovery
# verb is ``search``; modelled explicitly so routing stays declarative.
class CaixinAdapter(AgentGwAdapter):
    """Caixin: standard call grammar, ``search``-based discovery."""

    def describe(self) -> FetchResult:
        preflight = self._preflight("search")
        if preflight is not None:
            return preflight
        script = self.resolve_script()
        assert script is not None
        return self._run(
            [sys.executable, str(script), "search", "数据"],
            request=FetchRequest(api="search"),
            started=datetime.now(UTC),
        )


_KIND_TO_ADAPTER: dict[SourceKind, type[DataSourceAdapter]] = {
    SourceKind.AGENT_GW: AgentGwAdapter,
    SourceKind.CUSTOM_CLI: XhcjAdapter,
    SourceKind.SCENARIO_ROUTER: FinanceFetchAdapter,
    SourceKind.MCP: McpAdapter,
}


def build_adapter(spec: SourceSpec, runner: object | None = None) -> DataSourceAdapter:
    """Instantiate the adapter for *spec* (runner injectable for tests)."""
    if spec.name == "caixin":
        return CaixinAdapter(spec, runner=runner)  # type: ignore[arg-type]
    if spec.kind is SourceKind.CUSTOM_CLI and spec.name != "xhcj":
        raise ValueError(f"no adapter grammar registered for custom CLI {spec.name!r}")
    cls = _KIND_TO_ADAPTER[spec.kind]
    return cls(spec, runner=runner)  # type: ignore[arg-type]

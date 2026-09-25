"""Unified professional datasource layer for dipcatcher/fx-1.

Every installed finance plugin is a first-class, *honesty-gated* data source:

- Wind, iFinD, Gildata, S&P CapIQ, SEC EDGAR, Yahoo Finance, Dongcai (妙想),
  CLS (财联社), Caixin, Binance, IMF, World Bank, IGO/FRED, Xinhua Finance
  (XHCJ), finance-research, Tianyancha, the finance-fetch scenario router,
  and Finenter (进门投研, MCP).
- Sources are reached through their bundled CLI scripts (agent-gw gateway or
  local), never through model memory. A fetch that fails is an honest
  failure — adapters never synthesize data.
- Credentials are environment-only and are never printed, logged, or passed
  on any command line.
- Every successful fetch yields an immutable, hashed :class:`FetchResult`
  that can be chained into the corpus ledger for provenance.
"""

from fx1.data.sources.adapters import (
    AgentGwAdapter,
    CaixinAdapter,
    DataSourceAdapter,
    FinanceFetchAdapter,
    McpAdapter,
    XhcjAdapter,
    build_adapter,
)
from fx1.data.sources.base import (
    FetchRequest,
    FetchResult,
    LatencyClass,
    SourceKind,
    SourceProbe,
    SourceStatus,
)
from fx1.data.sources.ingest import (
    IngestDecision,
    fetch_to_example,
    record_fetch_in_ledger,
)
from fx1.data.sources.registry import REGISTRY, SourceSpec, get_spec, list_sources
from fx1.data.sources.router import (
    RoutingPlan,
    classify_need,
    fetch_routed,
    route,
)

__all__ = [
    "AgentGwAdapter",
    "CaixinAdapter",
    "DataSourceAdapter",
    "FetchRequest",
    "FetchResult",
    "FinanceFetchAdapter",
    "IngestDecision",
    "LatencyClass",
    "McpAdapter",
    "REGISTRY",
    "RoutingPlan",
    "SourceKind",
    "SourceProbe",
    "SourceSpec",
    "SourceStatus",
    "XhcjAdapter",
    "build_adapter",
    "classify_need",
    "fetch_routed",
    "fetch_to_example",
    "get_spec",
    "list_sources",
    "record_fetch_in_ledger",
    "route",
]

"""Registry of every professional datasource reachable from this repo.

One :class:`SourceSpec` per *underlying* datasource. Several installed
plugins expose the same source (e.g. Wind via wind-allskill,
financial-market-terminal and the xtt suites) — the registry deduplicates
them and lists every owning plugin, with script candidates tried in order.

Routing defaults follow the lab's market discipline:

- China A/H shares → iFinD, Wind, Gildata (specific beats general)
- US-listed (incl. ADRs) → S&P CapIQ → Gildata → SEC EDGAR → Yahoo
- Macro → IMF / World Bank / IGO+FRED / Wind EDB / Dongcai / Caixin
- News → CLS (fastest CN tape) / XHCJ / Dongcai / finance-research
- Crypto → Binance only
- Enterprise/credit → Tianyancha / Caixin
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from fx1.data.sources.base import LatencyClass, SourceKind

DEFAULT_PLUGIN_ROOT = "/app/.agents/plugins"
ENV_PLUGIN_ROOTS = "FX1_PLUGIN_ROOTS"

# Shared credential variable names (values never touch this process's logs).
AGENT_GW_CREDS = ("KIMI_API_KEY", "AGENT_GW_TOKEN")


class SourceSpec(BaseModel):
    """Static description of one datasource and how to reach it."""

    name: str
    display: str
    owner_plugins: list[str]
    kind: SourceKind
    script_candidates: list[str] = Field(
        description="Paths relative to each plugin root, tried in order"
    )
    markets: list[str] = Field(description="cn | hk | us | global | crypto")
    assets: list[str] = Field(
        description="equity | fund | bond | index | macro | news | filings | "
        "research | enterprise | crypto | sentiment | industry_chain"
    )
    latency: LatencyClass
    credentials: tuple[str, ...] = Field(
        default=(),
        description="Env var *names* the source reads; empty = auto-resolved",
    )
    requires_as_of: bool = Field(
        default=True,
        description="Time-stamped market data must carry as_of to enter the "
        "training corpus (leakage guard)",
    )
    describe_verb: str = "describe"
    notes: str = ""

    def plugin_roots(self) -> list[str]:
        raw = os.environ.get(ENV_PLUGIN_ROOTS, "")
        roots = [r for r in raw.split(os.pathsep) if r]
        return roots or [DEFAULT_PLUGIN_ROOT]


def _spec(
    name: str,
    display: str,
    owners: list[str],
    kind: SourceKind,
    scripts: list[str],
    markets: list[str],
    assets: list[str],
    latency: LatencyClass,
    credentials: tuple[str, ...] = AGENT_GW_CREDS,
    requires_as_of: bool = True,
    describe_verb: str = "describe",
    notes: str = "",
) -> SourceSpec:
    return SourceSpec(
        name=name,
        display=display,
        owner_plugins=owners,
        kind=kind,
        script_candidates=scripts,
        markets=markets,
        assets=assets,
        latency=latency,
        credentials=credentials,
        requires_as_of=requires_as_of,
        describe_verb=describe_verb,
        notes=notes,
    )


_SPECS: list[SourceSpec] = [
    _spec(
        "wind",
        "Wind 万得",
        ["wind-allskill", "financial-market-terminal", "xtt-public-markets-investing"],
        SourceKind.AGENT_GW,
        [
            "wind-allskill/skills/wind-mcp-skill/scripts/wind_tool.py",
            "financial-market-terminal/skills/mkt-datasource-wind/scripts/wind_tool.py",
        ],
        ["cn", "hk", "us"],
        ["equity", "fund", "bond", "index", "macro", "news", "filings"],
        LatencyClass.DAILY,
        notes="CN markets deepest coverage: quotes, K-line, funds, bonds, "
        "indices, EDB macro/industry, filings & news. Single-symbol per call.",
    ),
    _spec(
        "ifind",
        "iFinD 同花顺",
        ["financial-market-terminal", "xtt-public-markets-investing",
         "xtt-corporate-finance-accounting",
         "xtt-investment-banking-private-equity", "institutional-finance-kit"],
        SourceKind.AGENT_GW,
        [
            "financial-market-terminal/skills/mkt-datasource-ifind/scripts/ifind_tool.py",
            "xtt-public-markets-investing/skills/datasource-router/subskills/"
            "datasource-ifind/scripts/ifind_tool.py",
        ],
        ["cn", "hk"],
        ["equity", "fund", "index", "macro"],
        LatencyClass.DAILY,
        notes="CN microstructure: capital flow, shareholders, valuation "
        "percentile, broker targets, factor screening.",
    ),
    _spec(
        "gildata",
        "Gildata 恒生聚源",
        ["gildata-aifinmarket", "financial-market-terminal",
         "xtt-public-markets-investing"],
        SourceKind.AGENT_GW,
        [
            "gildata-aifinmarket/scripts/gildata_tool.py",
            "financial-market-terminal/skills/mkt-datasource-gildata/scripts/gildata_tool.py",
        ],
        ["cn", "hk"],
        ["equity", "fund", "index", "macro", "news", "filings", "research"],
        LatencyClass.DAILY,
        notes="CN A/H fundamentals, funds, research, screening, announcements.",
    ),
    _spec(
        "sp_data",
        "S&P Global Market Intelligence",
        ["sp_data", "institutional-finance-kit", "xtt-public-markets-investing"],
        SourceKind.AGENT_GW,
        ["sp_data/scripts/sp_data_tool.py"],
        ["us", "global"],
        ["equity", "research", "filings"],
        LatencyClass.FILINGS,
        notes="US/global standardized fundamentals, consensus estimates, "
        "transcripts, ownership incl. 13F, capital structure, M&A.",
    ),
    _spec(
        "sec_edgar",
        "SEC EDGAR",
        ["sec_edgar", "xtt-public-markets-investing"],
        SourceKind.AGENT_GW,
        ["sec_edgar/scripts/sec_edgar_tool.py"],
        ["us"],
        ["filings", "equity"],
        LatencyClass.FILINGS,
        credentials=(),
        notes="Official US filings, XBRL facts, insider trades, institutional "
        "holdings. Free official source — no credential required.",
    ),
    _spec(
        "yahoo_finance",
        "Yahoo Finance",
        ["yahoo_finance", "financial-market-terminal",
         "xtt-public-markets-investing"],
        SourceKind.AGENT_GW,
        ["yahoo_finance/scripts/yahoo_finance_tool.py"],
        ["us", "global", "hk"],
        ["equity", "index"],
        LatencyClass.DAILY,
        notes="Lightweight global quotes/profiles/fundamentals. Fallback for "
        "US only — S&P/Gildata/EDGAR outrank it.",
    ),
    _spec(
        "dongcai",
        "东方财富妙想",
        ["dongcai-mx-data", "financial-market-terminal"],
        SourceKind.AGENT_GW,
        [
            "dongcai-mx-data/scripts/dongcai_cli.py",
            "financial-market-terminal/skills/mkt-datasource-dongcai/scripts/dongcai_cli.py",
        ],
        ["cn", "hk", "us"],
        ["equity", "fund", "bond", "index", "macro", "news", "research",
         "sentiment"],
        LatencyClass.DAILY,
        describe_verb="desc",
        notes="Cross-market quotes/financials/macro/screener/news plus stock "
        "& fund diagnosis and hotspot discovery.",
    ),
    _spec(
        "cls",
        "财联社 CLS",
        ["cls-news", "financial-market-terminal"],
        SourceKind.AGENT_GW,
        [
            "financial-market-terminal/skills/mkt-datasource-cls/scripts/cls_tool.py",
            "xtt-public-markets-investing/skills/datasource-router/subskills/"
            "datasource-cls/scripts/cls_tool.py",
        ],
        ["cn"],
        ["news"],
        LatencyClass.REALTIME,
        credentials=(),
        notes="Fastest CN real-time telegraph stream + 12-column depth news. "
        "Credentials auto-resolved by runtime.",
    ),
    _spec(
        "caixin",
        "财新数据",
        ["caixin-data-agent", "financial-market-terminal",
         "xtt-public-markets-investing"],
        SourceKind.AGENT_GW,
        [
            "caixin-data-agent/scripts/caixin_tool.py",
            "financial-market-terminal/skills/mkt-datasource-caixin/scripts/caixin_tool.py",
        ],
        ["cn"],
        ["bond", "fund", "macro", "enterprise", "sentiment", "industry_chain",
         "equity", "research"],
        LatencyClass.DAILY,
        notes="CN bonds/funds/futures/macro/enterprise-integrity/sentiment/"
        "industry chains. APIs addressed by full Chinese name.",
    ),
    _spec(
        "binance_crypto",
        "Binance Crypto",
        ["binance_crypto", "xtt-public-markets-investing"],
        SourceKind.AGENT_GW,
        [
            "binance_crypto/scripts/binance_crypto_tool.py",
            "xtt-public-markets-investing/skills/datasource-router/subskills/"
            "datasource-binance/scripts/binance_crypto_tool.py",
        ],
        ["crypto"],
        ["crypto"],
        LatencyClass.REALTIME,
        notes="Crypto spot prices, K-lines, 24h stats, volume. The only "
        "sanctioned crypto source.",
    ),
    _spec(
        "imf",
        "IMF (WEO/COFER)",
        ["imf", "xtt-public-markets-investing"],
        SourceKind.AGENT_GW,
        ["imf/scripts/imf_tool.py"],
        ["global"],
        ["macro"],
        LatencyClass.MACRO,
        notes="WEO growth/inflation/debt/unemployment/trade across 190+ "
        "economies; COFER reserve currency shares.",
    ),
    _spec(
        "world_bank",
        "World Bank Open Data",
        ["world_bank_open_data", "xtt-public-markets-investing"],
        SourceKind.AGENT_GW,
        ["world_bank_open_data/scripts/world_bank_open_data_tool.py"],
        ["global"],
        ["macro"],
        LatencyClass.MACRO,
        credentials=(),
        notes="29k+ development indicators, 1960→present. Free official.",
    ),
    _spec(
        "igo_open_data",
        "IGO Open Data + FRED",
        ["igo_open_data", "financial-market-terminal",
         "xtt-public-markets-investing"],
        SourceKind.AGENT_GW,
        ["igo_open_data/scripts/igo_open_data_tool.py"],
        ["global"],
        ["macro"],
        LatencyClass.MACRO,
        notes="WHO/Eurostat/ECB/UNICEF/OECD/FAO/UNSD official statistics plus "
        "FRED macro time series.",
    ),
    _spec(
        "xhcj",
        "新华财经 Xinhua Finance",
        ["financial-market-terminal", "xtt-public-markets-investing"],
        SourceKind.CUSTOM_CLI,
        [
            "financial-market-terminal/skills/mkt-datasource-xhcj/scripts/xhcj_query.py",
            "xtt-public-markets-investing/skills/datasource-router/subskills/"
            "datasource-xhcj/scripts/xhcj_query.py",
        ],
        ["cn"],
        ["news", "filings"],
        LatencyClass.NEWS,
        credentials=(),
        notes="CN financial news, sector news, announcement keyword search, "
        "policy documents. Grammar: call <api> key=value.",
    ),
    _spec(
        "finance_research",
        "Finance Research (sell-side)",
        ["financial-market-terminal", "xtt-public-markets-investing",
         "institutional-finance-kit"],
        SourceKind.AGENT_GW,
        [
            "financial-market-terminal/skills/mkt-datasource-finance-research/"
            "scripts/finance_research_tool.py",
            "xtt-public-markets-investing/skills/datasource-router/subskills/"
            "datasource-finance-research/scripts/finance_research_tool.py",
        ],
        ["cn", "hk", "us", "global"],
        ["research"],
        LatencyClass.RESEARCH,
        requires_as_of=False,
        notes="Sell-side analyst reports, earnings reviews, morning-meeting "
        "notes. First source for analyst opinion.",
    ),
    _spec(
        "tianyancha",
        "天眼查",
        ["xtt-public-markets-investing", "xtt-corporate-finance-accounting",
         "xtt-investment-banking-private-equity"],
        SourceKind.AGENT_GW,
        [
            "xtt-public-markets-investing/skills/datasource-router/subskills/"
            "datasource-tianyancha/scripts/tianyancha_tool.py",
        ],
        ["cn"],
        ["enterprise"],
        LatencyClass.ENTERPRISE,
        notes="CN enterprise registry: corporate structure, risk, ownership.",
    ),
    _spec(
        "finance_fetch",
        "finance-fetch scenario router",
        ["institutional-finance-kit"],
        SourceKind.SCENARIO_ROUTER,
        ["institutional-finance-kit/skills/finance-data/scripts/finance_fetch.py"],
        ["cn", "hk", "us"],
        ["equity", "filings", "research", "news"],
        LatencyClass.FILINGS,
        credentials=("DATASOURCE_BASE_URL", "DATASOURCE_API_KEY") + AGENT_GW_CREDS,
        notes="Scenario + ticker → unified JSON envelope with built-in "
        "free↔paid degradation chains (three statements, consensus, peers, "
        "events, segments, K-line, quotes). Paid tier skips honestly when "
        "unconfigured.",
    ),
    _spec(
        "finenter",
        "进门投研 Finenter",
        ["comein-agent"],
        SourceKind.MCP,
        [],
        ["cn", "hk"],
        ["research", "news"],
        LatencyClass.RESEARCH,
        requires_as_of=False,
        notes="Roadshows, institutional research, quant screening via MCP "
        "connector. Offline probes report MCP_REQUIRED — never faked.",
    ),
]

REGISTRY: dict[str, SourceSpec] = {spec.name: spec for spec in _SPECS}


def get_spec(name: str) -> SourceSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown datasource {name!r}; known: {sorted(REGISTRY)}"
        ) from None


def list_sources() -> list[SourceSpec]:
    return [REGISTRY[name] for name in sorted(REGISTRY)]


# --------------------------------------------------------------------------
# Routing tables: (need, market) → ordered candidate source names.
# Order encodes the lab discipline: most specific/authoritative first.
# --------------------------------------------------------------------------
_ANY = ("cn", "hk", "us", "global", "crypto")

ROUTING_RULES: list[tuple[str, tuple[str, ...], list[str]]] = [
    # equity quotes / fundamentals / screening
    ("quote", ("cn",), ["ifind", "wind", "gildata", "dongcai"]),
    ("quote", ("hk",), ["wind", "ifind", "gildata", "dongcai"]),
    ("quote", ("us",), ["sp_data", "yahoo_finance", "dongcai"]),
    ("fundamentals", ("cn",), ["ifind", "gildata", "wind", "dongcai",
                               "finance_fetch"]),
    ("fundamentals", ("hk",), ["gildata", "wind", "ifind", "finance_fetch"]),
    ("fundamentals", ("us",), ["sp_data", "gildata", "sec_edgar",
                               "yahoo_finance", "finance_fetch"]),
    ("screen", ("cn", "hk"), ["gildata", "ifind", "wind", "dongcai"]),
    ("screen", ("us",), ["sp_data", "dongcai"]),
    # funds / bonds / indices
    ("fund", ("cn",), ["wind", "gildata", "dongcai", "caixin"]),
    ("bond", ("cn",), ["caixin", "wind", "dongcai"]),
    ("index", ("cn", "hk"), ["wind", "dongcai", "ifind"]),
    # filings & announcements
    ("filings", ("us",), ["sec_edgar", "sp_data", "finance_fetch"]),
    ("filings", ("cn", "hk"), ["wind", "gildata", "dongcai", "caixin",
                               "xhcj"]),
    # news & sentiment
    ("news", ("cn",), ["cls", "xhcj", "dongcai", "wind", "caixin"]),
    ("news", ("hk",), ["dongcai", "gildata", "wind"]),
    ("news", ("us", "global"), ["yahoo_finance", "finance_research"]),
    ("sentiment", ("cn",), ["caixin", "dongcai"]),
    # research opinion
    ("research", _ANY, ["finance_research", "gildata", "dongcai",
                        "finenter"]),
    # macro
    ("macro", _ANY, ["imf", "world_bank", "igo_open_data", "wind",
                     "dongcai", "caixin"]),
    # crypto
    ("crypto", ("crypto",), ["binance_crypto"]),
    # enterprise / credit
    ("enterprise", ("cn",), ["tianyancha", "caixin"]),
    # industry chains
    ("industry_chain", ("cn",), ["caixin", "gildata", "dongcai"]),
]


def routing_candidates(need: str, market: str) -> list[str]:
    """Ordered candidate source names for a (need, market) pair."""
    for rule_need, markets, sources in ROUTING_RULES:
        if rule_need == need and market in markets:
            return list(sources)
    return []


def roots_status() -> dict[str, object]:
    """Where adapters look for plugin scripts (audit-friendly)."""
    roots = REGISTRY["wind"].plugin_roots()
    return {
        "env_override": os.environ.get(ENV_PLUGIN_ROOTS) or None,
        "roots": roots,
        "existing": [r for r in roots if Path(r).exists()],
    }

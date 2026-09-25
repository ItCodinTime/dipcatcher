"""Routing: map a research question to the right datasource, honestly.

``route`` classifies the need and market, looks up the ordered candidate
list from the registry's routing rules, and probes each candidate.
``fetch_routed`` walks the plan in order: unavailable sources are skipped
*with their probe reason recorded*, the first genuinely successful fetch
wins, and if every candidate fails the aggregated result is an honest
failure naming each attempted source and why. No path ever fabricates data.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from fx1.data.sources.adapters import build_adapter
from fx1.data.sources.base import (
    FetchRequest,
    FetchResult,
    SourceProbe,
    SourceStatus,
)
from fx1.data.sources.registry import get_spec, routing_candidates

READY_STATES = (SourceStatus.READY, SourceStatus.READY_UNVERIFIED)

_NEED_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("crypto", ("crypto", "btc", "eth", "比特币", "加密", "币安", "binance")),
    ("sentiment", ("舆情", "sentiment", "负面新闻")),
    ("industry_chain", ("产业链", "industry chain", "供应链图谱")),
    ("enterprise", ("工商", "企业信息", "失信", "registry", "天眼查",
                    "enterprise")),
    ("research", ("研报", "research report", "analyst", "晨会", "路演",
                  "roadshow", "券商观点")),
    ("filings", ("公告", "年报", "季报", "招股", "10-k", "10-q", "8-k",
                 "filing", "披露", "announcement")),
    ("bond", ("债券", "bond", "转债", "国债", "信用债")),
    ("fund", ("基金", "fund", "etf", "lof", "公募")),
    ("index", ("指数", "板块", "index", "sector")),
    ("macro", ("宏观", "cpi", "ppi", "gdp", "pmi", "社融", "利率",
               "macro", "inflation", "unemployment", "fred")),
    ("news", ("新闻", "快讯", "电报", "news", "要闻", "热点")),
    ("screen", ("筛选", "选股", "screen", "screener", "条件")),
    ("fundamentals", ("财务", "利润表", "资产负债", "现金流", "fundamental",
                      "income statement", "balance sheet", "估值", "财务指标")),
    ("quote", ("行情", "价格", "quote", "price", "k线", "kline", "股价",
               "市值")),
]

_MARKET_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("crypto", ("btc", "eth", "bnb", "crypto", "加密", "比特币", "usdt")),
    ("cn", ("a股", "沪深", "上证", "深证", "科创", "创业板", "北交所",
            ".sh", ".sz", ".bj", "中国", "cn")),
    ("hk", ("港股", ".hk", "恒生", "hong kong", "hk")),
    ("us", ("美股", "纳斯达克", "纽交所", "us stock", ".o", ".n", "adr",
            "标普", "sp500")),
]


def classify_need(question: str) -> str:
    """Best-effort need classification from a natural-language question."""
    lowered = question.lower()
    for need, keywords in _NEED_KEYWORDS:
        if any(k in lowered for k in keywords):
            return need
    return "quote"


def classify_market(question: str) -> str:
    lowered = question.lower()
    for market, patterns in _MARKET_PATTERNS:
        if any(re.search(re.escape(p), lowered) for p in patterns):
            return market
    return "cn"


class Candidate(BaseModel):
    """One routed source with its live probe outcome."""

    source: str
    probe: SourceProbe
    rank: int

    @property
    def usable(self) -> bool:
        return self.probe.status in READY_STATES


class RoutingPlan(BaseModel):
    """Full routing decision: auditable, never silent."""

    question: str
    need: str
    market: str
    candidates: list[Candidate] = Field(default_factory=list)
    note: str = ""


def route(
    question: str,
    *,
    market: str | None = None,
    need: str | None = None,
) -> RoutingPlan:
    """Classify → look up rules → probe each candidate, in order."""
    resolved_need = need or classify_need(question)
    resolved_market = market or classify_market(question)
    names = routing_candidates(resolved_need, resolved_market)
    candidates: list[Candidate] = []
    for rank, name in enumerate(names):
        spec = get_spec(name)
        candidates.append(
            Candidate(source=name, probe=build_adapter(spec).probe(), rank=rank)
        )
    note = (
        ""
        if candidates
        else f"no routing rule for need={resolved_need!r} "
        f"market={resolved_market!r}; refusing to guess"
    )
    return RoutingPlan(
        question=question,
        need=resolved_need,
        market=resolved_market,
        candidates=candidates,
        note=note,
    )


def fetch_routed(
    plan: RoutingPlan,
    *,
    api_by_source: dict[str, str],
    params: dict[str, object] | None = None,
    as_of: str | None = None,
    runner: object | None = None,
) -> FetchResult:
    """Walk the plan in rank order until a source truly succeeds.

    ``api_by_source`` maps source name → source-native API name (the caller
    chooses the concrete endpoint per source; routing chooses *who* to ask).
    Every skip and failure is recorded; the final failure lists them all.
    """
    attempts: list[str] = []
    for candidate in sorted(plan.candidates, key=lambda c: c.rank):
        probe = candidate.probe
        if not candidate.usable:
            attempts.append(
                f"{candidate.source}: skipped ({probe.status} — {probe.detail})"
            )
            continue
        api = api_by_source.get(candidate.source)
        if not api:
            attempts.append(f"{candidate.source}: no API mapping provided")
            continue
        spec = get_spec(candidate.source)
        result = build_adapter(spec, runner=runner).fetch(  # type: ignore[arg-type]
            FetchRequest(api=api, params=params or {}, as_of=as_of)
        )
        if result.ok:
            return result
        attempts.append(f"{candidate.source}: failed — {result.error}")
    if not attempts:
        attempts.append(plan.note or "no candidates")
    return FetchResult.failure(
        source="router",
        api="fetch_routed",
        error="all routed sources failed or unavailable: " + " | ".join(attempts),
        status=SourceStatus.NO_SCRIPT,
    )

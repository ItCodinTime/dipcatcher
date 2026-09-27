# fx-1 Datasource Layer — 18 Professional Sources, One Honest Interface

Every installed finance plugin is wired into the repo as a first-class,
honesty-gated datasource. Corpus ingest records these sources for a future
training run. The harness fetches them when a source exists. No trained
fx-1 checkpoint is in this repository.

## Design invariants

1. **No fabrication, ever.** Adapters execute the plugins' bundled CLI
   scripts via subprocess. A failure returns `FetchResult(ok=False)` with
   the real error — never placeholder data.
2. **Credentials are environment-only.** Probes check env var *names*
   (`KIMI_API_KEY`, `AGENT_GW_TOKEN`, `DATASOURCE_BASE_URL`,
   `DATASOURCE_API_KEY`) and the agent-gw runtime config file; values are
   passed to child processes via scrubbed environments, never on argv,
   never printed, never logged.
3. **Point-in-time discipline.** Time-stamped sources
   (`requires_as_of=true`) cannot enter the training corpus without an
   explicit `as_of` observation date — the leakage guard refuses the
   ingest (exit 1) and records the exclusion in the ledger.
4. **Live-claim quarantine.** Payloads asserting live trading performance
   (`live_pnl_claim: true`, "live trading profit", 实盘收益 …) become
   *negative* training examples teaching refusal — never positives.
5. **Tamper-evident provenance.** Every ingested payload is hash-chained
   into the corpus ledger: payload hash → ingest-code hash → example hash.
6. **Honest degradation.** Routing walks candidates in authority order;
   every skip/failure is recorded with its reason and surfaced in the
   aggregated error. MCP-only sources report `MCP_REQUIRED` instead of
   pretending offline capability.

## Coverage matrix

| Source | Markets | Assets | Latency | Owner plugin(s) |
|---|---|---|---|---|
| Wind 万得 | cn, hk, us | equity, fund, bond, index, macro, news, filings | daily | wind-allskill, financial-market-terminal, xtt-public-markets-investing |
| iFinD 同花顺 | cn, hk | equity, fund, index, macro | daily | financial-market-terminal, xtt-* suites, institutional-finance-kit |
| Gildata 恒生聚源 | cn, hk | equity, fund, index, macro, news, filings, research | daily | gildata-aifinmarket, financial-market-terminal, xtt-public-markets-investing |
| S&P CapIQ | us, global | equity, research, filings | filings | sp_data, institutional-finance-kit, xtt-public-markets-investing |
| SEC EDGAR | us | filings, equity | filings | sec_edgar, xtt-public-markets-investing (free official, no creds) |
| Yahoo Finance | us, global, hk | equity, index | daily | yahoo_finance (US fallback only) |
| 东方财富妙想 | cn, hk, us | equity, fund, bond, index, macro, news, research, sentiment | daily | dongcai-mx-data, financial-market-terminal |
| 财联社 CLS | cn | news | realtime | cls-news, financial-market-terminal (fastest CN tape) |
| 财新数据 | cn | bond, fund, macro, enterprise, sentiment, industry_chain, equity, research | daily | caixin-data-agent, financial-market-terminal, xtt-public-markets-investing |
| Binance | crypto | crypto | realtime | binance_crypto (only sanctioned crypto source) |
| IMF (WEO/COFER) | global | macro | macro | imf, xtt-public-markets-investing |
| World Bank | global | macro | macro | world_bank_open_data (free official) |
| IGO + FRED | global | macro | macro | igo_open_data (WHO/Eurostat/ECB/UNICEF/OECD/FAO/UNSD + FRED) |
| 新华财经 XHCJ | cn | news, filings | news | financial-market-terminal, xtt-public-markets-investing |
| finance-research | cn, hk, us, global | research | research | financial-market-terminal, xtt-*, institutional-finance-kit |
| 天眼查 | cn | enterprise | enterprise | xtt-* suites |
| finance-fetch router | cn, hk, us | equity, filings, research, news | filings | institutional-finance-kit (scenario envelopes, free↔paid chains) |
| 进门投研 Finenter | cn, hk | research, news | research | comein-agent (**MCP-only** — honestly unavailable offline) |

## Routing discipline (authority order)

| Need | cn | hk | us |
|---|---|---|---|
| quote | ifind → wind → gildata → dongcai | wind → ifind → gildata → dongcai | sp_data → yahoo → dongcai |
| fundamentals | ifind → gildata → wind → dongcai → finance_fetch | gildata → wind → ifind → finance_fetch | sp_data → gildata → sec_edgar → yahoo → finance_fetch |
| filings | wind → gildata → dongcai → caixin → xhcj | same | sec_edgar → sp_data → finance_fetch |
| news | cls → xhcj → dongcai → wind → caixin | dongcai → gildata → wind | yahoo → finance_research |
| macro | imf → world_bank → igo → wind → dongcai → caixin (all markets) | | |
| research | finance_research → gildata → dongcai → finenter (all markets) | | |
| crypto | binance_crypto (only) | | |
| enterprise | tianyancha → caixin | — | — |
| sentiment | caixin → dongcai | — | — |
| industry chain | caixin → gildata → dongcai | — | — |

## CLI

```bash
fx1 sources list                      # all 18 sources + live probe status
fx1 sources probe [name]              # availability JSON (no secret values)
fx1 sources describe <name>           # the source's own capability docs
fx1 sources route --question "..."    # classified need + probed candidates
fx1 sources fetch <name> --api X --params-json '{...}' --as-of YYYY-MM-DD
fx1 sources scenarios                 # finance-fetch scenario coverage
fx1 corpus ingest-source <name> --api X --as-of ... \
    --ledger-path data/fx1/corpus_ledger.jsonl --out data/fx1/corpus.jsonl
```

## Verified live (2026-09-25)

- `sources describe cls` / `imf` → real capability docs via agent-gw.
- `sources fetch cls --api cls_telegraphs` → real 财联社 telegraph items.
- `sources fetch binance_crypto --api binance_crypto_price` → real BTCUSDT
  24h ticker, payload hash `bc31361c…`, 2.1s.
- `corpus ingest-source cls … --as-of 2026-09-25` → accepted example chained
  into the ledger (`chain_valid: true`).
- Wrong API names surface the source's own error verbatim
  (`API_NOT_FOUND … Available APIs: …`) — the failure is the message.

## Boundaries (stated honestly)

- Adapters are only as available as the host: missing scripts → `no_script`,
  missing creds → `no_credentials`, MCP-only → `mcp_required`.
- The routing table encodes authority order; the caller still picks the
  source-native API name (`api_by_source`) — read `sources describe` first.
- Point-in-time *vendor* snapshots (revisable fundamentals as first
  published) remain a data-procurement task; the `as_of` gate enforces the
  discipline but cannot manufacture a PIT vendor feed.

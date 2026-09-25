# Acceptance criteria v7 — professional datasource layer

Extends v6 (all prior criteria hold). New:

1. **Registry completeness:** 18 underlying datasources covering every
   installed finance plugin — Wind, iFinD, Gildata, S&P CapIQ, SEC EDGAR,
   Yahoo Finance, 东方财富妙想, 财联社 CLS, 财新数据, Binance, IMF,
   World Bank, IGO+FRED, 新华财经 XHCJ, finance-research, 天眼查,
   finance-fetch scenario router, 进门投研 Finenter (MCP).
2. **No fabrication:** adapters only execute bundled plugin CLIs; failures
   return `FetchResult(ok=False)` with the real error. Tested: nonzero
   exit, timeout, empty output, envelope `ok=false`, MCP-only sources.
3. **Credential discipline:** env-only, name-only probing, scrubbed child
   environments, config-file resolution; test proves probe JSON never
   contains secret values.
4. **PIT/leakage gate:** time-stamped sources cannot enter the corpus
   without `as_of`; refusals are recorded as ledger exclusions.
5. **Live-claim quarantine:** payloads asserting live performance become
   negative (refusal) examples, never positives.
6. **Routing discipline:** CN→iFinD/Wind/Gildata; US→S&P→Gildata→EDGAR→
   Yahoo; crypto→Binance only; macro→IMF/WB/IGO; news→CLS first.
   `fetch_routed` aggregates every skip/failure reason honestly.
7. **Provenance:** ingested payloads hash-chained into the corpus ledger
   (payload → transform-code → example); `chain_valid` verified.
8. **Live verification (2026-09-25):** real fetches succeeded through the
   layer — CLS telegraph (real 财联社 items), Binance BTCUSDT 24h ticker
   (payload hash recorded), IMF/CLS capability docs; wrong API names
   surface the source's own error verbatim.
9. **CLI:** `fx1 sources list|probe|describe|fetch|route|scenarios` and
   `fx1 corpus ingest-source` all covered by tests.
10. **Triple gate still green:** 137 tests, ruff, mypy (47 files).

# tests/fixtures/news — SYNTHETIC multilingual news corpus

Hand-written SYNTHETIC fixture headlines for `quant_fund.research.news_signals`.
**Nothing here is a real news article.** All records carry
`"data_label": "SYNTHETIC"` and `source: "synthetic_fixture"`. Headlines were
authored for this test suite to *resemble* wire copy; company names/tickers
appear only so the entity linker has realistic surface forms to resolve.

## Files

| file | language | purpose |
|---|---|---|
| `en.jsonl` | en | English wire-style headlines (US universe tickers) |
| `zh.jsonl` | zh | Chinese financial-press style (US + CN/HK names) |
| `ja.jsonl` | ja | Japanese financial-press style (US + TYO names) |
| `edge_cases.jsonl` | mixed | cashtags, qualified mentions, publisher-tagged symbols, ingest-lag boundary crossings, unlinked records |

## Schema (JSONL, one record per line)

`record_id`, `publish_ts` (ISO-8601, tz-aware), `ingest_ts`, `language`
(`en`/`zh`/`ja`), `source`, `headline`, optional `body`, `symbols`
(publisher-tagged tickers), `data_label`, `sentiment_label` (`pos`/`neu`/`neg`
gold label for the classifier harness).

Timestamps land in June 2024 to overlap the real `file_us_wide` daily bar
panel (session bars print 20:00 UTC under DST that month). A headline with
`effective_ts = max(publish_ts, ingest_ts)` strictly before bar `T` is usable
at `T`; anything at/after rolls to the next session — see
`src/quant_fund/research/news_signals/alignment.py`.

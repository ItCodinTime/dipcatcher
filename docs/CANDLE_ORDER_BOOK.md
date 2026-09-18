# Candle + order book research

This path is branded **Northset**. Canonical docs: [`NORTHSET.md`](NORTHSET.md)
and [ADR-021](decisions/021-northset-microstructure.md).

`quant_fund.microstructure` remains the implementation helpers (`bench_candle_order_book`,
`attach_candle_book_features`, `synthetic_lob`, `vendor_book_map`, `book_panel`). The research
family, CLI, and catalog name is **Northset**. Kyle/OFI diagnostics live in
`quant_fund.northset.kyle_ofi`.

## Fuse contract (`attach_candle_book_features`)

- Asof join: candle `decision_time` ← book `available_time` (backward), `by=security_id`,
  age capped by `max_book_age_seconds` (default 1d).
- Stamps `join_coverage`, `book_age_seconds`, `book_source`, `book_dgp`.
- Fail-closed: empty fuse; `join_coverage < min_join_coverage` (default **1.0** if book was
  synthesized, **0.5** if external); external panels need a single non-null `source`.
- Depth honesty is enforced when loading/validating panels (`validate_book_panel_depth_honesty`).

**Honesty:** date-level IC only; SYNTHETIC or offline-remapped books; `research_only`;
no live tape / no live P&L claim. See `NORTHSET.md` § Fail-closed integrity gates,
Date-level IC, Session L2, and Vendor remap.

Optional: set `northset.include_kyle_ofi: true` so `bench_northset` nests the full `kyle_ofi` research blob under the Northset receipt (see NORTHSET § Kyle / OFI).

## TOB size share rate

`bench_candle_order_book` stamps `tob_size_share_finite_rate` (fraction of eligible rows with finite `tob_size_share`). Distinct from northset receipt `mean_tob_size_share` (nanmean of the share). See DATA_CONTRACTS.

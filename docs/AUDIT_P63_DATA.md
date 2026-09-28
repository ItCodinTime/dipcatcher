# P6.3 data-integrity audit — `src/quant_fund/data/`

Scope: `ingest.py`, `point_in_time.py`, `universe.py`, `corporate_actions.py`,
`security_master.py`, `lake.py`, `calendars.py`, `sources/*`. Every file was
read end-to-end against its cited spec (PIT discipline, Binance/SEC/FRED/ALFRED
API contracts, total-return math, universe eligibility rules). Fix commits:
`76d568b`, `8e6f6ae`. Regression KATs live in
`tests/unit/data/test_p63_audit_regressions.py` (25 tests, deterministic,
offline, stubbed clients).

## Bugs fixed

1. `HttpClient._request` (sources/base.py) — **FIXED**. `http.client`
   exceptions that are not `OSError` (`IncompleteRead`, `BadStatusLine`,
   generic `HTTPException`) escaped the retry loop entirely and surfaced raw
   instead of `SourceError`, making transient transport faults fatal and
   contract-breaking for callers that only catch `SourceError`. Now
   `retry_on`/`except` cover `(OSError, http.client.HTTPException)`.
2. `pit_frame` (sources/base.py) — **FIXED**. A source row without
   `event_time` raised bare `KeyError`; an unparseable timestamp raised bare
   `ValueError` — both leaking past the `SourceError` contract. Missing or
   unparseable timestamps now raise `SourceError`.
3. `adjust_prices` (corporate_actions.py) — **FIXED**. A computed
   `_div_ret`-cumprod `close_total_return` column was unconditionally
   overwritten by the second (split-aware) computation — dead math. Worse,
   null/NaN/inf/zero/negative `close` silently propagated NaN/inf through the
   total-return cumprod (silent NaN swallowing). Removed the dead block; the
   function now fails closed on missing/non-finite/non-positive close.
4. `membership_asof` top-N (universe.py) — **FIXED**. The loop path sorted by
   `adv` only, so equal-ADV ties were resolved by Polars' unspecified stable
   order, while the vectorized `build_membership_panel` tied on
   `security_id`. Same inputs could return different universes between paths.
   Loop path now sorts `["adv", "security_id"]` — matches vectorized.
5. `_apply_vectorized_tickers` (universe.py) — **FIXED**. The vectorized
   ticker-change path never called `require_valid_ticker_changes`, so a blank
   `new_ticker` silently rewrote a member's ticker to `""` where the loop
   path raised `PointInTimeError`. Now both paths validate identically.
6. `_paginated_klines` (sources/adapters.py) — **FIXED**. `start_time=None`
   defaulted the cursor to `PERP_EARLIEST_MS` (2019-09-01, USDT-M futures
   launch) for **all** callers — silently truncating Binance spot history
   before 2019 (spot launched 2017-07-01). Callers now pass their product's
   floor (`SPOT_EARLIEST_MS` vs `PERP_EARLIEST_MS`); the helper defaults to
   epoch-0 so an unspecified floor can never truncate.
7. `BinancePerpUniverseSource` (sources/adapters.py) — **FIXED**. A member
   missing `onboardDate` was silently imputed to `PERP_EARLIEST_MS` — a
   fabricated listing event_time is a PIT integrity violation. Missing or
   invalid `onboardDate` now raises `SourceError`.
8. `BinanceFundingRateSource` (sources/adapters.py) — **FIXED**. A funding
   row missing `fundingTime`/`fundingRate` raised bare `KeyError`/`ValueError`
   mid-pagination. Now `SourceError` per row.
9. `SecEdgarSource` (sources/adapters.py) — **FIXED**. SEC submissions return
   *parallel* `filingDate`/`form`/`accessionNumber` arrays; `zip` without
   `strict` silently truncated a ragged payload to the shortest column —
   filings dropped without a trace. Now verifies equal length and uses
   `zip(..., strict=True)`.
10. `AlfredSource`/`FredSource` CSV path (sources/adapters.py) — **FIXED**.
    `AlfredSource.csv_endpoint` pointed at `fredgraph.csv` (FRED's endpoint on
    the FRED host's path shape); verified live that ALFRED serves
    `alfredgraph.csv` and that its data column is vintage-suffixed
    (`GDP_20260915`), never the bare `series_id` — so every ALFRED CSV fetch
    previously produced all-null values. The CSV path now resolves the single
    non-`observation_date` column when the bare id is absent, and raises
    `SourceError` on ambiguity.
11. Kline row validation (sources/adapters.py) — **FIXED**. The shape check
    verified `len(item) >= 7` but the open/close-time fields were cast with
    bare `int()` in the row comprehension / pagination cursor — non-numeric
    timestamps escaped as raw `ValueError`. `_require_kline_rows` validates
    shape + both timestamps on every path (paginated, legacy single-page,
    perp).
12. Treasury/CFTC/FINRA ragged rows (sources/adapters.py) — **FIXED**.
    `date_key`/`value_key` were probed on row 0 only; a later row missing the
    key raised bare `KeyError`. Now `.get()`; a missing timestamp surfaces as
    `SourceError` via `normalize_observations`, a missing value follows the
    FRED `"."` skip convention.
13. `WorldBankSource` (sources/adapters.py) — **FIXED**. `payload[1]` was
    iterated unchecked; a non-list second element escaped as raw
    `TypeError`/`AttributeError`. Now `SourceError` on unexpected shape.
14. `ingest` master attach (ingest.py) — **FIXED**. Attributes were attached
    only when the master contained a `sector` column, silently dropping
    `exchange`/`industry` from every bar when the vendor master lacked
    `sector`. `attach_master_attributes` already self-gates on its own
    `_ATTR_COLS`; the redundant `sector` gate is removed.
15. `get_bars`/`make_provider` (ingest.py) — **FIXED**. `source="ccxt"` /
    `"cryptofeed"` fell through `get_bars`' dispatch (`elif source not in
    {"ccxt","cryptofeed"}: raise`) into a misleading `PublicMarketProvider`
    path instead of failing at the boundary. Unknown/non-file sources now
    raise immediately, and `make_provider` raises `ValueError` early with a
    clear message (`config.supported_source` still accepts them — that's a
    config-model lane, flagged here not fixed there).
16. `write_parquet`/`write_source_frame` (lake.py, sources/storage.py) —
    **FIXED**. Parquet files and receipt JSON were written in place; a crash
    mid-write left torn artifacts at canonical paths. Both now write
    `<name>.tmp` then `os.replace`, cleaning up on failure.
17. `validate_feature_frame` (point_in_time.py) — **FIXED**. The
    whole-frame availability check went through `max_available_time`, which
    demands the full PIT column set — so a feature frame carrying
    `event_time`+`available_time` (the documented contract for features, per
    callers in pipeline/dataset.py and features/engine.py) died on a
    misleading "missing PIT columns" error before the actual PIT check. Now
    checks the availability column directly; `assert_pit_safe` semantics are
    unchanged (still raises on any availability after decision_time).

## Ledger

Verdicts: `correct`, `fixed` (KAT attached), `waived` (looser semantics are
intentional — reason noted), `suspicious-but-unproven`.

### corporate_actions.py

| claim checked | verdict | fix commit |
|---|---|---|
| `adjust_prices` total-return cumprod math (split-adjusted ratio method) | correct | — |
| dead `_div_ret`/`close_total_return` block | fixed | `76d568b` |
| silent NaN/inf on non-positive/non-finite/missing close | fixed | `76d568b` |
| dividends not restating earlier bars when discovered late | waived — PIT-correct: bars reflect what was knowable at their available_time | — |
| `require_valid_ticker_changes` blank/missing new_ticker | correct (raises) | — |
| `delisted_ids_asof`/`ticker_overrides_asof` as-of filtering | correct | — |

### universe.py

| claim checked | verdict | fix commit |
|---|---|---|
| loop vs vectorized membership equivalence | fixed — two real divergences | `76d568b` |
| top_n_adv tie-break determinism | fixed | `76d568b` |
| blank `new_ticker` handling parity | fixed | `76d568b` |
| min_price/min_adv/min_history_bars eligibility rules | correct | — |
| `attach_membership_flag` dtype/schema collisions | correct — SchemaError propagates (fail-closed) | — |
| `lookback=20` ADV window hardcoded vs `UniverseConfig` | waived — spec constant; config has no adv-lookback field | — |

### ingest.py

| claim checked | verdict | fix commit |
|---|---|---|
| master-attribute attach gating on `sector` | fixed | `76d568b` |
| `get_bars` dispatch coverage for non-file sources | fixed | `76d568b` |
| `make_provider` ccxt/cryptofeed routing | fixed (earlier ValueError); config validator accepting them is out-of-lane (`config/models.py`) — noted | `76d568b` |
| bronze→silver promotion order & schema | correct | — |

### point_in_time.py

| claim checked | verdict | fix commit |
|---|---|---|
| `validate_feature_frame` whole-frame check requires full PIT schema | fixed | `76d568b` |
| `assert_pit_safe(max_avail > decision)` semantics | correct | — |
| `require_pit_columns`/`filter_available` fail-closed | correct | — |

### security_master.py

| claim checked | verdict | fix commit |
|---|---|---|
| `_ATTR_COLS = (sector, industry, exchange)` — ticker/security_type not bar-attached | waived — intentional: universe filters read master via `snapshot_asof` | — |
| `snapshot_asof` validity-window filtering | correct | — |

### sources/base.py

| claim checked | verdict | fix commit |
|---|---|---|
| `HttpClient` retry coverage of `http.client.HTTPException` | fixed | `76d568b` |
| exhausted transport errors wrap in `SourceError` | fixed | `76d568b` |
| `pit_frame` missing/unparseable event_time | fixed | `76d568b` |
| `query_url` drops None params | correct | — |
| empty `pit_frame([])` | correct — collector `_validate_frame` rejects (fail-closed) | — |

### sources/adapters.py

| claim checked | verdict | fix commit |
|---|---|---|
| `_paginated_klines` default floor truncating spot pre-2019 | fixed | `76d568b` |
| kline timestamp fields cast bare | fixed | `8e6f6ae` |
| in-progress bar dropped by close-time filter | correct | — |
| `BinanceFundingRateSource` malformed row / non-finite rate | fixed / already fail-closed | `76d568b` |
| `BinancePerpUniverseSource` missing `onboardDate` imputed | fixed | `76d568b` |
| perp-universe survivorship truncation | waived — documented; stamped `available_time=now` | — |
| malformed `quoteVolume` entries silently skipped | waived — conservative: symbol excluded rather than ranked on garbage | — |
| `SecEdgarSource` ragged parallel arrays | fixed | `76d568b` |
| `FredSource` API + CSV paths; `AlfredSource` endpoint/vintage columns | fixed | `76d568b` |
| Treasury/CFTC/FINRA ragged-row KeyError | fixed | `8e6f6ae` |
| `WorldBankSource` payload shape | fixed | `8e6f6ae` |
| `GdeltSource`/`BeaSource`/`ItchSampleSource`/`Fi2010Source` | correct — fail via `normalize_*`/empty-frame contract | — |
| `_OptionalLibrarySource` (ccxt/cryptofeed/openbb) | correct — raises without payload | — |

### sources/normalize.py

| claim checked | verdict | fix commit |
|---|---|---|
| `normalize_ohlcv` dup-key, positivity, low≤open/close≤high checks | correct | — |
| `normalize_observations` null `event_time`/`available_time` | correct (raises) | — |
| missing `value` skips the row | waived — FRED `"."` unreported convention, consistent across sources | — |

### sources/registry.py

| claim checked | verdict | fix commit |
|---|---|---|
| `get_source` unknown name | correct — `ValueError` with valid names | — |
| lazy HF adapter install | correct | — |

### sources/storage.py

| claim checked | verdict | fix commit |
|---|---|---|
| non-atomic parquet/receipt writes | fixed | `76d568b` |
| receipt immutability contract | correct | — |

### lake.py

| claim checked | verdict | fix commit |
|---|---|---|
| non-atomic `write_parquet` | fixed | `76d568b` |
| layered bronze/silver/gold paths | correct | — |

### calendars.py

| claim checked | verdict | fix commit |
|---|---|---|
| weekday-only trading calendar | waived — documented convention; holiday calendars out of scope | — |
| `is_session`/day iteration boundary handling | correct | — |

## Out-of-lane issues surfaced (not fixed here)

- `config/models.py`: `DataConfig.supported_source` accepts `ccxt`/`cryptofeed`,
  which are source adapters, not market-bar providers — `get_bars`/`make_provider`
  now fail earlier with a clear message, but the config enum still advertises
  them.
- `collector.py` / `concurrent_io.py`: reviewed as dependencies; transport
  retry policy and PIT re-validation behave as their callers require — no fix
  needed.
- `tests/unit/pipeline/test_phase1_receipt_verification.py::
  test_completed_runs_and_index_survive_relocation` fails on clean `main`
  (7d2e01e) on macOS arm64 — pre-existing, unrelated to this lane.

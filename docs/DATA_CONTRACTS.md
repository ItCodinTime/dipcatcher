# Data contracts

## Point-in-time fields

Every stored observation includes:

| Field | Meaning |
|---|---|
| `event_time` | When the economic event occurred (bar close, print, filing timestamp) |
| `available_time` | When a market participant could know it |
| `ingested_time` | When this system ingested it |
| `source` | Adapter / vendor id |
| `security_id` | Immutable internal id |
| `symbol` | Ticker at `event_time` (not a join key) |
| `revision_id` | Vintage of restated data |

`available_time <= decision_time` is mandatory for every feature. For daily bars, default `available_time = event_time` (close). Ingestion lag is recorded but does not relax availability.

### Public file tape (Stooq)

`quant_fund.data.adapters.stooq` writes a PIT-shaped lake from Stooq daily CSVs.
If Stooq returns a JavaScript challenge instead of CSV, `yahoo_eod` writes the
same lake from Yahoo v8 daily charts (`revision_id=YAHOO_VENDOR_ADJ`).
`available_time` equals session close (16:00 America/New_York for US, 16:30
Europe/London for UK) — a close-published convention, **not** SIP as-of
vintages. Vendor EOD prints are **split-adjusted as received**; bronze stores
them with an explicit vendor-adjusted revision and silver split factors are
identity unless a separate corporate-action file is supplied. Universe
membership is a liquidity filter on this tape, not an index reconstitution.
This tape can be scored scientifically. It cannot mint a live P&L claim.

## Bars (bronze)

Columns: `security_id`, `symbol`, `event_time`, `available_time`, `ingested_time`, `source`, `revision_id`, `open`, `high`, `low`, `close`, `volume`, `currency`, `session`.

OHLC are **raw** (unadjusted). Volume is share volume. Every accepted bar must
also satisfy the candle envelope `low <= open <= high` and
`low <= close <= high`, in addition to positive finite prices and `high >= low`.
The file adapter rejects envelope violations with `PointInTimeError`; impossible
candles must not be persisted into downstream feature, label, or backtest inputs.

## Corporate actions (bronze)

`action_type` in `{split, cash_dividend, special_dividend, delist, ticker_change}`. Splits store `factor` (e.g. 2.0 for 2-for-1). Dividends store `amount` in the listing currency and `ex_date` as `event_time`. `available_time` is the announcement time when known; otherwise ex-date (conservative: do not assume earlier knowledge).

## Adjusted series (silver)

Computed, never replacing raw:

- `close_split_adjusted`
- `close_total_return` (splits + cash dividends reinvested)
- `open/high/low` split-adjusted with the same cumulative split factor

**Label mapping**

| Target | Series |
|---|---|
| Alpha / ranking / excess return | simple total-return |
| Distribution quantiles | log total-return |
| Realized volatility / HAR-RV | log total-return (or RV if intraday exists) |
| Liquidity / Amihud | raw dollar volume with simple raw return |

## Security master

`security_id`, `ticker`, `name`, `exchange`, `currency`, `sector`, `industry`, `valid_from`, `valid_to`, plus the universal PIT fields `available_time`, `ingested_time`, `source`, and `revision_id`. Ticker lookup is as-of dated: `valid_from`/`valid_to` define the economic window, while `available_time` is when the mapping could be known. Late-arriving restatements may have `valid_from` earlier than `available_time`; they must not rewrite pre-availability identity. The file adapter fails closed on missing PIT/identity columns, blank source/ticker/security_id, inverted windows, and duplicate `(security_id, valid_from)` or `(ticker, valid_from)` rows. Direct `FrameSecurityMaster` callers without `available_time` retain the legacy valid-window lookup.

## Universe membership

`effective_from`, `effective_to`, `security_id`, `symbol`, `sector`, `industry`, `exchange`. Membership at \(t\) uses only information with `available_time <= t` (trailing ADV, price, history, and PIT-visible `delist` / `ticker_change` corporate actions). Today's index constituents are never applied historically. Delisted names remain in history through the last listed session; late-arriving delists cannot rewrite pre-availability membership. Ingest persists `silver/universe.parquet` and drops post-delist bars from the silver panel once the event is knowable. `ticker_change` requires a non-blank `new_ticker` and overrides `symbol` only after both `event_time` and `available_time`. Gold features/labels, cross-sectional ranks, market aggregates, and the paper/backtest feature panel inner-join that membership on `(security_id, event_time)`; name-level rolling history is still computed on silver, but ineligible names cannot move decision-time ranks or remain in the persisted gold panel. An empty or duplicate-keyed universe artifact fails closed rather than silently training on the unfiltered silver panel. Cached gold loaded by training/forecast `panel()` is checked again against the current universe artifact: keys outside membership fail closed, and the panel cache is keyed by universe bytes so an in-place membership replacement cannot reuse a stale joined frame. The causal GARCH market overlay uses that same membership when the universe artifact exists: ineligible names on paper/backtest execution bars cannot move `max_predicted_vol`. The overlay as-of cache is keyed by a digest of the full causal return path.

## Feature frames (gold)

Must include `decision_time`, `max_source_available_time`, `security_id`, feature columns, and `feature_set_version`. Building a frame with `max_source_available_time > decision_time` raises `PointInTimeError`. Production `build_gold` persists only PIT-universe members at each `event_time`; cross-sectional and market aggregates use that same membership, not the full silver panel. Loading cached gold for training or `forecast_asof` re-checks those keys against `silver/universe.parquet` and refuses ineligible rows.

## Forecasts

See `AssetForecast` in `quant_fund.schemas.forecast`. Horizon keys are strings `1d`, `5d`, `20d`. All engines write `model_version`. Optional conformal fields `interval_lo`, `interval_hi`, `interval_alpha`, and `interval_method` (`mondrian_cqr` / `split_cqr`) are coverage-guaranteed sets and do not replace raw quantile PIT, pinball, or CRPS. When `robinhood_plus.enabled` and `blend_weight > 0`, fused forecasts may carry K-line path expected returns / quantiles / `P(R>0)` from split-adjusted lookback bars with `event_time` (and `available_time` when present) \(\le\) the decision clock. Every as-of stamps `robinhood_plus_n_ok` and `robinhood_plus_n_fallback`; a missing OHLC window is a fallback, never a fabricated path. Diagnostics stamp `core_engine=robinhood_plus` only when that name produced a finite path and the blend is applied; fusion still wraps and the risk gate still applies.

## Persisted model and target-weight artifacts

Joblib model loading verifies the optional SHA-256 sidecar before deserialization and
then requires the deserialized object to be an instance of the requested model
class. A checksum proves byte stability, not model-family identity or trust; wrong
family artifacts fail at the load boundary, while untrusted pickle files remain
outside the supported trust boundary.

The `/risk/portfolio` endpoint accepts a persisted target-weight panel only when it
contains `event_time`, `security_id`, and `target_weight`; timestamps and identity
fields are non-null, timestamps are datetimes, weights are finite numeric values,
and `(event_time, security_id)` keys are unique. Structural violations return a
controlled 422 response rather than producing a misleading measured risk report.

## Synthetic data

Rows produced by `SyntheticMarketProvider` set `source="synthetic"` and `revision_id="SYNTHETIC"`. Reports and CLI output prefix `SYNTHETIC` when any input row is synthetic. Synthetic results are not evidence of live edge.


## Northset canonical bars (`data_view`)

Silver/adjusted contract for the Northset bench entrypoint:

- Prefer full `open/high/low/close_split_adjusted` (partial set fails closed).
- Optional `close_total_return`, `volume_split_adjusted`, `split_factor`.
- Output view stamps `price_basis` / `return_basis` and materializes `return_open` / `return_close`.
- Config gate: `northset.require_adjusted_ohlc` (default true). Opt-out is fixture-only.

Does not replace the bronze raw bar contract above; it is the research view Northset scores on.

`bench_northset` stamps `price_basis` and `return_basis` from this view onto the family receipt (see `northset.benches`).


## Order-book snapshots (Northset)

L2 snapshots (`schemas.order_book.OrderBookSnapshot`) are PIT at `event_time` / `available_time`. Both timestamps must be non-null timezone-aware UTC. `(security_id,event_time)` is unique; required prices, spreads and depths are finite and positive; imbalances lie in `[-1,1]`. Bids are best→worse (descending price); asks best→worse (ascending). Best bid must be strictly below best ask (no crossed or locked book). SYNTHETIC books from `northset` / `microstructure.synthetic_lob` set `source="synthetic"` and `revision_id="SYNTHETIC_LOB_v1"`. They are aligned to bar timestamps and are not a vendor tape.

Candle/book fusion is an availability-time as-of join: select the latest snapshot with
`book_available_time <= decision_time`, subject to `northset.book_max_age_seconds`.
Future or stale snapshots cannot enter the feature row. External-panel join coverage
must meet `northset.book_join_coverage_floor`.

Columns on the metrics frame: `best_bid`, `best_ask`, `mid`, `spread`, `spread_bps`, `microprice`, `imbalance_top`, `imbalance_depth`, `bid_depth`, `ask_depth`.

## Session candles (Northset)

Pseudo-session candles reconstructed from a completed daily OHLC bar store `parent_event_time`, `session_index`, and the same PIT fields as bars. Their synthetic event timestamps end at the parent timestamp, but **every row is available only at the parent bar's availability time**; they are never an observable intraday tape. The session envelope must match daily open/high/low/close. `source="synthetic_reconstruction"` and `revision_id="NORTHSET_SESSION_v2_PARENT_CLOSE_AVAILABLE"`. See [`NORTHSET.md`](NORTHSET.md).


## Vendor quote remap → book panel (Northset)

Offline contract (`microstructure.vendor_book_map`). Maps Alpaca / Polygon / generic **top-of-book** columns into `BOOK_PANEL_REQUIRED`, then `validate_book_panel`.

| Northset target | Alpaca aliases | Polygon aliases |
|---|---|---|
| `security_id` | `S`, `symbol`, `security_id` | `ticker`, `T`, `symbol`, `security_id` |
| `event_time` | `t`, `timestamp`, `event_time` | `sip_timestamp`, `participant_timestamp`, `t`, `timestamp`, `event_time` |
| `available_time` | `t`, `timestamp`, `available_time` | `sip_timestamp`, `t`, `timestamp`, `available_time` |
| `best_bid` / `best_ask` | `bp` / `ap` (+ long names) | `bid` / `ask` (+ long names) |
| `top_bid_size` / `top_ask_size` | `bs` / `as` | `bid_size` / `ask_size` |

If `source` is absent, remap defaults it to the vendor preset name. Incomplete top-of-book fails closed. Beyond-top depth is **not** in this contract: `imbalance_depth` copies top imbalance; book slopes are NaN.

This remap is **not** a live SIP/ITCH subscription and does not authorize network pulls.

## Session L2 panel (Northset)

Session L2 rows are SYNTHETIC snapshots keyed by `(security_id, event_time)` with `parent_event_time` + `session_index`. They inherit PIT fields from session candles (`NORTHSET_SESSION_v1` / SYNTHETIC LOB labeling). Daily aggregates join on `event_time = parent_event_time`. Not RTH/ETH vendor clocks; not a live tape claim.

## Fail-closed book / session gates

| Gate | Module | Rule |
|---|---|---|
| `validate_session_book_counts` | `northset.identities` | Exactly `n_session_candles` snaps + unique indices per `(security_id, parent_event_time)` |
| `join_coverage` / `min_join_coverage` | `microstructure.candle_book_features` | Asof PIT join; coverage floor 1.0 (SYNTHETIC) / 0.5 (external) unless overridden |
| `validate_book_panel_depth_honesty` | `microstructure.book_panel` | Uses present `DEPTH_SHAPE_FIELDS`: thin ⇒ all NaN; deep ⇒ finite log-size/log-price slopes; tick-spacing may NaN on zero gaps; no-op if level cols absent |
| `DEPTH_SHAPE_FIELDS` | `microstructure.book_metrics` | Optional panel cols: log size/price slopes + mean log tick spacing; NaN-if-thin producer contract |
| `SIDE_STRUCTURE_FIELDS` | `microstructure.book_metrics` | `bid/ask_size_concentration_top`; finite when side depth > 0 |
| `depth_shape_finite_floor` / `concentration_top_*` / `queue_priority_*` / `side_notional_*` / `tob_size_share_finite_floor` | `northset` config → `bench_northset` | Optional [0,1] fail-closed per family; default None — see **Shape / structure floors matrix** |
| `enforce_session_l2_identity_floors` | `northset.benches` | When session L2 on: identity rates ≥ `session_l2_identity_floor` (default 0.99) or `ValueError` |
| Receipt eligibility stamps | `northset.benches` | `component_sources`, `book_hypothesis_eligible`, `session_book_hypothesis_eligible`, `sweep_evidence_scope`, `session_l2_identity_gate` |
| `include_kyle_ofi` nest | `northset.benches` | Config flag (default false): nest `bench_kyle_ofi_fused` under `receipt["kyle_ofi"]` with `research_only` assert; else stamp `include_kyle_ofi=false` only |
| External book `source` | `attach_candle_book_features` | Required, non-null, single source — refuse DGP mix |

Session L2 panels must carry `parent_event_time` + `session_index` for the count gate.
Candle/book fuse stamps `join_coverage` and `book_age_seconds` on the fused frame.


## Fuse flow aliases: `signed_volume` vs `signed_depth`

Same depth-imbalance formula on the candle↔book fuse — **two column labels**, not two signals:

| Label | Where produced | Formula | Used by |
|---|---|---|---|
| `signed_volume` | Always-on Northset fuse in `northset.benches` (`bench_northset` path into `_panel_kyle`) | `bid_depth - ask_depth` | Always-on receipt scalars `kyle_lambda` / `kyle_r2` (flow=`signed_volume`); OFI path still uses column `ofi` |
| `signed_depth` | `northset.kyle_ofi` fuse (`fuse_bars_with_book` / nest path) | `bid_depth - ask_depth` | Nested `receipt["kyle_ofi"]` when `include_kyle_ofi=true`: date-level λ + HAC / flow→Δmid IC (`flow="signed_depth"`) |

Both consume panel `bid_depth` / `ask_depth` from `book_metrics` (or session-aggregated equivalents). Do not treat a rename as a new microstructure feature. OFI remains Cont–Kukanov–Stoikov on consecutive tops and is distinct from this imbalance.

See NORTHSET disambiguation (always-on name-mean OLS vs nested date-level λ) and MATH_SPEC Kyle section.


## Always-on fuse vs `kyle_ofi` fuse columns

Side-by-side of columns **written or required** on each fuse path (not the full candle/LOB feature set).

| Role | Always-on (`bench_northset` after `attach_candle_book_features`) | `kyle_ofi.fuse_bars_l2_kyle_frame` |
|---|---|---|
| Depth imbalance | `signed_volume` = `bid_depth - ask_depth` | `signed_depth` = `bid_depth - ask_depth` (**same formula**) |
| Mid change | `delta_mid` = `mid.shift(-1) - mid` (per `security_id`) | `delta_mid` **and** `fwd_delta_mid` (same mid Δ) |
| Forward return | `fwd_ret_1` from **`candle_return_close`** ratio; also `abs_fwd_ret_1` | `fwd_ret_{1,2,3}` from mid; if `close` present, **overwritten** by close-to-close `fwd_close_ret_*` |
| OFI | From candle↔book attach (`ofi` Cont–Kukanov–Stoikov); plus `ofi_lag` = `ofi.shift(1)` (also `_IC_FEATURES` / `ofi_lag1_corr` panel) | Compute via `cont_ofi_by_security` if `ofi` absent; **no** `ofi_lag` column — nest lag1 is helper-internal; ≠ `ofi_lag1_corr` — **Always-on ofi_lag panel vs nest** |
| Effective spread / close–mid | **FIXED dual cols:** book `effective_spread` (ask−bid) preserved; candle `close_mid_abs_rel` = `2|C−mid|/mid`; receipts `mean_effective_spread` + `mean_close_mid_abs_rel` — see **dual columns** | Not minted on kyle fuse |
| Honesty stamps | `join_coverage`, `book_age_seconds` (from attach) | `book_source`, `book_dgp`, `join_coverage` literals |
| Depth fallback | Expects panel `bid_depth`/`ask_depth` from metrics | If missing, aliases `top_bid_size`/`top_ask_size` → depth |
| Join | PIT asof via `attach_candle_book_features` (+ amihud/vor/tr/sweeps/session agg joins) | Inner join on `(security_id, event_time)` when external `book` passed |
| Nest IC predictors (receipt only) | — (always-on scores fuse cols via `_IC_FEATURES`) | Nest stamps raw `ofi` / `signed_depth` ICs **and** residual_ofi / residual_depth ICs — see **Nest residual_ofi vs ofi_flow IC** |
| Forward mid on fuse | `delta_mid` (shift) used for always-on Kyle OLS | Nest requires both `delta_mid` and `fwd_delta_mid` (same mid Δ semantics for residual/predictive fwd paths) |

Always-on IC features (`_IC_FEATURES`) additionally score columns already on the fused frame (`imbalance_*`, `microprice_minus_mid_bps`, session aggs, sweeps, …) against `fwd_ret_1` — those are **not** re-derived on the kyle_ofi fuse. Nested nest estimators consume `signed_depth` / `ofi` / `delta_mid` / `fwd_delta_mid` / `fwd_ret_*`, including **residualized** ofi/depth ICs — never equate those to raw `ofi_flow_*` / `ofi_fwd_*` (see **Nest residual_ofi vs ofi_flow IC**).



## Nest residual_ofi vs ofi_flow IC (never equate)

All nest keys below live under `receipt["kyle_ofi"]` (or bare kyle receipt). Same `date_level_spearman_hac` method stamp does **not** make them interchangeable. Soft-verify (`kyle_residual_flow_honesty_errors`) gates labels/companions only — **no** numeric equality across families. research_only — never live Sharpe.

| Key family (examples) | Predictor | Target | Notes |
|---|---|---|---|
| `ofi_flow_delta_mid_*` | **raw** `ofi` | contemporaneous `delta_mid` (default `kyle_lambda_by_date` target) | Companion IC next to `kyle_lambda_ofi_mean`; name says flow_delta_mid |
| `ofi_fwd_delta_mid_*` / `ofi_fwd_ret_{1,2,3}_*` | **raw** `ofi` | **forward** `fwd_delta_mid` / `fwd_ret_*` | `predictive_flow_fwd_date_ic` — raw predictive |
| `ofi_delta_mid_lag0_*` / `lag1_*` | raw `ofi` (lagged for lag1) | contemporaneous `delta_mid` | Separate lag helper; ≠ residual; ≠ `ofi_flow` — see **Nest ofi_delta_mid_lag0/lag1 vs ofi_flow** |
| `residual_ofi_ex_depth_fwd_delta_mid_*` | **residual** = ofi − within-date linear proj onto `signed_depth` | **`fwd_delta_mid`** | `residual_flow_date_ic(flow=ofi, control=signed_depth)` |
| `residual_ofi_ex_depth_fwd_ret_{1,2,3}_*` | same residual ofi | `fwd_ret_*` | Horizon twins of residual mid IC |
| `residual_depth_ex_ofi_*` | residual depth ex ofi | fwd targets | **Swapped** flow/control — ≠ residual_ofi |

### Never-equate rules

| Do not treat as equal | Why |
|---|---|
| `residual_ofi_ex_depth_fwd_delta_mid_*` vs `ofi_flow_delta_mid_*` | Residualized + **fwd** target vs raw + **contemporaneous** Δmid |
| `residual_ofi_ex_depth_fwd_delta_mid_*` vs `ofi_fwd_delta_mid_*` | Same fwd target, but residual ≠ raw ofi predictor |
| `residual_ofi_ex_depth_fwd_ret_*` vs `ofi_fwd_ret_*` | Residual vs raw at matched horizon |
| `residual_ofi_*` vs `residual_depth_ex_ofi_*` | Flow/control swap |
| Any residual_* / nest `ofi_fwd_*` vs always-on `ofi_p_ic` | Nest date IC ≠ catalog discovery IC — **Nest ofi_fwd vs always-on ofi_p_ic** |
| Residual IC vs nest Kyle λ means | Spearman IC ≠ OLS λ |

Honesty: finite residual spearman → `kyle_residual_flow_honesty_errors` (research_only, claim, `*_t`/`*_n_dates`, no Sharpe/pnl). Cross-ref: **Kyle nest soft-verify suite**; fuse map below.


## Nest `ofi_delta_mid_lag0` / `lag1` vs `ofi_flow` (never equate)

Three nest IC families all score OFI against mid-change, but **lag axis / helper / receipt key** differ. Soft-verify does not assert equality. research_only — never live Sharpe.

| Receipt family | Helper | Feature | Target | Lag |
|---|---|---|---|---|
| `ofi_flow_delta_mid_*` | companion IC inside `kyle_lambda_by_date(flow=ofi, target=delta_mid)` | raw `ofi` | contemporaneous `delta_mid` | 0 (no shift) |
| `ofi_delta_mid_lag0_*` | `ofi_delta_mid_date_ic(lag=0)` | raw `ofi` | contemporaneous `delta_mid` | 0 |
| `ofi_delta_mid_lag1_*` | `ofi_delta_mid_date_ic(lag=1)` | `ofi.shift(1)` per `security_id` | contemporaneous `delta_mid` | **1** |
| `signed_depth_delta_mid_lag1_*` | `flow_delta_mid_date_ic(flow=signed_depth, lag=1)` | lagged depth | `delta_mid` | 1 |
| `depth_flow_delta_mid_*` | companion inside `kyle_lambda_by_date(flow=signed_depth, …)` | raw `signed_depth` | `delta_mid` (default) | 0 |

### Never-equate rules

| Pair | Why |
|---|---|
| `ofi_delta_mid_lag1_*` ≠ `ofi_flow_delta_mid_*` | Lagged predictor vs lag-0 companion to Kyle λ |
| `ofi_delta_mid_lag1_*` ≠ `ofi_delta_mid_lag0_*` | Explicit lag axis — prior ofi vs current Δmid |
| `ofi_delta_mid_lag0_*` ≠ `ofi_flow_delta_mid_*` | Separate helpers / sample construction (λ path drops nulls then ICs full finite sample; lag helper sorts + optional shift). Treat as **distinct receipt keys** even when lag=0 and estimates look close |
| `ofi_delta_mid_lag*` ≠ `ofi_fwd_*` | Contemporaneous Δmid (+ optional feature lag) ≠ stamped **forward** targets |
| `ofi_delta_mid_lag*` ≠ `residual_ofi_*` | Raw (±lag) ≠ residualized |
| `signed_depth_delta_mid_lag1_*` ≠ `depth_flow_delta_mid_*` | Same lag-1 vs lag-0 companion split for depth — **Nest signed_depth_delta_mid_lag1 vs depth_flow** |

Cross-ref: **Nest residual_ofi vs ofi_flow IC**; NORTHSET Kyle nest keys list.


## Nest `kyle_lambda_*_fwd_*` vs contemporaneous λ means (never equate)

`kyle_lambda_by_date` is parameterized by **target**. Default nest means use contemporaneous `delta_mid`; additional stamps use forward targets. Same flow (`ofi` / `signed_depth`), different y — never interchange.

| Contemporaneous (default nest λ) | Forward-target λ stamps | Target for OLS λ |
|---|---|---|
| `kyle_lambda_ofi_mean` (+ `_t`/`_p`/`_n_dates`) | `kyle_lambda_ofi_fwd_delta_mid_mean` (+ `_t`/`_std`) | `fwd_delta_mid` |
| same | `kyle_lambda_ofi_fwd_ret_1_mean` (+ `_t`/`_std`) | `fwd_ret_1` |
| `kyle_lambda_depth_mean` (+ companions) | `kyle_lambda_depth_fwd_delta_mid_mean` / `kyle_lambda_depth_fwd_ret_1_mean` | fwd mid / `fwd_ret_1` |

### Never-equate rules

| Rule | Detail |
|---|---|
| `kyle_lambda_ofi_mean` ≠ `kyle_lambda_ofi_fwd_*` | Contemporaneous Δmid λ ≠ forward-target λ |
| `kyle_lambda_*_fwd_delta_mid_*` ≠ `kyle_lambda_*_fwd_ret_1_*` | Mid-space ≠ return-space λ |
| Forward λ ≠ `ofi_fwd_*` / `residual_*` ICs | OLS λ mean ≠ Spearman IC |
| Forward λ ≠ always-on `kyle_ofi_lambda` | Still date-level nest ≠ name-mean always-on |

Companion `ofi_flow_delta_mid_*` rides only the **default** (`delta_mid`) λ path — do not read it as the IC for a fwd-target λ stamp.


## Nest `signed_depth_delta_mid_lag1` vs `depth_flow` (never equate)

Depth twin of the ofi lag / companion split. Soft-verify does not assert equality. research_only — never live Sharpe.

| Receipt family | Helper | Feature | Target | Lag |
|---|---|---|---|---|
| `depth_flow_delta_mid_*` | companion IC inside `kyle_lambda_by_date(flow=signed_depth, target=delta_mid)` | raw `signed_depth` | contemporaneous `delta_mid` | 0 |
| `signed_depth_delta_mid_lag1_*` | `flow_delta_mid_date_ic(flow=signed_depth, lag=1)` | `signed_depth.shift(1)` per name | contemporaneous `delta_mid` | **1** |
| `signed_depth_fwd_delta_mid_*` / `signed_depth_fwd_ret_*` | `predictive_flow_fwd_date_ic(flow=signed_depth, …)` | raw depth (lag 0) | **forward** targets | 0 feature / fwd y |
| `kyle_lambda_depth_mean` | OLS λ on same default path as `depth_flow_*` | depth | `delta_mid` | — (λ, not IC) |

### Never-equate rules

| Pair | Why |
|---|---|
| `signed_depth_delta_mid_lag1_*` ≠ `depth_flow_delta_mid_*` | Lagged depth vs lag-0 Kyle λ companion |
| `signed_depth_delta_mid_lag1_*` ≠ `signed_depth_fwd_*` | Feature lag on contemporaneous Δmid ≠ lag-0 depth vs **fwd** y |
| `depth_flow_delta_mid_*` ≠ `signed_depth_fwd_*` | Contemporaneous Δmid IC ≠ forward-target IC |
| `depth_flow_*` ≠ `kyle_lambda_depth_mean` | Spearman companion ≠ OLS λ mean — **Nest depth_flow vs kyle_lambda_depth_mean** |
| Depth lag/flow ≠ ofi lag/flow twins | `signed_depth_*` ≠ `ofi_*` / `ofi_flow_*` (different flow column; fuse alias `signed_depth` vs always-on `signed_volume`) |
| Depth nest ICs ≠ always-on imbalance / ofi discovery ICs | Nest diagnostic ≠ catalog H22/H27 surfaces |

Mirror: **Nest ofi_delta_mid_lag0/lag1 vs ofi_flow**.


## Nest `signed_depth_fwd_*` vs `depth_flow` (never equate)

| Receipt family | Helper | Feature | Target |
|---|---|---|---|
| `depth_flow_delta_mid_*` | companion IC in `kyle_lambda_by_date(flow=signed_depth, target=delta_mid)` | raw `signed_depth` | contemporaneous `delta_mid` |
| `signed_depth_fwd_delta_mid_*` | `predictive_flow_fwd_date_ic(flow=signed_depth, target=fwd_delta_mid)` | raw depth (feature lag 0) | **`fwd_delta_mid`** |
| `signed_depth_fwd_ret_{1,2,3}_*` | same helper, `fwd_ret_*` targets | raw depth | forward returns |
| `kyle_lambda_depth_fwd_*` | `kyle_lambda_by_date(…, target=fwd_*)` | depth | forward — **OLS λ**, not Spearman |
| `signed_depth_delta_mid_lag1_*` | `flow_delta_mid_date_ic(…, lag=1)` | lagged depth | contemporaneous `delta_mid` |

### Never-equate rules

| Pair | Why |
|---|---|
| `signed_depth_fwd_*` ≠ `depth_flow_delta_mid_*` | Forward-target IC ≠ contemporaneous Δmid companion |
| `signed_depth_fwd_delta_mid_*` ≠ `signed_depth_fwd_ret_*` | Mid-space fwd ≠ return-space fwd |
| `signed_depth_fwd_*` ≠ `kyle_lambda_depth_mean` / `kyle_lambda_depth_fwd_*` | Spearman IC ≠ OLS λ |
| `signed_depth_fwd_*` ≠ `signed_depth_delta_mid_lag1_*` | Lag-0 depth vs fwd y ≠ lagged depth vs current Δmid |
| `signed_depth_fwd_*` ≠ `ofi_fwd_*` / `depth_flow` ofi twins | Different flow column |
| Nest depth_fwd ≠ always-on imbalance / ofi discovery ICs | Nest diagnostic ≠ H22/H27 |

Mirror ofi: **Nest residual_ofi vs ofi_flow IC** (`ofi_fwd_*` vs `ofi_flow_*`). Sibling: **Nest signed_depth_delta_mid_lag1 vs depth_flow**.


## Nest `depth_flow` vs `kyle_lambda_depth_mean` (IC ≠ λ)

Same `kyle_lambda_by_date(flow=signed_depth, target=delta_mid)` call stamps **both** — they share dates/sample filters but are **different estimators**. Soft-verify never requires numeric agreement. research_only — never live Sharpe.

| Receipt keys | Estimator | Axis |
|---|---|---|
| `kyle_lambda_depth_mean` (+ `_t` / `_p` / `_n_dates`, std/iqr/dispersion) | Per-**date** OLS λ of `delta_mid` on `signed_depth` across names; then mean + HAC on the λ **series** | Price-impact slope |
| `depth_flow_delta_mid_*` (mean_spearman / t / p / n_dates) | Date-level **Spearman** IC of raw `signed_depth` vs `delta_mid` (`date_ic_series`) | Rank association |

OFI twin on the ofi λ path: `kyle_lambda_ofi_mean` ≠ `ofi_flow_delta_mid_*` (same IC≠λ split).

### Never-equate rules

| Pair | Why |
|---|---|
| `depth_flow_delta_mid_*` ≠ `kyle_lambda_depth_mean` | Spearman IC ≠ OLS λ (even same flow/target/date loop) |
| `ofi_flow_delta_mid_*` ≠ `kyle_lambda_ofi_mean` | Same split for ofi |
| `depth_flow_*` ≠ `kyle_lambda_depth_fwd_*` | Contemporaneous companion IC ≠ forward-target λ |
| IC companions ≠ always-on `kyle_lambda` / `kyle_ofi_lambda` | Nest date λ/IC ≠ name-mean always-on OLS |
| `*_n_dates` on IC ≠ `kyle_lambda_*_n_dates` | IC date count may differ from λ series length when λ drops non-finite slopes |

Dispersion / rolling HAC keys attach to the **λ series**, not to `depth_flow_*`. Cross-ref: **Nest kyle_lambda_*_fwd_* vs contemporaneous λ means**; **Nest signed_depth_fwd vs depth_flow**.


## Nest Kyle λ dispersion vs IC companions (never equate)

`kyle_lambda_dispersion` runs on the per-date **λ series** inside `kyle_lambda_by_date` (default window 10). Receipt prefixes keys by flow (`kyle_lambda_depth_*` / `kyle_lambda_ofi_*`). Soft-verify: `kyle_lambda_dispersion_honesty_errors` (+ rolling lo≤hi / window helpers). research_only — never live Sharpe.

| Family | Example keys | Object |
|---|---|---|
| **Dispersion deciles** | `kyle_lambda_depth_p10`…`p90` (ofi twin) | Percentiles of the cross-date λ series |
| **Rolling HAC band** | `kyle_lambda_*_rolling_mean`, `_rolling_hac_t`/`_p`, `_rolling_hac_lo`/`_hi` | Trailing-window mean_tstat on last `kyle_lambda_dispersion_window` finite λs; approx ±1.96·SE band |
| **Window stamp** | `kyle_lambda_dispersion_window` | Integer window used for rolling band (not an IC) |
| **Full-series λ mean/HAC** | `kyle_lambda_depth_mean` / `_t` / `_p` | mean_tstat on **all** finite date λs |
| **IC companions** | `depth_flow_delta_mid_*`, `ofi_flow_delta_mid_*` | Date Spearman of **raw flow** vs target — not λ |

### Never-equate rules

| Pair | Why |
|---|---|
| Dispersion p50 / IQR-style keys ≠ `depth_flow_*` / `ofi_flow_*` | λ-series shape ≠ flow→target Spearman IC |
| `kyle_lambda_*_rolling_hac_t` ≠ `depth_flow_*_t` / `ofi_flow_*_t` | HAC t on trailing λ means ≠ IC t on flow ranks |
| Rolling HAC ≠ full-series `kyle_lambda_*_t` | Trailing window vs full λ series inference |
| `kyle_lambda_dispersion_window` ≠ `hac_lags` | Dispersion trail length ≠ Newey–West lag stamp on nest |
| Depth dispersion ≠ ofi dispersion | Separate λ series per flow |
| Dispersion ≠ always-on `kyle_r2` / `kyle_ofi_r2` | Cross-date λ quantiles ≠ name-mean OLS R² |

Cross-ref: **Nest depth_flow vs kyle_lambda_depth_mean**; suite `kyle_lambda_dispersion_honesty_errors`.


## Nest `kyle_lambda_ofi_depth` corr vs dispersion (never equate)

| Family | Keys | Object |
|---|---|---|
| **ofi↔depth λ corr** | `kyle_lambda_ofi_depth_spearman` / `_pearson`, `kyle_lambda_ofi_depth_n_dates` (aligned), `kyle_lambda_ofi_depth_prod_hac_t`/`_p` | Corr of **per-date λ(ofi)** vs **per-date λ(signed_depth)** on shared dates (`kyle_lambda_ofi_depth_corr`); HAC on demeaned product proxy |
| **Per-flow dispersion** | `kyle_lambda_depth_p10`…`p90`, `kyle_lambda_ofi_p*`, `kyle_lambda_*_rolling_*`, `kyle_lambda_dispersion_window` | Shape / trailing inference of **one** λ series at a time |

Soft-verify: `kyle_lambda_ofi_depth_corr_honesty_errors` vs `kyle_lambda_dispersion_honesty_errors` — separate helpers. research_only — never live Sharpe.

### Never-equate rules

| Pair | Why |
|---|---|
| `kyle_lambda_ofi_depth_spearman` ≠ depth/ofi dispersion p50 / IQR | Cross-flow λ association ≠ within-flow λ quantile |
| `kyle_lambda_ofi_depth_prod_hac_t` ≠ `kyle_lambda_*_rolling_hac_t` / full-series `_t` | Product HAC on aligned pair ≠ single-series λ HAC |
| `kyle_lambda_ofi_depth_n_dates` ≠ `kyle_lambda_depth_n_dates` / `kyle_lambda_ofi_n_dates` | **Intersection** of dates with finite λ on both flows ≠ either side alone |
| Corr ≠ `depth_flow_*` / `ofi_flow_*` | λ↔λ corr ≠ raw flow→Δmid IC |
| Corr ≠ residual_ofi / ofi_fwd | Different diagnostic families |
| Depth dispersion ≠ ofi dispersion | Still never mix sides (sibling section) |

Cross-ref: **Nest Kyle λ dispersion vs IC companions**; suite ofi_depth_corr row.


## Always-on `n_securities` vs nest `n_dates` (never equate)

Count axes differ across always-on vs nest — do not translate one into the other.

| Surface | Count keys | Counts what |
|---|---|---|
| Always-on Kyle | `kyle_ofi_n_securities` (and signed_volume path via finite name λ count) | Names with finite per-name OLS λ |
| Always-on panel AR | `ofi_lag1_n_securities`, `mid_lag1_n_securities` | Names with finite lag1 corr |
| Nest λ | `kyle_lambda_ofi_n_dates`, `kyle_lambda_depth_n_dates` | Dates with finite cross-section λ |
| Nest date-series dump | `kyle_lambda_date_series_n_ofi` / `_n_depth` | Length of in-memory λ series (usually matches nest n_dates) |
| Nest IC companions | `ofi_flow_*_n_dates`, `depth_flow_*_n_dates`, residual/fwd `*_n_dates` | Dates surviving IC min_names filters |
| Nest ofi↔depth corr | `kyle_lambda_ofi_depth_n_dates` | Shared dates only |

### Never-equate rules

| Pair | Why |
|---|---|
| `kyle_ofi_n_securities` ≠ `kyle_lambda_ofi_n_dates` | Names ≠ dates |
| Always-on `*_n_securities` ≠ any nest `*_n_dates` | Panel name axis ≠ nest date axis |
| Nest IC `*_n_dates` ≠ nest λ `*_n_dates` | IC filter may keep dates λ drops (or vice versa) |
| Corr aligned n ≠ single-flow n | Intersection ≤ each side |
| Dump `date_series_n_*` ≠ always-on n_securities | Series length ≠ name count |

Already noted in **Always-on kyle_ofi_lambda vs nest**; this is the count-axis polish.


## Nest `join_coverage` vs always-on `join_coverage` (never equate)

Same key name on two surfaces — **different fuse paths**. Soft-verify does not require numeric equality.

| Surface | Where stamped | How computed | Floor / fail-closed | Soft-verify |
|---|---|---|---|---|
| **Always-on** northset receipt | `bench_northset` after `attach_candle_book_features` | Asof PIT join coverage (`n_fused / n_candle_rows`); also literal col on fused rows | `book_join_coverage_floor` (config; external re-check); attach default 1.0 synth / 0.5 external when `min_join_coverage` None | `join_coverage_honesty_errors` on main blob; receipt may echo `book_join_coverage_floor` |
| **Nest** `receipt["kyle_ofi"]` | `fuse_bars_l2_kyle_frame` / `bench_kyle_ofi_fused` | Inner-style bars↔book join: `n_fused / n_bar_rows` + literal col | Nest `min_join_coverage` arg (default 1.0 synth / 0.5 external) | `kyle_ofi_join_coverage_honesty_errors` — reuse open-unit checks **on nest blob** + nonempty `book_source` |

### Never-equate rules

| Pair | Why |
|---|---|
| Nest `join_coverage` ≠ always-on `join_coverage` | Kyle fuse ≠ candle asof attach — rates can differ on the same bars/book |
| Nest coverage ≠ always-on `book_join_coverage_floor` | Floor is a threshold, not a rate; nest may not echo the always-on floor key |
| Nest coverage ≠ `mean_book_age_seconds` / `max_book_age_seconds` | Share ≠ lag (always-on age stamps) |
| Passing always-on floor ≠ nest honesty OK | Nest helper still requires nest `book_source` when nest coverage finite |
| Nest join ≠ SYNTHETIC / claim stamps | Coverage is a rate; provenance/claim are separate honesty axes |

Cross-ref: fuse map Honesty stamps row; ops join_coverage / book_age; suite join row.


## Kyle nest claim / `ic_method` ops cheat-sheet

Quick ops read for `verify-research` honesty on `receipt["kyle_ofi"]` (fanned by `kyle_ofi_nest_honesty_errors`). research_only — **never live Sharpe** / no H-row mint.

| Stamp | Required value | When checked | Helper |
|---|---|---|---|
| `research_only` | `true` | Any nest diagnostic marker / residual / dispersion / corr / date_series / claim umbrella | nest claim + per-family helpers |
| `claim` | exactly `research_diagnostic_only` | Same umbrella + residual/dispersion/corr/date_series | `kyle_ofi_nest_claim_honesty_errors` (+ family helpers) |
| `ic_method` | exactly `date_level_spearman_hac` when present | IC markers finite **or** method key present; missing method with IC markers → error | `kyle_ofi_ic_method_honesty_errors` |
| `family` | exactly `kyle_ofi` | Nest diagnostics present | `kyle_ofi_family_honesty_errors` |
| `hac_lags` | finite, ≥0, integer-valued if present | Optional; also rolling HAC lo≤hi | `kyle_ofi_hac_lags_honesty_errors` |
| SYNTHETIC / SYN* | `synthetic_lob` ⇔ `data_source=SYNTHETIC`; SYN* label rules | dgp / label stamped | synthetic_source + label_synthetic helpers |
| Forbidden | no sharpe/sortino/calmar/pnl/nav/live_pnl_claim tokens in keys | claim / residual / date_series paths | `_assert_no_forbidden` + honesty |

CLI dump panel (`--dump-lambda-series`) must carry `research_only` + `claim=research_diagnostic_only` per row — aligns with date_series honesty.

Cross-ref: **Kyle nest soft-verify suite**; OPS nest suite bullet.


## Nest `book_source` / `book_dgp` vs always-on provenance (never equate)

Same-looking keys on nest vs main northset receipt — **different fuse + label matrices**. Soft-verify nest SYN/dgp helpers do not assert equality with always-on stamps.

| Key | Always-on (`bench_northset`) | Nest (`bench_kyle_ofi_fused` / kyle fuse) |
|---|---|---|
| `book_source` | Panel `source[0]` or `synthetic_lob` (synth path) | From `fuse_bars_l2_kyle_frame` / `_stamp_book_honesty` on **kyle** book path |
| `book_dgp` | `synthetic_lob` or `vendor_panel:{book_source}` | Same vocabulary from kyle fuse stamps |
| `dgp` | **Family** `family_dgp` (may be `mixed_sources` / `empirical` / `book_dgp`) | Typically **equals nest `book_dgp`** (book-scoped) |
| `label` / `data_source` | Evidence matrix (`SYNTHETIC` / `MIXED_SYNTHETIC_DERIVED` / bar_source) — bars×book×session rules | Nest: `data_source=SYNTHETIC` iff `book_dgp==synthetic_lob` else `book_source`; `label` often SYN* / caller arg |
| `component_sources` | `bars` / `book` / session_* map | **Absent** on nest |
| Eligibility | `book_hypothesis_eligible` / `session_book_hypothesis_eligible` | **Absent** on nest (no H-mint) |

### Never-equate rules

| Pair | Why |
|---|---|
| Nest `book_source` ≠ always-on `book_source` | Independent fuse path; nest may re-stamp even when northset used the same panel |
| Nest `book_dgp` ≠ always-on `book_dgp` | Same — do not assume string identity across blobs |
| Nest `dgp` ≠ always-on `dgp` | Nest book-scoped vs family matrix (esp. `mixed_sources`) |
| Nest `data_source` / `label` ≠ always-on `label` / `data_source` | SYN* / synthetic_lob rule ≠ evidence_label matrix — **Nest SYN* vs always-on MIXED** |
| Nest provenance ≠ `component_sources.book` | Nest has no component map; always-on book leg is one component among bars/session |
| Nest `book_source` ≠ `session_book_source` | Daily/kyle book ≠ last-snap session source (already: session vs receipt book_source) |
| Fuse-frame `book_dgp=external_panel` ≠ receipt `vendor_panel:…` | Always-on frame vs receipt cousin — still ≠ nest keys |
| Passing SYNTHETIC honesty on nest ≠ always-on eligibility | Nest SYN helpers ≠ `book_hypothesis_eligible` |

Honesty: `kyle_ofi_synthetic_source_honesty_errors` / `kyle_ofi_label_synthetic_honesty_errors` on **nest** only. Cross-ref: Family dgp; claim/ic_method cheat-sheet; join_coverage never-equate.


## Nest `n_fused` / `n_scored` vs always-on sizing (never equate)

| Surface | Keys | Meaning |
|---|---|---|
| Always-on | `n_bars`, `n_fused`, `n_scored` | Raw bars height; asof-attach fused height; rows with finite `fwd_ret_1` for `_IC_FEATURES` |
| Nest | `n_fused`, `n_scored` | Kyle-fuse height; rows with finite `delta_mid` drop_nulls for nest scoring |

### Never-equate rules

| Pair | Why |
|---|---|
| Nest `n_fused` ≠ always-on `n_fused` | Inner kyle join ≠ asof PIT attach |
| Nest `n_scored` ≠ always-on `n_scored` | `delta_mid` finite filter ≠ `fwd_ret_1` IC sample |
| Either ≠ `kyle_lambda_*_n_dates` / IC `*_n_dates` | Row counts ≠ date counts |
| Either ≠ `kyle_ofi_n_securities` | Rows ≠ names |

Cross-ref: **Nest join_coverage vs always-on**; **Always-on n_securities vs nest n_dates**.


## Nest `min_names` vs always-on `min_names` (never equate blindly)

| Surface | Where it lives | Typical value |
|---|---|---|
| Config | `northset.min_names` (models default **5**) | Drives always-on `date_ic_series` / `_ic_col` and is **passed through** to nest when `include_kyle_ofi=true` |
| Always-on receipt | **No** top-level `min_names` echo today | Threshold is config-only on the main blob |
| Nest receipt | `receipt["kyle_ofi"]["min_names"]` | Echo of the arg to `bench_kyle_ofi_fused` |
| Standalone `dipcatcher kyle-ofi` | Hardcodes `min_names=3` in CLI | May **differ** from config `northset.min_names` |
| Helper defaults | `kyle_ofi.py` kwargs default **3**; `date_ic_series` default **5** | Defaults ≠ each other |

### Never-equate rules

| Pair | Why |
|---|---|
| Nest `min_names` ≠ “always-on min_names on receipt” | Main blob usually has **no** `min_names` key — compare to **config**, not a missing receipt field |
| Nest echo ≠ standalone CLI default | Nested-from-northset uses config; bare `kyle-ofi` CLI uses 3 |
| Equal numeric threshold ≠ equal samples | Same `min_names` still yields different `*_n_dates` across IC vs λ vs always-on features (different filters/targets) |
| Nest `min_names` ≠ `kyle_lambda_*_n_dates` / IC `*_n_dates` | Threshold ≠ realized date count |
| Helper kwdefault 3 ≠ config 5 | Reading source defaults without the call site lies |

When documenting a notebook: cite nest echo **and** `northset.min_names` / CLI path separately.


## Nest SYN* / `label` vs always-on `MIXED_SYNTHETIC_DERIVED` (never equate)

| Surface | Label / data_source rule |
|---|---|
| Always-on evidence matrix | Bars×book×session → `SYNTHETIC`, **`MIXED_SYNTHETIC_DERIVED`**, or empirical `bar_source` (see Family dgp / evidence label) |
| Nest from `bench_northset` | `label=` **`SYNTHETIC` if bars synthetic else `bar_source`** — does **not** apply the MIXED branch |
| Nest `data_source` | `SYNTHETIC` iff nest `book_dgp==synthetic_lob` else `book_source` |
| Standalone `kyle-ofi` CLI | Often forces `label="SYNTHETIC"` |
| Soft-verify SYN* | `kyle_ofi_label_synthetic_honesty_errors` when nest `label` uppercases to **SYN*** |

### Never-equate rules

| Pair | Why |
|---|---|
| Nest `label=SYNTHETIC` ≠ always-on `MIXED_SYNTHETIC_DERIVED` | Empirical bars + synth book → always-on MIXED; nest label path ignores MIXED and may still say SYNTHETIC from bars flag or CLI |
| Nest SYN* honesty pass ≠ always-on eligibility / MIXED semantics | Nest SYN helpers check dgp↔data_source; they do **not** mint or deny H22/H27 |
| Nest `data_source=SYNTHETIC` ≠ always-on `data_source` | Book-dgp rule ≠ evidence_label matrix |
| Always-on SYNTHETIC ≠ nest research_only claim | Family evidence label ≠ `claim=research_diagnostic_only` |
| Vendor nest `label=bar_source` ≠ always-on vendor `label` when sessions reconstruct | Always-on branch 2 may keep bar_source with `dgp=book_dgp`; nest still book-scoped `data_source` |

Cross-ref: **Nest book_source/book_dgp vs always-on provenance**; evidence eligibility; SYNTHETIC source helpers.


## Nest `hac_lags` vs always-on IC inference (never equate)

| Surface | Behavior |
|---|---|
| Nest | Optional `hac_lags` stamp; soft-verify finite ≥0 integer when present; helpers pass `hac_lags` into `mean_tstat` / `date_ic_series` |
| Always-on northset ICs | `_ic_col` → `date_ic_series` without an echoed `hac_lags` receipt key (library default inside metrics) |
| Nest from `bench_northset` | Currently calls `bench_kyle_ofi_fused(..., min_names=…)` **without** passing `hac_lags` → nest helpers use their default (`None` → metrics default) |

### Never-equate rules

| Pair | Why |
|---|---|
| Nest `hac_lags` ≠ always-on (absent key) | Do not invent an always-on receipt twin |
| Nest `hac_lags` ≠ `kyle_lambda_dispersion_window` | NW lag ≠ trailing λ window |
| Matching defaults ≠ matching t/p | Different samples/targets still diverge |

Cross-ref: dispersion vs IC; claim/ic_method cheat-sheet.


## Nest `book_panel_path` vs always-on `book_panel_path` (never equate blindly)

| Surface | Stamp | How set |
|---|---|---|
| Always-on | top-level `book_panel_path` | `str(northset.book_panel_path)` when config/CLI `--book` loads a panel; else **`None`** (synth LOB path) |
| Nest | `receipt["kyle_ofi"]["book_panel_path"]` | Arg to `bench_kyle_ofi_fused` — when nested from northset, **passed through** the same `book_panel_path_str`; standalone `kyle-ofi` may set its own CLI `--book` string or None |

Path is an **audit echo** of which panel file was intended — the nest still fuses the in-memory `book` frame via kyle fuse (not a second load from that string inside the bench).

### Never-equate rules

| Pair | Why |
|---|---|
| Matching path strings ≠ matching `join_coverage` / `n_fused` | Same panel path, different fuse (asof vs kyle inner) |
| Nest path ≠ always-on path when CLI paths differ | Standalone `kyle-ofi --book` can diverge from config used for a prior northset run |
| `book_panel_path` ≠ `book_source` / `book_dgp` | File path ≠ provenance tags (`synthetic_lob` / `vendor_panel:…`) |
| `None` path ≠ “no book” | Synth path has book in memory with `book_panel_path=None` |
| Path present ≠ shape columns ensured | Always-on `shape_columns_ensured` is false when external path set; nest does not stamp that key |
| Path ≠ `min_join_coverage` / floor | External path gates whether floor is passed; path string is not the floor |

Cross-ref: External book_panel_path vs ensure; nest join_coverage; nest provenance.



## Kyle nest — do-not-invent backlog (parking lot)

**Status: IDLE inventing.** Docs-only lane does **not** invent never-equates, H-ids, or receipt means. Reopen only when the trigger is real.

| # | Gap (do not invent ahead of) | Reopen when |
|---|---|---|
| 1 | New nest receipt diagnostic keys | Sergeant/CoS land keys not yet in the soft-verify suite / never-equate index |
| 2 | Nest soft-verify **H-row** mint | Explicit productization (new kyle_* H-ids + agent mint + specs + ADR) — honesty suite ≠ H-mint |
| 3 | Always-on receipt echoes of `min_names` / `hac_lags` | Main blob starts stamping those keys (today config-only / absent) |
| 4 | Nest **loads** panel from `book_panel_path` string | Path stops being audit-echo-only and drives I/O inside nest bench |
| 5 | `include_kyle_ofi=false` but bare kyle keys on main blob | Should fail-closed today; document if behavior changes |

Cross-ref: **Kyle nest soft-verify suite**; OPS same backlog. Prior never-equate index remains valid — do not expand it speculatively.


## Always-on `kyle_r2` / `kyle_ofi_r2` vs nest HAC t/p (never equate)

| Surface | Keys | Estimator |
|---|---|---|
| Always-on | `kyle_r2`, `kyle_ofi_r2` | Per-name OLS fit R² (`kyle_lambda` helper) then **name-mean**; companions `kyle_lambda` / `kyle_ofi_lambda`, `*_n_securities` |
| Nest full-series | `kyle_lambda_depth_t`/`_p`, `kyle_lambda_ofi_t`/`_p` | HAC (Newey–West-style) t/p on the **mean of date-level λ** |
| Nest rolling | `kyle_lambda_*_rolling_hac_t`/`_p` | Same mean_tstat on **trailing** λ window only |

### Never-equate rules

| Pair | Why |
|---|---|
| `kyle_r2` / `kyle_ofi_r2` ≠ nest `kyle_lambda_*_t` / `_p` | Goodness-of-fit R² ≠ significance of date-λ mean |
| Always-on R² ≠ nest rolling HAC t/p | Name-mean fit quality ≠ trailing λ inference |
| Nest `_t`/`_p` ≠ Spearman `*_flow_*_t`/`_p` | λ-series HAC ≠ IC companion t/p |
| `kyle_ofi_r2` ≠ `kyle_lambda_ofi_mean` | R² ≠ λ level |
| No nest R² twin | Nest does **not** stamp an R² analogue of always-on `kyle_*_r2` |

Already flagged in **Always-on kyle_ofi_lambda vs nest kyle_lambda_ofi_mean**; this section is the R²/HAC-focused polish.


## Always-on `mid_lag1_corr` vs `ofi_lag1_corr` (never equate)

Both from `_panel_lag1` on the always-on candle↔book fuse: per-`security_id` `lag1_corr(series)` then name-mean. **Same estimator shape, different series.**

| Key | Series | Companion n |
|---|---|---|
| `mid_lag1_corr` | `mid` | `mid_lag1_n_securities` |
| `ofi_lag1_corr` | `ofi` | `ofi_lag1_n_securities` |

### Never-equate rules

| Pair | Why |
|---|---|
| `mid_lag1_corr` ≠ `ofi_lag1_corr` | Mid path AR ≠ OFI path AR — do not report one as the other |
| `mid_lag1_n_securities` ≠ `ofi_lag1_n_securities` | Finite-name counts can differ (missing ofi vs mid) |
| Either ≠ nest `ofi_delta_mid_lag1_*` / `signed_depth_delta_mid_lag1_*` | Panel AR ≠ nest date IC vs Δmid |
| Either ≠ always-on `ofi_lag_*_ic` / `ofi_p_ic` | Autocorr ≠ predictive IC |
| Either ≠ nest Kyle λ / depth_flow | Panel always-on ≠ nest diagnostic |

Cross-ref: **Always-on ofi_lag panel vs nest ofi_delta_mid_lag1**.


## Always-on `ofi_lag` panel vs nest `ofi_delta_mid_lag1` (never equate)

Three “lag-1 ofi” surfaces — different objects.

| Surface | Keys | What it is |
|---|---|---|
| Always-on **fuse col** | `ofi_lag` | `ofi.shift(1).over(security_id)` on candle↔book fuse; also in `_IC_FEATURES` → `ofi_lag_*_ic` vs **`fwd_ret_1`** |
| Always-on **panel lag-1 corr** | `ofi_lag1_corr`, `ofi_lag1_n_securities` | Per-name `lag1_corr(ofi)` then name-mean (`_panel_lag1`) — **autocorrelation of ofi**, not IC vs mid/return |
| Nest **lag1 Δmid IC** | `ofi_delta_mid_lag1_*` | `ofi_delta_mid_date_ic(lag=1)`: lagged ofi → contemporaneous **`delta_mid`**, date Spearman + HAC on kyle fuse |

### Never-equate rules

| Pair | Why |
|---|---|
| `ofi_lag1_corr` ≠ `ofi_delta_mid_lag1_*` | Name-mean AR(1)-style corr ≠ date-level IC vs Δmid |
| `ofi_lag_*_ic` ≠ `ofi_delta_mid_lag1_*` | Always-on lagged ofi → `fwd_ret_1` discovery IC ≠ nest lagged ofi → `delta_mid` |
| Fuse col `ofi_lag` ≠ nest lag1 keys | Column on always-on fuse; kyle fuse has **no** `ofi_lag` column (lag applied inside helper) |
| `ofi_lag1_n_securities` ≠ `ofi_delta_mid_lag1_n_dates` | Names with finite lag1 corr ≠ dates in nest IC |
| Always-on ofi_lag surfaces ≠ `ofi_flow_delta_mid_*` / `ofi_fwd_*` | Panel/fuse discovery ≠ nest Kyle companions / predictive |

Cross-ref: fuse map (no `ofi_lag` on kyle fuse); **Nest ofi_delta_mid_lag0/lag1 vs ofi_flow**; Session OFI vs bar-level.


## Nest `ofi_fwd_*` vs always-on `ofi_p_ic` (never equate)

| Surface | Keys | Fuse | Predictor → target | Product |
|---|---|---|---|---|
| **Always-on** northset receipt | `ofi_p_ic` (+ mean/t/n companions from `_ic_col`) | candle↔book `attach_candle_book_features` | raw `ofi` → **`fwd_ret_1`** (close-path fwd on always-on fuse) | Catalog discovery → **H27_northset_ofi** when finite + `book_hypothesis_eligible` |
| **Nest** `receipt["kyle_ofi"]` | `ofi_fwd_delta_mid_*`, `ofi_fwd_ret_{1,2,3}_*` | `fuse_bars_l2_kyle_frame` | raw `ofi` → `fwd_delta_mid` / mid- or close-stamped `fwd_ret_*` | research_only diagnostic; **no** kyle H-row mint |

### Never-equate rules

| Rule | Detail |
|---|---|
| `ofi_fwd_ret_1_*` ≠ `ofi_p_ic` | Nest date IC on kyle fuse ≠ always-on `_IC_FEATURES` IC — even when both say ofi→fwd return |
| `ofi_fwd_delta_mid_*` ≠ `ofi_p_ic` | Mid-space fwd ≠ always-on close `fwd_ret_1` discovery score |
| Nest ofi_fwd ≠ H27 soft-verify gate | Finite nest keys do **not** require H27; H27 keys off always-on `ofi_p_ic` only |
| Nest ofi_fwd ≠ `ofi_flow_delta_mid_*` / lag* / residual_* | Already covered in sibling never-equate sections |
| Always-on `kyle_ofi_lambda` ≠ nest ofi_fwd | Name-mean OLS λ ≠ Spearman IC |

Eligibility: H27 skipped when `book_hypothesis_eligible` is false; nest honesty still may run when nest present. Cross-ref: H-row finite gates; **Kyle nest soft-verify suite**.


## CLI surfaces for `session_l2_identity_floor`

`enforce_session_l2_identity_floors` is **not** part of `dipcatcher doctor` (doctor = config / dirs / manifest / import sanity only).

| Command | Surfaces the floor? | How |
|---|---|---|
| `dipcatcher northset` | **Yes** (when `northset.use_session_l2`) | `bench_northset` → `enforce_session_l2_identity_floors`; fail-closed `ValueError` before receipt write; stamps `session_l2_identity_gate` |
| `dipcatcher session-book` | **Yes** (always on this command) | After SYNTHETIC session L2 write: computes identity rates then calls `enforce_session_l2_identity_floors` with config floor; echoes rates + `gate=enforced` |
| `dipcatcher doctor` | **No** | Does not load session books or run identity rates |
| `dipcatcher verify-research` | **No live floor** | Post-hoc catalog/consistency on a saved notebook; may require finite identity *receipt fields* for some H-rows, but does not re-run the floor gate |

Config: `northset.session_l2_identity_floor` (default 0.99, ∈ [0,1]). Ops notes: OPERATIONS_RUNBOOK.



## `dipcatcher doctor`: `research_receipt` vs `shape_floors` split

Two different surfaces — do not conflate:

| Surface | Who | What | Gates exit code? |
|---|---|---|---|
| **`research_receipt`** | `pipeline.doctor` status key | Presence + soft-verify of `data/metadata/research/latest.json` via `verify_research_artifact` (`ok` / `missing` / `invalid`) | **Yes** — `healthy` requires `research_receipt == ok` (with dirs, manifest, core_imports, live≠REJECTED) |
| **`northset.shape_floors` echo** | CLI `doctor` after printing `info` | Config-only dump of `depth_shape_finite_floor`, `concentration_top_finite_floor`, `queue_priority_finite_floor`, `side_notional_finite_floor` (+ note that `shape_columns_ensured` is a **receipt** stamp when synth ensure runs) | **No** — informational; does not compute live rates or fail on NaN floors |

`shape_columns_ensured` is stamped on SYNTHETIC panel/session CLI paths when ensure runs — not a doctor health key. Live rate fail-closed floors run inside `bench_northset` / `dipcatcher northset`, not doctor.


## `verify-research` Northset H-row finite gates (post-hoc)

Soft notebook consistency in `research.catalog` / `research.verify` — **not** the live `session_l2_identity_floor` gate. Rule: if the northset family blob exposes a **finite** metric key → matching hypothesis row must exist (correct family). Non-finite / missing → skip. Not a live capital / promotion gate.

| Receipt field (finite → require H-row) | Hypothesis | Family | Extra gate |
|---|---|---|---|
| `ohlc_identity_rate` | H20_northset_ohlc | bound | — |
| `book_uncrossed_rate` | H21_northset_book | bound | — |
| `imbalance_top_p_ic` | H22_northset_imbalance | discovery | skip if `book_hypothesis_eligible` is false |
| `session_reconstructs_daily_rate` | H23_northset_session | bound | — |
| `session_volume_conservation_rate` | H24_northset_volume | bound | — |
| `microprice_p_ic` | H25_northset_microprice | discovery | skip if `book_hypothesis_eligible` is false |
| `wick_skew_p_ic` | H26_northset_wick | discovery | — |
| `ofi_p_ic` | H27_northset_ofi | discovery | skip if `book_hypothesis_eligible` is false |
| `dm_gk_vs_park_p` | H28_northset_gk | discovery | — |
| `session_chain_rate` | H29_northset_chain | bound | — |
| `clv_p_ic` | H30_northset_clv | discovery | — |
| `dm_split_vs_park_p` | H31_northset_overnight | discovery | — |
| `vpin_p_ic` | H32_northset_vpin | discovery | skip if `book_hypothesis_eligible` is false |
| `sweep_reject_signed_p_ic` | H33_northset_sweep_reject | discovery | — |
| `sweep_follow_signed_p_ic` | H34_northset_sweep_follow | discovery | — |
| `sweep_reject_event_p` | H35_northset_reject_event | discovery | — |
| `sweep_follow_event_p` | H36_northset_follow_event | discovery | — |
| `sweep_reject_placebo_p` | H37_northset_reject_placebo | discovery | — |
| `sweep_follow_placebo_p` | H38_northset_follow_placebo | discovery | — |
| `sweep_reject_cost_adjusted_mean_bps` | H39_northset_reject_cost | bound | — |
| `sweep_follow_cost_adjusted_mean_bps` | H40_northset_follow_cost | bound | — |
| `sweep_reject_fold_positive_fraction` | H41_northset_reject_stability | bound | — |
| `sweep_follow_fold_positive_fraction` | H42_northset_follow_stability | bound | — |
| `session_book_vpin_p_ic` | H43_northset_session_book_vpin | discovery | skip if `session_book_hypothesis_eligible` is false |

**Post-hoc vs live floor**

- Live (`dipcatcher northset` / `session-book`): `enforce_session_l2_identity_floors` fail-closes when session L2 on if identity **rates** &lt; `session_l2_identity_floor` (ohlc / session_ohlc / reconstruct / vol_cons / chain / book_uncrossed).
- Post-hoc (`verify-research`): does **not** re-apply that floor; only checks finite receipt → H-row minting. Nested `receipt["kyle_ofi"]` keys **must not** enter this soft-verify table unless/until explicitly productized (research_only nest stays diagnostic).


## Evidence eligibility stamps (`book_hypothesis_eligible` / `session_book_hypothesis_eligible`)

Set only by `bench_northset` in `northset.benches` (receipt honesty metadata — not alpha).

### Who sets them

| Flag | Expression | Inputs |
|---|---|---|
| `bars_synthetic` | `config.data.source.lower() == "synthetic"` | bars provenance |
| `book_synthetic` | `book_source.lower()` ∈ `{synthetic, synthetic_lob, synthetic_reconstruction}` | book panel `source` or synthesizer tag |
| `book_hypothesis_eligible` | `bars_synthetic or not book_synthetic` | true when bars are SYNTHETIC **or** book is vendor/external |
| `session_book_hypothesis_eligible` | `bars_synthetic` | true **only** when bars are SYNTHETIC |

Related: `component_sources` (`bars` / `book` / `session_candles=synthetic_reconstruction` / `session_book=synthetic_reconstruction|disabled`); `sweep_evidence_scope` (`synthetic` \| `empirical_adjusted` \| `fixture_raw_unadjusted`).

### When each is false (truth table)

| Bars | Book | `book_hypothesis_eligible` | `session_book_hypothesis_eligible` | Intent |
|---|---|---|---|---|
| SYNTHETIC | SYNTHETIC LOB | **true** | **true** | Lab path — discovery OK on synth |
| SYNTHETIC | vendor/external | **true** | **true** | Synth bars + real book still eligible |
| empirical | vendor/external | **true** | **false** | Real bars + real book: book hyps OK; session-book VPIN **not** discovery-eligible (session L2 is reconstructed plumbing) |
| empirical | SYNTHETIC LOB | **false** | **false** | Real bars + synthetic book → do **not** mint book discovery/bound hyps as empirical evidence (`MIXED_SYNTHETIC_DERIVED` label path) |

### Consumers (mint + soft-verify)

**Agent mint** (`research.agent` northset H-table): when the flag is false, treated as “metric absent” — H-row not minted even if the scalar is finite.

| Flag false → skip mint | Hypotheses |
|---|---|
| `book_hypothesis_eligible` | H21 (book uncrossed bound), H22 (imbalance), H25 (microprice), H27 (OFI), H32 (VPIN) |
| `session_book_hypothesis_eligible` | H43 (session_book_vpin) |

**Not gated by these flags** (mint on finite metric alone): H20 OHLC, H23/H24/H29 session reconstruct/volume/chain, H26 wick, H28 GK DM, H30 CLV, H31 overnight DM, H33–H42 sweep family.

**Soft verify-research** (`research.catalog`): same eligibility skips — finite field does **not** require the H-row when the matching flag is false (`northset_has_finite_*` / `NORTHSET_H23_H28_SPECS` book_metrics set `{microprice_p_ic, ofi_p_ic, vpin_p_ic}` + H21/H22/H43 helpers). Default if key missing: treat as eligible (`True`).


## Nested `kyle_ofi` soft-verify / H-row stance

**H-row mint:** still **not** productized — nest keys must **not** enter finite→H catalogs (no kyle_* hypotheses).

**Honesty suite:** **live** via `kyle_ofi_nest_honesty_errors` on northset when nest present — see **Kyle nest soft-verify suite** below.

- Nest remains diagnostic under `include_kyle_ofi` (default false); must stamp `research_only` + `claim=research_diagnostic_only`.
- Always-on scalars `kyle_lambda` / `kyle_ofi_lambda` are **not** soft-verify H sources.



## CoS `northset_receipt_honesty_errors` / `NORTHSET_RECEIPT_HONESTY_HELPERS`

**Live catalog count: 40** helpers in `NORTHSET_RECEIPT_HONESTY_HELPERS` (`research.catalog`, tuple + dispatcher at **EOF**).

### Placement rule (Lt NameError fix)

All helper **functions must be defined before** the tuple and `northset_receipt_honesty_errors` dispatcher. The tuple holds bare callables — defining it mid-module before helpers exist raises `NameError` at import. CoS/Lt keep tuple+dispatcher at catalog EOF after every helper body.

### Dispatcher contract

| Symbol | Role |
|---|---|
| `NORTHSET_RECEIPT_HONESTY_HELPERS` | Ordered tuple of 40 soft-verify callables |
| `northset_receipt_honesty_errors(blob)` | Fans every helper; concatenates error strings |
| `verify-research` | Calls dispatcher on family **`northset`** (`research.verify`) |

**Out of this dispatcher (separate lanes):**

| Lane | Dispatcher / wire | Count / note |
|---|---|---|
| Session path means | `northset_session_means_honesty_errors` | Separate fan |
| Kyle nest | `kyle_ofi_nest_honesty_errors` | **26** fans (not 40) |
| Direct verify wires | e.g. `mean_tob_notional_share_honesty_errors` | Sergeant free-lane — **not** in the 40-tuple today; called from `verify.py` on northset |

research_only — **never live Sharpe**. Do not equate the 40-count with kyle’s 26.

### Helper inventory (live tuple order)

| `mean_book_age_seconds_honesty_errors` |
| `book_age_seconds_honesty_errors` |
| `mean_microprice_minus_mid_honesty_errors` |
| `northset_log_size_slope_honesty_errors` |
| `northset_qlike_means_honesty_errors` |
| `northset_range_spread_honesty_errors` |
| `amihud_mean_honesty_errors` |
| `northset_queue_sweep_ofi_honesty_errors` |
| `depth_shape_finite_rate_honesty_errors` |
| `northset_overnight_rv_semi_honesty_errors` |
| `northset_book_shape_finite_rates_honesty_errors` |
| `session_volume_conservation_rate_honesty_errors` |
| `session_reconstructs_daily_rate_honesty_errors` |
| `book_uncrossed_rate_honesty_errors` |
| `ohlc_identity_rate_honesty_errors` |
| `session_chain_rate_honesty_errors` |
| `session_mean_jump_ratio_honesty_errors` |
| `sweep_follow_signed_mean_ic_honesty_errors` |
| `sweep_reject_signed_mean_ic_honesty_errors` |
| `sweep_reject_event_mean_bps_honesty_errors` |
| `sweep_follow_event_mean_bps_honesty_errors` |
| `sweep_follow_cost_adjusted_mean_bps_honesty_errors` |
| `sweep_reject_cost_adjusted_mean_bps_honesty_errors` |
| `northset_n_bars_scored_honesty_errors` |
| `ofi_p_ic_honesty_errors` |
| `northset_vpin_sweep_fold_honesty_errors` |
| `microprice_p_ic_honesty_errors` |
| `clv_p_ic_honesty_errors` |
| `ofi_mean_ic_honesty_errors` |
| `imbalance_top_mean_ic_honesty_errors` |
| `book_hypothesis_eligible_honesty_errors` |
| `northset_lag_corr_and_sweep_count_honesty_errors` |
| `gap_finite_rate_honesty_errors` |
| `vpin_mean_honesty_errors` |
| `northset_spread_means_honesty_errors` |
| `northset_session_identity_rates_honesty_errors` |
| `northset_half_spread_honesty_errors` |
| `northset_microprice_weight_balance_honesty_errors` |
| `northset_spread_bps_honesty_errors` |
| `northset_spread_receipt_honesty_errors` |

(Table is name-only; individual contracts stay in each helper docstring / sibling DATA_CONTRACTS sections.)


## Sergeant verify wire: `northset_receipt_honesty_errors` fan-in

**Status:** live in `research.verify` on family **`northset`**.

```text
for receipt_err in northset_receipt_honesty_errors(families.get("northset")):
    errors.append(receipt_err)
```

| Fact | Detail |
|---|---|
| Fan size | **40** helpers via `NORTHSET_RECEIPT_HONESTY_HELPERS` |
| Reach | All 40 now execute on every `verify-research` northset pass |
| Fan-in delta | **~28** helpers newly reach verify-research **only** through this dispatcher (were catalog-defined but not previously individually looped in `verify.py`) |
| Post-dedupe directs | Sergeant removed **duplicate** individual northset calls for helpers already in the 40. Remaining northset directs are **outside** the tuple (tob/queue/depth imbalance/…). Candle_order_book-only wires kept. See **Sergeant verify dedupe** |
| Placement | Tuple + dispatcher at catalog **EOF**; helpers defined first |
| Out of band | Kyle nest (`kyle_ofi_nest_honesty_errors`, 26); session-means fan; candle_order_book structure rates |

research_only — **never live Sharpe**. Cross-ref: **CoS northset_receipt_honesty_errors / NORTHSET_RECEIPT_HONESTY_HELPERS**.


## Sergeant verify dedupe: single northset receipt fan-in

**Live `research.verify` layout (post-Sergeant):**

| Lane | What runs | Notes |
|---|---|---|
| **Northset receipt fan-in** | `northset_receipt_honesty_errors(northset)` → **40** helpers | Single entry for gap/vpin/ohlc/spread/sweep+IC/book-age/… that live in `NORTHSET_RECEIPT_HONESTY_HELPERS` |
| **Removed duplicates** | Prior **individual** northset loops for helpers already in the 40 (e.g. gap_finite_rate / vpin_mean / ohlc_identity / …) | Dedupe — do not re-add one-off northset calls for tuple members |
| **Kept northset directs** | Helpers **outside** the 40 (e.g. tob size/notional, queue, depth imbalance, close_mid, …) | Still explicit loops until folded into the tuple |
| **Candle_order_book-only wires** | `structure_finite_rate_honesty_errors`, `depth_shape_finite_rate_honesty_errors`, `mean_microprice_minus_mid_honesty_errors`, `book_age_seconds_honesty_errors`, candle MWB | Kept — not replaced by the northset fan-in |
| **Other northset fans** | session means; kyle nest (26); top-level claim; shape/session_l2 floors | Unchanged parallel lanes |

research_only — **never live Sharpe**. Cross-ref: Sergeant fan-in (~28 new); CoS 40-helper inventory.


## Sergeant CLI rate echoes (`dipcatcher northset`)

Identity / reconstruct / structure rates on the northset receipt summary (never equate ohlc vs session_ohlc):

```text
ohlc_identity_rate=…
session_ohlc_identity_rate=…
book_uncrossed_rate=…
session_chain_rate=…
session_reconstructs_daily_rate=…
session_volume_conservation_rate=…
structure_finite_rate=…
```

| Key | Soft-verify / H notes |
|---|---|
| `ohlc_identity_rate` | ∈[0,1]; H20 when finite |
| `session_ohlc_identity_rate` | ≠ daily ohlc identity |
| `book_uncrossed_rate` | H21 when finite + eligible |
| `session_chain_rate` | H29 |
| `session_reconstructs_daily_rate` | H23; ≠ session_ohlc |
| `session_volume_conservation_rate` | H24 |
| `structure_finite_rate` | Stamped: northset = nanmean of concentration/queue/side_notional/tob finite rates; candle = nanmean of `finite_rate_*` companions. CLI echoes both. Distinct surfaces — never equate northset aggregate with candle companions. |
| Never-equate test | `tests/unit/test_structure_finite_rate_never_equate_surfaces.py` |

Also: research compact line still uses short aliases (`ohlc_ok` / `book_ok` / `chain_ok` / …). `session-book` CLI echoes live rates at gate time separately.


## Scoped northset CLI echo required + BLOB_ONLY (Sergeant)

| Frozenset | Role |
|---|---|
| `NORTHSET_CLI_ECHO_REQUIRED` | Core rates/structure/session/spread keys that must appear in `dipcatcher northset` CLI source (`benches.py`) |
| `NORTHSET_RECEIPT_BLOB_ONLY` | Explicit allowlist of stamped keys that may stay blob-only (DM stats, sweep p/t, floors, provenance, …) |
| `NORTHSET_CLI_ECHO_EXTRA` | Echoed companions not in REQUIRED (means/spreads/sweep CLI lines, …) |
| `NORTHSET_RECEIPT_CLASSIFIED` | REQUIRED ∪ EXTRA ∪ BLOB_ONLY |
| Soft-verify | `northset_receipt_key_classification_honesty_errors` — fail-closed on silent new stamps; wired on northset in verify-research |
| Invariant | Every stamped non-discovery / non-kyle key is CLI-echoed **or** blob-only; REQUIRED ∩ BLOB_ONLY = ∅ |
| Test | `tests/unit/test_northset_cli_echo_required_blob_only.py` |

Aliases: `mean_book_age_seconds` → `mean_book_age_s`, `max_book_age_seconds` → `max_book_age_s`. Discovery IC companions and `kyle_*` out of scope. Distinct from CoS mean_* 54/54 completeness.

## CoS `mean_*` CLI echo completeness — **54/54**

**Status:** CoS CLI mean_* echo batch complete (**54/54**; missing 0) — stamped northset `mean_*` companions (ex jump intermediates) are echoed on `dipcatcher northset`.

| Fact | Detail |
|---|---|
| Completeness | **54/54** receipt `mean_*` keys echoed (CoS stamp wave + CLI polish) |
| Surfaces | Shape/notional/imbalance/queue/slope/tick/spread/microprice/session means lines on `dipcatcher northset` |
| Exclusions | Jump intermediate keys / non-`mean_*` rates stay on other echo lines |
| Ops | Prefer CLI + receipt blob together; never invent a mean as echoed if not in live CLI |
| Test | `tests/unit/test_northset_cli_mean_echo_priority_batch.py` |

Cross-ref: NORTHSET **receipt means — CoS overnight stamp wave**; spread / tob / queue CLI sections.


## CoS: `mean_excess_bps` / `mean_diff_bps` → sweep CLI echoes

Nested sweep research rows carry raw bps fields; northset **flattens** them onto the receipt, and `dipcatcher northset` echoes the flat keys (not the nested names).

| Nested source | Receipt / CLI key | Path |
|---|---|---|
| event_studies `mean_excess_bps` (horizon=1) | `sweep_reject_event_mean_bps` / `sweep_follow_event_mean_bps` | `bench_northset` ← `row["mean_excess_bps"]` |
| matched_controls `mean_diff_bps` | `sweep_reject_control_diff_mean_bps` / `sweep_follow_control_diff_mean_bps` | ← `control["mean_diff_bps"]` |
| liquidity_matched_controls `mean_diff_bps` | `sweep_reject_liq_control_diff_mean_bps` / `sweep_follow_liq_control_diff_mean_bps` | lagged-dvol quartile |
| name_clustered / oot_holdouts / adv_participation | `sweep_follow_name_cluster_*`, `sweep_follow_oot_holdout_mean_bps`, `sweep_median_event_adv_participation` | H48/H49 + crowding disclosure |

### Never equate

| Pair | Why |
|---|---|
| `mean_excess_bps` ≠ `*_event_mean_bps` as dual receipt keys | Nested-only name; flat receipt uses `sweep_*_event_mean_bps` |
| `mean_diff_bps` ≠ `*_control_diff_mean_bps` naming | Same — control nested → flat `sweep_*_control_diff_mean_bps` |
| event_mean_bps ≠ control_diff_mean_bps | Excess vs matched event−control difference |
| event_mean_bps ≠ cost_adjusted_mean_bps | Raw event excess ≠ cost-adjusted |

CLI: CoS echoes both event + control_diff flat keys on the northset residual honesty line. Tests: `tests/unit/test_cli_mean_diff_excess_and_candle_gaps.py`. research_only — **never live Sharpe**.


## CoS candle-book CLI: `n_fused` + `min_names`

`dipcatcher candle-book` (`bench_candle_order_book`) now echoes:

```text
n_fused=… min_names=…
```

alongside `n_bars` / `n_scored` / join / book_source / means / CoS-stamped `finite_rate_*`.

| Key | Meaning |
|---|---|
| `n_fused` | Fused candle↔book row count on the candle_order_book receipt |
| `min_names` | Date-IC threshold used by the candle bench (default **4**) |

**Never equate** candle `n_fused` / `min_names` with northset or kyle nest twins (different fuse / defaults). CoS `finite_rate_*` stamps landed — see **CoS candle-book finite_rate_* stamps**; `structure_finite_rate_honesty` bites on synth.


## Commander residual #141–#150 (Mac property continuous)

Lt/Commander Mac property lane (off box merge ownership). Themes in #141–#150:

- OPTIONAL metrics not ±inf; queue bounds; tob shares ∈(0,1]; `n_levels`
- effective / quoted / touch aliases; `half_spread_bps`
- bid/ask `mean_log_tick_spacing` formulas

Docs: cross-link only after locks land in `book_metrics` / property tests — do not invent formulas ahead of Commander. Adjacent batches #151+ continue microprice/mid/spread/imbalance/queue/tob field-group locks.



## CoS candle-book `finite_rate_*` stamps — `structure_finite_rate_honesty` bites

**Live:** `bench_candle_order_book` stamps structure companions via `_finite_rate_column`:

| Receipt key | Column / gate |
|---|---|
| `finite_rate_microprice_minus_mid` | fraction finite `microprice_minus_mid` |
| `finite_rate_bid_size_concentration_top` | finite concentration among rows with `bid_depth` > 0 |
| `finite_rate_ask_size_concentration_top` | finite concentration among rows with `ask_depth` > 0 |

Also still stamps `depth_shape_finite_rate` (separate helper).

| Soft-verify | Behavior |
|---|---|
| `structure_finite_rate_honesty_errors` on family **`candle_order_book`** | Each present `finite_rate_*` key ∈ **[0, 1]** when finite; NaN skip; ±inf fail-closed |
| Effect | On SYNTHETIC candle receipts the keys are present + finite → helper **bites** (no longer a no-op) |
| Tests | `tests/unit/test_candle_book_structure_finite_rate_stamp.py` |
| CLI | `dipcatcher candle-book` echoes the three `finite_rate_*` keys |

research_only — **never live Sharpe**. Distinct from northset CLI `structure_finite_rate=` (single aggregate echo on northset receipt).


## Sergeant candle-book CLI: `family=` + non-ic parity guard

| Piece | Live |
|---|---|
| CLI echo | `dipcatcher candle-book` prints `family={receipt family}` (e.g. `candle_order_book`) on the summary line with n_bars / n_fused / min_names / join / finite_rate_* / … |
| Parity guard | `test_candle_book_cli_echoes_all_non_ic_receipt_keys` in `tests/unit/test_cli_mean_diff_excess_and_candle_gaps.py` |
| Rule | Every **non-`ic_*`** key stamped by `bench_candle_order_book` on a synth receipt must appear in the `candle_book` CLI source (`key=` / `get('key')` / `['key']`) |
| Sibling | `test_candle_book_cli_echoes_bench_mean_keys`, `test_candle_book_echoes_n_fused_min_names` |

Fails closed if a new bench stamp lands without a CLI echo — keep CLI and receipt aligned. research_only family (not in REQUIRED_BENCHMARK_FAMILIES).


## CoS / Sergeant candle-book — confirmed live (docs sync)

| Item | Status |
|---|---|
| CoS `finite_rate_microprice_minus_mid` + bid/ask size_concentration stamps | **Confirmed** on `bench_candle_order_book`; CLI echoes; `structure_finite_rate_honesty` bites on synth |
| Sergeant `family=` CLI + non-ic parity guard | **Confirmed** — `dipcatcher candle-book` echoes `family=`; `test_candle_book_cli_echoes_all_non_ic_receipt_keys` |
| Commander **#161–#175** | Mac property continuous (see section) — ahead of skipped box #73–#100 merge |

Cross-ref: **CoS candle-book finite_rate_***; **Sergeant candle-book family= + non-ic parity**; **Commander residual #161–#175**.









## Stamp-test wave — mean_* **COMPLETE** (0 remaining)

**Rule (Lt):** on-disk test file + pytest green → **GREEN**. Never keep stale INFLIGHT from an earlier assign.

**Stamp-test mean_* wave: COMPLETE.** No stamp-test INFLIGHT left.

### GREEN — Sergeant `structure_finite_rate` real stamp + derivation (**6/6**, NOT inflight)

| Path | Tests |
|---|---|
| `tests/unit/test_northset_finite_rates_receipt_stamp.py` | 4 — source stamps; synth unit-interval; candle structure+depth_shape; **northset structure derived from shape rates** |
| `tests/unit/test_candle_book_structure_finite_rate_stamp.py` | 2 — synth stamps; **structure_finite_rate == nanmean of `finite_rate_*` companions** |

**candle ≠ northset (never equate):**
- **candle:** nanmean of `finite_rate_microprice_minus_mid` + bid/ask `finite_rate_*_size_concentration_top`
- **northset:** nanmean of concentration / queue / side_notional / tob_size_share finite rates — **≠** candle companions

### GREEN — CoS `mean_session_*` batch (**4/4**, 11 keys) + `_nanmean` asarray

| Path | Status |
|---|---|
| `tests/unit/test_mean_session_means_receipt_stamp_batch.py` | **4/4 GREEN** |

11 receipt keys: `mean_session_book_snaps`, `mean_session_close_ask_depth`, `mean_session_close_bid_depth`, `mean_session_close_imbalance`, `mean_session_close_micro_bps`, `mean_session_close_mid`, `mean_session_close_spread_bps`, `mean_session_imbalance_mean`, `mean_session_imbalance_std`, `mean_session_ofi_abs_sum`, `mean_session_spread_bps_mean`.

**CoS `_nanmean` asarray fix:** `northset/benches.py` `_nanmean` uses `np.asarray(arr, dtype=float)` so list/tuple rate packs work (needed for northset `structure_finite_rate` aggregate).

### Also GREEN (close-out companions)

`test_northset_top_size_tr_spread_bps_receipt_stamp.py`; finite_rates gap/depth_shape; Lt spread means; fwd_ret_after; identity rates; slope/tick; levels; depth/MWB/notional/spread_over_mid; session_bv/rv/jump+sweep; size_concentration; not-receipt-keys.

### Soft-verify dual-family — **GREEN 11/11** (NOT inflight)

Sergeant `structure_finite_rate` dual-family soft-verify is **GREEN 11/11** (pytest-verified). Anchors include:
- `tests/unit/test_structure_finite_rate_soft_verify.py` (companions + aggregate + verify wire)
- `tests/unit/test_northset_structure_finite_rate_distinct_soft_verify.py` (northset ≠ candle companions)
- never-equate soft-verify cases in `test_structure_finite_rate_never_equate_surfaces.py`

**Never-equate surfaces:** `tests/unit/test_structure_finite_rate_never_equate_surfaces.py` **4/4 GREEN** (separate from the 11/11 soft-verify count).

### CLI partition + soft-verify — **GREEN** (no INFLIGHT here)

- Soft-verify `structure_finite_rate` dual-family **11/11 GREEN** (NOT inflight)
- CLI `NORTHSET_CLI_ECHO_REQUIRED` (**51**) + `BLOB_ONLY` (**90**) + disjoint **GREEN 4/4** — see **Northset CLI echo partition**

research_only — **never live Sharpe**. Box #73–#100 → Mac merge remains **SKIPPED**.



## Post–stamp-wave landings (Mac verified)

### Sergeant — soft-verify dual-family **11/11 GREEN** + never-equate surfaces **4/4 GREEN**

| Path | Status |
|---|---|
| Dual-family soft-verify suite | **11/11 GREEN** (NOT inflight) — `test_structure_finite_rate_soft_verify.py` + distinct + never-equate soft-verify cases |
| `tests/unit/test_structure_finite_rate_never_equate_surfaces.py` | **4/4 GREEN** — northset aggregate vs candle `finite_rate_*` companions; independent soft-verify per surface |

### Lt — candle `_FEATURE_COLS` **+5** structure LOB **2/2 GREEN**

| Path | `tests/unit/test_candle_book_structure_feature_cols.py` |
|---|---|
| Status | **2/2 GREEN** |
| +5 cols | `bid_size_concentration_top`, `ask_size_concentration_top`, `queue_priority_proxy`, `tob_size_share`, `notional_imbalance` |
| Source | `microstructure/bench.py` `_FEATURE_COLS` |

### CoS — microprice / notional / tob stamps + structure_finite distinct soft-verify

| Path | Status | Keys / role |
|---|---|---|
| `tests/unit/test_mean_microprice_notional_tob_receipt_stamp.py` | **3/3 GREEN** | `mean_microprice_minus_mid` (+ `_bps`), `mean_notional_imbalance`, `mean_tob_notional_share` |
| `tests/unit/test_northset_structure_finite_rate_distinct_soft_verify.py` | **4/4 GREEN** (part of dual-family 11/11) | Northset companions ≠ candle `finite_rate_*`; dispatcher wire |

### CLI partition — **GREEN** (was INFLIGHT)

Sergeant `NORTHSET_CLI_ECHO_REQUIRED` (**51**) + `NORTHSET_RECEIPT_BLOB_ONLY` (**90**) + disjoint contract — **GREEN 4/4**. See **Northset CLI echo partition**.

**No True INFLIGHT** in this post–stamp-wave lane (soft-verify dual-family already **11/11 GREEN**).

Off inventing. Stamp-test mean_* wave remains **COMPLETE**.


## Northset CLI echo partition — `NORTHSET_CLI_ECHO_REQUIRED` / `BLOB_ONLY`

**Status:** Sergeant **GREEN** (NOT INFLIGHT) — `test_northset_cli_echo_required_blob_only.py` **4/4** + `test_northset_cli_echo_required_finite_on_synth.py` **3/3**.

### Why full non-IC parity is wrong for northset

Candle-book can guard **all** non-IC stamped keys in CLI source (small surface). Northset receipts are a **large blob dump**: many stamped keys are diagnostics / provenance / DM stats / paths that must stay **receipt-only**, not `key=` CLI lines. A candle-style “every non-IC key must echo” rule would force noisy CLI spam or invent false ops signals.

**Design instead:** partition stamped non-discovery / non-kyle keys into:
1. **CLI-required** — must appear in `dipcatcher northset` CLI source (`key=` / `get('key')`, with `NORTHSET_CLI_ECHO_KEY_ALIASES` for age seconds → `_s`)
2. **BLOB_ONLY allowlist** — explicitly allowed to stay blob/receipt-only

CoS owns **mean_*** CLI echo completeness separately (54/54 wave) — not folded into this frozenset.

### Live sizes (Mac `benches.py`)

| Constant | n | Role |
|---|---|---|
| `NORTHSET_CLI_ECHO_REQUIRED` | **51** | literals ∪ `*SESSION_RECEIPT_KEYS` — core rates / structure / session / spread / shape |
| `NORTHSET_RECEIPT_BLOB_ONLY` | **90** | receipt-only allowlist (`book_dgp`, panel paths, DM keys, …) |

### Disjoint contract

| Invariant | Test |
|---|---|
| `REQUIRED ∩ BLOB_ONLY = ∅` | `test_required_and_blob_only_disjoint` |
| `SESSION_RECEIPT_KEYS ⊆ REQUIRED` | `test_session_receipt_keys_subset_of_required` |
| every REQUIRED key appears in northset CLI source | `test_required_keys_appear_in_northset_cli_source` |
| every stamped non-discovery/non-kyle key is CLI-classified or BLOB_ONLY | `test_stamped_keys_classified_cli_or_blob_only` |

research_only companions — **never live Sharpe**. Off inventing.

## CoS `candle_feature_cols_ic` soft-verify

| Path | Status |
|---|---|
| `catalog.candle_feature_cols_ic_honesty_errors` | wired in `verify.py` for candle family |
| `tests/unit/test_candle_feature_cols_ic_soft_verify.py` | **3/3 GREEN** |
| `tests/unit/test_candle_feature_cols_structure_ic_presence.py` | **2/2** presence / honesty on structure LOB IC keys |

Checks when present: `ic_*_p` ∈ [0,1]; `ic_*_n_dates` ≥ 0; signed IC / t / pearson finite-when-present (±inf fail-closed); `mean_abs_ic` ≥ 0; `best_feature_ic_key` empty or valid `ic_*` spearman key with matching |best| identity.

Pairs with Lt `_FEATURE_COLS` +5 structure LOB scoring. research_only — **never live Sharpe**.


## Sergeant REQUIRED finite-on-synth + CoS queue / FEATURE_COLS IC (Mac verified)

### Sergeant — CLI frozenset **GREEN** (NOT INFLIGHT)

| Path | Status |
|---|---|
| `tests/unit/test_northset_cli_echo_required_blob_only.py` | **4/4 GREEN** — REQUIRED **51** / BLOB_ONLY **90** / disjoint (see **Northset CLI echo partition**) |
| `tests/unit/test_northset_cli_echo_required_finite_on_synth.py` | **3/3 GREEN** — all `NORTHSET_CLI_ECHO_REQUIRED` present on synth; numeric finite; string keys nonempty |

### CoS — `queue_priority` stamp upgrade **GREEN**

| Path | Status | Keys |
|---|---|---|
| `tests/unit/test_mean_queue_priority_receipt_stamp.py` | **3/3 GREEN** | `mean_queue_priority_proxy`, `mean_ask_queue_priority_proxy` — source stamp + synth ∈[0,1] + honesty |

### CoS / Lt — candle FEATURE_COLS structure IC presence **5/5 GREEN**

| Path | n | Role |
|---|---|---|
| `tests/unit/test_candle_feature_cols_structure_ic_presence.py` | **2** | +5 structure LOB companions listed + synth scores `ic_*` keys |
| `tests/unit/test_candle_feature_cols_ic_completeness_soft_verify.py` | **3** | all `FEATURE_COLS` IC keys present; completeness helper + verify wire |

**5/5** = structure presence **2** + IC completeness **3**. Pairs with `candle_feature_cols_ic` honesty soft-verify.

### True INFLIGHT / IDLE

**Classify-or-echo / CLASSIFIED residual:** **GREEN** (`NORTHSET_RECEIPT_CLASSIFIED` **206** + soft-verify fail-closed). See **Northset receipt key CLASSIFIED partition**. **IDLE awaiting** next assign beyond that.

research_only — **never live Sharpe**. Off inventing.


## Northset receipt key CLASSIFIED partition (**206**)

**Status:** Sergeant **GREEN**.

### Union

| Set | n | Role |
|---|---|---|
| `NORTHSET_CLI_ECHO_REQUIRED` | **51** | scoped CLI must-echo (literals ∪ `SESSION_RECEIPT_KEYS`) |
| `NORTHSET_CLI_ECHO_EXTRA` | **65** | additional CLI-echo surface (not in REQUIRED) |
| `NORTHSET_RECEIPT_BLOB_ONLY` | **90** | explicit blob/receipt-only allowlist |
| `NORTHSET_RECEIPT_CLASSIFIED` | **206** | `REQUIRED ∪ EXTRA ∪ BLOB_ONLY` (pairwise disjoint) |

`REQUIRED ∩ EXTRA = ∅`, `REQUIRED ∩ BLOB_ONLY = ∅`, `EXTRA ∩ BLOB_ONLY = ∅` → **206 = 51+65+90**.

### Classification soft-verify (fail-closed)

| Piece | Detail |
|---|---|
| Helper | `northset.benches.northset_receipt_key_classification_honesty_errors` |
| Rule | every non-discovery / non-kyle receipt key must be in `NORTHSET_RECEIPT_CLASSIFIED`; unknown stamps → fail-closed |
| Wire | `verify.py` for northset family |
| Tests | `tests/unit/test_northset_receipt_key_classification_soft_verify.py` **3/3** (synth fully classified; unknown fail-closed; verify wires classification + candle join) |
| Partition tests | `tests/unit/test_northset_cli_echo_required_blob_only.py` (incl. `test_classified_union_matches_parts`) |

research_only — **never live Sharpe**. New stamps must be classified (CLI echo or blob-only); do not invent silent keys.

### Candle `join_coverage` verify wire

`join_coverage_honesty_errors` runs on **both** `northset` and `candle_order_book` families in `verify.py` (open unit interval or `[book_join_coverage_floor, 1]`). Anchors: `test_join_coverage_soft_verify.py`; classification soft-verify also asserts the candle join wire.

## CoS FEATURE_COLS IC completeness + `mid_lag1_corr` stamp

| Path | Status | Notes |
|---|---|---|
| `tests/unit/test_candle_feature_cols_ic_completeness_soft_verify.py` | **3/3 GREEN** | all `FEATURE_COLS` IC keys present; completeness helper + verify wire |
| `tests/unit/test_mid_lag1_corr_receipt_stamp.py` | **3/3 GREEN** | `mid_lag1_corr` (+ `mid_lag1_n_securities`); source + synth honesty; rejects out-of-unit |

Pairs with structure IC presence / `candle_feature_cols_ic` honesty. Never equate `mid_lag1_corr` vs `ofi_lag1_corr` (see existing never-equate). research_only — **never live Sharpe**.


## Sergeant classify-or-echo **GREEN** + EXTRA finite-on-synth

### Classify-or-echo suite — **pytest 11/11 GREEN**

| Path | n | Role |
|---|---|---|
| `test_northset_cli_echo_required_blob_only.py` | **5** | REQUIRED/EXTRA/BLOB partition + CLASSIFIED union + CLI source |
| `test_northset_receipt_key_classification_soft_verify.py` | **3** | `northset_receipt_key_classification_honesty_errors` fail-closed + candle join wire |
| `test_join_coverage_soft_verify.py` | **3** | `join_coverage` on northset + candle_order_book |

**11/11** = 5+3+3. Constants: `NORTHSET_CLI_ECHO_EXTRA` (**65**), `NORTHSET_RECEIPT_CLASSIFIED` (**206** = REQUIRED∪EXTRA∪BLOB_ONLY). See **Northset receipt key CLASSIFIED partition**.

### EXTRA finite-on-synth — **3/3 GREEN** (NOT INFLIGHT)

| Path | `tests/unit/test_northset_cli_echo_extra_finite_on_synth.py` |
|---|---|
| Status | **3/3 GREEN** |
| Asserts | EXTRA keys present on synth; numeric not ±inf when finite; non-numeric honest types |

Pairs with REQUIRED finite-on-synth **3/3**. research_only — **never live Sharpe**.

## CoS FEATURE_COLS + ask queue / MWB + bid/ask queue pair honesty

| Landing | Detail |
|---|---|
| `_FEATURE_COLS` adds | `ask_queue_priority_proxy`, `microprice_weight_balance` (plus prior +5 structure LOB) |
| Bid/ask queue pair honesty | `northset_queue_priority_bid_ask_pair_honesty_errors` — when `mean_queue_priority_proxy` finite, require `mean_ask_queue_priority_proxy` present; never collapse ask into bid; ∈[0,1] when finite |
| Tests | `tests/unit/test_ask_queue_priority_and_mwb_ic_honesty.py` **4/4** — FEATURE_COLS scores ask_queue+MWB; ask required when bid finite; northset synth pair + floors; verify wires MWB IC honesty |

CLASSIFIED partition already documented. Off inventing.


## CoS FEATURE_COLS Spearman/Pearson completeness + lag1 n_securities (**12/12**)

**Status:** CoS **GREEN 12/12** soft-verify suite (Mac verified).

### Suite breakdown

| Path | n | Role |
|---|---|---|
| `tests/unit/test_ask_queue_priority_and_mwb_ic_honesty.py` | **4** | FEATURE_COLS scores `ask_queue_priority_proxy` + `microprice_weight_balance`; bid/ask queue pair; MWB IC verify wire |
| `tests/unit/test_candle_feature_cols_ic_completeness_soft_verify.py` | **3** | all FEATURE_COLS IC keys present |
| `tests/unit/test_feature_cols_spearman_pearson_and_lag1_n_soft_verify.py` | **3** | Spearman/Pearson/**t** pack completeness; `mid_lag1_n_securities` / `ofi_lag1_n_securities` ≥ 0 |
| `tests/unit/test_candle_feature_cols_structure_ic_presence.py` | **2** | structure LOB companions scored + `ic_*` presence |

**12/12** = 4+3+3+2.

### Completeness contract

When candle FEATURE_COLS are scored, receipt must carry per-feature pack: `ic_<col>`, `ic_<col>_p`, `ic_<col>_n_dates`, `ic_<col>_pearson`, `ic_<col>_t` (missing pearson/t fail-closed).

### Lag1 n_securities

`northset_lag_corr_and_sweep_count_honesty_errors`: `mid_lag1_n_securities` / `ofi_lag1_n_securities` ≥ 0 when finite (±inf fail-closed). Pairs with `mid_lag1_corr` stamp.

### Also GREEN (cross-ref)

- Sergeant EXTRA finite-on-synth **3/3** — `test_northset_cli_echo_extra_finite_on_synth.py`
- CoS ask_queue + MWB in `_FEATURE_COLS` — see **CoS FEATURE_COLS + ask queue / MWB**

research_only — **never live Sharpe**. Off inventing.


## Sergeant BLOB_ONLY∉CLI leak lock + book_age fuse honesty (**5/5**)

**Status:** Sergeant **GREEN 5/5**.

| Path | n | Role |
|---|---|---|
| `tests/unit/test_northset_blob_only_never_in_cli.py` | **2** | `BLOB_ONLY` disjoint from REQUIRED; **BLOB_ONLY keys absent** from `dipcatcher northset` CLI source (no silent CLI leak) |
| `tests/unit/test_northset_book_age_fuse_honesty.py` | **3** | northset mean/max book_age fuse honesty (≥0; max≥mean); receipt dispatcher includes book_age helper |

**5/5** = 2+3. Outside frozenset theater — leak lock + fuse PIT residual.

Cross-ref: candle PIT book_age fuse polish in `test_candle_book_age_fuse_honesty.py` (separate). research_only — **never live Sharpe**.

### CoS FEATURE_COLS Spearman/Pearson + lag1 n — **12/12 GREEN** (confirmed)

See **CoS FEATURE_COLS Spearman/Pearson completeness + lag1 n_securities**. Still GREEN.



## CoS candle MWB fuse ⇒ mean unit + overnight/rv/semi + notional IC (**6/6**)

| Path | `tests/unit/test_candle_mwb_fuse_and_overnight_rv_synth.py` |
|---|---|
| Status | **6/6 GREEN** |
| Coverage | fuse `microprice_weight_balance` ∈[0,1]; scored MWB requires mean in unit; honesty flags missing/bad mean; northset overnight/rv/semi/bv companions honest on synth; `notional_imbalance` IC present when scored; verify wires MWB scored⇒mean helper |

research_only — **never live Sharpe**.

## Sergeant session L2 identity residual + candle book_age polish (**9/9**)

| Path | n | Role |
|---|---|---|
| `tests/unit/test_northset_session_identity_rates_soft_verify.py` | **6** | session L2 identity rates ∈[0,1]; synth clean; **never equate** `session_ohlc_identity_rate` vs daily OHLC identity key |
| `tests/unit/test_candle_book_age_fuse_honesty.py` | **3** | candle PIT mean/max book_age; max&lt;mean / ±inf fail-closed |

**9/9** = 6+3. **GREEN** — not INFLIGHT.

Sergeant BLOB_ONLY∉CLI + northset book_age fuse **5/5** already documented.



## Sergeant `candle_order_book_ic_method_honesty_errors` (**6/6 GREEN**)

| Path | `tests/unit/test_candle_order_book_ic_method_soft_verify.py` |
|---|---|
| Helper | `catalog.candle_order_book_ic_method_honesty_errors` |
| Status | **6/6 GREEN** |
| Coverage | synth candle `ic_method` honesty clean; missing `ic_method` with feature IC fail-closed; invalid method fail-closed; `n_dates` &lt; 1 with finite IC fail-closed; northset family skipped; verify wire |

Candle FEATURE_COLS date-IC / HAC meta soft-verify. research_only — **never live Sharpe**.

## CoS `spread_over_mid` FEATURE_COLS + IC⇒mean + `session_ofi_sum` IC (**4/4 GREEN**)

| Path | `tests/unit/test_spread_over_mid_ic_and_session_ofi_ic.py` |
|---|---|
| Status | **4/4 GREEN** |
| Coverage | `spread_over_mid` in `_FEATURE_COLS` + scored IC pack; `candle_spread_over_mid_ic_implies_mean_honesty_errors` (IC⇒`mean_spread_over_mid`); flags missing mean; `northset_session_ofi_sum_ic_honesty_errors` bounds + synth clean; never-equate `session_ofi_sum_p_ic` vs `ofi_p_ic`; verify wire |

research_only — **never live Sharpe**.

## CoS `depth_imbalance_abs` FEATURE_COLS + `ofi_lag` IC honesty (**4/4 GREEN**)

| Path | `tests/unit/test_depth_imbalance_ic_and_ofi_lag_ic.py` |
|---|---|
| Status | **4/4 GREEN** |
| Coverage | `depth_imbalance_abs` in `_FEATURE_COLS` + means; `candle_depth_imbalance_ic_implies_mean_honesty_errors` (flags missing means); `northset_ofi_lag_ic_honesty_errors` bounds + synth; verify wire |

research_only — **never live Sharpe**. Pairs with spread_over_mid / session_ofi IC **4/4** (already GREEN).

## CoS queue/vpin IC + candle structure IC⇒mean (**4/4 GREEN**)

| Path | `tests/unit/test_queue_vpin_ic_and_structure_means.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helpers | `candle_structure_ic_implies_mean_honesty_errors`; `northset_queue_imbalance_ic_honesty_errors`; `northset_vpin_ic_pack_honesty_errors` |
| Coverage | candle structure IC⇒means (e.g. imbalance_top / queue_imbalance / tob_size_share / …); flags missing mean; queue + VPIN IC packs honest on synth; verify wires structure IC⇒mean helper |

research_only — **never live Sharpe**.

### Sergeant free-lane

**IDLE awaiting** Sergeant’s next free-lane — promote in docs when GREEN. Off inventing.


## Sergeant best_feature IC identity (**8/8 GREEN**)

| Path | n | Role |
|---|---|---|
| `tests/unit/test_candle_feature_cols_best_feature_identity.py` | **5** | best_feature / `mean_abs_ic` identity residual — synth clean; missing key/IC fail-closed; mean_abs ≠ mean(|spearman|) fail-closed; still rejects not-max-abs |
| `tests/unit/test_candle_feature_cols_ic_soft_verify.py` | **3** | existing FEATURE_COLS IC honesty (p / n_dates / mean_abs / best-key) |

**8/8** = 5+3. Uses `candle_feature_cols_ic_honesty_errors`. research_only — **never live Sharpe**.

## Sergeant n_fused / n_bars / n_scored sizing (**6/6 GREEN**)

| Path | `tests/unit/test_northset_n_bars_scored_soft_verify.py` |
|---|---|
| Helper | `northset_n_bars_scored_honesty_errors` |
| Status | **6/6 GREEN** |
| Coverage | n_bars/n_scored ok; n_scored&gt;n_bars; n_scored&gt;n_fused fail-closed; n_fused&gt;n_bars fail-closed; n_fused not nonneg int fail-closed; synth sizing chain clean |

Chain: `n_fused ≤ n_bars`, `n_scored ≤ n_bars` (and scored vs fused fail-closed). research_only — **never live Sharpe**.

## CoS remaining FEATURE_COLS mean_* + amihud/depth/slope/body/vpin IC packs (**3/3 GREEN**)

| Path | `tests/unit/test_amihud_depth_slope_ic_packs_and_feature_means.py` |
|---|---|
| Status | **3/3 GREEN** |
| Helpers | `candle_feature_cols_ic_implies_mean_honesty_errors` (+ completeness); `northset_amihud_ic_pack_honesty_errors`; `northset_imbalance_depth_ic_pack_honesty_errors`; `northset_bid_log_size_slope_ic_pack_honesty_errors`; `northset_candle_body_ret_ic_pack_honesty_errors`; `northset_session_book_vpin_ic_pack_honesty_errors` |
| Coverage | all FEATURE_COLS have means when scored; IC pack bounds on synth; verify wire |

### Confirmed earlier (already in docs)

| Item | Status |
|---|---|
| Sergeant best_feature IC identity | **8/8 GREEN** |
| CoS imbalance/CLV/microprice IC packs + ofi/QP/slope means | **3/3 GREEN** |
| CoS queue/vpin + structure IC⇒mean | **4/4 GREEN** |


## CoS p_ic catchall + session_close / sweep / VoR packs (**3/3 GREEN**)

| Path | `tests/unit/test_p_ic_catchall_and_session_close_sweep_packs.py` |
|---|---|
| Status | **3/3 GREEN** |
| Helpers | `northset_all_p_ic_unit_interval_honesty_errors`; `northset_all_n_dates_nonneg_honesty_errors`; `northset_session_close_ic_packs_honesty_errors`; `northset_sweep_signed_ic_packs_honesty_errors`; `northset_volume_over_range_ic_packs_honesty_errors` |
| Coverage | catch-all `*_p_ic` ∈[0,1] / `*_n_dates` ≥0 fail-closed; packs clean on synth; session_close pack flags bad p |

research_only — **never live Sharpe**. Off inventing.



## Sergeant candle_order_book sizing + depth≥1 soft-verify (**8/8 GREEN**)

| Path | `tests/unit/test_candle_order_book_sizing_soft_verify.py` |
|---|---|
| Helper | `candle_order_book_sizing_honesty_errors` |
| Status | **8/8 GREEN** (prior 6 + depth **2**) |
| Coverage | synth sizing clean; `min_names`&lt;1 fail-closed; `n_scored`&gt;`n_fused`; `n_fused`&gt;`n_bars`; northset family skipped; **`depth` &lt;1 fail-closed**; synth `depth` stamped ≥1; verify wire |

Chain: `n_scored ≤ n_fused ≤ n_bars`; `depth` integer ≥1. Distinct from northset `northset_n_bars_scored_honesty_errors`. research_only — **never live Sharpe**.


## CoS dm_park + sweep_evidence_scope + candle join/chain (**4/4 GREEN**)

| Path | `tests/unit/test_dm_park_sweep_scope_and_candle_join_chain.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helpers | `northset_dm_park_honesty_errors`; `northset_sweep_evidence_scope_honesty_errors`; `candle_join_coverage_and_chain_honesty_errors` |
| Coverage | DM vs Park + sweep scope on synth; flags bad p/preferred; candle join_coverage + chain on synth; verify wire |

### Confirmed earlier (already in docs)

| Item | Status |
|---|---|
| Sergeant receipt bool-flags | **6/6 GREEN** |
| CoS candle claim + pearson + northset string enums | **4/4 GREEN** |


## Sergeant dgp / book_dgp ↔ data_source (**6/6 GREEN**)

| Path | `tests/unit/test_northset_receipt_dgp_data_source_soft_verify.py` |
|---|---|
| Status | **6/6 GREEN** |
| Coverage | synth clean; `dgp`/`book_dgp` mismatch fail-closed; SYNTHETIC ⇔ `synthetic_lob` / data_source rules; candle family skipped; helper registered |

research_only — **never live Sharpe**.

## CoS all_*_rate unit + sweep evidence blob (**3/3 GREEN**)

| Path | `tests/unit/test_pattern_rate_catchall_and_sweep_evidence_blob.py` |
|---|---|
| Status | **3/3 GREEN** |
| Coverage | pattern `*_rate` unit-interval catchall; sweep evidence blob honesty |

### Confirmed earlier (already in docs)

| Item | Status |
|---|---|
| Sergeant candle sizing + depth≥1 | **8/8 GREEN** |
| CoS dm_park + sweep_evidence_scope + candle join/chain | **4/4 GREEN** (`test_dm_park_sweep_scope_and_candle_join_chain.py`) |


## Sergeant use_session_l2 ↔ gate consistency (**6/6 GREEN**)

| Path | `tests/unit/test_northset_use_session_l2_gate_consistency.py` |
|---|---|
| Status | **6/6 GREEN** |
| Helper | `northset_use_session_l2_gate_consistency_errors` (registered) |
| Pair | `use_session_l2` ↔ `session_l2_identity_gate` |

| Config | Expected gate stamp |
|---|---|
| `use_session_l2=True` | `session_l2_identity_gate == "enforced"` |
| `use_session_l2=False` | `session_l2_identity_gate == "skipped"` |

Fail-closed codes: `use_session_l2_true_gate_not_enforced`, `use_session_l2_false_gate_not_skipped`. Incomplete pair (only one key present) is skipped. Synth benches cover on/off clean paths. research_only — **never live Sharpe**.

### Also confirmed this wave (already stamped)

| Item | Status |
|---|---|
| Sergeant dgp/book_dgp ↔ data_source | **6/6 GREEN** (`test_northset_receipt_dgp_data_source_soft_verify.py`) |
| CoS all_*_rate unit + sweep evidence blob | **3/3 GREEN** (`test_pattern_rate_catchall_and_sweep_evidence_blob.py`) |


## CoS all_*_share unit + candle spread alias honesty (**3/3 GREEN**)

| Path | `tests/unit/test_share_catchall_and_candle_spread_alias.py` |
|---|---|
| Status | **3/3 GREEN** |
| Helpers | `northset_all_share_unit_honesty_errors` (northset helpers registry); `candle_spread_alias_honesty_errors` (wired in `verify.py`) |

| Check | Behavior |
|---|---|
| `*_share` catch-all | Values must lie in **[0, 1]**; e.g. `sweep_low_reclaim_share_out_of_unit_interval` fail-closed |
| Candle spread aliases | `mean_half_spread` ≡ half of `mean_quoted_spread` (and related quoted/effective/half/bps identities on candle_order_book receipt) |
| verify wiring | `candle_spread_alias_honesty_errors` must appear in `src/quant_fund/research/verify.py` |

research_only — **never live Sharpe**.

## Sergeant component_sources (**6/6 GREEN**)

| Path | `tests/unit/test_northset_component_sources_soft_verify.py` |
|---|---|
| Status | **6/6 GREEN** |
| Helper | `northset_component_sources_honesty_errors` (registered) |

| Case | Expectation |
|---|---|
| synth + `use_session_l2=True` | `session_candles`/`session_book` = `synthetic_reconstruction`; `book` equals receipt `book_source` |
| synth + `use_session_l2=False` | `session_book` = `disabled` |
| not a dict | `component_sources_not_dict` |
| missing key | e.g. `component_sources_missing_session_book` |
| session_book vs flag | `component_sources_session_book_mismatch_use_session_l2` when True but `session_book=disabled` |

Always-on receipt map soft-verify (≠ nest). research_only — **never live Sharpe**.

### Confirmed earlier this wave

| Item | Status |
|---|---|
| Sergeant use_session_l2 ↔ gate consistency | **6/6 GREEN** (`test_northset_use_session_l2_gate_consistency.py`) |


## Sergeant northset depth (**6/6 GREEN**)

| Path | `tests/unit/test_northset_depth_soft_verify.py` |
|---|---|
| Status | **6/6 GREEN** |
| Helper | `northset_depth_honesty_errors` (registered) |
| Rule | Always-on northset receipt `depth` stamp: **integer ≥ 1** |

| Case | Behavior |
|---|---|
| synth clean | `int(depth) >= 1` |
| `depth < 1` or non-int | `northset_depth_lt_one_or_not_int` |
| non-finite | `northset_depth_non_finite` |
| `family == candle_order_book` | skipped (candle sizing has its own depth≥1 suite) |

research_only — **never live Sharpe**.

### Confirmed earlier this wave (already stamped)

| Item | Status |
|---|---|
| CoS all_*_share unit + candle spread alias | **3/3 GREEN** (`test_share_catchall_and_candle_spread_alias.py`) |
| Sergeant component_sources | **6/6 GREEN** (`test_northset_component_sources_soft_verify.py`) |


## CoS fraction catchall + queue_imbalance_mean alias (**3/3 GREEN**)

| Path | `tests/unit/test_fraction_catchall_and_queue_imbalance_mean.py` |
|---|---|
| Status | **3/3 GREEN** |
| Docstring | Honesty: *_fraction unit catch-all + queue_imbalance_mean alias. |
| Helpers | `northset_all_fraction_unit_honesty_errors`, `northset_queue_imbalance_mean_alias_honesty_errors` |
| Tests | `test_fraction_unit_ok_and_oob`, `test_queue_imbalance_mean_alias`, `test_synth_receipt_fraction_and_queue_honest` |

research_only — **never live Sharpe**.

### Confirmed earlier this wave

| Item | Status |
|---|---|
| Sergeant northset depth | **6/6 GREEN** (`test_northset_depth_soft_verify.py`) |



## Sergeant price_basis / return_basis (**6/6 GREEN**)

| Path | `tests/unit/test_northset_price_return_basis_soft_verify.py` |
|---|---|
| Status | **6/6 GREEN** |
| Helper | `northset_price_return_basis_honesty_errors` (registered) |

| Case | Expectation |
|---|---|
| synth + `require_adjusted_ohlc=False` | both stamps = `raw_fixture_opt_out` |
| invalid `price_basis` | `northset_price_basis_invalid` |
| raw_fixture pair mismatch | `northset_raw_fixture_price_return_basis_mismatch` |
| `split_adjusted` + `total_return` | clean |
| `split_adjusted` + `raw_fixture_opt_out` | `northset_split_adjusted_return_basis_invalid` |

Always-on northset receipt stamps. research_only — **never live Sharpe**.

### Confirmed earlier this wave

| Item | Status |
|---|---|
| CoS fraction catchall + queue_imbalance_mean alias | **3/3 GREEN** (`test_fraction_catchall_and_queue_imbalance_mean.py`) |
| Sergeant northset depth | **6/6 GREEN** (`test_northset_depth_soft_verify.py`) |


## CoS candle ic_*_p / t / n_dates catchalls (**4/4 GREEN**)

| Path | `tests/unit/test_candle_ic_p_t_ndates_catchalls.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helpers | `candle_all_ic_p_unit_honesty_errors`, `candle_all_ic_t_finite_honesty_errors`, `candle_all_ic_n_dates_nonneg_honesty_errors` (wired in `verify.py`) |

| Catch-all | Rule |
|---|---|
| `ic_*_p` | ∈[0,1]; `ic_*_pearson` is **not** treated as `_p` |
| `ic_*_t` | finite; non-finite → `*_non_finite_fail_closed` |
| `ic_*_n_dates` | ≥0 finite; else `*_negative_or_non_finite` |

Candle family only. research_only — **never live Sharpe**.


## Sergeant family / book_source + label nonempty (**8/8 GREEN**)

| Path | `tests/unit/test_northset_family_book_source_soft_verify.py` |
|---|---|
| Status | **8/8 GREEN** (was 6/6; +label nonempty) |
| Helper | `northset_family_book_source_honesty_errors` (registered) |

| Case | Expectation |
|---|---|
| synth clean | `family == "northset"`; non-empty str `book_source` |
| invalid family | `northset_family_invalid` |
| empty/whitespace book_source | `northset_book_source_empty_or_not_str` |
| empty/whitespace `label` | `northset_label_empty_or_not_str` |
| synth label nonempty | `label` non-empty strip |
| candle family / absent keys | skipped |

research_only — **never live Sharpe**.


## Sergeant candle family / provenance (**5/5 GREEN**)

| Path | `tests/unit/test_candle_order_book_family_provenance_soft_verify.py` |
|---|---|
| Status | **5/5 GREEN** |
| Helper | `candle_order_book_family_provenance_honesty_errors` (wired in `verify.py`) |

| Case | Expectation |
|---|---|
| synth clean | `family == "candle_order_book"`; non-empty `book_source` + `label` |
| non-candle family | skipped |
| empty/whitespace `book_source` | `candle_book_source_empty_or_not_str` |
| empty `label` | `candle_label_empty_or_not_str` |

Candle-only provenance stamps (≠ northset family/book_source helper). research_only — **never live Sharpe**.

## CoS mean_imbalance_top + shape_columns_ensured rates (**3/3 GREEN**)

| Path | `tests/unit/test_candle_imbalance_top_and_shape_ensured_rates.py` |
|---|---|
| Status | **3/3 GREEN** |
| Helpers | `mean_imbalance_top_honesty_errors` (candle); `northset_shape_columns_ensured_rates_honesty_errors` (northset) |

| Check | Rule |
|---|---|
| `mean_imbalance_top` | ∈[-1,1]; OOB → `mean_imbalance_top_out_of_unit_interval` |
| `shape_columns_ensured=True` | requires shape rate keys (e.g. `depth_shape_finite_rate`); missing → `*_missing_while_shape_columns_ensured` |
| `shape_columns_ensured=False` | skip |

Both wired in `verify.py`. research_only — **never live Sharpe**.

### Confirmed earlier this wave

| Item | Status |
|---|---|
| Sergeant northset family / book_source | **6/6 GREEN** (`test_northset_family_book_source_soft_verify.py`) |
| CoS candle ic_*_p / t / n_dates catchalls | **4/4 GREEN** (`test_candle_ic_p_t_ndates_catchalls.py`) |


## Sergeant candle dgp / data_source (**5/5 GREEN**)

| Path | `tests/unit/test_candle_order_book_dgp_data_source_soft_verify.py` |
|---|---|
| Status | **5/5 GREEN** |
| Helper | `candle_order_book_dgp_data_source_honesty_errors` (wired in `verify.py`) |
| Note | Candle twin of northset `dgp`/`book_dgp`↔`data_source` — **never equate** the two helpers |

| Case | Expectation |
|---|---|
| synth + `label=SYNTHETIC` | `data_source=SYNTHETIC`; `book_dgp=synthetic_lob` |
| `dgp` ≠ `book_dgp` | `candle_dgp_book_dgp_mismatch` |
| SYNTHETIC + non-`synthetic_lob` book_dgp | `candle_SYNTHETIC_data_source_book_dgp_not_synthetic_lob` |
| northset family | skipped |

research_only — **never live Sharpe**.

## CoS metrics_required_finite_ok ⇒ structure rates (**4/4 GREEN**)

| Path | `tests/unit/test_metrics_required_finite_ok_rates_honesty.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helper | `northset_metrics_required_finite_ok_rates_honesty_errors` (wired in `verify.py`) |

| Case | Expectation |
|---|---|
| synth clean | `metrics_required_finite_ok is True` ⇒ companion structure finite rates present/finite |
| True + missing rate | e.g. `structure_finite_rate_missing_while_metrics_required_finite_ok` |
| `metrics_required_finite_ok=False` | skip |

research_only — **never live Sharpe**.

### Confirmed earlier this wave

| Item | Status |
|---|---|
| Sergeant candle family / provenance | **5/5 GREEN** (`test_candle_order_book_family_provenance_soft_verify.py`) |
| CoS mean_imbalance_top + shape_columns_ensured | **3/3 GREEN** (`test_candle_imbalance_top_and_shape_ensured_rates.py`) |


## Sergeant sweep_evidence_scope stamp-contract (**10/10 GREEN**)

| Path | `tests/unit/test_northset_sweep_evidence_scope_stamp_contract.py` |
|---|---|
| Status | **10/10 GREEN** (Sergeant) |
| Helper | `northset_sweep_evidence_scope_honesty_errors` (registered) |
| Allowed enum (fixed) | `synthetic` \| `empirical_adjusted` \| `fixture_raw_unadjusted` |
| Legacy rejected | e.g. `vendor` → `sweep_evidence_scope_invalid` |

| Companion rule | Expectation |
|---|---|
| `data_source=SYNTHETIC` | scope must be `synthetic` (else `sweep_evidence_scope_not_synthetic_despite_SYNTHETIC_data_source`) |
| `price_basis=split_adjusted` | scope must be `empirical_adjusted` |
| `price_basis=raw_fixture_opt_out` | scope must be `fixture_raw_unadjusted` |
| synth bench | stamps `sweep_evidence_scope=synthetic` with `data_source=SYNTHETIC` |

Aligns allowed enums + companions; supersedes legacy vendor-style scopes. Distinct from CoS dm_park/sweep_scope join-chain suite and sweep evidence blob honesty. research_only — **never live Sharpe**.

### Confirmed earlier this wave

| Item | Status |
|---|---|
| Sergeant candle dgp / data_source | **5/5 GREEN** (`test_candle_order_book_dgp_data_source_soft_verify.py`) |
| CoS metrics_required_finite_ok ⇒ rates | **4/4 GREEN** (`test_metrics_required_finite_ok_rates_honesty.py`) |


## CoS depth_imbalance(+abs) + METRICS_REQUIRED finite-when-present (**3/3 GREEN**)

| Path | `tests/unit/test_candle_depth_imbalance_and_metrics_keys_present.py` |
|---|---|
| Status | **3/3 GREEN** |
| Helpers | `mean_depth_imbalance_honesty_errors`, `mean_depth_imbalance_abs_honesty_errors` (candle); `northset_metrics_required_keys_finite_when_present_honesty_errors` |

| Check | Rule |
|---|---|
| `mean_depth_imbalance` | ∈[-1,1]; OOB → `mean_depth_imbalance_out_of_unit_interval` |
| `mean_depth_imbalance_abs` | ∈[0,1] (non-neg unit); OOB → `mean_depth_imbalance_abs_out_of_unit_interval` |
| METRICS_REQUIRED keys | when present, must be finite; e.g. `spread_non_finite_metrics_required` |

Wired in `verify.py`. research_only — **never live Sharpe**.

## Sergeant shape_columns_ensured ↔ book_panel_path (**5/5 GREEN**)

| Path | `tests/unit/test_northset_shape_ensured_book_panel_path_soft_verify.py` |
|---|---|
| Status | **5/5 GREEN** |
| Helper | `northset_shape_columns_ensured_book_panel_path_honesty_errors` (registered) |

| `shape_columns_ensured` | `book_panel_path` | Result |
|---|---|---|
| True | None/empty | clean (synth default) |
| True | set | `book_panel_path_set_while_shape_columns_ensured` |
| False | missing | `book_panel_path_missing_while_shape_columns_not_ensured` |
| False | set | clean |

research_only — **never live Sharpe**.

## Sergeant include_kyle_ofi ↔ kyle_ofi nest (**5/5 GREEN**)

| Path | `tests/unit/test_northset_include_kyle_ofi_nest_presence_soft_verify.py` |
|---|---|
| Status | **5/5 GREEN** |
| Helper | `northset_include_kyle_ofi_nest_presence_honesty_errors` (registered) |

| `include_kyle_ofi` | Nest | Expectation |
|---|---|---|
| False | absent | clean |
| True | present dict with `research_only=True` | clean |
| True | missing | `kyle_ofi_nest_missing_while_include_kyle_ofi_true` |
| False | present | `kyle_ofi_nest_present_while_include_kyle_ofi_false` |

Presence-only soft-verify (≠ nest never-equates / overwrite suites). research_only — **never live Sharpe**.

## CoS research_only ⇒ claim (**10/10 GREEN**)

| Path | `tests/unit/test_research_only_implies_claim.py` |
|---|---|
| Status | **10/10 GREEN** (Sergeant/CoS reported) |
| Helpers | `northset_top_level_claim_honesty_errors`, `candle_order_book_claim_honesty_errors` |

| Rule | Fail-closed |
|---|---|
| `research_only=True` requires `claim` present | `*_claim_missing_while_research_only_true` |
| `claim` must be `research_diagnostic_only` | `*_claim_not_research_diagnostic_only` |
| synth northset + candle | both couple `research_only` ↔ claim clean |

research_only — **never live Sharpe**.

### Confirmed earlier this wave

| Item | Status |
|---|---|
| Sergeant sweep_evidence_scope stamp-contract | **10/10 GREEN** (`test_northset_sweep_evidence_scope_stamp_contract.py`; enum fixed) |


## Sergeant conservation ↔ reconstructs never-equate (**4/4 GREEN**)

| Path | `tests/unit/test_session_volume_conservation_vs_reconstructs_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helper | `session_volume_conservation_vs_reconstructs_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H23 reconstructs | `session_reconstructs_daily_rate` / `H23_HYPOTHESIS_ID` |
| H24 volume conservation | `session_volume_conservation_rate` / `H24_HYPOTHESIS_ID` |

**Never equate** the two keys or hypothesis ids (values may both be 1.0 on synth — identity is key/H-id). One key absent ⇒ skip. research_only — **never live Sharpe**.

## CoS control_sample_adequate ⇒ n / p / t (**3/3 GREEN**)

| Path | `tests/unit/test_sweep_control_sample_adequate_honesty.py` |
|---|---|
| Status | **3/3 GREEN** |
| Helper | `northset_sweep_control_sample_adequate_honesty_errors` (registered) |

| Flag | When True, companions must be honest |
|---|---|
| `sweep_*_control_sample_adequate` | `*_control_n_dates` ≥1; `*_control_diff_p` valid unit; `*_control_diff_t` finite |

False ⇒ skip. Fail-closed examples: `sweep_follow_control_n_dates_lt_one_while_sample_adequate`, `*_diff_p_invalid_*`, `*_diff_t_non_finite_*`. research_only — **never live Sharpe**.

## Sergeant impact_estimator_scope stamp-contract (**4/4 GREEN**)

| Path | `tests/unit/test_northset_impact_estimator_scope_stamp_contract.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helper | `northset_sweep_evidence_scope_honesty_errors` (same helper also covers `impact_estimator_scope`) |

| Case | Expectation |
|---|---|
| synth clean | `impact_estimator_scope == "per_security_equal_weight"` |
| invalid / empty | `impact_estimator_scope_invalid` |
| valid alone | clean |

Pairs with sweep_evidence_scope stamp-contract; empty/non-allowed estimator scopes fail-closed. research_only — **never live Sharpe**.

### Confirmed earlier this wave

| Item | Status |
|---|---|
| Sergeant include_kyle_ofi ↔ nest | **5/5 GREEN** |
| Sergeant shape_ensured ↔ book_panel_path | **5/5 GREEN** |
| CoS research_only ⇒ claim | **10/10 GREEN** |


## CoS fold_positive rates (**2/2 GREEN**)

| Path | `tests/unit/test_sweep_fold_positive_rates_honesty.py` |
|---|---|
| Status | **2/2 GREEN** |
| Helper | `northset_sweep_fold_positive_rates_honesty_errors` (registered) |

| Key | Rule |
|---|---|
| `sweep_*_fold_positive_fraction` | ∈[0,1]; OOB → `*_out_of_unit_interval` |
| `sweep_min_fold_positive_fraction` | ∈**(0,1]** (must be positive); `0.0` → `*_not_positive_unit` |
| NaN fractions | skipped |

Explicit soft-verify (complements fraction catch-all). research_only — **never live Sharpe**.

## Sergeant book_join_coverage_floor (**4/4 GREEN**)

| Path | `tests/unit/test_northset_book_join_coverage_floor_soft_verify.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helper | `northset_shape_and_session_l2_floors_honesty_errors` (floors soft-verify path) |

| Case | Expectation |
|---|---|
| synth clean | `book_join_coverage_floor` present ∈[0,1] |
| OOB / negative / inf | `book_join_coverage_floor_out_of_unit_interval` |

research_only — **never live Sharpe**.

### Confirmed earlier this wave (already stamped)

| Item | Status |
|---|---|
| Sergeant conservation ↔ reconstructs never-equate | **4/4 GREEN** |
| Sergeant impact_estimator_scope stamp-contract | **4/4 GREEN** |
| CoS control_sample_adequate ⇒ n/p/t | **3/3 GREEN** |


## CoS amihud / qlike / corwin pack (**8/8 GREEN**)

| Path | `tests/unit/test_amihud_qlike_range_spread_honesty_pack.py` |
|---|---|
| Status | **8/8 GREEN** (CoS reported) |
| Helpers | `amihud_mean_honesty_errors`, `northset_qlike_means_honesty_errors`, `northset_range_spread_honesty_errors` (all registered) |

| Family | Rule |
|---|---|
| `amihud_mean` | ≥0; else `amihud_mean_negative` |
| qlike means (e.g. `parkinson_qlike_vs_cc`) | ≥0; else `*_negative` |
| `corwin_schultz_spread` / `abdi_ranaldo_spread` | ∈[0,1] |
| `roll_spread` | ≥0 (**may exceed 1**) |

Synth receipt clean + helpers on `NORTHSET_RECEIPT_HONESTY_HELPERS`. research_only — **never live Sharpe**.

## Sergeant session_chain ↔ siblings never-equate (**4/4 GREEN**)

| Path | `tests/unit/test_session_chain_vs_siblings_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helper | `session_chain_vs_session_identity_siblings_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H29 chain | `session_chain_rate` / `H29_HYPOTHESIS_ID` |
| H23 reconstructs | `session_reconstructs_daily_rate` / `H23_HYPOTHESIS_ID` |
| H24 conservation | `session_volume_conservation_rate` / `H24_HYPOTHESIS_ID` |

**Never equate** H29 with H23/H24 siblings (key/hypothesis identity). Chain alone ⇒ skip. Complements conservation↔reconstructs never-equate. research_only — **never live Sharpe**.

### Confirmed earlier this wave

| Item | Status |
|---|---|
| Sergeant book_join_coverage_floor | **4/4 GREEN** (`test_northset_book_join_coverage_floor_soft_verify.py`) |
| CoS fold_positive rates | **2/2 GREEN** (`test_sweep_fold_positive_rates_honesty.py`) |




## CoS candle mean_*_frac pack (**3/3 GREEN**)

| Path | `tests/unit/test_candle_frac_and_spread_x_honesty.py` |
|---|---|
| Status | **3/3 GREEN** |
| Helper | `candle_frac_and_spread_x_honesty_errors` (wired in `verify.py`) |

| Key | Rule |
|---|---|
| `mean_candle_range_frac` / `mean_candle_body_frac` | ∈[0,1] |
| `mean_imbalance_x_body_frac` | ∈[-1,1] |
| `mean_spread_x_range` | ≥0 |

Fail-closed examples: `*_out_of_unit_interval`, `*_out_of_signed_unit`, `mean_spread_x_range_negative`. Synth candle clean. research_only — **never live Sharpe**.

### Confirmed earlier this wave

| Item | Status |
|---|---|
| Sergeant session_chain ↔ siblings never-equate | **4/4 GREEN** (`test_session_chain_vs_siblings_never_equate.py`) |
| CoS amihud / qlike / corwin pack | **8/8 GREEN** (`test_amihud_qlike_range_spread_honesty_pack.py`) |


## Sergeant ohlc ↔ session_ohlc never-equate (**4/4 GREEN**)

| Path | `tests/unit/test_ohlc_identity_vs_session_ohlc_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helper | `ohlc_identity_vs_session_ohlc_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| Daily OHLC identity | `ohlc_identity_rate` / `H20_HYPOTHESIS_ID` |
| Session OHLC identity | `session_ohlc_identity_rate` |

**Never equate** daily vs session OHLC identity keys; H20 distinct from H23/H24/H29 session binds. One key absent ⇒ skip. research_only — **never live Sharpe**.


## Sergeant ohlc ↔ gap_finite never-equate (**4/4 GREEN**)

| Path | `tests/unit/test_ohlc_identity_vs_gap_finite_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helper | `ohlc_identity_vs_gap_finite_never_equate_honesty_errors` (registered) |

| Axis | Key / note |
|---|---|
| Daily OHLC identity | `ohlc_identity_rate` / `H20_HYPOTHESIS_ID` |
| Gap finite | `gap_finite_rate` (not an H20 gate key) |

**Never equate** the two rates. `northset_has_finite_ohlc_identity_rate` is True only for `ohlc_identity_rate` (not `gap_finite_rate`). H20 ∉ {H21, H22}. One key absent ⇒ skip. Complements ohlc↔session_ohlc never-equate. research_only — **never live Sharpe**.

### Confirmed earlier this wave

| Item | Status |
|---|---|
| Sergeant ohlc ↔ session_ohlc never-equate | **4/4 GREEN** (`test_ohlc_identity_vs_session_ohlc_never_equate.py`) |
| Sergeant family / book_source + label nonempty | **8/8 GREEN** (`test_northset_family_book_source_soft_verify.py`) |


## CoS candle_direction (**4/4 GREEN**)

| Path | `tests/unit/test_candle_direction_mean_honesty.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helpers | `candle_direction_mean_honesty_errors` (wired in `verify.py`); IC⇒mean via `candle_feature_cols_ic_implies_mean_honesty_errors` |

| Check | Rule |
|---|---|
| `FEATURE_COLS` | includes `candle_direction` (ternary feature) |
| `mean_candle_direction` | ∈[-1,1]; OOB → `mean_candle_direction_out_of_signed_unit` |
| IC⇒mean path | fail-closed `mean_candle_direction_out_of_unit_interval` when mean OOB with IC present |

Synth candle clean. research_only — **never live Sharpe**.

### Confirmed earlier this wave

| Item | Status |
|---|---|
| Sergeant ohlc ↔ gap_finite never-equate | **4/4 GREEN** (`test_ohlc_identity_vs_gap_finite_never_equate.py`) |


## Sergeant H20 ↔ H21 never-equate (**4/4 GREEN**)

| Path | `tests/unit/test_ohlc_identity_vs_book_uncrossed_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helper | `ohlc_identity_vs_book_uncrossed_never_equate_honesty_errors` (registered) |

| Axis | Key / id / gate |
|---|---|
| H20 | `ohlc_identity_rate` / `northset_has_finite_ohlc_identity_rate` |
| H21 | `book_uncrossed_rate` / `northset_has_finite_book_uncrossed_rate` (needs `book_hypothesis_eligible`) |

**Never equate** H20 vs H21 keys, hypothesis ids, or finite-rate gates. One key absent ⇒ skip. research_only — **never live Sharpe**.

## CoS wick_skew + candle_body_ret (**3/3 GREEN**)

| Path | `tests/unit/test_wick_skew_and_candle_body_ret_finite_pack.py` |
|---|---|
| Status | **3/3 GREEN** |
| Helpers | `candle_wick_skew_and_body_ret_means_honesty_errors` (verify.py); `northset_wick_skew_ic_pack_honesty_errors`; `northset_candle_body_ret_ic_pack_honesty_errors` |

| Surface | Rule |
|---|---|
| Candle means | `mean_wick_skew` / `mean_candle_body_ret` finite; else `*_non_finite` |
| Northset IC packs | wick_skew + candle_body_ret IC honesty packs clean on synth |

research_only — **never live Sharpe**.

### Confirmed earlier this wave

| Item | Status |
|---|---|
| Sergeant ohlc ↔ gap_finite never-equate | **4/4 GREEN** |
| CoS candle_direction | **4/4 GREEN** (`test_candle_direction_mean_honesty.py`) |


## Sergeant session_ohlc ↔ reconstructs never-equate (**4/4 GREEN**)

| Path | `tests/unit/test_session_ohlc_vs_reconstructs_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helper | `session_ohlc_vs_reconstructs_never_equate_honesty_errors` (registered) |

| Axis | Key / note |
|---|---|
| Session OHLC identity | `session_ohlc_identity_rate` (≠ H23 bind) |
| H23 reconstructs | `session_reconstructs_daily_rate` / `H23_HYPOTHESIS_ID` |

**Never equate** session OHLC identity with reconstructs daily rate. One key absent ⇒ skip. Complements H23↔H24 and session_chain never-equate. research_only — **never live Sharpe**.

## CoS signed_vol_x_imbalance (**4/4 GREEN**)

| Path | `tests/unit/test_signed_vol_x_imbalance_finite_pack.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helper | `candle_signed_vol_x_imbalance_mean_honesty_errors` (wired in `verify.py`) |

| Check | Rule |
|---|---|
| `FEATURE_COLS` | includes `signed_vol_x_imbalance` |
| `mean_signed_vol_x_imbalance` | finite when stamped; else `mean_signed_vol_x_imbalance_non_finite` |
| Synth | clean with IC⇒mean + wick/body companions |

research_only — **never live Sharpe**.

### Confirmed earlier this wave

| Item | Status |
|---|---|
| Sergeant H20 ↔ H21 never-equate | **4/4 GREEN** (`test_ohlc_identity_vs_book_uncrossed_never_equate.py`) |
| CoS wick_skew + candle_body_ret | **3/3 GREEN** (`test_wick_skew_and_candle_body_ret_finite_pack.py`) |


## Sergeant H21 ↔ H22 never-equate (**4/4 GREEN**) — triad H20–H21–H22 complete

| Path | `tests/unit/test_book_uncrossed_vs_imbalance_p_ic_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helper | `book_uncrossed_vs_imbalance_p_ic_never_equate_honesty_errors` (registered) |

| Axis | Key / gate |
|---|---|
| H21 | `book_uncrossed_rate` / `northset_has_finite_book_uncrossed_rate` |
| H22 | `imbalance_top_p_ic` / `northset_has_finite_imbalance_p_ic` |

**Never equate** H21 vs H22 keys, hypothesis ids, or finite gates (`book_hypothesis_eligible` required for both gates). Completes triad with H20↔H21 and H20↔gap_finite. One key absent ⇒ skip. research_only — **never live Sharpe**.

## CoS candle ofi / queue imbalance means (**3/3 GREEN**)

| Path | `tests/unit/test_candle_ofi_and_queue_imbalance_means_honesty.py` |
|---|---|
| Status | **3/3 GREEN** |
| Helper | `candle_ofi_and_queue_imbalance_means_honesty_errors` (wired in `verify.py`) |

| Key | Rule |
|---|---|
| `mean_ofi` | finite; else `mean_ofi_non_finite` |
| `mean_queue_imbalance` | ∈[-1,1]; else `mean_queue_imbalance_out_of_signed_unit` |

Synth candle stamps both clean. research_only — **never live Sharpe**.

## CoS microprice_minus_mid finite (**4/4 GREEN**)

| Path | `tests/unit/test_microprice_minus_mid_finite_pack.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helpers | `candle_microprice_minus_mid_finite_pack_honesty_errors`; `mean_microprice_minus_mid_honesty_errors`; IC⇒mean via `candle_ofi_qp_slope_ic_implies_mean_honesty_errors` |

| Check | Rule |
|---|---|
| means (+bps) | finite; else `*_non_finite_fail_closed` |
| IC scored without mean | `mean_microprice_minus_mid_missing_while_ic_*_scored` |
| verify.py | wires candle + northset families |

research_only — **never live Sharpe**.

### Confirmed earlier this wave

| Item | Status |
|---|---|
| Sergeant session_ohlc ↔ reconstructs | **4/4 GREEN** |
| Sergeant H20 ↔ H21 never-equate | **4/4 GREEN** |
| CoS signed_vol_x_imbalance | **4/4 GREEN** |


## CoS spread_bps + log slopes (**3/3 GREEN**)

| Path | `tests/unit/test_candle_spread_bps_and_log_slopes_honesty.py` |
|---|---|
| Status | **3/3 GREEN** |
| Helpers | `candle_spread_bps_nonneg_honesty_errors`, `candle_log_slopes_finite_honesty_errors` (wired in `verify.py`); also exercises `mean_depth_imbalance_honesty_errors` on synth |

| Key | Rule |
|---|---|
| `mean_spread_bps` | ≥0 finite; else `mean_spread_bps_negative_or_non_finite` |
| candle log size/price slopes (e.g. `mean_bid_log_size_slope`) | finite; else `*_non_finite` |
| `mean_depth_imbalance` | ∈[-1,1] (companion on synth) |

research_only — **never live Sharpe**.


## Sergeant H20 ↔ H22 never-equate (**4/4 GREEN**) — triad diagonal complete

| Path | `tests/unit/test_ohlc_identity_vs_imbalance_p_ic_never_equate.py` |
|---|---|
| Status | **4/4 GREEN**; with H20↔H21 + H21↔H22 edge suites → **12/12** triad never-equate |
| Helper | `ohlc_identity_vs_imbalance_p_ic_never_equate_honesty_errors` (registered) |

| Axis | Key / gate |
|---|---|
| H20 | `ohlc_identity_rate` / `northset_has_finite_ohlc_identity_rate` |
| H22 | `imbalance_top_p_ic` / `northset_has_finite_imbalance_p_ic` |

**Never equate** H20 vs H22 (triad diagonal). H20–H21–H22 never-equate triad is **complete**. One key absent ⇒ skip. research_only — **never live Sharpe**.


## CoS tob_size_share + concentration tops + tick_spacing (**3/3 GREEN**)

| Path | `tests/unit/test_candle_tob_concentration_tick_spacing_honesty.py` |
|---|---|
| Status | **3/3 GREEN** |
| Helpers | `mean_tob_size_share_honesty_errors`, `size_concentration_top_honesty_errors`, `candle_log_tick_spacing_finite_honesty_errors` (wired in `verify.py`) |

| Key | Rule |
|---|---|
| `mean_tob_size_share` | ∈**(0,1]**; `0.0` → `mean_tob_size_share_out_of_open_unit_interval` |
| concentration tops (e.g. `mean_bid_size_concentration_top`) | ∈**(0,1]**; OOB → `*_out_of_open_unit_interval` |
| log tick spacing (e.g. `mean_bid_mean_log_tick_spacing`) | finite; else `*_non_finite` |

Synth candle pack clean (+ IC⇒mean companion). research_only — **never live Sharpe**.


## CoS notional_imbalance + queue_priority + MWB (**3/3 GREEN**)

| Path | `tests/unit/test_candle_notional_queue_mwb_means_honesty.py` |
|---|---|
| Status | **3/3 GREEN** |
| Helpers | `mean_notional_imbalance_honesty_errors`, `mean_queue_priority_honesty_errors`, `mean_microprice_weight_balance_honesty_errors` (candle wired in `verify.py`) |

| Key | Rule |
|---|---|
| `mean_notional_imbalance` | ∈[-1,1] |
| `mean_queue_priority_proxy` | ∈[0,1] |
| `mean_microprice_weight_balance` | ∈[0,1] |

research_only — **never live Sharpe**.

## Commander gap ↔ uncrossed never-equate (**4/4 GREEN**)

| Path | `tests/unit/test_gap_finite_vs_book_uncrossed_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helper | `gap_finite_rate_vs_book_uncrossed_never_equate_honesty_errors` (registered) |

| Axis | Key / note |
|---|---|
| Gap finite | `gap_finite_rate` |
| H21 uncrossed | `book_uncrossed_rate` / `northset_has_finite_book_uncrossed_rate` only |

**Never equate** gap finite with book uncrossed (numeric equality on synth allowed; identity is key/gate). One key absent ⇒ skip. research_only — **never live Sharpe**.

## Sergeant session_bulk_vpin ↔ siblings never-equate (**5/5 GREEN**)

| Path | `tests/unit/test_session_bulk_vpin_vs_siblings_never_equate.py` |
|---|---|
| Status | **5/5 GREEN** |
| Helpers | `session_bulk_vpin_vs_siblings_never_equate_honesty_errors`, `session_bulk_vpin_honesty_errors` (registered) |

| Axis | Key / note |
|---|---|
| Bulk VPIN | `session_bulk_vpin` ∈[0,1] |
| Siblings | `vpin_mean`, `session_book_vpin_mean` |
| Hyp ids | `H32_HYPOTHESIS_ID` ≠ `H43_HYPOTHESIS_ID` |

**Never equate** bulk vs sibling VPIN means; `northset_has_finite_session_book_vpin_p_ic` is not triggered by bulk alone. NaN bulk skipped. research_only — **never live Sharpe**.

## Sergeant CLV alias identity (**5/5 GREEN**)

| Path | `tests/unit/test_close_location_value_clv_alias_identity.py` |
|---|---|
| Status | **5/5 GREEN** |
| Helper | `close_location_value_clv_alias_identity_honesty_errors` (registered) |

| Alias pair | Rule |
|---|---|
| `close_location_value_p_ic` ↔ `clv_p_ic` | must match when both stamped |
| `close_location_value_t_ic` ↔ `clv_t_ic` | must match when both stamped |

Keys remain distinct names; mismatch → `*_clv_*_mismatch`. One side absent ⇒ skip. research_only — **never live Sharpe**.

### Confirmed earlier this wave

| Item | Status |
|---|---|
| Sergeant VPIN triad never-equate (`session_bulk_vpin`↔siblings) | **5/5 GREEN** (`test_session_bulk_vpin_vs_siblings_never_equate.py`) |
| Commander gap ↔ uncrossed | **4/4 GREEN** (`test_gap_finite_vs_book_uncrossed_never_equate.py`) |
| CoS candle_dir_x_imbalance + close_mid_abs_rel | **3/3 GREEN** (`test_candle_dir_x_imbalance_close_mid_join_honesty.py`) |


## Commander session_ohlc ↔ gap never-equate (**4/4 GREEN**) + identity #186–#190

| Path | `tests/unit/test_session_ohlc_vs_gap_finite_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helper | `session_ohlc_vs_gap_finite_never_equate_honesty_errors` (registered) |
| Backlog | Mac identity continuous **#186–#190** (`day_grind_progress.md`) |

| Axis | Key / note |
|---|---|
| Session OHLC identity | `session_ohlc_identity_rate` (≠ H23 bind) |
| Gap finite | `gap_finite_rate` |

**Never equate** the two rates. Neither key feeds H20 `northset_has_finite_ohlc_identity_rate` nor H21 book-uncrossed gate. One key absent ⇒ skip. Complements session_ohlc↔reconstructs and ohlc↔gap_finite. research_only — **never live Sharpe**.

## CoS effective / half / ofi (**3/3 GREEN**)

| Path | `tests/unit/test_candle_effective_spread_half_ofi_honesty.py` |
|---|---|
| Status | **3/3 GREEN** |
| Helpers | `candle_spread_alias_honesty_errors` (`mean_effective_spread` ≥0; half ≡ quoted/2); `candle_feature_ofi_finite_honesty_errors` (wired in `verify.py`) |

| Check | Rule |
|---|---|
| `mean_effective_spread` | ≥0 finite |
| `mean_half_spread` | half of `mean_quoted_spread` |
| FEATURE `mean_ofi` | finite |

Synth candle clean. research_only — **never live Sharpe**.

### Confirmed earlier this wave

| Item | Status |
|---|---|
| Sergeant CLV alias identity | **5/5 GREEN** (`test_close_location_value_clv_alias_identity.py`) |


## Sergeant impact_proxy_warning (**5/5 GREEN**)

| Path | `tests/unit/test_northset_impact_proxy_warning_soft_verify.py` |
|---|---|
| Status | **5/5 GREEN** |
| Helper | `northset_impact_proxy_warning_honesty_errors` (registered) |
| Expected token | `depth_or_ofi_proxy_not_signed_trade_flow` |

| Case | Expectation |
|---|---|
| synth | stamps exact expected token |
| unexpected token | `impact_proxy_warning_unexpected_token` |
| empty / non-str | `impact_proxy_warning_not_nonempty_str` |
| absent / non-northset family | skipped |

research_only — **never live Sharpe**. Not signed trade-flow impact.

## Sergeant product stamp (**5/5 GREEN**)

| Path | `tests/unit/test_northset_product_stamp_soft_verify.py` |
|---|---|
| Status | **5/5 GREEN** |
| Helper | `northset_product_stamp_honesty_errors` (registered) |
| Expected | `product == "Northset"` when stamped on northset family |

| Case | Expectation |
|---|---|
| synth | stamps `Northset` |
| unexpected token | `northset_product_unexpected_token` |
| empty / non-str | `northset_product_not_nonempty_str` |
| absent / other family | skipped |

research_only — **never live Sharpe**.

### Confirmed earlier this wave

| Item | Status |
|---|---|
| Commander session_ohlc ↔ gap (+ identity #186–#190) | **4/4 GREEN** |
| CoS effective / half / ofi | **3/3 GREEN** |


## Commander session_ohlc ↔ book_uncrossed never-equate (**4/4 GREEN**) + identity #191–#195

| Path | `tests/unit/test_session_ohlc_vs_book_uncrossed_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helper | `session_ohlc_vs_book_uncrossed_never_equate_honesty_errors` (registered) |
| Backlog | Mac identity continuous **#191–#195** (`day_grind_progress.md`) |

| Axis | Key / note |
|---|---|
| Session OHLC identity | `session_ohlc_identity_rate` (≠ H23 bind; ≠ H20 gate) |
| H21 uncrossed | `book_uncrossed_rate` / `northset_has_finite_book_uncrossed_rate` only |

**Never equate** session OHLC identity with book uncrossed. Complements session_ohlc↔gap and gap↔uncrossed. One key absent ⇒ skip. research_only — **never live Sharpe**.

## CoS finite_rate catchall + price_slope IC⇒mean (**3/3 GREEN**)

| Path | `tests/unit/test_candle_finite_rate_prefix_and_price_slope_ic.py` |
|---|---|
| Status | **3/3 GREEN** |
| Helpers | `candle_all_finite_rate_prefix_honesty_errors`; IC⇒mean via `candle_ofi_qp_slope_ic_implies_mean_honesty_errors` (wired in `verify.py`) |

| Check | Rule |
|---|---|
| `finite_rate_*` (candle) | ∈[0,1]; OOB → `*_out_of_unit_interval` |
| `ic_bid_log_price_slope` (etc.) | requires matching `mean_*` when scored |

Synth clean. research_only — **never live Sharpe**.

### Confirmed earlier this wave

| Item | Status |
|---|---|
| Sergeant impact_proxy_warning | **5/5 GREEN** |
| Sergeant product stamp | **5/5 GREEN** |


## Sergeant session_ohlc ↔ volume_conservation never-equate (**4/4 GREEN**)

| Path | `tests/unit/test_session_ohlc_vs_volume_conservation_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helper | `session_ohlc_vs_volume_conservation_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| Session OHLC identity | `session_ohlc_identity_rate` (≠ H23/H24 bind) |
| H24 volume conservation | `session_volume_conservation_rate` / `H24_HYPOTHESIS_ID` |

**Never equate** session OHLC identity with volume conservation. Complements session_ohlc↔reconstructs (H23) and H23↔H24. One key absent ⇒ skip. research_only — **never live Sharpe**.

### Confirmed earlier this wave

| Item | Status |
|---|---|
| Sergeant impact_proxy_warning | **5/5 GREEN** |
| Sergeant product stamp | **5/5 GREEN** |


## Sergeant session_ohlc ↔ session_chain never-equate (**4/4 GREEN**)

| Path | `tests/unit/test_session_ohlc_vs_session_chain_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helper | `session_ohlc_vs_session_chain_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| Session OHLC identity | `session_ohlc_identity_rate` (≠ H23/H29) |
| H29 chain | `session_chain_rate` / `H29_HYPOTHESIS_ID` |

**Never equate**. research_only — **never live Sharpe**.

## Sergeant ohlc ↔ session_reconstructs never-equate (**4/4 GREEN**)

| Path | `tests/unit/test_ohlc_identity_vs_session_reconstructs_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helper | `ohlc_identity_vs_session_reconstructs_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H20 daily OHLC identity | `ohlc_identity_rate` / `northset_has_finite_ohlc_identity_rate` |
| H23 reconstructs | `session_reconstructs_daily_rate` / `H23_HYPOTHESIS_ID` |

**Never equate** H20 vs H23. Distinct from session_ohlc↔reconstructs. research_only — **never live Sharpe**.

## Commander gap ↔ session_chain never-equate (**4/4 GREEN**) + identity #196–#200

| Path | `tests/unit/test_gap_finite_vs_session_chain_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helper | `gap_finite_rate_vs_session_chain_never_equate_honesty_errors` (registered) |
| Backlog | Mac identity continuous **#196–#200** |

| Axis | Key / id |
|---|---|
| Gap finite | `gap_finite_rate` |
| H29 chain | `session_chain_rate` / `H29_HYPOTHESIS_ID` |

**Never equate**. Neither key is H20/H21 gate. research_only — **never live Sharpe**.

## CoS IC unit tighten (**2/2 GREEN**)

| Path | `tests/unit/test_candle_ic_unit_interval_tighten.py` |
|---|---|
| Status | **2/2 GREEN** |
| Helper | `candle_feature_cols_ic_honesty_errors` |

| Key | Rule |
|---|---|
| bare `ic_*` / `ic_*_pearson` / `best_feature_ic` | ∈[-1,1] |
| `mean_abs_ic` | ∈[0,1] |

research_only — **never live Sharpe**.

## Reconnect: Sergeant H20 ↔ H29 ohlc ↔ session_chain (**4/4 GREEN**)

| Path | `tests/unit/test_ohlc_identity_vs_session_chain_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk reconnect) |
| Helper | `ohlc_identity_vs_session_chain_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H20 | `ohlc_identity_rate` |
| H29 | `session_chain_rate` |

**Never equate** H20 vs H29. Completes ohlc↔session_chain edge beside session_ohlc↔chain. research_only — **never live Sharpe**.



## CoS H20 ↔ H24 ohlc ↔ volume_conservation (**4/4 GREEN**)

| Path | `tests/unit/test_ohlc_identity_vs_volume_conservation_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `ohlc_identity_vs_volume_conservation_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H20 | `ohlc_identity_rate` |
| H24 | `session_volume_conservation_rate` / `H24_HYPOTHESIS_ID` |

**Never equate** daily OHLC envelope identity with session volume conservation. H24 binds volume only; H20 gate on daily ohlc only. research_only — **never live Sharpe**.


## Sergeant H21 ↔ H29 book_uncrossed ↔ session_chain (**4/4 GREEN**)

| Path | `tests/unit/test_book_uncrossed_vs_session_chain_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `book_uncrossed_vs_session_chain_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H21 | `book_uncrossed_rate` |
| H29 | `session_chain_rate` / `H29_HYPOTHESIS_ID` |

**Never equate** uncrossed book with session reconstruction chain. H21 finite gate on book only; H29 binds chain only. research_only — **never live Sharpe**.


## Commander gap ↔ H23 session_reconstructs (**4/4 GREEN**)

| Path | `tests/unit/test_gap_finite_vs_session_reconstructs_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `gap_finite_rate_vs_session_reconstructs_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| Gap finite | `gap_finite_rate` |
| H23 | `session_reconstructs_daily_rate` / `H23_HYPOTHESIS_ID` |

**Never equate** overnight gap finiteness with session envelope reconstructing the daily bar. research_only — **never live Sharpe**.


## General H21 ↔ H24 book_uncrossed ↔ volume_conservation (**4/4 GREEN**)

| Path | `tests/unit/test_book_uncrossed_vs_volume_conservation_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helper | `book_uncrossed_vs_volume_conservation_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H21 | `book_uncrossed_rate` |
| H24 | `session_volume_conservation_rate` / `H24_HYPOTHESIS_ID` |

**Never equate** uncrossed book with session volume conservation (H21 ≠ H24). H21 finite gate on book only; H24 binds volume only. Numeric equality on clean synth allowed. research_only — **never live Sharpe**.



## Commander gap ↔ volume_conservation (**4/4 GREEN**)

| Path | `tests/unit/test_gap_finite_vs_session_volume_conservation_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `gap_finite_rate_vs_session_volume_conservation_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| Gap finite | `gap_finite_rate` |
| H24 | `session_volume_conservation_rate` / `H24_HYPOTHESIS_ID` |

**Never equate** overnight gap finiteness with session volume conservation. Off H20/H21/H22 triad invent. research_only — **never live Sharpe**.


## Sergeant H22 ↔ H29 imbalance_top_p_ic ↔ session_chain (**4/4 GREEN**)

| Path | `tests/unit/test_imbalance_top_p_ic_vs_session_chain_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `imbalance_top_p_ic_vs_session_chain_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H22 | `imbalance_top_p_ic` |
| H29 | `session_chain_rate` / `H29_HYPOTHESIS_ID` |

**Never equate** imbalance discovery IC p with session reconstruction chain. H22 gate on imbalance only; H29 binds chain only. research_only — **never live Sharpe**.


## Commander H22 ↔ gap imbalance_top_p_ic ↔ gap_finite (**4/4 GREEN**)

| Path | `tests/unit/test_imbalance_top_p_ic_vs_gap_finite_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `imbalance_top_p_ic_vs_gap_finite_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H22 | `imbalance_top_p_ic` |
| Gap | `gap_finite_rate` (no H22 claim) |

**Never equate** imbalance discovery IC p with overnight gap finiteness. research_only — **never live Sharpe**.



## CoS H22 ↔ H24 imbalance_top_p_ic ↔ volume_conservation (**4/4 GREEN**)

| Path | `tests/unit/test_imbalance_top_p_ic_vs_session_volume_conservation_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `imbalance_top_p_ic_vs_session_volume_conservation_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H22 | `imbalance_top_p_ic` |
| H24 | `session_volume_conservation_rate` / `H24_HYPOTHESIS_ID` |

**Never equate** imbalance discovery IC p with session volume conservation. research_only — **never live Sharpe**.


## H22 ↔ session_ohlc imbalance_top_p_ic ↔ session_ohlc_identity (**4/4 GREEN**)

| Path | `tests/unit/test_imbalance_top_p_ic_vs_session_ohlc_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `imbalance_top_p_ic_vs_session_ohlc_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H22 | `imbalance_top_p_ic` |
| Session OHLC | `session_ohlc_identity_rate` (not H23/H24/H29) |

**Never equate** imbalance discovery IC p with session-candle OHLC identity. research_only — **never live Sharpe**.



## Lt H22 ↔ H23 imbalance_top_p_ic ↔ session_reconstructs (**4/4 GREEN**)

| Path | `tests/unit/test_imbalance_top_p_ic_vs_session_reconstructs_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `imbalance_top_p_ic_vs_session_reconstructs_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H22 | `imbalance_top_p_ic` |
| H23 | `session_reconstructs_daily_rate` / `H23_HYPOTHESIS_ID` |

**Never equate** imbalance discovery IC p with session envelope reconstructing the daily bar. Completes H22 mesh edges: gap / chain / volume / session_ohlc / reconstructs. research_only — **never live Sharpe**.



## Sergeant H25 ↔ gap microprice_p_ic ↔ gap_finite (**4/4 GREEN**)

| Path | `tests/unit/test_microprice_p_ic_vs_gap_finite_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `microprice_p_ic_vs_gap_finite_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H25 | `microprice_p_ic` |
| Gap | `gap_finite_rate` (no H25 claim) |

**Never equate** microprice discovery IC p with overnight gap finiteness. research_only — **never live Sharpe**.


## Sergeant H27 ↔ gap ofi_p_ic ↔ gap_finite (**4/4 GREEN**)

| Path | `tests/unit/test_ofi_p_ic_vs_gap_finite_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `ofi_p_ic_vs_gap_finite_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H27 | `ofi_p_ic` |
| Gap | `gap_finite_rate` (no H27 claim) |

**Never equate** OFI discovery IC p with overnight gap finiteness. research_only — **never live Sharpe**.


## Lt H26 ↔ gap wick_skew_p_ic ↔ gap_finite (**4/4 GREEN**)

| Path | `tests/unit/test_wick_skew_p_ic_vs_gap_finite_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `wick_skew_p_ic_vs_gap_finite_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H26 | `wick_skew_p_ic` |
| Gap | `gap_finite_rate` (no H26 claim) |

**Never equate** wick-skew discovery IC p with overnight gap finiteness. research_only — **never live Sharpe**.


## CoS H30 ↔ gap clv_p_ic ↔ gap_finite (**4/4 GREEN**)

| Path | `tests/unit/test_clv_p_ic_vs_gap_finite_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `clv_p_ic_vs_gap_finite_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H30 | `clv_p_ic` |
| Gap | `gap_finite_rate` (no H30 claim) |

**Never equate** CLV discovery IC p with overnight gap finiteness. IC↔gap fan-out (H25/H27/H26/H30) complete beside H22↔gap. research_only — **never live Sharpe**.



## H32 ↔ gap vpin_p_ic ↔ gap_finite (**4/4 GREEN**)

| Path | `tests/unit/test_vpin_p_ic_vs_gap_finite_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `vpin_p_ic_vs_gap_finite_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H32 | `vpin_p_ic` |
| Gap | `gap_finite_rate` (no H32 claim) |

**Never equate** VPIN discovery IC p with overnight gap finiteness. research_only — **never live Sharpe**.


## Sergeant H28 ↔ gap dm_gk_vs_park_p ↔ gap_finite (**4/4 GREEN**)

| Path | `tests/unit/test_dm_gk_vs_park_p_vs_gap_finite_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `dm_gk_vs_park_p_vs_gap_finite_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H28 | `dm_gk_vs_park_p` |
| Gap | `gap_finite_rate` (no H28 claim) |

**Never equate** GK vs Park DM discovery p with overnight gap finiteness. research_only — **never live Sharpe**.


## CoS H31 ↔ gap dm_split_vs_park_p ↔ gap_finite (**4/4 GREEN**)

| Path | `tests/unit/test_dm_split_vs_park_p_vs_gap_finite_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `dm_split_vs_park_p_vs_gap_finite_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H31 | `dm_split_vs_park_p` |
| Gap | `gap_finite_rate` (no H31 claim) |

**Never equate** overnight-split vs Park DM discovery p with overnight gap finiteness. research_only — **never live Sharpe**.


## Lt H33 ↔ gap sweep_reject_signed_p_ic ↔ gap_finite (**4/4 GREEN**)

| Path | `tests/unit/test_sweep_reject_signed_p_ic_vs_gap_finite_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `sweep_reject_signed_p_ic_vs_gap_finite_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H33 | `sweep_reject_signed_p_ic` |
| Gap | `gap_finite_rate` (no H33 claim) |

**Never equate** sweep-reject discovery IC p with overnight gap finiteness. research_only — **never live Sharpe**.



## Lt H34 ↔ gap sweep_follow_signed_p_ic ↔ gap_finite (**4/4 GREEN**)

| Path | `tests/unit/test_sweep_follow_signed_p_ic_vs_gap_finite_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `sweep_follow_signed_p_ic_vs_gap_finite_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H34 | `sweep_follow_signed_p_ic` |
| Gap | `gap_finite_rate` (no H34 claim) |

**Never equate** sweep-follow discovery IC p with overnight gap finiteness. Completes reject/follow IC↔gap pair with H33. research_only — **never live Sharpe**.



## H35 ↔ gap sweep_reject_event_p ↔ gap_finite (**4/4 GREEN**)

| Path | `tests/unit/test_sweep_reject_event_p_vs_gap_finite_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `sweep_reject_event_p_vs_gap_finite_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H35 | `sweep_reject_event_p` |
| Gap | `gap_finite_rate` (no H35 claim) |

**Never equate** sweep-reject event discovery p with overnight gap finiteness. research_only — **never live Sharpe**.


## H36 ↔ gap sweep_follow_event_p ↔ gap_finite (**4/4 GREEN**)

| Path | `tests/unit/test_sweep_follow_event_p_vs_gap_finite_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `sweep_follow_event_p_vs_gap_finite_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H36 | `sweep_follow_event_p` |
| Gap | `gap_finite_rate` (no H36 claim) |

**Never equate** sweep-follow event discovery p with overnight gap finiteness. research_only — **never live Sharpe**.


## Lt H37 ↔ gap sweep_reject_placebo_p ↔ gap_finite (**4/4 GREEN**)

| Path | `tests/unit/test_sweep_reject_placebo_p_vs_gap_finite_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `sweep_reject_placebo_p_vs_gap_finite_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H37 | `sweep_reject_placebo_p` |
| Gap | `gap_finite_rate` (no H37 claim) |

**Never equate** sweep-reject placebo discovery p with overnight gap finiteness. research_only — **never live Sharpe**.


## Sergeant H38 ↔ gap sweep_follow_placebo_p ↔ gap_finite (**4/4 GREEN**)

| Path | `tests/unit/test_sweep_follow_placebo_p_vs_gap_finite_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `sweep_follow_placebo_p_vs_gap_finite_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H38 | `sweep_follow_placebo_p` |
| Gap | `gap_finite_rate` (no H38 claim) |

**Never equate** sweep-follow placebo discovery p with overnight gap finiteness. research_only — **never live Sharpe**.


## Lt H40 ↔ gap sweep_follow_cost_adjusted_mean_bps ↔ gap_finite (**4/4 GREEN**)

| Path | `tests/unit/test_sweep_follow_cost_adjusted_mean_bps_vs_gap_finite_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `sweep_follow_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H40 | `sweep_follow_cost_adjusted_mean_bps` |
| Gap | `gap_finite_rate` (no H40 claim) |

**Never equate** sweep-follow cost-adjusted mean bps bound with overnight gap finiteness. research_only — **never live Sharpe**.



## CoS H39 ↔ gap sweep_reject_cost_adjusted_mean_bps ↔ gap_finite (**4/4 GREEN**)

| Path | `tests/unit/test_sweep_reject_cost_adjusted_mean_bps_vs_gap_finite_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `sweep_reject_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H39 | `sweep_reject_cost_adjusted_mean_bps` |
| Gap | `gap_finite_rate` (no H39 claim) |

**Never equate** sweep-reject cost-adjusted mean bps bound with overnight gap finiteness. Completes cost pair with H40. research_only — **never live Sharpe**.



## Lt H41 ↔ gap sweep_reject_fold_positive_fraction ↔ gap_finite (**4/4 GREEN**)

| Path | `tests/unit/test_sweep_reject_fold_positive_fraction_vs_gap_finite_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `sweep_reject_fold_positive_fraction_vs_gap_finite_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H41 | `sweep_reject_fold_positive_fraction` |
| Gap | `gap_finite_rate` (no H41 claim) |

**Never equate** sweep-reject fold-positive stability fraction with overnight gap finiteness. research_only — **never live Sharpe**.


## Sergeant H42 ↔ gap sweep_follow_fold_positive_fraction ↔ gap_finite (**4/4 GREEN**)

| Path | `tests/unit/test_sweep_follow_fold_positive_fraction_vs_gap_finite_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `sweep_follow_fold_positive_fraction_vs_gap_finite_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H42 | `sweep_follow_fold_positive_fraction` |
| Gap | `gap_finite_rate` (no H42 claim) |

**Never equate** sweep-follow fold-positive stability fraction with overnight gap finiteness. research_only — **never live Sharpe**.


## CoS H44 ↔ gap sweep_reject_control_diff_p ↔ gap_finite (**4/4 GREEN**)

| Path | `tests/unit/test_sweep_reject_control_diff_p_vs_gap_finite_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `sweep_reject_control_diff_p_vs_gap_finite_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H44 | `sweep_reject_control_diff_p` |
| Gap | `gap_finite_rate` (no H44 claim) |

**Never equate** sweep-reject matched-control diff p with overnight gap finiteness. research_only — **never live Sharpe**.


## Lt H45 ↔ gap sweep_follow_control_diff_p ↔ gap_finite (**4/4 GREEN**)

| Path | `tests/unit/test_sweep_follow_control_diff_p_vs_gap_finite_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `sweep_follow_control_diff_p_vs_gap_finite_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H45 | `sweep_follow_control_diff_p` |
| Gap | `gap_finite_rate` (no H45 claim) |

**Never equate** sweep-follow matched-control diff p with overnight gap finiteness. Completes H33–H45 sweep↔gap discovery/bound mesh; H43 session_book_vpin↔gap is a separate IC↔gap edge (stamped below). research_only — **never live Sharpe**.



## Lt H43 ↔ gap session_book_vpin_p_ic ↔ gap_finite (**4/4 GREEN**)

| Path | `tests/unit/test_session_book_vpin_p_ic_vs_gap_finite_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `session_book_vpin_p_ic_vs_gap_finite_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H43 | `session_book_vpin_p_ic` |
| Gap | `gap_finite_rate` (no H43 claim) |

**Never equate** session-book VPIN discovery IC p with overnight gap finiteness. Distinct from H32 `vpin_p_ic`↔gap. Completes IC↔gap SPECS ladder **H25–H45** (+H43). research_only — **never live Sharpe**.



## Sergeant H33 ≠ H34 sweep_reject_signed_p_ic ↔ sweep_follow_signed_p_ic (**4/4 GREEN**)

| Path | `tests/unit/test_sweep_reject_signed_p_ic_vs_sweep_follow_signed_p_ic_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `sweep_reject_signed_p_ic_vs_sweep_follow_signed_p_ic_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H33 | `sweep_reject_signed_p_ic` |
| H34 | `sweep_follow_signed_p_ic` |

**Never equate** reject vs follow signed-IC discovery. Sibling ladder start. research_only — **never live Sharpe**.


## Lt H35 ≠ H36 sweep_reject_event_p ↔ sweep_follow_event_p (**4/4 GREEN**)

| Path | `tests/unit/test_sweep_reject_event_p_vs_sweep_follow_event_p_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `sweep_reject_event_p_vs_sweep_follow_event_p_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H35 | `sweep_reject_event_p` |
| H36 | `sweep_follow_event_p` |

**Never equate** reject vs follow event discovery p. research_only — **never live Sharpe**.


## Sergeant H37 ≠ H38 sweep_reject_placebo_p ↔ sweep_follow_placebo_p (**4/4 GREEN**)

| Path | `tests/unit/test_sweep_reject_placebo_p_vs_sweep_follow_placebo_p_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `sweep_reject_placebo_p_vs_sweep_follow_placebo_p_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H37 | `sweep_reject_placebo_p` |
| H38 | `sweep_follow_placebo_p` |

**Never equate** reject vs follow placebo discovery p. research_only — **never live Sharpe**.


## CoS H39 ≠ H40 sweep_reject_cost ↔ sweep_follow_cost (**4/4 GREEN**)

| Path | `tests/unit/test_sweep_reject_cost_adjusted_mean_bps_vs_sweep_follow_cost_adjusted_mean_bps_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `sweep_reject_cost_adjusted_mean_bps_vs_sweep_follow_cost_adjusted_mean_bps_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H39 | `sweep_reject_cost_adjusted_mean_bps` |
| H40 | `sweep_follow_cost_adjusted_mean_bps` |

**Never equate** reject vs follow cost-adjusted mean bps bounds. Completes signed/event/placebo/cost sibling rung; fold/control (H41≠H42 / H44≠H45) stamped below — ladder COMPLETE. research_only — **never live Sharpe**.



## CoS candle spread_bps alias vs spread_over_mid (**4/4 GREEN**)

| Path | `tests/unit/test_candle_spread_bps_vs_over_mid_identity.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `candle_spread_alias_honesty_errors` (registered) |

| Axis | Rule |
|---|---|
| `mean_spread_bps` | ≈ `1e4 * mean_spread_over_mid` when both finite |
| Fail-closed | `mean_spread_bps_not_1e4_times_mean_spread_over_mid` |
| Partial | one key absent ⇒ skip |

book_metrics: `spread_bps = 1e4 * spread/mid`; `spread_over_mid = spread/mid`. research_only — **never live Sharpe**.


## Lt H41 ≠ H42 reject_fold ↔ follow_fold (**4/4 GREEN**)

| Path | `tests/unit/test_sweep_reject_fold_positive_fraction_vs_sweep_follow_fold_positive_fraction_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `sweep_reject_fold_positive_fraction_vs_sweep_follow_fold_positive_fraction_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H41 | `sweep_reject_fold_positive_fraction` |
| H42 | `sweep_follow_fold_positive_fraction` |

**Never equate** reject vs follow fold-positive stability fractions. research_only — **never live Sharpe**.


## Lt H44 ≠ H45 reject_control ↔ follow_control (**4/4 GREEN**)

| Path | `tests/unit/test_sweep_reject_control_diff_p_vs_sweep_follow_control_diff_p_never_equate.py` |
|---|---|
| Status | **4/4 GREEN** (on-disk confirm) |
| Helper | `sweep_reject_control_diff_p_vs_sweep_follow_control_diff_p_never_equate_honesty_errors` (registered) |

| Axis | Key / id |
|---|---|
| H44 | `sweep_reject_control_diff_p` |
| H45 | `sweep_follow_control_diff_p` |

**Never equate** reject vs follow matched-control diff p. Completes reject≠follow sibling ladder **H33≠H34 … H44≠H45** (H43 off sibling axis). research_only — **never live Sharpe**.


## Commander Residual #246–#255 — Mac identity batch

| Status | On-disk residual batch (Commander lane) |
|---|---|
| Scope | half_spread; mid; microprice; μ-mid[+bps]; spread_bps; depth_imbalance_abs; tops≤depths; ask>bid |

Documented from day_grind; no invent beyond listed props. Cross-ref IC↔gap SPECS closed; sibling ladder complete.


## CoS candle_dir_x_imbalance + close_mid_abs_rel (**3/3 GREEN**)

| Path | `tests/unit/test_candle_dir_x_imbalance_close_mid_join_honesty.py` |
|---|---|
| Status | **3/3 GREEN** |
| Helpers | `mean_candle_dir_x_imbalance_honesty_errors`, `mean_close_mid_abs_rel_honesty_errors`; companion `candle_join_coverage_and_chain_honesty_errors` |

| Key | Rule |
|---|---|
| `mean_candle_dir_x_imbalance` | ∈[-1,1] |
| `mean_close_mid_abs_rel` | ≥0 |
| join/chain | honesty companion on synth candle |

research_only — **never live Sharpe**.


## CoS tob / concentration / tick_spacing (**3/3 GREEN**)

| Path | `tests/unit/test_candle_tob_concentration_tick_spacing_honesty.py` |
|---|---|
| Status | **3/3 GREEN** |
| Helpers | `mean_tob_size_share_honesty_errors`, `size_concentration_top_honesty_errors`, `candle_log_tick_spacing_finite_honesty_errors` (wired in `verify.py`) |

| Key | Rule |
|---|---|
| `mean_tob_size_share` | ∈**(0,1]**; `0.0` → `mean_tob_size_share_out_of_open_unit_interval` |
| concentration tops (e.g. `mean_bid_size_concentration_top`) | ∈**(0,1]** |
| log tick spacing (e.g. `mean_bid_mean_log_tick_spacing`) | finite |

Synth candle pack clean (+ IC⇒mean companion). research_only — **never live Sharpe**.

### Confirmed / refreshed this wave

| Item | Status |
|---|---|
| CoS spread_bps + log slopes | **3/3 GREEN** (`test_candle_spread_bps_and_log_slopes_honesty.py`) |
| Sergeant H20 ↔ H22 never-equate | **4/4 GREEN** on-disk; triad diagonal complete; combined H20–H21–H22 edge suites **12/12** (`test_ohlc_identity_vs_imbalance_p_ic_never_equate.py` + H20↔H21 + H21↔H22) |


## CoS yang_zhang / overnight / QLIKE pack (**5/5 GREEN**)

| Path | `tests/unit/test_yz_park_overnight_rv_bv_semi_pack.py` |
|---|---|
| Status | **5/5 GREEN** (CoS reported) |
| Helpers | `northset_range_spread_honesty_errors` (YZ), `northset_qlike_means_honesty_errors` (park/gk/rs), `northset_overnight_rv_semi_honesty_errors` |

| Family | Rule |
|---|---|
| `yang_zhang_variance` | ≥0 |
| park / gk / rs `*_qlike_vs_cc` | ≥0 |
| overnight RV/BV/semi (e.g. `session_mean_rv`, `session_mean_bv`, `semi_up`) | ≥0 |

Synth pack clean + helpers registered. research_only — **never live Sharpe**.

### Confirmed earlier this wave

| Item | Status |
|---|---|
| CoS candle mean_*_frac pack | **3/3 GREEN** (`test_candle_frac_and_spread_x_honesty.py`) |


## CoS IC t/mean/rank + finite_rate/floor catchalls (**2/2 GREEN**)

| Path | `tests/unit/test_ic_finite_rate_floor_catchalls.py` |
|---|---|
| Status | **2/2 GREEN** |
| Helpers | `northset_all_t_ic_finite_honesty_errors`; `northset_all_mean_ic_finite_honesty_errors`; `northset_all_mean_rank_ic_unit_honesty_errors`; `northset_all_finite_rate_unit_honesty_errors`; `northset_all_floor_unit_honesty_errors` |
| Coverage | catch-alls flag bad values; clean on synth + wired into receipt honesty |

Contract: `*_t_ic` / `*_mean_ic` finite when present; `*_mean_rank_ic` ∈[-1,1]; `*_finite_rate` / `*_floor` ∈[0,1]. research_only — **never live Sharpe**.

### Confirmed: Sergeant candle_order_book sizing + depth≥1 (**8/8 GREEN**)

Already documented: `test_candle_order_book_sizing_soft_verify.py` **8/8**.


## Sergeant receipt bool-flags soft-verify (**6/6 GREEN**)

| Path | `tests/unit/test_northset_receipt_bool_flags_soft_verify.py` |
|---|---|
| Helper | `northset_receipt_bool_flags_honesty_errors` |
| Status | **6/6 GREEN** |
| Coverage | synth bool flags clean; `metrics_required_finite` ok≠bool fail-closed; `shape_columns_ensured` str fail-closed; `use_session_l2` None fail-closed; absent keys skipped; helper in receipt dispatcher |

research_only — **never live Sharpe**.

## CoS candle claim + pearson ∈[-1,1] + northset string enums (**4/4 GREEN**)

| Path | `tests/unit/test_candle_claim_pearson_and_northset_enums.py` |
|---|---|
| Status | **4/4 GREEN** |
| Helpers | `candle_order_book_claim_honesty_errors`; `candle_all_ic_pearson_unit_honesty_errors`; `northset_receipt_string_enum_honesty_errors` |
| Coverage | candle claim + Pearson IC ∈[-1,1] on synth; northset string enum honesty |

### Confirmed: CoS IC t/mean/rank + finite_rate/floor catchalls (**2/2 GREEN**)

Already documented: `test_ic_finite_rate_floor_catchalls.py` **2/2**.


## CoS imbalance/CLV/microprice IC packs + ofi/queue/slope means (**3/3 GREEN**)

| Path | `tests/unit/test_imbalance_clv_microprice_ic_packs_and_ofi_means.py` |
|---|---|
| Status | **3/3 GREEN** |
| Helpers | `northset_imbalance_top_ic_pack_honesty_errors`; `northset_clv_ic_pack_honesty_errors`; `northset_microprice_bps_ic_pack_honesty_errors`; `candle_ofi_qp_slope_ic_implies_mean_honesty_errors` |
| Coverage | candle ofi / queue_priority / slope means + IC⇒mean; IC pack bounds on synth; verify wire |

Confirmed earlier: CoS queue/vpin IC + structure IC⇒mean **4/4** (`test_queue_vpin_ic_and_structure_means.py`) already in docs.


## Sergeant `session_book_snaps`↔`n_session` consistency — **PROMOTED GREEN 7/7** (INFLIGHT cleared)

| Path | `tests/unit/test_session_book_snaps_n_session_consistency.py` |
|---|---|
| Helper | `northset_session_book_snaps_n_session_consistency_errors` |
| Status | **7/7 GREEN** (NOT INFLIGHT) |
| Coverage | synth session_l2-on consistent; l2-off skip; rows 0/missing despite mean fail-closed; rows≠candles fail-closed; mean NaN despite rows fail-closed; helper in session-means dispatcher |

Never equate `mean_session_book_snaps` ↔ `n_session_book_rows` / `n_session_candles` without the consistency helper.

### Status board

| Item | Status |
|---|---|
| Candle `ic_method` HAC soft-verify | **6/6 GREEN** — not INFLIGHT |
| CoS spread_over_mid FEATURE_COLS + session_ofi IC | **4/4 GREEN** |
| CoS depth_imbalance_abs + ofi_lag IC | **4/4 GREEN** |
| Sergeant snaps vs n_session | **7/7 GREEN** — promoted |

**No True INFLIGHT** in this lane — IDLE awaiting next assign. Off inventing.


## Sergeant candle `ic_method` HAC soft-verify (**6/6 GREEN**)

Confirmed: `candle_order_book_ic_method_honesty_errors` — `tests/unit/test_candle_order_book_ic_method_soft_verify.py` **6/6**. See prior section.

### True INFLIGHT now (Sergeant)

**None** — snaps vs n_session **promoted GREEN 7/7**. See below.


(Replaces broader snaps/uncrossed/HAC meta parking with this concrete assign.)


## CoS session_l2_enforced + notional IC⇒mean (**5/5 GREEN**)

| Path | `tests/unit/test_session_l2_enforced_and_notional_ic_mean.py` |
|---|---|
| Status | **5/5 GREEN** |
| Coverage | enforced gate requires session identity rate keys; synth session_l2-on stamps identity rates; candle stamps `mean_notional_imbalance` and IC⇒mean; honesty flags missing mean; verify wires |

Cross-ref: CoS MWB fuse **6/6**; Sergeant session L2 identity + candle book_age **9/9** (already GREEN — not INFLIGHT).

### True INFLIGHT now (Sergeant)

**None** — snaps vs n_session **promoted GREEN 7/7**. See below.



## Commander residual #161–#175 (Mac property continuous)

| Batch | Themes |
|---|---|
| **#161–#170** | depth1 tob shares == 1; tops/depths > 0; concentration ∈(0,1]; half_spread > 0; microprice ∈[bid,ask]; weight ∈[0,1]; notional_imbalance ∈[-1,1]; depth_imbalance_abs ≥ 0; `assert_metrics_required_finite` |
| **#171–#175** | SIDE_NOTIONAL / QUEUE / TOB_SHARE / SIDE_STRUCTURE field presence + bounds; all values float |

Lt owns box merge; Mac property continuous stayed off merge. Docs cross-link only live locks — do not invent formulas ahead of Commander.


## Commander box #73–#100 → Mac merge — **SKIPPED** (Lt)

Lieutenant **SKIPPED** merging Commander box property-work **#73–#100** onto Mac.

| Fact | Detail |
|---|---|
| Decision | **SKIPPED** — do not apply box #73–#100 tarball/merge onto Mac |
| Why | Mac property continuous already through **#175**; box mirror was **behind** Mac |
| Docs | Prefer Mac-live locks; do not re-introduce older box residual wording as authoritative |
| Follow-ons | Commander **#161–#175** (and later) are the Mac-ahead batches — see that section |

Off inventing. If box later catches up, Lt re-evaluates merge — not automatic.


## Sergeant free-lane: `mean_tob_notional_share` soft-verify + receipt stamp

| Piece | Status |
|---|---|
| Receipt stamp | `bench_northset` stamps `mean_tob_notional_share` = nanmean(`tob_notional_share`) |
| Soft-verify helper | `mean_tob_notional_share_honesty_errors` — finite ⇒ ∈ **(0, 1]**; ±inf / 0 fail-closed; NaN/absent skip |
| Never equate | ≠ `mean_tob_size_share` (Commander #62 — size vs notional) |
| `verify-research` wire | **Direct** in `research.verify` on family northset (`for mtn_err in mean_tob_notional_share_honesty_errors(...)`) — **not** inside `NORTHSET_RECEIPT_HONESTY_HELPERS` (40) today |
| Unit tests | `tests/unit/test_mean_tob_notional_share_receipt_stamp.py` — source stamp assert; honesty bounds; verify.py wire assert. Soft-verify twin: `test_mean_tob_notional_share_soft_verify.py` |
| Property | `tests/property/test_order_book_metrics_identity.py` — `tob_notional_share` ∈(0,1] geometry |

Cross-ref: NORTHSET receipt means **Notional / TOB share**; **mean_tob_size_share vs tob_size_share_finite_rate**.

## Kyle nest soft-verify suite (`kyle_ofi_nest_honesty_errors`)

**Live** on `verify-research` for family **northset** when `receipt["kyle_ofi"]` is present (or a bare kyle-shaped blob). Dispatcher `kyle_ofi_nest_honesty_errors` fans **26** honesty helpers (Sergeant nest sync pack). **`test_kyle_ofi.py` 47/47** green on Mac after sync — suite **completeness cross-link** (tests ⊇ helpers + fuse/CLI paths). **Still not** a soft-verify **H-row** mint source (no kyle_* hypotheses) — honesty/labels only. research_only — **never live Sharpe**.

| Helper | Checks (short) |
|---|---|
| `kyle_residual_flow_honesty_errors` | residual_* spearman → research_only + claim + companions; no Sharpe/pnl |
| `kyle_lambda_dispersion_honesty_errors` | p50 / window markers → research_only + claim |
| `kyle_lambda_ofi_depth_corr_honesty_errors` | ofi↔depth corr → claim + n_dates / HAC companions |
| `kyle_ofi_join_coverage_honesty_errors` | finite join → open-unit + nonempty `book_source` |
| `kyle_lambda_date_series_honesty_errors` | date_series_n_* → research_only + claim |
| `kyle_ofi_synthetic_source_honesty_errors` | synthetic_lob ⇔ SYNTHETIC; dgp match |
| `kyle_ofi_label_synthetic_honesty_errors` | SYN* label ↔ data_source / dgp |
| `kyle_ofi_nest_claim_honesty_errors` | umbrella research_only + `claim=research_diagnostic_only` |
| `kyle_ofi_ic_method_honesty_errors` | `ic_method=date_level_spearman_hac` when IC present |
| `kyle_ofi_family_honesty_errors` | `family=kyle_ofi` |
| `kyle_ofi_hac_lags_honesty_errors` | hac_lags ≥0 int; rolling lo≤hi |
| `kyle_ofi_min_names_honesty_errors` | min_names ≥1 int when benchish |
| `kyle_ofi_n_fused_scored_honesty_errors` | n_scored ≤ n_fused; companions |
| `kyle_ofi_book_panel_path_honesty_errors` | if path set → nonempty string |
| `kyle_ofi_min_join_coverage_pair_honesty_errors` | floor ∈[0,1]; join ≥ floor when both finite |
| `kyle_ofi_dispersion_window_honesty_errors` | dispersion_window ≥1 int when markers |
| `kyle_ofi_diagnostic_string_honesty_errors` | optional diagnostic nonempty, no forbidden tokens |
| `kyle_ofi_lambda_decile_order_honesty_errors` | p10≤p50≤p90 per side when all finite |
| `kyle_ofi_std_iqr_honesty_errors` | λ std/iqr ≥0 when finite |
| `kyle_ofi_rolling_mean_hac_band_honesty_errors` | rolling mean finite when lo/hi finite; lo≤hi |
| `kyle_ofi_n_dates_companion_honesty_errors` | finite spearman/t ⇒ n_dates ≥1 |
| `kyle_ofi_pvalue_honesty_errors` | `*_p` ∈[0,1] when finite |
| `kyle_ofi_spearman_honesty_errors` | spearman/pearson ∈[-1,1] when finite |
| `kyle_ofi_tstat_honesty_errors` | `*_t` finite when present; spearman⇒t companion |
| `kyle_ofi_label_nonempty_honesty_errors` | label nonempty when diagnostics present |
| `kyle_ofi_date_series_counts_honesty_errors` | date_series_n_* ≥0 ints when present |

Ops: Sergeant **47/47** nest sync completeness. Never-equate index below + parking-lot backlog — do not invent further helpers in docs before code.

### Never equate / stance

| Rule | Detail |
|---|---|
| Nest honesty ≠ H-row mint | Suite checks labels/companions; does **not** invent kyle H-ids |
| Nest `kyle_lambda_ofi_mean` ≠ always-on `kyle_ofi_lambda` | Date-cross-section λ + HAC ≠ name-mean OLS; also `n_dates` ≠ `n_securities`, nest HAC ≠ always-on R² — see **Always-on kyle_ofi_lambda vs nest** |
| Nest join_coverage ≠ northset join_coverage | Separate fuse path — **Nest join_coverage vs always-on join_coverage** |
| Sweep honesty ≠ kyle nest | Sweep helpers do not touch `kyle_*` |
| residual_ofi ≠ ofi_flow / ofi_fwd | Residualized+fwd ≠ raw contemporaneous/predictive — **Nest residual_ofi vs ofi_flow IC** |
| kyle_lambda_*_fwd_* ≠ contemporaneous λ means | Forward-target OLS λ ≠ default delta_mid λ — **Nest kyle_lambda_*_fwd_*** |
| nest ofi_fwd ≠ always-on ofi_p_ic | Kyle fuse IC ≠ H27 discovery — **Nest ofi_fwd vs always-on ofi_p_ic** |
| ofi_lag1_corr / ofi_lag_*_ic ≠ nest ofi_delta_mid_lag1 | Panel AR / always-on IC ≠ nest lag1 Δmid IC — **Always-on ofi_lag panel vs nest** |
| mid_lag1_corr ≠ ofi_lag1_corr | Same panel AR shape, different series — **Always-on mid_lag1_corr vs ofi_lag1_corr** |
| always-on kyle_r2 ≠ nest HAC t/p | Name-mean OLS R² ≠ date-λ HAC — **Always-on kyle_r2 vs nest HAC** |
| always-on n_securities ≠ nest n_dates | Name axis ≠ date axis — **Always-on n_securities vs nest n_dates** |
| nest join_coverage ≠ always-on join_coverage | Kyle fuse rate ≠ asof attach — **Nest join_coverage vs always-on** |
| nest n_fused/n_scored ≠ always-on sizing | Kyle fuse rows ≠ asof fuse / fwd_ret_1 sample — **Nest n_fused/n_scored vs always-on** |
| nest hac_lags ≠ always-on IC HAC echo | Optional nest stamp; always-on has no twin key — **Nest hac_lags vs always-on** |
| nest book_panel_path ≠ always-on path blindly | Audit echo; same string ≠ same fuse; CLI may diverge — **Nest book_panel_path vs always-on** |
| nest SYN* label ≠ always-on MIXED | Nest label path skips MIXED branch — **Nest SYN* vs always-on MIXED** |
| nest min_names ≠ always-on min_names echo | Nest stamps; main blob usually config-only; CLI may use 3 — **Nest min_names vs always-on** |
| nest book_source/book_dgp ≠ always-on provenance | Independent fuse + label matrix — **Nest book_source/book_dgp vs always-on** |
| kyle_lambda_ofi_depth corr ≠ dispersion | λ↔λ corr ≠ within-flow quantiles/rolling — **Nest kyle_lambda_ofi_depth corr vs dispersion** |
| kyle λ dispersion ≠ IC companions | Deciles/rolling HAC on λ series ≠ depth_flow/ofi_flow — **Nest Kyle λ dispersion vs IC** |
| depth_flow ≠ kyle_lambda_depth_mean | Spearman IC ≠ OLS λ (ofi twin too) — **Nest depth_flow vs kyle_lambda_depth_mean** |
| signed_depth_fwd ≠ depth_flow | Fwd-target IC ≠ contemporaneous companion — **Nest signed_depth_fwd vs depth_flow** |
| signed_depth_lag1 ≠ depth_flow | Depth lag twin — **Nest signed_depth_delta_mid_lag1 vs depth_flow** |
| ofi_delta_mid_lag* ≠ ofi_flow | Lag axis / separate helper — **Nest ofi_delta_mid_lag0/lag1 vs ofi_flow** |

Cross-ref: NORTHSET **Kyle / OFI family**; **Always-on fuse vs kyle_ofi fuse columns**.

## Always-on `kyle_ofi_lambda` vs nest `kyle_lambda_ofi_mean` (never equate)

Two Kyle λ surfaces on the northset receipt — **different estimators**. Soft-verify does **not** require numeric agreement.

| Always-on (main receipt; flag irrelevant) | Nested `receipt["kyle_ofi"]` (`include_kyle_ofi=true`) | Why not interchangeable |
|---|---|---|
| `kyle_ofi_lambda` | `kyle_lambda_ofi_mean` (+ `_t` / `_p` / `_n_dates`) | Name-mean OLS λ on fused `ofi` (`_panel_kyle`) vs **date**-cross-section λ then mean/HAC on the λ series (`kyle_lambda_by_date`) |
| `kyle_ofi_r2` / `kyle_r2` | nest HAC t/p (full or rolling) — **not** an R² twin | See **Always-on kyle_r2 / kyle_ofi_r2 vs nest HAC t/p** |
| `kyle_ofi_n_securities` | `kyle_lambda_ofi_n_dates` / `kyle_lambda_date_series_n_ofi` | Names vs dates — **Always-on n_securities vs nest n_dates** |
| `kyle_lambda` / `kyle_r2` | `kyle_lambda_depth_mean` (+ companions) | Same split for `signed_volume` (always-on) vs nest `signed_depth` flow alias |
| Absent when? | Nest absent when `include_kyle_ofi=false` | Always-on scalars **remain**; comparing a missing nest key to always-on is a doc/ops error |

**Fuse alias:** always-on labels imbalance `(bid_depth - ask_depth)` as `signed_volume`; nest / `kyle_ofi` fuse uses `signed_depth` — same formula, different column name.

**Soft-verify:** nest honesty suite (`kyle_ofi_nest_honesty_errors`) gates labels/companions on the nest; it does **not** assert `kyle_ofi_lambda ≈ kyle_lambda_ofi_mean`. Always-on scalars are still not H-row sources.

Cross-ref: NORTHSET **Always-on scalars vs nested kyle_ofi blob**; OPS **`--dump-lambda-series` / date_series honesty**.




## Evidence label / `component_sources` / `sweep_evidence_scope` matrix

All stamped by `bench_northset` (`northset.benches`). Honesty metadata — not live Sharpe or promotion evidence.

### Inputs

| Symbol | Definition |
|---|---|
| `bar_source` | `str(config.data.source)` (e.g. `synthetic`, vendor name) |
| `bars_synthetic` | `bar_source.lower() == "synthetic"` |
| `book_source` | Panel `source` or synthesizer tag (`synthetic_lob`, …) |
| `book_synthetic` | `book_source.lower()` ∈ `{synthetic, synthetic_lob, synthetic_reconstruction}` |
| `session_synthetic` | `northset.use_session_l2` (default true) — enables session L2 path |
| `price_basis` | From `canonical_northset_bars` / `NorthsetMarketView` |

### `component_sources` (always)

| Key | Value |
|---|---|
| `bars` | `bar_source` |
| `book` | `book_source` |
| `session_candles` | always `synthetic_reconstruction` (never a vendor RTH tape) |
| `session_book` | `synthetic_reconstruction` if `use_session_l2` else `disabled` |

Session candle/book provenance is **always** reconstructed plumbing in this dict — even when receipt `label` is an empirical vendor string (vendor book + empirical bars path).

### `label` / `data_source` / `dgp` branch order

Exact order in code:

1. **`bars_synthetic and book_synthetic`** → `label=data_source=SYNTHETIC`, `dgp=synthetic_lob`
2. **`not book_synthetic`** (vendor/external book) → `label=data_source = SYNTHETIC if bars_synthetic else bar_source`, `dgp=book_dgp` (vendor panel DGP wins over a vague mix tag)
3. **else if `bars_synthetic or book_synthetic or session_synthetic`** → `label=data_source=MIXED_SYNTHETIC_DERIVED`, `dgp=mixed_sources`
4. **else** → `label=data_source=bar_source`, `dgp=empirical`

### Practical matrix

| Bars | Book | Session L2 | Typical `label` / `data_source` | `dgp` | Notes |
|---|---|---|---|---|---|
| SYNTHETIC | SYNTHETIC LOB | on/off | `SYNTHETIC` | `synthetic_lob` | Branch 1 |
| SYNTHETIC | vendor/external | on/off | `SYNTHETIC` | `book_dgp` (vendor) | Branch 2 — book provenance authoritative |
| empirical | vendor/external | on/off | `bar_source` (e.g. alpaca) | `book_dgp` | Branch 2 — **not** MIXED even if session_candles are reconstructed |
| empirical | SYNTHETIC LOB | on/off | `MIXED_SYNTHETIC_DERIVED` | `mixed_sources` | Branch 3 — synth book on real bars |
| empirical | SYNTHETIC LOB | off | `MIXED_SYNTHETIC_DERIVED` | `mixed_sources` | Still MIXED (`book_synthetic`) |
| empirical | vendor | off | `bar_source` | `book_dgp` / else empirical only if somehow no synth flags | Branch 2 or 4 |

Branch 3 also fires if bars were synth but book synth was already handled by branch 1; the residual MIXED case in practice is **empirical bars + synthetic book** (session flag alone cannot MIX when book is vendor — see branch 2).

### `sweep_evidence_scope`

Independent of MIXED label; driven by **bars** + **price_basis**:

| Condition | `sweep_evidence_scope` |
|---|---|
| `bars_synthetic` | `synthetic` |
| else and `price_basis == split_adjusted` | `empirical_adjusted` |
| else (`raw_fixture_opt_out`) | `fixture_raw_unadjusted` |

### Forbidden claims

- Do **not** treat `SYNTHETIC` or `MIXED_SYNTHETIC_DERIVED` receipts as live / capacity / promotion evidence.
- Do **not** describe MIXED runs as pure empirical microstructure (synthetic book or mix is in the stack).
- Do **not** treat `component_sources.session_candles=synthetic_reconstruction` as a vendor intraday tape — even when `label` is an empirical vendor string.
- Do **not** read `sweep_evidence_scope=synthetic` or `fixture_raw_unadjusted` as corp-action-safe empirical sweep alpha.
- Vendor book on empirical bars may still show reconstructed session keys in `component_sources`; that does **not** by itself force `MIXED_SYNTHETIC_DERIVED` (branch 2).
- Pair with eligibility stamps: MIXED empirical+synth-book → `book_hypothesis_eligible=false` (see eligibility section).


## `price_basis` / `return_basis` stamp matrix

Setter: `canonical_northset_bars` → `NorthsetMarketView`; copied onto the northset receipt by `bench_northset`.

| Path | Gate | `price_basis` | `return_basis` | Frame effects |
|---|---|---|---|---|
| Full split-adjusted OHLC quartet present | default / `require_adjusted_ohlc=true` | `split_adjusted` | `total_return` if `close_total_return` present, else `split_adjusted` | Overwrites `open/high/low/close` with `*_split_adjusted`; materializes `return_open` / `return_close`; volume prefers `volume_split_adjusted` else `volume*split_factor` else raw |
| Partial adjusted quartet | any | **fail-closed** `ValueError` | — | — |
| No adjusted cols + `require_adjusted=true` | default | **fail-closed** | — | — |
| No adjusted cols + `require_adjusted=false` | fixture opt-out only | `raw_fixture_opt_out` | `raw_fixture_opt_out` | `return_open`/`return_close` alias raw open/close |
| Adjusted frame with `ohlc_identity_rate < 1` | any | **fail-closed** | — | — |

**Claims**

- Split-adjusted prices drive candle geometry and sweep detection; TR close drives forward exits when present.
- `raw_fixture_opt_out` is visible honesty for narrow fixtures without corp actions — **not** empirical adjusted evidence (`sweep_evidence_scope` becomes `fixture_raw_unadjusted` when bars are non-synthetic).
- These keys are receipt metadata beside `label` / `dgp` / `component_sources`; they do not authorize live trading.

Config: `northset.require_adjusted_ohlc` (default true). Synthetic fixtures typically run with require_adjusted false or provide adjusted cols per lab setup.



## Family `dgp` / receipt `book_dgp` (and fuse-frame cousin)

Honesty provenance — not alpha. Two layers:

| Layer | Setter | Key | Typical values |
|---|---|---|---|
| **Receipt** | `bench_northset` | `book_dgp`, `dgp` (= `family_dgp`) | see below |
| **Fuse frame** | `attach_candle_book_features` | column `book_dgp` / `book_source` | `synthetic_lob` or `external_panel` |

### Receipt `book_dgp` (set before fuse, then stamped on receipt)

| Condition | `book_dgp` |
|---|---|
| `northset.book_panel_path` set (loaded panel) | `vendor_panel:{book_source}` where `book_source` is panel `source[0]` (else `parquet`) |
| else (synthesize L2 from bars) | `synthetic_lob` |

### Family `dgp` (= receipt `dgp`) — same branch order as evidence `label`

| Branch | `dgp` (`family_dgp`) |
|---|---|
| synth bars + synth book | `synthetic_lob` |
| vendor/external book (`not book_synthetic`) | **`book_dgp`** (so usually `vendor_panel:…`) — book provenance wins |
| else MIXED path | `mixed_sources` |
| else pure empirical | `empirical` |

### Fuse-frame `book_dgp` (attach)

| Condition | Frame `book_dgp` |
|---|---|
| `book is None` (synthesize inside attach) | `synthetic_lob` |
| external panel, `source` ∈ `{synthetic, synthetic_lob}` | `synthetic_lob` |
| external panel, other single `source` | `external_panel` |
| missing / null / **mixed** `source` values | **fail-closed** `ValueError` (refuse silent DGP mix) |

**String mismatch is intentional:** receipt may say `vendor_panel:alpaca` while the fused frame column says `external_panel`. Do not equate the strings; both mean “not synthetic_lob synthesizer output.” Prefer receipt `book_dgp` + `component_sources.book` for notebook honesty; frame stamps block silent mix inside the fuse.

### Forbidden claim mix

- Do not report `dgp=synthetic_lob` as vendor tape evidence.
- Do not report `dgp=vendor_panel:*` / frame `external_panel` as pure SYNTHETIC LOB.
- Do not report `dgp=mixed_sources` as clean empirical microstructure.
- Do not invent a blended DGP string — attach refuses multi-`source` panels.
- Family `dgp` following vendor `book_dgp` while `label=SYNTHETIC` (synth bars + vendor book) is allowed: candle label SYNTHETIC, book DGP authoritative — do not collapse to `mixed_sources` in prose.


## `join_coverage` / `book_age` receipt fields (ops)

Produced by `attach_candle_book_features`; summarized onto the northset receipt by `bench_northset`.

### Definitions

| Field | Meaning |
|---|---|
| `join_coverage` | `n_fused / n_candle_rows` after asof PIT join (also literal column on every fused row) |
| `book_age_seconds` | per-row `(decision_time - book_event_time)` in seconds |
| Receipt `mean_book_age_seconds` / `max_book_age_seconds` | nanmean / nanmax of fused ages |
| Receipt `book_join_coverage_floor` | config floor echoed for audit |

Join: Polars `join_asof` on `decision_time` ↔ `book_available_time`, `by=security_id`, `strategy=backward`, `tolerance=max_book_age_seconds`. Rows with null book after asof are dropped before coverage.

### Config / defaults

| Knob | Default | Role |
|---|---|---|
| `northset.book_max_age_seconds` | `86400` (≥ 1) | asof tolerance + stale fail-closed ceiling |
| `northset.book_join_coverage_floor` | `0.5` ∈ [0,1] | floor passed as `min_join_coverage` when **external** `book_panel_path` set |

Attach defaults when `min_join_coverage` is `None`: **1.0** if book synthesized inside attach, **0.5** if external book passed. `bench_northset` passes `min_join_coverage=book_join_coverage_floor` only when `book_panel_path` is set; synth path leaves `None` → attach’s 1.0 default.

### Fail-closed (data-contract bugs, not alpha)

| Check | Where | Trigger |
|---|---|---|
| Empty fuse | attach | no matching book within tolerance |
| Coverage &lt; floor | attach | `join_coverage < min_join_coverage` |
| Coverage &lt; floor (again) | `bench_northset` | external path only: NaN or `join_coverage < book_join_coverage_floor` |
| PIT lookahead | attach | `book_age_seconds < 0` (book_event after decision) |
| Stale book | attach | `book_age_seconds > max_book_age_seconds` |

### Ops notes

- Low coverage / stale ages → fix panel clocks, availability stamps, or loosen floors **only** via config; do not edit receipt scalars to pass.
- Synth lab path should stay at coverage ≈ 1.0; external panels may use 0.5+ and must still pass age/lookahead gates.
- CLI: `dipcatcher northset` surfaces `join_coverage` / age stamps on the receipt; failures raise before a clean notebook write.
- Nested kyle fuse has its own join coverage rules (`kyle_ofi.fuse_bars_l2_kyle_frame`) — see fuse alias / kyle sections; do not assume identical floors.

## `vpin_mean` soft-verify (daily ≠ session)

**CoS live.** Daily fused VPIN mean on the northset receipt — **not** session book VPIN.

| Layer | Detail |
|---|---|
| Receipt | `vpin_mean` (always-on / daily fuse path) |
| Soft-verify | `vpin_mean_honesty_errors` on family **`northset`**: finite → **∈ [0, 1]**; NaN skip; error `vpin_mean_out_of_unit_interval` |
| Hypothesis | **H32** discovery (gated by `book_hypothesis_eligible`) |

### Never equate

| Object | Clock / role |
|---|---|
| `vpin_mean` | Daily fused VPIN mean — H32 |
| `session_book_vpin_mean` | Session-path VPIN mean — H43; soft-verify separate (`northset_session_book_vpin_mean_honesty_errors`) |
| `session_book_vpin_p_ic` | Session fuse-col IC p — H43 gate, not the daily mean |
| `vpin_proxy` / CLI compact `vpin=` | Same daily family as `vpin_mean` — still ≠ session |

research_only — **never live Sharpe**. Cross-ref: ofi/vpin soft-verify; H32 vs H43.

## `ohlc_identity_rate` vs `gap_finite_rate`

**Never equate.** Both are northset bar **rates** ∈[0,1] when finite, but different identities.

| Rate | Definition | Soft-verify / H |
|---|---|---|
| **`ohlc_identity_rate`** | Fraction of bars with classic OHLC envelope OK (`high/low` envelope `open/close`; positive open/close; finite) — `identities.ohlc_identity_rate` | H20 bound when finite (`ohlc_identity_rate ≥ 1` claimed); not the same helper as gap |
| **`gap_finite_rate`** | Among bars with non-null overnight `gap = open/prev_close − 1`, fraction with **finite** gap | `gap_finite_rate_honesty_errors` ∈[0,1]; **no** H20 claim |

A bar can fail OHLC identity and still have a finite gap (or the reverse). Empty bars → NaN rates (skip soft-verify). research_only — **never live Sharpe**. Cross-ref: **gap_finite_rate soft-verify**.






## `book_uncrossed_rate` / H21 soft-verify

**CoS live.** Fraction of book snapshots with best bid **strictly below** best ask (`identities.book_uncrossed_rate` / `book_identity_frame`).

| Layer | Detail |
|---|---|
| Definition | Finite bid/ask and `bid < ask` (and finite positive `spread` when that col present) |
| Receipt | `book_uncrossed_rate` on northset |
| Soft-verify | `book_uncrossed_rate_honesty_errors`: finite → **∈ [0, 1]** (reject ±inf); NaN skip; error `book_uncrossed_rate_out_of_unit_interval` |
| Hypothesis | **H21** bound (`H21_northset_book`): finite rate + `book_hypothesis_eligible` → require H21 discovery/bound row (`h21_hypothesis_consistency_errors`) |

### Never equate

| Object | Why separate |
|---|---|
| `book_uncrossed_rate` | Book TOB integrity rate — H21 |
| `ohlc_identity_rate` | Daily candle envelope — H20 |
| `join_coverage` | Fuse share — not uncrossed |
| Crossed book / locked quote diagnostics | Different checks |

research_only — **never live Sharpe**. Cross-ref: H20/H21/H22 soft-verify family.

## `session_ohlc_identity_rate` vs `ohlc_identity_rate`

**Never equate.** Same identity function (`ohlc_identity_rate`), **two frames**.

| Receipt key | Frame | Bench |
|---|---|---|
| **`ohlc_identity_rate`** | Daily bars | `ohlc_identity_rate(bars)` → H20 when finite |
| **`session_ohlc_identity_rate`** | Session candles | `ohlc_identity_rate(session)` when session height > 0; else NaN |

Both are OHLC envelope rates ∈[0,1] when finite — different clocks (daily vs intraday session candles). Related but still distinct: `session_reconstructs_daily_rate` (session envelope reconstructs daily OHLC). research_only — **never live Sharpe**. Cross-ref: **ohlc_identity_rate vs gap_finite_rate**; **book_uncrossed_rate / H21**.



## H20 / H21 / H22 soft-verify triad

Research-receipt consistency: **finite gate metric → require matching H-row**. Non-finite / missing metric → skip. Not a live capital / promotion gate. Invoked from `dipcatcher verify-research` via the three `h2*_hypothesis_consistency_errors` helpers.

| H-id | Family | Gate (finite → require row) | Eligibility | Consistency error token |
|---|---|---|---|---|
| **H20** `H20_northset_ohlc` | **bound** | `ohlc_identity_rate` | always (no book gate) | `hypothesis_h20_missing_despite_finite_ohlc_identity_rate` |
| **H21** `H21_northset_book` | **bound** | `book_uncrossed_rate` | `book_hypothesis_eligible` is not false | `hypothesis_h21_missing_despite_finite_book_uncrossed_rate` |
| **H22** `H22_northset_imbalance` | **discovery** | `imbalance_top_p_ic` | `book_hypothesis_eligible` is not false | `hypothesis_h22_missing_despite_finite_imbalance_p_ic` |

### Companion rate soft-verify (unit-interval)

| Rate | Soft-verify helper | Rule when finite |
|---|---|---|
| `ohlc_identity_rate` | (H20 gate; bound claim ≥1 in notebook) | see identities |
| `book_uncrossed_rate` | `book_uncrossed_rate_honesty_errors` | ∈[0,1] |
| `imbalance_top_p_ic` | (H22 gate; IC p-value) | not a unit-interval rate |

### Never equate across the triad

| Never mix | Why |
|---|---|
| H20 ↔ H21 ↔ H22 | Different gates (OHLC envelope vs uncrossed book vs imbalance IC) |
| Rate soft-verify ↔ H-row consistency | Unit-interval / finiteness ≠ hypothesis table presence |
| Daily `ohlc_identity_rate` ↔ `session_ohlc_identity_rate` | See prior never-equate |

research_only — **never live Sharpe**. Cross-ref: **book_uncrossed_rate / H21**; **ohlc_identity_rate vs gap_finite_rate**.

## `session_reconstructs_daily_rate` vs `session_ohlc_identity_rate`

**Never equate.** Both session-related rates; different predicates.

| Receipt key | Definition | Soft-verify |
|---|---|---|
| **`session_ohlc_identity_rate`** | `ohlc_identity_rate(session)` — fraction of **session candles** with classic OHLC envelope OK | (no dedicated unit helper required beyond frame identity; ≠ reconstructs) |
| **`session_reconstructs_daily_rate`** | Fraction of **daily** bars whose session envelope matches daily OHLC (`sess_open/first`, `sess_close/last`, `sess_high/max`, `sess_low/min` within atol) | `session_reconstructs_daily_rate_honesty_errors`: ∈[0,1] when finite; error `session_reconstructs_daily_rate_out_of_unit_interval` |

Session candles can all pass OHLC identity while the session envelope fails to reconstruct the daily bar (or the reverse on edge cases). Related third: daily `ohlc_identity_rate`. research_only — **never live Sharpe**. Cross-ref: **session_ohlc_identity_rate vs ohlc_identity_rate**; H20/H21/H22 triad.



## `session_volume_conservation_rate` soft-verify

**CoS live.** Sibling of `session_reconstructs_daily_rate` / session chain — **volume** conservation, not OHLC envelope.

| Layer | Detail |
|---|---|
| Definition | `identities.session_volume_conservation_rate`: fraction of days where `sum(session volume) ≈ daily volume` (relative tol) |
| Receipt | `session_volume_conservation_rate` |
| Soft-verify | `session_volume_conservation_rate_honesty_errors`: finite → **∈ [0, 1]**; NaN skip; error `session_volume_conservation_rate_out_of_unit_interval` |
| Hypothesis | **H24** bound (`H24_northset_volume`): finite rate → require H24 row via `northset_h23_h28_consistency_errors` |

### Never equate

| Object | Kind |
|---|---|
| `session_volume_conservation_rate` | session volumes **sum to** daily volume |
| `session_reconstructs_daily_rate` | session OHLC **envelope matches** daily OHLC — H23 |
| `session_ohlc_identity_rate` | per session-candle OHLC identity |
| `session_chain_rate` | session chain continuity — H29 |

research_only — **never live Sharpe**. Cross-ref: **session_reconstructs_daily_rate vs session_ohlc_identity_rate**; **H23+ expansion**.

## H23+ soft-verify expansion pointer

Beyond the H20/H21/H22 triad, catalog v2 expands soft H-row consistency through **`northset_h23_h28_consistency_errors`** over **`NORTHSET_H23_H28_SPECS`** (name is historical — tuple includes H23–H49-class gates). Same rule: **finite metric → require matching H-row** (skip if non-finite / missing; book-gated metrics skip when `book_hypothesis_eligible` is false).

| Band | Examples (gate → H-id) | Family |
|---|---|---|
| Session **bound** | `session_reconstructs_daily_rate` → **H23**; `session_volume_conservation_rate` → **H24**; `session_chain_rate` → **H29** | bound |
| Core **discovery** | `microprice_p_ic` → H25; `wick_skew_p_ic` → H26; `ofi_p_ic` → H27; `dm_gk_vs_park_p` → H28; `clv_p_ic` → H30; `vpin_p_ic` → H32 | discovery |
| Sweep **discovery / bound** | reject/follow signed IC (H33/H34 descriptive), event p (H35/H36), placebo (H37/H38), cost-adjusted means (H39/H40), fold stability (H41/H42), eligible-non-swept controls (H44/H45 primary), liq-quartile controls (H46/H47), OOT same-sign (H48), name cluster (H49) | mixed |
| Session book VPIN | `session_book_vpin_p_ic` → **H43** (separate helper path; eligibility `session_book_hypothesis_eligible`) | discovery |

Do **not** treat SPECS membership as live Sharpe. Full id list lives in `research.catalog` (`H23_HYPOTHESIS_ID` …). research_only — **never live Sharpe**. Cross-ref: **H20/H21/H22 soft-verify triad**; **session_volume_conservation_rate**.



## Sweep follow event vs cost-adjusted vs reject

**Never equate.** CoS soft-verify **live** on northset receipt event/cost stamps. Three axes: **signal** (follow vs reject), **statistic** (pre-cost event excess vs post-cost), **H-row** (H35/H36 vs H39/H40).

| Receipt key | Soft-verify | Rule when present/finite | H-id (finite gate) |
|---|---|---|---|
| `sweep_follow_event_mean_bps` | `sweep_follow_event_mean_bps_honesty_errors` | finite (signed OK; ±inf dishonest) | H36 uses `sweep_follow_event_p` |
| `sweep_follow_cost_adjusted_mean_bps` | `sweep_follow_cost_adjusted_mean_bps_honesty_errors` | finite (signed OK; ±inf dishonest) | **H40** bound |
| `sweep_reject_event_mean_bps` | `sweep_reject_event_mean_bps_honesty_errors` | finite (signed OK; ±inf fail-closed) | H35 uses `sweep_reject_event_p` |
| `sweep_reject_cost_adjusted_mean_bps` | `sweep_reject_cost_adjusted_mean_bps_honesty_errors` | finite (signed OK) | **H39** bound |

### Never-equate matrix

| Pair | Why |
|---|---|
| `sweep_follow_event_mean_bps` ↔ `sweep_follow_cost_adjusted_mean_bps` | **Pre-cost** event excess ≠ **post-cost** haircut mean (equality allowed if cost=0 but still different stats — do not soft-fail on coincidence) |
| `sweep_follow_*` ↔ `sweep_reject_*` | Different signals (follow-through vs reclaim/reject) |
| `*_event_mean_bps` ↔ `*_event_p` / `*_signed_mean_ic` | Event mean bps ≠ event p-value ≠ date IC |
| Event/cost stamps ↔ kyle nest keys | Sweep honesty helpers do **not** touch `kyle_*` |

CLI may echo `sweep_follow_event=…bps` beside `sweep_follow_costed=…bps` on one line — adjacency ≠ identity. research_only — **never live Sharpe**. Cross-ref: **H23+ soft-verify expansion pointer**; sweep evidence nested blob.


## Depth honesty ops checklist (`DEPTH_SHAPE_FIELDS` / `SIDE_STRUCTURE_FIELDS`)

Modules: `book_metrics` (producers + rates), `book_panel.validate_book_panel_depth_honesty`,
`bench_northset` optional floors. Honesty / plumbing — not alpha.

### Field sets

| Set | Fields | Finite when |
|---|---|---|
| `DEPTH_SHAPE_FIELDS` | `bid/ask_log_size_slope`, `bid/ask_log_price_slope`, `bid/ask_mean_log_tick_spacing` | Side `n_*_levels ≥ 2` for slopes; tick spacing also needs all adjacent gaps > 0 |
| `SIDE_STRUCTURE_FIELDS` | `bid/ask_size_concentration_top` (= top_size / side_depth) | Side `*_depth > 0` (thin-safe NaN if depth ≤ 0) |

### Producer / validator checklist

- [ ] Panel carries `n_bid_levels` / `n_ask_levels` when shape cols are present (else depth-honesty **no-op** — legacy top-of-book remaps OK).
- [ ] **Thin → NaN:** `n_*_levels < 2` ⇒ every present `DEPTH_SHAPE` field on that side is null/NaN (`validate_book_panel_depth_honesty` fail-closed if finite).
- [ ] **Deep → finite slopes:** `n_*_levels ≥ 2` ⇒ present `*_log_size_slope` and `*_log_price_slope` are finite (fail-closed if null/NaN).
- [ ] **Tick spacing may NaN on zero-gap:** deep side may keep `*_mean_log_tick_spacing` NaN if any adjacent `|Δp| ≤ 0`; thin-side NaN still required.
- [ ] **Concentration when depth > 0:** `SIDE_STRUCTURE` finite iff side depth > 0; distinct from shape (does **not** need n≥2).
- [ ] Vendor top-of-book remap typically stamps `n_*=1` and NaN slopes — honesty passes; do not claim multi-level geometry.
- [ ] SYNTHETIC / session L2 paths should run `ensure_book_panel_shape_columns` so concentration cols exist before `concentration_top_finite_rate`.

### Northset receipt rates + optional floors

| Receipt key | Definition | Config floor (optional) |
|---|---|---|
| `depth_shape_finite_rate` | Among deep-side eligible (row, field) cells with `n_*≥2`, fraction finite (`depth_shape_finite_rate`) | `northset.depth_shape_finite_floor` ∈ [0,1] or **None** (default: no gate) |
| `concentration_top_finite_rate` | Among sides with depth > 0, fraction of finite concentration cells | `northset.concentration_top_finite_floor` ∈ [0,1] or **None** |

When a floor is set: NaN rate or rate < floor → `ValueError` fail-closed in `bench_northset`. Missing shape/concentration columns → rate NaN (floor then fails if set). Receipt also echoes the floor values (or null).

### Ops notes

- Failures are data-contract bugs (thin carrying slopes, deep missing slopes, concentration missing on positive depth when floor set).
- Do not edit receipt rates to pass; fix panel producers or disable optional floors.
- Top-of-book-only / external `book_panel_path`: leave floors **None**; expect NaN depth_shape rate (no deep eligibles). **Never invent** shape cols on vendor panels — see **External book_panel_path vs ensure_book_panel_shape_columns**.



## Shape / structure floors matrix (`DEPTH_SHAPE` vs SIDE / QUEUE / NOTIONAL / TOB)

**Never mix families.** Depth-shape slopes need multi-level books (`n_*≥2`); concentration / queue / notional / tob-share honesty rates do **not**. Gate each rate only with its own floor. Concentration vs queue formulas stay closed (Commander #59) — this matrix is about **rates + floors**, not field algebra.

### Field sets → rates → floors

| Family | Fields (tuple) | Receipt rate | Config floor (default **None**) | Eligible cells | Needs `n_*≥2`? |
|---|---|---|---|---|---|
| **`DEPTH_SHAPE`** | `bid/ask_log_size_slope`, `bid/ask_log_price_slope`, `bid/ask_mean_log_tick_spacing` | `depth_shape_finite_rate` | `depth_shape_finite_floor` | (row, field) with matching side `n_*_levels ≥ 2` | **Yes** |
| **`SIDE_STRUCTURE`** | `bid/ask_size_concentration_top` | `concentration_top_finite_rate` | `concentration_top_finite_floor` | side `*_depth > 0` | No |
| **`QUEUE_STRUCTURE`** | `queue_priority_proxy`, `ask_queue_priority_proxy` | `queue_priority_finite_rate` | `queue_priority_finite_floor` | side `*_depth > 0` | No |
| **`SIDE_NOTIONAL`** | `side_notional_proxy_bid/ask` | `side_notional_finite_rate` | `side_notional_finite_floor` | `best_*>0` ∩ `*_depth>0` | No |
| **`TOB_SHARE`** | `tob_size_share` only | `tob_size_share_finite_rate` | `tob_size_share_finite_floor` | both depths finite and `bid_depth+ask_depth > 0` | No |

Helpers: `depth_shape_finite_rate`, `concentration_top_finite_rate`, `queue_priority_finite_rate`, `side_notional_finite_rate`, `tob_size_share_finite_rate` in `microstructure.book_metrics`. Floors gated in `bench_northset` (NaN rate or rate < floor → fail-closed). Doctor echoes all five floors under `northset.shape_floors` (config-only; does not compute live rates).

### Synth vs external

| Path | `ensure_book_panel_shape_columns` | Typical rates | Floors stance |
|---|---|---|---|
| **SYNTHETIC** daily / session L2 | **Yes** — DEPTH + SIDE + QUEUE + NOTIONAL + TOB cols expected | Finite rates over eligibles (depth-shape needs real multi-level synth) | Optional; set only when you intend fail-closed honesty |
| **External** `book_panel_path` / vendor TOB remap | **No** — never invent | `depth_shape_*` often **NaN** (`n_*=1` / missing cols); SIDE/QUEUE/NOTIONAL/TOB **NaN** if cols absent | **Leave all five floors unset** unless panel truly carries the family |

### Never-mix rules

1. Do **not** interpret a finite `concentration_top_finite_rate` as evidence of multi-level geometry (that is `DEPTH_SHAPE` only).
2. Do **not** gate QUEUE with SIDE floors or vice versa (same eligibility class, different fields/formulas — #59).
3. Do **not** treat `tob_size_share_finite_rate` as interchangeable with `mean_tob_size_share` (coverage vs level — see that section).
4. Do **not** call ensure on vendor frames to force finite depth-shape rates.
5. Floors unset → rates still stamped (may be NaN); **no** fail-close.

Cross-ref: Depth honesty ops checklist; External `book_panel_path` vs ensure; **queue_priority_finite_rate / side_notional_finite_rate**; **bid/ask_size_concentration_top vs queue_priority_proxy**; CLI shape echo line.



## Doctor `shape_floors` vs CLI shape-line echo (parity)

Same five floor **keys**; different **what** is printed.

| Surface | Rates | Floors |
|---|---|---|
| **`dipcatcher doctor`** `northset.shape_floors:` | **None** (config dump only) | Always echoes all five config values (`depth_shape`, `concentration_top`, `queue_priority`, `side_notional`, `tob_size_share` `*_finite_floor`), including `None` |
| **`dipcatcher northset` shape line** | Always echoes all five `*_finite_rate` (may be NaN) | Appends each `*_finite_floor=…` **only when** receipt floor is non-null |

Parity rules: doctor ≠ live rates; CLI rates ≠ proof floors were set. Unset floors → doctor shows `None`; CLI omits floor tokens. Matrix of families: **Shape / structure floors matrix**.


## `max_book_age` vs session L2 daily aggregation clocks

Two different join clocks — do not conflate:

| Path | Clock / join | Role of `northset.book_max_age_seconds` |
|---|---|---|
| Daily candle ↔ L2 fuse | `join_asof` `decision_time` ← `book_available_time` (backward), tolerance = `book_max_age_seconds`; age = `decision_time - book_event_time` | **Gates** coverage, stale, and lookahead fail-closed |
| Session L2 → daily path stats | `synthesize_session_l2` on session stamps (`event_time` = session, `parent_event_time` = daily); `aggregate_session_book_to_daily` then **exact** left-join onto fused daily on `(security_id, event_time=parent)` | **Does not** apply asof tolerance — parent key equality, not age window |

Session candle/book availability remains parent-close PIT semantics (rows visible at parent availability). Path OFI/imbalance/VPIN are computed within `(security_id, parent_event_time)` ordered by `session_index`, then published on the daily `event_time`. Widening `book_max_age_seconds` does **not** change session aggregation; tightening it only affects the daily asof fuse to the primary book panel / synth daily L2.



## External `book_panel_path` vs `ensure_book_panel_shape_columns` (floors)

Aligns CoS / ops honesty for top-of-book vendor panels.

### Who calls `ensure_book_panel_shape_columns`

| Path | Calls ensure? | Why |
|---|---|---|
| SYNTHETIC daily L2 (`synthesize_l2_from_bars` in `bench_northset`) | **Yes** | Multi-level synth must expose full `DEPTH_SHAPE` + `SIDE_STRUCTURE` (plumbing bug if missing) |
| Session L2 (`synthesize_session_l2`) | **Yes** | Same |
| External `northset.book_panel_path` | **No** | Vendor/external panel is authoritative; do **not** invent shape columns |

Wrappers: `ensure_depth_shape_columns` / `ensure_side_structure_columns` / combined `ensure_book_panel_shape_columns` in `synthetic_lob`.

### Vendor remap top-of-book contract (`vendor_book_map`)

Offline remap stamps:

- `n_bid_levels = n_ask_levels = 1`
- `bid/ask_log_size_slope = NaN` (no depth ladder)
- `bid_depth` / `ask_depth` copy top sizes; `imbalance_depth` copies top imbalance
- Does **not** synthesize multi-level price slopes, tick spacings, or fake ladders

`validate_book_panel_depth_honesty`: thin (`n_*<2`) → present shape fields must be NaN (passes). Missing optional shape cols → depth-honesty **no-op** for those fields.

### Implications for Northset floors

| Situation | `depth_shape_finite_rate` | `concentration_top_finite_rate` | If floor set |
|---|---|---|---|
| Top-of-book vendor panel (typical remap / external TOB) | **NaN** — no deep (`n≥2`) eligible cells, and/or not all `DEPTH_SHAPE_FIELDS` present on fuse | **NaN** if `SIDE_STRUCTURE` cols absent; else may be finite when depth > 0 | **Fail-closed** (NaN or below floor) |
| Multi-level SYNTHETIC (ensure ran) | Finite rate over deep eligibles | Finite rate over depth>0 sides | Gates apply when floors configured |
| Floors left **None** (default) | Rate still stamped (may be NaN) | Same | **No** fail-closed on these rates |

**Product / docs stance**

- External `book_panel_path` honesty: top-of-book → expect NaN depth-shape rate; **never invent** shape / concentration columns on vendor panels to force a finite rate.
- Keep `depth_shape_finite_floor` / `concentration_top_finite_floor` **unset** for TOB vendor runs unless the panel truly carries multi-level metrics.
- Do not call `ensure_book_panel_shape_columns` on remapped vendor frames to “fill in” NaNs with fabricated deep geometry.


## Sweep evidence battery → H33–H42 receipt keys

`bench_northset` runs `sweep_evidence_battery` → nests `receipt["sweep_evidence"]`, then **flattens** horizon-1 / placebo scalars onto the family receipt for agent mint + soft-verify.

| Hypothesis | Receipt field(s) used for mint / soft-verify | Source inside battery |
|---|---|---|
| H33 discovery | `sweep_reject_signed_p_ic` (+ `*_t_ic` for statistic) | Date-level IC on fused `sweep_reject_signed` vs `fwd_ret_1` (`_IC_FEATURES` path) |
| H34 discovery | `sweep_follow_signed_p_ic` (+ `*_t_ic`) | Same for `sweep_follow_signed` |
| H35 discovery | `sweep_reject_event_p` (+ `*_event_t`, `*_event_mean_bps`) | `event_studies` row: signal `sweep_reject_signed`, **horizon=1** |
| H36 discovery | `sweep_follow_event_p` (+ t / mean_bps) | `event_studies`: `sweep_follow_signed`, horizon=1 |
| H37 discovery | `sweep_reject_placebo_p` (+ `sweep_reject_placebo_observed_ic`) | `permutation_placebos["sweep_reject_signed"]` |
| H38 discovery | `sweep_follow_placebo_p` (+ observed IC) | `permutation_placebos["sweep_follow_signed"]` |
| H39 bound | `sweep_reject_cost_adjusted_mean_bps` (meets if > 0) | Flattened from horizon-1 event study `cost_adjusted_mean_bps` |
| H40 bound | `sweep_follow_cost_adjusted_mean_bps` | Same for follow |
| H41 bound | `sweep_reject_fold_positive_fraction` ≥ `sweep_min_fold_positive_fraction` | Horizon-1 `positive_fraction` |
| H42 bound | `sweep_follow_fold_positive_fraction` | Same for follow |

Also stamped (not separate H-ids): nested `sweep_evidence` blob (`event_studies`, `volatility_regimes`, `permutation_placebos`), `sweep_evidence_scope`, rates / conditional fwd means. Research-only; not live fills. Soft-verify finite→H-row table: earlier DATA_CONTRACTS section.





## `sweep_evidence.permutation_placebos` nested key schema

Dict under `receipt["sweep_evidence"]["permutation_placebos"]`, keyed by signal
(`sweep_reject_signed`, `sweep_follow_signed`). Each value is the return of
`within_date_permutation_test` (preserves within-date score and target marginals).

| Key | Meaning |
|---|---|
| `observed_mean_ic` | Mean across dates of within-date corr(score, target); NaN if &lt; 3 dates |
| `placebo_p_value` | Two-sided permutation p: `(1 + #{|placebo| ≥ |observed|}) / (n_perm + 1)`; NaN if inadequate |
| `n_dates` | Dates kept (≥ `min_names` finite pairs) |
| `n_permutations` | Config `sweep_n_permutations` |

Flatten to soft-verify / agent: H37/H38 use `sweep_*_placebo_p` + `sweep_*_placebo_observed_ic` from these nests. Nested-only extras (n_dates / n_permutations) are not separate H-ids.


## `sweep_evidence.event_studies` full row schema

List under `receipt["sweep_evidence"]["event_studies"]`. One row per `(signal, horizon)` with
`signal ∈ {sweep_reject_signed, sweep_follow_signed}` and `horizon ∈ northset.sweep_horizons`
(default often includes 1, 5, 20).

| Key | Meaning |
|---|---|
| `signal` | `sweep_reject_signed` or `sweep_follow_signed` |
| `horizon` | Exit horizon h (bars) |
| `entry` | Always `next_open` (event known at close t) |
| `exit` | `close_t_plus_{h}` |
| `control` | `same_date_cross_sectional_mean` |
| `n_events` / `n_dates` | Raw event count / finite daily-series dates |
| `sample_adequate` | `n_events ≥ sweep_min_events` and `n_dates ≥ sweep_min_dates` |
| `min_events` / `min_dates` | Config thresholds echoed |
| `mean_excess_bps` | Mean date-level XS excess × 1e4 |
| `hac_t` / `p_value` | HAC t / two-sided p (NaN if sample inadequate) |
| `bootstrap_lo_bps` / `bootstrap_hi_bps` | Block-bootstrap mean CI (bps) |
| `hit_rate` | Fraction of finite daily excess > 0 |
| `round_trip_cost_bps` | Modeled RT cost used for adjustment |
| `cost_adjusted_mean_bps` | Mean (excess − cost) × 1e4 |
| `cost_adjusted_hac_t` | HAC t on cost-adjusted series |
| `cost_adjusted_p_greater` | One-sided (greater) p for cost-adjusted mean |
| `n_folds` / `positive_fraction` / `worst_mean_bps` / `fold_means_bps` / `purge_bars` | Chronological fold summary (`_fold_summary`) |
| `reject_fdr` | BH-FDR flag among finite event-study p-values (battery-level) |

### Relation to H35–H42 flatten

Only **horizon = 1** rows are flattened onto the family receipt for soft-verify / agent mint:

- H35/H36 ← `p_value` / `hac_t` / `mean_excess_bps` as `sweep_*_event_*`
- H39/H40 ← `cost_adjusted_mean_bps`
- H41/H42 ← `positive_fraction` (vs `sweep_min_fold_positive_fraction`)

Horizons ≠ 1, bootstrap bands, hit_rate, fold_means_bps, FDR flags, and `sample_adequate` stay **nested-only** (no extra H-ids). Do not mint soft-verify rows from those nested-only keys until productized.


## External panel CLI / `vendor-book-map` dry-run ops

Offline CoS path for shipping `northset.book_panel_path` honesty (ADR-021). **No network.**

### Commands

| Command | Role |
|---|---|
| `dipcatcher vendor-book-map` | Dry-run alias coverage; optional remap parquet → Northset panel |
| `dipcatcher book-panel` | Write **SYNTHETIC** multi-level panel (runs `ensure_book_panel_shape_columns`) — not the vendor TOB path |
| `dipcatcher northset --book …` | Sets `cfg.northset.book_panel_path` to that parquet for the run |
| `dipcatcher northset --book … --vendor alpaca\|polygon\|generic` | Remap raw vendor quotes → `*_remapped_{vendor}.parquet`, then set `book_panel_path` |

### Recommended CoS sequence (external TOB)

```bash
# 1) Inspect preset aliases (no parquet)
uv run dipcatcher vendor-book-map --vendor alpaca

# 2) Dry-run column coverage (JSON report: mapped / missing / notes)
uv run dipcatcher vendor-book-map --vendor alpaca --columns S,t,bp,ap,bs,as

# 3) Remap vendor-shaped parquet → panel (n_*=1, NaN size slopes)
uv run dipcatcher vendor-book-map --vendor alpaca --parquet path/to/quotes.parquet   --out data/book_panel/alpaca_remapped.parquet

# 4) Run Northset on the panel — leave depth/concentration floors unset for TOB
uv run dipcatcher northset --config configs/research.yaml   --book data/book_panel/alpaca_remapped.parquet
# equivalent one-shot remap+run:
uv run dipcatcher northset --book path/to/quotes.parquet --vendor alpaca
```

Config alternative: set `northset.book_panel_path` in YAML to a pre-remapped panel path (same honesty).

### Optional CLI floors (use with care)

`dipcatcher northset` accepts `--depth-shape-floor` / `--concentration-floor` (overrides config for that run). On **top-of-book** remapped panels, rates are typically **NaN** → setting these floors **fail-closes**. Leave unset unless the panel truly carries multi-level shape / concentration columns. Never invent shape cols to pass floors.

### Honesty expectations on success

- Receipt `book_dgp` ≈ `vendor_panel:{source}`; fuse may stamp `external_panel`
- `depth_shape_finite_rate` often NaN (n_*=1 / incomplete DEPTH_SHAPE on fuse)
- Join coverage gated by `book_join_coverage_floor` (default 0.5)
- Do not confuse with `dipcatcher book-panel` SYNTHETIC output (`shape_columns_ensured=true`, finite rates expected)



## `dipcatcher northset` CLI echo: depth_shape / concentration (external runs)

After a successful `bench_northset`, the CLI prints a dedicated shape line (then dumps the receipt blob):

```text
depth_shape_finite_rate=<float|nan> concentration_top_finite_rate=<float|nan> queue_priority_finite_rate=<float|nan> side_notional_finite_rate=<float|nan>
```

If a floor was set (YAML or `--depth-shape-floor` / `--concentration-floor`), the same line appends:

```text
… depth_shape_finite_floor=<f> concentration_top_finite_floor=<f> queue_priority_finite_floor=<f> side_notional_finite_floor=<f>
```

(only the floors that are non-null appear).

### External / TOB expectations

| Observation | Meaning |
|---|---|
| `depth_shape_finite_rate=nan` | Typical top-of-book / `n_*=1` / incomplete `DEPTH_SHAPE` on fuse — **not** a silent pass |
| `concentration_top_finite_rate=nan` | Missing `SIDE_STRUCTURE` cols or no positive-depth eligibles |
| Floor keys **absent** from the echo | Floors unset (default) — NaN rates do **not** fail-close |
| Floor keys **present** + process `ValueError` before echo | Rate NaN or below floor → fail-closed inside `bench_northset` (no clean shape line / blob dump) |

Ops: on external TOB remaps, leave floors unset and accept NaN in the echo. Do not invent shape columns to force finite rates. Related: External panel CLI ops + depth honesty checklist.




## `queue_priority_finite_rate` / `side_notional_finite_rate` (CLI + floors)

**Ops refresh (2026-09-16):** Treat `queue_priority_finite_rate` as the QUEUE twin of `concentration_top_finite_rate` — same eligibility (side depth > 0), different field set / formula. Never gate QUEUE floors on SIDE_STRUCTURE rates or vice versa. External TOB: leave `queue_priority_finite_floor` unset (NaN rate expected). Disambiguation: **bid/ask_size_concentration_top vs queue_priority_proxy**.


Honesty rates parallel to concentration — **not** multi-level DEPTH_SHAPE (no `n_*≥2` requirement).

### Field sets

| Set | Fields | Finite when | Formula (producer) |
|---|---|---|---|
| `QUEUE_STRUCTURE_FIELDS` | `queue_priority_proxy`, `ask_queue_priority_proxy` | Side `*_depth > 0` | `top_size / (top_size + side_depth)` |
| `SIDE_NOTIONAL_FIELDS` | `side_notional_proxy_bid`, `side_notional_proxy_ask` | `best_* > 0` and `*_depth > 0` | `best_price * side_depth` |

Distinct from `SIDE_STRUCTURE` concentration (`top/side_depth`). Thin-safe NaN when depth/price ≤ 0.

### Ensure vs vendor TOB

| Path | `ensure_book_panel_shape_columns` | Queue/notional cols |
|---|---|---|
| SYNTHETIC daily/session L2 | **Yes** — now wraps depth + side structure + **queue** + **side notional** | Required; missing → plumbing `ValueError` |
| External `book_panel_path` / vendor remap | **No** — never invent | Remap does **not** stamp these proxies → rates typically **NaN** |

### Receipt rates + optional floors

| Rate | Eligibility | Config floor (default **None**) |
|---|---|---|
| `queue_priority_finite_rate` | Finite cells among sides with depth > 0 | `northset.queue_priority_finite_floor` |
| `side_notional_finite_rate` | Finite cells among best>0 ∩ depth>0 | `northset.side_notional_finite_floor` |

When floor set: NaN rate or rate < floor → `bench_northset` fail-closed. External TOB: leave floors **unset**.

### CLI / doctor surfaces

**`dipcatcher northset` shape line** (after join line) always includes:

```text
… queue_priority_finite_rate=<f|nan> side_notional_finite_rate=<f|nan>
```

Appends `queue_priority_finite_floor=…` / `side_notional_finite_floor=…` only when non-null.

**`dipcatcher doctor`** echoes config floors (does **not** compute live rates):

```text
northset.shape_floors: depth_shape_finite_floor=… concentration_top_finite_floor=…
  queue_priority_finite_floor=… side_notional_finite_floor=…
  (shape_columns_ensured is a receipt stamp when synth ensure runs)
```



## `tob_size_share` vs `tob_notional_share` (Commander #53 / #54; #62 TOB notional lock)

**Honesty confirm:** no active “may exceed 1” / may>1 claim remains for `tob_notional_share` — both shares **∈ (0, 1]** when finite (touch-priced TOB). Never-equate size vs notional kept.

Touch concentration diagnostics from `book_metrics_from_snapshot`. Do **not** treat them as interchangeable — **both** are unit-interval when finite after Commander **#62**.

| Metric | In `TOB_SHARE_FIELDS`? | Formula | Range when finite | Honesty |
|---|---|---|---|---|
| **`tob_size_share`** (#53 / legacy #34) | **Yes** (sole member of `TOB_SHARE_FIELDS`) | `(top_bid_size + top_ask_size) / (bid_depth + ask_depth)` | **∈ (0, 1]** | Size-space touch share |
| **`tob_notional_share`** (#54 / legacy #35; **#62**) | **No** (adjacent diagnostic) | `top_of_book_notional_proxy / (side_notional_proxy_bid + side_notional_proxy_ask)` with **touch-priced** TOB = `best_bid·top_bid + best_ask·top_ask` and side = `best·side_depth` | **∈ (0, 1]** | Notional-space touch share — **retired** “may exceed 1” (old mid·TOB numerator) |

NaN when inputs non-finite or denominators ≤ 0 (thin-safe). `tob_size_share` also requires each of top_bid, top_ask, bid_depth, ask_depth > 0.

**#62 lock:** `top_of_book_notional_proxy = best_bid·top_bid_size + best_ask·top_ask_size` (not `mid·(top_bid+top_ask)`). With touch-priced TOB and best·depth side notionals, `tob_notional_share ∈ (0, 1]` when finite (TOB ≤ side notionals when tops ≤ depths on each side).


### Never equate (Commander #53 / #54 / #62 honesty)

- [ ] `tob_size_share` is the honest ∈(0, 1] **size** touch share (`TOB_SHARE_FIELDS`)
- [ ] `tob_notional_share` is the honest ∈(0, 1] **notional** touch share (touch-priced TOB / side notionals) — **never equate** with size share (different spaces)
- [ ] Floors/rates still key off `tob_size_share` / `tob_size_share_finite_rate` (`TOB_SHARE_FIELDS`) — notional share is adjacent, not that tuple
- [ ] `mean_tob_size_share` (northset) ≠ `tob_size_share_finite_rate` (candle-book) — different benches (see dedicated disambiguation)
- [ ] Do **not** revive mid-based TOB notional or “may exceed 1” language — superseded by #62


### Panel / rates / receipt

- `BOOK_PANEL_OPTIONAL` includes `*TOB_SHARE_FIELDS` (`tob_size_share` only) — not `tob_notional_share` as a required optional tuple member for share-rate helpers.
- `tob_size_share_finite_rate`: among rows with finite depths and `bid_depth+ask_depth > 0`, fraction with finite `tob_size_share` (microstructure bench).
- Northset receipt stamps `mean_tob_size_share` (nanmean on fused) when the column is present.
- Vendor TOB remap may lack these columns → mean/rate NaN; do **not** invent shares to pass honesty.



## Notional proxies: `top_of_book_notional_proxy` / `side_notional_proxy_*` / `notional_imbalance`

Produced by `book_metrics_from_snapshot` (thin-safe NaN). Building blocks for `tob_notional_share`. **Commander #62** aligned TOB notional to touch prices.

| Field | Formula | Finite when | Thin-safe NaN when |
|---|---|---|---|
| `side_notional_proxy_bid` | `best_bid · bid_depth` | best>0 and depth>0, both finite | price/depth non-finite or ≤0 |
| `side_notional_proxy_ask` | `best_ask · ask_depth` | same on ask | same |
| `top_of_book_notional_proxy` | **`best_bid · top_bid_size + best_ask · top_ask_size`** (#62; was mid·sum tops) | best_bid>0, best_ask>0, both finite | either best ≤0 / non-finite |
| `notional_imbalance` | `(bid_notional − ask_notional) / (bid_notional + ask_notional)` | both notionals finite and sum > 0 | either non-finite or sum ≤0 |

Notes:

- Side depths are **sum of level sizes** on that side (not top alone).
- `tob_notional_share` = `top_of_book_notional_proxy / (side_notional_proxy_bid + side_notional_proxy_ask)` — **∈ (0, 1]** when finite with touch-priced TOB (#62). See **tob_size_share vs tob_notional_share**.
- Size-space `tob_size_share` and notional-space `tob_notional_share` are **both** unit-interval — still **never equate** (different spaces).
- `SIDE_NOTIONAL_FIELDS` = the two side proxies only; TOB notional + imbalance are adjacent exports, not that tuple.
- SYNTHETIC ensure requires `SIDE_NOTIONAL_FIELDS` (+ best/depth); vendor TOB may omit → related rates NaN.



## Spread / imbalance aliases (`half_spread`, `quoted_spread_bps`, …)

From `book_metrics_from_snapshot` (panel columns). Identities vs core `spread` / `imbalance_top`.

| Field | Formula / identity | Notes |
|---|---|---|
| `spread` | `best_ask − best_bid` | Core quoted spread (price units) |
| `half_spread` | `spread / 2` | `mid ± half_spread` recovers best ask / bid |
| `effective_spread` (**book_metrics**) | **Alias of `spread`** | Same value as quoted spread on the snapshot |
| `spread_bps` | `1e4 · spread / mid` | NaN path if mid ≤0 / non-finite (via snapshot) |
| `quoted_spread_bps` | **Alias of `spread_bps`** | Same |
| `half_spread_bps` | `1e4 · half_spread / mid` | NaN if mid ≤0 / non-finite |
| `spread_over_mid` | `spread / mid` | Equals `spread_bps / 1e4` when mid>0 |
| `imbalance_top` | `(top_bid − top_ask) / (top_bid + top_ask)` | Top-of-book size imbalance |
| `touch_size_imbalance` | **Alias of `imbalance_top`** (= `2w−1` when `w` finite) | Checklist: **touch_size_imbalance alias surface** |




## Reminder: `quoted_spread` alias vs dual-column `effective_spread` / `close_mid_abs_rel`

Post-fix contract (current) — **never equate** book quoted width with the candle close–mid diagnostic.

| Object | Layer | Identity | Receipt / CLI |
|---|---|---|---|
| `spread` | book_metrics | `best_ask − best_bid` | — |
| `quoted_spread` | book_metrics | **≡ `spread`** (northset-safe name) | `mean_quoted_spread` |
| `effective_spread` | book_metrics + northset fuse | **≡ `spread`** (ask−bid alias; **preserved**, not overwritten) | `mean_effective_spread` ≈ `mean_quoted_spread` (soft-verify) |
| `close_mid_abs_rel` | northset fuse only | **`2 · \|candle_close − mid\| / mid`** — candle diagnostic | `mean_close_mid_abs_rel` |

### Lock

- On the book snapshot: **`quoted_spread == effective_spread == spread`** (price units).
- **`mean_close_mid_abs_rel` is candle \|close−mid\| only** — relative units, ×2 convention — **never** a quoted/effective spread mean.
- Soft-verify: `northset_spread_receipt_honesty_errors` checks `mean_effective_spread ≈ mean_quoted_spread`; it does **not** equate either to `mean_close_mid_abs_rel`.

### Wait (session imbalance)

**Lieutenant** (not CoS) is shipping `mean_session_imbalance_mean` — **do not** expand session_imbalance receipt docs until that lands. Prior “CoS-owned session_imbalance_mean gap” notes are superseded by this ownership.

Cross-ref: **Northset fuse dual columns**; **mean_spread_bps / mean_effective_spread / mean_close_mid_abs_rel**; Spread receipt means never-equate; CLI echo quoted/half/close–mid.


## Northset fuse dual columns: `effective_spread` + `close_mid_abs_rel` (**FIXED**)

**Status (CoS landed 2026-09-16):** `bench_northset` **no longer overwrites** book `effective_spread`. Candle close–mid lives on its own column. Historical “collision / overwrite” language is obsolete — keep this section as the current contract.

| Column / receipt | Meaning | Formula / source |
|---|---|---|
| Fuse `effective_spread` | Book quoted-spread **alias** (ask−bid) | From `book_metrics` / attach; **preserved** through northset enrich |
| Fuse `close_mid_abs_rel` | Candle \|close−mid\| diagnostic | `2 · \|candle_close − mid\| / mid` (new col; does not touch `effective_spread`) |
| Receipt `mean_effective_spread` | Mean **book** effective/quoted alias | `nanmean(effective_spread)` ≡ mean ask−bid (price) |
| Receipt `mean_close_mid_abs_rel` | Mean candle close–mid | `nanmean(close_mid_abs_rel)` |
| Also | `mean_quoted_spread` / `mean_spread_bps` | nanmean of `spread` / `spread_bps` (quoted; unchanged) |

Kyle fuse does **not** mint `close_mid_abs_rel`. Soft-verify does not currently gate on these keys.

### Historical note (pre-fix)

Older builds aliased the candle formula onto fuse column `effective_spread`, destroying the book alias on that name. Do not use archived receipts’ `mean_effective_spread` as quoted spread without checking build date / presence of `mean_close_mid_abs_rel`.


## `mean_spread_bps` / `mean_effective_spread` / `mean_close_mid_abs_rel` (side-by-side)

Post-fix dual-column contract (current):

| Receipt key | Source column | Formula (row → nanmean) | Units | What it measures |
|---|---|---|---|---|
| `mean_quoted_spread` | `spread` | `best_ask − best_bid` | price | Mean quoted touch width |
| `mean_spread_bps` | `spread_bps` | `1e4 · spread / mid` when mid>0 | bps | Mean quoted spread in bps |
| `mean_effective_spread` | `effective_spread` (book alias, **not** overwritten) | same as `spread` (ask−bid) | price | Mean book effective/quoted alias |
| `mean_close_mid_abs_rel` | `close_mid_abs_rel` | `2 · \|candle_close − mid\| / mid` | relative | Mean close–mid deviation |

Related: `spread_over_mid = spread/mid` (= `spread_bps/1e4`) ≠ `close_mid_abs_rel` (uses \|close−mid\| and the ×2 convention).

**Honesty:** Prefer `mean_spread_bps` / `mean_quoted_spread` for quoted width in ops copy; `mean_effective_spread` now agrees with the book alias (ask−bid). Use `mean_close_mid_abs_rel` for the candle diagnostic formerly mis-labeled under `mean_effective_spread`.




## Spread receipt means: never equate blindly

| Receipt key | Units | Identity / relation | ≠ |
|---|---|---|---|
| `mean_quoted_spread` | price | nanmean(`spread`) = ask−bid | — |
| `mean_effective_spread` | price | Post-fix: **same object as quoted** (book alias); soft-verify ≈ `mean_quoted_spread` | ≠ `mean_close_mid_abs_rel` |
| `mean_half_spread` | price | nanmean(`half_spread`); soft-verify ≈ `0.5 * mean_quoted_spread` | ≠ half of CS/Roll |
| `mean_spread_bps` | bps | nanmean(`spread_bps`); soft-verify ≈ `2 * mean_half_spread_bps` | ≠ `1e4 * mean_quoted / mean_mid` (ratio-of-means) |
| `mean_half_spread_bps` | bps | nanmean(`half_spread_bps`) | pair with `mean_spread_bps` |
| `mean_close_mid_abs_rel` | relative | candle `2\|C−mid\| / mid` | **not** a quoted spread mean |

**Never equate blindly:** treat price-space and bps-space as separate; do not swap `mean_effective_spread` for close–mid; do not treat half means as “half of estimator spreads.”

**Soft-verify (catalog):** `northset_spread_bps_honesty_errors`, `northset_half_spread_honesty_errors`, `northset_spread_receipt_honesty_errors` — research diagnostic only.


### Identity notes (research_only — **never live Sharpe**)

Row-wise book_metrics imply receipt-mean identities when both sides are finite:

| Identity | Soft-verify helper | Error token |
|---|---|---|
| `mean_half_spread ≈ 0.5 × mean_quoted_spread` | `northset_half_spread_honesty_errors` | `mean_half_spread_not_half_of_mean_quoted_spread` |
| `mean_spread_bps ≈ 2 × mean_half_spread_bps` | `northset_spread_bps_honesty_errors` | `mean_spread_bps_not_double_mean_half_spread_bps` |
| `mean_effective_spread ≈ mean_quoted_spread` (post dual-col fix) | `northset_spread_receipt_honesty_errors` | (effective ≠ quoted / close-mid confusion) |

Prefer mean-of-ratios identities above — **not** `1e4 × mean_quoted_spread / mean_mid` (ratio-of-means ≠ mean-of-ratios).

**Ops path:** `dipcatcher verify-research` → `research.verify` runs the three helpers on family blobs `northset` **and** `candle_order_book` (soft errors; research diagnostic only). Northset receipt already stamps `research_only: true` / `claim: research_diagnostic_only`. These identities are **not** live risk/Sharpe gates.



## verify-research ops: candle dual-IC alias (docs-only)

Family `candle_order_book` may stamp both `ic_imbalance_top*` and `ic_touch_size_imbalance*` — **redundant IC** (alias), not independent alpha. **No** soft-verify helper asserts equality today; ops must not double-count. research_only — **never live Sharpe**. Full note: **Candle dual-IC alias honesty**.


## `verify-research` ops: `northset_spread_*_honesty_errors`

Soft receipt-mean identities (research diagnostic only — **never live Sharpe**). Implemented in `research.catalog`; invoked from `research.verify` when running `dipcatcher verify-research`.

| Helper | Families checked | When both sides finite, requires | Error token |
|---|---|---|---|
| `northset_half_spread_honesty_errors` | `northset`, `candle_order_book` | `mean_half_spread ≈ 0.5 × mean_quoted_spread` | `mean_half_spread_not_half_of_mean_quoted_spread` |
| `northset_spread_bps_honesty_errors` | same | `mean_spread_bps ≈ 2 × mean_half_spread_bps` | `mean_spread_bps_not_double_mean_half_spread_bps` |
| `northset_spread_receipt_honesty_errors` | same | `mean_effective_spread ≈ mean_quoted_spread` (post dual-col) | `mean_effective_spread_diverges_from_mean_quoted_spread` |

Missing / non-finite either side → **skip** (no error). Failures append to the soft-verify error list (do not re-run live northset floors). Cross-ref: **Spread receipt means: never equate blindly** identity notes; CLI echo matrix for which means are stamped.

Also on this path: `northset_microprice_weight_balance_honesty_errors` (`northset` only, ∈[0,1]) — full CLI/receipt detail: **mean_microprice_weight_balance**.



**CLI:** CoS wired / is wiring these onto `dipcatcher northset`, research compact, and candle-order-book echo lines — see **CLI echo: quoted / half / close–mid / microprice means**. Prefer that section for which keys appear on which line (`mean_quoted_spread` missing from research compact).


## `coverage_guarantee_scope` soft-verify ops (Jackknife+ / CV+)

Research-only honesty (Day Wave 41–42) — **never live Sharpe / never live capital**.

### What the flag means

`coverage_guarantee_scope` is a **receipt honesty stamp**, not a performance metric. When set to **`marginal_exchangeable`**, it asserts that Jackknife+ / CV+ **coverage** / **coverage_floor** claims are interpreted as **marginal coverage under exchangeability** (H10/H15), **not** training-conditional / distribution-free guarantees, and **not** a live-trading Sharpe or capital gate.

### Soft-verify behavior

| Item | Detail |
|---|---|
| Helper | `coverage_guarantee_scope_consistency_errors` (`research.catalog`) |
| Invoked from | `research.verify` / `dipcatcher verify-research` |
| Families checked | nonempty `jackknife_plus`, `cv_plus` only |
| Fires when | family blob exposes `coverage` **or** `coverage_floor` (key present even if value is NaN) |
| Required stamp | `coverage_guarantee_scope == "marginal_exchangeable"` |
| Errors | `coverage_guarantee_scope_missing:<fam>` / `coverage_guarantee_scope_invalid:<fam>` |
| **Skips** (no error) | family absent / empty `{}`; **or** neither `coverage` nor `coverage_floor` key present; other conformal families out of scope |

Soft-verify H-table sprawl remains paused. Pointers: MATH_SPEC Day Wave 42; RESEARCH_CENTRE Wave 41–42. Cross-ref: NORTHSET / OPS / ADR-021.


## CLI echo: quoted / half / close–mid / microprice means (current)

**Status:** `mean_spread_bps` is **no longer blob-only** on northset / research / candle_order_book — docs below match live CLI.

### `dipcatcher northset` (after shape-rate line)

```
mean_quoted_spread=… mean_effective_spread=… mean_half_spread=… mean_half_spread_bps=…
mean_spread_bps=… mean_close_mid_abs_rel=… mean_microprice_weight_balance=…
```

| Echo key | Receipt | Meaning |
|---|---|---|
| `mean_quoted_spread` | same | nanmean(`spread`) ask−bid |
| `mean_effective_spread` | same | nanmean(book `effective_spread` alias) — quoted |
| `mean_half_spread` | same | nanmean(`half_spread`) = mean spread/2 |
| `mean_half_spread_bps` | same | nanmean(`half_spread_bps`) |
| `mean_spread_bps` | same | nanmean(`spread_bps`) — **quoted bps** |
| `mean_close_mid_abs_rel` | same | nanmean(`close_mid_abs_rel`) candle diagnostic |
| `mean_microprice_weight_balance` | same | nanmean(`microprice_weight_balance`) |

### Research notebook northset compact line

Ends with (among other keys):

```
… mean_effective_spread=… mean_half_spread=… mean_half_spread_bps=…
mean_spread_bps=… mean_close_mid_abs_rel=… mean_microprice_weight_balance=…
```

`mean_quoted_spread` is **not** on this compact line (use `dipcatcher northset` or blob). `vpin=` remains `vpin_mean` only.

### `dipcatcher candle-order-book` (SYNTH path)

Prints `mean_quoted_spread` / `mean_effective_spread` / `mean_half_spread` / `mean_half_spread_bps` / `mean_spread_bps` (no `mean_close_mid_abs_rel` on that family line unless receipt carries it).

### Honesty

- Prefer `mean_spread_bps` + `mean_quoted_spread` for quoted width; `mean_close_mid_abs_rel` for candle–mid only.
- Half-spread means are aliases of half quoted width — not Roll/CS/AR estimators.
- Older docs that said `mean_spread_bps` was northset blob-only are **obsolete**.


## Touch means vs OHLC / mid estimator spreads

**Not interchangeable.** Touch means average observed book width; estimator panel keys infer a spread from mid or OHLC paths without needing L2.

| Receipt key | Family | Inputs | Aggregation | Units / shape |
|---|---|---|---|---|
| `mean_quoted_spread` | **Touch mean** | fused `spread` = ask−bid | nanmean over fused rows | price |
| `mean_spread_bps` | **Touch mean** | fused `spread_bps` | nanmean over fused rows | bps |
| `mean_effective_spread` | **Touch** (book alias, post-fix) | `effective_spread` = ask−bid | nanmean | price |
| `mean_close_mid_abs_rel` | Candle–mid diagnostic | `close_mid_abs_rel` = `2\|C−mid\|/mid` | nanmean | relative |
| `roll_spread` | **Estimator** (Roll 1984) | fused `mid` per `security_id` | per-name `2√(-γ₁)` then nanmean; `roll_n_securities` = finite names | price-like (mid units) |
| `corwin_schultz_spread` | **Estimator** (CS 2012) | bar `high`/`low` only | mean of two-day pair spreads over all finite pairs | relative ∈(0,1) |
| `abdi_ranaldo_spread` | **Estimator** (AR 2017) | bar `high`/`low`/`close` | mean of two-day relative spreads | relative ∈[0,1) |

### Ops / honesty rules

- Touch means require a book join (`spread` present). Estimators need only mid (Roll) or OHLC bars (CS/AR) — they can be finite when book is thin/missing.
- Do **not** claim “quoted ≈ Roll/CS/AR” without an explicit calibration study; synthetic LOB and vendor TOB will diverge systematically.
- CLI northset echoes `cs_spread=…` among estimator lines; touch means appear under spread/mean keys in the blob.
- Formulas: MATH_SPEC Roll / Corwin–Schultz / Abdi–Ranaldo sections. Touch aliases: DATA_CONTRACTS **Spread / imbalance aliases**.



## Session OFI vs bar-level `ofi` / `queue_imbalance` (validity + fail-closed)

Three different objects. Do not mix names or treat session aggregates as bar Cont OFI.

### Definitions

| Signal | Source module | Clock / grain | Formula (short) |
|---|---|---|---|
| `queue_imbalance` | `estimators.queue_imbalance` (also alias of `imbalance_top`) | **Bar / daily book row** (snapshot level) | `(top_bid_size − top_ask_size) / (top_bid_size + top_ask_size)` |
| `ofi` | `order_flow_imbalance` on daily book (or attach size-diff proxy) | **Bar / daily** Cont between consecutive daily tops | Cont–Kukanov–Stoikov TOB OFI (or `Δbid_size − Δask_size` proxy) |
| `ofi_lag` | `bench_northset` enrich | Daily | `ofi.shift(1).over(security_id)` |
| `session_step_ofi` | `synthetic_lob.aggregate_session_book_to_daily` | **Session snapshot** within parent day | Cont-style OFI between consecutive session tops (same parent) |
| `session_ofi_sum` | same aggregate → join to daily fuse | **Daily** (sum over session snaps) | `sum(session_step_ofi)` |
| `session_book_vpin` | same | Daily | `|session_ofi_sum| / session_ofi_abs_sum` |

### When each is valid

| Signal | Valid when | Invalid / absent when |
|---|---|---|
| `queue_imbalance` | Daily book has TOB sizes (or `imbalance_top`); `queue_imbalance()` ran on book | Missing sizes and no `imbalance_top`; empty book |
| `ofi` / `ofi_lag` | Daily book chronology with Cont inputs (or attach proxy from sizes) | No book join / empty fuse; first bar of name → OFI null then lag null |
| `session_ofi_*` | `northset.use_session_l2=true` **and** session candles non-empty **and** `synthesize_session_l2` + `aggregate_session_book_to_daily` succeed (SYNTHETIC session path today) | `use_session_l2=false`; session empty with bars present → **fail-closed** before receipt; external daily-only book without session path → columns absent, means NaN |

### Fail-closed (session path)

| Condition | Behavior |
|---|---|
| `use_session_l2` and bars>0 but `session_candles_from_daily` empty | `ValueError` fail-closed (“session-L2 enabled but session… empty”) |
| Session L2 on + identity rates < `session_l2_identity_floor` | `enforce_session_l2_identity_floors` → `ValueError` |
| Shape / join / coverage floors (unrelated but same bench) | Separate fail-closed in `bench_northset` — see floors tables |

No fail-closed solely because `session_ofi_sum` is NaN on a row after a successful aggregate (first session snap has null step OFI by construction; sum may still be finite).

### Receipt / fuse field names

| Field | Kind | Notes |
|---|---|---|
| `queue_imbalance` | fuse col | IC feature |
| `queue_imbalance_mean` | receipt | nanmean |
| `ofi`, `ofi_lag` | fuse cols | IC / Kyle path; **no** `ofi_mean` |
| `ofi_mean_ic` / `ofi_lag_*_ic` | receipt IC | Compact CLI may show `ofi_ic=` |
| `ofi_lag1_corr`, `ofi_lag1_n_securities` | receipt | Panel lag-1 corr of `ofi` |
| `kyle_ofi_lambda`, `kyle_ofi_r2`, `kyle_ofi_n_securities` | receipt | Always-on OLS on daily `ofi` |
| `session_ofi_sum` | fuse col (session join) | IC feature when present |
| `session_ofi_sum_mean` | receipt | NaN if column absent |
| `session_book_vpin`, `session_book_vpin_mean` | fuse / receipt | ≠ `vpin_mean` (bar/session bulk path) |
| `n_session_book_snaps`, `mean_session_book_snaps` | fuse / receipt | Coverage of session path |
| `use_session_l2`, `session_l2_identity_gate` | receipt stamps | Config / gate status |

**Honesty:** all are book-size / TOB proxies (`impact_proxy_warning`), not signed trade prints. Session aggregates from `synthetic_lob` are SYNTHETIC session DGP unless/until an empirical session book path is productized.



## `session_book_vpin_mean` vs `vpin_mean` / `vpin_proxy` (H43 vs H32)

**Do not equate.** Three VPIN-adjacent objects exist on Northset; H-rows attach to two of them.

### Definitions

| Object | Builder | Formula (short) | Grain |
|---|---|---|---|
| `vpin` (fuse) / `vpin_proxy` | `estimators.vpin_proxy` on **daily** book | Rolling mean of Cont-style \|buy−sell\|/(buy+sell) over TOB updates | Daily book chronology |
| `vpin_mean` | receipt | `nanmean(vpin)` | Scalar |
| `session_book_vpin` | `aggregate_session_book_to_daily` | `|session_ofi_sum| / session_ofi_abs_sum` | Daily row from **session L2** path |
| `session_book_vpin_mean` | receipt | `nanmean(session_book_vpin)` (NaN if col absent) | Scalar |
| `session_bulk_vpin` | `session_vpin(session)` candles | Mean over parent days of \|signed session volume\| / total volume (sign from close vs open) | Scalar — **not** H32/H43 |

### When each exists

| Object | Present when | Absent / NaN when |
|---|---|---|
| `vpin` / `vpin_mean` / `vpin_*_ic` | Daily book path runs `vpin_proxy` after queue_imbalance | No book / empty fuse |
| `session_book_vpin*` | `use_session_l2` + successful session synthesize/aggregate | Session L2 off or fail-closed earlier; means NaN if col missing |
| `session_bulk_vpin` | Session candles non-empty | No session candles |

### Eligibility → H32 vs H43

| Flag (from `bench_northset`) | True when | Gates |
|---|---|---|
| `book_hypothesis_eligible` | `bars_synthetic or not book_synthetic` | **H32** (`vpin_p_ic`) + other book discovery rows (H21/H22/H25/H27, …) |
| `session_book_hypothesis_eligible` | `bars_synthetic` only | **H43** (`session_book_vpin_p_ic`) |

| H-id | Metric | Family | Skip when |
|---|---|---|---|
| `H32_northset_vpin` | finite `vpin_p_ic` | discovery | `book_hypothesis_eligible` is false |
| `H43_northset_session_book_vpin` | finite `session_book_vpin_p_ic` | discovery | `session_book_hypothesis_eligible` is false |

Soft `verify-research`: same skips — finite metric does not demand the H-row when the flag is false. `session_bulk_vpin` has **no** H-id.

### CLI vs blob

| Surface | Echoes |
|---|---|
| Research northset compact line | `vpin=` → **`vpin_mean` only** |
| `dipcatcher northset` summary lines | Dual-col means / shape / join — **not** VPIN means |
| Full receipt blob | `vpin_mean`, `session_book_vpin_mean`, `session_bulk_vpin`, ICs, eligibility stamps |

Do not read compact `vpin=` as session-book VPIN.

### Never equate (ops checklist)

- [ ] Compact CLI `vpin=` → always `vpin_mean` (daily `vpin_proxy`), **not** `session_book_vpin_mean`
- [ ] Finite `vpin_p_ic` → H32 only if `book_hypothesis_eligible`; finite `session_book_vpin_p_ic` → H43 only if `session_book_hypothesis_eligible`
- [ ] `session_bulk_vpin` (candle signed-volume) is a **third** scalar — no H-id; not interchangeable with either mean above
- [ ] `session_close_*` / path means are **not** VPIN (see **session_close_*** section)



## `session_close_*` daily aggregates (last snap vs path means vs VPIN)

All produced by `synthetic_lob.aggregate_session_book_to_daily` and left-joined onto the daily northset fuse when `use_session_l2` succeeds. Join key: `event_time` = parent daily bar.

### Three families (do not mix)

| Family | What it is | Examples | ≠ VPIN? |
|---|---|---|---|
| **Last snap** (`session_close_*`) | Metrics from the **max `session_index`** snapshot that day | `session_close_mid`, `session_close_spread_bps`, `session_close_imbalance`, `session_close_micro_bps`, `session_close_bid_depth`, `session_close_ask_depth` | Yes — level/state at session close, not toxicity |
| **Path means / counts** | Reduce across **all** session snaps that day | `session_imbalance_mean`, `session_imbalance_std`, `session_spread_bps_mean`, `n_session_book_snaps` | Yes |
| **Path OFI / session-book VPIN** | Cont-style steps summed over the session | `session_ofi_sum`, `session_ofi_abs_sum`, `session_book_vpin` | `session_book_vpin` is the session-book toxicity proxy (H43) — still ≠ daily `vpin` / H32 |

Also stamped: `session_book_source` on the last-snap row.

### Field table

| Fuse column | Family | Source expression |
|---|---|---|
| `session_close_mid` | last snap | last `mid` |
| `session_close_spread_bps` | last snap | last `spread_bps` |
| `session_close_imbalance` | last snap | last `imbalance_top` |
| `session_close_micro_bps` | last snap | last `microprice_minus_mid_bps` |
| `session_close_bid_depth` / `session_close_ask_depth` | last snap | last depths |
| `session_imbalance_mean` / `session_imbalance_std` | path | mean / std of `imbalance_top` |
| `session_spread_bps_mean` | path | mean of `spread_bps` |
| `n_session_book_snaps` | path | count of session indices |
| `session_ofi_sum` / `session_ofi_abs_sum` | path OFI | sum / abs-sum of `session_step_ofi` |
| `session_book_vpin` | path VPIN | `|session_ofi_sum| / session_ofi_abs_sum` |
| `session_imbalance_std` | path | std of `imbalance_top` — **not** IC-scored; see **session_imbalance_std / session_book_source** |
| `session_book_source` | last snap stamp | last snap `source` (or `"synthetic"`) — ≠ receipt `book_source` |

### IC / receipt / CLI

| Item | Notes |
|---|---|
| IC features (subset) | `session_ofi_sum`, `session_imbalance_mean`, `session_book_vpin`, `session_close_imbalance`, `session_spread_bps_mean` are in `_IC_FEATURES` |
| Receipt means | `session_ofi_sum_mean`, `session_book_vpin_mean`, `mean_session_book_snaps` (and related); **no** dedicated `mean_session_close_*` scalars today |
| CLI | Session aggregate fields are **blob-only** on `dipcatcher northset` summary lines; research compact line does not echo `session_close_*` |

### Honesty

- `session_close_imbalance` ≠ `session_imbalance_mean` ≠ daily `queue_imbalance` / `imbalance_top`.
- `session_close_spread_bps` ≠ `session_spread_bps_mean` ≠ daily `mean_spread_bps`.
- None of the last-snap or path-mean fields are `vpin_mean` or `session_book_vpin`. Cross-ref: **session_book_vpin_mean vs vpin_mean** and **Session OFI vs bar-level**.



## `session_imbalance_std` / `session_book_source` stamps + session_* IC list

### Field honesty

| Fuse / stamp | When present | Definition | ≠ |
|---|---|---|---|
| `session_imbalance_std` | Session L2 on + aggregate join succeeds | `std(imbalance_top)` across session snaps that parent day (path) | ≠ `session_imbalance_mean`, ≠ `session_close_imbalance`, ≠ daily `imbalance_top` / `queue_imbalance` |
| `session_book_source` | Same (last-snap row) | `source` from last session book snap, else lit `"synthetic"` | ≠ receipt `book_source` (daily L2 panel provenance: `synthetic_lob` / vendor `source`) |
| Receipt `book_source` | Always on northset receipt | Daily book panel provenance used for DGP / CLI echo | Not overwritten by `session_book_source` |

**Session L2 off** (`use_session_l2=false` or path skipped): both fuse columns absent; no fail-closed solely for missing std/source. Identity / empty-session fail-closed still apply when session L2 is enabled (see Session OFI fail-closed table).

**CLI:** `session_imbalance_std` (fuse col) is **not** echoed as its own token. Receipt **`mean_session_imbalance_std`** is echoed on both northset (~661) and research-family blob (~957) — **never equate** fuse col ↔ receipt mean. `session_book_source` remains fuse/blob-only (no receipt mean).

### session_* columns: IC-scored vs fuse/receipt-only

From `bench_northset` `_IC_FEATURES` (date IC vs `fwd_ret_1` when column present on scored fuse):

| session_* column | In `_IC_FEATURES`? | Typical receipt scalar | Notes |
|---|---|---|---|
| `session_ofi_sum` | **Yes** | `session_ofi_sum_mean` | Path OFI sum |
| `session_imbalance_mean` | **Yes** | (no dedicated mean-of-mean today) | Path mean imbalance |
| `session_book_vpin` | **Yes** | `session_book_vpin_mean` | → H43 when eligible |
| `session_close_imbalance` | **Yes** | (no `mean_session_close_*`) | Last-snap imbalance |
| `session_spread_bps_mean` | **Yes** | (no dedicated receipt mean) | Path mean spread bps |
| `session_imbalance_std` | **No** | **`mean_session_imbalance_std`** | Path std fuse col; receipt = `nanmean` — **never equate**; CLI echoes the mean |
| `session_ofi_abs_sum` | **No** | — | Denominator for session_book_vpin |
| `session_close_mid` / `_spread_bps` / `_micro_bps` / `_bid_depth` / `_ask_depth` | **No** | — | Last-snap state |
| `n_session_book_snaps` | **No** | `mean_session_book_snaps` | Count; receipt mean only |
| `session_book_source` | **No** | — | Provenance string on fuse |

Related non-session but often confused: daily `imbalance_top`, `queue_imbalance`, `spread_bps`, `vpin` **are** IC-scored; receipt `book_source` is stamp-only.

Cross-ref: **session_close_*** (families), **session_book_vpin_mean vs vpin_mean**, **Session OFI vs bar-level**, CLI **join_coverage / book_source**.

## `session_imbalance_std` vs `mean_session_imbalance_std`

**Never equate.** Fuse path std ≠ receipt panel mean. Parallel to fuse `n_session_book_snaps` vs receipt `mean_session_book_snaps`.

| Object | Layer | What it is | CLI / IC |
|---|---|---|---|
| **`session_imbalance_std`** | Fuse / path agg col | `std(imbalance_top)` over snaps per `(security_id, parent_event_time)` | **not** echoed as its own CLI token; **not** in `_IC_FEATURES` |
| **`mean_session_imbalance_std`** | Receipt (`SESSION_RECEIPT_KEYS`) | `nanmean(session_imbalance_std)` over the fused panel; **NaN** if session L2 off / col absent (key still stamped) | **yes** — both northset (~661) and research-family blob echo (~957); receipt-only (no IC companion) |

### Never equate (clocks)

| Object | Kind |
|---|---|
| `session_imbalance_std` | per-day path **dispersion** (fuse col) |
| `mean_session_imbalance_std` | panel **mean of those dispersions** (receipt) |
| `session_imbalance_mean` / `mean_session_imbalance_mean` | path **location** |
| `session_close_imbalance` / `mean_session_close_imbalance` | last-snap **level** |
| daily `imbalance_top` | daily book |

### Honesty

- Soft-verify (`northset_session_imbalance_std_honesty_errors`): finite receipt **≥ 0**; NaN skips; error `mean_session_imbalance_std_negative_or_non_finite`.
- Polars `std` on a single snap is null/NaN — not a fake 0.
- Never treat the fuse col as the receipt mean, or mint an H-row off either.
- research_only — **never live Sharpe**. Cross-ref: **session_imbalance_std / session_book_source**; parallel book_snaps count vs mean.





## Docs gap: `session_imbalance_mean` receipt scalar

`session_imbalance_mean` is an IC-scored fuse column (session L2 path) but Northset does **not** currently stamp a dedicated receipt mean (unlike `session_ofi_sum_mean` / `session_book_vpin_mean`). **CoS owns** landing that receipt code — docs-only lane notes the gap only; do not invent a mean in docs as if shipped.


## Candle geometry IC family: `wick_skew` / `candle_body_ret` / `close_location_value`

Bar-only geometry from `northset.candles` (also on candle↔book fuse). All three are in northset `_IC_FEATURES` (date IC vs `fwd_ret_1`).

| Fuse column | Formula (short) | IC keys | H-row / soft-verify | Receipt mean? |
|---|---|---|---|---|
| `wick_skew` | `lower_wick_frac − upper_wick_frac` | `wick_skew_*_ic` | **H26_northset_wick** on finite `wick_skew_p_ic` (not book-eligibility gated) | No dedicated `mean_wick_skew` |
| `candle_body_ret` | `(close − open) / close` | `candle_body_ret_*_ic` | **No** soft-verify H-id today (IC-scored / discovery-adjacent only) | No plain mean |
| `close_location_value` (CLV) | `(2·close − high − low) / (high − low)` | `close_location_value_*_ic`; aliases `clv_p_ic` / `clv_t_ic` on receipt | **H30_northset_clv** on finite `clv_p_ic` (not book-eligibility gated) | No plain mean |

Related fuse cols (often scored elsewhere / candle_order_book): `candle_body_frac`, wick fracs, `candle_gap`, `candle_direction` — see NORTHSET feature set.
### Scored vs receipt-only (summary)

| Column | `_IC_FEATURES` | Dedicated receipt mean | Catalog H-row |
|---|---|---|---|
| `wick_skew` | yes | no | H26 (`wick_skew_p_ic`) |
| `candle_body_ret` | yes | no | **none** |
| `close_location_value` | yes | no (`clv_p_ic`/`clv_t_ic` aliases only) | H30 |

**Honesty:** geometry ICs are research discovery diagnostics — not live Sharpe; not touch-spread means.


## Candle geometry H-gap stub: `candle_body_frac` / wick fracs

Companion to **Candle geometry IC family** (`wick_skew` / `candle_body_ret` / CLV).

| Fuse column | Northset IC? | candle_order_book IC? | Receipt mean? | Soft-verify H-row | Gap |
|---|---|---|---|---|---|
| `wick_skew` | **scored** | **scored** | no | **H26** | — |
| `candle_body_ret` | **scored** | **scored** | no | **none** | IC without H-id |
| `close_location_value` | **scored** | via `clv_*` | no (aliases `clv_p_ic`/`clv_t_ic`) | **H30** | — |
| `candle_body_frac` | **not scored** | **scored** | no | **none** | fuse/bench only; missing H-id |
| `candle_upper_wick_frac` | **not scored** | not separate H | no | **none** | input to `wick_skew`; missing H-id |
| `candle_lower_wick_frac` | **not scored** | not separate H | no | **none** | input to `wick_skew`; missing H-id |

**H-gap:** inventing H-ids for `candle_body_frac` or individual wick fracs needs explicit productization (catalog + agent mint + soft-verify). Until then: treat as fuse/diagnostic; do not soft-verify finite→H for them. Prefer H26 on `wick_skew` when the question is wick asymmetry.


## Illiquidity means on Northset receipts (Amihud-style / range)

Bar-only diagnostics. **Research diagnostic only — no live Sharpe, no PnL/NAV claim.**

Northset receipt stamps `research_only: true` and `claim: research_diagnostic_only`. These means must not be promoted to live risk/Sharpe gates or soft-verify “edge” without an explicit product ADR.

| Fuse / concept | Formula | Receipt / IC keys | Honesty |
|---|---|---|---|
| `amihud` | `\|r\| / (close · volume)` (lagged close for `r`) | `amihud_mean`; ICs `amihud_*` and abs `amihud_abs_*` | Classic Amihud illiquidity ≠ quoted spread |
| `volume_over_range` | `volume / (high − low)` | **No** plain mean; abs IC `volume_over_range_abs_*` | Liquidity proxy (volume vs range), not spread×bps PnL |
| `true_range` | Wilder TR | `mean_true_range` | Range width; not a spread estimator (≠ Roll/CS/AR/touch) |
| Related touch | `mean_spread_bps` / `mean_quoted_spread` / `mean_effective_spread` | Dual-col docs | Quoted width — still not live Sharpe |

Forbidden: labeling `amihud_mean` or `mean_true_range` as strategy Sharpe, capacity, or production edge. Discovery may use ICs; soft-verify H-rows only where catalog already maps finite fields (do not invent Amihud H-ids here).


## `depth_imbalance_abs` / `spread_over_mid`

| Field | Formula | Identity / notes |
|---|---|---|
| `depth_imbalance_abs` | `\|imbalance_depth\|` | Absolute full-depth size imbalance; NaN if `imbalance_depth` non-finite |
| `spread_over_mid` | `spread / mid` | Equals `spread_bps / 1e4` when mid>0; NaN if mid ≤0 / non-finite |

Neither is overwritten by the northset `effective_spread` fuse step. Prefer these (plus `spread` / `imbalance_depth`) when you need unambiguous quoted-spread or depth-imbalance geometry on the fused frame after northset enrich.




## microprice vs mid vs `microprice_weight_balance` (triad refresh)

Three related but **non-interchangeable** objects from `book_metrics` / `OrderBookSnapshot`. Floors matrix stays closed.

### Definitions (as implemented)

| Object | Formula | Units / range when finite |
|---|---|---|
| **`mid`** | `0.5 * (best_bid + best_ask)` | price; midpoint of the touch |
| **`microprice_weight_balance`** (`w`) | `top_bid_size / (top_bid_size + top_ask_size)` | dimensionless **∈ [0, 1]**; NaN if tops non-finite or denom ≤ 0 |
| **`microprice`** (`μ`) | `(ask·top_bid + bid·top_ask) / (top_bid+top_ask)` ≡ `ask·w + bid·(1−w)` when denom > 0; **else falls back to `mid`** | price; **∈ [best_bid, best_ask]** when tops positive (convex combo) |

Derived diagnostics (not substitutes for `w`):

| Field | Formula | Notes |
|---|---|---|
| `microprice_minus_mid` | `μ − mid` | price gap |
| `microprice_minus_mid_bps` | `1e4 * (μ − mid) / mid` (0 if mid ≤ 0) | IC feature (`_IC_FEATURES` → H25 aliases `microprice_p_ic` / `microprice_t_ic`) |

### Identities (honesty)

1. **μ ∈ [bid, ask]** when top sizes > 0 and book uncrossed (convex combination of touch prices).
2. **mid** is always the arithmetic midpoint — independent of sizes.
3. **`w` alone is not a price.** Algebraic link when denom > 0:
   `μ − mid = spread · (w − 0.5)` with `spread = ask − bid`.
   **Never equate** `microprice_minus_mid` (or `_bps`) **with** `microprice_weight_balance` — different units; `w=0.5` ⇒ μ=mid, but `w` is not “how many ticks off mid.”
4. **`w` vs `imbalance_top`:** when both finite, `imbalance_top = 2w − 1` (locked — see **imbalance_top vs microprice_weight_balance**); still ≠ μ−mid.
5. **`w` ≠ queue_priority / size_concentration** (same-side depth denoms) — see those sections.

### CLI / receipt means

| Surface | What appears |
|---|---|
| Fuse cols | `mid`, `microprice`, `microprice_minus_mid(_bps)`, `microprice_weight_balance` |
| Northset receipt | `mean_microprice_weight_balance` = `nanmean(microprice_weight_balance)` when col present |
| `dipcatcher northset` means line | echoes **`mean_microprice_weight_balance`** (with quoted/half/bps/close–mid) — **not** a separate `mean_microprice` / `mean_mid` echo today |
| Research compact | same MWB mean on northset line |
| Soft-verify | `northset_microprice_weight_balance_honesty_errors`: finite mean ∈[0,1]; skip if NaN; research_only |

Cross-ref: **microprice_weight_balance export**; **mean_microprice_weight_balance**; CLI echo quoted/half/close–mid/microprice means; queue vs MWB; H25 microprice IC.




## `touch_size_imbalance` alias surface checklist

**Alias of `imbalance_top`** — same float from `book_metrics_from_snapshot` (`touch_size_imbalance: float(imb_top)`). **Never a separate signal.** tob_notional may>1 fix stays closed.

### Identity chain (when finite)

| Link | Statement | Where |
|---|---|---|
| Alias | `touch_size_imbalance ≡ imbalance_top` | producer + `METRICS_REQUIRED_FINITE_KEY_DOCS` |
| Weight | `imbalance_top = 2·microprice_weight_balance − 1` | when both finite — **imbalance_top vs microprice_weight_balance** |
| Therefore | `touch_size_imbalance = 2w − 1` when `w` finite | same lock |
| Not μ−mid | ≠ `microprice_minus_mid(_bps)` | triad: `μ−mid = spread·(w−0.5)` |

### Surfaces (as implemented)

| Surface | `imbalance_top` | `touch_size_imbalance` | Honesty |
|---|---|---|---|
| **Producer** `book_metrics_from_snapshot` | set to `imb_top` | set to **same** `imb_top` | Bitwise alias |
| **`METRICS_REQUIRED_FINITE_KEYS`** | member | member (alias still required present + finite) | Both gated by `assert_metrics_required_finite` |
| **`METRICS_REQUIRED_FINITE_KEY_DOCS`** | top imbalance formula | `"Alias of imbalance_top"` | Docs string lock |
| **Northset** `_IC_FEATURES` / H22 | **yes** (`imbalance_top`) | **absent** | Canonical IC path — do not add touch alias as a second H-row |
| **Candle+LOB** `bench._FEATURE_COLS` | **yes** | **no** (removed from IC list) | Fuse still stamps alias col; candle bench scores **`imbalance_top` only** — no `ic_touch_size_imbalance*` |
| **CLI** `dipcatcher northset` / doctor / means lines | not a dedicated mean echo | **no** `mean_touch_size_imbalance` | Do not invent a distinct mean |
| **Soft-verify** | (imbalance honesty via other helpers if any) | none for alias | MWB ∈[0,1] gates `w`, not this alias |
| **Docs** | Spread/imbalance aliases; 2w−1 lock | this checklist + NORTHSET/OPS/ADR | Never Cont queue priority / size concentration |

### Checklist

- [ ] Fuse/panel rows: where both finite, values **equal**
- [ ] REQUIRED catalog keeps both keys — removing the alias from REQUIRED is a product change, not silent drift
- [ ] Northset hypotheses stay on **`imbalance_top`** only
- [x] Candle `_FEATURE_COLS` scores `imbalance_top` only (alias col may still exist on fuse — not a second IC)
- [ ] No CLI mean / soft-verify invented solely for the alias name
- [ ] `= 2w − 1` when `w` finite; still ≠ μ−mid

Cross-ref: **Spread / imbalance aliases**; **imbalance_top vs microprice_weight_balance**; **microprice vs mid vs microprice_weight_balance**.



## Candle dual-IC alias honesty (closed — `_FEATURE_COLS` no longer dual-lists)

**Code lock (current):** `microstructure.bench._FEATURE_COLS` scores **`imbalance_top` only** — `touch_size_imbalance` was removed from the IC feature list (alias still stamped on the fuse by `book_metrics`). Northset `_IC_FEATURES` was already imbalance_top-only.

| Item | Status |
|---|---|
| Alias | `touch_size_imbalance ≡ imbalance_top` (= `2w−1` when `w` finite) — fuse/REQUIRED keys unchanged |
| Candle IC keys | `ic_imbalance_top*` only — **no** `ic_touch_size_imbalance*` from current bench |
| Soft-verify equality helper | Still **none** (and not needed while dual IC absent) |
| Ops | Do not cite stale “redundant dual IC” as live behavior; do not invent a second alpha |

Historical note: older docs described dual listing in `_FEATURE_COLS`; treat that as superseded. research_only — **never live Sharpe**.

Cross-ref: **touch_size_imbalance alias surface checklist**; **Candle+LOB `_FEATURE_COLS` inventory**.


## Candle+LOB `_FEATURE_COLS` inventory (HF honesty)

`bench_candle_order_book` date-IC vs `fwd_ret_1` (`ic_method=date_level_spearman_hac`). Optional family — not in `REQUIRED_BENCHMARK_FAMILIES`. research_only — **never live Sharpe**.

| Column | IC keys | Notes |
|---|---|---|
| `candle_body_ret`, `candle_body_frac`, `candle_range_frac`, `candle_direction`, `wick_skew` | `ic_*` | Geometry; wick_skew adjacent to H26 on northset path |
| `imbalance_top`, `imbalance_depth` | `ic_*` | Top ≠ depth; touch_size alias **not** IC-scored here |
| `spread_bps`, `microprice_minus_mid`, `microprice_minus_mid_bps` | `ic_*` | μ−mid ≠ bps companion |
| `candle_dir_x_imbalance`, `spread_x_range`, `imbalance_x_body_frac`, `signed_vol_x_imbalance` | `ic_*` | Interaction features |
| `ofi`, `queue_imbalance` | `ic_*` | Flow vs level — never equate |
| `bid_log_size_slope`, `ask_log_size_slope` | `ic_*` | DEPTH_SHAPE size slopes |
| `bid_log_price_slope`, `ask_log_price_slope` | `ic_*` | Commander residual #2 — ≠ size slopes |
| `bid_mean_log_tick_spacing`, `ask_mean_log_tick_spacing` | `ic_*` | Commander residual #3 — may NaN on ≤0 gaps |

Receipt companions (not IC): spread means, `mean_microprice_weight_balance`, `mean_microprice_minus_mid`, `mean_close_mid_abs_rel`, join/age, `depth_shape_finite_rate`, `mean_abs_ic` / `best_feature_ic*`.

### Soft-verify on `candle_order_book` (`verify-research`)

| Helper | Wired |
|---|---|
| `mean_microprice_weight_balance_honesty_errors` | yes (parity with northset) |
| `mean_microprice_minus_mid_honesty_errors` | yes |
| spread half/quoted identities | yes (shared helpers) |
| `join_coverage_honesty_errors` / `book_age_seconds_honesty_errors` / `depth_shape_finite_rate_honesty_errors` | yes |
| `structure_finite_rate_honesty_errors` | yes — skips if `finite_rate_*` keys absent (candle bench may not stamp them) |

### CLI `dipcatcher candle-book`

Echoes `family`, n_bars/n_scored/n_fused/min_names/depth/book_source/dgp/join/age, `mean_microprice_minus_mid`, `depth_shape_finite_rate`, `ic_method`, `mean_abs_ic`/`best`, spread-alias + MWB means line, then finite `ic_*` list. Does **not** echo northset-only session/kyle keys.

Never equate candle `ic_*` with northset discovery H-row ICs (`*_p_ic`) — different family surfaces.


## `imbalance_top` vs `microprice_weight_balance` (`2w − 1` lock)

Verified against `book_metrics_from_snapshot` / `_microprice_weight_balance` — **not invented**. Triad refresh stays closed; this is the imbalance ↔ weight link.

### As implemented

| Field | Code path | Formula |
|---|---|---|
| `imbalance_top` | inline in `book_metrics_from_snapshot` | `(top_bid_size − top_ask_size) / (top_bid_size + top_ask_size)` |
| `touch_size_imbalance` | alias | **same value** as `imbalance_top` |
| `microprice_weight_balance` (`w`) | `_microprice_weight_balance` | `top_bid_size / (top_bid_size + top_ask_size)`; **NaN** if tops non-finite or denom ≤ 0 |

### Locked identity

When **both** are finite (valid uncrossed snapshot with `top_bid_size + top_ask_size > 0`):

```text
imbalance_top = 2 · microprice_weight_balance − 1
```

Equivalently `w = (imbalance_top + 1) / 2`. Algebra: `(tb−ta)/(tb+ta) = 2·tb/(tb+ta) − 1`.

### Never-equate / caveats

- **Still never equate** `imbalance_top` with `microprice` or `μ − mid` — see triad (`μ − mid = spread · (w − 0.5)`).
- **`w` ∈ [0,1]** when finite; **`imbalance_top` ∈ [−1,1]** when finite — same information, different centering.
- Thin denom: MWB helper returns **NaN**; imbalance is computed without that helper and is in `METRICS_REQUIRED_FINITE_KEYS` (valid snaps are expected to keep tops usable). Do not invent a separate “imbalance NaN when denom≤0” contract beyond what the producer does.
- Soft-verify today gates **`mean_microprice_weight_balance` ∈ [0,1]** only — not a parallel mean-imbalance unit-interval check unless/until productized.

Cross-ref: **microprice vs mid vs microprice_weight_balance**; **microprice_weight_balance export**.


## `microprice_weight_balance` export

Part of the **microprice vs mid vs microprice_weight_balance** triad — see that section for μ/mid/`w` identities.


| Field | Formula | Thin-safe |
|---|---|---|
| `microprice_weight_balance` | `top_bid_size / (top_bid_size + top_ask_size)` | NaN if sizes non-finite or denom ≤ 0 |

Identity: with weight `w = microprice_weight_balance`,

`microprice = best_ask · w + best_bid · (1 − w)`

(same convex combination as Cont-style size-weighted mid when tops are positive). Distinct from `queue_priority_proxy` / `ask_queue_priority_proxy` (`top/(top+side_depth)` — never equate; see **queue_priority vs microprice_weight_balance**) and from `imbalance_top`.


## `mean_microprice_weight_balance` (CLI / receipt / soft-verify)

| Layer | Key | Notes |
|---|---|---|
| Fuse | `microprice_weight_balance` | `top_bid_size / (top_bid_size + top_ask_size)` ∈[0,1] when finite; see export section |
| Receipt | `mean_microprice_weight_balance` | `nanmean` of fuse col when present; else NaN |
| CLI | echoed | On `dipcatcher northset` means line and research compact northset line (see CLI echo matrix) |

### Soft-verify ∈[0,1]

| Helper | Family | Rule | Error token |
|---|---|---|---|
| `mean_microprice_weight_balance_honesty_errors` | **`northset` + `candle_order_book`** (`verify-research` both) | If finite, require `mean_microprice_weight_balance ∈ [0, 1]` | `mean_microprice_weight_balance_out_of_unit_interval` |

Missing / NaN → skip. Invoked from `research.verify` during `dipcatcher verify-research`. research_only — **never live Sharpe**. Cross-ref: verify-research spread honesty table (sibling helpers).





## `bid/ask_size_concentration_top` vs `queue_priority_proxy` / `ask_queue_priority_proxy`

**Never equate.** Both are same-side top-of-book diagnostics from `book_metrics`; they use **different denominators**. As implemented, queue priority uses **same-side** depth — **not** the opposite top (that is `microprice_weight_balance`).

**Commander #59:** formulas locked in `microstructure.book_metrics` (`_size_concentration_top` = `top/side_depth`; `_queue_priority_proxy` = `top/(top+side_depth)`). Docs must match that lock — do not invent opposite-side queue priority.

| Field | Tuple | Formula (as implemented) | Bounds when finite | Thin → NaN when |
|---|---|---|---|---|
| `bid_size_concentration_top` | `SIDE_STRUCTURE_FIELDS` | `top_bid_size / bid_depth` | typically **∈ (0, 1]** when top ≤ depth | `bid_depth ≤ 0` or non-finite |
| `ask_size_concentration_top` | `SIDE_STRUCTURE_FIELDS` | `top_ask_size / ask_depth` | same | `ask_depth ≤ 0` or non-finite |
| `queue_priority_proxy` | `QUEUE_STRUCTURE_FIELDS` | `top_bid_size / (top_bid_size + bid_depth)` | **∈ [0, 1)** (∈[0,1] if edge cases allow) | `bid_depth ≤ 0` or non-finite |
| `ask_queue_priority_proxy` | `QUEUE_STRUCTURE_FIELDS` | `top_ask_size / (top_ask_size + ask_depth)` | same | `ask_depth ≤ 0` or non-finite |

### Why they diverge

- Concentration: share of **side depth sitting at the touch** (`top/side_depth`).
- Queue priority proxy: Cont-style **priority weight** with denom = top + full side depth (`top/(top+side_depth)`). When side depth includes the touch, this is strictly **≤** concentration for the same snap (same top, larger denom).
- Neither uses the opposite side. Cross-side top mix = **`microprice_weight_balance`** — see **queue_priority vs microprice_weight_balance**.

### Honesty rates / floors (parallel, not interchangeable)

| Rate | Fields | Eligible cells | Optional floor (default unset) |
|---|---|---|---|
| `concentration_top_finite_rate` | SIDE_STRUCTURE | side `*_depth > 0` | `northset.concentration_top_finite_floor` |
| `queue_priority_finite_rate` | QUEUE_STRUCTURE | side `*_depth > 0` | `northset.queue_priority_finite_floor` |

Missing columns (external TOB remap) → rate **NaN**; floors unset → no fail-close. Floors set + NaN/below → `bench_northset` fail-closed. Cross-ref: **`queue_priority_finite_rate` / `side_notional_finite_rate` (CLI + floors)**; OPS shape floors.


## `queue_priority_proxy` / `ask_queue_priority_proxy` vs `microprice_weight_balance`

**Never equate.** Queue-priority proxies measure top-of-queue **position on one side**; microprice weight is the **cross-side** top-size mix used in microprice.

| Field | Tuple / role | Formula | Bounds when finite | Thin → NaN when |
|---|---|---|---|---|
| `queue_priority_proxy` | `QUEUE_STRUCTURE_FIELDS` (bid) | `top_bid_size / (top_bid_size + bid_depth)` | **∈ (0, 1]** typically (∈[0,1] if top=0 allowed by inputs) | `bid_depth ≤ 0` or non-finite |
| `ask_queue_priority_proxy` | `QUEUE_STRUCTURE_FIELDS` (ask) | `top_ask_size / (top_ask_size + ask_depth)` | same | `ask_depth ≤ 0` or non-finite |
| `microprice_weight_balance` | book metric / microprice weight | `top_bid_size / (top_bid_size + top_ask_size)` | **∈ [0, 1]** | denom ≤ 0 or non-finite tops |

### Honesty

- Queue priority ≠ imbalance_top / queue_imbalance (those are bid−ask over bid+ask).
- Queue priority ≠ `bid/ask_size_concentration_top` (`top/side_depth` vs `top/(top+side_depth)` — see **bid/ask_size_concentration_top vs queue_priority_proxy**).
- Queue priority ≠ microprice weight (same-side depth in denom vs opposite top size).
- Finite-rate floors on northset use `queue_priority_finite_rate` for QUEUE fields — not MWB.
- Receipt mean soft-verify ∈[0,1] exists for **`mean_microprice_weight_balance`** only today (`northset_microprice_weight_balance_honesty_errors`); queue-priority means are not similarly gated unless/until productized.

Cross-ref: DEPTH/QUEUE floors checklist; **mean_microprice_weight_balance**.


## `mean_tob_size_share` vs `tob_size_share_finite_rate`

Same underlying column `tob_size_share` — **different aggregations on different benches**. Do not equate.

| Key | Bench / module | Definition | Interprets |
|---|---|---|---|
| **`mean_tob_size_share`** | `northset.benches.bench_northset` (family `northset` receipt) | `nanmean` of fused `tob_size_share` values (absent column → key omitted / path skips) | Typical **level** of touch size share |
| **`tob_size_share_finite_rate`** | `microstructure.bench.bench_candle_order_book` (candle+LOB family receipt) | `tob_size_share_finite_rate`: among rows with finite depths and `bid_depth+ask_depth > 0`, fraction with **finite** `tob_size_share` | **Coverage / honesty** of the share column — not the mean share |

Northset does **not** currently stamp `tob_size_share_finite_rate` on the family receipt; candle-order-book bench does **not** stamp `mean_tob_size_share`. Comparing a northset mean to a candle-book finite rate is a category error.


## `dipcatcher northset` CLI echo: join_coverage / book_source (and where `book_dgp` lives)

Summary line printed after the banner (before the shape-rate line and the full receipt dump):

```text
join_coverage=<f> mean_book_age_s=<f> max_book_age_s=<f> book_source=<str> include_kyle_ofi=<bool>
```

**`book_dgp` is not on this summary line.** It appears on the receipt (`typer.echo(blob)` / notebook family) as `book_dgp`, alongside family `dgp` / `label`. Read the blob (or YAML config path) for DGP strings.

### Typical values: SYNTHETIC vs external

| Field (summary echo) | SYNTHETIC daily L2 (no `book_panel_path`) | External `book_panel_path` / `--book` (TOB remap) |
|---|---|---|
| `join_coverage` | ≈ **1.0** (attach default floor 1.0 when synth) | ≥ `book_join_coverage_floor` (default **0.5**) or fail-closed before echo |
| `mean_book_age_s` / `max_book_age_s` | Small / bar-aligned ages typical | May be larger; stale/lookahead fail-closed inside fuse |
| `book_source` | `synthetic_lob` | Panel `source` (e.g. `alpaca`, `polygon`, remapped preset name) or `parquet` if empty |
| `include_kyle_ofi` | Config flag (default false) | Same |

| Receipt field (blob, not summary line) | SYNTHETIC | External |
|---|---|---|
| `book_dgp` | `synthetic_lob` | `vendor_panel:{book_source}` |
| `dgp` (family) | `synthetic_lob` or follows label branch | Usually follows `book_dgp` when vendor book |

Optional one-shot remap prefix: `remapped vendor=<preset> → <path>` before the northset banner.

Fail-closed join/age/coverage errors raise inside `bench_northset` — **no** join summary line / blob dump. See join_coverage / book_age ops + family dgp sections.


## `sweep_evidence.volatility_regimes` keys (no H-ids yet)

Nested under `receipt["sweep_evidence"]["volatility_regimes"]` from `sweep_evidence_battery`.

### Product stance (docs-only — same bar as nested `kyle_ofi`)

Until explicitly productized (new catalog H-ids + agent mint + soft-verify specs + ADR),
`volatility_regimes` rows **must not** mint soft-verify H-rows and **must not** be added to
`verify-research` finite→H catalogs. Treat as `research_only` diagnostics inside the sweep
battery. H33–H42 continue to use flattened IC / event / placebo / cost / fold keys only.
Do not treat regime `p_value` / `hac_t` as interchangeable with those flattened discovery bounds.

### Regime definition (PIT)

- Lagged vol: `shift(1)` then rolling std of log close-to-close returns (`vol_lookback`, min_samples = lookback) per `security_id`
- Same-date cross-sectional median of lagged vol → `sweep_vol_regime` = `high` if ≥ median else `low` (null if lagged vol null)
- Event subset filtered by `regime` before horizon-1 excess-return HAC (`sweep_excess_ret_1`)

### Row schema (list of dicts)

| Key | Meaning |
|---|---|
| `signal` | `sweep_reject_signed` or `sweep_follow_signed` |
| `regime` | `low` or `high` |
| `n_events` | Event count in that signal×regime slice |
| `n_dates` | Dates with finite daily series points |
| `mean_excess_bps` | HAC mean of date-level excess returns × 1e4 |
| `hac_t` | HAC t-stat |
| `p_value` | Two-sided p from mean_tstat |

Cardinality: 2 signals × 2 regimes = **4** rows (when battery completes). Sibling keys on the same blob: `event_studies`, `permutation_placebos`, `timing_contract`, `control_contract`, `horizons`, `episode_cooldown_bars`, `round_trip_cost_bps`, `fdr_cutoff`, `research_only`. Productizing regime H-ids would need an explicit ADR + agent mint + soft-verify specs (out of scope until approved).


## Sweep evidence blob (Northset receipt)

`bench_northset` attaches `sweep_evidence` from `sweep_evidence_battery`. Timing:
event known at close \(t\) → enter `open[t+1]` → exit `close[t+h]`; control is same-date
cross-sectional mean. Placebos are within-date score permutations. Not a live fill model;
`research_only`.

## Missingness

Do not `fillna(0)` by default. Policies: train-period median, missing indicator, model-native NA, drop feature, drop row. Policy is config `missing.policy`.

## Data manifest

`dipcatcher ingest` writes `data/metadata/data_manifest.json` with
`schema_version: 1`. The manifest must identify the configured `source` and
contain the canonical `bars`, `actions`, `master`, and `silver` artifacts. Each
artifact records its absolute path, row count, columns, and a SHA-256 digest.

`dipcatcher doctor` accepts the manifest only when every canonical artifact exists,
has a valid digest and nonnegative row count, hashes to the recorded digest,
resolves inside the configured data root, and its declared columns and row
count match the actual Parquet schema and row count. Unknown or missing
manifest schema versions are invalid. A valid manifest is lineage evidence,
not proof of data quality beyond the checks listed here.

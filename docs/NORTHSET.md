# Northset

**Northset** is Dipcatcher's order-book and candlestick research slice.
It is not a live matching engine and not a P&L claim. Research family blobs
use `research_only` / `claim=research_diagnostic_only` (they cannot carry a
`live_pnl_claim` key — `pnl` is a forbidden research-headline token).

See [ADR-021](decisions/021-northset-microstructure.md).

## What it scores

**Identities (bound)**

1. Daily OHLC envelope
2. Uncrossed L2 (best bid &lt; best ask)
3. Session candles reconstruct the daily open/high/low/close
4. Session volumes sum to the daily volume
5. Session candles chain (`close[i] = open[i+1]`)

**Candles**

6. Geometry: body, wicks, wick skew, gap, CLV, doji / hammer / engulfing / spinning-top / marubozu / shooting-star *rates* (shape flags, not trade signals)
7. Date-level IC of body, gap, wick skew, CLV vs next-bar return
8. Parkinson, Garman–Klass, Rogers–Satchell, Yang–Zhang vs close-to-close RV (QLIKE + DM)
9. Overnight² + open-to-close² vs Parkinson (QLIKE + DM); overnight share
10. Session realized variance vs daily close-to-close (QLIKE)
11. Session BNS jump share (RV vs bipower)
12. Upside / downside realized semivariance
13. Wilder true range (level)

**Books / impact / liquidity**

14. Date-level IC of top imbalance, depth imbalance, microprice−mid, OFI, OFI lag, queue imbalance, VPIN, book-slope, volume/range
15. Kyle λ on signed depth and on OFI
16. Roll, Corwin–Schultz, Abdi–Ranaldo spreads; quoted vs effective
17. Amihud illiquidity (level and IC vs |next return|)
18. Book-panel VPIN proxy and session bulk VPIN
19. Mid and OFI lag-1 autocorrelation

**Liquidity sweeps**

20. Prior-`sweep_lookback`-bar extreme takeouts (PIT: shifted rolling high/low)
21. Resolution: reclaim (close back inside) vs follow-through (close beyond), with sweep depth
22. Sweep rates, reclaim/follow shares, mean depth, conditional mean next-bar returns
23. Date-level IC of signed sweep-reject and signed sweep-follow depth

No Sharpe / Sortino / Calmar / pnl / nav keys. Date-level IC is never stacked name-dates.

## Sweep evidence battery

Producer flags: `northset.sweeps.liquidity_sweep_frame` (lookback default from
`northset.sweep_lookback` / CLI). PIT prior extreme = shifted rolling max/min over
`lookback` (≥2). Outputs include `sweep_eligible`, high/low/both flags, reclaim/follow
flags (unambiguous only — `sweep_both` excluded from reclaim/follow), depths, and signed
scores `sweep_reject_signed` / `sweep_follow_signed` / `sweep_depth_signed`
(`sweep_output_columns()`). Rates via `sweep_rates` (eligible-denominator).

Institutional battery: `northset.sweep_research.sweep_evidence_battery(sweep_frame, config)`
→ stamped on the Northset receipt as `sweep_evidence` (also drives scope via
`sweep_evidence_scope`).

| Blob key | Meaning |
|---|---|
| `timing_contract` | `event_close_then_next_open` — no event-close fill |
| `control_contract` | `same_date_cross_sectional_mean` excess return |
| `horizons` | From `northset.sweep_horizons` (default 1, 5, 20) |
| `round_trip_cost_bps` | Median PIT hurdle; each event uses lagged volatility: 2×(commission+half_spread+bps_per_turnover) + 2×impact_y·σ[t−1]·√participation·1e4 |
| `event_studies` | Per (`sweep_reject_signed`\|`sweep_follow_signed`) × horizon: next-open→close[t+h] XS mean, HAC t (lags=horizon), circular block bootstrap CI, hit rate, cost-adjusted one-sided p, chronological fold summary, `reject_fdr` |
| `volatility_regimes` | Same signals × horizon-1 split by PIT `sweep_vol_regime` low/high (lagged vol vs date median) |
| `permutation_placebos` | `within_date_permutation_test` per signal vs `sweep_excess_ret_1` — shuffle scores within date, keep return marginals; `placebo_p_value` |
| `fdr_cutoff` | Benjamini–Hochberg α=0.05 over event-study p-values |
| `inference_index` | `calendar_including_idle_zeros` — HAC/bootstrap/folds run on the panel calendar with idle dates as 0; `mean_excess_bps` stays event-conditional |
| `research_only` | Always true |

Sample adequacy: `n_events ≥ sweep_min_events` (default 30) and `n_dates ≥ sweep_min_dates` (default 20); else t/CI/p are NaN. Default `sweep_n_permutations ≥ 50` (config floor), typically 200. Statistical FDR rejection ≠ economic cost pass (`cost_adjusted_*`).

Repeated same-direction events inside `sweep_cooldown_bars` (default 5) count as one
episode. Chronological fold summaries purge `horizon` observations at fold boundaries.
Split-adjusted OHLC drives detection; total-return-aligned open/close drives exits.

Beyond the event studies, the battery carries institutional evidence blocks:

| Blob key | Meaning |
|---|---|
| `matched_controls` | Per signal: direction-matched event-minus-control excess difference vs same-date **eligible non-swept** names, HAC t (H44/H45). **Primary executable test is H45.** |
| `liquidity_matched_controls` | Same design restricted to the event's lagged dollar-volume quartile (H46/H47) |
| `name_clustered` | Per-security mean signed excess, iid t across names (`lags=0`; H49 follow) |
| `two_way_clustered` | Cameron–Gelbach–Miller two-way clustered t on event-level signed excess (date × name; H50 follow). Wild-cluster p is a few-cluster companion, not a new H-id. |
| `overnight_gaps` | Signed close-to-next-open excess on the idle-zero calendar (H51 follow). Untradeable under next-open entry. |
| `oot_holdouts` | Last chronological `1/n_folds` of event dates as a frozen holdout; same-sign bound (H48 follow) |
| `lead_diagnostics` | Negative control: today's signal vs *yesterday's* excess return — pre-trend loading disclosure, not an edge claim |
| `parameter_sensitivity` | Horizon-1 event study re-run across `sweep_sensitivity_lookbacks`; every cell is a disclosed trial (`n_parameter_trials`, `sign_stable_by_signal`) |
| `coverage` | Event counts per side, ambiguous double-sweeps, unique securities, top-security event share, event/calendar date coverage |
| `adv_participation` | Modeled fill vs lagged ADV on event rows (`participation_rate * event dvol / lagged ADV`) |
| `primary_test` | Predeclared `H45_northset_follow_control` (next-open, horizon 1, eligible non-swept controls) |
| `trial_ledger` | Counted secondaries; FDR family remains `northset_sweep_event_studies_only` |

Date-level IC of signed depths (H33/H34) remains a **descriptive** discovery score via `date_ic_series` on the fused Northset frame — not a substitute for the executable battery.




### How the battery lands on the Northset receipt

`bench_northset` stores the full blob under `sweep_evidence`, then **flattens horizon-1**
event rows and placebos onto top-level keys for catalog/agent consumption:

| Flattened key | Source |
|---|---|
| `sweep_reject_event_{p,t,mean_bps}` | `event_studies` row `signal=sweep_reject_signed`, `horizon=1` |
| `sweep_follow_event_{p,t,mean_bps}` | same for `sweep_follow_signed` |
| `sweep_*_cost_adjusted_mean_bps` / `_p` | cost-adjusted mean and one-sided p |
| `sweep_*_fold_positive_fraction` | chronological fold positive fraction |
| `sweep_reject_placebo_p` / `_observed_ic` | `permutation_placebos['sweep_reject_signed']` |
| `sweep_follow_placebo_p` / `_observed_ic` | `permutation_placebos['sweep_follow_signed']` |
| `sweep_min_fold_positive_fraction` | config echo (default 0.75) |
| `sweep_reject_control_diff_{p,t,mean_bps}` | `matched_controls['sweep_reject_signed']` |
| `sweep_follow_control_diff_{p,t,mean_bps}` | `matched_controls['sweep_follow_signed']` (primary executable test H45) |
| `sweep_*_liq_control_diff_{p,t,mean_bps}` | lagged-ADV quartile matched controls (H46/H47) |
| `sweep_follow_name_cluster_{p,t}` | per-security mean t (H49) |
| `sweep_follow_two_way_cluster_{p,t}` | date×name two-way clustered t (H50) |
| `sweep_follow_two_way_wild_p` | date-cluster Rademacher wild bootstrap companion to H50 |
| `sweep_follow_overnight_gap_{p,t,mean_bps}` | close-to-next-open untradeable gap (H51) |
| `sweep_overnight_gap_method` | `event_close_to_next_open` |
| `sweep_two_way_inference_index` | `event_rows_not_calendar_zeros` |
| `corwin_schultz_pair_scope` | `prior_and_current_bar` |
| `sweep_follow_oot_holdout_mean_bps` | last-fold holdout mean (H48 bound) |
| `sweep_primary_test_id` / `sweep_n_counted_trials` | predeclaration + trial ledger |
| `sweep_median_event_adv_participation` | crowding disclosure vs lagged ADV |
| `sweep_inference_index` | `calendar_including_idle_zeros` (HAC/bootstrap/folds on idle-zero calendar) |
| `sweep_n_ohlc_quarantined` | invalid OHLC prints excluded from PIT extremes |

Research agent bound checks (not FDR discoveries): **H39/H40** cost hurdle
(`*_cost_adjusted_mean_bps > 0`); **H41/H42** fold stability
(`*_fold_positive_fraction ≥ sweep_min_fold_positive_fraction`); **H48**
out-of-time same-sign. H33/H34 remain **descriptive** date-level IC discoveries
on signed depths, not executable event studies.

## Date-level IC (candle + LOB)

Institutional score for fused candle/LOB features is **date-level** IC, not a pooled panel Spearman.

**Implementation:** `quant_fund.metrics.cross_section.date_ic_series` (also used by `microstructure.bench.bench_candle_order_book` and `northset.benches`).

| Piece | Contract |
|---|---|
| Cross-section | Within each calendar date key, Spearman (rank IC) and Pearson across names with finite score and target |
| Aggregation | Mean of the date IC series; Newey–West HAC t/p on that series (`mean_tstat`) |
| Default `min_names` | 5 in `date_ic_series`; candle+LOB bench default `min_names=4`; Northset config may override |
| Target | Next-bar simple close return: `close[t+1]/close[t] - 1` per `security_id` (shifted, then drop last null) |
| Alignment | `attach_candle_book_features` asof-joins candles (`decision_time`) to book (`book_available_time`) by `security_id`, backward, with `max_book_age_seconds` (default 86400). Stamps `join_coverage`, `book_age_seconds`, `book_source`, `book_dgp`. Empty fuse or coverage below floor → `ValueError` |
| Receipt keys | `ic_method=date_level_spearman_hac`; per-feature `ic_*`, `ic_*_pearson`, `ic_*_t`, `ic_*_p`, `ic_*_n_dates` |
| Forbidden | Sharpe / Sortino / Calmar / `pnl` / `nav` / `live_pnl_claim` on research blobs |

**What date-level IC is not**

- Not stacked name-dates treated as i.i.d. rows (that inflates n and understates SE).
- Not a live-tape or live-L2 claim — default books are SYNTHETIC (`synthetic_lob`) or offline vendor-remapped panels.
- Not a trade signal or capacity estimate; discovery scores (H22/H25–H27/H30/H32–H34) can be ≠ 0 on SYNTHETIC without implying edge.
- ICIR annualization (`×√252`) is only when explicitly reported; candle+LOB bench headlines use mean Spearman + HAC t.

Feature set scored by `bench_candle_order_book` includes candle geometry (`candle_body_ret`, wicks, …), book metrics (`imbalance_*`, `ofi`, slopes, …), and cross terms (`candle_dir_x_imbalance`, …). Northset’s broader bench adds sweep / session-path ICs under the same date-level rule.



## Canonical market view (corp-action-safe)

`quant_fund.northset.data_view.canonical_northset_bars` → `NorthsetMarketView`.

| Field | Contract |
|---|---|
| Input | Raw bronze bars plus, when present, full split-adjusted OHLC quartet |
| Default | `require_adjusted=True` (config `northset.require_adjusted_ohlc`, default true) |
| Adjusted path | Overwrites `open/high/low/close` with `*_split_adjusted`; volume prefers `volume_split_adjusted`, else `volume * split_factor` if present, else raw volume |
| `return_close` | `close_total_return` if present, else `close_split_adjusted` |
| `return_open` | Scaled open when TR close exists: `open_split_adjusted * close_total_return / close_split_adjusted`, else split-adjusted open |
| `price_basis` | `split_adjusted` |
| `return_basis` | `total_return` if TR close present, else `split_adjusted` |
| Raw opt-out | Only if `require_adjusted=False`: `price_basis=return_basis=raw_fixture_opt_out` (fixtures without corp actions) |
| Fail closed | Missing required raw cols; **partial** adjusted quartet; adjusted required but absent; `ohlc_identity_rate(frame) < 1` |

Split-adjusted prices drive candle geometry and sweep detection; TR close drives forward exits when available. `bench_northset` routes bars through this view before scoring. Not a live feed claim. Full stamp matrix: DATA_CONTRACTS **`price_basis` / `return_basis` stamp matrix**.


### Receipt stamps (`bench_northset`)

`northset.benches.bench_northset` always stamps the canonical view onto the family receipt:

| Key | Source |
|---|---|
| `price_basis` | `NorthsetMarketView.price_basis` (`split_adjusted` or `raw_fixture_opt_out`) |
| `return_basis` | `NorthsetMarketView.return_basis` (`total_return`, `split_adjusted`, or `raw_fixture_opt_out`) |

These sit beside `dgp` / `book_dgp` / `component_sources` / `label`. They are honesty metadata, not live Sharpe. Fixture opt-out remains visible as `raw_fixture_opt_out` on both keys.


### Evidence label / scope matrix

`bench_northset` stamps `label` (= `data_source`), `dgp`, `component_sources`, and `sweep_evidence_scope` together. Full branch order + forbidden claims: DATA_CONTRACTS **Evidence label / component_sources / sweep_evidence_scope matrix**.

| `label` (headline) | When |
|---|---|
| `SYNTHETIC` | Synth bars+synth book; or synth bars+vendor book (label stays SYNTHETIC, `dgp` follows book) |
| `MIXED_SYNTHETIC_DERIVED` | Empirical bars + synthetic book (session L2 alone does not MIX a vendor-book empirical run) |
| empirical `bar_source` | Empirical bars + vendor/external book |

`component_sources.session_candles` is always `synthetic_reconstruction`. `sweep_evidence_scope` ∈ `{synthetic, empirical_adjusted, fixture_raw_unadjusted}` from bars + `price_basis` (not from MIXED).



### Family `dgp` / `book_dgp` values

| Receipt key | Common values |
|---|---|
| `book_dgp` | `synthetic_lob` (synthesized) or `vendor_panel:{source}` (panel path) |
| `dgp` | `synthetic_lob` / `vendor_panel:…` (follows book when vendor) / `mixed_sources` / `empirical` |

Fuse-frame column may say `external_panel` while receipt says `vendor_panel:…` — not a contradiction. Full matrix + forbidden mixes: DATA_CONTRACTS **Family dgp / receipt book_dgp**.

### Evidence eligibility + session-L2 identity floor

Also stamped by `bench_northset` (honesty, not alpha):

| Receipt key | Contract |
|---|---|
| `component_sources` | Dict: `bars`, `book`, `session_candles` (`synthetic_reconstruction`), `session_book` (`synthetic_reconstruction` or `disabled`) |
| `book_hypothesis_eligible` | `bars_synthetic or not book_synthetic` — SYNTHETIC book on empirical bars → **false** |
| `session_book_hypothesis_eligible` | `bars_synthetic` only — session multi-snap L2 is reconstructed plumbing |
| `sweep_evidence_scope` | `synthetic` / `empirical_adjusted` / `fixture_raw_unadjusted` from bars + price_basis |
| `session_l2_identity_floor` | Config `northset.session_l2_identity_floor` (default **0.99**) |
| `session_l2_identity_gate` | `enforced` when `use_session_l2` else `skipped` |

`book_synthetic` when `book_source` ∈ `{synthetic, synthetic_lob, synthetic_reconstruction}`.

**End-to-end:** setter `bench_northset` → agent mint skips H21/H22/H25/H27/H32 when `book_hypothesis_eligible` false; skips H43 when `session_book_hypothesis_eligible` false → soft `verify-research` uses the same skips (finite metric does not demand the H-row). Full truth table + ungated H-ids: DATA_CONTRACTS **Evidence eligibility stamps**.

When enforced, `enforce_session_l2_identity_floors` fail-closes if any of
`ohlc_identity_rate`, `session_ohlc_identity_rate`, `session_reconstructs_daily_rate`,
`session_volume_conservation_rate`, `session_chain_rate`, `book_uncrossed_rate` is NaN or
below the floor. Rates below floor are **data-contract bugs**, not discovery scores.

## Fail-closed integrity gates

### Session book counts — `validate_session_book_counts`

In `quant_fund.northset.identities`. Every `(security_id, parent_event_time)` must have
**exactly** `n_session_candles` rows and that many unique `session_index` values
(`n_session_candles >= 2`). Empty frame / missing keys / any parent mismatch → `ValueError`.
Called from `bench_northset` and CLI `session-book` after `synthesize_session_l2`.

Session identity rates (same module) remain bound checks: `session_reconstructs_daily_rate`,
`session_volume_conservation_rate`, `session_chain_rate` — failing these is a **data-contract
bug**, not alpha.

### Candle/book join coverage — `attach_candle_book_features`

| Rule | Contract |
|---|---|
| Join | Polars `join_asof` on `decision_time` ↔ `book_available_time`, `by=security_id`, `strategy=backward`, `tolerance=max_book_age_seconds` |
| Coverage | `join_coverage = n_fused / n_candle_rows`, stamped on every fused row |
| Default floor | SYNTHETIC synthesize path: `min_join_coverage=1.0`; external panel: `0.5` (override via arg) |
| Fail closed | Empty fuse, coverage below floor, missing PIT `available_time`, missing/mixed/null external `source` |
| Honesty stamps | `book_source`, `book_dgp`, `book_age_seconds` |

Receipts (`bench_candle_order_book` / `bench_northset`) surface `join_coverage`, `mean_book_age_seconds`, `max_book_age_seconds`, and `book_join_coverage_floor`. Ops / fail-closed list: DATA_CONTRACTS **join_coverage / book_age receipt fields**.

### Book panel depth honesty — `validate_book_panel_depth_honesty`

Invoked from `validate_book_panel` when `n_bid_levels` / `n_ask_levels` exist and any
`DEPTH_SHAPE_FIELDS` columns are present (`book_metrics.DEPTH_SHAPE_FIELDS`, also listed
in `BOOK_PANEL_OPTIONAL`):

`bid_log_size_slope`, `ask_log_size_slope`, `bid_log_price_slope`, `ask_log_price_slope`,
`bid_mean_log_tick_spacing`, `ask_mean_log_tick_spacing`.

| Side depth | Contract |
|---|---|
| thin (`n_*_levels < 2`) | Every present shape field on that side must be null/NaN |
| deep (`n_*_levels >= 2`) | `*_log_size_slope` and `*_log_price_slope` must be **finite** |
| deep + tick spacing | `*_mean_log_tick_spacing` may stay NaN if any adjacent gap is ≤0 (Commander residual #3); thin-side NaN still required |

Absent level columns → no-op (legacy top-of-book remaps). Producer: `book_metrics_from_snapshot`
(Commander residual #2 price geometry + #3 `DEPTH_SHAPE_FIELDS` export).


**`SIDE_STRUCTURE_FIELDS`:** `bid_size_concentration_top`, `ask_size_concentration_top`
(= top size / side depth). Finite when that side’s depth > 0; NaN if depth ≤ 0. Not gated by `n_*≥2`.

**Northset optional floors** (config default **None** = off): `depth_shape_finite_floor`,
`concentration_top_finite_floor`, `queue_priority_finite_floor`, `side_notional_finite_floor`,
`tob_size_share_finite_floor`. **Never mix** DEPTH_SHAPE slopes with SIDE/QUEUE/NOTIONAL/TOB rates —
DATA_CONTRACTS **Shape / structure floors matrix**.
When set, `bench_northset` fail-closes if the matching receipt rate is NaN or below the floor.
Ops: DATA_CONTRACTS **queue_priority / side_notional finite rates** + depth honesty checklist.


### External panel vs SYNTHETIC ensure (floors)

`ensure_book_panel_shape_columns` runs on **SYNTHETIC** daily/session L2 only — not on
`book_panel_path` (ensures DEPTH_SHAPE + SIDE/QUEUE/NOTIONAL structure). Vendor remap uses
`n_*=1` and NaN size slopes; does **not** invent queue/notional proxies. Top-of-book →
`depth_shape` / `queue_priority` / `side_notional` rates typically **NaN**; optional floors
fail-closed only when set. Details: DATA_CONTRACTS queue/notional + external-panel sections.

### Kyle / OFI family — `quant_fund.northset.kyle_ofi`

Standalone diagnostics: `bench_kyle_ofi_fused` (also CLI `dipcatcher kyle-ofi`).

- Cont–Kukanov–Stoikov OFI: `cont_ofi_series` / `cont_ofi_by_security` on consecutive tops
- Kyle λ: per-date OLS `Δmid = λ · flow` for `flow ∈ {signed_depth, ofi}`, then HAC on the λ series (`kyle_lambda_by_date`)
- Date-level IC: `date_ic_series` of flow→Δmid and OFI→Δmid at lag 0 / lag 1 (`ofi_delta_mid_date_ic`)
- Receipts: `research_only=true`, `claim=research_diagnostic_only`, `ic_method=date_level_spearman_hac`; forbidden Sharpe/pnl/nav/live_pnl_claim keys assert-fail

#### Optional nest on the Northset receipt (`include_kyle_ofi`)

Config: `northset.include_kyle_ofi` (**default `false`**). When true, `bench_northset` calls
`bench_kyle_ofi_fused` on the same bars/book (honoring `book_panel_path` / join floor) and:

| Receipt key | When flag true | When flag false |
|---|---|---|
| `kyle_ofi` | Full `bench_kyle_ofi_fused` blob (must already carry `research_only`) | **absent** |
| `include_kyle_ofi` | `true` | `false` |

Fail-closed: nested blob missing or without `research_only` → `AssertionError`. The nest is **not** a separate catalog-v2 required family; it rides inside family `northset`. Research CLI `dipcatcher northset` echoes `include_kyle_ofi=…`. Standalone `dipcatcher kyle-ofi` always runs the fused bench without needing the flag.

**Distinct from** always-on Northset scalars `kyle_ofi_lambda` / `kyle_ofi_r2` / `kyle_ofi_n_securities` (from the main bench’s OFI/Kyle OLS path on the fused frame). Those stay even when `include_kyle_ofi=false`.

#### Always-on scalars vs nested `kyle_ofi` blob

| Surface | Keys (examples) | Estimator | Aggregation |
|---|---|---|---|
| **Always-on** Northset receipt (flag irrelevant) | `kyle_lambda`, `kyle_r2`; `kyle_ofi_lambda`, `kyle_ofi_r2`, `kyle_ofi_n_securities` | Per-`security_id` OLS `kyle_lambda(Δmid, q)` with `q∈{signed_volume, ofi}` via `_panel_kyle` | Mean of finite per-name λ / R²; `n` = count of names with finite λ |
| **Nested** `receipt["kyle_ofi"]` when `include_kyle_ofi=true` | `kyle_lambda_depth_mean/_t/_p`, `kyle_lambda_ofi_mean/_t/_p`, `*_n_dates`; `depth_flow_delta_mid_*`; `ofi_flow_delta_mid_*`; `ofi_delta_mid_lag0_*` / `lag1_*`; `ic_method`; `book_source`/`book_dgp`/`join_coverage` | `kyle_lambda_by_date` (per-**date** cross-section λ + HAC on λ series) + `date_ic_series` flow→Δmid / OFI→Δmid | Date-level institutional IC/HAC — **not** the same object as always-on name-mean λ |

**Never equate (pair matrix):**

| Always-on | Nest | Axis mismatch |
|---|---|---|
| `kyle_ofi_lambda` | `kyle_lambda_ofi_mean` (+ `_t`/`_p`/`_n_dates`) | name-mean OLS vs date λ + HAC |
| `kyle_ofi_r2` | nest HAC t/p | R² ≠ Newey–West inference |
| `kyle_ofi_n_securities` | `kyle_lambda_ofi_n_dates` / `kyle_lambda_date_series_n_ofi` | names vs dates |
| `kyle_lambda` | `kyle_lambda_depth_mean` | same split for signed_volume vs signed_depth |

Nested blob absent when `include_kyle_ofi=false`; always-on scalars remain. Soft-verify does **not** require numeric agreement. DATA_CONTRACTS **Always-on kyle_ofi_lambda vs nest kyle_lambda_ofi_mean**; OPS **`--dump-lambda-series` / date_series honesty**.

Same imbalance formula `(bid_depth - ask_depth)` is labeled **`signed_volume`** on the always-on fuse path and **`signed_depth`** inside `kyle_ofi` / nested nest — alias mismatch, not a different signal.

Full side-by-side fuse column map (fwd returns, OFI lag, honesty stamps): DATA_CONTRACTS **Always-on fuse vs kyle_ofi fuse columns**.

**Product stance:** nested `receipt["kyle_ofi"]` **must not** mint soft-verify **H-rows** unless/until productized. **Honesty suite is live:** `verify-research` runs `kyle_ofi_nest_honesty_errors` (join_coverage, residual, dispersion, ofi_depth_corr, SYNTHETIC/SYN* label, claim umbrella, `ic_method`, `family`, `hac_lags`). Nest stays diagnostic under `include_kyle_ofi`. DATA_CONTRACTS **Kyle nest soft-verify suite**.


Not a live tape. No invented empirics — score meanings only.


## SYNTHETIC books

`synthesize_l2_from_bars` builds one L2 snapshot per daily bar from range, body, and volume.
Optional `northset.book_panel_path` loads a vendor-shaped parquet panel with the same columns.
Tag every synthetic number SYNTHETIC.

Fusion selects the latest snapshot satisfying `book_available_time <= decision_time`
and `book_max_age_seconds`. Book rows must be unique, non-null, UTC, finite, positive,
and uncrossed. Reconstructed session rows are available only at parent-bar availability.
Empirical bars mixed with **synthetic books** are stamped `MIXED_SYNTHETIC_DERIVED` (`dgp=mixed_sources`); vendor book on empirical bars keeps an empirical/`bar_source` label even though `session_candles` stay `synthetic_reconstruction`. Ineligible synthetic-book hypotheses are not minted as empirical evidence. See evidence-label matrix. Receipts hash adjusted silver content and external book bytes.

## CLI

```bash
uv run dipcatcher northset --config configs/research.yaml
# optional: set northset.include_kyle_ofi: true to nest receipt["kyle_ofi"]
uv run dipcatcher kyle-ofi --config configs/research.yaml
uv run dipcatcher research --config configs/research.yaml   # includes family northset
uv run dipcatcher book-panel --config configs/research.yaml
```

## Catalog

Family `northset` is required in benchmark catalog **version 2**.

| Hypothesis | Family | Meaning |
|---|---|---|
| H20_northset_ohlc | bound | `ohlc_identity_rate ≥ 1` |
| H21_northset_book | bound | `book_uncrossed_rate ≥ 1` |
| H22_northset_imbalance | discovery | date-level imbalance IC ≠ 0 |
| H23_northset_session | bound | session envelope reconstructs daily OHLC |
| H24_northset_volume | bound | session volumes sum to daily volume |
| H25_northset_microprice | discovery | microprice−mid date IC ≠ 0 |
| H26_northset_wick | discovery | wick-skew date IC ≠ 0 |
| H27_northset_ofi | discovery | OFI date IC ≠ 0 |
| H28_northset_gk | discovery | Garman–Klass vs Parkinson QLIKE (DM) |
| H29_northset_chain | bound | session candles chain |
| H30_northset_clv | discovery | close-location-value date IC ≠ 0 |
| H31_northset_overnight | discovery | overnight+OC split vs Parkinson QLIKE (DM) |
| H32_northset_vpin | discovery | VPIN date IC ≠ 0 |
| H33_northset_sweep_reject | discovery | signed sweep-reclaim depth date IC ≠ 0 (**descriptive**, not executable) |
| H34_northset_sweep_follow | discovery | signed sweep follow-through depth date IC ≠ 0 (**descriptive**, not executable) |
| H35_northset_reject_event | discovery | reclaim event-study p (finite → soft-verify) |
| H36_northset_follow_event | discovery | follow event-study p |
| H37_northset_reject_placebo | discovery | reclaim placebo p |
| H38_northset_follow_placebo | discovery | follow placebo p |
| H39_northset_reject_cost | bound | reclaim cost-adjusted mean bps > 0 |
| H40_northset_follow_cost | bound | follow cost-adjusted mean bps > 0 |
| H41_northset_reject_stability | bound | reclaim fold-positive fraction ≥ `sweep_min_fold_positive_fraction` |
| H42_northset_follow_stability | bound | follow fold-positive fraction ≥ floor |
| H43_northset_session_book_vpin | discovery | session_book_vpin date IC (eligible session path) |
| H44_northset_reject_control | discovery | reclaim beats direction-matched same-date eligible non-swept controls |
| H45_northset_follow_control | discovery | follow-through beats matched non-event controls (**primary executable**) |
| H46_northset_reject_liq_control | discovery | reclaim beats lagged-dvol-quartile matched controls |
| H47_northset_follow_liq_control | discovery | follow-through beats lagged-dvol-quartile matched controls |
| H48_northset_follow_oot | bound | follow-through last-fold holdout mean keeps in-sample sign |
| H49_northset_follow_name_cluster | discovery | follow-through per-security mean excess ≠ 0 (iid across names) |

**VPIN disambiguation:** `vpin_mean`/`vpin_proxy` → **H32** (`book_hypothesis_eligible`); `session_book_vpin_mean` → **H43** (`session_book_hypothesis_eligible`); `session_bulk_vpin` is a third scalar (candle volume sign) — no H-id. Compact CLI `vpin=` is `vpin_mean` only; session means are blob-only. `session_close_*` = last session snap aggregates (not VPIN). DATA_CONTRACTS **session_book_vpin_mean vs vpin_mean**.

Flattened sweep battery → H33–H42 field map: DATA_CONTRACTS **Sweep evidence battery → H33–H42**. Full nested `event_studies` row schema (all horizons): DATA_CONTRACTS **event_studies full row schema**. `permutation_placebos` nest: DATA_CONTRACTS **permutation_placebos nested key schema**. `sweep_evidence.volatility_regimes` **must not** mint soft-verify H-rows until productized (same stance as nested `kyle_ofi`): DATA_CONTRACTS **volatility_regimes keys**. Soft `verify-research` consistency: finite northset receipt field → matching H-row (see DATA_CONTRACTS verify-research table). Distinct from live `session_l2_identity_floor`. Nested `kyle_ofi` blob keys **must not** be soft-verified H sources unless/until productized.

## Session multi-snapshot L2

When `northset.use_session_l2` is true (default), Northset builds **one SYNTHETIC L2 snapshot per session candle**, then aggregates path stats onto the daily bar.

**Semantics (honest)**

| Term | Meaning here |
|---|---|
| Session candles | Intraday OHLC reconstructed from the daily bar (`session_candles_from_daily`); not a vendor RTH/ETH tape |
| Session L2 | `synthesize_session_l2`: L2 derived from each session candle’s OHLC/volume (`revision_id` / source still SYNTHETIC) |
| Clocks | Snapshot `event_time` = session stamp; daily join key is `parent_event_time` → daily `event_time` |
| Depth | Configurable ladder depth (default 5) on the synthesizer — not exchange depth replay |
| Aggregation | `aggregate_session_book_to_daily`: last snapshot for close state; path OFI / imbalance / spread / VPIN over the day |

Daily research columns from `aggregate_session_book_to_daily`:

- **Last snap:** `session_close_mid`, `session_close_spread_bps`, `session_close_imbalance`, `session_close_micro_bps`, `session_close_bid_depth`, `session_close_ask_depth`
- **Path means/counts:** `session_imbalance_mean` (/`_std`), `session_spread_bps_mean`, `n_session_book_snaps`
- **Path OFI / session-book VPIN:** `session_ofi_sum`, `session_ofi_abs_sum`, `session_book_vpin` (H43) — ≠ daily `vpin` (H32)

Full table + IC/CLI notes: DATA_CONTRACTS **session_close_*** and **session_book_vpin_mean vs vpin_mean**. Also **session_imbalance_std / session_book_source** (IC vs receipt-only). OPS: **VPIN: H32 vs H43** / **session_close aggregates**.

These are **research diagnostics on SYNTHETIC microstructure**, not live L2, not SIP, not a matching-engine replay. Disable with `northset.use_session_l2=false` (receipt then has `n_session_book_rows=0`).

```bash
uv run dipcatcher session-book --out data/book_panel/session_l2.parquet \
  --daily-out data/book_panel/session_l2_daily.parquet
```



### Session L2 clocks vs `book_max_age_seconds`

Daily candle↔book fuse uses asof + `book_max_age_seconds`. Session multi-snap L2 aggregates
to daily on **exact** `(security_id, parent_event_time → event_time)` — not an age window.
Changing `book_max_age_seconds` does not retune session path stats. Details: DATA_CONTRACTS
**max_book_age vs session L2 daily aggregation clocks**.

## Book metrics public surface

`quant_fund.microstructure.book_metrics` is the L2 snapshot → flat metrics producer.

**Always-on top / depth totals:** `best_bid/ask`, `mid`, `spread`, `spread_bps`, `microprice`,
`microprice_minus_mid(_bps)`, `imbalance_top`, `imbalance_depth`, `bid_depth`, `ask_depth`,
`top_bid_size`, `top_ask_size`, `n_bid_levels`, `n_ask_levels`.

**Aliases (book_metrics):** `half_spread` (= spread/2), `quoted_spread_bps` (= spread_bps), `effective_spread` (= spread on snapshot), `touch_size_imbalance` (= imbalance_top), `half_spread_bps`, `spread_over_mid`. Full table: DATA_CONTRACTS **Spread / imbalance aliases**.

**Northset dual columns (FIXED 2026-09-16):** fuse keeps book `effective_spread` (ask−bid alias). Candle diagnostic is `close_mid_abs_rel` = `2|candle_close−mid|/mid`. Receipts: `mean_effective_spread` = nanmean(book alias); `mean_close_mid_abs_rel` = nanmean(candle col); `mean_spread_bps` / `mean_quoted_spread` remain quoted. DATA_CONTRACTS **dual columns** + **mean_spread_bps / mean_effective_spread / mean_close_mid_abs_rel**.
**Spread families:** touch means `mean_quoted_spread` / `mean_spread_bps` ≠ OHLC/mid estimators `roll_spread` / `corwin_schultz_spread` / `abdi_ranaldo_spread`. **Flow vs level:** `queue_imbalance_mean` (TOB level) ≠ OFI (flow; no plain `ofi_mean`; see `ofi_lag1_corr` / `kyle_ofi_*` / ICs). DATA_CONTRACTS **Touch means vs OHLC** and **queue_imbalance vs ofi**.

**CLI spread means echo (current):** `dipcatcher northset` prints `mean_quoted_spread` / `mean_effective_spread` / `mean_half_spread` / `mean_half_spread_bps` / `mean_spread_bps` / `mean_close_mid_abs_rel` / `mean_microprice_weight_balance`. Research compact line includes `mean_spread_bps` + half/close-mid/microprice means (not `mean_quoted_spread`). DATA_CONTRACTS **CLI echo: quoted / half / close–mid / microprice means**. `session_ofi_sum_mean` remains blob-only.




## Northset receipt means — CoS overnight stamp wave

Always-on `bench_northset` nanmeans on the fused/book frame (research_only honesty — **never live Sharpe**). Companion never-equates live in DATA_CONTRACTS; this table is the receipt inventory.

| Family | Receipt means (examples) | Notes |
|---|---|---|
| **Depth / levels** | `mean_bid_depth`, `mean_ask_depth`, `mean_top_bid_size`, `mean_top_ask_size`, `mean_n_bid_levels`, `mean_n_ask_levels` | ≥0 when finite; levels count ≠ slope |
| **Imbalance** | `mean_imbalance_top`, `mean_depth_imbalance`, `mean_depth_imbalance_abs`, `mean_notional_imbalance` | Top ≠ depth ≠ abs ≠ notional; `imbalance_top = 2w−1` vs MWB when finite |
| **Concentration / queue** | `mean_bid_size_concentration_top`, `mean_ask_size_concentration_top`, `mean_queue_priority_proxy`, `mean_ask_queue_priority_proxy` | concentration `top/side` ≠ queue `top/(top+side)` (Commander #59) |
| **Notional / TOB share** | `mean_tob_size_share`, `mean_tob_notional_share`, `mean_side_notional_proxy_*`, `mean_top_of_book_notional_proxy` | size ≠ notional; `mean_tob_notional_share` soft-verify ∈(0,1] + Sergeant receipt-stamp / verify wire — DATA_CONTRACTS **Sergeant free-lane mean_tob_notional_share** |
| **Spread / mid** | `mean_quoted_spread`, `mean_effective_spread`, `mean_half_spread`, `mean_half_spread_bps`, `mean_spread_bps`, `mean_spread_over_mid`, `mean_close_mid_abs_rel` | Dual-col + identity soft-verify; `spread_over_mid` ≠ bps scale |
| **Microprice** | `mean_microprice_weight_balance`, `mean_microprice_minus_mid`, `mean_microprice_minus_mid_bps` | `w` ≠ μ−mid ≠ bps companion |
| **Book slope / tick** | `mean_bid_log_size_slope`, `mean_ask_log_size_slope`, `mean_bid_log_price_slope`, `mean_ask_log_price_slope`, `mean_bid_mean_log_tick_spacing`, `mean_ask_mean_log_tick_spacing` | size slope ≠ price slope ≠ tick spacing; thin→NaN |
| **Join age** | `mean_book_age_seconds` (+ `max_book_age_seconds`) | mean ≠ max ≠ join_coverage |
| **Illiquidity / session** | `mean_true_range`, `amihud_mean`, session means (`mean_session_*`, `session_*_mean`) | Session vs daily never-equate elsewhere |

**Do not invent** receipt means that are not stamped (e.g. historical gap notes for session_imbalance_mean — check live benches before claiming). Soft-verify honesty helpers may gate a subset only.


**Session vs bar flow/level (validity):** `session_ofi_sum*` (synthetic_lob session aggregate) ≠ daily `ofi`/`ofi_lag` ≠ `queue_imbalance` (level). Fail-closed when `use_session_l2` + empty session; identity floor enforced. Receipt name map: DATA_CONTRACTS **Session OFI vs bar-level**. **Illiquidity means:** `amihud_mean` / `mean_true_range` (research_only — **no live Sharpe**). DATA_CONTRACTS **Illiquidity means**.





`depth_imbalance_abs` = |imbalance_depth|; `spread_over_mid` = spread/mid (= spread_bps/1e4 when mid>0) — not overwritten by that fuse step.

**`microprice_weight_balance`:** top_bid/(top_bid+top_ask); microprice = ask·w + bid·(1−w). DATA_CONTRACTS **microprice_weight_balance export**. Triad refresh: DATA_CONTRACTS **microprice vs mid vs microprice_weight_balance**.

**`DEPTH_SHAPE_FIELDS` (NaN-if-thin):**

| Field | Definition (side has ≥2 positive finite levels) |
|---|---|
| `*_log_size_slope` | OLS slope of \(\log(\mathrm{size})\) on level index 0=inside |
| `*_log_price_slope` | OLS slope of \(\log(\mathrm{price})\) on level index (Commander residual #2) |
| `*_mean_log_tick_spacing` | Mean \(\log|p_i-p_{i-1}|\) over adjacent levels; NaN if any gap ≤0 (#3) |

**`SIDE_STRUCTURE_FIELDS`:** `bid/ask_size_concentration_top` — finite when side depth > 0.

**`QUEUE_STRUCTURE_FIELDS`:** `queue_priority_proxy` / `ask_queue_priority_proxy` — `top/(top+side_depth)`; finite when depth > 0.

**`SIDE_NOTIONAL_FIELDS`:** `side_notional_proxy_bid/ask` — `best·side_depth`; finite when best>0 and depth>0.

**`TOB_SHARE_FIELDS`:** `tob_size_share` only — `(top_bid+top_ask)/(bid_depth+ask_depth)`, honest **∈(0,1]** (Commander **#53** / legacy #34). Adjacent **`tob_notional_share`** (touch-priced TOB / best·depth notionals) also **∈(0,1]** after Commander **#62** (**#54** / legacy #35) — not in `TOB_SHARE_FIELDS`. Size-share ≠ notional-share — never equate; both unit-interval. Full checklist: DATA_CONTRACTS **tob_size_share vs tob_notional_share**; OPS **TOB size vs notional share**.

Notional building blocks: `top_of_book_notional_proxy` = `best_bid·top_bid + best_ask·top_ask` (#62); `side_notional_proxy_*` = best·side_depth; `notional_imbalance` = (bid−ask)/(bid+ask) notionals. Full formulas: DATA_CONTRACTS **Notional proxies**.

Receipt disambiguation: northset stamps **`mean_tob_size_share`**; candle+LOB bench stamps **`tob_size_share_finite_rate`** — not interchangeable (DATA_CONTRACTS).

Empty / crossed / locked books are rejected before metrics (`OrderBookSnapshot` + metrics fail-closed).
Panel interchange: required top-of-book columns in `BOOK_PANEL_REQUIRED`; shape fields optional via
`BOOK_PANEL_OPTIONAL = (…, *DEPTH_SHAPE_FIELDS, …)`. MATH_SPEC states the same NaN-if-thin rule;
validators enforce it when columns are present.

**CoS receipt honesty dispatcher:** `northset_receipt_honesty_errors` fans `NORTHSET_RECEIPT_HONESTY_HELPERS` (**40** live). Tuple+dispatcher at catalog **EOF** (helpers defined first — Lt NameError fix). Separate from kyle nest (**26**) and session-means fans. DATA_CONTRACTS **CoS northset_receipt_honesty_errors**.

**Sergeant verify fan-in:** `northset_receipt_honesty_errors` wired in `verify-research` — all **40** receipt helpers reach northset checks (~28 newly via fan-in). **Candle gap:** `structure_finite_rate_honesty` expects `finite_rate_*` stamps candle bench may still omit (stamp = CoS/Sergeant code). **Lt:** Commander box #73–#100 → Mac merge **SKIPPED** (Mac through #175; box behind). DATA_CONTRACTS those sections. **Never-equate mesh (confirm):** CoS H20↔H24; Sergeant H21↔H29; Commander gap↔H23; General H21↔H24; Commander gap↔volume; Sergeant H22↔H29; Commander H22↔gap; CoS H22↔H24; Lt H22↔H23 (mesh edges complete); IC↔gap SPECS H25–H45 closed; reject≠follow siblings COMPLETE H33≠H34…H44≠H45; CoS spread_bps alias — DATA_CONTRACTS.

**Stamp-test wave mean_* COMPLETE (0 remaining):** Sergeant `structure_finite_rate` real stamp + derivation **6/6 GREEN** (NOT inflight); CoS `test_mean_session_means_receipt_stamp_batch.py` **4/4 GREEN** (11 keys) + `_nanmean` asarray; candle ≠ northset. True INFLIGHT only: Sergeant soft-verify `structure_finite_rate` on northset / CLI polish. DATA_CONTRACTS **Stamp-test wave — mean_* COMPLETE**.

**CLI frozenset GREEN (NOT INFLIGHT):** blob_only **4/4** + REQUIRED finite-on-synth **3/3**. CoS `queue_priority` stamp upgrade **3/3**; FEATURE_COLS structure IC presence **5/5**. IDLE awaiting next assign (no honesty-residual INFLIGHT). DATA_CONTRACTS **Sergeant REQUIRED finite-on-synth**.
**CLASSIFIED GREEN:** `NORTHSET_RECEIPT_CLASSIFIED` **206** = REQUIRED∪EXTRA∪BLOB_ONLY (51+65+90, disjoint) + classification soft-verify fail-closed; candle `join_coverage` verify wire. CoS FEATURE_COLS IC completeness **3/3** + `mid_lag1_corr` stamp **3/3**. DATA_CONTRACTS **Northset receipt key CLASSIFIED partition**.
**Classify-or-echo GREEN 11/11;** EXTRA finite-on-synth **3/3 GREEN** (NOT INFLIGHT). CoS FEATURE_COLS +`ask_queue_priority_proxy` +`microprice_weight_balance`; bid/ask queue pair honesty. DATA_CONTRACTS **Sergeant classify-or-echo**.
**FEATURE_COLS IC pack 12/12 GREEN:** Spearman/Pearson/t completeness + lag1 n_securities; ask_queue+MWB in FEATURE_COLS. EXTRA finite-on-synth **3/3**. DATA_CONTRACTS **CoS FEATURE_COLS Spearman/Pearson completeness**.
**BLOB_ONLY∉CLI leak lock + book_age fuse 5/5 GREEN.** FEATURE_COLS Spearman/Pearson+lag1 **12/12** confirmed. INFLIGHT: Sergeant session L2 / max_book_age / claim+floors. DATA_CONTRACTS **Sergeant BLOB_ONLY∉CLI leak lock**.
**CoS MWB fuse⇒mean + overnight/rv/semi + notional IC 6/6 GREEN.** **Sergeant session L2 identity + candle book_age 9/9 GREEN** (promotes prior session-L2 INFLIGHT). DATA_CONTRACTS **CoS candle MWB fuse** / **session L2 identity**.
**ic_method honesty 6/6 GREEN** (`candle_order_book_ic_method_honesty_errors`). **session_l2_enforced + notional IC⇒mean 5/5 GREEN.** Session L2 + candle book_age **9/9** stays GREEN (not INFLIGHT). Snaps vs n_session **7/7 GREEN** (not INFLIGHT). DATA_CONTRACTS **candle_order_book_ic_method**.
**spread_over_mid FEATURE_COLS + IC⇒mean + session_ofi_sum IC 4/4 GREEN.** Candle ic_method HAC **6/6** confirmed. DATA_CONTRACTS **spread_over_mid FEATURE_COLS**.
**depth_imbalance_abs FEATURE_COLS + ofi_lag IC 4/4 GREEN.** Sergeant snaps vs n_session **promoted 7/7 GREEN** (not INFLIGHT). ic_method HAC stays **6/6 GREEN**. DATA_CONTRACTS **depth_imbalance_abs FEATURE_COLS**.
**queue/vpin IC + candle structure IC⇒mean 4/4 GREEN** (`test_queue_vpin_ic_and_structure_means.py`). IDLE awaiting Sergeant next free-lane. DATA_CONTRACTS **CoS queue/vpin IC**.
**best_feature IC identity 8/8 GREEN**; **imbalance/CLV/microprice IC packs + ofi/QP/slope means 3/3 GREEN**. queue/vpin + structure IC⇒mean **4/4** already in. DATA_CONTRACTS **Sergeant best_feature IC identity**.
**n_fused/n_bars/n_scored sizing 6/6 GREEN**; **remaining FEATURE_COLS means + amihud/depth/slope/body/vpin IC packs 3/3 GREEN**. best_feature **8/8** already promoted. DATA_CONTRACTS **Sergeant n_fused sizing**.
**p_ic catchall + session_close/sweep/VoR packs 3/3 GREEN** (`test_p_ic_catchall_and_session_close_sweep_packs.py`). DATA_CONTRACTS **CoS p_ic catchall**.
**candle_order_book sizing + depth≥1 soft-verify 8/8 GREEN** (`candle_order_book_sizing_honesty_errors`; ≠ northset n_bars/n_scored helper). p_ic catchall + session_close/sweep packs **3/3** confirmed. DATA_CONTRACTS **Sergeant candle_order_book sizing**.
**IC t/mean/rank + finite_rate/floor catchalls 2/2 GREEN** (`test_ic_finite_rate_floor_catchalls.py`). candle_order_book sizing + depth≥1 **8/8** confirmed. DATA_CONTRACTS **CoS IC t/mean/rank catchalls**.
**receipt bool-flags 6/6 GREEN**; **candle claim + pearson∈[-1,1] + northset string enums 4/4 GREEN**. IC floor catchalls **2/2** confirmed. DATA_CONTRACTS **Sergeant receipt bool-flags**.
**dm_park + sweep_evidence_scope + candle join/chain 4/4 GREEN** (`test_dm_park_sweep_scope_and_candle_join_chain.py`). Bool-flags **6/6** + claim/pearson/enums **4/4** confirmed. DATA_CONTRACTS **CoS dm_park + sweep_evidence_scope**.
**dgp/book_dgp↔data_source 6/6 GREEN**; **all_*_rate unit + sweep evidence blob 3/3 GREEN**. Sizing+depth **8/8** + dm_park/sweep_scope/join-chain **4/4** confirmed. DATA_CONTRACTS **Sergeant dgp / book_dgp**.
**use_session_l2↔gate consistency 6/6 GREEN** (`northset_use_session_l2_gate_consistency_errors`: True⇒enforced, False⇒skipped). dgp/data_source **6/6** + all_*_rate/sweep blob **3/3** confirmed. DATA_CONTRACTS **Sergeant use_session_l2 ↔ gate consistency**.
**all_*_share unit + candle spread alias 3/3 GREEN**; **component_sources 6/6 GREEN**. use_session_l2↔gate **6/6** confirmed. DATA_CONTRACTS **CoS all_*_share** / **Sergeant component_sources**.
**northset depth 6/6 GREEN** (`northset_depth_honesty_errors`: int ≥1; candle family skipped). component_sources **6/6** + share/spread alias **3/3** confirmed. DATA_CONTRACTS **Sergeant northset depth**.
**fraction catchall + queue_imbalance_mean alias 3/3 GREEN**. northset depth **6/6** confirmed. DATA_CONTRACTS **CoS fraction catchall + queue_imbalance_mean alias**.
**price_basis/return_basis 6/6 GREEN**. fraction catchall + queue_imbalance_mean **3/3** + northset depth **6/6** confirmed. DATA_CONTRACTS **Sergeant price_basis / return_basis**.
**candle ic_*_p/t/n_dates catchalls 4/4 GREEN**; **family/book_source 6/6 GREEN**. price_basis/return_basis **6/6** confirmed. DATA_CONTRACTS **CoS candle ic_*_p** / **Sergeant family / book_source**.
**candle family/provenance 5/5 GREEN**; **mean_imbalance_top + shape_columns_ensured rates 3/3 GREEN**. northset family/book_source **6/6** + candle ic catchalls **4/4** confirmed. DATA_CONTRACTS **Sergeant candle family / provenance**.
**candle dgp/data_source 5/5 GREEN**; **metrics_required_finite_ok⇒rates 4/4 GREEN**. candle provenance **5/5** + mean_imbalance_top/shape_columns **3/3** confirmed. DATA_CONTRACTS **Sergeant candle dgp / data_source**.
**sweep_evidence_scope stamp-contract 10/10 GREEN** (enum fixed: `synthetic` / `empirical_adjusted` / `fixture_raw_unadjusted`). candle dgp **5/5** + metrics_required_finite_ok **4/4** confirmed. DATA_CONTRACTS **Sergeant sweep_evidence_scope stamp-contract**.
**depth_imbalance(+abs)+METRICS_REQUIRED finite-when-present 3/3 GREEN**; **shape_ensured↔book_panel_path 5/5 GREEN**; **include_kyle_ofi↔nest 5/5 GREEN**; **research_only⇒claim 10/10 GREEN**. sweep_evidence_scope stamp-contract **10/10** confirmed. DATA_CONTRACTS latest Sergeant/CoS soft-verify wave.
**conservation↔reconstructs never-equate 4/4 GREEN** (H24≠H23); **control_sample_adequate⇒n/p/t 3/3 GREEN**; **impact_estimator_scope stamp-contract 4/4 GREEN** (`per_security_equal_weight`). include_kyle↔nest / shape_ensured↔path / research_only⇒claim confirmed. DATA_CONTRACTS latest soft-verify wave.
**fold_positive rates 2/2 GREEN**; **book_join_coverage_floor 4/4 GREEN**. conservation↔reconstructs **4/4** + impact_estimator_scope **4/4** + control_sample_adequate **3/3** confirmed. DATA_CONTRACTS **CoS fold_positive rates** / **Sergeant book_join_coverage_floor**.
**amihud/qlike/corwin pack 8/8 GREEN**; **session_chain↔siblings never-equate 4/4 GREEN** (H29≠H23/H24). book_join_coverage_floor **4/4** + fold_positive **2/2** confirmed. DATA_CONTRACTS **CoS amihud / qlike / corwin pack**.
**candle mean_*_frac pack 3/3 GREEN**. session_chain↔siblings **4/4** + amihud/qlike/corwin **8/8** confirmed. DATA_CONTRACTS **CoS candle mean_*_frac pack**.
**family/book_source+label nonempty 8/8 GREEN**; **ohlc↔session_ohlc never-equate 4/4 GREEN**; **yang_zhang/overnight/QLIKE pack 5/5 GREEN**. candle mean_*_frac **3/3** confirmed. DATA_CONTRACTS latest soft-verify wave.
**ohlc↔gap_finite never-equate 4/4 GREEN**. ohlc↔session_ohlc **4/4** + label nonempty **8/8** confirmed. DATA_CONTRACTS **Sergeant ohlc ↔ gap_finite never-equate**.
**candle_direction 4/4 GREEN** (`mean_candle_direction` ∈[-1,1]; in FEATURE_COLS). ohlc↔gap_finite never-equate **4/4** confirmed. DATA_CONTRACTS **CoS candle_direction**.
**H20↔H21 never-equate 4/4 GREEN** (`ohlc_identity_rate`≠`book_uncrossed_rate`); **wick_skew+candle_body_ret 3/3 GREEN**. ohlc↔gap_finite **4/4** + candle_direction **4/4** confirmed. DATA_CONTRACTS **Sergeant H20 ↔ H21 never-equate**.
**session_ohlc↔reconstructs never-equate 4/4 GREEN**; **signed_vol_x_imbalance 4/4 GREEN**. H20↔H21 **4/4** + wick_skew+body_ret **3/3** confirmed. DATA_CONTRACTS **Sergeant session_ohlc ↔ reconstructs**.
**H21↔H22 never-equate 4/4 GREEN** (triad H20–H21–H22 complete); **candle ofi/queue means 3/3 GREEN**; **microprice_minus_mid finite 4/4 GREEN**. session_ohlc↔reconstructs **4/4** + H20↔H21 **4/4** + signed_vol **4/4** confirmed. DATA_CONTRACTS **Sergeant H21 ↔ H22 never-equate**.
**spread_bps + log slopes 3/3 GREEN**; **H20↔H22 never-equate 4/4 GREEN** (triad diagonal — H20–H21–H22 complete). DATA_CONTRACTS **CoS spread_bps + log slopes** / **Sergeant H20 ↔ H22**.
**tob/concentration/tick_spacing 3/3 GREEN**; spread_bps+slopes **3/3** + H20↔H22 **4/4** (triad 12/12) confirmed. DATA_CONTRACTS **CoS tob / concentration / tick_spacing**.
**tob_size_share + concentration tops + tick_spacing 3/3 GREEN**. DATA_CONTRACTS **CoS tob_size_share + concentration tops + tick_spacing**.
**notional/queue/MWB 3/3 GREEN**; **gap↔uncrossed never-equate 4/4 GREEN**; **session_bulk_vpin↔siblings 5/5 GREEN**; **candle_dir_x_imbalance + close_mid_abs_rel 3/3 GREEN**. DATA_CONTRACTS latest soft-verify wave.
**CLV alias identity 5/5 GREEN** (`close_location_value_*` ≡ `clv_*` when both stamped). VPIN triad **5/5** + gap↔uncrossed **4/4** + candle_dir_x_imbalance **3/3** confirmed. DATA_CONTRACTS **Sergeant CLV alias identity**.
**session_ohlc↔gap never-equate 4/4 GREEN** (identity #186–#190); **effective/half/ofi 3/3 GREEN**. CLV alias **5/5** confirmed. DATA_CONTRACTS **Commander session_ohlc ↔ gap**.
**impact_proxy_warning 5/5 GREEN** (`depth_or_ofi_proxy_not_signed_trade_flow`); **product stamp 5/5 GREEN** (`product=Northset`). session_ohlc↔gap **4/4** + effective/half/ofi **3/3** confirmed. DATA_CONTRACTS **Sergeant impact_proxy_warning** / **product stamp**.
**session_ohlc↔book_uncrossed never-equate 4/4 GREEN** (identity #191–#195); **finite_rate catchall + price_slope IC⇒mean 3/3 GREEN**. impact_proxy_warning **5/5** + product stamp **5/5** confirmed. DATA_CONTRACTS **Commander session_ohlc ↔ book_uncrossed**.
**session_ohlc↔volume_conservation never-equate 4/4 GREEN**. impact_proxy_warning **5/5** + product stamp **5/5** confirmed. DATA_CONTRACTS **Sergeant session_ohlc ↔ volume_conservation**.
**session_ohlc↔chain 4/4**; **ohlc↔session_reconstructs 4/4**; **gap↔session_chain 4/4** (#196–#200); **IC unit tighten 2/2**; **H20↔H29 ohlc↔session_chain 4/4** (reconnect on-disk). DATA_CONTRACTS catch-up wave.




























































**Sergeant verify dedupe:** single `northset_receipt_honesty_errors` fan-in for the 40; duplicate individual gap/vpin/… northset calls removed; candle_order_book-only wires kept. **CLI rate echoes:** ohlc / session_ohlc / book_uncrossed / chain / reconstructs / volume_conservation / `structure_finite_rate` on `dipcatcher northset`. **CoS mean_* CLI 54/54** complete. DATA_CONTRACTS those sections.

**CoS sweep CLI map:** nested `mean_excess_bps` → `sweep_*_event_mean_bps`; `mean_diff_bps` → `sweep_*_control_diff_mean_bps`. **Candle-book CLI:** echoes `n_fused` + `min_names` (+ CoS `finite_rate_*` stamps). **mean_* 54/54** CLI complete. **Commander #141–#150** Mac property continuous (optional/queue/tob/n_levels/aliases/tick spacing). DATA_CONTRACTS those sections.

**CoS candle finite_rate_*:** stamps microprice_minus_mid + bid/ask size_concentration rates; `structure_finite_rate_honesty` bites on synth. **Sergeant candle-book CLI:** `family=` echo + non-ic parity guard (`test_candle_book_cli_echoes_all_non_ic_receipt_keys`). **Commander #161–#175** Mac property continuous (depth1/bounds/field groups). DATA_CONTRACTS those sections.





## Vendor Alpaca / Polygon remap (offline)

`quant_fund.microstructure.vendor_book_map` remaps **top-of-book quote shapes** onto the Northset book panel. **No network pulls** in this path — live vendor ingest stays gated elsewhere.

Remap honesty: `n_bid/ask_levels=1`, NaN log-size slopes — **no invented ladder**. Do not run `ensure_book_panel_shape_columns` on remapped frames. Floor implications: DATA_CONTRACTS external-panel section.

### Presets

| Preset | Id / time aliases (first match wins) | Bid / ask / size |
|---|---|---|
| `alpaca` | `S`/`symbol`; `t`/`timestamp` | `bp`/`ap`; `bs`/`as` |
| `polygon` | `ticker`/`T`; **event**=`participant_timestamp` then `sip_timestamp`; **available**=`sip_timestamp` (participant-only fails closed) | `bid`/`ask`; `bid_size`/`ask_size` |
| `generic` | broader alias table in `VENDOR_COLUMN_ALIASES` | same targets |

Required direct targets before derive: `security_id`, `event_time`, `available_time`, `best_bid`, `best_ask`, `top_bid_size`, `top_ask_size`. Missing any → `ValueError` (fail closed).

### Derive after top-of-book

From best bid/ask + top sizes: `mid`, `spread`, `spread_bps`, `microprice`, `microprice_minus_mid(_bps)`, `imbalance_top`, `imbalance_depth` (= top when no ladder), `bid_depth`/`ask_depth` (= top sizes). Depth slopes are **NaN**; `n_*_levels=1`. Panel must pass `validate_book_panel`.

### Honesty stamps

| Receipt / column | Meaning |
|---|---|
| `book_source` | Panel `source` (e.g. `alpaca`) |
| `book_dgp` / `dgp` | `vendor_panel:{book_source}` when a panel is used — **not** `synthetic_lob` |
| `data_source` / `label` | Still describe the **bars** feed (often SYNTHETIC) |
| Dry-run blob | `research_only=true`, `claim=research_diagnostic_only` |
| `event_clock` / `available_clock` | `participant`/`sip` when both Polygon clocks exist; `sip`/`sip` if SIP-only; participant-only remap is rejected |

`remap_vendor_bars` applies the same SIP vs participant clocks to OHLCV. Polygon bars: event=`participant_timestamp`, available=`sip_timestamp`; participant-only fails closed. Alpaca bars use `t` for both clocks.

`vendor_panel_from_bars` builds SYNTHETIC L2, disguises columns as Alpaca/Polygon, then remaps — fixture path only; timestamps stay bar-aligned.

```bash
uv run dipcatcher vendor-book-map --vendor alpaca
uv run dipcatcher vendor-book-map --vendor alpaca --columns S,t,bp,ap,bs,as
uv run dipcatcher vendor-book-map --vendor alpaca --parquet tests/fixtures/northset/alpaca_quotes.parquet \
  --out data/book_panel/alpaca_remapped.parquet
uv run dipcatcher northset --book data/book_panel/alpaca_remapped.parquet
uv run dipcatcher northset --book tests/fixtures/northset/alpaca_quotes.parquet --vendor alpaca
```

### External panel ops (CoS)

Full dry-run / remap / `--book` sequence + floor caution: DATA_CONTRACTS **External panel CLI / vendor-book-map dry-run ops**.

- TOB remapped panels: leave `--depth-shape-floor` / `--concentration-floor` (and YAML floors) **unset**
- `book-panel` CLI writes SYNTHETIC multi-level panels (ensure + finite rates) — different path from vendor TOB


### CLI shape-line echo (external)

Successful runs print `depth_shape_finite_rate=… concentration_top_finite_rate=…` (and floor keys only when set). External TOB: expect **nan** rates with floors absent. Floors set + NaN → fail-closed before that line. Details: DATA_CONTRACTS **dipcatcher northset CLI echo**. Dual-col means line: DATA_CONTRACTS **CLI echo: mean_quoted_spread / mean_effective_spread / mean_close_mid_abs_rel**.


### CLI join / book_source echo

Summary line: `join_coverage=… mean_book_age_s=… max_book_age_s=… book_source=… include_kyle_ofi=…`.
**`book_dgp` is receipt-only** (blob dump), not this line. SYNTHETIC → `book_source=synthetic_lob`, coverage ≈1; external → vendor `source`, coverage ≥ join floor. Details: DATA_CONTRACTS **CLI echo: join_coverage / book_source**.


### Doctor: research_receipt vs shape_floors

`research_receipt` (doctor status) gates health via verify of `latest.json`. The
`northset.shape_floors` line is config-only echo and does **not** gate exit.
Details: DATA_CONTRACTS **doctor research_receipt vs shape_floors split**.

**session_imbalance_std / session_book_source:** fuse `session_imbalance_std` ≠ receipt `mean_session_imbalance_std` (CLI echoes mean; not IC; ≥0 soft-verify). Parallel to book_snaps count vs mean. Last-snap `session_book_source` ≠ receipt `book_source`. DATA_CONTRACTS **`session_imbalance_std` vs `mean_session_imbalance_std`** + **session_imbalance_std / session_book_source**; OPS cross-link.

**Spread means never-equate:** `mean_spread_bps` ↔ `2× mean_half_spread_bps`; `mean_half_spread` ↔ `½ mean_quoted_spread`; `mean_effective_spread` ↔ quoted (≠ `mean_close_mid_abs_rel`). Soft-verify honesty helpers in catalog; `verify-research` runs them on northset + candle_order_book — research_only, **never live Sharpe**. CoS CLI echo: DATA_CONTRACTS **Spread receipt means** + **CLI echo: quoted / half…**. **Candle IC:** `wick_skew`→H26; `close_location_value`/`clv_*`→H30; `candle_body_ret` IC-scored, no H-id. DATA_CONTRACTS **Candle geometry IC family** + **H-gap stub** (`candle_body_frac` / wick fracs — no H-ids).

**TOB share honesty (#53/#54/#62):** `tob_size_share` ∈(0,1] vs `tob_notional_share` ∈(0,1] (touch-priced TOB) — never equate (size ≠ notional). DATA_CONTRACTS **tob_size_share vs tob_notional_share**. **verify-research spread honesty:** half/bps/effective helpers on northset + candle_order_book — DATA_CONTRACTS **verify-research ops: northset_spread_*_honesty_errors**; OPS same.

**MWB soft-verify:** `mean_microprice_weight_balance` CLI/receipt + `northset_microprice_weight_balance_honesty_errors` ∈[0,1] on verify-research (northset only). DATA_CONTRACTS **mean_microprice_weight_balance**. **session_imbalance_mean** receipt mean still a CoS gap (IC fuse only). Candle H-gap table expanded: DATA_CONTRACTS **Candle geometry H-gap stub**.

**Queue vs MWB:** `queue_priority_proxy` / `ask_queue_priority_proxy` (side queue position) ≠ `microprice_weight_balance` (cross-top weight); both thin→NaN / ∈[0,1] when finite. DATA_CONTRACTS **queue_priority vs microprice_weight_balance**. **coverage_guarantee_scope:** verify-research soft stub — DATA_CONTRACTS **coverage_guarantee_scope soft-verify ops**; OPS.

**Concentration vs queue priority (Commander #59 lock):** `bid/ask_size_concentration_top` = `top/side_depth`; `queue_priority_proxy` / `ask_queue_priority_proxy` = `top/(top+side_depth)` (same-side, not opposite). Never equate; ∈[0,1] when finite; thin→NaN. Floors: `concentration_top_finite_*` vs `queue_priority_finite_*` are parallel, not interchangeable. DATA_CONTRACTS **bid/ask_size_concentration_top vs queue_priority_proxy**.

**Microprice triad:** `mid` = midpoint; `microprice_weight_balance` `w` = `top_bid/(top_bid+top_ask)` ∈[0,1]; `microprice` `μ` = `ask·w+bid·(1−w)` ∈[bid,ask] (fallback mid if denom≤0). Never equate `μ−mid` with `w` alone (`μ−mid = spread·(w−0.5)`). CLI echoes `mean_microprice_weight_balance`. DATA_CONTRACTS **microprice vs mid vs microprice_weight_balance**.

**imbalance_top vs MWB:** when both finite, `imbalance_top = 2·microprice_weight_balance − 1` (verified in book_metrics). `touch_size_imbalance` aliases imbalance_top. Still ≠ μ−mid. DATA_CONTRACTS **imbalance_top vs microprice_weight_balance**. **Doctor vs CLI floors:** doctor always dumps five config floors; northset shape line always rates, floors only if set — DATA_CONTRACTS **Doctor shape_floors vs CLI shape-line echo**.

**touch_size_imbalance surfaces:** ≡ `imbalance_top` = `2w−1` when finite — never a separate signal. REQUIRED both fuse keys; northset `_IC_FEATURES` and candle `_FEATURE_COLS` score **imbalance_top only** (no dual `ic_touch_size_imbalance*`). No CLI mean. DATA_CONTRACTS **touch_size_imbalance alias surface checklist** + **Candle+LOB `_FEATURE_COLS` inventory**. tob_notional may>1 stays closed.

**Quoted vs dual-column reminder:** book `quoted_spread == effective_spread == spread`; `mean_close_mid_abs_rel` is candle `|close−mid|` only — never equate. DATA_CONTRACTS **Reminder: quoted_spread alias vs dual-column**. **session_imbalance docs:** wait for Lieutenant `mean_session_imbalance_mean` (not CoS).

**Candle dual-IC (closed):** `_FEATURE_COLS` no longer dual-lists touch_size — imbalance_top IC only. MWB soft-verify runs on candle_order_book + northset. DATA_CONTRACTS **Candle dual-IC alias honesty** + **Candle+LOB `_FEATURE_COLS` inventory**. Dual-col quoted/effective stays closed.

**`session_imbalance_std` vs `mean_session_imbalance_std`:** fuse per-day path std ≠ receipt `nanmean`; CLI echoes the mean (not the fuse col); not IC; ≥0 when finite. Parallel to `n_session_book_snaps` vs `mean_session_book_snaps`. DATA_CONTRACTS **`session_imbalance_std` vs `mean_session_imbalance_std`**.
**`vpin_mean` soft-verify:** daily VPIN ∈[0,1] when finite; ≠ `session_book_vpin_mean` (H32 ≠ H43). **`ohlc_identity_rate` vs `gap_finite_rate`:** OHLC envelope rate ≠ overnight gap finite rate. DATA_CONTRACTS **`vpin_mean` soft-verify** + **`ohlc_identity_rate` vs `gap_finite_rate`**.
**`book_uncrossed_rate` / H21:** soft-verify ∈[0,1]; finite + eligible → H21 bound row. **`session_ohlc_identity_rate` vs `ohlc_identity_rate`:** same OHLC identity fn, session candles ≠ daily bars. DATA_CONTRACTS **`book_uncrossed_rate` / H21** + **`session_ohlc_identity_rate` vs `ohlc_identity_rate`**.
**H20/H21/H22 soft-verify triad:** finite `ohlc_identity_rate` → H20 bound; finite `book_uncrossed_rate` (+ eligible) → H21 bound; finite `imbalance_top_p_ic` (+ eligible) → H22 discovery. **`session_reconstructs_daily_rate` vs `session_ohlc_identity_rate`:** reconstruct daily envelope ≠ session-candle OHLC identity. DATA_CONTRACTS **H20/H21/H22 soft-verify triad** + **`session_reconstructs_daily_rate` vs `session_ohlc_identity_rate`**.
**`session_volume_conservation_rate`:** soft-verify ∈[0,1]; H24 bound when finite; ≠ reconstructs / session_ohlc / chain. **H23+ expansion:** `northset_h23_h28_consistency_errors` / `NORTHSET_H23_H28_SPECS` (H23–H49-class finite→H-row). DATA_CONTRACTS **`session_volume_conservation_rate` soft-verify** + **H23+ soft-verify expansion pointer**.
**Sweep follow event vs cost vs reject:** never equate `sweep_follow_event_mean_bps` ↔ `sweep_follow_cost_adjusted_mean_bps` ↔ reject twins; soft-verify finite on each. DATA_CONTRACTS **Sweep follow event vs cost-adjusted vs reject**.
**Kyle nest soft-verify suite:** `kyle_ofi_nest_honesty_errors` fans **26** helpers (Sergeant nest sync; **`test_kyle_ofi` 47/47**). Includes join/residual/dispersion/corr/SYN/claim/`ic_method`/`family`/`hac_lags`/min_names/n_fused/path/deciles/std-iqr/rolling/n_dates/p/spearman/t/label/date_series counts. Not an H-row mint. DATA_CONTRACTS **Kyle nest soft-verify suite** + **do-not-invent backlog**.
**Nest residual_ofi ≠ ofi_flow IC:** `residual_ofi_ex_depth_fwd_*` is within-date ofi residual ex `signed_depth` vs **fwd** targets; `ofi_flow_delta_mid_*` is raw ofi vs contemporaneous `delta_mid`; `ofi_fwd_*` is raw ofi vs fwd targets. Never equate. DATA_CONTRACTS **Nest residual_ofi vs ofi_flow IC**.
**Nest ofi_delta_mid_lag0/lag1 ≠ ofi_flow:** lag1 is prior ofi vs current Δmid; lag0 is a separate helper from `ofi_flow_delta_mid_*` (Kyle λ companion). **Nest kyle_lambda_*_fwd_* ≠** default `kyle_lambda_ofi_mean` / `kyle_lambda_depth_mean` (forward-target OLS λ). DATA_CONTRACTS those sections.
**Nest signed_depth_delta_mid_lag1 ≠ depth_flow:** lagged depth vs lag-0 Kyle companion (ofi lag twin). **Nest ofi_fwd ≠ always-on ofi_p_ic / H27:** kyle-fuse predictive IC ≠ candle-fuse discovery IC. DATA_CONTRACTS those sections.
**Nest signed_depth_fwd ≠ depth_flow:** forward-target depth IC ≠ contemporaneous `depth_flow_delta_mid_*`. **Always-on ofi_lag / ofi_lag1_corr ≠ nest ofi_delta_mid_lag1:** fuse col + panel AR ≠ nest lagged-ofi→Δmid IC. DATA_CONTRACTS those sections.
**Nest depth_flow ≠ kyle_lambda_depth_mean:** Spearman companion IC ≠ OLS λ (ofi twin: ofi_flow ≠ kyle_lambda_ofi_mean). **mid_lag1_corr ≠ ofi_lag1_corr:** same `_panel_lag1` shape, different series. DATA_CONTRACTS those sections.
**Nest Kyle λ dispersion ≠ IC companions:** p10–p90 / rolling HAC band on the λ series ≠ `depth_flow_*` / `ofi_flow_*`. **Always-on kyle_r2 / kyle_ofi_r2 ≠ nest HAC t/p** (no nest R² twin). DATA_CONTRACTS those sections.
**Nest kyle_lambda_ofi_depth corr ≠ dispersion:** cross-flow date-λ correlation ≠ within-flow λ deciles/rolling HAC. **Always-on n_securities ≠ nest n_dates** (names ≠ dates; corr uses aligned intersection). DATA_CONTRACTS those sections.
**Nest join_coverage ≠ always-on join_coverage:** kyle fuse rate ≠ candle asof attach (separate soft-verify + nest requires `book_source`). **Kyle claim/ic_method cheat-sheet:** `research_only` + `claim=research_diagnostic_only` + `ic_method=date_level_spearman_hac`. DATA_CONTRACTS those sections.
**Nest book_source/book_dgp ≠ always-on provenance:** kyle fuse stamps ≠ family evidence matrix (`dgp`/`label`/`data_source`/`component_sources`). **Nest n_fused/n_scored ≠ always-on sizing** (kyle join vs asof / delta_mid vs fwd_ret_1). DATA_CONTRACTS those sections.
**Nest min_names ≠ always-on echo:** nest stamps `min_names`; main receipt usually does not (use config); standalone `kyle-ofi` may hardcode 3. **Nest SYN* ≠ always-on MIXED:** nest label path skips `MIXED_SYNTHETIC_DERIVED`. **Nest hac_lags** optional; no always-on receipt twin. DATA_CONTRACTS those sections.
**Nest book_panel_path ≠ always-on path blindly:** audit echo passed through when nested; same string ≠ same fuse metrics; standalone CLI may differ. Kyle nest never-equate docs **IDLE** pending new keys/productization — DATA_CONTRACTS **Nest book_panel_path** + **Kyle nest docs grind — IDLE**.

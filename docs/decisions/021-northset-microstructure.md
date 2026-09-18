# ADR-021: Northset — order-book and candlestick research inside Dipcatcher

## Status

Accepted

## Date

2026-09-16

## Context

Dipcatcher already scores daily ranking, distribution, conformal coverage, and execution
shortfall. It did not have a first-class research slice for **candlesticks (OHLCV)** and
**L2 order-book snapshots**. A plumbing path existed under `quant_fund.microstructure`
but was optional, unnamed, and outside the required benchmark catalog. Neural nets remain
blocked (ADR-007). PIT fields and SYNTHETIC labeling stay mandatory. Sharpe / live P&L
are not lab headlines.

## Options considered

### Option A: Keep microstructure optional, outside the catalog

- Pros: no catalog version bump; existing receipts stay valid.
- Cons: the candle/book path can rot; `dipcatcher research` never has to run it.

### Option B: Vendor matching-engine replay as the research object

- Pros: closer to live microstructure.
- Cons: no vendor book in this lab; would invent fills; collides with ADR-005 next-bar fills.

### Option C: Named research product **Northset** on PIT bars + labeled SYNTHETIC L2 (chosen)

Northset is Dipcatcher's microstructure slice. Daily OHLCV identities, session candles
that reconstruct the daily envelope and chain, uncrossed L2 snapshots, date-level
imbalance / microprice / wick / OFI / CLV / VPIN ICs, Kyle λ (depth and OFI),
Roll / Corwin–Schultz / Abdi–Ranaldo spreads, Amihud, Parkinson / Garman–Klass /
Rogers–Satchell / Yang–Zhang / overnight-split QLIKE, and session BNS jumps.
Catalog version 2 adds family `northset`. H20/H21/H23/H24/H29 are bound identity
checks; H22/H25–H28/H30–H32 are discovery scores.

- Pros: same honesty contract as the rest of the lab; no vendor feed required; identities
  fail closed.
- Cons: SYNTHETIC L2 is derived from bar OHLC/volume (plumbing, not live edge). Catalog
  receipts from version 1 must be regenerated.

## Decision

Use **Option C**.

- Package: `quant_fund.northset` (re-exports `microstructure` helpers).
- Config: `northset:` on `AppConfig`.
- CLI: `dipcatcher northset` and the `northset` family inside `dipcatcher research`.
- SYNTHETIC LOB / session candles set `source=synthetic`. Research blobs use
  `research_only` and omit `pnl`/`nav` key tokens (ADR dual-catalog).
- Vendor books later must emit the same metric columns and PIT timestamps.

`optimizer.py` and `risk_gate.py` are not edited. Northset does not place orders.

## Consequences

Research receipts bump `benchmark_catalog_version` to 2 and must include `northset`.
OHLC/book identity failures are data-contract bugs, not alpha. Imbalance IC on
SYNTHETIC books is a scientific score, not evidence of live microstructure edge.


## Session L2 and vendor remap (same decision)

- **Session L2** (`northset.use_session_l2`, default true): one SYNTHETIC L2 snapshot per session candle, aggregated to daily path stats. Not vendor RTH/ETH L2 and not a live book.
- **Vendor remap** (`vendor_book_map`): offline Alpaca/Polygon/generic top-of-book → Northset panel; fail closed on incomplete quotes; no network in this module.
- **Candle+LOB IC**: always date-level Spearman/Pearson + HAC via `date_ic_series`; never pooled stacked IC.
- Receipts remain `research_only` / `claim=research_diagnostic_only` with dual-catalog forbidden keys (no `pnl`/`nav`/`live_pnl_claim` as research headlines).

## Integrity gates (post-acceptance)

Fail-closed helpers that must stay documented with the Northset decision:

- `validate_session_book_counts` — session L2 cardinality per parent day
- `attach_candle_book_features` `join_coverage` / `min_join_coverage` — asof PIT fuse
- `validate_book_panel_depth_honesty` — thin→NaN / deep→finite slopes
- `northset.kyle_ofi` — Cont OFI + date-level Kyle λ / OFI→Δmid IC, `research_only` receipts

- `canonical_northset_bars` / `NorthsetMarketView` — corp-action-safe split/TR view; `require_adjusted_ohlc` fail-closed (fixture opt-out only).

- `DEPTH_SHAPE_FIELDS` / Commander residuals #2–#3 — log-price slopes + mean log tick spacing; NaN-if-thin; panel optional + depth-honesty validator.
- `bench_northset` receipt stamps `price_basis` / `return_basis` from `canonical_northset_bars`.
- `enforce_session_l2_identity_floors` + eligibility stamps (`book_hypothesis_eligible`, `session_book_hypothesis_eligible`, `sweep_evidence_scope`, `component_sources`) — research receipts must expose SYNTHETIC vs empirical scope.
- Sweep evidence: `liquidity_sweep_frame` + `sweep_evidence_battery` (event_studies / volatility_regimes / permutation_placebos, BH-FDR, cost hurdle); H33/H34 remain date-level IC discovery scores.
- Sweep battery flatten + bounds: horizon-1 event/placebo keys on the Northset receipt; H39/H40 cost hurdle and H41/H42 fold-stability (`sweep_min_fold_positive_fraction`) are bound checks, not FDR discoveries.
- Optional `northset.include_kyle_ofi` (default false): nest `bench_kyle_ofi_fused` under Northset receipt[`kyle_ofi`]; catalog v2 still requires family `northset` only — kyle_ofi is not a separate required family. Standalone CLI: `dipcatcher kyle-ofi`.
- Disambiguation: always-on `kyle_lambda` / `kyle_ofi_lambda` are per-name OLS means (`_panel_kyle`); nested `kyle_ofi` uses date-level λ + HAC / date IC — not interchangeable.
- Ops: `session_l2_identity_floor` documented in OPERATIONS_RUNBOOK.
- Fuse alias: `signed_volume` (always-on `_panel_kyle`) ≡ `signed_depth` (kyle_ofi nest) = `bid_depth - ask_depth`; documented in DATA_CONTRACTS + MATH_SPEC.
- Identity floor CLI: enforced on `northset` + `session-book`; not on `doctor` / `verify-research` live path.
- Fuse column side-by-side (always-on vs kyle_ofi) + verify-research H20–H43 finite→H-row catalog in DATA_CONTRACTS; NORTHSET catalog lists H35–H38/H43.
- Eligibility stamps end-to-end: `book_hypothesis_eligible` / `session_book_hypothesis_eligible` setter + agent/verify skip lists in DATA_CONTRACTS.
- Stance: nested `kyle_ofi` must **not** mint soft-verify H-rows until explicitly productized.
- Evidence label / component_sources / sweep_evidence_scope matrix + price_basis/return_basis stamp matrix documented in DATA_CONTRACTS (SYNTHETIC vs MIXED vs empirical; forbidden claims).
- Family `dgp` / receipt `book_dgp` (`synthetic_lob` / `vendor_panel:*` / `mixed_sources` / `empirical`) + fuse-frame `external_panel` documented; join_coverage / book_age fail-closed ops in DATA_CONTRACTS + OPERATIONS_RUNBOOK.
- Depth honesty ops checklist: DEPTH_SHAPE + SIDE_STRUCTURE; optional depth_shape/concentration floors; session L2 clocks vs book_max_age documented.
- External book_panel_path: ensure_book_panel_shape_columns is SYNTHETIC-only; vendor TOB n_levels=1 → NaN depth_shape rates; floors fail-closed only when set; never invent shape cols. H33–H42 sweep flatten map in DATA_CONTRACTS.
- External panel CLI / vendor-book-map dry-run ops documented; sweep_evidence.volatility_regimes schema noted (no H-ids yet).
- Stance: `sweep_evidence.volatility_regimes` must **not** mint soft-verify H-rows until productized (same bar as nested kyle_ofi). Northset CLI shape-line echo documented for external TOB NaN / floor fail-closed.
- Northset CLI join_coverage/book_source echo (book_dgp receipt-only) + sweep_evidence.event_studies full row schema documented.
- queue_priority / side_notional finite rates + floors on CLI shape/doctor; SYNTHETIC ensure includes QUEUE/NOTIONAL; vendor TOB → NaN unless floors unset. permutation_placebos nested schema documented.
- Commander #53/#54 (legacy #34/#35) + **#62**: `tob_size_share` ∈(0,1] vs `tob_notional_share` ∈(0,1] (touch-priced TOB; retired may>1). Never equate size vs notional. Doctor research_receipt vs shape_floors split documented.
- Notional proxies (TOB / side / imbalance) + mean_tob_size_share (northset) vs tob_size_share_finite_rate (candle-book bench) documented.
- Spread/imbalance aliases (half_spread, quoted_spread_bps, touch_size_imbalance, …) + microprice_weight_balance.
- Honesty **FIXED**: dual columns — book `effective_spread` preserved; candle `close_mid_abs_rel`; receipts `mean_effective_spread` + `mean_close_mid_abs_rel`. Also `depth_imbalance_abs` / `spread_over_mid`.
- Receipt disambiguation (post-fix): `mean_spread_bps`/`mean_quoted_spread`/`mean_effective_spread` = quoted/book alias; `mean_close_mid_abs_rel` = candle `2|close−mid|/mid`.
- Touch means (`mean_quoted_spread`/`mean_spread_bps`) vs OHLC/mid estimators (`roll_spread`/`corwin_schultz_spread`/`abdi_ranaldo_spread`) documented as non-interchangeable; `queue_imbalance_mean` vs OFI flow keys likewise.
- `session_ofi_sum`/`session_ofi_sum_mean` vs always-on `ofi`/`ofi_lag` documented (not interchangeable); Amihud / volume_over_range / true_range receipt family noted.
- CLI echo documents dual-col means (`mean_effective_spread` + `mean_close_mid_abs_rel`) on `dipcatcher northset` / research northset line.
- Session OFI (synthetic_lob aggregate) vs bar `ofi`/`queue_imbalance`: validity, fail-closed, receipt names. Illiquidity means (`amihud_mean`, …) documented as research_only — no live Sharpe.
- `session_book_vpin_mean` vs `vpin_mean`/`vpin_proxy`: H43 vs H32, eligibility, CLI vs blob; `session_close_*` pointer; `session_bulk_vpin` noted as third scalar.
- Expanded `session_close_*` daily aggregates (last snap vs path means vs session-book VPIN) with DATA_CONTRACTS/NORTHSET/OPS cross-links; VPIN never-equate checklist tightened.
- `session_imbalance_std` / `session_book_source` honesty + session_* IC vs receipt-only list documented (DATA_CONTRACTS/NORTHSET/OPS).
- CLI echo matrix updated: `mean_spread_bps` / half-spread means / `mean_microprice_weight_balance` now on northset + research lines (docs match live CLI).
- Spread receipt means never-equate matrix + soft-verify identities; CoS CLI echo note. Candle geometry IC family: wick H26, CLV H30, candle_body_ret IC-only.
- Half/quoted/bps identity notes + verify-research soft path documented (research_only, never live Sharpe); candle IC scored vs receipt-only summary.
- Cross-link: DATA_CONTRACTS / NORTHSET / OPS **tob_size_share vs tob_notional_share** (#53/#54).
- `mean_microprice_weight_balance` CLI/receipt + soft-verify ∈[0,1] (`northset_microprice_weight_balance_honesty_errors`); candle H-gap table (body_frac/wick fracs); session_imbalance_mean receipt gap noted (CoS).
- queue_priority_proxy / ask_queue_priority_proxy vs microprice_weight_balance (never equate; ∈[0,1]; thin→NaN). coverage_guarantee_scope soft-verify ops stub (marginal_exchangeable; research_only).
- bid/ask_size_concentration_top vs queue_priority_proxy / ask_queue_priority_proxy (top/side_depth vs top/(top+side_depth); never equate; ∈[0,1]; thin→NaN; same-side not opposite). queue_priority_finite_rate floors ops refresh (parallel to concentration_top_finite_*).
- Commander #59 locks size_concentration_top vs queue_priority_proxy formulas in book_metrics (docs align).
- Shape/structure floors matrix: DEPTH_SHAPE vs SIDE_STRUCTURE / QUEUE_STRUCTURE / SIDE_NOTIONAL / TOB_SHARE (rates+floors; never mix shape slopes with concentration/queue; synth ensure vs external unset).
- microprice vs mid vs microprice_weight_balance triad refresh (μ ∈[bid,ask]; mid midpoint; w=top_bid/(top_bid+top_ask); never equate μ−mid with w; CLI mean_microprice_weight_balance).
- imbalance_top vs microprice_weight_balance: locked identity imbalance_top = 2w−1 when both finite (book_metrics); touch_size_imbalance alias. Doctor shape_floors vs CLI shape-line echo parity.
- Commander #62: top_of_book_notional_proxy = best_bid·top_bid+best_ask·top_ask; tob_notional_share ∈(0,1] with touch-priced TOB; docs retire “may exceed 1”.
- touch_size_imbalance alias surface checklist (≡ imbalance_top; =2w−1 when finite). #62 tob honesty confirm: may>1 retired; both shares ∈(0,1]; never-equate kept.
- touch_size_imbalance: REQUIRED fuse alias; candle+northset IC imbalance_top-only (no dual IC); ≡ 2w−1; never separate signal.
- Reminder: quoted_spread == effective_spread == spread (book); mean_close_mid_abs_rel candle-only — never equate. session_imbalance docs wait on Lieutenant mean_session_imbalance_mean (not CoS).
- Candle dual-IC alias honesty: FEATURE_COLS lists imbalance_top + touch_size_imbalance (redundant IC). Quoted vs dual-column reminder confirmed (quoted==effective==spread; close_mid candle-only).
- Candle dual-IC verify-research ops note: no soft-verify equality helper yet; redundant IC not independent alpha; research_only never live Sharpe.
- Stance: `sweep_evidence.volatility_regimes` must **not** mint soft-verify H-rows until productized (same bar as nested kyle_ofi). Northset CLI shape-line echo documented for external TOB NaN / floor fail-closed.
- session_imbalance_std (fuse per-day path std) ≠ mean_session_imbalance_std (receipt nanmean); CLI echoes mean; not IC; ≥0 soft-verify; parallel to n_session_book_snaps vs mean_session_book_snaps.
- vpin_mean soft-verify ∈[0,1] (daily; H32) ≠ session_book_vpin_mean (H43). ohlc_identity_rate vs gap_finite_rate never-equate (OHLC envelope ≠ overnight gap finiteness).
- book_uncrossed_rate soft-verify ∈[0,1] (H21 companion); session_ohlc_identity_rate vs ohlc_identity_rate never-equate (same identity fn; session candles ≠ daily bars).
- H20/H21/H22 soft-verify triad: finite ohlc_identity_rate→H20; book_uncrossed_rate(+eligible)→H21; imbalance_top_p_ic(+eligible)→H22. session_reconstructs_daily_rate ≠ session_ohlc_identity_rate (daily envelope match ≠ session-candle OHLC identity).
- session_volume_conservation_rate soft-verify ∈[0,1] (H24 companion); ≠ session_reconstructs_daily_rate / session_ohlc_identity_rate / session_chain_rate. H23+ expansion pointer: NORTHSET_H23_H28_SPECS / northset_h23_h28_consistency_errors (finite metric → H-row; H23–H49-class).
- Sweep follow event vs cost-adjusted vs reject never-equate: pre-cost event mean_bps ≠ post-cost adjusted; follow ≠ reject; soft-verify finite helpers live (H35/H36/H39/H40 gates separate).
- Kyle nest soft-verify suite live (kyle_ofi_nest_honesty_errors): join_coverage, residual_flow, dispersion, ofi_depth_corr, SYNTHETIC/SYN* label, claim umbrella, ic_method=date_level_spearman_hac, family=kyle_ofi, hac_lags; still no kyle H-row mint.
- Never-equate refresh: always-on `kyle_ofi_lambda` (name-mean OLS) ≠ nest `kyle_lambda_ofi_mean` (date λ + HAC); `n_securities` ≠ `n_dates`; dump `--dump-lambda-series` aligns with `kyle_lambda_date_series_honesty_errors`.
- Nest residual_ofi (ex depth, fwd targets) never-equate raw ofi_flow (contemporaneous) or ofi_fwd (raw predictive); fuse map notes residual IC consumers.
- Nest ofi_delta_mid_lag0/lag1 never-equate ofi_flow companion IC; kyle_lambda_*_fwd_* never-equate contemporaneous kyle_lambda_*_mean.
- Nest signed_depth_delta_mid_lag1 never-equate depth_flow companion; nest ofi_fwd never-equate always-on ofi_p_ic / H27.
- Nest signed_depth_fwd never-equate depth_flow companion; always-on ofi_lag1_corr / ofi_lag IC never-equate nest ofi_delta_mid_lag1.
- Nest depth_flow companion IC never-equate kyle_lambda_depth_mean (ofi twin); always-on mid_lag1_corr never-equate ofi_lag1_corr.
- Nest kyle_lambda dispersion/rolling HAC never-equate flow IC companions; always-on kyle_r2 never-equate nest HAC t/p.
- Nest kyle_lambda_ofi_depth corr never-equate per-flow dispersion; always-on n_securities never-equate nest n_dates (aligned corr n is intersection).
- Nest join_coverage never-equate always-on join_coverage (separate fuse); kyle claim/ic_method ops cheat-sheet documented.
- Nest book_source/book_dgp/label matrix never-equate always-on provenance; nest n_fused/n_scored never-equate always-on sizing.
- Nest min_names echo never-equate always-on (config-only / CLI 3); nest SYN* label never-equate always-on MIXED_SYNTHETIC_DERIVED; nest hac_lags never-equate absent always-on twin.
- Nest book_panel_path audit echo never-equate always-on path blindly; Kyle nest never-equate docs marked IDLE pending new keys/productization.
- Kyle nest do-not-invent backlog parked (5 triggers); soft-verify suite docs synced to 26 helpers / test_kyle_ofi 47/47; CoS overnight receipt means inventory in NORTHSET.
- Free-lane candle+LOB docs: dual-IC FEATURE_COLS listing closed; MWB soft-verify northset+candle parity; FEATURE_COLS inventory incl. price-slope/tick ICs + CLI candle-book.
- Documented CoS NORTHSET_RECEIPT_HONESTY_HELPERS (40) + northset_receipt_honesty_errors at catalog EOF (helpers-before-tuple); Sergeant mean_tob_notional_share verify wire + receipt-stamp tests; kyle nest remains 26-fan separate lane.
- Sergeant wired northset_receipt_honesty_errors into verify-research (40; ~28 new via fan-in); docs note candle structure_finite_rate_* stamp gap; Lt SKIPPED Commander #73–#100 Mac merge (Mac through #175; box behind).
- Sergeant verify dedupe to single northset receipt fan-in (candle wires kept); CLI rate echoes for identity/chain/volume/structure_finite_rate; CoS mean_* CLI 54/54.
- CoS: mean_excess_bps/mean_diff_bps map to sweep_*_event/control_diff CLI echoes; candle-book n_fused+min_names (+ finite_rate_* stamps); mean_* CLI 54/54; Commander #141–#150 Mac property continuous noted.
- CoS candle-book finite_rate_* stamps (structure_finite_rate_honesty bites); Sergeant candle-book family= CLI + non-ic parity guard; Commander #161–#175 noted.
- Lt SKIPPED Commander box #73–#100 → Mac merge (Mac ahead through #175); confirmed CoS candle finite_rate_* + Sergeant family= parity; Commander #161–#175 noted.
- Stamp-test wave: CoS mean_bid/ask_depth 3/3; Sergeant identity rates 3/3; slope/tick stamp tests on Mac; backlog CoS notional/spread_over_mid + Lt half/quoted/effective; box #73–#100 merge remains SKIPPED.
- Stamp-test wave inventory: CoS depth + spread_over_mid/tob/side_notional 3/3; Sergeant identity+slope/tick 3/3; Sergeant n_levels/imbalance/age/close_mid INFLIGHT; Lt half/quoted/effective in flight.
- Stamp-test wave refresh: CoS MWB+depth_abs GREEN; CoS notional/spread_over_mid+side proxies GREEN; Sergeant levels/imbalance/age/close_mid GREEN; Sergeant fwd_ret_after_* INFLIGHT; Lt half-spread still landing (Mac auth flap).
- Stamp-test inventory hard-refresh: Sergeant levels + fwd_ret_after GREEN 3/3; Lt test_northset_spread_means_receipt_stamp.py GREEN 3/3; CoS MWB+depth_imbalance(+abs) GREEN; Sergeant finite_rate stamp tests INFLIGHT.
- Stamp-test inventory re-verified on Mac: GREEN fwd_ret_after 3/3, Lt spread means 3/3, CoS session_bv/rv/jump+sweep excess 5/5; INFLIGHT Sergeant finite_rate, CoS mean_session_* batch, Lt top_size/true_range/mean_spread_bps (files may exist — Lt assign authority).
- Stamp-test correction: Sergeant finite_rates 3/3 GREEN; CoS session_bv/rv/jump+sweep + not-receipt-keys + size_concentration GREEN; Lt top_size/TR/spread_bps 3/3 GREEN; INFLIGHT only structure_finite_rate real stamp from finite_rate_* companions + CoS mean_session_* batch.
- Stamp-test wave mean_* COMPLETE (0 remaining): finite_rates, top_size/tr/spread_bps, session means batch, structure_finite_rate real stamp candle≠northset; CoS _nanmean asarray fix; Lt rule: pytest-green file → GREEN.
- Stamp-test mean_* COMPLETE hard-correct: structure_finite_rate real+derivation 6/6 GREEN (NOT inflight); CoS session means batch 4/4 (11 keys) + _nanmean asarray; true INFLIGHT only Sergeant soft-verify structure_finite_rate on northset / CLI polish.
- Post–stamp-wave: Sergeant never-equate structure surfaces 4/4; Lt candle FEATURE_COLS +5 structure LOB 2/2; CoS microprice/notional/tob stamps + structure_finite distinct soft-verify.
- Soft-verify structure_finite_rate dual-family GREEN 11/11; never-equate surfaces 4/4; Lt FEATURE_COLS +5 noted; True INFLIGHT now Sergeant CLI required-key frozenset + BLOB_ONLY allowlist.
- Northset CLI echo partition GREEN: REQUIRED 51 + BLOB_ONLY 73 disjoint; full non-IC CLI parity wrong for northset blob dump; CoS candle_feature_cols_ic soft-verify; structure soft-verify stays 11/11 GREEN.
- Promote CLI frozenset GREEN 4/4 + REQUIRED finite-on-synth 3/3 (NOT INFLIGHT); CoS queue_priority stamp upgrade; FEATURE_COLS IC presence 5/5; IDLE awaiting (no honesty residual INFLIGHT).
- CLASSIFIED 183 = REQUIRED∪EXTRA∪BLOB_ONLY + classification soft-verify fail-closed; candle join_coverage verify wire; CoS FEATURE_COLS IC completeness + mid_lag1_corr stamp.
- Classify-or-echo GREEN 11/11; EXTRA finite-on-synth 3/3 GREEN; CoS FEATURE_COLS +ask_queue_priority_proxy +microprice_weight_balance + bid/ask queue pair honesty.
- CoS FEATURE_COLS Spearman/Pearson pair completeness + lag1 n_securities soft-verify 12/12; ask_queue+MWB FEATURE_COLS; EXTRA finite-on-synth 3/3 noted.
- Sergeant BLOB_ONLY∉CLI leak lock + book_age fuse honesty 5/5; CoS Spearman/Pearson+lag1 12/12 confirmed; INFLIGHT session L2 / max_book_age / claim+floors.
- CoS candle MWB fuse⇒mean unit + overnight/rv/semi + notional IC 6/6; Sergeant session L2 identity + candle book_age polish 9/9 (promotes prior INFLIGHT).
- candle_order_book_ic_method_honesty_errors GREEN 6/6; CoS session_l2_enforced + notional IC⇒mean 5/5; session L2+candle book_age stays 9/9 GREEN (not INFLIGHT); True INFLIGHT session_book_snaps/uncrossed/HAC meta.
- CoS spread_over_mid FEATURE_COLS + IC⇒mean + session_ofi_sum IC honesty 4/4; candle ic_method HAC 6/6 confirmed; INFLIGHT session_book_snaps vs n_session.
- CoS depth_imbalance_abs FEATURE_COLS + ofi_lag IC honesty 4/4; Sergeant session_book_snaps vs n_session promoted GREEN 7/7; ic_method HAC stays 6/6 GREEN not INFLIGHT.
- CoS queue_imbalance/vpin IC + candle structure IC⇒mean 4/4 (test_queue_vpin_ic_and_structure_means.py); IDLE awaiting Sergeant free-lane.
- Sergeant best_feature IC identity GREEN 8/8; CoS imbalance/CLV/microprice IC packs + ofi/queue/slope means 3/3; queue/vpin structure IC⇒mean 4/4 already in.
- Sergeant n_fused/n_bars/n_scored sizing GREEN 6/6; CoS remaining FEATURE_COLS mean_* + amihud/depth/slope/body/vpin IC packs 3/3; best_feature 8/8 already promoted.
- CoS p_ic catchall + session_close/sweep/volume_over_range IC packs 3/3 (test_p_ic_catchall_and_session_close_sweep_packs.py).
- Sergeant candle_order_book sizing soft-verify GREEN 6/6 (≠ northset n_bars/n_scored); CoS p_ic catchall + session_close/sweep packs 3/3 confirmed.
- CoS IC t/mean/rank + finite_rate/floor catchalls GREEN 2/2 (test_ic_finite_rate_floor_catchalls.py); candle_order_book sizing 6/6 confirmed.
- Sergeant receipt bool-flags GREEN 6/6; CoS candle claim + pearson∈[-1,1] + northset string enums 4/4; IC floor catchalls 2/2 confirmed.
- Sergeant candle sizing + depth≥1 GREEN 8/8; CoS dm_park + sweep_evidence_scope + candle join/chain 4/4; bool-flags 6/6 + claim/pearson/enums 4/4 confirmed.
- Sergeant dgp/book_dgp↔data_source GREEN 6/6; CoS all_*_rate unit + sweep evidence blob 3/3; sizing+depth 8/8 + dm_park/sweep_scope/join-chain 4/4 confirmed.
- Sergeant use_session_l2↔gate consistency GREEN 6/6; dgp/data_source 6/6 + all_*_rate/sweep blob 3/3 confirmed.
- CoS all_*_share unit + candle spread alias GREEN 3/3; Sergeant component_sources GREEN 6/6; use_session_l2↔gate 6/6 confirmed.
- Sergeant northset depth GREEN 6/6; component_sources 6/6 + all_*_share/candle spread alias 3/3 confirmed.
- CoS fraction catchall + queue_imbalance_mean alias GREEN 3/3; Sergeant northset depth 6/6 confirmed.
- Sergeant price_basis/return_basis GREEN 6/6; fraction catchall + queue_imbalance_mean 3/3 + northset depth 6/6 confirmed.
- CoS candle ic_*_p/t/n_dates catchalls GREEN 4/4; Sergeant family/book_source GREEN 6/6; price_basis/return_basis 6/6 confirmed.
- Sergeant candle family/provenance GREEN 5/5; CoS mean_imbalance_top + shape_columns_ensured rates GREEN 3/3; northset family/book_source 6/6 + candle ic catchalls 4/4 confirmed.
- Sergeant candle dgp/data_source GREEN 5/5; CoS metrics_required_finite_ok⇒rates GREEN 4/4; candle provenance 5/5 + mean_imbalance_top/shape_columns 3/3 confirmed.
- Sergeant sweep_evidence_scope stamp-contract GREEN 10/10 (enum: synthetic/empirical_adjusted/fixture_raw_unadjusted); candle dgp 5/5 + metrics_required_finite_ok 4/4 confirmed.
- CoS depth_imbalance(+abs)+METRICS_REQUIRED finite-when-present GREEN 3/3; Sergeant shape_ensured↔book_panel_path GREEN 5/5; include_kyle_ofi↔nest GREEN 5/5; CoS research_only⇒claim GREEN 10/10; sweep_evidence_scope stamp-contract 10/10 confirmed.
- Sergeant conservation↔reconstructs never-equate GREEN 4/4; CoS control_sample_adequate⇒n/p/t GREEN 3/3; Sergeant impact_estimator_scope stamp-contract GREEN 4/4; include_kyle↔nest + shape_ensured↔path + research_only⇒claim confirmed.
- CoS fold_positive rates GREEN 2/2; Sergeant book_join_coverage_floor GREEN 4/4; conservation↔reconstructs 4/4 + impact_estimator_scope 4/4 + control_sample_adequate 3/3 confirmed.
- CoS amihud/qlike/corwin pack GREEN 8/8; Sergeant session_chain↔siblings never-equate GREEN 4/4; book_join_coverage_floor 4/4 + fold_positive 2/2 confirmed.
- CoS candle mean_*_frac pack GREEN 3/3; session_chain↔siblings 4/4 + amihud/qlike/corwin 8/8 confirmed.
- Sergeant family/book_source+label nonempty GREEN 8/8; ohlc↔session_ohlc never-equate GREEN 4/4; CoS yang_zhang/overnight/QLIKE GREEN 5/5; candle mean_*_frac 3/3 confirmed.
- Sergeant ohlc↔gap_finite never-equate GREEN 4/4; ohlc↔session_ohlc 4/4 + label nonempty 8/8 confirmed.
- CoS candle_direction GREEN 4/4; Sergeant ohlc↔gap_finite never-equate 4/4 confirmed.
- Sergeant H20↔H21 never-equate GREEN 4/4; CoS wick_skew+candle_body_ret GREEN 3/3; ohlc↔gap_finite 4/4 + candle_direction 4/4 confirmed.
- Sergeant session_ohlc↔reconstructs never-equate GREEN 4/4; CoS signed_vol_x_imbalance GREEN 4/4; H20↔H21 4/4 + wick_skew+body_ret 3/3 confirmed.
- Sergeant H21↔H22 never-equate GREEN 4/4 (H20–H21–H22 triad complete); CoS candle ofi/queue means GREEN 3/3; CoS microprice_minus_mid finite GREEN 4/4; session_ohlc↔reconstructs 4/4 + H20↔H21 4/4 + signed_vol 4/4 confirmed.
- CoS spread_bps + log slopes GREEN 3/3; Sergeant H20↔H22 never-equate GREEN 4/4 (H20–H21–H22 triad complete).
- CoS tob/concentration/tick_spacing GREEN 3/3; spread_bps+slopes 3/3 + H20↔H22 4/4 confirmed (triad never-equate 12/12).
- CoS tob_size_share + concentration tops + tick_spacing GREEN 3/3.
- CoS notional/queue/MWB GREEN 3/3; Commander gap↔uncrossed GREEN 4/4; Sergeant session_bulk_vpin↔siblings GREEN 5/5; CoS candle_dir_x_imbalance + close_mid_abs_rel GREEN 3/3.
- Sergeant CLV alias identity GREEN 5/5; VPIN triad never-equate 5/5 + gap↔uncrossed 4/4 + candle_dir_x_imbalance 3/3 confirmed.
- Commander session_ohlc↔gap never-equate GREEN 4/4 (identity #186–#190); CoS effective/half/ofi GREEN 3/3; CLV alias 5/5 confirmed.
- Sergeant impact_proxy_warning GREEN 5/5; Sergeant product stamp GREEN 5/5; session_ohlc↔gap 4/4 + effective/half/ofi 3/3 confirmed.
- Commander session_ohlc↔book_uncrossed never-equate GREEN 4/4 (identity #191–#195); CoS finite_rate catchall + price_slope IC⇒mean GREEN 3/3; impact_proxy_warning 5/5 + product stamp 5/5 confirmed.
- Sergeant session_ohlc↔volume_conservation never-equate GREEN 4/4; impact_proxy_warning 5/5 + product stamp 5/5 confirmed.
- Catch-up: session_ohlc↔chain 4/4; ohlc↔session_reconstructs 4/4; gap↔session_chain 4/4 (#196–#200); IC unit tighten 2/2; H20↔H29 ohlc↔session_chain 4/4 reconnect.
- Sweep inference index `calendar_including_idle_zeros` (HAC/bootstrap/folds + matched-control HAC); invalid OHLC quarantined from PIT rolling extremes; Yang–Zhang QLIKE is per-security expanding OOS (not pooled constant); VPIN honesty stamp `count_window_bulk_ofi_proxy`; CLASSIFIED 206 = 51+65+90.
- H50 two-way clustered t (Cameron–Gelbach–Miller) on event-level signed excess; Polygon remap uses participant time for `event_time` and SIP time for `available_time` (participant-only fails closed); `vpin_proxy(bucket_volume=)` is a real volume clock.
- H51 untradeable overnight gap (close→next-open) on the idle-zero calendar; two-way wild-cluster bootstrap companion; Corwin–Schultz pairs are PIT `(t-1,t)`; `remap_vendor_bars` applies the same SIP/participant clocks to OHLCV.

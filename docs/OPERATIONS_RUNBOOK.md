# Dipcatcher operations runbook

Dipcatcher is Artificial Hedge's proprietary research lab. This runbook covers
the supported research and simulated paper/shadow workflows. It does not
authorize live trading.

## Pre-run checks

Run from the repository root with the locked environment:

```bash
uv sync --frozen --all-groups
uv run dipcatcher doctor --config configs/research.yaml
uv run dipcatcher verify-research
```

Do not proceed when `data_manifest` or `research_receipt` is not `ok`,
`live_allowed` is true in an unexpected profile, or research verification reports errors.

## Reproducible research

Use the configured source and retain the generated receipt:

```bash
uv run dipcatcher research --config configs/research.yaml
uv run dipcatcher verify-research
```

The receipt binds configuration, dataset content, Git/worktree state, runtime
packages, point-in-time status, and the versioned 23-family benchmark catalog.
Synthetic output is diagnostic evidence only; it is not a live-performance
claim.

## Paper and shadow operation

Run the simulated broker with the paper profile:

```bash
uv run dipcatcher paper --config configs/paper.yaml --max-steps 2
```

Before reviewing any result, validate the ledger and inspect the promotion
receipt. Required invariants are `research_only: true`, `live_pnl_claim: false`,
and `would_promote_live: false`. A kill-switch halt or ledger validation error
is an incident requiring investigation, not a performance result.

## Incident containment

1. Stop new paper orders with the configured kill switch.
2. Preserve the complete run directory and immutable receipts.
3. Record the run ID, configuration hash, dataset manifest, and failure output.
4. Re-run `dipcatcher doctor` and `dipcatcher verify-research` before resuming.
5. Do not alter a receipt to make a gate pass; create a new run after the cause
   is corrected.


## Northset session-L2 identity floor

When running Northset with session multi-snapshot L2 enabled (`northset.use_session_l2`,
default **true**), `bench_northset` enforces `enforce_session_l2_identity_floors` against
config `northset.session_l2_identity_floor` (default **0.99**, must be in \([0,1]\)).

Rates gated: daily OHLC identity, session OHLC identity, session→daily reconstruct,
session volume conservation, session chain, book uncrossed. Any NaN or rate below the
floor → fail-closed `ValueError` (data-contract bug, not alpha).

Ops notes:

- Disable session L2 with `northset.use_session_l2: false` (receipt `session_l2_identity_gate=skipped`).
- Tighten/loosen only via `session_l2_identity_floor`; do not edit receipts to pass.
- After a gate failure: preserve the run dir, fix bars/session/book inputs, re-run `dipcatcher northset` and/or `dipcatcher session-book` (both enforce the floor). `dipcatcher doctor` and `dipcatcher verify-research` do **not** re-run the live floor (see DATA_CONTRACTS CLI map).
- Optional Kyle nest: `northset.include_kyle_ofi: true` adds `receipt["kyle_ofi"]`; see NORTHSET
  disambiguation vs always-on `kyle_*` scalars. Not a live path.

```bash
uv run dipcatcher northset --config configs/research.yaml
```




## External book panel (CoS dry-run)

```bash
uv run dipcatcher vendor-book-map --vendor alpaca --parquet quotes.parquet \
  --out data/book_panel/remapped.parquet
uv run dipcatcher northset --config configs/research.yaml \
  --book data/book_panel/remapped.parquet
```

Leave depth/concentration floors unset for top-of-book remaps (NaN rates). Full ops:
DATA_CONTRACTS **External panel CLI / vendor-book-map dry-run ops**.

On a successful `dipcatcher northset` run, look for the shape echo line `depth_shape_finite_rate=… concentration_top_finite_rate=…`. External TOB: **nan** is expected when floors are unset. Setting `--depth-shape-floor` / `--concentration-floor` with NaN rates fail-closes inside the bench (no clean echo).

## Northset depth-shape / concentration honesty

Checklist (thin→NaN shape, deep→finite size/price slopes, tick spacing may NaN on zero-gap,
concentration finite when side depth > 0): DATA_CONTRACTS **Depth honesty ops checklist**.

Optional fail-closed floors (default off): `depth_shape_finite_floor`, `concentration_top_finite_floor`, `queue_priority_finite_floor`, `side_notional_finite_floor`, `tob_size_share_finite_floor`. Never mix DEPTH_SHAPE with SIDE/QUEUE/NOTIONAL/TOB. Matrix: DATA_CONTRACTS **Shape / structure floors matrix**. Receipt rates on CLI shape line + doctor `northset.shape_floors` echo.

External `book_panel_path` / vendor TOB: leave floors unset; rates often NaN; **never invent** shape cols (`ensure_book_panel_shape_columns` is SYNTHETIC-only).

Session L2 daily aggregates do **not** use `book_max_age_seconds` (exact parent join);
asof age only gates the daily candle↔primary book fuse.

## Northset book join coverage / age

Candle↔book fuse (`attach_candle_book_features`) fail-closes on empty join, coverage below
floor, PIT lookahead (`book_age_seconds < 0`), or age above `northset.book_max_age_seconds`
(default 86400). External panels also re-check `northset.book_join_coverage_floor` (default 0.5)
inside `bench_northset`. Synth path expects coverage ≈ 1.0.

Receipt fields: `join_coverage`, `mean_book_age_seconds`, `max_book_age_seconds`,
`book_join_coverage_floor`. Provenance: `book_dgp` / family `dgp` — see DATA_CONTRACTS.

```bash
uv run dipcatcher northset --config configs/research.yaml
```

## Promotion and live boundary

Promotion remains fail-closed. No local command enables live capital, creates a
broker order, or establishes live P&L evidence. Live readiness additionally
requires authorized point-in-time vendor data, authenticated broker/fill
reconciliation, non-synthetic holdout and forward evidence, venue-specific
cost/liquidity measurements, and an independently reviewable authorization.
Join/book summary echo: `join_coverage=… book_source=…` (`book_dgp` only in receipt blob). SYNTHETIC ≈ coverage 1 / `synthetic_lob`; external ≥ join floor / vendor source. See DATA_CONTRACTS.

## TOB size vs notional share

Prefer documenting both as ∈(0,1] after #62: `tob_size_share` (size) vs `tob_notional_share` (notional, touch-priced TOB) — Commander #53/#54/#62; never equate. DATA_CONTRACTS **tob_size_share vs tob_notional_share**; ADR-021.

Doctor: `research_receipt` gates health; `northset.shape_floors` is config echo only.

Notional proxies + `mean_tob_size_share` vs `tob_size_share_finite_rate`: DATA_CONTRACTS **Notional proxies** / **mean_tob_size_share vs tob_size_share_finite_rate**.

Spread/imbalance aliases + microprice_weight_balance: DATA_CONTRACTS. Post-fix: book `effective_spread` **==** quoted/`spread`; candle diagnostic is `close_mid_abs_rel` only — never equate. DATA_CONTRACTS **Reminder: quoted_spread alias vs dual-column**.

## Northset spreads: dual columns (FIXED)

Fuse keeps book `effective_spread` (ask−bid). Candle diagnostic is `close_mid_abs_rel`. Receipts: `mean_effective_spread` (book), `mean_close_mid_abs_rel` (candle), plus `mean_spread_bps` / `mean_quoted_spread`. DATA_CONTRACTS **dual columns**.


## Northset spreads: touch vs estimators; queue vs OFI

Do not equate `mean_quoted_spread`/`mean_spread_bps` with `roll_spread`/`corwin_schultz_spread`/`abdi_ranaldo_spread`. Do not read `queue_imbalance_mean` as mean OFI. DATA_CONTRACTS **Touch means vs OHLC** / **queue_imbalance vs ofi**.

## Session OFI vs bar `ofi` / `queue_imbalance`; Amihud honesty

Validity + fail-closed + receipt names: DATA_CONTRACTS **Session OFI vs bar-level**. Illiquidity means (`amihud_mean`, …) are research_only — **no live Sharpe**. DATA_CONTRACTS **Illiquidity means**.

## Northset CLI: spread means echo

`dipcatcher northset` echoes quoted + half + `mean_spread_bps` + `mean_close_mid_abs_rel` + `mean_microprice_weight_balance` on one line. Research compact line includes `mean_spread_bps` (not `mean_quoted_spread`). `session_ofi_sum_mean` remains blob-only. DATA_CONTRACTS **CLI echo: quoted / half / close–mid / microprice means**.

## VPIN: H32 vs H43

Compact `vpin=` is always-on `vpin_mean` (H32 / book eligibility). `session_book_vpin_mean` (H43 / session eligibility) is blob-only — do not equate. DATA_CONTRACTS **session_book_vpin_mean vs vpin_mean**.

## Session daily aggregates: last snap vs path vs VPIN

`session_close_*` = last session snapshot; `session_*_mean` / `n_session_book_snaps` = path reduces; `session_book_vpin` = path OFI toxicity (H43) — not daily `vpin_mean` (H32). Field table: DATA_CONTRACTS **session_close_***. VPIN means: **session_book_vpin_mean vs vpin_mean** / OPS **VPIN: H32 vs H43**.

## session_imbalance_std / session_book_source

Fuse `session_imbalance_std` (per-day path std) ≠ receipt `mean_session_imbalance_std` (`nanmean`; CLI echoes the mean). Not IC; soft-verify ≥0 when finite. Parallel to `n_session_book_snaps` vs `mean_session_book_snaps`. `session_book_source` ≠ CLI/receipt `book_source`. DATA_CONTRACTS **`session_imbalance_std` vs `mean_session_imbalance_std`** + **session_imbalance_std / session_book_source**.

## Spread means + candle geometry ICs

Do not equate `mean_spread_bps` / half / quoted / effective / close_mid blindly — DATA_CONTRACTS **Spread receipt means: never equate blindly**. Wick→H26, CLV→H30; `candle_body_ret` IC only. **Candle geometry IC family**.

## verify-research: spread mean identities (`northset_spread_*_honesty_errors`)

`dipcatcher verify-research` → `research.verify` soft-appends errors from:

1. `northset_half_spread_honesty_errors` — `mean_half_spread ≈ ½ mean_quoted_spread`
2. `northset_spread_bps_honesty_errors` — `mean_spread_bps ≈ 2× mean_half_spread_bps`
3. `northset_spread_receipt_honesty_errors` — `mean_effective_spread ≈ mean_quoted_spread`

Runs on family blobs **`northset`** and **`candle_order_book`**. Skip if either side missing/NaN. research_only — **never live Sharpe**. Full table: DATA_CONTRACTS **verify-research ops: northset_spread_*_honesty_errors**.

## TOB size vs notional share (#53 / #54)

Never equate `tob_size_share` ∈(0,1] with `tob_notional_share` ∈(0,1] (#62 touch-priced TOB) — size ≠ notional space. Floors/rates use size share. DATA_CONTRACTS **tob_size_share vs tob_notional_share**.

## verify-research: microprice_weight_balance ∈[0,1]

`northset_microprice_weight_balance_honesty_errors` on family `northset` only — finite `mean_microprice_weight_balance` must lie in [0,1]. Skip if NaN. DATA_CONTRACTS **mean_microprice_weight_balance**. `session_imbalance_mean` receipt mean: CoS-owned gap.

## Queue priority vs microprice weight

Never equate `queue_priority_proxy` / `ask_queue_priority_proxy` with `microprice_weight_balance`. DATA_CONTRACTS **queue_priority vs microprice_weight_balance**.


## verify-research: coverage_guarantee_scope

**Flag:** `marginal_exchangeable` = marginal coverage under exchangeability (not training-conditional; not live Sharpe).
**Skips:** empty JP/CV+ family, or no `coverage`/`coverage_floor` keys.
**Errors:** missing/invalid stamp when coverage keys present on nonempty Jackknife+/CV+.
Research_only — **never live Sharpe**. DATA_CONTRACTS **coverage_guarantee_scope soft-verify ops**.

## Concentration vs queue priority (Commander #59)

`bid/ask_size_concentration_top` (`top/side_depth`) ≠ `queue_priority_proxy` / `ask_queue_priority_proxy` (`top/(top+side_depth)`). Same-side only — opposite tops are MWB. DATA_CONTRACTS **bid/ask_size_concentration_top vs queue_priority_proxy**.


## queue_priority_finite_rate floors (ops refresh)

QUEUE twin of `concentration_top_finite_rate`: eligible when side depth > 0; optional `queue_priority_finite_floor` (default unset). External TOB → NaN rate — leave floor unset. Do not mix SIDE_STRUCTURE and QUEUE floors. DATA_CONTRACTS **queue_priority_finite_rate / side_notional_finite_rate** + concentration vs queue section.

## Shape / structure floors matrix

Five parallel honesty families: DEPTH_SHAPE (`n≥2` slopes) vs SIDE_STRUCTURE / QUEUE_STRUCTURE / SIDE_NOTIONAL / TOB_SHARE (depth>0 class — no n≥2). Synth ensure → cols present; external TOB → leave floors unset. Never gate one family with another's rate. DATA_CONTRACTS **Shape / structure floors matrix**.

## Microprice / mid / weight triad

`mid` midpoint ≠ `microprice` (size-weighted, ∈[bid,ask]) ≠ `microprice_weight_balance` `w`. Never equate `μ−mid` with `w` (`μ−mid = spread·(w−0.5)`). CLI: `mean_microprice_weight_balance`. DATA_CONTRACTS **microprice vs mid vs microprice_weight_balance**.

## imbalance_top vs microprice_weight_balance

When both finite: `imbalance_top = 2w − 1` with `w = microprice_weight_balance` (locked in book_metrics). Never equate with μ−mid. DATA_CONTRACTS **imbalance_top vs microprice_weight_balance**.


## Doctor vs CLI shape floors

`doctor` always echoes five `northset.shape_floors` config values (incl. None). `dipcatcher northset` shape line always echoes five rates; floor tokens only when set. DATA_CONTRACTS **Doctor shape_floors vs CLI shape-line echo**.

## Commander #62 TOB notional honesty

`top_of_book_notional_proxy = best_bid·top_bid + best_ask·top_ask` (not mid·sum). `tob_notional_share ∈ (0,1]` when finite — retire “may exceed 1”. Still never equate with `tob_size_share`. DATA_CONTRACTS **tob_size_share vs tob_notional_share** / **Notional proxies**.

## touch_size_imbalance alias surfaces

`touch_size_imbalance ≡ imbalance_top` = `2w − 1` when finite — **never a separate signal**.
REQUIRED: both fuse keys. Northset + candle IC: `imbalance_top` only (touch_size not in `_FEATURE_COLS`). No CLI `mean_touch_size_imbalance`. DATA_CONTRACTS **touch_size_imbalance alias surface checklist**.


## Honesty confirm #62 (tob_notional_share)

Active docs: both `tob_size_share` and `tob_notional_share` ∈(0,1] when finite; may>1 language retired. Never equate size vs notional. DATA_CONTRACTS **tob_size_share vs tob_notional_share**.

## Reminder: quoted vs dual-column

Book: `quoted_spread == effective_spread == spread`. `mean_close_mid_abs_rel` = candle `|close−mid|` only — never equate with quoted/effective means. Wait: Lieutenant shipping `mean_session_imbalance_mean` before session_imbalance docs. DATA_CONTRACTS **Reminder: quoted_spread alias vs dual-column**.

## Candle dual-IC alias honesty (verify-research)

Candle `_FEATURE_COLS` scores `imbalance_top` only (touch_size alias may exist on fuse, not dual-IC).
`dipcatcher verify-research`: **no** soft-verify helper asserts IC equality today; do not double-count receipt IC blocks.
research_only — **never live Sharpe**. DATA_CONTRACTS **Candle dual-IC alias honesty** + **verify-research ops: candle dual-IC alias**.

## `vpin_mean` soft-verify + ohlc vs gap

`vpin_mean` soft-verify ∈[0,1] (daily / H32) — never equate to `session_book_vpin_mean` (H43). `ohlc_identity_rate` (envelope / H20) ≠ `gap_finite_rate` (overnight gap finiteness). DATA_CONTRACTS **`vpin_mean` soft-verify** + **`ohlc_identity_rate` vs `gap_finite_rate`**.

## `book_uncrossed_rate` / H21 + session vs daily OHLC identity

`book_uncrossed_rate` soft-verify ∈[0,1]; H21 when finite + book eligible. `session_ohlc_identity_rate` ≠ `ohlc_identity_rate` (session candles vs daily bars; same identity fn). DATA_CONTRACTS **`book_uncrossed_rate` / H21** + **`session_ohlc_identity_rate` vs `ohlc_identity_rate`**.

## H20/H21/H22 triad + session reconstructs vs session OHLC

Finite gate → require H-row: H20 (`ohlc_identity_rate`), H21 (`book_uncrossed_rate` + eligible), H22 (`imbalance_top_p_ic` + eligible). `session_reconstructs_daily_rate` ≠ `session_ohlc_identity_rate`. DATA_CONTRACTS **H20/H21/H22 soft-verify triad** + **`session_reconstructs_daily_rate` vs `session_ohlc_identity_rate`**.

## session_volume_conservation + H23+ pointer

`session_volume_conservation_rate` soft-verify ∈[0,1]; H24 when finite. ≠ reconstructs/session_ohlc/chain. H23+ finite→H-row expansion via `northset_h23_h28_consistency_errors`. DATA_CONTRACTS **`session_volume_conservation_rate` soft-verify** + **H23+ soft-verify expansion pointer**.

## Sweep follow event vs cost vs reject

Never equate pre-cost `sweep_follow_event_mean_bps` with post-cost `sweep_follow_cost_adjusted_mean_bps`, or follow with reject. Soft-verify finite on each (CoS live). DATA_CONTRACTS **Sweep follow event vs cost-adjusted vs reject**.

## Kyle nest soft-verify suite

`verify-research` → `kyle_ofi_nest_honesty_errors` on northset nest: **26** helpers (Sergeant sync; **`test_kyle_ofi` 47/47**). Honesty only — not H-row mint. DATA_CONTRACTS **Kyle nest soft-verify suite**.

## `--dump-lambda-series` / nest date_series honesty

CLI: `dipcatcher kyle-ofi --dump-lambda-series PATH` writes a **research_only** parquet panel via `kyle_lambda_date_series_frame` (columns: `event_time`, `kyle_lambda`, `flow`, `target`, `book_source`, `book_dgp`, `research_only`, `claim=research_diagnostic_only`). No Sharpe/pnl columns.

Receipt companions (when nest present): `kyle_lambda_date_series_n_depth` / `kyle_lambda_date_series_n_ofi` = lengths of the in-memory λ series used for nest means/HAC — **not** the always-on `kyle_ofi_n_securities` name count.

Soft-verify: `kyle_lambda_date_series_honesty_errors` (fanned by `kyle_ofi_nest_honesty_errors`) — when either `n_*` is finite, require nest `research_only` + `claim=research_diagnostic_only` and no Sharpe/pnl-token keys. Aligns dump panel contract with nest stamps. research_only — never live Sharpe.

Never equate dump panel λ rows with always-on `kyle_ofi_lambda` (different aggregation axis). DATA_CONTRACTS **Always-on kyle_ofi_lambda vs nest kyle_lambda_ofi_mean**.

## Nest residual_ofi vs ofi_flow IC

When reading nest diagnostics: `residual_ofi_ex_depth_fwd_*` ≠ `ofi_flow_delta_mid_*` ≠ `ofi_fwd_*` (residualized+fwd vs raw contemporaneous vs raw predictive). Soft-verify honesty only. DATA_CONTRACTS **Nest residual_ofi vs ofi_flow IC**.

## Nest ofi_delta_mid_lag* vs ofi_flow; kyle_lambda_*_fwd_* vs contemporaneous λ

- `ofi_delta_mid_lag1_*` ≠ `ofi_flow_delta_mid_*` / `ofi_delta_mid_lag0_*` (feature lag).
- `ofi_delta_mid_lag0_*` ≠ `ofi_flow_delta_mid_*` as receipt keys (separate helpers).
- `kyle_lambda_ofi_fwd_*` / `kyle_lambda_depth_fwd_*` ≠ default `kyle_lambda_*_mean` (fwd target vs contemporaneous Δmid).

DATA_CONTRACTS **Nest ofi_delta_mid_lag0/lag1 vs ofi_flow**; **Nest kyle_lambda_*_fwd_* vs contemporaneous λ means**.

## Nest signed_depth_lag1 vs depth_flow; nest ofi_fwd vs always-on ofi_p_ic

- `signed_depth_delta_mid_lag1_*` ≠ `depth_flow_delta_mid_*` (feature lag vs Kyle companion).
- Nest `ofi_fwd_*` ≠ always-on `ofi_p_ic` / H27 (kyle fuse diagnostic ≠ candle-fuse discovery).

DATA_CONTRACTS **Nest signed_depth_delta_mid_lag1 vs depth_flow**; **Nest ofi_fwd vs always-on ofi_p_ic**.

## Nest signed_depth_fwd vs depth_flow; always-on ofi_lag vs nest lag1

- `signed_depth_fwd_*` ≠ `depth_flow_delta_mid_*` (fwd y vs contemporaneous companion).
- Always-on `ofi_lag1_corr` / `ofi_lag_*_ic` ≠ nest `ofi_delta_mid_lag1_*` (panel AR / fwd_ret IC ≠ nest Δmid IC).

DATA_CONTRACTS **Nest signed_depth_fwd vs depth_flow**; **Always-on ofi_lag panel vs nest ofi_delta_mid_lag1**.

## Nest depth_flow vs kyle_lambda_depth_mean; mid_lag1_corr vs ofi_lag1_corr

- `depth_flow_delta_mid_*` ≠ `kyle_lambda_depth_mean` (Spearman IC ≠ OLS λ; ofi twin likewise).
- Always-on `mid_lag1_corr` ≠ `ofi_lag1_corr` (mid AR ≠ ofi AR).

DATA_CONTRACTS **Nest depth_flow vs kyle_lambda_depth_mean**; **Always-on mid_lag1_corr vs ofi_lag1_corr**.

## Nest Kyle λ dispersion vs IC; always-on kyle_r2 vs nest HAC

- Dispersion deciles / rolling HAC on nest λ series ≠ `depth_flow_*` / `ofi_flow_*` IC companions.
- Always-on `kyle_r2` / `kyle_ofi_r2` ≠ nest `kyle_lambda_*_t`/`_p` or rolling HAC (no nest R² twin).

DATA_CONTRACTS **Nest Kyle λ dispersion vs IC companions**; **Always-on kyle_r2 / kyle_ofi_r2 vs nest HAC t/p**.

## Nest kyle_lambda_ofi_depth corr vs dispersion; n_securities vs n_dates

- `kyle_lambda_ofi_depth_spearman` / prod HAC ≠ depth/ofi dispersion p* or rolling HAC.
- Always-on `*_n_securities` ≠ nest `*_n_dates` (corr n = shared dates only).

DATA_CONTRACTS **Nest kyle_lambda_ofi_depth corr vs dispersion**; **Always-on n_securities vs nest n_dates**.

## Nest join_coverage vs always-on; Kyle claim / ic_method cheat-sheet

- Nest `join_coverage` (kyle fuse) ≠ always-on `join_coverage` (asof attach). Nest honesty also requires nonempty `book_source`.
- Nest stamps for verify-research: `research_only=true`, `claim=research_diagnostic_only`, `ic_method=date_level_spearman_hac` (when IC present), `family=kyle_ofi`; optional `hac_lags` ≥0 int; no Sharpe/pnl keys.

DATA_CONTRACTS **Nest join_coverage vs always-on join_coverage**; **Kyle nest claim / ic_method ops cheat-sheet**.

## Nest book_source/book_dgp vs always-on; nest n_fused/n_scored

- Nest `book_source` / `book_dgp` / nest `dgp`/`data_source`/`label` ≠ always-on family provenance (`component_sources`, evidence_label matrix).
- Nest `n_fused` / `n_scored` ≠ always-on `n_fused` / `n_scored` (kyle fuse vs asof; `delta_mid` vs `fwd_ret_1`).

DATA_CONTRACTS **Nest book_source/book_dgp vs always-on provenance**; **Nest n_fused/n_scored vs always-on sizing**.

## Nest min_names; SYN* vs MIXED; nest hac_lags

- Nest `min_names` echo ≠ always-on receipt (config-only on main); `kyle-ofi` CLI may use 3 vs config 5.
- Nest `label` SYN* / bars-driven path ≠ always-on `MIXED_SYNTHETIC_DERIVED` evidence matrix.
- Nest `hac_lags` optional stamp; always-on ICs have no mirrored receipt key.

DATA_CONTRACTS **Nest min_names vs always-on**; **Nest SYN* vs always-on MIXED**; **Nest hac_lags vs always-on**.

## Nest book_panel_path; Kyle nest docs IDLE

- Nest `book_panel_path` is an audit echo (passed through from northset when nested). Same path ≠ same `join_coverage`/`n_fused`. Standalone `kyle-ofi --book` may diverge.
- Kyle nest never-equate grind: **IDLE** unless new nest keys or H-row productization — DATA_CONTRACTS idle list.

DATA_CONTRACTS **Nest book_panel_path vs always-on**; **Kyle nest docs grind — IDLE**.

## Kyle nest — do-not-invent backlog

IDLE inventing. Do **not** invent never-equates / H-ids / means ahead of:

1. New nest receipt keys (Sergeant/CoS)
2. Nest H-row productization
3. Always-on echoing `min_names` / `hac_lags`
4. Nest loading from `book_panel_path` (today echo-only)
5. `include_kyle_ofi=false` with bare kyle keys on main blob

DATA_CONTRACTS **Kyle nest — do-not-invent backlog**.

## Northset receipt means (CoS overnight wave)

Depth/imbalance/concentration/queue/notional/spread_over_mid/microprice/slopes/tick/n_levels means inventory: NORTHSET **Northset receipt means — CoS overnight stamp wave**.

## Candle+LOB free-lane honesty (HF)

- Dual-IC touch_size listing **closed** in code; MWB soft-verify on candle_order_book + northset.
- FEATURE_COLS includes log_price_slope + mean_log_tick_spacing ICs; CLI `candle-book` echoes means/join/best IC.
- DATA_CONTRACTS **Candle+LOB `_FEATURE_COLS` inventory**.

## CoS northset receipt honesty dispatcher (40)

`verify-research` → `northset_receipt_honesty_errors(northset)` fans `NORTHSET_RECEIPT_HONESTY_HELPERS` (**40** live in catalog). Tuple + dispatcher live at catalog **EOF** — all helpers must be defined **before** the tuple (import NameError otherwise).

Separate lanes: `northset_session_means_honesty_errors`; `kyle_ofi_nest_honesty_errors` (**26**); direct wires such as Sergeant `mean_tob_notional_share_honesty_errors` (∈(0,1], ≠ size-share; receipt-stamp + verify wire tests).

DATA_CONTRACTS **CoS northset_receipt_honesty_errors**; **Sergeant free-lane mean_tob_notional_share**.

## Sergeant receipt-honesty verify fan-in; candle finite_rate_* gap; Commander merge

- `verify-research` calls `northset_receipt_honesty_errors(northset)` → all **40** helpers (~28 newly via fan-in vs prior direct-only wires).
- Candle `structure_finite_rate_honesty_errors` looks for `finite_rate_microprice_minus_mid` / `finite_rate_*_size_concentration_top` — candle bench may still omit those stamps (helper no-ops until CoS/Sergeant stamp). `depth_shape_finite_rate` is stamped and checked separately.
- Lt **SKIPPED** Commander box **#73–#100** → Mac merge (Mac through #175; box mirror behind).

DATA_CONTRACTS **Sergeant verify wire northset_receipt_honesty_errors fan-in**; **Candle-bench gap structure_finite_rate_honesty**; **Commander box #73–#100**.

## Sergeant verify dedupe; CLI rates; CoS mean_* 54/54

- `verify-research`: one northset receipt fan-in (40); duplicates for tuple members removed; candle_order_book-only wires kept.
- `dipcatcher northset` rate line: `ohlc_identity_rate` / `session_ohlc_identity_rate` / `book_uncrossed_rate` / `session_chain_rate` / `session_reconstructs_daily_rate` / `session_volume_conservation_rate` / `structure_finite_rate`.
- CoS **mean_* CLI echo 54/54** complete.

DATA_CONTRACTS **Sergeant verify dedupe**; **Sergeant CLI rate echoes**; **CoS mean_* CLI 54/54**.

## CoS sweep mean_diff/excess echoes; candle-book n_fused/min_names; Commander #141–#150

- Northset CLI: `sweep_*_event_mean_bps` ← nested `mean_excess_bps`; `sweep_*_control_diff_mean_bps` ← `mean_diff_bps`.
- `dipcatcher candle-book`: echoes `n_fused` + `min_names` (and CoS `finite_rate_*` when stamped).
- CoS mean_* CLI **54/54** complete.
- Commander Mac property **#141–#150** continuous (bounds/aliases/tick); Lt owns box merge.

DATA_CONTRACTS **CoS mean_excess/mean_diff → sweep echoes**; **CoS candle-book n_fused+min_names**; **Commander #141–#150**.

## CoS candle finite_rate_*; Sergeant family= parity; Commander #161–#175

- Candle-book stamps + CLI: `finite_rate_microprice_minus_mid` / `finite_rate_bid|ask_size_concentration_top` — `structure_finite_rate_honesty` bites on synth.
- `dipcatcher candle-book` echoes `family=`; parity guard requires every non-ic bench stamp in CLI source.
- Commander Mac property **#161–#175** continuous (bounds / SIDE_NOTIONAL / QUEUE / TOB / SIDE_STRUCTURE).

DATA_CONTRACTS **CoS candle-book finite_rate_***; **Sergeant candle-book family= + non-ic parity**; **Commander #161–#175**.

## Lt SKIPPED box #73–#100 Mac merge; candle finite_rate + family= confirmed; #161–#175

- **SKIPPED:** Commander box #73–#100 → Mac property merge — Mac already through #175; box was behind.
- **Confirmed:** CoS candle `finite_rate_*` stamps + `structure_finite_rate_honesty` bites; Sergeant candle-book `family=` + non-ic parity guard.
- Commander **#161–#175** Mac property continuous (Mac-ahead).

DATA_CONTRACTS **Commander box #73–#100 SKIPPED**; **CoS/Sergeant candle-book confirmed live**; **Commander #161–#175**.








## Stamp-test wave — mean_* COMPLETE (0 remaining)

**GREEN (pytest-verified; NOT inflight):**
- Sergeant `structure_finite_rate` real stamp + derivation asserts **6/6** (`test_northset_finite_rates_receipt_stamp.py` + `test_candle_book_structure_finite_rate_stamp.py`); candle ≠ northset
- CoS `tests/unit/test_mean_session_means_receipt_stamp_batch.py` **4/4** (11 `mean_session_*` keys) + `_nanmean` asarray fix

**True INFLIGHT only (not stamp-tests):** Sergeant soft-verify `structure_finite_rate` on northset / CLI polish.

DATA_CONTRACTS **Stamp-test wave — mean_* COMPLETE**. Off inventing. No commits.



## Post–stamp-wave landings

GREEN: soft-verify dual-family **11/11**; never-equate **4/4**; Lt `_FEATURE_COLS` +5 **2/2**. No frozenset INFLIGHT — see CLI partition below.

## Northset CLI echo partition (Sergeant GREEN)

- `NORTHSET_CLI_ECHO_REQUIRED` **51** ∪ session keys; `NORTHSET_RECEIPT_BLOB_ONLY` **90**; **disjoint**.
- Why not full non-IC parity: northset is a large blob dump — diagnostics stay blob-only; CLI echoes a scoped ops set. CoS mean_* echo is separate.
- Tests: `test_northset_cli_echo_required_blob_only.py` **4/4**.

## CoS candle_feature_cols_ic soft-verify

- `candle_feature_cols_ic_honesty_errors` on candle family; tests **3/3** (+ structure IC presence **2/2**).

Soft-verify structure_finite dual-family **11/11 GREEN** (not INFLIGHT). DATA_CONTRACTS. Off inventing. No commits.

## Sergeant CLI frozenset + finite-on-synth (GREEN, NOT INFLIGHT)

- `test_northset_cli_echo_required_blob_only.py` **4/4** (REQUIRED 51 / BLOB_ONLY 90 / disjoint)
- `test_northset_cli_echo_required_finite_on_synth.py` **3/3** (present + finite / nonempty on synth)

## CoS queue_priority + FEATURE_COLS IC presence

- `test_mean_queue_priority_receipt_stamp.py` **3/3**
- FEATURE_COLS structure IC presence **5/5** (`structure_ic_presence` 2 + `ic_completeness` 3)

**IDLE awaiting** next assign — no open honesty-residual INFLIGHT in this lane. DATA_CONTRACTS. Off inventing. No commits.

## Northset CLASSIFIED partition (Sergeant GREEN)

- `NORTHSET_RECEIPT_CLASSIFIED` **206** = `REQUIRED`(**51**) ∪ `EXTRA`(**65**) ∪ `BLOB_ONLY`(**90**), pairwise disjoint
- Soft-verify fail-closed: `northset_receipt_key_classification_honesty_errors` (unknown non-IC/non-kyle stamps fail)
- Candle `join_coverage` honesty also wired on `candle_order_book` in `verify-research`
- Tests: `test_northset_receipt_key_classification_soft_verify.py` **3/3**; blob_only suite includes union match

## CoS FEATURE_COLS IC completeness + mid_lag1_corr

- `test_candle_feature_cols_ic_completeness_soft_verify.py` **3/3**
- `test_mid_lag1_corr_receipt_stamp.py` **3/3** (`mid_lag1_corr` / `mid_lag1_n_securities`)

DATA_CONTRACTS. Off inventing. No commits.

## Sergeant classify-or-echo + EXTRA finite-on-synth (GREEN)

- Classify-or-echo **pytest 11/11** (blob_only 5 + classification 3 + join_coverage 3); EXTRA **65**; CLASSIFIED **206**
- EXTRA finite-on-synth `test_northset_cli_echo_extra_finite_on_synth.py` **3/3 GREEN** (NOT INFLIGHT)

## CoS FEATURE_COLS ask_queue + MWB + queue pair honesty

- `_FEATURE_COLS` includes `ask_queue_priority_proxy` + `microprice_weight_balance`
- `northset_queue_priority_bid_ask_pair_honesty_errors`; `test_ask_queue_priority_and_mwb_ic_honesty.py` **4/4**

DATA_CONTRACTS. Off inventing. No commits.

## CoS FEATURE_COLS Spearman/Pearson + lag1 n_securities (12/12 GREEN)

- Completeness requires `ic_*` / `_p` / `_n_dates` / `_pearson` / `_t` per FEATURE_COLS feature
- Lag honesty: `mid_lag1_n_securities` + `ofi_lag1_n_securities` ≥ 0
- Suite **12/12**: ask_queue+MWB honesty 4 + completeness 3 + spearman/pearson/lag1 3 + structure IC presence 2
- EXTRA finite-on-synth remains **3/3 GREEN**

DATA_CONTRACTS. Off inventing. No commits.

## Sergeant BLOB_ONLY∉CLI + book_age fuse (5/5 GREEN)

- `test_northset_blob_only_never_in_cli.py` **2** — BLOB_ONLY keys must not appear in northset CLI source
- `test_northset_book_age_fuse_honesty.py` **3** — mean/max book_age fuse honesty + dispatcher fan-in
- CoS FEATURE_COLS Spearman/Pearson + lag1 n **12/12** confirmed

**Session L2 + candle book_age:** **9/9 GREEN** (not INFLIGHT). Snaps vs n_session **7/7 GREEN** (not INFLIGHT).

DATA_CONTRACTS. Off inventing. No commits.

## CoS MWB fuse + overnight/rv/semi (6/6 GREEN)

- `test_candle_mwb_fuse_and_overnight_rv_synth.py` **6/6** — fuse MWB unit; scored⇒mean; overnight/rv/semi/bv synth; notional_imbalance IC presence

## Sergeant session L2 identity + candle book_age (9/9 GREEN)

- `test_northset_session_identity_rates_soft_verify.py` **6** — session L2 identity; never-equate daily OHLC key
- `test_candle_book_age_fuse_honesty.py` **3** — candle PIT book_age fuse polish
- Prior INFLIGHT session L2 / candle book_age → **promoted GREEN**

BLOB_ONLY∉CLI + northset book_age fuse **5/5** still GREEN. DATA_CONTRACTS. Off inventing. No commits.

## Sergeant candle ic_method honesty (6/6 GREEN)

- `candle_order_book_ic_method_honesty_errors` — `test_candle_order_book_ic_method_soft_verify.py` **6/6**

## CoS spread_over_mid FEATURE_COLS + session_ofi_sum IC (4/4 GREEN)

- `test_spread_over_mid_ic_and_session_ofi_ic.py` **4/4** — FEATURE_COLS score; IC⇒mean; session_ofi_sum_* IC honesty; never-equate vs ofi_p_ic
- Sergeant candle ic_method HAC soft-verify **6/6** confirmed

**Snaps vs n_session:** promoted **7/7 GREEN**. No True INFLIGHT. DATA_CONTRACTS. Off inventing. No commits.

## CoS depth_imbalance_abs + ofi_lag IC (4/4 GREEN)

- `test_depth_imbalance_ic_and_ofi_lag_ic.py` **4/4**

## CoS queue/vpin IC + candle structure IC⇒mean (4/4 GREEN)

- `test_queue_vpin_ic_and_structure_means.py` **4/4**
- Helpers: structure IC⇒mean; queue_imbalance IC; vpin IC pack
- **IDLE awaiting** Sergeant next free-lane (promote when GREEN)

DATA_CONTRACTS. Off inventing. No commits.

## Sergeant best_feature IC identity (8/8 GREEN)

- `test_candle_feature_cols_best_feature_identity.py` **5** + `test_candle_feature_cols_ic_soft_verify.py` **3** = **8/8**

## Sergeant n_fused / n_bars / n_scored sizing (6/6 GREEN)

- `test_northset_n_bars_scored_soft_verify.py` **6/6** — `northset_n_bars_scored_honesty_errors`

## CoS remaining FEATURE_COLS means + amihud/depth/slope/body/vpin IC packs (3/3 GREEN)

- `test_amihud_depth_slope_ic_packs_and_feature_means.py` **3/3**
- best_feature IC identity **8/8** already GREEN

DATA_CONTRACTS. Off inventing. No commits.

## CoS p_ic catchall + session_close/sweep/VoR packs (3/3 GREEN)

- `test_p_ic_catchall_and_session_close_sweep_packs.py` **3/3**
- Catch-all `*_p_ic` ∈[0,1] / `*_n_dates` ≥0; session_close / sweep signed / volume_over_range IC packs

DATA_CONTRACTS. Off inventing. No commits.

## Sergeant candle_order_book sizing + depth≥1 (8/8 GREEN)

- `test_candle_order_book_sizing_soft_verify.py` **8/8** — sizing chain + `depth`≥1
- Never equate with northset `northset_n_bars_scored_honesty_errors`
- CoS p_ic catchall + session_close/sweep packs **3/3** confirmed

DATA_CONTRACTS. Off inventing. No commits.

## CoS dm_park + sweep_evidence_scope + candle join/chain (4/4 GREEN)

- `test_dm_park_sweep_scope_and_candle_join_chain.py` **4/4**
- Sergeant candle sizing + depth≥1 **8/8**; bool-flags **6/6**; claim/pearson/enums **4/4** confirmed

DATA_CONTRACTS. Off inventing. No commits.

## Sergeant dgp/book_dgp↔data_source (6/6 GREEN)

- `test_northset_receipt_dgp_data_source_soft_verify.py` **6/6**

## CoS all_*_rate unit + sweep evidence blob (3/3 GREEN)

- `test_pattern_rate_catchall_and_sweep_evidence_blob.py` **3/3**
- Candle sizing+depth **8/8** + dm_park/sweep_scope/join-chain **4/4** confirmed

DATA_CONTRACTS. Off inventing. No commits.

## Sergeant use_session_l2 ↔ gate consistency (6/6 GREEN)

- `test_northset_use_session_l2_gate_consistency.py` **6/6**
- Helper: `northset_use_session_l2_gate_consistency_errors`
- True ⇒ `session_l2_identity_gate=enforced`; False ⇒ `skipped`
- Confirmed: dgp/data_source **6/6**; all_*_rate + sweep blob **3/3**

DATA_CONTRACTS. Off inventing. No commits.

## CoS all_*_share unit + candle spread alias (3/3 GREEN)

- `test_share_catchall_and_candle_spread_alias.py` **3/3**
- Helpers: `northset_all_share_unit_honesty_errors`, `candle_spread_alias_honesty_errors` (verify.py)

## Sergeant component_sources (6/6 GREEN)

- `test_northset_component_sources_soft_verify.py` **6/6**
- Helper: `northset_component_sources_honesty_errors`
- Confirmed: use_session_l2↔gate **6/6**

DATA_CONTRACTS. Off inventing. No commits.

## Sergeant northset depth (6/6 GREEN)

- `test_northset_depth_soft_verify.py` **6/6**
- Helper: `northset_depth_honesty_errors` — receipt `depth` int ≥1
- Confirmed: component_sources **6/6**; all_*_share + candle spread alias **3/3**

DATA_CONTRACTS. Off inventing. No commits.

## CoS fraction catchall + queue_imbalance_mean alias (3/3 GREEN)

- `test_fraction_catchall_and_queue_imbalance_mean.py` **3/3**
- Confirmed: Sergeant northset depth **6/6**

DATA_CONTRACTS. Off inventing. No commits.

## Sergeant price_basis / return_basis (6/6 GREEN)

- `test_northset_price_return_basis_soft_verify.py` **6/6**
- Confirmed: fraction catchall + queue_imbalance_mean **3/3**; northset depth **6/6**

DATA_CONTRACTS. Off inventing. No commits.

## CoS candle ic_*_p / t / n_dates catchalls (4/4 GREEN)

- `test_candle_ic_p_t_ndates_catchalls.py` **4/4**
- Helpers: `candle_all_ic_p_unit_honesty_errors`, `candle_all_ic_t_finite_honesty_errors`, `candle_all_ic_n_dates_nonneg_honesty_errors`


## Sergeant family / book_source + label nonempty (8/8 GREEN)

- `test_northset_family_book_source_soft_verify.py` **8/8** (+label nonempty)


## Sergeant candle family / provenance (5/5 GREEN)

- `test_candle_order_book_family_provenance_soft_verify.py` **5/5**
- Helper: `candle_order_book_family_provenance_honesty_errors`

## CoS mean_imbalance_top + shape_columns_ensured rates (3/3 GREEN)

- `test_candle_imbalance_top_and_shape_ensured_rates.py` **3/3**
- Helpers: `mean_imbalance_top_honesty_errors`, `northset_shape_columns_ensured_rates_honesty_errors`
- Confirmed: northset family/book_source **6/6**; candle ic catchalls **4/4**

DATA_CONTRACTS. Off inventing. No commits.

## Sergeant candle dgp / data_source (5/5 GREEN)

- `test_candle_order_book_dgp_data_source_soft_verify.py` **5/5**
- Helper: `candle_order_book_dgp_data_source_honesty_errors` (≠ northset twin)

## CoS metrics_required_finite_ok ⇒ structure rates (4/4 GREEN)

- `test_metrics_required_finite_ok_rates_honesty.py` **4/4**
- Helper: `northset_metrics_required_finite_ok_rates_honesty_errors`
- Confirmed: candle provenance **5/5**; mean_imbalance_top + shape_columns_ensured **3/3**

DATA_CONTRACTS. Off inventing. No commits.

## Sergeant sweep_evidence_scope stamp-contract (10/10 GREEN)

- `test_northset_sweep_evidence_scope_stamp_contract.py` **10/10**
- Helper: `northset_sweep_evidence_scope_honesty_errors`
- Allowed: `synthetic` | `empirical_adjusted` | `fixture_raw_unadjusted` (legacy `vendor` fail-closed)
- Confirmed: candle dgp/data_source **5/5**; metrics_required_finite_ok⇒rates **4/4**

DATA_CONTRACTS. Off inventing. No commits.

## CoS depth_imbalance(+abs) + METRICS_REQUIRED finite-when-present (3/3 GREEN)

- `test_candle_depth_imbalance_and_metrics_keys_present.py` **3/3**

## Sergeant shape_columns_ensured ↔ book_panel_path (5/5 GREEN)

- `test_northset_shape_ensured_book_panel_path_soft_verify.py` **5/5**

## Sergeant include_kyle_ofi ↔ kyle_ofi nest (5/5 GREEN)

- `test_northset_include_kyle_ofi_nest_presence_soft_verify.py` **5/5**

## CoS research_only ⇒ claim (10/10 GREEN)

- `test_research_only_implies_claim.py` **10/10**
- Confirmed: sweep_evidence_scope stamp-contract **10/10**

DATA_CONTRACTS. Off inventing. No commits.

## Sergeant conservation ↔ reconstructs never-equate (4/4 GREEN)

- `test_session_volume_conservation_vs_reconstructs_never_equate.py` **4/4**
- H24 `session_volume_conservation_rate` ≠ H23 `session_reconstructs_daily_rate`

## CoS control_sample_adequate ⇒ n/p/t (3/3 GREEN)

- `test_sweep_control_sample_adequate_honesty.py` **3/3**

## Sergeant impact_estimator_scope stamp-contract (4/4 GREEN)

- `test_northset_impact_estimator_scope_stamp_contract.py` **4/4**
- Allowed stamp: `per_security_equal_weight`
- Confirmed: include_kyle↔nest **5/5**; shape_ensured↔path **5/5**; research_only⇒claim **10/10**

DATA_CONTRACTS. Off inventing. No commits.

## CoS fold_positive rates (2/2 GREEN)

- `test_sweep_fold_positive_rates_honesty.py` **2/2**
- Helper: `northset_sweep_fold_positive_rates_honesty_errors`

## Sergeant book_join_coverage_floor (4/4 GREEN)

- `test_northset_book_join_coverage_floor_soft_verify.py` **4/4**
- Via `northset_shape_and_session_l2_floors_honesty_errors`
- Confirmed: conservation↔reconstructs **4/4**; impact_estimator_scope **4/4**; control_sample_adequate **3/3**

DATA_CONTRACTS. Off inventing. No commits.

## CoS amihud / qlike / corwin pack (8/8 GREEN)

- `test_amihud_qlike_range_spread_honesty_pack.py` **8/8**
- Helpers: `amihud_mean_honesty_errors`, `northset_qlike_means_honesty_errors`, `northset_range_spread_honesty_errors`

## Sergeant session_chain ↔ siblings never-equate (4/4 GREEN)

- `test_session_chain_vs_siblings_never_equate.py` **4/4**
- H29 `session_chain_rate` ≠ H23/H24 siblings
- Confirmed: book_join_coverage_floor **4/4**; fold_positive rates **2/2**

DATA_CONTRACTS. Off inventing. No commits.

## CoS candle mean_*_frac pack (3/3 GREEN)

- `test_candle_frac_and_spread_x_honesty.py` **3/3**
- Confirmed: session_chain↔siblings **4/4**; amihud/qlike/corwin **8/8**

DATA_CONTRACTS. Off inventing. No commits.

## Sergeant family / book_source + label nonempty (8/8 GREEN)

- `test_northset_family_book_source_soft_verify.py` **8/8** (+label nonempty)

## Sergeant ohlc ↔ session_ohlc never-equate (4/4 GREEN)

- `test_ohlc_identity_vs_session_ohlc_never_equate.py` **4/4**
- H20 `ohlc_identity_rate` ≠ `session_ohlc_identity_rate`

## Sergeant ohlc ↔ gap_finite never-equate (4/4 GREEN)

- `test_ohlc_identity_vs_gap_finite_never_equate.py` **4/4**
- `ohlc_identity_rate` ≠ `gap_finite_rate`
- Confirmed: ohlc↔session_ohlc **4/4**; label nonempty **8/8**

DATA_CONTRACTS. Off inventing. No commits.

## CoS candle_direction (4/4 GREEN)

- `test_candle_direction_mean_honesty.py` **4/4**
- Helper: `candle_direction_mean_honesty_errors` (verify.py)
- Confirmed: ohlc↔gap_finite never-equate **4/4**

DATA_CONTRACTS. Off inventing. No commits.

## Sergeant H20 ↔ H21 never-equate (4/4 GREEN)

- `test_ohlc_identity_vs_book_uncrossed_never_equate.py` **4/4**
- H20 `ohlc_identity_rate` ≠ H21 `book_uncrossed_rate`

## CoS wick_skew + candle_body_ret (3/3 GREEN)

- `test_wick_skew_and_candle_body_ret_finite_pack.py` **3/3**
- Confirmed: ohlc↔gap_finite **4/4**; candle_direction **4/4**

DATA_CONTRACTS. Off inventing. No commits.

## Sergeant session_ohlc ↔ reconstructs never-equate (4/4 GREEN)

- `test_session_ohlc_vs_reconstructs_never_equate.py` **4/4**
- `session_ohlc_identity_rate` ≠ H23 `session_reconstructs_daily_rate`

## CoS signed_vol_x_imbalance (4/4 GREEN)

- `test_signed_vol_x_imbalance_finite_pack.py` **4/4**
- Helper: `candle_signed_vol_x_imbalance_mean_honesty_errors`
- Confirmed: H20↔H21 **4/4**; wick_skew+body_ret **3/3**

DATA_CONTRACTS. Off inventing. No commits.

## Sergeant H21 ↔ H22 never-equate (4/4 GREEN) — triad complete

- `test_book_uncrossed_vs_imbalance_p_ic_never_equate.py` **4/4**
- H21 `book_uncrossed_rate` ≠ H22 `imbalance_top_p_ic`

## CoS candle ofi / queue imbalance means (3/3 GREEN)

- `test_candle_ofi_and_queue_imbalance_means_honesty.py` **3/3**

## CoS microprice_minus_mid finite (4/4 GREEN)

- `test_microprice_minus_mid_finite_pack.py` **4/4**
- Confirmed: session_ohlc↔reconstructs **4/4**; H20↔H21 **4/4**; signed_vol **4/4**

DATA_CONTRACTS. Off inventing. No commits.

## CoS spread_bps + log slopes (3/3 GREEN)

- `test_candle_spread_bps_and_log_slopes_honesty.py` **3/3**
- Helpers: `candle_spread_bps_nonneg_honesty_errors`, `candle_log_slopes_finite_honesty_errors`


## Sergeant H20 ↔ H22 never-equate (4/4 GREEN) — triad diagonal complete

- `test_ohlc_identity_vs_imbalance_p_ic_never_equate.py` **4/4**
- With H20↔H21 + H21↔H22 → triad never-equate **12/12**


## CoS tob_size_share + concentration tops + tick_spacing (3/3 GREEN)

- `test_candle_tob_concentration_tick_spacing_honesty.py` **3/3**
- Helpers: `mean_tob_size_share_honesty_errors`, `size_concentration_top_honesty_errors`, `candle_log_tick_spacing_finite_honesty_errors`
- tob/concentration ∈(0,1]; log tick spacing finite

DATA_CONTRACTS. Off inventing. No commits.

## CoS notional_imbalance + queue_priority + MWB (3/3 GREEN)

- `test_candle_notional_queue_mwb_means_honesty.py` **3/3**

## Commander gap ↔ uncrossed never-equate (4/4 GREEN)

- `test_gap_finite_vs_book_uncrossed_never_equate.py` **4/4**

## Sergeant session_bulk_vpin ↔ siblings never-equate (5/5 GREEN)

- `test_session_bulk_vpin_vs_siblings_never_equate.py` **5/5**

## Sergeant CLV alias identity (5/5 GREEN)

- `test_close_location_value_clv_alias_identity.py` **5/5**
- `close_location_value_{p,t}_ic` must match `clv_{p,t}_ic` when both stamped
- Confirmed: VPIN triad never-equate **5/5**; gap↔uncrossed **4/4**; candle_dir_x_imbalance **3/3**

DATA_CONTRACTS. Off inventing. No commits.

## Commander session_ohlc ↔ gap never-equate (4/4 GREEN) + identity #186–#190

- `test_session_ohlc_vs_gap_finite_never_equate.py` **4/4**
- `session_ohlc_identity_rate` ≠ `gap_finite_rate`

## CoS effective / half / ofi (3/3 GREEN)

- `test_candle_effective_spread_half_ofi_honesty.py` **3/3**
- Confirmed: Sergeant CLV alias identity **5/5**

DATA_CONTRACTS. Off inventing. No commits.

## Sergeant impact_proxy_warning (5/5 GREEN)

- `test_northset_impact_proxy_warning_soft_verify.py` **5/5**
- Token: `depth_or_ofi_proxy_not_signed_trade_flow`

## Sergeant product stamp (5/5 GREEN)

- `test_northset_product_stamp_soft_verify.py` **5/5**
- `product` must be `Northset` when stamped
- Confirmed: session_ohlc↔gap **4/4**; effective/half/ofi **3/3**

DATA_CONTRACTS. Off inventing. No commits.

## Commander session_ohlc ↔ book_uncrossed never-equate (4/4 GREEN) + identity #191–#195

- `test_session_ohlc_vs_book_uncrossed_never_equate.py` **4/4**
- `session_ohlc_identity_rate` ≠ H21 `book_uncrossed_rate`

## CoS finite_rate catchall + price_slope IC⇒mean (3/3 GREEN)

- `test_candle_finite_rate_prefix_and_price_slope_ic.py` **3/3**
- Helper: `candle_all_finite_rate_prefix_honesty_errors`
- Confirmed: impact_proxy_warning **5/5**; product stamp **5/5**

DATA_CONTRACTS. Off inventing. No commits.

## Sergeant session_ohlc ↔ volume_conservation never-equate (4/4 GREEN)

- `test_session_ohlc_vs_volume_conservation_never_equate.py` **4/4**
- `session_ohlc_identity_rate` ≠ H24 `session_volume_conservation_rate`
- Confirmed: impact_proxy_warning **5/5**; product stamp **5/5**

DATA_CONTRACTS. Off inventing. No commits.

## Sergeant session_ohlc ↔ session_chain (4/4 GREEN)

- `test_session_ohlc_vs_session_chain_never_equate.py` **4/4**

## Sergeant ohlc ↔ session_reconstructs (4/4 GREEN)

- `test_ohlc_identity_vs_session_reconstructs_never_equate.py` **4/4**

## Commander gap ↔ session_chain (4/4 GREEN) + #196–#200

- `test_gap_finite_vs_session_chain_never_equate.py` **4/4**

## CoS IC unit tighten (2/2 GREEN)

- `test_candle_ic_unit_interval_tighten.py` **2/2**
- bare ic/pearson/best ∈[-1,1]; mean_abs_ic ∈[0,1]

## Reconnect H20 ↔ H29 ohlc ↔ session_chain (4/4 GREEN)

- `test_ohlc_identity_vs_session_chain_never_equate.py` **4/4**

DATA_CONTRACTS. Off inventing. No commits.


## CoS H20 ↔ H24 ohlc ↔ volume_conservation (4/4 GREEN)

- `test_ohlc_identity_vs_volume_conservation_never_equate.py` **4/4** (confirm)
- H20 `ohlc_identity_rate` ≠ H24 `session_volume_conservation_rate`

## Sergeant H21 ↔ H29 book_uncrossed ↔ session_chain (4/4 GREEN)

- `test_book_uncrossed_vs_session_chain_never_equate.py` **4/4** (confirm)
- H21 `book_uncrossed_rate` ≠ H29 `session_chain_rate`

## Commander gap ↔ H23 session_reconstructs (4/4 GREEN)

- `test_gap_finite_vs_session_reconstructs_never_equate.py` **4/4** (confirm)
- `gap_finite_rate` ≠ H23 `session_reconstructs_daily_rate`

## General H21 ↔ H24 book_uncrossed ↔ volume_conservation (4/4 GREEN)

- `test_book_uncrossed_vs_volume_conservation_never_equate.py` **4/4**
- Helper: `book_uncrossed_vs_volume_conservation_never_equate_honesty_errors`
- H21 ≠ H24; Lt H20↔H29 already **12/12** reconnect suite

DATA_CONTRACTS. Off inventing. No commits.



## Commander gap ↔ volume_conservation (4/4 GREEN)

- `test_gap_finite_vs_session_volume_conservation_never_equate.py` **4/4** (confirm)
- `gap_finite_rate` ≠ H24 `session_volume_conservation_rate`

## Sergeant H22 ↔ H29 imbalance ↔ session_chain (4/4 GREEN)

- `test_imbalance_top_p_ic_vs_session_chain_never_equate.py` **4/4** (confirm)
- H22 `imbalance_top_p_ic` ≠ H29 `session_chain_rate`

## Commander H22 ↔ gap imbalance ↔ gap_finite (4/4 GREEN)

- `test_imbalance_top_p_ic_vs_gap_finite_never_equate.py` **4/4** (confirm)
- H22 `imbalance_top_p_ic` ≠ `gap_finite_rate`

DATA_CONTRACTS. Off inventing. No commits.



## CoS H22 ↔ H24 imbalance ↔ volume_conservation (4/4 GREEN)

- `test_imbalance_top_p_ic_vs_session_volume_conservation_never_equate.py` **4/4** (confirm)
- H22 `imbalance_top_p_ic` ≠ H24 `session_volume_conservation_rate`

## H22 ↔ session_ohlc imbalance ↔ session_ohlc_identity (4/4 GREEN)

- `test_imbalance_top_p_ic_vs_session_ohlc_never_equate.py` **4/4** (confirm)
- H22 `imbalance_top_p_ic` ≠ `session_ohlc_identity_rate`

DATA_CONTRACTS. Off inventing. No commits.



## Lt H22 ↔ H23 imbalance ↔ session_reconstructs (4/4 GREEN)

- `test_imbalance_top_p_ic_vs_session_reconstructs_never_equate.py` **4/4** (confirm)
- H22 `imbalance_top_p_ic` ≠ H23 `session_reconstructs_daily_rate`
- H22 mesh edges complete: gap / chain / volume / session_ohlc / reconstructs

DATA_CONTRACTS. Off inventing. No commits.



## Sergeant H25 ↔ gap microprice ↔ gap_finite (4/4 GREEN)

- `test_microprice_p_ic_vs_gap_finite_never_equate.py` **4/4** (confirm)
- H25 `microprice_p_ic` ≠ `gap_finite_rate`

## Sergeant H27 ↔ gap ofi ↔ gap_finite (4/4 GREEN)

- `test_ofi_p_ic_vs_gap_finite_never_equate.py` **4/4** (confirm)
- H27 `ofi_p_ic` ≠ `gap_finite_rate`

## Lt H26 ↔ gap wick_skew ↔ gap_finite (4/4 GREEN)

- `test_wick_skew_p_ic_vs_gap_finite_never_equate.py` **4/4** (confirm)
- H26 `wick_skew_p_ic` ≠ `gap_finite_rate`

## CoS H30 ↔ gap clv ↔ gap_finite (4/4 GREEN)

- `test_clv_p_ic_vs_gap_finite_never_equate.py` **4/4** (confirm)
- H30 `clv_p_ic` ≠ `gap_finite_rate`
- IC↔gap fan-out (H25/H27/H26/H30) **16/16** beside H22↔gap

DATA_CONTRACTS. Off inventing. No commits.



## H32 ↔ gap vpin ↔ gap_finite (4/4 GREEN)

- `test_vpin_p_ic_vs_gap_finite_never_equate.py` **4/4** (confirm)
- H32 `vpin_p_ic` ≠ `gap_finite_rate`

## Sergeant H28 ↔ gap dm_gk ↔ gap_finite (4/4 GREEN)

- `test_dm_gk_vs_park_p_vs_gap_finite_never_equate.py` **4/4** (confirm)
- H28 `dm_gk_vs_park_p` ≠ `gap_finite_rate`

## CoS H31 ↔ gap dm_split ↔ gap_finite (4/4 GREEN)

- `test_dm_split_vs_park_p_vs_gap_finite_never_equate.py` **4/4** (confirm)
- H31 `dm_split_vs_park_p` ≠ `gap_finite_rate`

## Lt H33 ↔ gap sweep_reject ↔ gap_finite (4/4 GREEN)

- `test_sweep_reject_signed_p_ic_vs_gap_finite_never_equate.py` **4/4** (confirm)
- H33 `sweep_reject_signed_p_ic` ≠ `gap_finite_rate`

DATA_CONTRACTS. Off inventing. No commits. Commander through #245 noted in day_grind.



## Lt H34 ↔ gap sweep_follow ↔ gap_finite (4/4 GREEN)

- `test_sweep_follow_signed_p_ic_vs_gap_finite_never_equate.py` **4/4** (confirm)
- H34 `sweep_follow_signed_p_ic` ≠ `gap_finite_rate`
- Reconfirmed on-disk: H32 vpin / H33 reject / H28 dm_gk / H31 dm_split (all **4/4**)

DATA_CONTRACTS. Off inventing. No commits.



## H35 ↔ gap reject_event ↔ gap_finite (4/4 GREEN)

- `test_sweep_reject_event_p_vs_gap_finite_never_equate.py` **4/4** (confirm)
- H35 `sweep_reject_event_p` ≠ `gap_finite_rate`

## H36 ↔ gap follow_event ↔ gap_finite (4/4 GREEN)

- `test_sweep_follow_event_p_vs_gap_finite_never_equate.py` **4/4** (confirm)
- H36 `sweep_follow_event_p` ≠ `gap_finite_rate`

## Lt H37 ↔ gap reject_placebo ↔ gap_finite (4/4 GREEN)

- `test_sweep_reject_placebo_p_vs_gap_finite_never_equate.py` **4/4** (confirm)
- H37 `sweep_reject_placebo_p` ≠ `gap_finite_rate`

## Sergeant H38 ↔ gap follow_placebo ↔ gap_finite (4/4 GREEN)

- `test_sweep_follow_placebo_p_vs_gap_finite_never_equate.py` **4/4** (confirm)
- H38 `sweep_follow_placebo_p` ≠ `gap_finite_rate`

## Lt H40 ↔ gap follow_cost ↔ gap_finite (4/4 GREEN)

- `test_sweep_follow_cost_adjusted_mean_bps_vs_gap_finite_never_equate.py` **4/4** (confirm)
- H40 `sweep_follow_cost_adjusted_mean_bps` ≠ `gap_finite_rate`

DATA_CONTRACTS. Off inventing. No commits.


## CoS candle_dir_x_imbalance + close_mid_abs_rel

## CoS H39 ↔ gap reject_cost ↔ gap_finite (4/4 GREEN)

- `test_sweep_reject_cost_adjusted_mean_bps_vs_gap_finite_never_equate.py` **4/4** (confirm)
- H39 `sweep_reject_cost_adjusted_mean_bps` ≠ `gap_finite_rate`
- Sweep event/placebo/cost IC↔gap through H35–H40 complete

DATA_CONTRACTS. Off inventing. No commits.

 (3/3 GREEN)

- `test_candle_dir_x_imbalance_close_mid_join_honesty.py` **3/3**

DATA_CONTRACTS. Off inventing. No commits.


## Lt H41 ↔ gap reject_fold ↔ gap_finite (4/4 GREEN)

- `test_sweep_reject_fold_positive_fraction_vs_gap_finite_never_equate.py` **4/4** (confirm)
- H41 `sweep_reject_fold_positive_fraction` ≠ `gap_finite_rate`
- Reconfirmed: H38 follow_placebo / H39 reject_cost / H40 follow_cost already stamped **4/4**

## Sergeant H42 ↔ gap follow_fold ↔ gap_finite (4/4 GREEN)

- `test_sweep_follow_fold_positive_fraction_vs_gap_finite_never_equate.py` **4/4** (confirm)
- H42 `sweep_follow_fold_positive_fraction` ≠ `gap_finite_rate`

## CoS H44 ↔ gap reject_control ↔ gap_finite (4/4 GREEN)

- `test_sweep_reject_control_diff_p_vs_gap_finite_never_equate.py` **4/4** (confirm)
- H44 `sweep_reject_control_diff_p` ≠ `gap_finite_rate`

## Lt H45 ↔ gap follow_control ↔ gap_finite (4/4 GREEN)

- `test_sweep_follow_control_diff_p_vs_gap_finite_never_equate.py` **4/4** (confirm)
- H45 `sweep_follow_control_diff_p` ≠ `gap_finite_rate`
- Sweep↔gap through H33–H42 + H44/H45; H43 session_book_vpin↔gap stamped below

DATA_CONTRACTS. Off inventing. No c

## Lt H43 ↔ gap session_book_vpin ↔ gap_finite (4/4 GREEN)

- `test_session_book_vpin_p_ic_vs_gap_finite_never_equate.py` **4/4** (confirm)
- H43 `session_book_vpin_p_ic` ≠ `gap_finite_rate` (≠ H32 daily `vpin_p_ic`)
- Reconfirmed: H41 reject_fold / H42 follow_fold / H45 follow_control already stamped **4/4**
- IC↔gap SPECS ladder largely complete **H25–H45** (+H43)

DATA_CONTRACTS. Off inventing. No commits. Sergeant→H33≠H34; CoS→post-gap honesty.

ommits.



## Sergeant H33 ≠ H34 reject_signed ↔ follow_signed (4/4 GREEN)

- `test_sweep_reject_signed_p_ic_vs_sweep_follow_signed_p_ic_never_equate.py` **4/4** (confirm)
- H33 ≠ H34 signed IC siblings
- Reconfirmed: Lt H43 session_book_vpin↔gap already stamped **4/4**; IC↔gap SPECS closed

## Lt H35 ≠ H36 reject_event ↔ follow_event (4/4 GREEN)

- `test_sweep_reject_event_p_vs_sweep_follow_event_p_never_equate.py` **4/4** (confirm)

## Sergeant H37 ≠ H38 reject_placebo ↔ follow_placebo (4/4 GREEN)

- `test_sweep_reject_placebo_p_vs_sweep_follow_placebo_p_never_equate.py` **4/4** (confirm)

## CoS H39 ≠ H40 reject_cost ↔ follow_cost (4/4 GREEN)

- `test_sweep_reject_cost_adjusted_mean_bps_vs_sweep_follow_cost_adjusted_mean_bps_never_equate.py` **4/4** (confirm)
- Sibling ladder signed/event/placebo/cost **16/16**; fold/control below → COMPLETE

DATA_CONTRACT

## CoS candle spread_bps alias vs over_mid (4/4 GREEN)

- `test_candle_spread_bps_vs_over_mid_identity.py` **4/4** (confirm)
- Helper: `candle_spread_alias_honesty_errors` — `mean_spread_bps ≈ 1e4 * mean_spread_over_mid`
- Reconfirmed: Sergeant H33≠H34 + Lt H35≠H36 already stamped **4/4**

## Lt H41 ≠ H42 reject_fold ↔ follow_fold (4/4 GREEN)

- `test_sweep_reject_fold_positive_fraction_vs_sweep_follow_fold_positive_fraction_never_equate.py` **4/4** (confirm)
- Reconfirmed: Sergeant H37≠H38 + CoS H39≠H40 already stamped **4/4**

## Lt H44 ≠ H45 reject_control ↔ follow_control (4/4 GREEN)

- `test_sweep_reject_control_diff_p_vs_sweep_follow_control_diff_p_never_equate.py` **4/4** (confirm)
- Reject≠follow sibling ladder **COMPLETE** (H33≠H34 … H44≠H45)

## Commander Residual #246–#255

- Mac identity batch: half_spread; mid; microprice; μ-mid[+bps]; spread_bps; depth_imbalance_abs; tops≤depths; ask>bid

DATA_CONTRACTS. Off inventing. No commits.

S. Off inventing. No commits.


## CoS tob / concentration / tick_spacing (3/3 GREEN)

- `test_candle_tob_concentration_tick_spacing_honesty.py` **3/3**
- Confirmed: spread_bps+log slopes **3/3**; H20↔H22 **4/4** (triad edges **12/12**)

DATA_CONTRACTS. Off inventing. No commits.

## CoS yang_zhang / overnight / QLIKE pack (5/5 GREEN)

- `test_yz_park_overnight_rv_bv_semi_pack.py` **5/5**
- Confirmed: candle mean_*_frac pack **3/3**

DATA_CONTRACTS. Off inventing. No commits.

## CoS IC t/mean/rank + finite_rate/floor catchalls (2/2 GREEN)

- `test_ic_finite_rate_floor_catchalls.py` **2/2**
- `*_t_ic` / `*_mean_ic` finite; `*_mean_rank_ic` ∈[-1,1]; `*_finite_rate` / `*_floor` ∈[0,1]
- Sergeant candle_order_book sizing + depth≥1 **8/8** confirmed

DATA_CONTRACTS. Off inventing. No commits.

## Sergeant receipt bool-flags (6/6 GREEN)

- `test_northset_receipt_bool_flags_soft_verify.py` **6/6** — `northset_receipt_bool_flags_honesty_errors`

## CoS candle claim + pearson + northset string enums (4/4 GREEN)

- `test_candle_claim_pearson_and_northset_enums.py` **4/4**
- IC t/mean/rank + finite_rate/floor catchalls **2/2** confirmed

DATA_CONTRACTS. Off inventing. No commits.

## CoS imbalance/CLV/microprice IC packs + ofi/QP/slope means (3/3 GREEN)

- `test_imbalance_clv_microprice_ic_packs_and_ofi_means.py` **3/3**
- queue/vpin + structure IC⇒mean **4/4** already documented

DATA_CONTRACTS. Off inventing. No commits.

## Sergeant snaps vs n_session (promoted GREEN 7/7)

- `test_session_book_snaps_n_session_consistency.py` **7/7**
- Candle ic_method HAC **6/6 GREEN** (not INFLIGHT); spread_over_mid+session_ofi IC **4/4 GREEN**

**No True INFLIGHT** this lane. DATA_CONTRACTS. Off inventing. No commits.

## CoS session_l2_enforced + notional IC⇒mean (5/5 GREEN)

- `test_session_l2_enforced_and_notional_ic_mean.py` **5/5**
- MWB fuse **6/6** and session L2 + candle book_age **9/9** remain GREEN (not INFLIGHT)

**Snaps vs n_session:** **7/7 GREEN** (not INFLIGHT).

DATA_CONTRACTS. Off inventing. No commits.

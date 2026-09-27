# P6.5 exec/microstructure audit — `execution/` + `microstructure/` + `northset/`

Scope: `execution/almgren_chriss.py`, `microstructure/{book_metrics, book_panel,
candle_book_features, bench, synthetic_lob, vendor_book_map}.py`,
`northset/{estimators, kyle_ofi, benches, sweeps, sweep_research, candles,
data_view, identities, __init__}.py`. Each function was checked against the
method cited in its docstring (Cont–Kukanov–Stoikov 2014 OFI, Kyle 1985,
Roll 1984, Corwin–Schultz 2012, Amihud 2002, Easley–López de Prado VPIN,
Almgren–Chriss 2000). Verdicts: `correct`, `fixed`, `waived` (documented looser
semantics). KATs live in
`tests/unit/microstructure/test_p65_exec_microstructure_audit.py`
(27 tests, deterministic, offline, synthetic data only).

## Bugs fixed

| fix commit | change |
|---|---|
| `e50e79a` | `almgren_chriss_trajectory`: fail-closed param validation + stable expm1 form |
| `c938f52` | `attach_candle_book_features`: real CKS OFI; `forward_close_return_labels`; `bench.py` label on bars; `synthesize_l2_from_bars` empty raise; **estimators**: CKS vpin legs, amihud dollar-volume floor, CS zero-pairs, `session_vpin` raise |
| `e2ffd95` | `bench_northset` label on canonical bars; `_attach_event_costs` honors `sweep_vol_lookback`; `northset.__all__` export fixes; `kyle_ofi` docstring clarifies forward-target semantics |
| `9837a6f` | KAT regression file (27 tests) |

### `execution/almgren_chriss.py`

| claim checked | verdict | evidence / fix |
|---|---|---|
| `x(t) = x0·sinh(κ(T−t))/sinh(κT)`, κ²=λσ²/η | **fixed** | `sinh(κT)` overflows for κT>~710 and the `isfinite(denom)` guard silently returned TWAP — the worst response to extreme urgency (the AC limit is immediate liquidation). Also NaN/inf params slipped past `risk_aversion <= 0` (NaN comparisons are False) into the same TWAP fallback, and fractional `n_slices` was truncated by `np.arange`. Now: finite/>0 validation on all params, `n_slices` must be an integral non-bool, negative `risk_aversion` raises, κ=0 handled as exact TWAP, and x(t) evaluated as `e^{-κt}(1−e^{−2κ(T−t)})/(1−e^{−2κT})` via `expm1` — bounded and monotone at any κT, with a final `isfinite` fail-closed check. KAT: κT≈3e5 → h[1] < 1e-3·x0 (was 0.9·x0 under the bug); κT=2 matches `sinh` form to 1e-9. |
| `twap_trajectory`, `front_loaded_trajectory` wrappers | correct | wrappers pass `risk_aversion=0` / fixed positive params — semantics preserved. |
| `slice_trades` = −Δx | correct | trivially consistent; callers receive sell-side inventory released per slice. |

### `northset/estimators.py`

| claim checked | verdict | evidence / fix |
|---|---|---|
| `order_flow_imbalance` — CKS (2014) decomposition | correct | `𝟙{b≥pb}q^B_n − 𝟙{b≤pb}q^B_{n−1} − 𝟙{a≤pa}q^A_n + 𝟙{a≥pa}q^A_{n−1}`; ties on the unchanged side cancel correctly (flat ask contributes −q^A_n+q^A_{n−1}). KAT: [12, −10, 6] on engineered book. |
| `vpin_proxy` buy/sell legs — CKS | **fixed** | Ask legs were swapped: `buy` got `ask_dn·q^A_{n−1}` and `sell` got `ask_up·q^A_n` — wrong condition AND wrong size index on both legs, so `buy−sell ≠ ofi` whenever the ask moved. Now `buy = bid_up·q^B_n + ask_up·q^A_{n−1}`, `sell = bid_dn·q^B_{n−1} + ask_dn·q^A_n`, so `buy−sell ≡ OFI`. KAT (bucket mode): tox [12/32, 10/26, 6/22]. |
| `amihud_illiquidity` — Amihud 2002, \|r\|/dollar-volume | **fixed** | `dvol.clip(lower_bound=1e-12)` minted `\|r\|·1e12` on zero/negative dollar volume; now `null` when `dvol ≤ 0` or non-finite (fail-closed). KAT: volume 0 and −5 → null. |
| `corwin_schultz_spread` — CS 2012, pairs (t−1,t) | **fixed** | filter `(spread>0)&(spread<1)` dropped negative-α pairs; CS convention sets them to 0 **and keeps them in the mean** — the old filter upward-biased panel means (and returned NaN when all pairs were negative). Now clips at 0 over finite pairs. KAT: all-jump panel → 0.0; mixing zero-pair and positive-pair securities halves the mean (previously equal). |
| `session_vpin` — bulk-volume VPIN | **fixed** | missing required columns silently returned NaN while siblings raise; now `ValueError` (empty frame still NaN — honest unmeasured). KAT both paths. |
| `session_vpin` sign convention | correct | `|Σ sign(C−O)·V| / ΣV` per parent day; KAT: all-up days → 1.0. |
| `roll_spread` — Roll 1984 `2√(−cov)` | correct | returns NaN (not 0) on positive autocovariance — honest unmeasured. KATs already exist in `test_northset.py`. |
| `kyle_lambda`-adjacent `lag1_corr`, `signed_*` helpers | correct | signatures verified vs call sites in `benches.py`/`sweep_research.py`. |

### `microstructure/candle_book_features.py`

| claim checked | verdict | evidence / fix |
|---|---|---|
| fused `ofi` semantics | **fixed** | when the book panel lacked `ofi`, the fuse stamped `diff(top_bid_size) − diff(top_ask_size)` — a size-delta proxy, not the CKS estimator used everywhere else (`kyle_ofi.cont_ofi_by_security`, `estimators.order_flow_imbalance`). Now computes `order_flow_imbalance` on the book's own event-time chronology before the as-of join. KAT: fused `ofi` == CKS OFI joined on `book_event_time`; book-supplied `ofi` is used verbatim. |
| `forward_close_return_labels` (new helper) | fixed | extracted so the 1-bar label is derived on the full bar panel; see bench/benches rows. |
| `queue_imbalance` fallback → `imbalance_top` | correct | same formula per BOOK_PANEL docs — kept as alias with comment. |
| PIT as-of join + age/lookahead checks | correct | backward join on `book_available_time ≤ decision_time`, empty-join and coverage-floor raise, `book_event_time > decision_time` and staleness raise. |

### `microstructure/bench.py` and `northset/benches.py`

| claim checked | verdict | evidence / fix |
|---|---|---|
| `fwd_ret_1` label = MATH_SPEC `y_{i,t+1} = C_{t+1}/C_t − 1` | **fixed** (both files) | label was `close.shift(-1)` on the **fused** frame after the book join dropped candle rows — a sparse external book silently minted multi-day spans as "1-bar" returns. Now `forward_close_return_labels` on the pre-join bar frame (`close` in `bench.py`; canonical `return_close` in `benches.py`), joined onto fused. KATs pin day0→day1 label with day2 absent from fused. |
| receipt honesty (no P&L/Sharpe keys, `research_only`) | correct | forbidden-key assertion exists; receipts verified unchanged in shape. |

### `microstructure/synthetic_lob.py`, `book_panel.py`, `book_metrics.py`, `vendor_book_map.py`

| claim checked | verdict | evidence / fix |
|---|---|---|
| `synthesize_l2_from_bars` empty input | **fixed** | vectorized path declined (`None`), snapshot path returned `[]`, then `pl.DataFrame([]).sort(...)` raised a cryptic `ColumnNotFoundError`; now an explicit `ValueError("zero snapshots")`. |
| `synthesize_snapshots_from_bars` seeding | correct | `np.random.default_rng(seed)` per call; deterministic; `book_metrics_from_snapshot` non-finite → NaN documented. |
| `book_panel` validate/required columns | correct | `BOOK_PANEL_REQUIRED`/`OPTIONAL` partition enforced; `ofi`/`queue_imbalance`/`vpin` optional (callers compute). |
| `vendor_book_map` remap honesty | correct | fail-closed on unknown vendor fields; `dry_run` used in tests. |

### `northset/kyle_ofi.py`

| claim checked | verdict | evidence / fix |
|---|---|---|
| `cont_ofi_series` — CKS on book stream | correct | identical decomposition to `estimators.order_flow_imbalance`; first obs = 0.0 (documented, vs NaN in estimators — kept: receipts/tests pin it). NaN inputs → NaN. |
| `delta_mid` == `fwd_delta_mid`, `fwd_ret_*` on fused clock | **waived** | `delta_mid` is the mid change to the NEXT fused row (book-event clock), so the default `kyle_lambda` target is predictive, not contemporaneous-Δp. Renaming would break receipt keys/tests; docstring now states the forward-looking semantics explicitly. |
| inner-join `fwd_ret_*` after row drops | waived | same filtered-frame caveat as benches, but targets are mid-price book events — coherent as book-clock labels; documented in docstring. |

### `northset/sweep_research.py`, `sweeps.py`, `candles.py`, `data_view.py`, `identities.py`

| claim checked | verdict | evidence / fix |
|---|---|---|
| `_attach_event_costs` lagged-vol lookback | **fixed** | fallback recomputed `sweep_lagged_vol` with hardcoded `rolling_std(20)` ignoring `config.northset.sweep_vol_lookback`; now `max(3, config.northset.sweep_vol_lookback)`. KAT: lookback=5 → first finite at index 6. |
| sweep windows strictly-before-event lagging | correct | `_lagged` paths verified `.shift(1)` before rolling; no lookahead. |
| `ohlc_identity_rate`, conservation/reconstruct rates, `geometry_rates` | correct | identities hold exactly on synthetic data; rates are honest-NaN on missing columns. |
| `canonical_northset_bars` adjusted-close requirement | correct | fail-closed when `require_adjusted_ohlc` and columns absent. |

### `northset/__init__.py`

| claim checked | verdict | fix |
|---|---|---|
| `__all__` ↔ `_EXPORT_MODULES` consistency | **fixed** | several public estimators (`abdi_ranaldo_spread`, `aggregate_session_book_to_daily`, `load_book_panel`, `snapshots_to_panel`, `synthesize_session_l2`, `validate_book_panel`, `write_book_panel`) were exported via `_LAZY`/`_EXPORT_MODULES` but missing from `__all__`; added. KAT: every `__all__` name resolves and the list has no duplicates. |

## Cross-cutting contract checks (verdict: correct unless noted)

- `benches.py` ↔ `sweep_research.py` ↔ `estimators.py` column names and metric
  helper signatures (`mean_tstat`, `date_ic_series`, `qlike`, `benjamini_hochberg`,
  `bootstrap_mean_ci`, `two_way_clustered_mean_tstat`, `wild_cluster_bootstrap_two_way_p`,
  `onesided_from_twosided`) verified against `metrics/` definitions.
- Vendor fixture `tests/fixtures/northset/alpaca_remapped_panel.parquet` covers
  every bar (65/65) — sparse-panel label behavior is pinned by the new KATs,
  not the fixture.
- Permanent-impact algebra `Σ n_k (c_k − n_k/2) = X²/2` is the continuous-AC
  convention — trajectory-independent, not a bug.

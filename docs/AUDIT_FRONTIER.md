# Frontier audit ledger — P6 code review

Scope: line-level review of the statistical core and money path, feeding the
ULTRAPLAN frontier program. Findings are stated conservatively; severity tags:
BUG (wrong result), RISK (wrong under plausible input), NIT (style/perf),
OK (verified correct).

## P6.2 Statistical core (`src/quant_fund/metrics/`)

### inference.py — OK with minor observations

- `newey_west_variance`: Bartlett kernel, bandwidth `floor(1.5 n^(1/3))`
  (nonstandard but documented), clamps omega at 0 with a correct comment
  explaining why no epsilon floor (fail-closed downstream). OK.
- `stationary_bootstrap_indices`: vectorized Politis–Romano construction
  verified — Bernoulli block-start mask, per-block uniform origin, position
  offsets modulo n. `new_block[:, 0] = True` forces a fresh start at index 0.
  OK.
- `optimal_block_length`: Politis–White 2004 flat-top kernel with the m̂ rule
  (`K_N = max(10, ceil(√n))`, threshold `2√(log10 n / n)`), `M = 2m̂`,
  `b̂ = (2Ĝ²/D̂²)^{1/3} n^{1/3}` — matches the paper. FFT-based autocovariance.
  OK.
- `diebold_mariano`: DM = HAC mean t-stat on aligned loss diffs, finite-row
  filter, honest `inconclusive`. OK.
- `two_way_clustered_mean_tstat`: CGM sandwich `V = Va + Vb − Vwhite` on
  demeaned residuals, df = min(Ga,Gb)−1. OK.
- `wild_cluster_bootstrap_two_way_p`: restricted Rademacher on cluster_a.
  NIT: the t* recomputes `mean(y*w)` inside `two_way_clustered_mean_tstat`,
  which re-centers the weighted residuals — slightly conservative relative to
  the canonical restricted bootstrap (residual under H0 IS y). Documented as
  an "honesty companion", so acceptable — flag for awareness, not a bug.
- `benjamini_hochberg`: correct BH step-up; NaN-safe ordering; validates
  p ∈ [0,1]. OK.
- `mean_difference_t`, `two_proportion_test`: correct fallbacks (Fisher when
  sparse), fail-closed on degenerate inputs. OK.

### snooping.py — OK

- `reality_check`: recentered stationary-bootstrap max statistic,
  `p = (1+#{V*≥V})/(B+1)` — never mints 0. OK.
- `spa_test`: three recenterings verified — `g_lower = max(mean,0)`,
  `g_consistent` threshold `√(2 log log n / n)` matches Hansen (2005) eq. for
  the consistent variant on the studentized scale; `g_upper = mean`.
  Shared-draw ordering `p_lower ≤ p_consistent ≤ p_upper` holds by
  construction. Zero-variance columns dropped and disclosed. OK.
- `stepm`: ascending sort + prefix-max over `t_boot[:, order]` is the correct
  remaining-set null (at the step testing ascending position m the not-yet-
  rejected set is the prefix {0..m}); `p_adj` = suffix running-max of raw
  step p-values. Verified — this is the subtle part of Romano–Wolf and it is
  right. OK.
- `model_confidence_set`: HLN range statistic; `d_ij = mean_j − mean_i`
  (positive = i worse); equivalence detected from DATA (`ptp`/`array_equal`)
  not bootstrap noise — avoids minting giant t-stats on constant diffs;
  `invalid` pairs break elimination with honest NaN p-values (fail-closed).
  Monotone running_p. OK.
- `_bootstrap_means`: count-matrix × data formulation — scale-invariant and
  fast. OK.

### scoring.py — OK + extended this session

- Existing suite (pinball, coverage, crossing, CRPS Gaussian/Student-t/
  empirical, QLIKE, PIT, Fissler–Ziegel) reviewed — consistent fail-closed
  contracts and equal-length validation. OK.
- ADDED `crps_gaussian_mixture` (closed-form mixture CRPS:
  Σw·E|X_k−y| − ½Σᵢⱼwᵢwⱼ·A(μᵢ−μⱼ, σᵢ²+σⱼ²)) and `gaussian_mixture_quantiles`
  (bisection on the monotone mixture CDF). Verified: K=1 bitwise equals
  `crps_gaussian`; MC |err| < 5e-4 vs 200k draws; quantile inversion +
  monotonicity; 10 invalid-input fail-closed cases. 19/19 metrics tests green
  on the remote evalenv.

## P6.1 Money path

- `backtest/engine.py` + `backtest/fast_replay.py`: fast replay proven
  BITWISE-identical to the reference (7/7 conformance + real 11-asset NAV and
  fills bit-equal), 3.15× faster than reference. Semantics preserved:
  decision-at-close/fill-at-next-open, sorted-sid order, sequential cash,
  participation caps, risk gates, kill switch, carried marks,
  total-return precedence. The float64-vs-np.float64 `sum()` Neumaier
  propagation is replicated exactly.
- `portfolio/risk_gate.py` — OK. Fail-closed on non-finite inputs, staleness
  bounds (price_age_bars, model_age_hours), and the full cap hierarchy
  (notional, name weight, gross, net, participation, predicted vol). Sell-side
  signed quantity handled explicitly.
- `execution/simulated_broker.py` — OK. Participation-capped child orders
  pass the gate (not the uncapped parent); limit orders rest unless the bar
  range touches them and gap-through fills take the adverse open; expired and
  gate-rejected resting orders cancel instead of re-recording every bar;
  shadow slots record intent but never move cash; kill switch blocks resting
  fills too. `not (price > 0)` correctly rejects NaN/missing marks.
- `backtest/carry_engine.py` — OK with one NIT. Pair-unit accounting
  (long spot + short perp) is honest: spot leg fully paid from cash with a
  buffer clamp, perp settles realized PnL on reductions, VWAP entries tracked
  per leg, funding `u·mark·rate` flows to cash on held pairs. Wick-paranoid
  liquidation marks the short at the bar HIGH and unwinds the hedge at bar
  CLOSE — deliberately avoids assuming simultaneous opposite extremes on two
  venues (comment documents why). MMR deficit loop unwinds largest-notional
  first (adverse ordering). Per-symbol attribution decomposes
  funding/realized/fees/liquidation.
  NIT: a pair filled at a bar's open receives that bar's funding settlement —
  intra-bar settlement timing is unknowable from OHLCV; slightly generous,
  convention documented in code.
- `paper/loop.py`, `paper/ledger.py`, `backtest/perp_engine.py`,
  `backtest/sleeves.py`: PENDING detailed pass.

## Challenger lane (P1.x) — status

Added to the arena (`sota_eval_kronos.py` / `sota_eval_native.py` /
`_challenger_col.py` / `splice_challenger_column.py`):

- `dip_gmm_k`: BIC-selected Gaussian mixture (K∈{1,2,3}, seeded) on the
  garch window, closed-form mixture CRPS. EVALUATED on 27 shards:
  mid-pack — d1fix pooled 0.015875 (8th/12), h4f 0.005324 (~empirical
  family). Not a pooled winner; honest arena member.
- `dip_skt`: Azzalini skew-t MLE (Nelder–Mead, xi/omega/alpha/nu) on the
  garch window, quantile-grid copy CRPS. In flight.
- `dip_qar`: quantile autoregression (statsmodels QuantReg) at the 9 lgbm
  levels, `crps_from_quantiles` convention, rearranged. In flight.
- `dip_conf_t`: split-conformal PIT-warped Student-t (2/3 fit + 1/3
  calibration, causal). In flight.
- `dip_regime`: two-state EWMA-vol regime mixture, transition-persistence
  weight clipped [0.05, 0.95]. In flight.

Splice discipline: column artifacts recompute the shard's deterministic origin
grid and bind via bars_sha256 + protocol fields + row count + asset identity;
`splice_challenger_column.py` appends columns with `appended_columns`
provenance (file hashes, n_finite) while preserving the scoring-contract
string — no existing score is modified.

## P6.1 cont. — perp_engine / sleeves / costs / forecast (2026-09-23)

### perp_engine.py — CLEAN, two NITs
- Signed-position accounting verified: flip realizes `current*(price-entry)` +
  re-anchors at fill; add → vwap entry; partial close → proportional realize.
- Leverage cap applied before participation cap (delta only shrinks → cap
  ordering safe); margin reject when no headroom; participation re-costs the
  capped delta.
- Wick-paranoid liquidation: longs tested at bar low, shorts at bar high;
  forced-unwind loop pops largest position until MMR deficit clears; ruin
  declared only after close marking (`nav_close <= 0 → ruined, break`).
- Stale held marks raise StaleValuationError before any exec — fail-closed.
- NIT-1 (LOW): funding events are applied via `fund_map.get(dt)` — a funding
  timestamp that doesn't exactly equal a bar `event_time` is silently dropped
  (no drop counter; relies on upstream alignment). Recommend counting
  dropped/unmatched funding rows.
- NIT-2 (LOW): `funding_paid` accumulates outflows only; funding received is
  invisible in metrics (net funding not surfaced).

### sleeves.py — CLEAN, one doc nit
- All causal: funding sleeves join_asof backward (rate known at decision);
  momentum/trend use shifted windows; cross-sectional ops are same-timestamp
  only; hysteresis book exits on missing/weak rates; rebalance_band bounds
  weight drift of fixed-unit pairs.
- liquidity_sweep_frame (northset): PIT-clean — shifted rolling extremes over
  *valid* bars only (bogus prints excluded before entering the window),
  full-lookback requirement, both-sweep ambiguity gated.
- DOC-NIT: funding_spike_fade_weights docstring claims weight "decays" between
  prints — implementation carries the last z-score flat until the next funding
  event. No leak; docstring overclaims.

### costs.py — CLEAN
- All components validate finiteness/sign; sqrt_impact coerces `abs(float(q))`
  (consistent with both engines' impact path); frictionless early-returns with
  explicit label after input validation.

### pipeline/forecast.py — PIT-CLEAN
- forecast_asof: decision-day features only via slice_day; explicit-asof miss
  raises (no latest-date fallback); ranker/RL artifacts are fixed loads with
  feature-contract enforcement — a stale/mismatched artifact raises rather
  than degrading to the momentum heuristic.
- history chain: sort contract enforced (`_require_assume_sorted_contract`
  raises on unsorted frames), prefix binary-search equals `<= asof` filter.
- history_for_calibration: cutoff at `times[idx-horizon-1]` — exact for
  off-grid asofs, one-bar conservative on-grid. Labels whose realization bar
  is `asof` itself are excluded — safe direction, tiny data loss.
- conformal_sets_asof: realized-label window only; Mondrian tercile cuts from
  cal rows; content-hash cache keys (no id(frame) reuse); cal_event_times
  provenance recorded.
- probability calibrator: fixed artifact, staleness-gated (`fit_end` must
  precede asof, ≤ max_age_days) — a future-fit calibrator is rejected.
- optimize_asof/build_causal_weight_panel: per-date asof loop, w_prev chained
  causally, covariance from trailing ret_1 with *logged* homoskedastic
  fallback (never silent).

### native-eval note (disclosure, not a bug)
- `dip_ewma_t` reports n_origins=0 in native-protocol receipts: its point
  forecast is a flat path (mu=0 by design) and Spearman rank-IC is undefined
  on a constant — honest NaN, vol metrics still finite. Disclosed, not faked.

## P6.x — returns / analytics / catalog / adapters (2026-09-23)

- metrics/returns.py — CLEAN: honest-NaN discipline throughout (non-finite
  propagates, never silently dropped); constant/empty/short series → NaN;
  `flag_high_sharpe` is a sanity flag; every public function is labeled
  research-diagnostic.
- metrics/analytics.py — CLEAN: `book_diagnostics` hard-stamps
  `research_only=True`/`live_pnl_claim=False`; `export_analytics_dict` forces
  both on every export regardless of input; `validate_analytics_export`
  fail-closes incl. sha256 digest verification.
- research/catalog.py — CLEAN: 10.7k-line verification/receipt layer. Zero
  pickle/eval/exec/network/RNG sites. Enforces the dual-catalog honesty split
  (research blobs forbid pnl/nav/sharpe keys outright; paper exports carry
  them only under live_pnl_claim=false).
- data/sources/adapters.py — CLEAN: in-progress klines dropped
  (close_time <= now), available_time = kline close (conservative PIT),
  malformed rows/finite-rate validation, non-empty enforced.

### Audit coverage summary (cumulative)
Verified: scoring, inference, snooping, returns, analytics, risk_gate,
simulated_broker, carry_engine, perp_engine, engine (bitwise-level via
fast_replay), sleeves, costs, forecast.py PIT chain, sweeps, catalog,
adapters. Remaining lower-priority surface: pipeline/train.py, cli/main.py,
research/benches.py, research/agent.py, northset internals.

### research/benches.py — CLEAN, one NIT (2026-09-23)
- Split discipline verified across all benches: `_holdout`/`_triple_split`
  chronological; fits consume `tr`/`cal` only, `te` evaluated only;
  `oos_rank_scores` uses walk-forward folds w/ embargo; `_policy_ridge_
  topk_rewards` fits on an expanding prefix, scores current date BEFORE
  appending it (causal).
- Conformal benches honor the train/cal/test contract (SplitCQR/ACI/
  Mondrian/Jackknife+/CV+ calibrate on `cal`, evaluate on `te`;
  `_vol_or_width` documents the PIT-safe scale-covariate fallback).
- Oracle/planted columns appear only as labeled diagnostics
  (`oracle_definition`, `oracle_column`, `leak_*` keys recorded in output).
- Honest-NaN + `research_only=True` stamps throughout; DM pairings align
  on common date keys, never positional truncation.
- NIT: `_scaled_fill` (l.138) and the inline fill at l.830 take the NaN-fill
  median over the full tr+te covariate vector — a look-ahead in the *fill
  value* only (labels untouched, median robust, immaterial). Would be
  strictly cleaner to use the tr-slice median.

### pipeline/train.py — CLEAN (2026-09-23)
- All model families (ranker/distribution/calibration/vol/alpha/regime/
  tail/liquidity/RL) go through date-level purged walk-forward:
  `walk_forward` + `purge_mask` + explicit embargo (defaults to horizon;
  embargo_bars=0 stays 0 — no silent substitution).
- Rankers/calibrators fit on `train_mask`/`fold.train_times` only;
  `np.isin` date-level masks prevent intra-day leakage. LambdaRank fits
  on mergesort-stable date order with correct group sizes.
- `purge_mask` (validation/purging.py) verified: label-window overlap vs
  test interval in session-index space; insertion-point endpoints are
  conservative when test bounds aren't observed sessions; label_end_times
  alignment enforced.
- `set_global_seed` at each family entry; no global RNG drift across
  models. Oracle columns never reach fit inputs.

### cli/main.py — CLEAN (2026-09-23)
- Zero dangerous call sites (no shell=True/pickle.load/eval/exec/
  verify=False/allow_pickle=True).
- `api` refuses non-loopback binds unless QUANT_API_KEY is set —
  fail-closed against unauthenticated exposure.
- `paper` forces research_only=True + live_pnl_claim=False in the emitted
  metrics JSON; --halt/--clear-halt mutually exclusive; --resume requires
  explicit run_id or prior latest_run.json; SYNTHETIC data label is
  loudly disclosed on stdout.
- monitor/report/tearsheet paths read ledgers via schema-validated
  loaders (digest-checked upstream in analytics/catalog).

### Coverage update (cumulative)
Now verified: scoring, inference, snooping, returns, analytics,
risk_gate, simulated_broker, carry_engine, perp_engine, engine,
sleeves, costs, forecast.py, sweeps, catalog, adapters, purging,
train.py, cli/main.py, research/benches.py, research/agent.py,
research/verify.py (immutable-receipt digest recompute + provenance/
scorecard/full-equality checks, fail-closed). Remaining lower-priority:
northset internals, execution/paper loop details (partially covered via
fault injection).

### Funding-observability fixes landed (2026-09-23, was: NITs)
- carry_engine + perp_engine: `funding_events_dropped` now counts funding
  prints whose timestamp never equals a bar time — feed misalignment is
  visible in receipts instead of silently fabricating zero funding.
  Verified: 2 aligned + 1 off-bar event -> dropped=1.
- perp_engine: `funding_received_total` + `funding_net` added (previously
  only outflows surfaced); carry_engine already had them.
- sleeves.funding_spike_fade_weights docstring corrected: weight is carried
  FLAT between prints via backward as-of join, not decaying.
- 39/39 carry/perp/sleeve/funding tests green; ruff+mypy clean.

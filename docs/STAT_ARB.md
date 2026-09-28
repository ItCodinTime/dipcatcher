# STAT_ARB — pairs-trading / statistical-arbitrage research pack

`src/quant_fund/research/pairs/` is a point-in-time research pack for
pairs trading: cointegration screening, hedge-ratio estimation,
Ornstein–Uhlenbeck spread diagnostics, and a z-score band signal. All
outputs are labeled **SYNTHETIC** correctness evidence — detection truth
and signal-alignment statistics only, never a P&L, Sharpe, or
live-trading claim (repo honesty contract).

## Methods

### Cointegration screening (`cointegration.py`)

1. **Correlation pre-filter** — candidate pairs are those whose |Pearson
   correlation of first differences of log prices| ≥ `min_corr`. The
   correlation is computed on *differences*, not levels: level
   correlation between I(1) series is spurious by construction.
2. **Engle–Granger residual ADF** — per surviving pair, OLS
   `y = α + βx + ε`, then `statsmodels adfuller(resid, regression="n")`
   (AIC lag selection). Both directions (`a ~ b`, `b ~ a`) are tested
   because the EG procedure is not symmetric; the smaller p-value's
   direction is kept.
3. **Multiple-testing control** — `benjamini_hochberg` (BH 1995, direct
   implementation, no extra deps) and `bonferroni` adjusted p-values are
   attached to every row.

**Honest p-value caveats** (also carried in the receipt `notes`):

- `adf_pvalue` applies the ordinary Dickey–Fuller surface to residuals
  of an *estimated* regression and is **liberal**. The strict gate is
  `passes_eg_cv`: `tau < EG 5% CV = −3.34` (MacKinnon 1991 response
  surface for the two-variable EG case; 1% = −3.90, 10% = −3.05 are
  exposed as `EG_CV_1PCT`/`EG_CV_10PCT`).
- BH/Bonferroni control error within the post-pre-filter family only;
  the correlation filter is itself selection, so adjusted p-values are
  optimistic relative to the full asset universe.

### Hedge ratio (`hedge.py`)

- `ols_hedge_ratio` — static Engle–Granger cointegrating regression.
- `kalman_hedge_ratio` — scalar Kalman filter, `β_t` random walk with
  process variance `q` and measurement variance `r` (Elliott, van der
  Hoek, Malcolm 2005). The intercept and `r` are calibrated by OLS on
  the burn-in prefix only, so the filter is strictly causal inside a
  call: `beta[t]` depends on observations `≤ t` plus frozen burn-in.

### OU spread diagnostics + signal (`spread.py`)

- `ar1_fit` → `ou_half_life = −ln2/ln(ρ)` (Vidyamurthy 2004); `inf`
  when `ρ ≤ 0` or `ρ ≥ 1`. `ou_params` maps the AR(1) onto continuous OU
  `(θ, μ, σ)` descriptively.
- `zscore_trailing(spread, window)` — `z_t` uses only `[t−w+1, t]`;
  NaN during warmup. `bands_position(z, entry, exit)` — the causal
  `{-1,0,+1}` state machine.

### Point-in-time pipeline (`pit.py`)

`pit_pair_signals` produces per-date `alpha`, `beta`, `spread`, `z`,
`position`. Every estimate at index `t` is a deterministic function of
data `≤ t`. The proof is a test: `tests/unit/research/pairs/test_pit.py`
mutates prices after a cut index and asserts the outputs at `≤ t` are
**bit-identical** (including the Kalman variant, whose burn-in
calibration is prefix-only). `tests/property/test_stat_arb_pairs.py`
extends the no-lookahead guarantee to randomized Hypothesis inputs.

## Evaluation (`evaluation.py`)

`run_pairs_eval` builds a seeded panel (`fixtures.planted_pair_panel`)
with one planted cointegrated pair — `logA = f + ε`, `logB = f + s + ε`
with `f` a random walk and `s` a stationary AR(1) — screens it, then
scores the PIT z-score signal:

- **Detection truth**: planted pair's rank by `p_bh` and pass flags.
- **Alignment**: Pearson/Spearman IC between the reversion direction
  `−z_t` and the `eval_horizon`-step forward spread change, plus the
  in-band hit rate (`|z| ≥ entry`).

`write_pairs_receipt` seals a `stat_arb_pairs_eval.v1` receipt under
`receipts/` (canonical payload, sha256 filename, atomic write) — same
contract as `cross_sectional.write_rankic_receipt`: it fails closed on
missing `data_label="SYNTHETIC"`, `live_pnl_claim≠False`, or any
forbidden-metric key.

## Measured results (reproduce exactly)

```bash
# 56-pack tests
uv run --no-sync pytest tests/unit/research/pairs \
    tests/property/test_stat_arb_pairs.py -q

# CLI eval on the default SYNTHETIC panel (seals a receipt)
dipcatcher pairs --n-dates 300 --window 100 --out-dir /tmp/pairs_rcpt
```

Observed on seed 0, `n_assets=8`, `n_dates=300`, `window=100`
(SYNTHETIC — correctness evidence, not market evidence):

| Metric | Value |
|---|---|
| Planted pair rank by `p_bh` | 1 (detected, `passes_bh`, `passes_eg_cv`) |
| `adf_tau` / `p_bh` | −4.25 / 2.66e-05 |
| Hedge ratio (true = 1.0) | 1.003 |
| Spearman IC (`−z` vs 1-step fwd spread change) | +0.285 |
| Pearson IC | +0.261 |
| In-band hit rate (`|z|≥2`, n=10) | 0.80 |
| Estimated half-life vs true | 2.8 vs 6.6 (AR(1) OLS is downward-biased — expected) |

Seed 0, `n_dates=600`, `window=120`: `adf_tau` −4.81, `p_bh` 2.52e-06,
Spearman IC +0.234, in-band hit rate 0.71 (n=34), estimated half-life
4.7 vs true 6.6.

## Limitations

- **No market evidence.** All numbers are on labeled SYNTHETIC panels;
  nothing here says a real pair is tradeable.
- **Liberal p-values** by construction (see caveats above); the EG
  critical-value flag is the conservative criterion.
- **Selection bias** from the correlation pre-filter — rejected-family
  sizes are post-filter.
- **AR(1) half-life is downward-biased** at these sample sizes; treat
  `half_life` as a ranking diagnostic, not a calibrated duration.
- **Costs/borrow/latency absent.** Positions are signal states, not an
  execution or capacity model; no fill or borrow assumptions are made.
- Univariate spread per pair only — no portfolio netting, no regime
  detection for broken cointegration.

## References

- Engle, R.F., Granger, C.W.J. (1987). *Co-integration and error
  correction.* Econometrica 55(2).
- MacKinnon, J.G. (1991/2010). *Critical values for cointegration tests*
  (response-surface EG critical values).
- Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and
  Analysis* (AR(1) half-life, z-score bands).
- Elliott, R.J., van der Hoek, J., Malcolm, W.P. (2005). *Pairs
  trading.* Quantitative Finance 5(3) (Kalman hedge ratio).
- Benjamini, Y., Hochberg, Y. (1995). *Controlling the false discovery
  rate.* JRSS-B 57(1).
- Ornstein, L.S., Uhlenbeck, G.E. (1930); Chan, E. (2013) for the
  AR(1) ↔ OU half-life bridge.

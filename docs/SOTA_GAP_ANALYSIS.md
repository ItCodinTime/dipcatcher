# SOTA gap analysis — Day Wave 21 (2026-09-16)

## Day Wave 165 — holdout must clear too — 2026-09-22

- A ranker `promote` flag now requires the same name to clear the CS
  rule on the selection window and on the frozen holdout. Book overlays
  no longer flip on a reality-check p-value alone: every holdout excess
  mean must be positive, and SPA must clear as well. `blend_weight`
  stays 0. The stored 20-day LambdaRank flag is restamped false because
  its mean IC is negative.

## Day Wave 164 — promotion requires positive IC on the same name — 2026-09-22

- `quant_fund.hedge_lab.promotion.clears_cs_promotion` is the CS gate.
  The same challenger must have mean IC > 0, win pairwise DM of −IC
  versus ridge, and sit in the StepM rejection set, with White RC and
  SPA also below 5%. A less-negative IC does not clear. Overlapping
  labels use Hansen–Hodrick lags. Kronos `blend_weight` also stays 0
  when its IC is not positive. No champion alias is written.

## Day Wave 163 — characteristic long-only and Faber timing — 2026-09-22

- Lagged low-vol, low-beta, 52-week high, residual momentum, 1-day
  reversal, Faber 10-month SPY timing, and a 200-day above-SMA book.
  Holdout versus SPY: White RC p 0.82, SPA p 0.75. Not a promotion.
  `blend_weight` 0. Artifact: `artifacts/hedge_lab/char_books.json`.
- Full sample: residual momentum Sharpe +0.82, ~4×, DD −33%. Faber
  Sharpe +0.71, DD −21%. Low-vol Sharpe +0.33, DD −21%. Reversal
  Sharpe −0.68, DD −85%. None beat SPY’s full-sample Sharpe +0.79
  with a holdout reality check. Holdout residual momentum +0.91 versus
  SPY +0.98.
- Champion stays public ridge.

## Day Wave 162 — stock k-grid, ETF dual momentum, UCB freeze — 2026-09-22

- Pre-registered set versus SPY on the 2025–26 holdout: White RC p 0.27,
  SPA p 0.28. Not a promotion. `blend_weight` 0.
  Artifact: `artifacts/hedge_lab/more_sleeves.json`.
- Full sample: stock top-1 Sharpe +1.13, CAGR 58%, ~115×, DD −66%.
  Holdout Sharpe +0.80, which loses to SPY +0.98. ETF dual top-1 / top-3
  full-sample Sharpe +0.78, in line with SPY +0.79. Their higher holdout
  Sharpes (+1.46 / +1.54) are inside a set the reality check does not
  clear. UCB1 froze `stock_top3` after a training path of Sharpe −0.09
  and DD −83%.
- Champion stays public ridge.

## Day Wave 161 — HMM gate, linear sleeve, XLK lead — 2026-09-22

- Pre-registered books on the 55-name tape, fit only before 2025-01-02.
  Holdout versus SPY: White RC p 0.089, SPA p 0.124. Not a promotion.
  `blend_weight` 0. Artifact: `artifacts/hedge_lab/sleeve_lane.json`.
- Holdout Sharpe: top-5 12–1 **+1.38** (DD −31%); linear sleeve +1.28
  (it picked the top-5 book on 2543 of 2634 days); HMM-gated top-5
  **+0.25** (the selection state did not persist); XLK-lead +0.29;
  SPY +0.98.
- Champion stays public ridge.

## Day Wave 160 — v2 tree lane vs ridge — 2026-09-22

- `configs/sota_protocol_v2.yaml` (`dipcatcher.sota.v2`) was hashed
  before the fit (`protocol_sha256`
  `6007e652f2d7974de8fddef3e904be2a25c28e7a68ff8f9b69b8c8e00b538eb2`).
  Engines: ridge, gbrt, lambdarank, xgboost. v1 protocol file was not
  edited. 54 names, rolling 252, delay 1, 10 bp, `future_idio_return_1`.
- IC: ridge −0.0023 (t −0.25); gbrt +0.0053 (t 0.65); lambdarank
  +0.0021 (t 0.34); xgboost −0.0063 (t −0.77). DM p-values 0.40, 0.58,
  0.67. RC p 0.39, SPA p 0.40, StepM []. **promote false.** 10 bp CS
  LS Sharpe: ridge −1.81, gbrt −2.00, lambdarank −1.92, xgboost −3.19.
- Same engines, pre-registered horizons, one shot.
  `future_idio_return_5`: lambdarank IC +0.010 (t 0.80), DM p 0.037,
  SPA p 0.048, RC p 0.059, StepM rejects lambdarank. RC misses 5%.
  **promote false.** 10 bp Sharpe −0.34.
  `future_idio_return_20`: lambdarank IC **−0.008** (t −0.37) beats
  ridge IC −0.060 (t −2.09) on DM p 0.009, RC p 0.018, SPA p 0.019,
  StepM. The receipt flag `promote` is true because the gate is
  relative to ridge. Absolute IC is still negative. 10 bp CS LS
  Sharpe −1.40, max DD −99%. Overlapping 20-day labels make the HAC
  t-stat optimistic. **Champion was not moved. `blend_weight` stays 0.**
- Artifacts: `ml_lane.json`, `ml_lane_h5.json`, `ml_lane_h20.json`.

## Day Wave 159 — trend/crash and vol Pareto; wide tape — 2026-09-21

- 200-day SMA + −20%/10d crash on `topk5_12_1` did **not** cut the
  drawdown (55-name DD −38% vs −39%) and cut Sharpe 1.03 → 0.92.
- Vol-target sweep of the same 55-name top-5 book, delay 1, 10 bp:
  raw Sharpe 1.03 / CAGR 30% / ~16× / DD −39%; 20% vol Sharpe 0.99 /
  CAGR 21% / ~7× / DD −30%; 10% vol Sharpe 0.98 / CAGR 11% / ~2.9× /
  DD −16%. Sharpe stays ~1 as leverage falls. `hit_sharpe5` false.
- Wide 421-name labels (survivorship-biased, not a PIT vintage):
  top-5 12–1 Sharpe 1.01, CAGR 35%, ~23×, DD −41%, and that
  drawdown is in the 2025–26 holdout. Not better skill.
- Champion stays public ridge. `blend_weight` 0.
  Artifacts: `target_hunt.json`, `target_hunt_wide.json`.

## Day Wave 158 — directional TSMOM / top-k long (not CS-idio) — 2026-09-21

- The CS-idio long-short was the wrong object for 10×: it shorts
  losers on a bull tape and pays 10 bp two ways. Directional
  total-return books, delay 1, monthly rebalance, 10 bp:
  `topk5_12_1` Sharpe **+1.04**, CAGR **30%**, total **+1491%** (~16×),
  DD **−39%**; holdout Sharpe **+1.38**. Beats SPY buy-hold Sharpe
  **+0.79**, total **+270%**, DD **−34%**. `dir_riskparity` Sharpe
  **+1.03**, DD **−23%**, ~4×. Vol-target 2.5% mix: Sharpe **+0.98**,
  DD **−4.6%**, CAGR **2.6%**. `hit_sharpe5` still false. TQQQ 3×
  ~39× / DD **−61%**. Champion stays public ridge. `blend_weight` 0.
- CLI: `dipcatcher ls hunt`. Artifact: `artifacts/hedge_lab/target_hunt.json`.

## Day Wave 157 — tsmom holdout confirmation — 2026-09-21

- Pre-declared confirmation of Wave 156 `tsmom` vs public ridge on the
  frozen Lightspeed cut (`SELECTION_END=2024-12-31`,
  `HOLDOUT_START=2025-01-02`). CLI: `dipcatcher ls confirm`.
  54 names, delay 1, 10 bp, rolling 252. Artifact:
  `artifacts/hedge_lab/holdout_confirm.json`.
- **Selection** (n=945, through 2024-12-31): ridge IC −0.0020 (t −0.20);
  tsmom +0.0283 (t 2.49); nautica +0.0147 (t 1.34). DM tsmom p=0.013;
  RC p=0.027; SPA p=0.014; StepM rejects `tsmom`. That window would
  have cleared the four gates. Wave 156 already used the full OOS IC,
  so this is not a license to promote after the fact.
- **Holdout** (n=189, from 2025-01-02): ridge IC −0.0038 (t −0.22);
  tsmom +0.0105 (t 0.54 p 0.59); nautica −0.0050 (t −0.23). DM tsmom
  p=0.60; RC p=0.44; SPA p=0.41; StepM []. **Did not persist.**
  10 bp CS LS holdout: tsmom Sharpe +0.56 DD −11%; nautica +0.61 DD
  −8.4%; ridge −2.67. Risk-stack tsmom Sharpe +0.26 DD −1.4%.
- Champion stays public ridge. `blend_weight` 0. Not a live P&L claim.

## Day Wave 156 — causal risk gates + tsmom/vme/krauss — 2026-09-21

- `quant_fund.risk.gates`: delay-1 vol target, DD halt / remaining
  budget, ES cap, fractional Kelly (negative μ → 0), CRC size,
  nautica crash analog, expanding-window StepM allow/deny.
  `BookRiskOverlay` takes `min` with Kelly and CRC (lookback 63).
  Sharpe is scale-invariant for constant leverage; vol targeting
  cannot mint Sharpe 5 from IC≈0. ADR-036.
- CS challengers `tsmom` (MOP 12–1), `vme` (AMP 2013 public proxy:
  12–1 + George–Hwang 52w, no B/M), `krauss` (Krauss–Do–Huck 2017
  linear logistic on the same purged WF as ridge). CLI:
  `dipcatcher ls hunt|book|race|confirm`.
- 55-name file_us v4 1d race (54 names after dropping SPY, 1134 OOS
  dates, rolling 252, delay 1, 10 bp): ridge IC −0.00232 (t −0.25);
  nautica +0.0114 (t 1.14); **tsmom +0.0254 (t 2.53)**; vme +0.0142
  (t 1.38); krauss +0.0036 (t 0.40). Pairwise DM of −IC vs ridge:
  tsmom p=0.014 (preferred tsmom); others p>0.17. White RC p=0.057;
  SPA consistent p=0.028; StepM rejects `tsmom` only. **Not promoted**
  (RC misses 5%). Champion stays public ridge. `blend_weight` 0.
- DD-safe CS LS: tsmom StepM-gated Sharpe +0.541, DD −1.94%, CAGR
  0.63%. Hunt riskstack: `ew_close_riskstack` Sharpe +0.994, DD
  −3.54%, CAGR 2.23%; `combo_voltgt` Sharpe +0.808, DD −3.42%, CAGR
  2.75%. hit_sharpe5=false. TQQQ 3× Sharpe +0.978, DD −61%. Not a
  live P&L claim. No Alpaca.

## Day Wave 155 — SLP3 Appendix A discrete HMM — 2026-09-21

- Implemented Jurafsky & Martin SLP3 Appendix A
  (https://web.stanford.edu/~jurafsky/slp3/A.pdf) as `quant_fund.hmm`:
  forward, backward, Viterbi, Baum–Welch. Eisner ice-cream numbers
  match Fig. A.5 / A.8 (`3 1 3` → P(O) 0.028562, path H C H).
  CLI: `dipcatcher hmm eisner`. Discrete; does not replace
  `GaussianHMMRegime` (hmmlearn, continuous).

## Day Wave 154 — Lightspeed engines — 2026-09-21

- Ported [cosmic-hydra/lightspeed](https://github.com/cosmic-hydra/lightspeed)
  (Artificial Hedge private book) into `quant_fund.lightspeed`:
  frozen TQQQ 20/180 SMA-seeded EMA rotation, nautica /
  stock-momentum top-1 63d + 200 SMA + crash −20%/10d, AFML
  metalabel reduce-only gate. CLI: `dipcatcher ls`. ADR-034.
- CS challenger `nautica` is a priori `cs_z_mom_60`. Champion
  stays public ridge. `blend_weight` 0. No Alpaca ALL-LIVE.
  Lightspeed Yahoo dollar paths are not Dipcatcher P&L.
- `dipcatcher ls book` on the 55-name v4 1d tape (2026-09-21):
  ridge IC −0.002 t −0.25; nautica +0.011 t 1.14; DM p 0.35;
  RC 0.22 SPA 0.23 StepM []. Not promoted. 10bp CS LS: ridge
  Sharpe −1.81, nautica +0.10 (lower turnover, not a gate win).
  Frozen concentrated momentum / reconstructed 3× TQQQ are
  reported with the Lightspeed IS/holdout cut; they are not
  the CS champion. Momentum holdout is only 82 days (Sharpe
  3.04 vs SPY 3.00). TQQQ holdout Sharpe 0.65 loses to SPY
  0.98. `blend_weight` stays 0.

## Day Wave 153 — quant-models engines — 2026-09-21

- Ported [davidalmeida90/quant-models](https://github.com/davidalmeida90/quant-models)
  and the README sibling repos into `quant_fund.quant_models`: BSM +
  implied vol, first/higher-order greeks (FD-validated), CRR European /
  American, Heston little-trap CF, GBM + discrete delta-hedge error,
  raw SVI, NSS, HRP / HCAA / ERC / long-only MVO, GEX + last-hour
  *decision* (no IBKR), TSMOM, Krauss linear window, GKX \(R^2\) vs
  zero, Blume beta. CLI: `dipcatcher qm`. ADR-033.
- Not a CS-ranker champion. Neural vol / deep hedging stay behind
  ADR-007. `blend_weight` 0. Not a live P&L claim.

## Day Wave 152 — discard sign-flip book; 1-day public CS — 2026-09-21

- The Wave 151 sign-flip / anti-book is discarded as a trading strategy.
  Costs are even under a score flip; random high-turnover Sharpe ≈ −6.7
  both ways. It is kept only as a diagnostic identity (ADR-032).
- Path is the public-feature CS rankers on the 55-name tape. Live
  `ranking_target` is explicit `future_idio_return_1`. Rolling 252-bar
  window. New challenger `ridge_neut`: within-date residualize public
  characteristics on size / price / vol, then T-ridge. Champion remains
  public ridge. `blend_weight` 0. Not a live P&L claim.

## Day Wave 151 — sign-flip mirror books — 2026-09-21

- Identity, not a free lunch: frictionless CS long-short Sharpe is odd
  under score sign-flip and leverage-invariant (rf = 0). Spread /
  commission / impact are even, so both the book and the mirror pay
  them. A 5% drawdown halt cannot coexist with −150% total return.
  Nested anti-univariate (worst train IC) is best-train IC after the
  flip. `dipcatcher hedge-lab --mirror` backtests negated
  `target_weight`.
- 54-name Yahoo 1-day idio, rolling 252, dollar-neutral top/bottom 20%,
  117635 rows. Frictionless Sharpes sum to 0 by construction. Best
  frictionless mirror is `classic_st` at +0.87 (+47% total, DD −36%).
  After 10 bp one-way turnover both sides of every card are negative;
  random scores go to Sharpe −6.7 *both* ways (costs, not alpha).
  Nested `anti_univ` OOS Sharpe +0.61 frictionless — the train loser
  did not stay a loser. No card has Sharpe > 5, max DD 5%, and +150%
  together. Champion remains public ridge. `blend_weight` 0. Not a
  live P&L claim.

## Day Wave 149 — small-N CS rankers, skip/residual momentum, T-ridge — 2026-09-20

- 54 names × 39 collinear CS-z made pooled `Ridge(α=1)` OLS and OLS
  Fama–MacBeth nearly unidentified (\(N\approx p+2\)). Champion ridge
  now date-demeans \(y\) and uses \(\alpha T\) when dates are passed.
  New challengers: `fm_ridge` (per-date ridge FM), `classic` (a priori
  signed public characteristics), `combo_ic` (Rapach train-IC weights).
- `features.v4` adds skip-momentum (20d excluding last week), CAPM
  residual 20d momentum, and rank-space reversal / overnight. Public
  card is 43 columns.
- 55-name 5-day v4 race (54 names, 1134 OOS dates, HAC 20): ridge IC
  −0.002 (t=−0.10). Best `combo_ic` 0.039 (t=1.93), principal
  portfolios 0.036 (t=1.75), joint IPCA-α 0.031 (t=1.59). Pairwise DM
  vs ridge does not reject at 5% (`combo_ic` p=0.056). White RC p=0.21;
  SPA consistent p=0.25; StepM rejects none. `classic` is ~0 — daily
  5-day idio on 54 Yahoo names does not carry the monthly factor zoo.
  Champion remains public ridge. `blend_weight` 0. Not a live P&L claim.

## Day Wave 148 — Gram LASSO, scale-stable DS, wide tape — 2026-09-20

- sklearn L1 challengers (`alasso`, `ds_lasso`) use Gram-precomputed
  coordinate descent. Naive \(n\)-path CD could not finish adaptive
  LASSO on the 55-name 5d tape (process killed mid-fit after combo) or
  the 420-name tape. `p≈39`, expanding \(n\sim10^5\)–\(10^6\).
- `ds_lasso` column-standardizes before selection / OLS. Pooled OLS on
  mixed-scale CS columns produced \(|\hat\beta|\sim10^{-13}\) and date IC
  identical to the null engines (`sdf_en` / `rp_pca` / `gx3pass`).
  PCR / alasso were already standardized.
- 55-name Yahoo session-close tape, `features.v3` gold (MAD fallback;
  54 names, 2629 dates, 117k rows, 39 public features, 5-day idio
  label, 1134 OOS dates, HAC 20): ridge mean date IC −0.011 (t=−0.61).
  Point-best GX 3-pass 0.029 (t=1.53), RP-PCA 0.028 (t=1.50), SDF
  0.026 (t=1.36). Pairwise DM vs ridge does not reject at 5% (GX
  p=0.057, RP-PCA p=0.057, SDF p=0.067). White RC p=0.13; SPA
  consistent p=0.23; StepM rejects none. `alasso` finished in 6s
  (Gram path). Wave 146’s +0.012 ridge IC was `features.v2` (exploded
  CS-z). Champion remains public ridge. `blend_weight` 0. Not a live
  P&L claim.
- Wide Yahoo tape (424 names / 1.13M bars, current-constituent
  survivorship, research-only) v3 gold rebuilt (1.05M rows). 5d horse
  race in flight.

## Day Wave 146 — joint IPCA-\(\alpha\), combo, adaptive LASSO, file-tape IC — 2026-09-20

- Unrestricted IPCA is joint ALS on \(F_{\mathrm{aug},t}=(1,f_t)'\)
  (Kelly–Pruitt–Su), not the nested residual \(\Gamma_\alpha\) of Wave
  144. Catalog adds `combo` (Rapach–Strauss–Zhou equal-weight univariate
  OLS) and `alasso` (Zou 2006 adaptive LASSO). Walk-forward row masks use
  integer-nanosecond membership so a 129k-row expanding horse race can
  finish.
- Causal public-feature walk-forward on the Yahoo session-close tape
  (54 names, 2633 dates, 1134 OOS IC dates, `PUBLIC_FEATURES` only):
  ridge mean date IC 0.012 (t=1.51); RP-PCA 0.020 (t=2.42); GX 3-pass
  0.020 (t=2.40); SDF ridge 0.019 (t=2.27); joint IPCA-\(\alpha\) 0.007;
  RFF 0.005. Pairwise DM on −IC vs ridge does not reject equal accuracy
  (RP-PCA p=0.14, GX p=0.15, SDF p=0.10; HAC 15 lags). SYNTHETIC IPCA
  was a DGP artifact. `blend_weight` 0. `forecast_asof` still loads
  ridge. No Sharpe in metadata. Overlay Sharpe stays in
  `hedge_lab_analytics` and is not live P&L.

## Day Wave 145 — FM, PCR/PLS, 3PRF, GBRT, principal portfolios — 2026-09-19

- Causal public-feature challengers add `fm` (Fama–MacBeth 1973), `pcr`
  / `pls` / `gbrt` (Gu–Kelly–Xiu RFS 2020 / NBER w25398; Huber GBRT,
  no NN), `tprf` (Kelly–Pruitt JoE 2015 automatic-proxy 3PRF), and `pp`
  (Kelly–Malamud–Pedersen JoF 2023 / NBER w27388 truncated SVD of
  \(\Pi=E[RS']\)). `fnw` empty selection and `sdf_en` zero loadings no
  longer emit constant scores. `bench_ranking` still pairwise-DM vs
  ridge. `forecast_asof` stamps Spearman only. `blend_weight` 0. No
  Sharpe in metadata. SYNTHETIC 16×120 public card: IPCA mean date IC
  0.400 (t=15.1) vs ridge 0.375 (t=9.5); Fama–MacBeth 0.395; RFF
  negative. Not a live P&L claim.

## Day Wave 144 — RP-PCA, FNW, 3-pass, DS-LASSO, KNS-EN, IPCA-\(\alpha\) — 2026-09-19

- Public-feature causal challengers now include `rff` / `rff_ridgeless`,
  `sdf_ridge` / `sdf_en`, `ipca` / `ipca_alpha`, `rp_pca` (Lettau–Pelger
  RFS 2020, NBER w24858), `fnw` (Freyberger–Neuhierl–Weber RFS 2020,
  NBER w23227), `gx3pass` (Giglio–Xiu JPE 2021, NBER w23527), and
  `ds_lasso` (Feng–Giglio–Xiu JoF 2020). `bench_ranking` walk-forward
  scores each on `PUBLIC_FEATURES` and pairwise-DM vs ridge. `forecast_asof`
  stamps Spearman vs ridge when a `ranker_<name>.joblib` exists and **does
  not blend**. `blend_weight` stays 0. No Sharpe in metadata.
- Completes VoC ridgeless \(z=0\), KNS eq. 28 ISTA, nested IPCA
  \(\Gamma_\alpha\). Deviations in MATH_SPEC. Not a live P&L claim.

## Day Wave 143 — VoC RFF, KNS SDF ridge, IPCA ALS rankers — 2026-09-19

- Kelly–Malamud–Zhou JoF 2024 (DOI 10.1111/jofi.13298, OA HTML), Kozak–Nagel–Santosh
  JFE 2020 (NBER w24070), and Kelly–Pruitt–Su JFE 2019 (NBER w24540) are
  implemented as causal ranking challengers `rff` / `sdf_ridge` / `ipca`.
  MATH_SPEC records stacked-panel \(T\), public CS features, and no
  \(\Gamma_\alpha\) IPCA. Catalog names only — `forecast_asof` still loads
  ridge. `blend_weight` stays 0. No Sharpe in model metadata.
- Dual ridge when \(P>T\); KNS extra shrinkage is \((\Sigma+zI)^{-1}\mu\);
  IPCA ALS with \(\Gamma'\Gamma=I_K\). Unit tests recover a trig signal,
  PC shrinkage factors \(d_j/(d_j+z)\), and a planted IPCA expected return.
  This is a paper-engine wiring claim, not a live P&L claim.

## Day Wave 142 — robinhood+ challenger card, leakage stamps, Jackknife+ units — 2026-09-19

- Causal SYNTHETIC ridge-only vs robinhood+ (blend_weight 1) on the same
  panel and as-ofs: date-level IC, pinball/CRPS, Diebold–Mariano. No Sharpe.
  numpy Markov lost (IC −0.35 vs ridge +0.77). Default ``blend_weight`` is 0.
  The engine stays enabled as a stamped challenger and does not size the book.
- Every ``forecast_asof`` stamps ``robinhood_plus_n_ok`` /
  ``robinhood_plus_n_fallback``. Poison-next-bar and unpublished
  ``available_time`` tests gate leakage.
- ``backend: torch`` is wired for local Kronos-mini only
  (``allow_network: false``). On this workstation Kronos-mini loaded from
  ``third_party/kronos_weights``: date IC −0.26 vs ridge +0.71 vs numpy Markov
  −0.51. It still does not size the book. Missing weights fail closed and
  never fall back to numpy.
- Jackknife+ coverage 0.07 vs ≥0.78 in the research notebook was a
  residual-unit bug: LOO location stayed in y/vol space while the band width
  was in returns. Scaling loc by the test-time vol restores coverage ~0.92
  on the 36×220 DGP (floor 0.80). That is receipt integrity, not a robinhood+
  score.

## Day Wave 141 — robinhood+ Kronos K-line engine — 2026-09-19

- Kronos (Shi et al., 2025, arXiv:2508.02739, MIT) is implemented in-repo as
  **robinhood+**: hierarchical BSQ K-line tokens + s1-then-s2 autoregression.
  Default backend is NumPy (no Hub download, no GPU). Optional torch loads
  official NeoQuasar/Kronos-* weights only with `[nn]` and `allow_network` or
  local paths. `forecast_asof` consumes robinhood+ as a core rank/alpha/
  distribution engine; fusion and the risk gate still apply. Name-level
  `vol_20` is unchanged. Optional research family `robinhood_plus` (not
  required). Internal name only — not affiliated with Robinhood Markets, Inc.
- This is a foundation-model wiring claim, not a live-performance claim.

## Day Wave 140 — named analytical nonlinear Ledoit–Wolf path — 2026-09-19

- `ledoit_wolf_nonlinear` is now a stamped catalog estimator, matching `oas` /
  `sample`. Ledoit–Wolf (2020) analytical spectral shrinkage on the
  listwise-complete trailing window returns trailing \(\Sigma\) with
  `family=ledoit_wolf_nonlinear` / `spec=ledoit_wolf_2020_analytical` /
  `covariance_object=trailing` / `sample=listwise_complete`. This is not
  2004 linear shrinkage and not numerical QuEST.
- Named `optimizer.covariance=ledoit_wolf_nonlinear` / `/risk/portfolio`
  consumes that matrix plus the GARCH/RGARCH overlay. Family or 2004-spec
  mismatch fails closed. Short or failed fits fail closed rather than
  Ledoit–Wolf 2004. When \(T\le N\) the estimator stays on the singular-case
  analytical map. Generic `shrinkage` / `ledoit_wolf_2017` / `quest` stay
  unknown. Default `ledoit_wolf` stays 2004. Factor stays unwired. This does
  not invent high-frequency RV.
- This is covariance-spec honesty, not a live-performance claim.

## Day Wave 139 — default Ledoit–Wolf stays Ledoit–Wolf when T≤N — 2026-09-19

- `ledoit_wolf` is now a stamped catalog estimator, matching `oas` / `sample`.
  sklearn `LedoitWolf` 2004 linear shrinkage on the listwise-complete trailing
  window returns trailing \(\Sigma\) with `family=ledoit_wolf` /
  `spec=ledoit_wolf_2004_linear` / `covariance_object=trailing` /
  `sample=listwise_complete` and the fitted shrinkage intensity.
- Default `optimizer.covariance=ledoit_wolf` / `/risk/portfolio` now stays
  Ledoit–Wolf when \(T\le N\) rather than silently switching to unbiased
  sample covariance. The named `sample` path is unchanged. Family mismatch
  fails closed. Short or failed default fits stay the homoskedastic proxy
  rather than sample. Overlay still applies (trailing LW has no \(D_{t+1}\)).
  This is not nonlinear Ledoit–Wolf 2017. Factor stays unwired. This does
  not invent high-frequency RV.
- This is covariance-spec honesty, not a live-performance claim.

## Day Wave 138 — named unrestricted CES AG-DCC optimize_asof path — 2026-09-19

- `optimize_asof` and `/risk/portfolio` now honor a named
  `optimizer.covariance=agdcc_full` path. It consumes the Wave 137 two-stage
  CES (2006) unrestricted AG-DCC one-step \(H_{t+1}=D_{t+1}R_{t+1}D_{t+1}\) and
  does **not** apply the GARCH/RGARCH overlay (AG-DCC already supplies
  \(D_{t+1}\)). Weights and `/risk/portfolio` stamp
  `covariance_estimator=agdcc_full` / `covariance_object=one_step_ahead` /
  `covariance_spec=cappiello_engle_sheppard_2006_full_agdcc`.
- This is not diagonal AG-DCC with a `parameterization` stamp: the named path
  does not call `agdcc`, `adcc`, `dcc_gaussian`, `dcc_student_t`, or `ccc`.
  Family mismatch (including an `agdcc` stamp), short history, incomplete asof
  rows, and failed fits fail closed rather than Ledoit–Wolf. Generic `dcc`
  stays unknown. Factor stays unwired. This does not invent high-frequency RV.
- This is covariance-spec honesty, not a live-performance claim.

## Day Wave 137 — unrestricted CES AG-DCC catalog estimator — 2026-09-19

- `agdcc_full` is now a two-stage catalog estimator, not an unspecified
  DCC alias. Stage 1 is univariate Gaussian GARCH(1,1). Stage 2 QML uses
  the CES (2006) unrestricted AG-DCC recursion
  \(Q_t=(\bar Q-A\bar Q A^\top-B\bar Q B^\top-G\bar N G^\top)
  +A z_{t-1}z_{t-1}^\top A^\top + B Q_{t-1} B^\top
  + G n_{t-1}n_{t-1}^\top G^\top\). Diagonal \(A,B,G\) recover Wave 135
  diagonal AG-DCC; nonzero off-diagonals do not. The public matrix is
  one-step \(H_{t+1}=D_{t+1}R_{t+1}D_{t+1}\) on the trailing contiguous
  complete-case window. Params stamp `family=agdcc_full` /
  `spec=cappiello_engle_sheppard_2006_full_agdcc` /
  `parameterization=full` / `asymmetric=true`.
- This is not diagonal AG-DCC with a `parameterization` stamp:
  `agdcc_full` does not call `agdcc`, `adcc`, `dcc_gaussian`,
  `dcc_student_t`, or `ccc`. `/models` lists it among implemented
  covariance estimators. `optimize_asof` / `/risk/portfolio` stay
  unwired (`unwired_optimizer_covariance:agdcc_full`) so unrestricted
  AG-DCC cannot silently size as diagonal AG-DCC, scalar ADCC, Gaussian
  DCC, CCC, or Ledoit–Wolf. Generic `dcc` stays unknown. Factor stays
  unwired. This does not invent high-frequency RV.
- This is covariance-spec honesty, not a live-performance claim.

## Day Wave 136 — named diagonal CES AG-DCC optimize_asof path — 2026-09-19

- `optimize_asof` and `/risk/portfolio` now honor a named
  `optimizer.covariance=agdcc` path. It consumes the Wave 135 two-stage
  CES (2006) diagonal AG-DCC one-step \(H_{t+1}=D_{t+1}R_{t+1}D_{t+1}\) and
  does **not** apply the GARCH/RGARCH overlay (AG-DCC already supplies
  \(D_{t+1}\)). Weights and `/risk/portfolio` stamp
  `covariance_estimator=agdcc` / `covariance_object=one_step_ahead` /
  `covariance_spec=cappiello_engle_sheppard_2006_diagonal_agdcc`.
- This is not scalar ADCC with a `parameterization` stamp: the named path
  does not call `adcc`, `dcc_gaussian`, `dcc_student_t`, or `ccc`. Family
  mismatch (including an `adcc` stamp), short history, incomplete asof
  rows, and failed fits fail closed rather than Ledoit–Wolf. Generic
  `dcc` stays unknown. Unrestricted full-matrix AG-DCC (`agdcc_full`)
  stays unspecified. Factor stays unwired. This does not invent
  high-frequency RV.
- This is covariance-spec honesty, not a live-performance claim.

## Day Wave 135 — diagonal CES AG-DCC catalog estimator — 2026-09-19

- `agdcc` is now a two-stage catalog estimator, not an unknown DCC alias.
  Stage 1 is univariate Gaussian GARCH(1,1). Stage 2 QML uses the CES (2006)
  diagonal AG-DCC recursion \(Q_t=(\bar Q-A\bar Q A-B\bar Q B-G\bar N G)
  +A z_{t-1}z_{t-1}^\top A+B Q_{t-1}B+G n_{t-1}n_{t-1}^\top G\) with
  diagonal \(A,B,G\). Equal diagonals recover scalar CES ADCC; heterogeneous
  diagonals do not. The public matrix is one-step \(H_{t+1}=D_{t+1}R_{t+1}D_{t+1}\)
  on the trailing contiguous complete-case window. Params stamp `family=agdcc`
  / `spec=cappiello_engle_sheppard_2006_diagonal_agdcc` /
  `parameterization=diagonal` / `asymmetric=true`.
- This is not scalar ADCC with a `parameterization` stamp: `agdcc` does not
  call `adcc`, `dcc_gaussian`, or `dcc_student_t`. `/models` lists it among
  implemented covariance estimators. `optimize_asof` / `/risk/portfolio` stay
  unwired (`unwired_optimizer_covariance:agdcc`) so diagonal AG-DCC cannot
  silently size as scalar ADCC, Gaussian DCC, CCC, or Ledoit–Wolf.
  Unrestricted full-matrix AG-DCC (`agdcc_full`) stays unspecified. Generic
  `dcc` stays unknown. Factor stays unwired. This does not invent
  high-frequency RV.
- This is covariance-spec honesty, not a live-performance claim.

## Day Wave 134 — named Bollerslev CCC optimize_asof path — 2026-09-18

- `optimize_asof` and `/risk/portfolio` now honor a named
  `optimizer.covariance=ccc` path. It consumes the Wave 133 two-stage
  Bollerslev (1990) CCC one-step \(H_{t+1}=D_{t+1} R D_{t+1}\) and does
  **not** apply the GARCH/RGARCH overlay (CCC already supplies
  \(D_{t+1}\)). Weights and `/risk/portfolio` stamp
  `covariance_estimator=ccc` / `covariance_object=one_step_ahead` /
  `covariance_spec=bollerslev_1990_ccc`.
- This is not Gaussian DCC with \(a=b=0\): the named path does not call
  `dcc_gaussian`, `dcc_student_t`, or `adcc`. Family mismatch, short
  history, incomplete asof rows, and failed fits fail closed rather than
  Ledoit–Wolf. Generic `dcc` stays unknown. Factor stays unwired. This
  does not invent high-frequency RV or implement matrix AG-DCC.
- This is covariance-spec honesty, not a live-performance claim.

## Day Wave 133 — Bollerslev CCC catalog estimator — 2026-09-18

- `ccc` is now a two-stage catalog estimator, not an unknown DCC alias.
  Stage 1 is univariate Gaussian GARCH(1,1). Stage 2 is Bollerslev (1990)
  constant \(R=\mathrm{corr}(z)\) of standardized residuals. There is no
  \(Q\) recursion and no \(a,b\) QML. The public matrix is one-step
  \(H_{t+1}=D_{t+1} R D_{t+1}\) on the trailing contiguous complete-case
  window. Params stamp `family=ccc` / `spec=bollerslev_1990_ccc` /
  `dynamic_correlation=false` / `covariance_object=one_step_ahead`.
- This is not Gaussian DCC with \(a=b=0\): `ccc` does not call
  `dcc_gaussian`, `dcc_student_t`, or `adcc`. `/models` lists it among
  implemented covariance estimators. `optimize_asof` / `/risk/portfolio`
  stay unwired (`unwired_optimizer_covariance:ccc`) so CCC cannot silently
  size as Gaussian DCC, Student-t DCC, scalar ADCC, or Ledoit–Wolf.
  Generic `dcc` stays unknown. This does not invent high-frequency RV,
  implement matrix AG-DCC, or wire factor covariance.
- This is covariance-spec honesty, not a live-performance claim.

## Day Wave 132 — named unbiased sample optimize_asof path — 2026-09-18

- `optimize_asof` and `/risk/portfolio` now honor a named
  `optimizer.covariance=sample` path. It consumes unbiased (`ddof=1`)
  sample covariance on the listwise-complete trailing `ret_1` window and
  **does** apply the GARCH/RGARCH overlay (trailing sample has no
  \(D_{t+1}\)). Weights and `/risk/portfolio` stamp
  `covariance_estimator=sample` / `covariance_object=trailing` /
  `covariance_spec=unbiased_sample`.
- This is not Ledoit–Wolf or OAS with a different stamp: the named path
  calls `sample`, refuses a Ledoit–Wolf family stamp, and fails closed on
  a short or failed fit rather than substituting Ledoit–Wolf, OAS, EWMA,
  or DCC. When \(T>N\) the estimator stays sample rather than silently
  switching to Ledoit–Wolf. The default Ledoit–Wolf path still falls back
  to this same unbiased spec when \(T\le N\). Factor stays unwired so
  missing PIT factor returns cannot be invented. Default remains trailing
  Ledoit–Wolf plus the overlay. This does not invent high-frequency RV or
  implement matrix AG-DCC.
- This is optimizer covariance-object identity, not a live-performance claim.

## Day Wave 131 — named Chen OAS optimize_asof path — 2026-09-18

- `optimize_asof` and `/risk/portfolio` now honor a named
  `optimizer.covariance=oas` path. It consumes Chen–Wiesel–Eldar–Hero (2010)
  Oracle Approximating Shrinkage on the listwise-complete trailing `ret_1`
  window and **does** apply the GARCH/RGARCH overlay (trailing shrinkage has
  no \(D_{t+1}\)). Weights and `/risk/portfolio` stamp
  `covariance_estimator=oas` / `covariance_object=trailing` /
  `covariance_spec=chen_wiesel_eldar_hero_2010`.
- This is not Ledoit–Wolf with a different stamp: the named path calls `oas`,
  refuses a Ledoit–Wolf family stamp, and fails closed on a short or failed
  fit rather than substituting Ledoit–Wolf, sample, EWMA, or DCC. When
  \(T\le N\) the estimator stays OAS rather than silently switching to
  sample. Generic `shrinkage` stays unknown. Default remains trailing
  Ledoit–Wolf plus the overlay. This does not invent high-frequency RV or
  implement matrix AG-DCC.
- This is optimizer covariance-object identity, not a live-performance claim.

## Day Wave 130 — named RiskMetrics EWMA optimize_asof path — 2026-09-18

- `optimize_asof` and `/risk/portfolio` now honor a named
  `optimizer.covariance=ewma` path. It consumes one-step RiskMetrics
  \(H_{t+1}=\lambda H_t+(1-\lambda)r_t r_t'\) on the trailing contiguous
  complete-case window (`features.ewma_lambda`). That matrix is **not**
  GARCH/RGARCH overlay-scaled. Weights and `/risk/portfolio` stamp
  `covariance_estimator=ewma` / `covariance_object=one_step_ahead` /
  `covariance_spec=jpmorgan_riskmetrics_1996`.
- Sequential EWMA no longer listwise-deletes holes or omits asof \(r_t\):
  an incomplete terminal row fails closed, and interior holes are not
  concatenated. The named path does not call Gaussian DCC, Student-t DCC,
  scalar ADCC, or Ledoit–Wolf, and a short or failed fit fails closed
  rather than substituting those estimators. Sample and factor stay
  unwired. Default remains trailing Ledoit–Wolf plus the overlay. This
  does not invent high-frequency RV or implement matrix AG-DCC.
- This is optimizer covariance-object identity, not a live-performance claim.

## Day Wave 129 — named scalar ADCC optimize_asof path — 2026-09-18

- `optimize_asof` and `/risk/portfolio` now honor a named
  `optimizer.covariance=adcc` path. It consumes the Wave 128 two-stage
  Cappiello–Engle–Sheppard scalar ADCC estimator (Gaussian GARCH stage 1 +
  CES stage-2 QML) and returns one-step \(H_{t+1}\) on the trailing
  contiguous complete-case window. That matrix is **not** GARCH/RGARCH
  overlay-scaled. Weights and `/risk/portfolio` stamp
  `covariance_estimator=adcc` / `covariance_object=one_step_ahead` /
  `covariance_spec=cappiello_engle_sheppard_2006`.
- This is not Gaussian or Student-t DCC with an `asymmetric` stamp: the
  named path calls `adcc`, refuses a non-ADCC family stamp, and fails
  closed on a short or failed fit rather than substituting `dcc_gaussian`,
  `dcc_student_t`, or Ledoit–Wolf. Generic `dcc` stays fail-closed.
  Default remains trailing Ledoit–Wolf plus the overlay. This is scalar
  CES ADCC, not matrix AG-DCC, and does not invent high-frequency RV.
- This is optimizer covariance-object identity, not a live-performance claim.

## Day Wave 128 — Cappiello–Engle–Sheppard scalar ADCC — 2026-09-18

- `adcc` is now a two-stage catalog estimator, not a named fail-closed stub.
  Stage 1 is univariate Gaussian GARCH(1,1). Stage 2 QML uses the CES (2006)
  scalar recursion \(Q_t=(1-a-b)\bar Q - g\bar N + a z_{t-1}z_{t-1}^\top +
  b Q_{t-1} + g n_{t-1}n_{t-1}^\top\) with \(n_t=I[z_t<0]\odot z_t\) and
  \(\bar N=\mathbb{E}[n_t n_t^\top]\). PD uses \(a+b+\kappa g<1\). The public
  matrix remains one-step \(H_{t+1}\) on the trailing contiguous complete-case
  window. Params stamp `family=adcc` / `spec=cappiello_engle_sheppard_2006` /
  `asymmetric=true` / `g` / `kappa`.
- This is not Gaussian DCC with an `asymmetric` stamp: `adcc` does not call
  `dcc_gaussian` or `dcc_student_t`. `/models` lists it among implemented
  covariance estimators. `optimize_asof` / `/risk/portfolio` stay unwired
  (`unwired_optimizer_covariance:adcc`) so ADCC cannot silently size as
  Gaussian DCC, Student-t DCC, or Ledoit–Wolf. This is scalar CES ADCC, not
  matrix AG-DCC, and does not invent high-frequency RV.
- This is DCC likelihood honesty, not a live-performance claim.

## Day Wave 127 — named Student-t DCC optimize_asof path — 2026-09-18

- `optimize_asof` and `/risk/portfolio` now honor a named
  `optimizer.covariance=dcc_student_t` path. It consumes the Wave 126 two-stage
  Student-t DCC estimator (univariate Student-t GARCH + covariance-t QML) and
  returns Engle one-step \(H_{t+1}\) on the trailing contiguous complete-case
  window. That matrix is **not** GARCH/RGARCH overlay-scaled. Weights and
  `/risk/portfolio` stamp `covariance_estimator=dcc_student_t` /
  `covariance_object=one_step_ahead` / `covariance_spec=engle_2002_student_t_dcc`.
- This is not Gaussian DCC with a `dist` stamp: the named path calls
  `dcc_student_t`, refuses a Gaussian family stamp, and fails closed on a
  short or failed fit rather than substituting `dcc_gaussian` or Ledoit–Wolf.
  Ambiguous aliases (`t`, `student_t`) are rejected so they cannot be read as
  a return law. Generic `dcc` and ADCC stay fail-closed. Default remains
  trailing Ledoit–Wolf plus the overlay. This does not invent high-frequency RV.
- This is optimizer covariance-object identity, not a live-performance claim.

## Day Wave 126 — Student-t DCC likelihood — 2026-09-18

- `dcc_student_t` is now a two-stage catalog estimator, not a named
  fail-closed stub. Stage 1 is univariate Student-t GARCH(1,1). Stage 2 QML
  uses the covariance Student-t correlation likelihood
  (`student_t_corr_nll`, \(ν>2\), scale \(((ν-2)/ν)R\)). The public matrix
  remains Engle one-step \(H_{t+1}\) on the trailing contiguous complete-case
  window. Params stamp `family=dcc_student_t`, `dist=student_t`,
  `spec=engle_2002_student_t_dcc`, and `nu`.
- This is not Gaussian DCC with a `dist` stamp: `dcc_student_t` does not call
  `dcc_gaussian`. `/models` lists it among implemented covariance estimators.
  `optimize_asof` / `/risk/portfolio` stay unwired (`unwired_optimizer_covariance:dcc_student_t`)
  so Student-t DCC cannot silently size as Gaussian DCC or Ledoit–Wolf.
  ADCC remains unspecified. This does not invent high-frequency RV.
- This is DCC likelihood honesty, not a live-performance claim.

## Day Wave 125 — DCC trailing complete-window honesty — 2026-09-18

- Gaussian DCC no longer listwise-deletes holes and concatenates the survivors
  as if they were adjacent observations. The estimation sample is the trailing
  contiguous complete-case window ending at the last row
  (`sample=trailing_complete_window`). An incomplete terminal row fails closed
  (`incomplete_terminal_row`) so advertised one-step \(H_{t+1}\) cannot drop
  asof \(z_t\). A hole that leaves fewer than 50 contiguous complete rows
  fails closed rather than stitching earlier dates onto the suffix.
- `optimize_asof` / `/risk/portfolio` share that contract, and trailing
  return pivots are sorted by `event_time` so reversed history cannot make
  the earliest bar look like \(z_t\). Ledoit–Wolf/sample still listwise-delete
  because they are not sequential likelihoods. This does not invent
  high-frequency RV or implement Student-t DCC / ADCC.
- This is DCC sample-path honesty, not a live-performance claim.

## Day Wave 124 — named Gaussian DCC optimize_asof path — 2026-09-18

- `optimize_asof` and `/risk/portfolio` now honor a named `optimizer.covariance`
  path. Default remains trailing Ledoit–Wolf (or sample when \(T\le N\)) plus
  the GARCH/RGARCH overlay. `dcc_gaussian` uses Engle (2002) one-step-ahead
  \(H_{t+1}\) on the same PIT-filtered trailing `ret_1` and does **not** scale
  that matrix by the market overlay (univariate GARCH already sits on
  \(D_{t+1}\)). Weights and `/risk/portfolio` stamp `covariance_estimator` /
  `covariance_object` / `covariance_spec`.
- Student-t DCC, ADCC, generic `dcc`, and catalog estimators that are not
  optimizer-wired (`sample` / `ewma` / `factor`) fail closed rather than
  silently running Ledoit–Wolf or Gaussian DCC. A short or failed Gaussian DCC
  fit fails closed rather than substituting the overlay path. `/models` lists
  `optimizer_covariance` separately from the estimator catalog.
- This is optimizer covariance-object identity, not a live-performance claim,
  high-frequency RV, or an implemented Student-t / ADCC likelihood.

## Day Wave 123 — one-step-ahead Gaussian DCC — 2026-09-18

- `dcc_gaussian` now returns Engle (2002) one-step-ahead \(H_{t+1}=D_{t+1}R_{t+1}D_{t+1}\), not in-sample last \(H_t\). \(Q_{t+1}\) uses the last standardized residual \(z_t\); \(D_{t+1}\) is the univariate GARCH one-step sigma. Params stamp `covariance_object=one_step_ahead` and `horizon=1`. A failed one-step univariate forecast fails closed.
- Stage-1 QML, named fail-closed Student-t / ADCC, and `/models` listing are unchanged. This does not wire DCC into `optimize_asof`, replace Ledoit–Wolf, or invent high-frequency RV.
- This is covariance-object honesty, not a live-performance claim or an implemented Student-t / ADCC likelihood.

## Day Wave 122 — named fail-closed DCC specs — 2026-09-18

- Student-t DCC and Cappiello–Engle–Sheppard ADCC are now named specs
  (`dcc_student_t`, `adcc`, `require_implemented_dcc_spec`) that fail closed
  with `unspecified_dcc_spec:*`. They must not silently run Engle (2002)
  Gaussian DCC. `dcc_gaussian` stamps `family=dcc_gaussian`,
  `spec=engle_2002_gaussian_dcc`, `dist=normal`, and `asymmetric=false`.
- `/models` lists `dcc_gaussian` among implemented covariance estimators and
  `dcc_student_t` / `adcc` under `covariance_unspecified`, so a generic `dcc`
  catalog entry cannot be read as a Student-t or asymmetric fit. This does
  not wire DCC into `optimize_asof`, replace Ledoit–Wolf, or invent
  high-frequency RV.
- This is covariance-spec honesty, not a live-performance claim or an
  implemented Student-t / ADCC likelihood.

## Day Wave 121 — ranker cache content digest — 2026-09-18

- Process-local `ranker_ridge.joblib` cache now keys by SHA-256 of the
  artifact bytes, not `(path, mtime)`. An in-place ranker replacement that
  preserves mtime cannot reuse a stale ridge for `forecast_asof` /
  `optimize_asof` alpha. Missing artifacts still return None so the
  momentum heuristic remains the explicit no-model path.
- GARCH/RGARCH spec caches already used this digest (Wave 113); ranker
  identity is now the same contract. This does not change `vol_20`, the
  GARCH/RGARCH overlay, or `max_predicted_vol`.
- This is ranker provenance and cache integrity, not a live-performance
  claim, high-frequency RV, Student-t DCC, or ADCC.

## Day Wave 120 — /risk/portfolio overlay identity — 2026-09-18

- `/risk/portfolio` now scales trailing Ledoit–Wolf with the same causal
  GARCH/RGARCH market overlay `optimize_asof` uses
  (`apply_market_variance_overlay_to_covariance`). The response stamps
  `market_risk_overlay` (`realized_garch` vs `garch` vs null) so unscaled
  sample risk cannot be mistaken for the optimizer's covariance object.
- A present Realized GARCH artifact still fail-closes on missing OHLC, the
  wrong `series_scope`, or a high-frequency RV claim rather than reporting
  unscaled Ledoit–Wolf. Per-name `vol_20` remains the impact/cost sigma.
- This is research-lab risk-object identity, not a live-performance claim,
  high-frequency RV, Student-t DCC, or ADCC.

## Day Wave 119 — optimizer covariance available_time PIT — 2026-09-18

- Trailing name-covariance in `optimize_asof` and `/risk/portfolio` now drops
  `available_time > asof` restatements of earlier `ret_1`. Wave 111 already
  closed that hole for the GARCH/RGARCH overlay scale; unpublished revisions
  can no longer move relative Ledoit–Wolf/sample risk after the overlay is
  PIT-clean. Null availability among usable `ret_1` rows fails closed.
  Frames without `available_time` keep the legacy event-time path.
- This is optimizer/API covariance observability, not a live-performance
  claim, high-frequency RV, Student-t DCC, or ADCC.

## Day Wave 118 — Realized GARCH forecast/optimize overlay — 2026-09-18

- `forecast_asof` and `optimize_asof` now consume the same causal Parkinson
  Realized GARCH market overlay that paper/backtest `check_order` prefers when
  `vol_realized_garch.joblib` is present. Precedence matches Wave 117:
  Realized GARCH, else return-only GARCH, else name-level `vol_20`.
- `MarketState.market_risk_overlay` stamps `realized_garch` versus `garch` so
  covariance scaling cannot be mistaken for the return-only overlay. A present
  Realized GARCH artifact still fail-closes on missing OHLC, the wrong
  `series_scope`, or a high-frequency RV claim; it does not fall back to GARCH
  because ranges are absent. Per-name `vol_20` remains the impact/cost sigma.
  The realized measure remains one-day Parkinson from daily OHLC
  (`intraday_realized_variance=false`).
- This is a research-lab risk-object identity fix, not a live-performance or
  high-frequency realized-variance claim.

## Day Wave 117 — Realized GARCH check_order overlay — 2026-09-18

- Paper and backtest `check_order` now consume the causal date-level
  Hansen–Huang–Shek Realized GARCH one-step market sigma when
  `vol_realized_garch.joblib` is present. The realized measure remains
  one-day Parkinson from daily OHLC (`intraday_realized_variance=false`).
- Precedence is explicit: Realized GARCH, else return-only GARCH, else
  name-level `vol_20`. A present Realized GARCH artifact still fail-closes
  on missing OHLC, the wrong `series_scope`, or a high-frequency RV claim;
  it does not silently fall back to GARCH because ranges are absent.
  Metrics stamp `realized_garch_risk_overlay_dates` separately from
  `garch_risk_overlay_dates`. `forecast_asof` / `optimize_asof` still use
  `vol_garch.joblib`. Per-name `vol_20` remains the impact/cost sigma.
- This is a research-lab risk overlay, not a live-performance or
  high-frequency realized-variance claim.

## Day Wave 116 — DCC stage-1 arch GARCH — 2026-09-18

- Gaussian DCC(1,1) stage 1 now fits univariate GARCH(1,1) via `arch` /
  `GARCHVol` on each name's decimal returns and uses those standardized
  residuals for stage-2 QML. RiskMetrics EWMA is no longer a silent substitute
  for Engle (2002) stage-1 innovations (ADR-004 / MATH_SPEC).
- Failed, short, zero-variance, or fallback univariate fits fail closed.
  Params stamp `stage1=garch`. Last \(H_t\) still uses in-sample GARCH sigma
  with PSD repair; this does not replace `vol_20`, the GARCH market overlay,
  or `max_predicted_vol`. No Student-t DCC or ADCC.
- This is a research-lab covariance estimator, not a live-performance claim.

## Day Wave 115 — Realized GARCH on daily Parkinson — 2026-09-18

- Causal `RealizedGARCHVol` is Hansen–Huang–Shek log-linear Realized GARCH(1,1)
  with a Gaussian return law. The realized measure is one-day Parkinson variance
  from daily OHLC (`high_split_adjusted`/`low_split_adjusted`, else `high`/`low`).
  Diagnostics stamp `realized_measure=parkinson_daily_ohlc` and
  `intraday_realized_variance=false`. Close-to-close \(r_t^2\) is not inferred,
  and the 20-day `vol_parkinson` feature is not the measure.
- `realized_garch_market_forecast_asof` clones `vol_realized_garch.joblib` and
  refits on date-level equal-weight `ret_1` paired with same-name Parkinson.
  Origin-bar OHLC cannot enter the fit. Unpublished `available_time` restatements
  cannot move the overlay. This namespace does not replace `vol_garch.joblib`,
  `vol_20`, or `max_predicted_vol`. Missing OHLC fail-closed. Walk-forward
  `train_volatility(..., "realized_garch")` uses the same overlap-aware QLIKE
  and one-step density contract as the return-only GARCH overlay.
- This is a research-lab range-based Realized GARCH, not a live-performance or
  high-frequency realized-variance claim.

## Day Wave 114 — name-level GARCH walk-forward QLIKE/density — 2026-09-18

- Per-security GARCH now has a walk-forward scorer that is not the date-level
  market overlay. `garch_name_walk_forward` refits each name on that name's
  strictly prior PIT-observable `ret_1` and scores QLIKE against the name's
  `future_realized_var_h`. One-step log-score, CRPS, and PIT KS use that
  name's origin `ret_1`, never the equal-weight cross-section and never
  `future_realized_var_h`. Hansen–Lunde stride is applied independently per
  name. Metrics stamp `scoring_scope=security_level_ret_1`.
- Date-level `overlap_aware_qlike` still fail-closes when forecasts disagree
  within a date, so name-level forecasts cannot be silently collapsed into the
  market QLIKE. Duplicate `(security_id, event_time)` keys, missing origin
  `ret_1`, and empty prior history fail closed. `available_time` restatements
  cannot enter a name's expanding fit. This does not replace `vol_20`,
  `max_predicted_vol`, or `train_volatility` overlay scores.
- This is a research-lab per-name scoring namespace, not a live-performance
  or realized-GARCH claim.

## Day Wave 113 — GARCH spec-cache content digest — 2026-09-18

- Process-local GARCH spec and as-of caches now key `vol_garch.joblib` by
  SHA-256 of the artifact bytes, not `(path, mtime)` or a partial
  `(p, q, dist, vol, min_obs)` tuple. An in-place spec replacement that
  preserves mtime cannot reuse a stale variance family, mean, or power for
  `garch_market_forecast_asof` or `garch_name_forecasts_asof`.
- Mean/power are cloned into as-of refits but were previously absent from the
  as-of key, so a same-mtime Constant→Zero rewrite could keep the old overlay.
  History digest (Wave 110) is unchanged. Missing artifacts still return
  empty/None; wrong `series_scope` still fails closed.
- This is overlay provenance and cache integrity, not a live-performance,
  realized-GARCH, or name-level walk-forward scoring claim.

## Day Wave 112 — per-security GARCH namespace — 2026-09-18

- Causal `garch_name_forecasts_asof` clones the date-level `vol_garch.joblib`
  specification and refits each name on that name's strictly prior `ret_1`.
  Output `series_scope` is `security_level_ret_1`, so these forecasts cannot be
  mistaken for the equal-weight market overlay. They do not replace `vol_20`
  or `max_predicted_vol`; `forecast_asof` / `check_order` stay on the date-level
  overlay.
- PIT universe membership and `available_time <= asof` follow the overlay
  contract. Duplicate `(security_id, event_time)` keys and blank ids fail
  closed. Missing artifacts return an empty mapping; a present artifact with
  the wrong scope fails closed. Explicit ids with no strictly-prior history
  fail closed; implicit scans omit names without history.
- This is a research-lab per-name variance namespace, not a live-performance
  or realized-GARCH claim.

## Day Wave 111 — GARCH overlay available_time PIT — 2026-09-18

- Causal GARCH market overlay and walk-forward OOS fits now keep only
  `ret_1` rows with `event_time < asof` **and** `available_time <= asof` when
  that column is present. A late restatement of an earlier date cannot move
  `max_predicted_vol`, `optimize_asof` covariance scaling, or origin-level
  QLIKE/density fits. Null `available_time` among otherwise usable prior rows
  fails closed. Frames without the column keep the legacy event-time path.
- Density scoring still uses the origin's date-level `ret_1` outcome
  (evaluation vintage). This is overlay/walk-forward PIT honesty, not a
  live-performance, per-name GARCH, or realized-GARCH claim.

## Day Wave 110 — GARCH overlay PIT membership + history digest — 2026-09-18

- Causal `garch_market_forecast_asof` now restricts the date-level equal-weight
  `ret_1` series to the current `silver/universe.parquet` when that artifact
  exists. Paper/backtest execution bars may still carry ineligible ADV/listing
  names for marks; those names cannot move `max_predicted_vol`. Empty universe
  or a present universe without `security_id` fails closed. Missing universe
  keeps the caller frame for legacy fixtures.
- The process-local as-of cache is keyed by a SHA-256 digest of the full
  causal date/return path. A last-date / last-value / length fingerprint can
  reuse a stale overlay after an earlier membership or return rewrite that
  leaves the terminal mean unchanged.
- This is overlay provenance and cache integrity, not a live-performance,
  per-name GARCH, or realized-GARCH claim.

## Day Wave 109 — gold panel load-time PIT membership — 2026-09-18

- Training/forecast `panel()` now fail-closes unless persisted gold feature and
  label keys are a subset of the current `silver/universe.parquet` membership.
  Wave 108 inner-joined at materialization; a later universe shrink or
  hand-edited gold file can no longer silently train, rank, or score names the
  PIT universe rejected. Missing universe beside gold fails closed rather than
  re-ingesting silver under stale gold.
- The process-local gold cache key includes the universe parquet digest, so an
  in-place membership replacement invalidates cached training inputs even when
  gold bytes are unchanged. Extra membership rows remain allowed (incomplete
  gold is not leakage). Direct feature callers that omit membership keep the
  legacy unfiltered frame.
- This is decision-universe honesty on the consumption path, not a
  live-performance, per-name GARCH, or realized-GARCH claim.

## Day Wave 108 — gold/CS consume PIT universe membership — 2026-09-18

- `build_gold` now inner-joins features and labels to `silver/universe.parquet`
  on `(security_id, event_time)`. Name-level rolling history still uses the
  full silver panel, but ineligible ADV/listing/history names cannot remain in
  gold or move cross-sectional ranks, market aggregates, or idio-label means.
  Empty, duplicate-keyed, or schema-invalid membership fails closed instead of
  silently training on unfiltered silver. Paper/backtest CLI feature panels
  use the same membership.
- This is decision-universe honesty after Wave 107's listing-action artifact,
  not a live-performance, per-name GARCH, or realized-GARCH claim.

## Day Wave 107 — delist/ticker_change on production panel — 2026-09-18

- File-adapter `delist` and `ticker_change` events now apply on the production
  silver path instead of dying after the parquet contract gate. Ingest overlays
  PIT-visible ticker changes onto `symbol`, drops leftover bars after a knowable
  delist (last listed session kept), and persists `silver/universe.parquet`.
  `membership_asof` consumes the same actions: late announcements cannot rewrite
  pre-availability membership, and `include_delisted=False` excludes a name as
  soon as the announcement is available.
- `ticker_change` fails closed on missing or blank `new_ticker`. Doctor now
  requires the universe artifact. This is bronze/silver listing integrity, not a
  live-performance, per-name GARCH, or realized-GARCH claim.

## Day Wave 106 — APARCH/FIGARCH variance specs — 2026-09-18

- Causal `GARCHVol` now accepts APARCH and FIGARCH alongside GARCH/EGARCH/GJR.
  Config `garch_vol` enumerates the same set. APARCH uses the asymmetric power
  specification (`o=1`); FIGARCH allows only \(p,q\in\{0,1\}\) and rejects
  higher orders fail-closed.
- Fit gates stay specification-appropriate: GARCH/GJR/APARCH use the symmetric
  persistence check (GJR still adds \(\frac12\gamma\)); APARCH requires
  \(\delta>0\); FIGARCH requires \(0<d<1\) rather than \(\alpha+\beta\).
  Multi-step APARCH forecasts use seeded simulation, matching EGARCH, because
  `arch` has no analytic path beyond one step. Density/QLIKE consumers still
  clone the persisted spec and refit on strictly prior `ret_1`.
- This is a research-lab variance-family expansion, not a live-performance,
  per-name GARCH, or realized-GARCH claim.

## Day Wave 105 — GARCH one-step density calibration — 2026-09-18

- Walk-forward GARCH now scores the one-step date-level equal-weight `ret_1`
  predictive density in addition to overlap-aware *h*-step QLIKE. Log-score,
  CRPS, and PIT KS use the origin law fitted on strictly prior returns; the
  density target is never `future_realized_var_h` and never a Gaussian
  approximation to `cumulative_variance`.
- Gaussian CRPS is closed-form; t/skew-t CRPS is quantile-Riemann from the
  fitted `arch` ppf. Student-t log-score uses the standardized GARCH
  innovation, not textbook location-scale *t*. Fallback fits remain explicitly
  Gaussian. Missing origin `ret_1` fails closed.
- This is scoring honesty for the advertised predictive law, not a
  live-performance, per-name GARCH, or APARCH/FIGARCH claim.

## Day Wave 104 — overlap-aware multi-horizon vol scoring — 2026-09-18

- Date-level GARCH walk-forward QLIKE no longer concatenates name-level rows
  onto a market forecast. Primary `qlike` is the equal-weight cross-section
  realized variance on Hansen–Lunde nonoverlapping origins (session stride =
  label horizon). Overlapping-date QLIKE remains a diagnostic. Within-date
  forecast disagreement fails closed.
- `bench_volatility` collapses holdout observations to one date before QLIKE
  and Diebold–Mariano, and uses Hansen–Hodrick HAC lags of at least \(h-1\).
  Nonoverlapping companion keys are stamped. This is scoring honesty, not a
  live-performance or per-name GARCH claim.

## Day Wave 103 — check_order GARCH market overlay — 2026-09-18

- Paper and backtest `check_order` now compare `max_predicted_vol` to the causal
  date-level GARCH one-step market sigma when `vol_garch.joblib` is present.
  Callers still pass per-name `vol_20` as `predicted_vol`; the optional
  `market_predicted_vol` overlay replaces it for the vol gate only.
- Impact/cost models keep name-level `vol_20`. Missing artifacts and origins
  with no strictly-prior `ret_1` fall back to name vol. A present artifact with
  the wrong `series_scope`, or with no `ret_1` on the execution panel, fails
  closed. Late returns on or after the decision origin cannot change the gate.
- Regression coverage includes overlay-vs-name unit edges, broker passthrough,
  backtest admit/reject identity, late-return non-leakage, wrong-scope and
  missing-`ret_1` fail-closed, and a paper-loop overlay admit path. This is
  simulated/paper integrity only; it is not a live-performance or per-name
  GARCH claim.

## Day Wave 102 — causal GARCH market overlay — 2026-09-18

- `forecast_asof` now consumes `vol_garch.joblib` as a date-level equal-weight
  market overlay: it clones the artifact specification and refits on `ret_1`
  dates strictly before the decision origin. Persisted full-sample parameters
  cannot leak post-asof returns into a historical as-of.
- Per-security `vol_20` remains the name-level volatility field. The GARCH
  one-step variance is stamped on `MarketState` and forecast diagnostics with
  `series_scope=date_level_equal_weight_cross_section`. A present artifact with
  any other scope fails closed.
- `optimize_asof` scales trailing name-covariance so equal-weight market
  variance matches the causal GARCH level (PSD-repaired). This is a research-lab
  risk overlay, not a live-performance or asset-specific GARCH claim.
- Regression coverage includes overlay identity, name-vol preservation, late
  return non-leakage, full-sample fit non-reuse, wrong-scope fail-closed, and
  persisted overlay columns.

## Day Wave 101 — security-master PIT/schema fail-closed — 2026-09-18

- File-adapter security master now requires identity + PIT columns (`security_id`, `ticker`, `valid_from`/`valid_to`, `available_time`, `ingested_time`, `source`, `revision_id`). Blank identity/source, non-temporal timestamps, inverted windows, and duplicate `(security_id, valid_from)` or `(ticker, valid_from)` rows fail closed instead of silently joining.
- Lookup and universe/ingest consumers respect `available_time`: late restatements cannot rewrite pre-availability ticker identity or sector attributes, and vintaged masters no longer explode the bar panel. Direct callers without `available_time` retain the legacy valid-window lookup.
- Regression coverage includes parquet contract fixtures, closed-form restatement edges, and membership/ingest PIT joins. This is bronze-input integrity only; it is not live-data or live-performance evidence.

## Day Wave 98 — corporate-action contract fail-closed — 2026-09-18

- File-adapter corporate actions now require the DATA_CONTRACTS schema: `action_type` in `{split, cash_dividend, special_dividend, delist, ticker_change}`, PIT timestamps, and non-blank source/security identity. Unknown or blank types, non-positive/non-finite split factors, negative/non-finite dividend amounts, and duplicate `(security_id, event_time, action_type)` rows fail closed instead of silently no-oping or exploding the bar panel.
- Adjustment still respects `available_time`: late-arriving splits/dividends cannot rewrite pre-availability history. Same-day cash and special dividends are summed onto one bar instead of duplicating rows. Direct callers with a missing `action_type` column remain a documented no-op.
- Regression coverage includes parquet PIT/contract fixtures and closed-form split/dividend edges. This is bronze-input integrity only; it is not live-data or live-performance evidence.

## Day Wave 95 — canonical lineage-bound research inputs — 2026-09-17

- Research provenance now fingerprints materialized frames using canonical sorted columns, schema dtypes, and a sorted multiset of canonicalized rows. The digest is invariant to dataframe row/column ordering, counts duplicate rows, and normalizes non-finite values to null.
- The research receipt's `dataset_content_sha256` and Northset silver-frame content digest use this canonical representation instead of implementation-specific dataframe hash seeds.
- The process-local gold panel cache is keyed by SHA-256 digests of the feature and label parquet bytes, not mtimes alone. In-place replacements therefore invalidate cached training inputs even when timestamps or file sizes are reused.
- Added regression coverage for ordering invariance, duplicate preservation, byte-level cache invalidation, and hash-seed-independent execution. This is reproducibility and cache-integrity evidence only; it does not establish live alpha or live execution readiness.

## Day Wave 96 — sparse-label endpoint purging — 2026-09-17

- `build_labels` now persists exact observed-row `label_end_time_{h}` metadata for each forward horizon, preserving causal endpoints for securities with asynchronous or missing observations.
- Purging accepts those endpoints explicitly; walk-forward date folds conservatively use the latest endpoint observed on each decision date, including the short-sample fallback, and ranking, distribution, volatility, and tail training pass aligned endpoint arrays into every fold.
- Regression coverage verifies endpoint alignment and sparse-label purge behavior. This closes a leakage class in validation; it is not live-data or live-performance evidence.

## Day Wave 97 — bounded paper valuation under sparse marks — 2026-09-17

- Paper NAV, borrow, exposure, cash identity, and persisted snapshots now use valuation marks that carry forward missing marks only for the configured `risk_gate.stale_price_bars` bound.
- Execution remains fresh-mark-only: stale prices cannot create new orders, while held positions remain valuable across short data gaps and fail closed after the bounded age expires.
- Pre-trade valuation uses fresh execution marks plus bounded prior marks; close/total-return marks are applied only after fills, preventing same-bar close leakage under `NEXT_OPEN`.
- Per-security mark ages are persisted in `broker_state.json` and restored with validation, so split/resumed paper runs cannot silently reset stale-age accounting; legacy `mark_ages: null` is treated as an empty age map for compatibility.
- A restored mark without historical age is deliberately bootstrapped at age `-1`, becoming age `0` on the first resumed bar; this grants one explicit baseline observation opportunity even when `stale_price_bars=0`, and is covered by regression tests rather than presented as a live-data guarantee.
- This is paper/simulation integrity only, not live execution evidence.

## Day Wave 94 — point-in-time aggregate eligibility — 2026-09-16

- Cross-sectional winsorization, robust z-scores, ranks, percentiles, and sector transforms now exclude rows whose `available_time` is later than their `event_time`; late rows are returned as null rather than contaminating the decision-time universe.
- Market return, volatility, dispersion, breadth, mean-return, and sector aggregates now use the same availability filter; benchmark values are also masked when either the benchmark or output row is unavailable.
- Added regression fixtures proving a late-arriving asset cannot move same-day aggregate values, ranks, denominators, or benchmark features. Frames without `available_time` retain the legacy behavior for compatibility.
- This is a PIT-integrity improvement only; it does not create live-data evidence or live-performance claims.

## Day Wave 93 — cumulative receipt-history resume and date-correct ranker DM

- `SimulatedBroker` now persists JSON-safe order receipts, rejection metadata, fills, and slots; resume reconstructs validated `Order`/`Fill` models and preserves cumulative receipt counts.
- Legacy broker state without history remains readable through explicit counter baselines; modern state fails closed on malformed history or count mismatches.
- Ledger schema validation covers receipt-history shape and count consistency, with an uninterrupted-vs-two-leg regression.
- Ranker benchmark Diebold–Mariano contrasts align IC losses by common date keys rather than positional truncation, avoiding invalid comparisons when feature sets have different missing-date patterns.
- All evidence remains simulated/paper-only; no live broker connectivity or live-performance claim is introduced.

## Day Wave 92 — deterministic and append-safe paper resume

- Target panels now reject duplicate `(event_time, security_id)` keys, normalize security IDs, and construct event mappings in stable sorted order.
- Backtest and simulated-broker exposure/target reductions no longer depend on set/hash iteration order.
- Paper resume loads all existing orders, equity, shadow equity, positions, and cash-ledger rows before appending the next leg; `flush()` is the single durable table writer.
- Shadow equity is persisted through the same ledger path, and returned order frames retain cumulative fill-receipt history across resumed legs.
- Regression coverage verifies two-leg row preservation and ledger-schema validity. This is still simulated/paper-only evidence; it does not establish live execution readiness.

Scope: Artificial Hedge / dipcatcher **research lab**. Vendor market-data and live broker fills are **out of scope**. Day Waves 1–18 (panel honesty, MinTRL + Acerbi–Székely, Fissler–Ziegel FZ0, e-process DM, MinTRL bench smoke, CPCV/PBO residual honesty, closed-form/empirical CRPS, closed-form CRPS bench wiring + DM, volatility-bench e-process DM + paper/shadow honesty smoke + Bernoulli miss-clip docs, dual honesty catalogs research-vs-analytics_export, distribution-bench CRPS e-process DM) landed 2026-09-16.

## Already have (overnight Waves 1–43 baseline)

| Area | Status (from overnight progress / memory) |
|------|-------------------------------------------|
| Conformal / CRC / online CRC / weighted / localized / rank / portfolio conformal | Present + many extremes fixtures |
| Jackknife+ / CV+ edges | Present |
| CPCV purge/embargo + PBO/TrialLedger/DSR/PSR | Present + Day Wave 6 residual edges / known PBO fraction |
| CRPS / pinball / PIT / distribution families | Present; Day Wave 7–8 CRPS+DM; Day Wave 18 `e_dm_crps_*`; Day Wave 19 scaled `dm_crps_scaled_*` / `e_dm_crps_scaled_*`; soft CRPS e-process verify companion (Wave19 harden) |
| Kupiec POF + Christoffersen CC | Present |
| Almgren–Chriss trajectory / ES_ac | Present + extremes |
| E-values / Ville | Present + edges; Day Wave 4 loss-diff e-process |
| DM / Jobson–Korkie–Memmel / bootstrap CI | Helpers present; pairwise DM may need expansion |
| Research agent catalogs + forbidden metrics | Present (Wave 32) |
| Paper/shadow multi-challenger, analytics_export fail-closed | Phase 17 largely done |
| PERF panel/optimize/wrappee caches | Phase 18 largely done (wave:39 bench) |
| ~1310+ `pytest -m 'not network'` green (Wave 42+) | Baseline to preserve |

## GARCH-family upgrade — 2026-09-17

The volatility path now has a research-grade causal contract rather than a claim of
state-of-the-art forecasting: `GARCHVol` fits historical decimal returns, applies
explicit percent scaling at the `arch` boundary, validates convergence/finite
parameters/specification-appropriate persistence, and records fail-closed fallback
reasons. Direct protocol callers must pass `returns=` explicitly; the compatible `y`
argument is never implicitly treated as a likelihood series because it may be a
forward variance label. Its `forecast(horizon=...)` API returns decimal per-bar
variance, sigma, cumulative variance, and optional Student-t/skew-t predictive
quantiles; `pit()` provides bounded probability-integral-transform diagnostics. The
walk-forward GARCH path rebuilds an expanding causal history from `ret_1` at each
test-date origin, and uses `future_realized_var_h` only for horizon-matched variance
QLIKE evaluation. Return history is aggregated from the full feature panel before
supervised design-matrix label filtering, so the latest finite returns remain in the
persisted fit even when their forward labels are structurally unavailable; each OOS
origin still selects only return dates strictly earlier than that origin. Because the
current implementation pools securities into a date-level equal-weight return series
on the PIT-universe gold panel (Wave 108). Cached gold loaded by `panel()`
fail-closes if its keys sit outside the current universe artifact (Wave 109).
Its saved artifact and diagnostics now declare `series_scope` explicitly rather than
silently presenting the forecast as security-specific volatility.

This does **not** establish superiority, live deployability, or true intraday realized
volatility. The current panel is cross-sectional. `forecast_asof` / `optimize_asof` consume the saved GARCH artifact as a
date-level equal-weight market overlay (Wave 102), and paper/backtest
`check_order` uses that same causal one-step market sigma for
`max_predicted_vol` (Wave 103). Walk-forward QLIKE and the volatility bench
now score that overlay on date-level, overlap-aware *h*-step origins
(Wave 104). Walk-forward also scores the one-step date-level `ret_1`
predictive density with log-score, CRPS, and PIT KS (Wave 105). APARCH and
FIGARCH are first-class `GARCHVol` variance specs (Wave 106). The overlay
series is PIT-universe restricted when membership exists, and the as-of
cache digests the full causal history (Wave 110). Overlay and walk-forward
fits also drop unpublished `available_time > asof` restatements (Wave 111).
Per-security causal forecasts exist as `garch_name_forecasts_asof` with
`series_scope=security_level_ret_1` (Wave 112) and do not replace the market
overlay, `vol_20`, or `max_predicted_vol`. Spec and as-of caches key the
joblib by content digest rather than mtime (Wave 113). Walk-forward QLIKE
and one-step density now exist for that name-level namespace
(`garch_name_walk_forward`, Wave 114) with per-name Hansen–Lunde stride;
they do not replace date-level overlay scores. Realized-GARCH now exists as a
Parkinson-daily-OHLC namespace (Wave 115) and still does not replace the
return-only overlay, `vol_20`, or `max_predicted_vol`. DCC stage-1 now uses
univariate `arch` GARCH rather than EWMA residuals (Wave 116); a failed
univariate fit fails closed. Paper/backtest `check_order` now uses the
Parkinson Realized GARCH overlay when `vol_realized_garch.joblib` is present
(Wave 117). `forecast_asof` / `optimize_asof` consume that same overlay for
market variance and covariance scaling, stamped as
`market_risk_overlay=realized_garch` (Wave 118), else the return-only GARCH
overlay, else `vol_20`. Trailing name-covariance in `optimize_asof` and
`/risk/portfolio` now refuses unpublished `available_time > asof`
restatements of earlier `ret_1` (Wave 119), matching the overlay PIT
contract. `/risk/portfolio` now also applies that same overlay scale and
stamps `market_risk_overlay` (Wave 120), so the research diagnostic cannot
report unscaled Ledoit–Wolf while the optimizer sized on GARCH/RGARCH.
The ridge ranker cache now keys `ranker_ridge.joblib` by SHA-256 bytes
rather than mtime (Wave 121), matching the GARCH spec-cache contract so a
same-mtime rewrite cannot reuse stale alpha. Named `dcc_student_t` and
`adcc` entry points fail closed rather than masquerading as Gaussian DCC
(Wave 122); `/models` lists `dcc_gaussian` as the implemented spec.
Gaussian DCC now returns one-step-ahead \(H_{t+1}\) rather than in-sample
last \(H_t\) (Wave 123), stamped `covariance_object=one_step_ahead`.
Named `optimizer.covariance=dcc_gaussian` sizes `optimize_asof` /
`/risk/portfolio` on that matrix without the GARCH/RGARCH overlay
(Wave 124); default remains Ledoit–Wolf plus overlay. The DCC sample is
the trailing contiguous complete-case window; incomplete asof rows and
stitched holes fail closed (Wave 125). Student-t DCC is a catalog estimator
with a real multivariate-t stage-2 likelihood (Wave 126); named
`optimizer.covariance=dcc_student_t` now sizes `optimize_asof` /
`/risk/portfolio` on that matrix without overlay (Wave 127) rather than
silently substituting Gaussian DCC or Ledoit–Wolf. Scalar Cappiello–Engle–
Sheppard ADCC is a catalog estimator with a real CES stage-2 likelihood
(Wave 128); named `optimizer.covariance=adcc` now sizes `optimize_asof` /
`/risk/portfolio` on that matrix without overlay (Wave 129) rather than
silently substituting Gaussian DCC, Student-t DCC, or Ledoit–Wolf.
Named `optimizer.covariance=ewma` now sizes `optimize_asof` /
`/risk/portfolio` on one-step RiskMetrics \(H_{t+1}\) without overlay
(Wave 130) rather than listwise-deleting holes, dropping asof \(r_t\), or
silently substituting Ledoit–Wolf or DCC.
Named `optimizer.covariance=oas` now sizes `optimize_asof` /
`/risk/portfolio` on trailing Chen OAS plus the overlay (Wave 131) rather
than silently substituting Ledoit–Wolf, sample, EWMA, or DCC, and rather
than switching to sample when \(T\le N\).
Named `optimizer.covariance=sample` now sizes `optimize_asof` /
`/risk/portfolio` on trailing unbiased sample covariance plus the overlay
(Wave 132) rather than silently substituting Ledoit–Wolf, OAS, EWMA, or
DCC, and rather than switching to Ledoit–Wolf when \(T>N\).
Bollerslev (1990) CCC is a catalog estimator with constant \(R\) and
one-step \(D_{t+1}\) (Wave 133); named `optimizer.covariance=ccc` now sizes
`optimize_asof` / `/risk/portfolio` on that matrix without overlay
(Wave 134) rather than silently substituting Gaussian DCC, Student-t DCC,
scalar ADCC, or Ledoit–Wolf.
Diagonal CES AG-DCC is a catalog estimator with matrix diagonals \(A,B,G\)
(Wave 135); named `optimizer.covariance=agdcc` now sizes
`optimize_asof` / `/risk/portfolio` on that matrix without overlay
(Wave 136) rather than silently substituting scalar ADCC, Gaussian DCC,
CCC, or Ledoit–Wolf. Unrestricted full-matrix AG-DCC is a catalog
estimator (Wave 137); named `optimizer.covariance=agdcc_full` now sizes
`optimize_asof` / `/risk/portfolio` on that matrix without overlay
(Wave 138) rather than silently substituting diagonal AG-DCC, scalar
ADCC, Gaussian DCC, CCC, or Ledoit–Wolf. Default `optimizer.covariance=ledoit_wolf`
now stays Ledoit–Wolf 2004 when \(T\le N\) rather than silently switching
to unbiased sample (Wave 139); the named `sample` path is unchanged.
Named `optimizer.covariance=ledoit_wolf_nonlinear` now sizes
`optimize_asof` / `/risk/portfolio` on trailing analytical 2020 nonlinear
shrinkage plus the overlay (Wave 140) rather than silently substituting
2004 linear Ledoit–Wolf, OAS, sample, EWMA, or DCC, and rather than
switching to sample or 2004 when \(T\le N\).
High-frequency RV remains unavailable. Factor stays unwired.

## Volatility baseline contract — 2026-09-17

The rolling and EWMA baselines now use their intended supervised features rather than
implicitly reading the final `vol_of_vol` column. Their predictions are sigma-like, so
walk-forward evaluation squares them before calling variance QLIKE against
`future_realized_var_h`; this keeps the response and forecast dimensionally aligned.
`vol_of_vol` remains a separate feature and cannot substitute for either baseline.
The HAR estimator likewise consumes the declared production feature matrix directly;
its forward realized-variance target is response-only, and its artifact metadata is
versioned as the v2 supervised input contract.

## Missing / thin (highest-ROI Day Wave 1 ship targets)

1. **Closed-form fixtures** for any remaining thin conformal/CRC/e-value/CPCV paths (exact coverage or known λ edges).
2. **CPCV/PBO fail-closed** — DONE Day Wave 6 (known PBO fraction fixture; shape/NaN edges ≠ silent 0.0; CPCV aggressive-purge fold-count honesty).
3. **Distribution scoring vs MATH_SPEC**: CRPS empirical/Gaussian closed forms — DONE Day Wave 7; bench wiring `crps_gaussian_closed` / `dm_crps_*` — DONE Day Wave 8; pinball empty/mismatch already edged; PIT KS present + Day Wave 7 spiked-vs-uniform fixture.
4. **Acerbi–Székely Z1/Z2** — DONE Day Wave 2 (`acerbi_szekely_z1`/`z2`, edges + known Z1≠Z2 fixture; research-only).
4b. **Bailey–LdP MinTRL** — DONE Day Wave 2 (`min_track_record_length`, Gaussian closed-form + honest NaN edges).
5. **Research agent**: ensure all scorecard families run; forbidden Sharpe/pnl keys stay out of lab headlines (extend Wave 32/40 benches if any family drifted).

## Out of scope (vendor / live)

- Live broker fills, Alpaca/vendor market-data pulls, forged `live_pnl_claim=true`
- Cursor cloud agents / commits / pushes
- Parallel causal with `w_prev` unsafe paths
- Advertising SYNTHETIC Sharpe as live performance

## Day Wave 2 DONE (2026-09-16)

- MinTRL hardened + Gaussian closed-form fixtures; exported from `metrics`
- Acerbi–Székely Z1/Z2 extreme edges + distinct known-ratio fixture; docs cited
- No fake live Sharpe; research-diagnostic labels only

## Day Wave 3 DONE (2026-09-16)

- Fissler–Ziegel FZ0 joint VaR/ES scoring (`fissler_ziegel_loss` / `mean_fissler_ziegel`); closed-form + edges
- Thin research hook `fissler_ziegel_mean` on `var_backtest_hooks` when ES supplied
- MATH_SPEC + RESEARCH_REFERENCES (Fissler–Ziegel / Nolde–Ziegel) cited; no live capital claim

## Day Wave 4 DONE (2026-09-16)

- Choe–Ramdas-style anytime-valid e-process on forecast loss differentials (`e_process_loss_diff` / `e_process_dm`); Ville via existing threshold helper
- Closed-form / RNG fixtures + edges in `tests/unit/test_eprocess_dm.py`
- Thin research flag `include_e_process` on `pairwise_diebold_mariano` (optional fields only)
- MATH_SPEC + RESEARCH_REFERENCES linked; no live capital / Sharpe claim

## Day Wave 5 DONE (2026-09-16)

- Thin MinTRL research smoke: `min_trl_from_returns` → nested `book_diagnostics["min_trl"]` with keys `min_track_record_length` / `min_trl` / `track_record_bars`
- Gaussian fixture smoke + forbidden-key hygiene (`family_blob_forbidden_metrics_absent`); edges remain in `test_overfitting_edges.py`
- OT / multivariate conformal **skipped** (no ADR greenlight)
- No fake live Sharpe / live capital claim

## Day Wave 23 DONE (2026-09-16)

- **Promotion receipt fail-closed hardening:** `validate_candidate` now treats a
  missing research receipt as invalid rather than defaulting it to valid. A
  non-synthetic candidate cannot promote on metrics, causal-panel, and fold
  evidence alone; it must be bound to a verified immutable research receipt.
  Regression covered by `test_validate_never_promotes_without_research_receipt`.

## Day Wave 6 DONE (2026-09-16)

- PBO residual honesty: known closed-form fraction (PBO=0.5) + valid 0/1; invalid 1×N / N×1 / mismatch / all-NaN / partial non-finite → NaN (never silent 0.0) in `tests/unit/test_pbo_edges.py`
- CPCV: aggressive purge/embargo may yield fewer (or zero) folds than `C(n,k)` — documented + tested; empty groups / bad n_test already ValueError
- MATH_SPEC CSCV/PBO note updated; OT conformal **skipped**
- No fake live Sharpe / live capital claim

## Day Wave 7 DONE (2026-09-16)

- Closed-form Gaussian CRPS (`crps_gaussian` / `mean_crps_gaussian`) + empirical ensemble CRPS (`crps_empirical`) in `metrics/scoring.py`
- Hand-check fixtures: N(0,1)@0 → (√2−1)/√π; 2-point ensemble CRPS=0.5; scale homogeneity; bad-σ → NaN (never silent 0.0)
- Thin PIT: reuse `pit_ks` — uniform fixture retained + spiked U-shape rejects Uniform(0,1); no new API / mean-PIT helper skipped
- MATH_SPEC + RESEARCH_REFERENCES (Gneiting–Raftery) updated; OT / multivariate conformal **skipped**
- No fake live Sharpe / live capital claim

## Day Wave 8 DONE (2026-09-16)

- Distribution bench wires research-only `crps_gaussian_closed` (+ optional `crps_scaled_gaussian_closed`) beside quantile Riemann approx keys
- Diebold–Mariano on per-obs quantile-CRPS losses: `dm_crps_preferred` / `dm_crps_p` / `dm_crps_stat` (gaussian vs empirical)
- Thin smoke: `tests/unit/test_bench_crps_closed.py`; forbidden-key hygiene retained
- OT / multivariate conformal **skipped**; PERF rebench **skipped**
- No fake live Sharpe / live capital claim

## Day Wave 9 DONE (2026-09-16)

- **Paper/shadow day-grind honesty smoke:** `tests/unit/test_day_paper_shadow_honesty.py` — minimal `run_paper_loop` + shadow; broker_state `allow_capital=False` / cash=0 / n_fills=0; `live_pnl_claim=false` on metrics + analytics_export (+ disk JSON); slim challenger blob `family_blob_forbidden_metrics_absent`
- Volatility bench wires research-only `e_dm_final` / `e_dm_reject` / `e_dm_n` via Wave4 `e_process_dm` on ewma vs rolling losses (beside existing DM); thin smoke `tests/unit/test_bench_volatility_eprocess.py`
- Bernoulli e-process honesty: docstring notes silent miss clip to [0,1] in `_step_e`; edge `test_miss_out_of_range_clips_like_soft_coverage` (no public API change; soft coverage preserved)
- Conformal residual **skipped** (no concrete untested ValueError/NaN fixture found in skim)
- OT / multivariate conformal **skipped**; PERF rebench **skipped**
- No fake live Sharpe / live capital claim

## Day Wave 10 DONE (2026-09-16)

- **`/ready` missing-manifest fail-closed harden:** `resolve_allowed_config_path` always `.resolve()`s `_CONFIGS_DIR` before `relative_to` (macOS `/var`→`/private/var` flake → spurious 400 without `checks`); test isolates repo lake, asserts `report["data_manifest"]=="missing"`, plus unresolved-configs-dir regression
- **Dual honesty catalogs:** research family/scorecard blobs keep `FORBIDDEN_RESEARCH_METRIC_KEYS` / `family_blob_forbidden_metrics_absent`; paper `analytics_export` may nest equity `nav_*` / stress `*_pnl` diagnostics but `validate_analytics_export` fails closed on `live_pnl_claim=true`
- Docstrings on `research/catalog.py` + `ANALYTICS_SCHEMA_KEYS` / `validate_analytics_export`; thin proof `tests/unit/test_honesty_catalog_dual.py`; INSTITUTIONAL_READINESS dual-catalog note
- MATH_SPEC + RESEARCH_CENTRE dual-catalog note; conformal residual / OT / PERF **skipped** (no concrete gap / prefer skip)
- No fake live Sharpe / live capital claim

## Day Wave 11 DONE (2026-09-16)

- **H-table Kupiec skip-on-nonfinite:** H16–H18 (and H4/H7/H8/H11/H12) mint only when `kupiec_p` is **finite**; `None` / NaN / inf skip the row (same as empty-panel). Prevents NaN-p success string "coverage consistent with nominal alpha" and FDR pollution.
- Shared `_hyp` hardened: non-finite p → decision `"Inference unavailable (non-finite p-value)."` (never yes/no success claim).
- Tests in `tests/unit/test_hypothesis_semantics.py` (nan skip, empty skip, finite present, other Kupiec nan).
- OT / PERF / sprawling conformal **skipped**.
- No fake live Sharpe / live capital claim

## Day Wave 12 DONE (2026-09-16)

- **H9 e-process skip-on-nonfinite:** NaN/inf `e_sup` omitted — Python `max(nan, 1.0)`→1.0 previously minted p=1 calibration **success** ("consistent with α").
- **H10/H15 bound skip-on-nonfinite:** NaN coverage (or NaN CV+ floor) omitted — previously claimed "below floor".
- **H19 FDR skip-on-nonfinite:** NaN FDR omitted (no "unavailable" bound mint); finite still at-or-below / exceeds.
- Tests in `tests/unit/test_hypothesis_semantics.py` (H9/H10/H15/H19 NaN skips + finite still present).
- OT / PERF / sprawling conformal **skipped**.
- No fake live Sharpe / live capital claim

## Day Wave 13 DONE (2026-09-16)

- **Discovery DM/IC skip-on-nonfinite:** H1 (`p_ic`), H2 (`ls_p`) independent gates; H3 (`dm_p`); pairwise `H_rank_dm_*` (`p_value`) — NaN/inf **skip** mint (align with Kupiec; prefer omit over "unavailable" discovery rows). BH finite-mask unchanged.
- Tests in `tests/unit/test_hypothesis_semantics.py` (H3 nan/finite, pairwise nan/finite, H1/H2 independence).
- OT / PERF / sprawling conformal **skipped**.
- No fake live Sharpe / live capital claim

## Day Wave 14 DONE (2026-09-16)

- **Contrast-inference skip-on-nonfinite:** H5/H6/H13/H14 — `_finite_number` on input scalars before contrast; NaN/inf computed `p` **skip** mint (align discovery DM hygiene; prefer omit over "unavailable"). Finite inputs + finite p still present.
- Tests in `tests/unit/test_hypothesis_semantics.py` (H5 nan brier / no-series; H6 nan advantage / no-series / with-series; H13/H14 nan inputs; shared contrast nan/finite).
- OT / PERF / sprawling conformal **skipped**.
- No fake live Sharpe / live capital claim

## Day Wave 15 DONE (2026-09-16)

- **`bench_tail` ES diagnostics:** Acerbi–Székely Z1/Z2 + mean Fissler–Ziegel FZ0 wired beside Kupiec on holdout losses (scalar broadcast or scaled path arrays). Alpha matches analytics: coverage 0.95 for FZ; miss level 0.05 for Acerbi Z1. Primary keys prefer scaled when `vol_20` present; `*_unscaled` / `*_scaled` mirrors. Honest NaN on empty/no-hits/non-positive ES.
- Thin tests: `tests/unit/test_bench_tail_es_diagnostics.py` (keys finite-or-NaN; forbidden-metrics absent; Kupiec retained).
- OT / PERF / sprawling conformal **skipped**.
- No fake live Sharpe / live capital claim

## Day Wave 16 DONE (2026-09-16)

- **`bench_tail` Christoffersen ind+CC:** `christoffersen_ind_lr`/`ind_p` + `christoffersen_cc_lr`/`cc_p` wired beside Kupiec on the same holdout hit series (unscaled + scaled paths). Miss level \(p_{miss}=0.05\). Primary keys prefer scaled when `vol_20` present; `*_unscaled` / `*_scaled` mirrors. Honest NaN on empty/short series.
- Thin tests: `tests/unit/test_bench_tail_es_diagnostics.py` (Christoffersen keys + mirrors; Kupiec+Acerbi+FZ retained; forbidden-metrics absent).
- H-table mint for CC/ind **skipped** this wave (prefer omit; no invented p).
- OT / PERF / sprawling conformal **skipped**.
- No fake live Sharpe / live capital claim

## Day Wave 17 DONE (2026-09-16)

- **H-table `H4b_var_christoffersen_cc`:** mints from finite `families['tail'].christoffersen_cc_p` (calibration; `_finite_number` skip-on-nonfinite; Kupiec-style yes/no on conditional coverage). **CC only** — no ind hyp (avoids BH double-count; CC nests ind+Kupiec). Never invent p.
- Tests: `tests/unit/test_hypothesis_semantics.py` (`test_h4b_nan_christoffersen_cc_p_skips`, `test_h4b_finite_christoffersen_cc_p_mints_calibration`).
- OT / PERF / sprawling conformal **skipped**.
- No fake live Sharpe / live capital claim

## Day Wave 18 DONE (2026-09-16)

- Distribution bench wires research-only `e_dm_crps_final` / `e_dm_crps_reject` / `e_dm_crps_n` via Wave4 `e_process_dm` on gaussian vs empirical per-obs quantile-CRPS losses (beside existing `dm_crps_*`; gate `n_te>=3`; omit on e-process failure; `research_only=True`; no `live_pnl_claim`). Prefixed to avoid vol-bench `e_dm_*` clash.
- Thin smoke: `tests/unit/test_bench_distribution_eprocess.py`
- **Soft VaR-battery verify honesty:** when nonempty `families["tail"]` contains `kupiec_p` or `kupiec_lr` (even NaN), `verify_research_artifact` requires key *presence* of `christoffersen_cc_p`/`cc_lr` + preferred `christoffersen_ind_p`/`ind_lr` (values may be NaN). Empty `{}` skips. Fail-closed `tail_var_battery_incomplete:<key>`. Helpers: `catalog.tail_var_battery_missing_keys` / `tail_var_battery_keys_present`; soft scorecard `tail_var_battery_ok`. Not a live promotion gate.
- Tests: `tests/unit/test_research_verify.py` (complete battery; Kupiec-without-CC fails; `kupiec_lr` marker; empty ok; NaN presence ok; forbidden-metrics hygiene unchanged).
- OT / PERF / Acerbi–FZ H-table / sprawling conformal **skipped**.
- No fake live Sharpe / live capital claim

## Day Wave 19 DONE (2026-09-16)

- Distribution bench wires research-only `dm_crps_scaled_preferred` / `dm_crps_scaled_p` / `dm_crps_scaled_stat` and `e_dm_crps_scaled_final` / `e_dm_crps_scaled_reject` / `e_dm_crps_scaled_n` via `diebold_mariano` + Wave4 `e_process_dm` on ScaledGaussian vs ScaledStudentT per-obs quantile-CRPS losses when `vol_20` path present (`n_te>=3`; Day Wave 20: NaN/False/0 sentinels on e-process failure; `research_only=True`; no `live_pnl_claim`). Diagnostics only — wrappee selection unchanged. Unscaled `e_dm_crps_*` retained.
- Thin smoke: `tests/unit/test_bench_distribution_eprocess.py` (scaled + unscaled hygiene).
- OT / PERF / Acerbi–FZ H-table / sprawling conformal **skipped**.
- No fake live Sharpe / live capital claim

## Soft CRPS e-process verify (Wave19 companion / soft-verify harden)

- Soft distribution CRPS e-process verify: when nonempty `families["distribution"]` has `dm_crps_p` and/or `dm_crps_scaled_p` (even NaN), require matching `e_dm_crps_*` / `e_dm_crps_scaled_*` key *presence* (NaN/False/0 ok). Empty `{}` skips. Helpers `catalog.dist_crps_eprocess_missing_keys` / `dist_crps_eprocess_keys_present`; `verify_research_artifact` → `dist_crps_eprocess_incomplete:<key>`; soft scorecard `dist_crps_eprocess_ok`. **Not** a live promotion gate.
- Bench harden: on e-process failure beside DM, emit NaN/False/0 sentinels (presence) instead of silent omit.

## Day Wave 20 DONE (2026-09-16)

- Soft distribution CRPS e-process verify: when nonempty `families["distribution"]` has `dm_crps_p` and/or `dm_crps_scaled_p` (even NaN), require matching `e_dm_crps_*` / `e_dm_crps_scaled_*` key *presence* (NaN/False/0 ok). Empty `{}` skips. Helpers `catalog.dist_crps_eprocess_missing_keys` / `dist_crps_eprocess_keys_present`; `verify_research_artifact` → `dist_crps_eprocess_incomplete:<key>`; soft scorecard `dist_crps_eprocess_ok`. **Not** a live promotion gate.
- Bench harden: on e-process failure beside DM, emit NaN/False/0 presence sentinels.
- Tests: `tests/unit/test_research_verify.py` (empty/no-DM; DM-without-e; scaled; NaN ok; empty distribution).
- OT / PERF / Acerbi–FZ H-table / sprawling conformal / soft ES-battery (→ Wave21) **skipped**.
- No fake live Sharpe / live capital claim

## Day Wave 21 DONE (2026-09-16)

- **Soft ES-battery verify honesty:** when nonempty `families["tail"]` contains `es_95` OR `realized_es` OR `var_95` (even NaN), `verify_research_artifact` requires key *presence* of `acerbi_szekely_z1`/`z2`, `fissler_ziegel_mean`, `es_hit_count` (values may be NaN). Empty `{}` skips. Fail-closed `tail_es_battery_incomplete:<key>`. Helpers: `catalog.tail_es_battery_missing_keys` / `tail_es_battery_keys_present`; soft scorecard `tail_es_battery_ok`. **Orthogonal** to VaR-battery (Kupiec⇒Christoffersen; ES markers⇒Acerbi/FZ). Not a live promotion gate.
- Tests: `tests/unit/test_research_verify.py` (complete; ES-marker-without-Acerbi fails; empty ok; NaN presence ok; both batteries independent; VaR-battery still green; forbidden-metrics hygiene unchanged).
- OT / PERF / Acerbi–FZ H-table / sprawling conformal **skipped**.
- No fake live Sharpe / live capital claim

## Day Wave 22 candidates

- Conformal residual edges **only if** a concrete fail-closed gap remains (prefer skip sprawl)
- H-table for `dm_crps_scaled_p` / ES Acerbi/FZ **only if** honest `_finite_number` skip (prefer skip invented p)
- `/ready` missing-manifest flake harden **only if** reproduced
- Further multi-model / pairwise distribution scoring **only if** a concrete ROI gap remains (prefer thin)
- OT / multivariate conformal **prefer skip** unless ADR
- PERF rebench **only if** wrappee / `history_prefix` change materially
- Vendor adapters remain gated; never unblock live without separate ADR

## Day Wave 22 DONE (2026-09-16)

- **`/ready` integrity hardening:** readiness now requires
  `research_receipt == ok` in addition to the data manifest, directories, and
  core imports. Missing receipts return structured `503` readiness with an
  explicit `checks.research_receipt=false`; regression covered by
  `test_readiness_is_fail_closed_for_missing_research_receipt`. This is a
  research-runtime integrity check only and does not authorize live capital.

## Day Wave 25 DONE (2026-09-16) — soft battery_ok forge fail-closed (day_grind DayWave23)

- **Numbering note:** day_grind progress keeps **DayWave23** (INFLIGHT banner); SOTA uses **Wave25** to avoid colliding with existing Wave23 promotion-receipt and Wave24 run_id sections.
- Suite: **1204** non-network green (~11s wall); ruff clean on touch set. Error codes: `scorecard_tail_var_battery_flag_forged:tail` / `scorecard_tail_es_battery_flag_forged:tail` / `scorecard_dist_crps_eprocess_flag_forged:distribution`.

- **Scorecard soft-battery forge honesty:** when `scorecard["tail"]["tail_var_battery_ok"]` / `tail_es_battery_ok` or `scorecard["distribution"]["dist_crps_eprocess_ok"]` is **True** but the matching catalog helper (`tail_var_battery_keys_present` / `tail_es_battery_keys_present` / `dist_crps_eprocess_keys_present`) is False, `verify_research_artifact` fail-closes with `scorecard_tail_var_battery_flag_forged:tail` / `scorecard_tail_es_battery_flag_forged:tail` / `scorecard_dist_crps_eprocess_flag_forged:distribution` (parallel to `scorecard_forbidden_flag_forged`). Incomplete families without forged True still fail via existing `*_incomplete:<key>` only. Not a live promotion gate.
- Tests: forge True+incomplete → `*_flag_forged`; honest True+complete → no forge; absent/False flags → no forge (incomplete:* still fires): `test_verify_rejects_forged_tail_var_battery_ok_when_incomplete`, `test_verify_rejects_forged_tail_es_battery_ok_when_incomplete`, `test_verify_rejects_forged_dist_crps_eprocess_ok_when_incomplete`, `test_verify_honest_battery_ok_true_when_keys_present_no_forge`, `test_verify_absent_battery_ok_flags_no_forge_even_if_incomplete`, `test_verify_false_battery_ok_flags_no_forge_even_if_incomplete`.
- OT / PERF / Acerbi–FZ H-table / sprawling conformal **skipped**.
- No fake live Sharpe / live capital claim

## Day Wave 26 candidates → DONE (H4b notebook consistency; see below)



## Day Wave 24 DONE (2026-09-16)

- **Receipt-to-candidate identity binding:** promotion now requires the
  candidate `run_id` to exactly match `provenance.run_id` in the verified
  immutable research receipt. Missing or mismatched identities produce
  `research_receipt_run_id_unbound` and force `promote=false`; regression
  covered by `test_validate_rejects_receipt_run_id_mismatch`.

## Day Wave 27 DONE (2026-09-16) — causal-panel integrity (sibling; preserved)

- **Causal-panel integrity:** promotion evidence now rejects null or empty
  security identifiers, null timestamps, non-finite weights, and duplicate
  `(event_time, security_id)` rows. This prevents ambiguous or silently
  double-counted target-weight panels from satisfying the causal gate; coverage
is in `test_validate_rejects_ambiguous_causal_panel`.

## Day Wave 26 DONE (2026-09-16) — H4b notebook consistency (day_grind DayWave26)

- **Soft H4b H-table consistency:** when `families["tail"]` is a dict with **finite**
  `christoffersen_cc_p`, `verify_research_artifact` requires a hypothesis with
  `id == "H4b_var_christoffersen_cc"` and `family == "calibration"` (Day Wave 17 mint).
  Missing → `hypothesis_h4b_missing_despite_finite_christoffersen_cc_p`. Non-finite /
  missing `cc_p` → skip. Helpers: `catalog.tail_has_finite_christoffersen_cc_p` /
  `hypotheses_include_h4b` / `h4b_hypothesis_consistency_errors`. Not a live promotion gate.
- Tests: finite cc_p without H4b → fail; with calibration H4b → ok; NaN cc_p without H4b → ok;
  wrong-family H4b → fail (`test_verify_rejects_finite_christoffersen_cc_p_without_h4b`,
  `test_verify_accepts_finite_christoffersen_cc_p_with_h4b`,
  `test_verify_nan_christoffersen_cc_p_without_h4b_ok`,
  `test_verify_rejects_finite_cc_p_with_h4b_wrong_family`).
- OT / PERF / Acerbi–FZ invented-p sprawl **skipped**.
- No fake live Sharpe / live capital claim.

## Day Wave 29 DONE (2026-09-16) — H4 Kupiec notebook consistency (day_grind DayWave29)

- **Soft H4 Kupiec H-table consistency:** when `families["tail"]` is a dict with **finite**
  `kupiec_p`, `verify_research_artifact` requires a hypothesis with
  `id == "H4_var_kupiec"` and `family == "calibration"` (agent mint).
  Missing → `hypothesis_h4_missing_despite_finite_kupiec_p`. Non-finite /
  missing `kupiec_p` → skip. Helpers: `catalog.tail_has_finite_kupiec_p` /
  `hypotheses_include_h4` / `h4_hypothesis_consistency_errors`. Not a live promotion gate.
- Tests: finite kupiec_p without H4 → fail; with calibration H4 → ok; NaN kupiec_p without H4 → ok;
  wrong-family H4 → fail (`test_verify_rejects_finite_kupiec_p_without_h4`,
  `test_verify_accepts_finite_kupiec_p_with_h4`,
  `test_verify_nan_kupiec_p_without_h4_ok`,
  `test_verify_rejects_finite_kupiec_p_with_h4_wrong_family`).
- OT / PERF / Acerbi–FZ invented-p sprawl **skipped**.
- No fake live Sharpe / live capital claim.

## Day Wave 28 DONE (2026-09-16) — conformal set metrics (sibling; preserved)

- **Conformal set-ordering invariant:** shared interval expansion now
  defensively normalizes malformed base bounds so conformal outputs cannot be
  inverted or report negative-width sets. Regression coverage is in
  `test_expand_interval_normalizes_inverted_base_bounds`; valid intervals are
  unchanged.
- **Conformal diagnostic honesty:** `set_metrics` now excludes inverted
  intervals from coverage and width statistics instead of allowing negative
  widths to improve a benchmark. Regression coverage is in
  `test_set_metrics_excludes_inverted_intervals`.
- **Conformal shape integrity:** `set_metrics` now rejects mismatched
  observation/bound shapes instead of relying on NumPy broadcasting; covered
  by `test_set_metrics_rejects_shape_mismatch`.

- **Interval-cap diagnostic honesty:** `bench_interval_caps` now excludes
  inverted or non-finite interval geometry from `mean_width` rather than
  allowing invalid negative widths to improve the portfolio diagnostic;
  covered by `test_bench_interval_caps_excludes_inverted_widths`.

- **Coverage shape integrity:** `covered` now rejects mismatched observation
  and bound shapes instead of relying on NumPy broadcasting; covered by
  `test_covered_rejects_shape_mismatch`.

- **General scoring width honesty:** `metrics.scoring.interval_width` now
  excludes inverted or non-finite bounds and returns honest `NaN` when no
  valid interval remains; covered by the scoring edge suite.

- **Quantile crossing honesty:** `quantile_crossing_rate` now excludes rows
  with non-finite quantiles and returns `NaN` when no valid rows remain,
  preventing missing forecasts from masquerading as monotone quantiles.

- **PIT integrity:** `pit_values` now rejects malformed tau grids and emits
  `NaN` for non-finite or crossing quantile rows instead of interpolating
  invalid forecast geometry; covered by the scoring edge suite.

- **Quantile-axis integrity:** `quantile_crossing_rate` now requires finite,
  nondecreasing tau levels, preventing semantically invalid quantile axes from
  producing a benchmark statistic.

- **CRPS-axis integrity:** `crps_from_quantiles` now requires finite tau
  levels strictly inside `(0, 1)` and nondecreasing, preventing negative or
  undefined integration weights.

- **PIT probability-domain integrity:** `pit_values` now applies the same
  strict `(0, 1)` tau-domain requirement, keeping PIT interpolation aligned
  with probability semantics.

- **Promotion receipt freshness binding:** `validate_candidate` now compares
  the receipt's immutable worktree fingerprint with the current checkout and
  fails closed on stale or unavailable checkout state, preventing a valid
  receipt from being replayed after code changes.

## Day Wave 30 DONE (2026-09-16) — H3 vol-DM notebook consistency (day_grind DayWave30)

- **Soft H3 vol-DM H-table consistency:** when `families["volatility"]` is a dict with **finite**
  `dm_p`, `verify_research_artifact` requires a hypothesis with
  `id == "H3_vol_dm"` and `family == "discovery"` (agent mint).
  Missing → `hypothesis_h3_missing_despite_finite_dm_p`. Non-finite /
  missing `dm_p` → skip. Helpers: `catalog.volatility_has_finite_dm_p` /
  `hypotheses_include_h3` / `h3_hypothesis_consistency_errors`. Not a live promotion gate.
- Tests: finite dm_p without H3 → fail; with discovery H3 → ok; NaN dm_p without H3 → ok;
  wrong-family H3 → fail (`test_verify_rejects_finite_dm_p_without_h3`,
  `test_verify_accepts_finite_dm_p_with_h3`,
  `test_verify_nan_dm_p_without_h3_ok`,
  `test_verify_rejects_finite_dm_p_with_h3_wrong_family`).
- **Skipped:** `dm_crps_*` hyp consistency (no concrete hyp ID); paper/shadow already DayWave9; OT / PERF / Acerbi–FZ invented-p sprawl.
- No fake live Sharpe / live capital claim.

## Day Wave 31 DONE (2026-09-16) — H1/H2 oracle ranking notebook consistency (day_grind DayWave31)

- **Soft H1/H2 oracle ranking H-table consistency:** find `oracle_raw` in `notebook["rankers"]`
  (skip names starting with `_`; oracle is **not** in families). When `p_ic` is **finite**,
  `verify_research_artifact` requires hypothesis `id == "H1_ranking_oracle"` and
  `family == "discovery"` (agent mint). Missing → `hypothesis_h1_missing_despite_finite_p_ic`.
  When `ls_p` is **finite**, require `id == "H2_decile_mono"` and `family == "discovery"`.
  Missing → `hypothesis_h2_missing_despite_finite_ls_p`. Gates are **independent**.
  Non-finite / missing / no `oracle_raw` → skip each. Helpers: `catalog.rankers_oracle_raw` /
  `oracle_has_finite_p_ic` / `oracle_has_finite_ls_p` / `hypotheses_include_h1` /
  `hypotheses_include_h2` / `h1_hypothesis_consistency_errors` / `h2_hypothesis_consistency_errors`.
  Not a live promotion gate.
- Tests: finite p_ic without H1 → fail; with discovery H1 → ok; NaN p_ic without H1 → ok;
  wrong-family H1 → fail; finite ls_p without H2 → fail; with discovery H2 → ok; NaN ls_p without H2 → ok;
  wrong-family H2 → fail; both gates independent (`test_research_verify.py`).
- **Skipped:** OT / PERF / Acerbi–FZ invented-p sprawl; H7 ACI consistency deferred.
- No fake live Sharpe / live capital claim.

## Day Wave 32 DONE (2026-09-16) — H7 ACI notebook consistency (day_grind DayWave32)

- **Soft H7 ACI miss-Kupiec H-table consistency:** when `families["conformal"]["aci"]` has
  **finite** `kupiec_p`, `verify_research_artifact` requires hypothesis
  `id == "H7_aci_coverage"` and `family == "calibration"` (agent mint). Missing →
  `hypothesis_h7_missing_despite_finite_aci_kupiec_p`. Non-finite / missing / no `aci` → skip.
  Helpers: `catalog.conformal_aci_blob` / `aci_has_finite_kupiec_p` /
  `hypotheses_include_h7` / `h7_hypothesis_consistency_errors`. Not a live promotion gate.
- Tests: finite ACI kupiec_p without H7 → fail; with calibration H7 → ok; NaN without H7 → ok;
  wrong-family H7 → fail (`test_research_verify.py`).
- **Skipped:** Mondrian H8 consistency (separate hyp); OT / PERF / Acerbi–FZ invented-p sprawl.
- No fake live Sharpe / live capital claim.

## Day Wave 33 DONE (2026-09-16) — H8 Mondrian high-vol notebook consistency (day_grind DayWave33)

- **Soft H8 Mondrian high-X miss-Kupiec H-table consistency:** when `families["conformal"]["mondrian_aci"]` has
  **finite** `high_x_kupiec_p`, `verify_research_artifact` requires hypothesis
  `id == "H8_mondrian_high_vol"` and `family == "calibration"` (agent mint). Missing →
  `hypothesis_h8_missing_despite_finite_high_x_kupiec_p`. Non-finite / missing / no `mondrian_aci` → skip.
  Helpers: `catalog.conformal_mondrian_aci_blob` / `mondrian_aci_has_finite_high_x_kupiec_p` /
  `hypotheses_include_h8` / `h8_hypothesis_consistency_errors`. Not a live promotion gate.
- Tests: finite high_x_kupiec_p without H8 → fail; with calibration H8 → ok; NaN without H8 → ok;
  wrong-family H8 → fail (`test_research_verify.py`).
- **Skipped:** OT / PERF / Acerbi–FZ invented-p / sprawling conformal beyond H8.
- No fake live Sharpe / live capital claim.

## Day Wave 34 DONE (2026-09-16) — H11 CRC notebook consistency (day_grind DayWave34)

- **Soft H11 CRC miss-Kupiec H-table consistency:** when `families["crc"]` has
  **finite** `kupiec_p`, `verify_research_artifact` requires hypothesis
  `id == "H11_crc_var"` and `family == "calibration"` (agent mint). Missing →
  `hypothesis_h11_missing_despite_finite_crc_kupiec_p`. Non-finite / missing → skip.
  Helpers: `catalog.crc_has_finite_kupiec_p` / `hypotheses_include_h11` /
  `h11_hypothesis_consistency_errors`. Not a live promotion gate.
- Tests: finite CRC kupiec_p without H11 → fail; with calibration H11 → ok; NaN without H11 → ok;
  wrong-family H11 → fail; catalog smoke for finite / skip-on-nonfinite (`test_research_verify.py`).
- **Skipped:** H12 weighted CQR consistency (separate hyp); OT / PERF / Acerbi–FZ invented-p sprawl.
- No fake live Sharpe / live capital claim.

## Day Wave 35 DONE (2026-09-16) — H12 weighted CQR notebook consistency (day_grind DayWave35)

- **Soft H12 weighted CQR miss-Kupiec H-table consistency:** when `families["weighted_conformal"]` has
  **finite** `kupiec_p`, `verify_research_artifact` requires hypothesis
  `id == "H12_weighted_cqr"` and `family == "calibration"` (agent mint). Missing →
  `hypothesis_h12_missing_despite_finite_wcqr_kupiec_p`. Non-finite / missing → skip.
  Helpers: `catalog.weighted_conformal_has_finite_kupiec_p` / `hypotheses_include_h12` /
  `h12_hypothesis_consistency_errors`. Not a live promotion gate.
- Tests: finite WCQR kupiec_p without H12 → fail; with calibration H12 → ok; NaN without H12 → ok;
  wrong-family H12 → fail; catalog smoke for finite / skip-on-nonfinite (`test_research_verify.py`).
- **Skipped:** OT / PERF / Acerbi–FZ invented-p sprawl; H16–H18 panel Kupiec consistency deferred to Wave36 if ROI (or stop soft-verify sprawl).
- No fake live Sharpe / live capital claim.

## Day Wave 36 DONE (2026-09-16) — H9 e-process ACI notebook consistency (day_grind DayWave36)

- **Soft H9 e-process ACI H-table consistency:** when `families["evalues"]` has
  **finite** `e_sup`, `verify_research_artifact` requires hypothesis
  `id == "H9_eprocess_aci"` and `family == "calibration"` (agent mint). Missing →
  `hypothesis_h9_missing_despite_finite_e_sup`. Non-finite / missing → skip.
  Helpers: `catalog.evalues_has_finite_e_sup` / `hypotheses_include_h9` /
  `h9_hypothesis_consistency_errors`. Not a live promotion gate.
- Tests: finite e_sup without H9 → fail; with calibration H9 → ok; NaN without H9 → ok;
  wrong-family H9 → fail; catalog smoke for finite / skip-on-nonfinite (`test_research_verify.py`).
- **Skipped:** OT / PERF / Acerbi–FZ invented-p sprawl; H10 jackknife / H13 interval deferred to Wave37 if ROI.
- No fake live Sharpe / live capital claim.

## Day Wave 37 DONE (2026-09-16) — H10 Jackknife+ coverage notebook consistency (day_grind DayWave37)

- **Soft H10 Jackknife+ coverage H-table consistency:** when `families["jackknife_plus"]` has
  **finite** `coverage`, `verify_research_artifact` requires hypothesis
  `id == "H10_jackknife_coverage"` and `family == "bound"` (agent mint — **bound**, not
  calibration). Missing → `hypothesis_h10_missing_despite_finite_coverage`. Non-finite /
  missing → skip. Helpers: `catalog.jackknife_plus_has_finite_coverage` /
  `hypotheses_include_h10` / `h10_hypothesis_consistency_errors`. Not a live promotion gate.
- Tests: finite coverage without H10 → fail; with bound H10 → ok; NaN without H10 → ok;
  wrong-family H10 → fail; catalog smoke for finite / skip-on-nonfinite (`test_research_verify.py`).
- **Skipped:** OT / PERF / Acerbi–FZ invented-p sprawl; H16–H18 panel Kupiec deferred; H13/H15 if ROI later.
- No fake live Sharpe / live capital claim.

## Day Wave 38 DONE (2026-09-16) — H15 CV+ floor notebook consistency (day_grind DayWave38)

- **Soft H15 CV+ floor H-table consistency:** when `families["cv_plus"]` has
  **finite** `coverage` **and** **finite** `coverage_floor`, `verify_research_artifact` requires hypothesis
  `id == "H15_cv_plus_floor"` and `family == "bound"` (agent mint — both fields finite; **bound**).
  Missing → `hypothesis_h15_missing_despite_finite_coverage_and_floor`. Non-finite / missing either → skip.
  Helpers: `catalog.cv_plus_has_finite_coverage_and_floor` /
  `hypotheses_include_h15` / `h15_hypothesis_consistency_errors`. Not a live promotion gate.
- Tests: finite coverage+floor without H15 → fail; with bound H15 → ok; NaN coverage or NaN floor without H15 → ok;
  wrong-family H15 → fail; catalog smoke for both-finite / skip-on-nonfinite (`test_research_verify.py`).
- **Skipped:** OT / PERF / Acerbi–FZ invented-p sprawl; H13 contrast prefer skip; stop soft-verify sprawl OK; H19 FDR if ROI.
- No fake live Sharpe / live capital claim.
- Soft-fail token: `hypothesis_h15_missing_despite_finite_coverage_and_floor` (both-fields gate).
- Verify accepts JSON `null` hyp `p_value` as unavailable (agent `_jsonable` nan→null) so bound H10/H15/H19 receipts stay doctor-ok.

## Day Wave 39 DONE (2026-09-16) — H16–H18 panel Kupiec notebook consistency (day_grind DayWave39)

- **Soft H16–H18 panel Kupiec H-table consistency (batch):** when `families["localized_conformal"]` /
  `online_crc` / `portfolio_conformal` has **finite** `kupiec_p` and `dgp != "fixture"`,
  `verify_research_artifact` requires matching hypothesis
  `H16_localized_cqr` / `H17_online_crc` / `H18_portfolio_conformal` with `family == "calibration"`
  (agent mint). Missing → `hypothesis_h16_missing_despite_finite_kupiec_p` /
  `hypothesis_h17_missing_despite_finite_kupiec_p` /
  `hypothesis_h18_missing_despite_finite_kupiec_p`. Fixture DGP / non-finite / missing → skip.
  Helpers: `catalog.panel_family_has_finite_kupiec_p` / `hypotheses_include_h16|h17|h18` /
  `h16_h18_panel_kupiec_consistency_errors` (one parameterized helper). Not a live promotion gate.
- Tests: finite panel kupiec_p without H16–H18 → fail; with calibration rows → ok; fixture DGP → ok;
  NaN → ok; wrong-family H16 → fail; catalog smoke (`test_research_verify.py`).
- **Skipped:** OT / PERF / Acerbi–FZ invented-p sprawl; H13 interval caps deferred; stop further soft-verify sprawl preferred next.
- No fake live Sharpe / live capital claim.
- **Race note:** parent preferred H16–H18 as Wave37; sibling closed Wave37=H10 and Wave38=H15; this wave lands preferred panel batch as Wave39.

## Day Wave 40 DONE (2026-09-16) — H19 conformal-rank FDR bound notebook consistency (day_grind DayWave40)

- **Soft H19 conformal-rank FDR bound H-table consistency:** when `families["conformal_rank"]` has
  **finite** `fdr` and `dgp != "fixture"`, `verify_research_artifact` requires
  `H19_conformal_rank` with `family == "bound"` (agent mint). Missing/wrong-family →
  `hypothesis_h19_missing_despite_finite_fdr`. Fixture / non-finite / missing → skip.
  Helpers: `catalog.conformal_rank_has_finite_fdr` / `hypotheses_include_h19` /
  `h19_hypothesis_consistency_errors`. Wired via `families.get("conformal_rank")`.
  Research diagnostic — not a live promotion gate.
- **Tests:** finite fdr without H19 → fail; with bound H19 → ok; NaN without H19 → ok;
  fixture dgp without H19 → ok; wrong-family → fail; catalog smoke (`test_research_verify.py`).
- **Skipped:** OT / PERF / Acerbi–FZ invented-p; H5/H6/H13/H14 discovery contrasts deferred.
  Soft-verify H-table sprawl **paused** after this last bound-family mint twin.
- **Suite:** `uv run pytest -q -m 'not network'` → **1303 green** (~17.4s wall); `ruff check` clean on touch set.
- No Cursor / commits / fake live Sharpe. notify_user=no.

## Day Wave 41 DONE (2026-09-16) — marginal vs training-conditional coverage honesty (day_grind DayWave41)

- **Honesty keys:** `JackknifePlus` / `CVPlus` metadata and nonempty `bench_jackknife_plus` /
  `bench_cv_plus` surface `coverage_guarantee_scope="marginal_exchangeable"` plus a short
  claim that the coverage floor is **marginal under exchangeability**, **not**
  training-conditional (Barber–Candès–Ramdas–Tibshirani 2021; Bian–Barber 2023 caveat).
  `research_only=True`; no `live_pnl_claim`; forbidden-metrics hygiene. H10/H15 floors inherit
  the same scope — lab must not imply training-conditional guarantees.
- **Tests:** thin unit/fixture asserts on meta + `bench_cv_plus()` fixture path
  (`test_coverage_guarantee_scope.py`; extended jackknife/cv_plus unit asserts).
- **Docs:** MATH_SPEC H10/H15 marginal note; RESEARCH_REFERENCES Barber–Candès Jackknife+ +
  Bian–Barber row; RESEARCH_CENTRE Wave41.
- **Skipped:** soft-verify H-table sprawl (H5/H6/H13/H14); OT / PERF / Acerbi invented-p.
  Preserve Waves 32–40.
- **Suite:** `uv run pytest -m 'not network'` → **1310 green** (~26.3s wall); `ruff check` clean on touch set.
- No fake live Sharpe / live capital claim. notify_user=no.

## Day Wave 42 DONE (2026-09-16) — Jackknife+/CV+ coverage_guarantee_scope receipt verify (day_grind DayWave42)

- **Soft receipt verify:** when nonempty `families["jackknife_plus"]` / `families["cv_plus"]`
  exposes `coverage` **or** `coverage_floor` (key present — NaN ok), `verify_research_artifact`
  requires `coverage_guarantee_scope == "marginal_exchangeable"` (Wave41 honesty key).
  Missing → `coverage_guarantee_scope_missing:<fam>`; wrong → `coverage_guarantee_scope_invalid:<fam>`.
  Empty `{}` / no coverage keys → skip. Helpers:
  `catalog.jp_cv_blob_requires_marginal_coverage_scope` /
  `coverage_guarantee_scope_is_marginal` /
  `coverage_guarantee_scope_consistency_errors`. Parallel to VaR/ES battery presence gates.
  Research diagnostic — **not** a live promotion gate. No forged soft scorecard flag.
- **Tests:** complete ok; coverage without scope fails; wrong scope fails; empty ok;
  NaN coverage still requires scope; forbidden hygiene unchanged (`test_research_verify.py`).
- **Skipped:** soft-verify H-table sprawl (H5/H6/H13/H14); OT / PERF / Acerbi invented-p.
  Preserve Waves 32–41.
- **Suite:** `uv run pytest -q -m 'not network'` → **1321 green** (~28.8s wall); `ruff check` clean on touch set.
- No fake live Sharpe / live capital claim. notify_user=no.

## Day Wave 43 DONE (2026-09-16) — assume_sorted history prefix fail-closed (day_grind DayWave43)

- **Fail-closed:** `history_upto` / `history_for_calibration` / `optimize_asof` trailing-hist:
  when `assume_sorted=True` on a nonempty frame with `event_time` that is **not**
  `under_history_sort_contract`, raise `ValueError` (via `_require_assume_sorted_contract`)
  instead of taking `history_prefix_upto` binary search. Empty frames OK. Sorted +
  `assume_sorted=True` still matches filter. No-`day_index` path unchanged (filter).
  Happy path: one contract check when `assume_sorted=True`.
- **Tests:** unsorted+assume_sorted+day_index raises; sorted assume_sorted matches filter;
  no day_index unsorted unchanged; calibration unsorted assume_sorted raises; empty OK
  (`test_history_upto.py`).
- **Docs:** MATH_SPEC / PERF (no rebench) / RESEARCH_CENTRE / SOTA Wave43 DONE.
- **Skipped:** soft-verify H-table sprawl (H5/H6/H13/H14); OT / PERF rebench / Acerbi invented-p.
  Preserve Waves 32–42.
- **Suite:** (filled after full run).
- No fake live Sharpe / live capital claim. notify_user=no.

## Day Wave 44 DONE (2026-09-16) — soft scorecard executed/nonempty/finite_observation forge

- Closed by CoS Day Wave 48 docs/suite pass (helpers+verify+tests were parked/pre-landed).
- See Day Wave 48 DONE for suite proof.

## Day Wave 45 DONE (2026-09-16) — closed-form Student-t CRPS (day_grind DayWave43 race)

- **Helpers:** `crps_student_t` / `mean_crps_student_t` in `metrics/scoring.py` (Jordan–Krüger–Lerch /
  scoringRules closed form; ν>2 fail-closed NaN; bad σ → NaN; empty → []/NaN; mismatch → ValueError).
  Exported from `metrics/__init__.py`.
- **Bench:** research-only `crps_scaled_student_t_closed` beside quantile Riemann
  `crps_scaled_student_t` on scaled `vol_20` path (`_distribution_horizon_scores`). Dual view;
  forbidden-metrics hygiene unchanged. Not a live capital claim.
- **Tests:** median ν=5 closed form; scale homogeneity; large-ν ≈ Gaussian; edges;
  bench key finite when scaled-t present (`test_crps_closed_form.py`, `test_bench_crps_closed.py`).
- **Numbering note:** day_grind INFLIGHT banner was **DayWave43** (Student-t CRPS); sibling claimed
  SOTA **Wave43** for `assume_sorted` history prefix and parked Wave44 soft scorecard forge —
  this wave is **Wave45** in SOTA to avoid collision.
- **Skipped:** soft-verify H-table sprawl; OT / PERF / Acerbi invented-p. Preserve Waves 32–42
  (+ sibling 43–44).
- **Suite:** (filled after full run); `ruff check` clean on touch set.
- No fake live Sharpe / live capital claim. notify_user=no.

## Day Wave 46 DONE (2026-09-16) — causal input and cache invariants

- **Calibration horizon validation:** `history_for_calibration` now rejects negative,
  fractional, NaN, infinite, and otherwise non-integral `horizon_bars` values instead
  of truncating them through `int(...)`. Regression coverage includes all invalid forms.
- **Event-time fast-path validation:** supplied `event_times` must be strictly increasing;
  unsorted or duplicated sequences fail closed before horizon indexing can shift the
  calibration cutoff.
- **Wrappee cache proof:** executable LRU regression verifies that a hot fit survives
  capacity pressure while the oldest untouched fit is evicted. Content-digest identity,
  causal family reselection, and `live_pnl_claim=false` remain unchanged.
- **Verification:** full non-network CI green after these changes; no new performance
  numbers claimed because the checked-in Wave 39 benchmark predates the cache-policy work.
- **Skipped:** vendor/live adapters, parallel causal dates with sequential `w_prev`,
  soft H-table sprawl, OT/multivariate conformal, and unmeasured PERF claims.

## Day Wave 47 DONE (2026-09-16) — paper ledger identity binding

- **Cross-artifact identity:** `validate_ledger_schema` now binds the run-directory
  name to `meta.json.run_id`, and binds `broker_state.json.run_id` to that same
  metadata identity. Promotion receipts were already required to match metadata.
- **Failure mode:** copied or mixed-run paper artifacts fail closed with explicit
  `meta_run_id_mismatch` or `broker_state_run_id_mismatch` errors rather than
  validating as a coherent ledger.
- **Verification:** ledger schema suite passes (15 tests); Ruff, format checks,
  mypy, and the preceding full CI run were clean. No performance or live-readiness
  claim is implied.

## Day Wave 48 DONE (2026-09-16) — fail-closed count evidence

- **Promotion-gate hardening:** walk-forward and multi-fold evidence counts now
  reject booleans, fractional values, NaN, infinity, and non-numeric strings
  instead of accepting Python coercions such as `True == 1`.
- **Regression coverage:** adversarial count fixtures prove malformed metadata
  cannot satisfy causal or stability gates; valid positive integers and integral
  floats remain accepted for JSON compatibility.
- **Verification:** `make ci` passed with 1,355 non-network tests, 87.97% total
  coverage, Ruff, format, and mypy clean. This changes evidence validation only;
  it makes no performance or live-readiness claim.

## Day Wave 49 DONE (2026-09-16) — strict typed evidence counts

- **Evidence boundary:** promotion fold/date counts now require actual JSON
  numeric values (positive integers or integral floats); booleans, numeric
  strings, fractions, NaN, and infinity are rejected without coercion.
- **Regression coverage:** count-gate tests include coercion-shaped values and
  valid numeric compatibility cases.
- **Verification:** `make ci` passed with 1,356 non-network tests, 87.95% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 50 DONE (2026-09-16) — registry authorization hardening

- **Independent promotion boundary:** `promotion_decision` now rejects boolean,
  numeric-string, fractional, NaN, and infinite fold counts; fold stability must
  also be finite numeric evidence. Malformed metrics cannot bypass the registry
  even if an upstream wrapper is skipped.
- **Receipt typing:** emitted `evidence_complete` and `synthetic` fields now use
  strict boolean semantics rather than generic truthiness.
- **Verification:** `make ci` passed with 1,357 non-network tests, 87.98% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 51 DONE (2026-09-16) — total promotion decisions

- **No-crash fail-closed gate:** malformed persisted metric values such as
  arbitrary objects or nonnumeric strings now normalize to unavailable evidence
  instead of raising during threshold comparison.
- **Regression coverage:** promotion tests verify malformed metrics return a
  structured non-promote decision with missing-metric reasons.
- **Verification:** `make ci` passed with 1,368 non-network tests, 87.98% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 52 DONE (2026-09-16) — wrappee resolve fail-closed (CoS)

- Numbering note: Lieutenant owned Waves 50–51 (registry/promotion). CoS wrappee
  hardening ships as **Wave 52** to avoid collision.
- Added `_require_wrappee_resolve_inputs` on `resolve_wrappee_reselect_cached`: empty taus,
  empty train/cal, y vs scale length mismatch, alpha/min_coverage outside (0, 1) raise ValueError.
- `fit_scaled_wrappee`: unknown family name raises ValueError (no silent gaussian fallback).
- Tests in `tests/unit/test_wrappee_reselect.py` (Wave 50-labeled edges in file; suite green).
  Full `pytest -m 'not network'` green; ruff clean on touch set.
- Soft H-table / OT / PERF / Acerbi invented-p skipped. Not a live capital claim.

## Day Wave 53 DONE (2026-09-16) — wrappee resolve residual edges (CoS)

- Added fail-closed tests for empty cal and min_coverage outside (0, 1) on
  `resolve_wrappee_reselect_cached` (completes Wave 52 input guards).
- Suite: wrappee reselect edges green; ruff clean on touch set.
- Soft H-table / OT / PERF skipped. Not a live capital claim.

## Day Wave 54 DONE (2026-09-16) — cross-layer boolean integrity

- **Validation consistency:** outer `validate_candidate` gates now require
  literal booleans for walk-forward completion, causal-panel claims,
  synthetic markers, and evidence completeness. Truthy strings cannot make a
  research result appear valid while the registry rejects it.
- **Regression coverage:** poisoned string flags are explicitly rejected and
  cannot set the causal/walk-forward gate or research `ok` status.
- **Verification:** `make ci` passed with 1,369 non-network tests, 87.99% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 55 candidates

- paper/receipt schema hygiene residuals, or history_prefix edges if concrete gap
- Soft-verify H-table sprawl remains paused; OT / PERF / Acerbi invented-p prefer skip

## Day Wave 55 DONE (2026-09-16) — history_prefix asof type guard (CoS)

- `history_prefix_upto` rejects non-datetime `asof` with TypeError (fail-closed).
- Test: `test_history_prefix_upto_rejects_non_datetime_asof`. Ruff clean.
- Soft H-table / OT / PERF skipped. Not a live capital claim.

## Day Wave 56 DONE (2026-09-16) — total paper-ledger validation

- **Corruption safety:** `validate_ledger_schema` now reports non-object
  `meta.json` / `broker_state.json` payloads instead of raising.
- **Strict schema typing:** schema versions must be actual non-negative integer
  JSON values; string and boolean coercions are rejected.
- **Verification:** `make ci` passed with 1,376 non-network tests, 88.08% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 57 DONE (2026-09-16) — typed paper receipt identity

- **Run identity:** promotion dry-run receipts now reject non-string `run_id`
  values instead of stringifying them before path validation.
- **Regression coverage:** numeric identifiers are explicitly rejected, keeping
  receipt identity aligned with ledger directory/meta/broker-state bindings.
- **Verification:** `make ci` passed with 1,377 non-network tests, 88.09% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 58 DONE (2026-09-16) — latest-run identity safety

- **Corruption handling:** `latest_run_id` now returns no identity for malformed
  JSON, non-object payloads, or non-string IDs instead of raising or coercing
  arbitrary values.
- **Path security preserved:** traversal IDs still raise the established
  path-safety error rather than being silently swallowed.
- **Verification:** `make ci` passed with 1,379 non-network tests, 88.08% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 59 DONE (2026-09-16) — fail-closed broker-state reader

- **Resume safety:** `load_broker_state` now returns no state for malformed JSON
  or non-object payloads instead of handing arbitrary data to resume logic.
- **Security contract preserved:** path-traversal run IDs still raise the
  established path-safety error; this wave only softens corruption handling.
- **Verification:** `make ci` passed with 1,381 non-network tests, 86.77% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 60 DONE (2026-09-16) — resume receipt corruption boundary

- **Paper resume safety:** corrupt or non-JSON `promotion_dry_run.json` now
  produces a controlled invalid-receipt error before resume state is used.
- **Contract preservation:** valid receipts continue to resume normally, while
  receipt validation remains the single source of truth for resume acceptance.
- **Verification:** `make ci` passed with 1,387 non-network tests, 87.98% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 61 DONE (2026-09-16) — Northset and resume integrity integration

- **Resume restoration:** malformed champion/shadow broker state now produces a
  controlled invalid-state error instead of raw deserialization exceptions.
- **Northset catalog integration:** the new Northset family is catalog-versioned
  and its research receipt omits forbidden `live_pnl_claim` keys while retaining
  explicit research-only labeling.
- **Type/quality gate:** corrected shared Northset typing and stale catalog/CLI
  expectations; doctor now surfaces an unready default checkout with exit code 1.
- **Verification:** `make ci` passed with 1,401 non-network tests, 87.71% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 62 DONE (2026-09-16) — resume run-identity binding

- **Cross-artifact invariant:** resume now requires the requested run ID to
  match `broker_state.json.run_id`; mixed-run state fails closed before broker
  restoration or replay.
- **Regression coverage:** paper resume tests cover mismatched state identity,
  corrupt broker state, and corrupt promotion receipts.
- **Verification:** `make ci` passed with 1,402 non-network tests, 87.71% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 63 DONE (2026-09-16) — strict resume cursor validation

- **Replay safety:** persisted `step` must be a non-negative integer, and
  `last_decision` / `last_exec` must be valid ISO timestamps when present.
  Ambiguous cursor metadata now fails closed instead of silently replaying or
  skipping history.
- **Regression coverage:** resume tests cover invalid step and timestamp fields
  in addition to mixed-run identity and corrupt broker/receipt state.
- **Verification:** `make ci` passed with 1,405 non-network tests, 87.72% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 64 DONE (2026-09-16) — resume fingerprint integrity

- **Provenance boundary:** persisted `resume_fingerprint` must be a lowercase
  64-character SHA-256 digest when present; malformed values fail before replay.
- **Regression coverage:** resume tests now cover invalid fingerprints alongside
  cursor timestamps, step types, run identity, broker state, and promotion receipts.
- **Verification:** `make ci` passed with 1,406 non-network tests, 87.74% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 65 DONE (2026-09-16) — persisted analytics schema boundary

- **Artifact integrity:** `validate_ledger_schema` now validates the persisted
  `analytics_export.json` through the canonical analytics-export validator.
- **Fail-closed corruption handling:** malformed JSON, non-object payloads, and
  honesty/schema violations are reported as ledger errors; legacy runs without
  the optional artifact remain visible through a warning.
- **Regression coverage:** schema tests cover a valid paper run plus corrupted,
  non-object, and live-claim analytics exports.
- **Verification:** `make ci` passed with 1,409 non-network tests, 87.73% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 66 DONE (2026-09-16) — Northset integration hardening

- **OHLC contract restored:** Northset volatility diagnostics now receive the
  complete OHLC input required by the estimator, including `open`.
- **Import safety:** Northset public exports are lazy, removing the
  `microstructure.candle_book_features` package-initialization cycle while
  preserving the existing import surface.
- **Type/robustness:** candle geometry rate aggregation handles empty means
  explicitly and remains type-safe.
- **Verification:** `make ci` passed with 1,412 non-network tests, 87.29% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 67 DONE (2026-09-16) — paper analytics provenance binding

- **Artifact identity:** paper `analytics_export.json` now carries the paper
  `run_id`, while shared backtest exports remain compatible without one.
- **Swap detection:** ledger validation rejects a valid analytics export copied
  from another run instead of treating it as an independently valid artifact.
- **Regression coverage:** paper roundtrip and ledger-schema tests cover the
  emitted identity and cross-run tampering.
- **Verification:** `make ci` passed with 1,416 non-network tests, 87.67% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 68 DONE (2026-09-16) — analytics artifact tamper seal

- **Same-run integrity:** paper analytics exports now carry a self-excluding
  SHA-256 digest, mirroring promotion-receipt integrity protection.
- **Fail-closed validation:** malformed, mismatched, or tampered digests are
  reported by ledger schema validation; legacy exports without a digest remain
  backward-compatible.
- **Regression coverage:** tests cover emitted digests, cross-run identity
  binding, same-run content tampering, malformed JSON, and non-object payloads.
- **Verification:** `make ci` passed with 1,421 non-network tests, 87.45% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 69 DONE (2026-09-16) — integrity primitive API contract

- **Public API:** `analytics_export_digest` is now exported from
  `quant_fund.metrics` for consistent paper/backtest tooling.
- **Canonical contract:** direct tests lock deterministic ordering, nested-value
  sensitivity, and self-exclusion of the digest field.
- **Verification:** `make ci` passed with 1,422 non-network tests, 87.45% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 70 DONE (2026-09-16) — backtest/paper integrity parity

- **Consistent persistence:** `export_backtest_metrics_json` now emits the same
  self-excluding analytics SHA-256 seal as paper exports.
- **Regression coverage:** backtest export tests verify the digest after honesty
  stamping and metadata normalization.
- **Boundary preserved:** this seals persisted backtest artifacts without
  changing in-memory result schemas or introducing live-performance claims.
- **Verification:** `make ci` passed with 1,422 non-network tests, 87.45% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 71 DONE (2026-09-16) — canonical digest validation

- **Single trust boundary:** `validate_analytics_export` now verifies optional
  analytics SHA-256 seals for paper, backtest, and external callers alike.
- **Fail-closed behavior:** malformed or mismatched digest fields become
  validation errors instead of being silently accepted.
- **Regression coverage:** direct validator tests cover same-run tampering while
  preserving unsigned legacy export compatibility.
- **Verification:** `make ci` passed with 1,423 non-network tests, 87.27% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 72 DONE (2026-09-16) — API artifact trust boundary

- **Serving integrity:** `GET /backtest/{id}` now validates nested analytics
  exports through the canonical research-only and digest-aware validator before
  returning persisted results.
- **Fail-closed API:** poisoned honesty flags, missing schema, or tampered
  analytics seals return a structured 422 instead of being served.
- **Regression coverage:** API tests cover poisoned nested analytics alongside
  existing identity, path-containment, and parquet hash checks.
- **Verification:** `make ci` passed with 1,424 non-network tests, 87.40% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 73 DONE (2026-09-16) — outer API receipt integrity

- **Receipt provenance:** persisted FastAPI backtest receipts now carry a
  self-excluding SHA-256 digest covering identity, scope, metrics, and data
  references.
- **Lookup hardening:** `GET /backtest/{id}` rejects invalid or mismatched
  receipt digests before returning the artifact; legacy unsigned receipts remain
  readable for compatibility.
- **Regression coverage:** API tests cover emitted digest parity and same-receipt
  tampering in addition to nested analytics and parquet hash validation.
- **Verification:** `make ci` passed with 1,433 non-network tests, 87.44% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 74 DONE (2026-09-16) — research receipt read-race hardening

- **API fail-closed behavior:** `/research/latest` now converts post-verification
  unreadable, malformed, or non-object receipts into controlled 422 responses.
- **Regression coverage:** API tests exercise the verification/read race without
  weakening the canonical research artifact verifier.
- **Verification:** `make ci` passed with 1,435 non-network tests, 87.51% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 75 DONE (2026-09-16) — research API honesty envelope

- **Consistent API contract:** `/research/latest` now applies the same enforced
  `research_only=true` / `live_pnl_claim=false` envelope as other metric routes,
  even when a verified receipt contains poisoned flags.
- **Regression coverage:** endpoint tests cover both read-race failure and
  honesty-stamp behavior after successful verification.
- **Verification:** `make ci` passed with 1,436 non-network tests, 87.04% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 76 DONE (2026-09-16) — vendor microstructure receipt honesty

- **Receipt completeness:** candle/order-book vendor benches preserve both
  `book_source` and `book_dgp` in the returned research receipt.
- **Rebrand continuity:** user-facing docs and metadata consistently identify
  Dipcatcher as Artificial Hedge's proprietary research lab; executable legacy
  commands remain compatibility aliases.
- **Verification:** `make ci` passed with 1,463 non-network tests, 86.82% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 77 DONE (2026-09-16) — scientific display and provenance hardening

- **P-value honesty:** research receipts and CLI output render floating-point
  underflow as a strict lower bound instead of the misleading literal `p=0`.
- **Fixture boundary:** Northset synthetic fixtures explicitly opt out of the
  production adjusted-OHLC requirement; provider/production inputs remain
  fail-closed.
- **Vendor provenance:** mixed candle/book runs retain `data_source=SYNTHETIC`
  for synthetic candles while reporting `dgp=vendor_panel:<source>` for the
  external book component.
- **Verification:** `make ci` passed with 1,507 non-network tests, 85.58% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 78 DONE (2026-09-16) — non-parametric conditional density comparator

- **Distribution breadth:** added `ScaledEmpiricalDistribution`, which learns
  standardized residual quantiles and rescales them by the PIT-safe volatility
  covariate.
- **Proper diagnostics:** distribution benches now report empirical-residual
  CRPS, PIT KS, pinball, coverage, and a Student-t comparison without changing
  the operational scaled-t wrappee.
- **Safety boundary:** the comparator is explicitly research-only and carries
  no conditional-coverage or live-performance claim.
- **Verification:** `make ci` passed with 1,538 non-network tests, 86.09% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 79 DONE (2026-09-16) — density benchmark verification

- **End-to-end wiring:** standardized empirical residual scoring is exercised
  through the research notebook path, not only through isolated model tests.
- **Comparator integrity:** the new CRPS/PIT/coverage diagnostics coexist with
  the existing scaled Gaussian and Student-t paths without changing wrappee
  selection or FDR hypotheses.
- **Verification:** `make ci` passed with 1,538 non-network tests, 86.09% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 80 DONE (2026-09-16) — date-clustered calibration diagnostics

- **Panel inference hygiene:** added `grouped_mean_tstat`, collapsing correlated
  name-level observations into one miss-rate observation per date before HAC
  inference.
- **Localized conformal:** panel benches now expose date-clustered miss rate,
  t-statistic, p-value, and effective date count alongside legacy Kupiec fields.
- **Compatibility boundary:** legacy iid Kupiec fields remain unchanged; the
  clustered result is clearly diagnostic and does not mint a finite-sample
  coverage guarantee or promotion decision.
- **Verification:** `make ci` passed with 1,578 non-network tests, 85.96% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 81 DONE (2026-09-16) — panel calibration coverage

- **Online CRC:** date-clustered miss-rate HAC diagnostics now align with the
  original per-observation evaluation dates, while one update remains shared
  per timestamp.
- **Portfolio conformal:** date-level book sets now expose the same clustered
  calibration fields without pretending book dates are iid name observations.
- **Backward compatibility:** legacy Kupiec fields and H-table semantics remain
  unchanged; clustered fields are diagnostic only.
- **Verification:** `make ci` passed with 1,595 non-network tests, 85.34% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 82 DONE (2026-09-16) — clustered calibration becomes authoritative

- **Hypothesis semantics:** H16–H18 now prefer date-clustered HAC evidence for
  modern panel receipts, so correlated names cannot silently inflate the
  calibration sample size.
- **Compatibility:** legacy receipts without clustered fields still use their
  finite Kupiec values; verifier consistency markers remain backward-compatible.
- **FDR boundary:** the hypotheses remain in the calibration family, where
  failure to reject is the desired outcome; no discovery or promotion meaning
  was added.
- **Verification:** `make ci` passed with 1,616 non-network tests, 85.95% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 83 DONE (2026-09-16) — clustered receipt consistency

- **Verifier closure:** panel receipt consistency now triggers on either modern
  `date_clustered_p` or legacy `kupiec_p`, preventing clustered H16–H18 evidence
  from bypassing hypothesis-presence checks.
- **Compatibility:** the existing helper name and legacy error tokens remain
  stable for older receipts and downstream tooling.
- **Regression coverage:** clustered-only receipts are explicitly tested for
  fail-closed H16–H18 consistency.
- **Verification:** `make ci` passed with 1,631 non-network tests, 85.12% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 84 DONE (2026-09-16) — latest receipt pointer integrity

- **Pointer trust boundary:** when declared, `artifacts.json` in `latest.json`
  must resolve back to the receipt being validated.
- **Backward compatibility:** legacy receipts that omit the optional pointer
  remain valid; immutable run binding and SHA-256 checks are unchanged.
- **Regression coverage:** explicit latest-pointer tampering now fails closed.
- **Verification:** `make ci` passed with 1,657 non-network tests, 85.59% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness claim.

## Day Wave 85 DONE (2026-09-16) — artifact schema strictness

- **Artifact boundary:** present research-receipt artifact pointers must be
  strings, and present SHA-256 fields must be valid lowercase digests; omitted
  legacy optional fields remain compatible.
- **Fail-closed behavior:** malformed values are rejected before path or hash
  interpretation, preventing structurally poisoned receipts from being treated
  as valid evidence.
- **Regression coverage:** verifier tests cover malformed JSON pointers and
  artifact digests in addition to latest-pointer tampering.
- **Verification:** `make ci` passed with 1,706 non-network tests, 86.73% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness
  claim.

## Day Wave 86 DONE (2026-09-16) — Kyle-lambda alignment boundary

- **Alignment safety:** OFI and signed-depth Kyle-lambda date/value vectors are
  checked for equal length before correlation mapping; silent truncation is
  impossible, and mismatches produce a stable domain-specific error.
- **Regression coverage:** a malformed-vector test exercises the public
  correlation diagnostic fail-closed path.
- **Verification:** focused Kyle-OFI suite passes (21 tests), with Ruff, mypy,
  and diff checks clean. No performance or live-readiness claim.

## Day Wave 87 DONE (2026-09-16) — causal-panel marker hardening

- **Evidence boundary:** `validate_candidate` no longer treats a metrics
  `causal_panel=true` marker as causal evidence; only the structurally
  validated target-weight parquet panel can satisfy that gate.
- **Fail-closed behavior:** forged or stale metadata cannot substitute for the
  configured panel artifact.
- **Regression coverage:** explicit forged-marker rejection is covered in the
  validation-gate suite.
- **Verification:** focused validation-gate suite passes (32 tests), with Ruff
  and mypy clean. No performance or live-readiness claim.

## Day Wave 88 DONE (2026-09-16) — walk-forward evidence marker hardening

- **Evidence boundary:** `walk_forward_complete=true` no longer satisfies the
  causal-or-walk-forward gate by itself; valid positive date/fold counts or
  verified ranker evidence are required.
- **Fail-closed behavior:** completion metadata cannot substitute for temporal
  evaluation evidence.
- **Regression coverage:** explicit forged walk-forward-marker rejection is
  covered alongside the causal-panel marker test.
- **Verification:** focused validation-gate suite passes (33 tests), with Ruff
  and mypy clean. No performance or live-readiness claim.

## Day Wave 89 DONE (2026-09-16) — multi-fold marker hardening

- **Stability boundary:** `walk_forward_complete=true` without a positive fold
  count no longer satisfies the multi-fold stability gate.
- **Evidence semantics:** research-success fixtures now declare explicit fold
  evidence; completion metadata remains descriptive rather than authoritative.
- **Regression coverage:** marker-only multi-fold evidence fails closed, while
  valid two-fold evidence continues to pass.
- **Verification:** validation-gate suite passes (34 tests), with Ruff and mypy
  clean. No performance or live-readiness claim.

## Day Wave 90 DONE (2026-09-16) — completeness-marker hardening

- **Evidence boundary:** `evidence_complete=true` cannot substitute for finite
  core evidence when no verified immutable research receipt is present.
- **Required metrics:** receipt-less research validation now requires finite
  `mean_ic`, `net_spread`, and `turnover` values before `research_ok` can pass.
- **Regression coverage:** forged completeness markers fail closed while valid
  metric-backed cases remain covered.
- **Verification:** validation-gate suite passes (35 tests), with Ruff and mypy
  clean. No performance or live-readiness claim.

## Day Wave 91 DONE (2026-09-16) — fold/date semantic separation

- **Stability semantics:** ranker `n_dates` and `n_ic_dates` no longer satisfy
  the multi-fold stability gate; only explicit `n_folds` evidence does.
- **Evidence preservation:** date counts remain valid for temporal
  walk-forward coverage, but are not misrepresented as independent validation
  splits.
- **Regression coverage:** ranker date-count masquerading is explicitly
  rejected.
- **Verification:** validation-gate suite passes (36 tests), with Ruff and mypy
  clean. No performance or live-readiness claim.

## Day Wave 92 DONE (2026-09-16) — cumulative validation gate verification

- **Global verification:** cumulative marker, fold-count, date-count, and
  completeness hardening passes the full repository gate.
- **Verification:** `make ci` passed with 1,725 non-network tests, 86.54% total
  coverage, Ruff, format, and mypy clean. No performance or live-readiness
  claim.

## Day Wave 93 DONE (2026-09-16) — promotion data-source consistency

- **Promotion boundary:** `promotion_decision` now rejects missing or
  whitespace-only `data_source` values, matching champion receipt approval.
- **Normalization:** source labels are trimmed before synthetic detection and
  are emitted in normalized form, preventing layer-specific decisions.
- **Regression coverage:** missing-source and whitespace-padded synthetic
  source cases fail closed.
- **Verification:** registry and validation suites pass (56 tests), with Ruff
  and mypy clean. No performance or live-readiness claim.

## Day Wave 94 DONE (2026-09-16) — registry decision consistency audit

- **Audit result:** champion approval already requires the complete promotion
  receipt identity, while the lower-level metric decision remains intentionally
  receipt-independent for direct registry testing and validation composition.
- **Closed gap:** the lower-level decision now agrees with champion approval on
  non-empty normalized `data_source` and synthetic classification.
- **Verification:** registry/validation tests remain green; no promotion or
  live-readiness claim is inferred from synthetic evidence.

## Day Wave 95 DONE (2026-09-16) — champion-source normalization closure

- **Champion boundary:** `promotion_is_approved` now trims and normalizes
  `data_source` before checking synthetic eligibility and non-empty identity.
- **Fail-closed behavior:** whitespace-padded synthetic receipts cannot evade
  the champion safety barrier.
- **Regression coverage:** direct forged whitespace-padded synthetic receipt
  rejection is covered.
- **Verification:** registry/promotion suites pass (34 tests), with Ruff and
  mypy clean. No performance or live-readiness claim.

## Day Wave 96 DONE (2026-09-16) — leakage-flag coercion hardening

- **Leakage boundary:** promotion decisions now accept only the literal boolean
  `True` as a leakage pass; truthy strings, integers, objects, and other
  coercion-shaped values fail closed.
- **Receipt consistency:** emitted `leakage_ok` is now the same strict boolean
  used by the gate and champion approval.
- **Regression coverage:** string, integer, and object leakage flags are
  explicitly rejected.
- **Verification:** promotion/registry suites pass (35 tests), with Ruff and
  mypy clean. No performance or live-readiness claim.

## Day Wave 97 DONE (2026-09-16) — receipt version type strictness

- **Schema boundary:** receipt `schema_version` and provenance
  `benchmark_catalog_version` now require exact integer types; JSON booleans
  cannot exploit Python's `True == 1` equality.
- **Fail-closed behavior:** malformed version markers invalidate the research
  artifact before downstream evidence interpretation.
- **Regression coverage:** boolean markers are explicitly rejected for both
  version fields.
- **Verification:** research-verifier suite passes (232 tests), with Ruff and
  mypy clean. No performance or live-readiness claim.

## Day Wave 98 DONE (2026-09-16) — synthetic-flag type strictness

- **Provenance boundary:** an explicitly present `synthetic` field must be a
  literal boolean; strings, numbers, and objects cannot be interpreted through
  truthiness.
- **Fail-closed behavior:** malformed synthetic declarations invalidate the
  promotion decision while preserving the explicit data-source check.
- **Regression coverage:** malformed string synthetic flags are rejected.
- **Verification:** promotion/registry suites pass (36 tests), with Ruff and
  mypy clean. No performance or live-readiness claim.

## Day Wave 99 DONE (2026-09-16) — receipt source normalization

- **Provenance boundary:** research receipt source/synthetic consistency now
  trims and normalizes `data_source` before classification.
- **Fail-closed behavior:** whitespace-padded synthetic labels cannot evade the
  receipt synthetic mismatch check.
- **Regression coverage:** padded-source mismatch is explicitly rejected.
- **Verification:** research-verifier suite passes (233 tests), with Ruff and
  mypy clean. No performance or live-readiness claim.

## Day Wave 100 candidates

- Soft-verify H-table sprawl remains paused; OT / PERF / Acerbi invented-p prefer skip

## Definition of done (Day Wave 1)

- Docs updated on Mac checkout
- Highest-ROI fixtures/hardening landed
- SYNTHETIC `dipcatcher research` + paper smoke recorded in `day_grind_progress.md`
- `pytest -m 'not network'` green; ruff clean
- `INFLIGHT` cleared; Wave 2 gaps listed

## Day Wave 73 DONE (2026-09-16) — book panel load honesty

- `load_book_panel` fail-closed on missing parquet path (shared Northset/CLI/provider contract).

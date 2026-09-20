# Dipcatcher — Artificial Hedge's proprietary research lab

Dipcatcher is Artificial Hedge's **proprietary research lab**. It measures forecast quality with proper scoring rules. It does not exist to manufacture Sharpe ratios.

## Families (`dipcatcher research`)

| Family | Scientific scores |
|---|---|
| Ranking | Date-level IC / RankIC, HAC t, decile monotonicity. Public-feature challengers: `rff`, `rff_ridgeless`, `sdf_ridge`, `sdf_en`, `ipca`, `ipca_alpha`, `rp_pca`, `fnw`, `gx3pass`, `ds_lasso`, `fm`, `pcr`, `pls`, `tprf`, `gbrt`, `pp` (ADR-026/027/028). Champion remains public ridge until a non-SYNTHETIC card wins. Pairwise DM is on −IC, not Sharpe. |
| Alpha | Holdout MSE vs historical mean, Pearson IC |
| Volatility | QLIKE, Diebold–Mariano |
| Distribution | Pinball, CRPS, interval coverage, crossing, PIT KS (raw Gaussian + vol-scaled Gaussian / Student-t / standardized empirical residuals); 1d and 5d scored as separate keys |
| Regime | HMM AIC/BIC, holdout average log-likelihood |
| Tail | Historical VaR/ES; headline Kupiec is vol-scaled; unscaled kept as diagnostic |
| Drawdown | Brier, log-loss, ECE vs base rate |
| Liquidity | Amihud correlation, Almgren–Chriss vs TWAP shortfall |
| Reinforcement | LinUCB top-k on **public** features; regret vs planted oracle; advantage vs uniform |
| Conformal | Operational CQR/ACI/Mondrian wrap scaled (t) bands; `cqr_raw`/`aci_raw` show misspecification repair; `|Y|` slice is not an X-validity claim |
| E-values | Anytime-valid miss e-process on ACI sets (Ville); coverage, \(E_n\), ever-cross |
| Jackknife+ | Leave-one-out conformal coverage and width; finite-sample floor \(1-2\alpha\) (bound check, not a Kupiec null) |
| CRC | CRC on scaled (t) VaR bounds (same wrappee as two-sided); homoskedastic Gaussian is diagnostic |
| Weighted conformal | Likelihood-ratio split CQR coverage and width vs exchangeable CQR |
| Interval risk | Equal-weight \(1/n\) per date, then interval caps; bind_wide vs bind_tight (no P&L) |
| Quantile bandit | Quantile Thompson on **public** features; regret vs residual oracle |
| CV+ / JAW | Date-folded CV+; minmax floor (1-\alpha), plus/JAW floor (1-2\alpha) |
| Localized conformal | RBF-weighted CQR on PIT-safe volatility; coverage and width |
| Conformal rank sets | Date-grouped top-k set size, FDR, and oracle hit; no P&L |
| Online CRC | Sequential monotone-loss risk control by date; no cross-sectional time stacking |
| Portfolio conformal | One CQR set per date for the scalar book return (w^\top r) |
| Northset | OHLC / book / session identities; candle geometry + CLV; Parkinson / Garman–Klass / Rogers–Satchell / Yang–Zhang / overnight-split QLIKE; Kyle λ, Roll, Corwin–Schultz, Abdi–Ranaldo, Amihud, OFI, VPIN; liquidity sweeps; date-level ICs; BNS jumps |
| CPCV audit | Combinatorial purged/embargoed date folds; integrity only, no return claim |
| robinhood+ (optional) | Kronos-derived K-line path IC on split-adjusted OHLCV; hierarchical tokens + autoregression; not a live claim |

## What is not a lab headline

Sharpe, PSR, DSR, CSCV-PBO, and simulated P&L stay out of the research notebook. Those belong to optional execution backtests, not to the scientific benches.

## SYNTHETIC oracle

`planted_signal` at close \(t\) is a labeled residual oracle for \(r_{t+1}\). Recovering date-level IC is a **correctness** test of ranking only (H1), never a public-feature fit column. Alpha, drawdown, LinUCB, and Quantile Thompson train on `PUBLIC_FEATURES`. The H6/H14 headline is whether the bandit beats a **static public ridge top-k** on the same features and dates; vs-uniform and vs-planted (or y-greedy) regret stay diagnostics. Tag every such number SYNTHETIC.

## Hypothesis families

BH-FDR is **not** one table. **Calibration** (H4, H4b, H7, H8, H9, H11, H12): fail-to-reject is success. **Discovery** (H1–H3, H5, H6, H13, H14): reject is a finding. **H16–H18** (localized / online CRC / portfolio conformal) are panel Kupiec calibrations. **H19** top-k FDR is a bound check vs α. **H10 Jackknife+** is a coverage-floor boolean vs \(1-2\alpha\), not either family. **H20–H21 / H23–H24 / H29** are Northset identity bounds; **H22 / H25–H28 / H30–H38 / H43** are Northset discovery tests (including sweep IC, executable event studies, permutation placebos and range-variance DM); **H39–H42** are modeled-cost and chronological-stability bounds. **H44/H45** are matched-control sweep discoveries (event vs same-date eligible non-swept names; **H45 is the predeclared primary executable test**). **H46/H47** restrict those controls to the event's lagged dollar-volume quartile. **H48** is an out-of-time same-sign bound on the last chronological fold. **H49** is name-clustered follow-through inference. Discovery p-values enter BH-FDR together.

## Data labels

See [`docs/DATA_SOURCE_LABELS.md`](DATA_SOURCE_LABELS.md) for SYNTHETIC vs public/file labeling and fail-closed promotion rules.

## Overnight honesty (2026-09-15 → 16)

- **Benches** report coverage, width, IC/RankIC, pinball/CRPS, QLIKE, Kupiec/CC
  fixtures, etc. They do **not** headline Sharpe / Sortino / Calmar / NAV / P&L.
- **BH-FDR families stay split**: calibration vs discovery vs bound (H10
  Jackknife+ floor). Never pool into one FDR table.
- **SYNTHETIC** labels every planted-oracle / synthetic-bar number. Research and
  paper smokes are infrastructure / correctness — **not** live edge.
- **`live_pnl_claim=false`** on paper ledger, analytics export, promotion
  dry-run, and PERF benches. `--claim-live` on SYNTHETIC fails closed.
- **Dual catalogs:** research scorecard blobs forbid sharpe/sortino/calmar/pnl/nav
  key tokens (`family_blob_forbidden_metrics_absent`); paper `analytics_export`
  may carry equity/stress pnl/nav diagnostics but validators fail-closed on
  `live_pnl_claim=true` (see `validate_analytics_export`).
- Weighted / localized conformal benches: coverage + width only (no Sharpe keys).
- **Panel-or-skip:** `cv_plus`, `localized_conformal`, `conformal_rank`, `online_crc`,
  and `portfolio_conformal` in `run_research()` are wired to the **lab gold panel**.
  Standalone toy generators (Gaussian / exponential / planted-rank) are labeled
  `dgp=fixture` for unit tests only and **must not** share the research H-table with
  panel Kupiec tests (H4/H7/H8/H11/H12/H16–H18).
- **H16–H18** are Kupiec POF calibrations on panel coverage/risk (calibration family).
  **H19** is a bound check `FDR ≤ alpha` (bound family), not NaN-p discovery FDR.
- **Skip-on-nonfinite Kupiec (Day Wave 11):** H4/H7/H8/H11/H12/H16–H18 mint only when
  `kupiec_p` is finite; `None`/NaN/inf → omit the H-row (same as empty panel). Avoids
  NaN-p "consistent with nominal" success strings and FDR pollution.
- **Skip-on-nonfinite e-process / bounds (Day Wave 12):** H9 skips non-finite `e_sup`
  (`max(nan,1)` would fake p=1 success). H10/H15 skip non-finite coverage(/floor);
  H19 skips non-finite FDR (no "unavailable" mint).
- **Skip-on-nonfinite discovery DM/IC (Day Wave 13):** H1/H2/H3 and pairwise
  `H_rank_dm_*` mint only when the relevant p is finite; NaN/inf → omit (align with Kupiec).
- **Skip-on-nonfinite contrast inference (Day Wave 14):** H5/H6/H13/H14 require finite input scalars before contrast and skip mint when computed contrast `p` is non-finite (same discovery omit-over-unavailable hygiene as Wave 13 DM/IC).
- **`bench_tail` ES diagnostics (Day Wave 15):** research family blob reports `acerbi_szekely_z1`/`z2`, `fissler_ziegel_mean`, `es_hit_count` beside Kupiec (coverage α for FZ; miss 1−α for Acerbi Z1; primary prefers scaled). Research-diagnostic only.
- **`bench_tail` Christoffersen (Day Wave 16):** same holdout hits as Kupiec expose `christoffersen_ind_lr`/`ind_p` and `christoffersen_cc_lr`/`cc_p` (miss level 0.05; primary prefers scaled; `*_unscaled`/`*_scaled` mirrors). Honest NaN on empty/short. Research-diagnostic only.
- **H4b Christoffersen CC H-table (Day Wave 17):** mint `H4b_var_christoffersen_cc` when `_finite_number(tail.christoffersen_cc_p)` (calibration family; Kupiec-style skip-on-nonfinite). **CC only** — no separate ind hyp (CC nests ind+Kupiec; avoids BH calibration FDR double-count with ind). Never invent p.
- **`bench_distribution` CRPS e-process DM (Day Wave 18):** when `n_te>=3` and `dm_crps_*` are present, also surface research-only `e_dm_crps_final` / `e_dm_crps_reject` / `e_dm_crps_n` via Wave4 `e_process_dm` on the same per-obs quantile-CRPS losses (gaussian vs empirical); omit on failure; `research_only=True`; no `live_pnl_claim`. Prefixed to avoid vol-bench `e_dm_*` clash.
- **VaR-battery verify honesty (Day Wave 18):** nonempty `families['tail']` with `kupiec_p`/`kupiec_lr` must expose `christoffersen_cc_p`/`cc_lr` + preferred `christoffersen_ind_p`/`ind_lr` key presence (NaN ok); empty `{}` skips; missing → `tail_var_battery_incomplete:<key>` via `catalog.tail_var_battery_missing_keys` / `tail_var_battery_keys_present` / `verify_research_artifact`; soft scorecard `tail_var_battery_ok`. Not a live promotion gate.
- **`bench_distribution` scaled multi-model CRPS DM+e-process (Day Wave 19):** when `vol_20` path fits both ScaledGaussian and ScaledStudentT and `n_te>=3`, surface research-only `dm_crps_scaled_preferred` / `dm_crps_scaled_p` / `dm_crps_scaled_stat` and `e_dm_crps_scaled_final` / `e_dm_crps_scaled_reject` / `e_dm_crps_scaled_n` via `diebold_mariano` + Wave4 `e_process_dm` on per-obs quantile-CRPS losses; on e-process failure emit NaN/False/0 presence sentinels (Day Wave 20); `research_only=True`; no `live_pnl_claim`; does **not** change wrappee selection (diagnostics beside existing wrappee key).
- **Soft distribution CRPS e-process verify (Day Wave 20):** nonempty `families["distribution"]` with `dm_crps_p` and/or `dm_crps_scaled_p` (even NaN) must expose matching `e_dm_crps_*` / `e_dm_crps_scaled_*` key presence; helpers `dist_crps_eprocess_missing_keys` / `dist_crps_eprocess_keys_present`; verify → `dist_crps_eprocess_incomplete:<key>`; soft scorecard `dist_crps_eprocess_ok`. Empty `{}` skips. Research-receipt fail-closed on presence only — **not** a live promotion gate.
- **ES-battery verify honesty (Day Wave 21):** nonempty `families['tail']` with `es_95` OR `realized_es` OR `var_95` (even NaN) must expose `acerbi_szekely_z1`/`z2` + `fissler_ziegel_mean` + `es_hit_count` key presence (NaN ok); empty `{}` skips; missing → `tail_es_battery_incomplete:<key>` via `catalog.tail_es_battery_missing_keys` / `tail_es_battery_keys_present` / `verify_research_artifact`; soft scorecard `tail_es_battery_ok`. Orthogonal to VaR-battery (does not require Christoffersen). Not a live promotion gate.
- **H4b notebook consistency verify (Day Wave 26):** when `families['tail']` has **finite** `christoffersen_cc_p`, require hypothesis `H4b_var_christoffersen_cc` with `family == "calibration"`; missing → `hypothesis_h4b_missing_despite_finite_christoffersen_cc_p` via `catalog.h4b_hypothesis_consistency_errors` / `verify_research_artifact`. Non-finite / missing `cc_p` skips. Closes H-table regression risk vs Day Wave 17 mint. Not a live promotion gate.
- **H4 Kupiec notebook consistency verify (Day Wave 29):** when `families['tail']` has **finite** `kupiec_p`, require hypothesis `H4_var_kupiec` with `family == "calibration"`; missing → `hypothesis_h4_missing_despite_finite_kupiec_p` via `catalog.h4_hypothesis_consistency_errors` / `verify_research_artifact`. Non-finite / missing `kupiec_p` skips. Closes H-table regression risk vs agent mint. Not a live promotion gate.
- **H3 vol-DM notebook consistency verify (Day Wave 30):** when `families['volatility']` has **finite** `dm_p`, require hypothesis `H3_vol_dm` with `family == "discovery"`; missing → `hypothesis_h3_missing_despite_finite_dm_p` via `catalog.h3_hypothesis_consistency_errors` / `verify_research_artifact`. Non-finite / missing `dm_p` skips. Closes H-table regression risk vs agent mint. Not a live promotion gate.
- **Promotion receipt freshness binding:** promotion now requires the verified receipt's `provenance.git_worktree_sha256` to equal the current checkout fingerprint, in addition to the candidate `run_id` binding. Stale or unverifiable checkout state fails closed as `research_receipt_worktree_unbound`.
- **H11 CRC notebook consistency verify (Day Wave 34):** when `families['crc']` has **finite** `kupiec_p`, require hypothesis `H11_crc_var` with `family == "calibration"`; missing → `hypothesis_h11_missing_despite_finite_crc_kupiec_p` via `catalog.h11_hypothesis_consistency_errors` / `verify_research_artifact`. Non-finite / missing `kupiec_p` skips. Closes H-table regression risk vs agent mint. Not a live promotion gate.
- **H12 weighted CQR notebook consistency verify (Day Wave 35):** when `families['weighted_conformal']` has **finite** `kupiec_p`, require hypothesis `H12_weighted_cqr` with `family == "calibration"`; missing → `hypothesis_h12_missing_despite_finite_wcqr_kupiec_p` via `catalog.h12_hypothesis_consistency_errors` / `verify_research_artifact`. Non-finite / missing `kupiec_p` skips. Closes H-table regression risk vs agent mint. Not a live promotion gate.
- **H9 e-process ACI notebook consistency verify (Day Wave 36):** when `families['evalues']` has **finite** `e_sup`, require hypothesis `H9_eprocess_aci` with `family == "calibration"`; missing → `hypothesis_h9_missing_despite_finite_e_sup` via `catalog.h9_hypothesis_consistency_errors` / `verify_research_artifact`. Non-finite / missing `e_sup` skips. Closes H-table regression risk vs agent mint. Not a live promotion gate.
- **H10 Jackknife+ coverage notebook consistency verify (Day Wave 37):** when `families['jackknife_plus']` has **finite** `coverage`, require hypothesis `H10_jackknife_coverage` with `family == "bound"`; missing → `hypothesis_h10_missing_despite_finite_coverage` via `catalog.h10_hypothesis_consistency_errors` / `verify_research_artifact`. Non-finite / missing `coverage` skips. Closes H-table regression risk vs agent mint. Not a live promotion gate.
- **H15 CV+ floor notebook consistency verify (Day Wave 38):** when `families['cv_plus']` has **finite** `coverage` **and** **finite** `coverage_floor`, require hypothesis `H15_cv_plus_floor` with `family == "bound"`; missing → `hypothesis_h15_missing_despite_finite_coverage_and_floor` via `catalog.h15_hypothesis_consistency_errors` / `verify_research_artifact`. Non-finite / missing either field skips. Closes H-table regression risk vs agent mint. Not a live promotion gate.
- **H16–H18 panel calibration (Day Wave 82):** modern panel receipts prefer the finite `date_clustered_p` HAC test (one aggregated miss-rate observation per date) for H16/H17/H18; legacy receipts fall back to finite `kupiec_p`. Receipt consistency remains keyed to the legacy marker for backward compatibility. Not a live promotion gate.
- **H19 conformal-rank FDR bound notebook consistency verify (Day Wave 40):** when `families['conformal_rank']` has **finite** `fdr` and `dgp != "fixture"`, require hypothesis `H19_conformal_rank` with `family == "bound"`; missing → `hypothesis_h19_missing_despite_finite_fdr` via `catalog.h19_hypothesis_consistency_errors` / `verify_research_artifact`. Fixture DGP / non-finite / missing skips. Closes H-table regression risk vs agent mint (alpha defaults 0.20 if missing at mint). Not a live promotion gate.
- **Jackknife+/CV+ marginal coverage scope honesty (Day Wave 41):** `coverage_floor` / H10 / H15 are **marginal under exchangeability**, not training-conditional (Barber–Candès 2021; Bian–Barber 2023 caveat). Nonempty `bench_jackknife_plus` / `bench_cv_plus` and model metadata surface `coverage_guarantee_scope=marginal_exchangeable` + claim string; `research_only=True`; no `live_pnl_claim`. Soft-verify H-table sprawl remains paused.
- **Jackknife+/CV+ coverage_guarantee_scope receipt verify (Day Wave 42):** when nonempty `families['jackknife_plus']` / `families['cv_plus']` exposes `coverage` **or** `coverage_floor` (even NaN), require `coverage_guarantee_scope == "marginal_exchangeable"`; missing → `coverage_guarantee_scope_missing:<fam>`; wrong → `coverage_guarantee_scope_invalid:<fam>` via `catalog.coverage_guarantee_scope_consistency_errors` / `verify_research_artifact`. Empty `{}` skips. Parallel to VaR/ES battery presence gates. Not a live promotion gate; soft-verify H-table sprawl remains paused; no forged soft scorecard flag.
- **Soft scorecard executed/nonempty/finite_observation forge honesty (Day Wave 44 — parked/pre-landed):** `verify_research_artifact` fail-closes when scorecard claims those flags True but `catalog.family_blob_executed` / `family_blob_nonempty` / `family_blob_has_finite_observation` disagree — tokens `scorecard_executed_flag_forged:<fam>` / `scorecard_nonempty_flag_forged:<fam>` / `scorecard_finite_observation_flag_forged:<fam>`. Parallel to forbidden/battery forge; False/absent not forgeries. Agent `_benchmark_scorecard` DRYs to the same helpers. Not a live promotion gate; soft-verify H-table sprawl remains paused.
- **Closed-form Student-t CRPS (Day Wave 45; day_grind DayWave43 race):** research-only `crps_student_t` / `mean_crps_student_t` (Jordan–Krüger–Lerch / scoringRules; ν>2; honest NaN on bad σ/ν); distribution bench surfaces `crps_scaled_student_t_closed` beside quantile Riemann `crps_scaled_student_t` on scaled `vol_20` path. Dual view — not a replacement; not a live capital claim. Soft-verify H-table sprawl remains paused. (SOTA Wave45: sibling claimed Wave43 for `assume_sorted` history prefix; Wave44 soft scorecard forge parked/pre-landed.)
- **`assume_sorted` history prefix fail-closed (Day Wave 43):** `history_upto` / `history_for_calibration` / optimize trailing-hist refuse `assume_sorted=True` on nonempty unsorted frames (`ValueError` via `_require_assume_sorted_contract`); sorted assume_sorted still matches filter; no-`day_index` stays filter. One contract check on happy path. Not a live P&L claim; soft-verify H-table sprawl remains paused.
- **H1/H2 oracle ranking notebook consistency verify (Day Wave 31):** find `oracle_raw` in `notebook["rankers"]` (skip `_` names; **not** in families). Finite `p_ic` → require `H1_ranking_oracle` (`family == "discovery"`); missing → `hypothesis_h1_missing_despite_finite_p_ic`. Finite `ls_p` → require `H2_decile_mono` (discovery); missing → `hypothesis_h2_missing_despite_finite_ls_p`. Independent gates via `catalog.rankers_oracle_raw` / `oracle_has_finite_p_ic` / `oracle_has_finite_ls_p` / `hypotheses_include_h1` / `hypotheses_include_h2` / `h1_hypothesis_consistency_errors` / `h2_hypothesis_consistency_errors` / `verify_research_artifact`. Non-finite / missing / no oracle skips each. Closes H-table regression risk vs agent mint. Not a live promotion gate.
- **H7 ACI notebook consistency verify (Day Wave 32):** when `families['conformal']['aci']` has **finite** `kupiec_p`, require hypothesis `H7_aci_coverage` with `family == "calibration"`; missing → `hypothesis_h7_missing_despite_finite_aci_kupiec_p` via `catalog.h7_hypothesis_consistency_errors` / `verify_research_artifact`. Non-finite / missing / no `aci` skips. Closes H-table regression risk vs agent mint. Not a live promotion gate.
- **H8 Mondrian high-vol notebook consistency verify (Day Wave 33):** when `families['conformal']['mondrian_aci']` has **finite** `high_x_kupiec_p`, require hypothesis `H8_mondrian_high_vol` with `family == "calibration"`; missing → `hypothesis_h8_missing_despite_finite_high_x_kupiec_p` via `catalog.h8_hypothesis_consistency_errors` / `verify_research_artifact`. Non-finite / missing / no `mondrian_aci` skips. Closes H-table regression risk vs agent mint. Not a live promotion gate.
- **Soft battery_ok forge fail-closed (Day Wave 25; day_grind DayWave23):** if scorecard claims `tail_var_battery_ok` / `tail_es_battery_ok` / `dist_crps_eprocess_ok` **True** while the matching catalog helper says False, `verify_research_artifact` appends `scorecard_tail_var_battery_flag_forged:tail` / `scorecard_tail_es_battery_flag_forged:tail` / `scorecard_dist_crps_eprocess_flag_forged:distribution` (parallel to `scorecard_forbidden_flag_forged`). Absent or False `*_ok` flags are not forgeries; incomplete families still fail via `*_incomplete:<key>` only. Research diagnostic — **not** a live promotion gate.

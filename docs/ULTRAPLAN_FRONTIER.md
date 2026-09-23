# ULTRAPLAN — Frontier upgrades for dipcatcher

**Status:** active. **Created:** 2026-09-22. **Owner lane:** frontier expansion —
supersedes nothing; absorbs the open tails of `MEGAPLAN_SOTA.md` (phases A–G),
`MEGAPLAN_SHARPE5.md` (performance lane), and the `.dsh-24x7` industry-grade bar
into one ordered program.

## Mission

Make dipcatcher the most *rigorously evidenced* open quant research lab: the
largest set of independently-verifiable PROVEN claims across four bars.

| bar | current status | ultraplan target |
|---|---|---|
| Forecasting SOTA | PROVEN (scoped: Binance 1d+4h next-bar CRPS vs 4 TSFMs) | Wider & deeper: more cells, horizons, targets, published protocols, second domain |
| Industry-grade | NOT PROVEN (vbt parity+reliability win, ~1.8–5.4× latency loss; qlib parity+104× win; zipline not-fair) | PROVEN: close/argue latency, add NautilusTrader lane, UX+security evidence |
| Strategy performance | NOT PROVEN (carry dev Sharpe 1.7–2.1, holdout ~0; Sharpe-5 honestly failed) | Highest *honest* result achievable; multi-sleeve + vol-target + basis-carry lanes, locked holdout |
| Scientific infra | strong (receipts, hashing, MCS/SPA/DM, fail-closed) | frontier-grade: CI repro, mutation testing of money paths, adversarial self-audits as routine |

**Honesty contract (unchanged, absolute):** every claim traces to a receipt with
embedded hashes; negative results are recorded, not hidden; no live-PnL claim;
locked holdouts are never retuned. "Beating every hedge fund" is not a
verifiable claim — the plan instead maximizes *scoped, auditable* proofs a
skeptic cannot dismiss.

## Research basis (2026-09-22 scan)

Forecasting targets — the field moved since the current slate was frozen:
- **Moirai-2.0** (Salesforce, arXiv:2511.11698): decoder-only, quantile output,
  #1 non-leaking on GIFT-Eval MASE; `Salesforce/moirai-2.0-R-small` on HF.
- **TiRex-2** (NX-AI): xLSTM, zero-shot SOTA claims, ships **decontaminated
  checkpoints** (TiRex-2-g/-f) — lets us answer the contamination objection.
- **Sundial** (THU-MT, flow-matching generative), **Toto** (Datadog),
  **YingLong** (Alibaba), **TabPFN-TS** (PriorLabs, 11M, tabular-PFN
  formulation, strong covariate-informed results).
- Community benchmark framing: GIFT-Eval + fev-bench; TIME benchmark
  (arXiv:2602.12147) argues for task-centric strict zero-shot cells — aligns
  with our protocol.
- **"Heads, not backbones"** (arXiv:2606.30037): on fat-tailed returns at
  short horizons, output *distribution head* dominates architecture —
  Gaussian→mixture adds ~2.4% CRPS, largest in high-vol regimes. Directly
  motivates a mixture challenger: cheap, causal, exactly our lane.
- **Two-stage quantile-NN + spline CDF** (arXiv:2408.07497): SOTA-claiming
  distributional forecaster for stock returns; adaptable as a challenger.
- Conformal frontier: online-CP-via-online-optimization (Areces et al. ICML
  2025), KOWCPI kernel-weighted CP (ICLR 2025), O2CP cross-horizon CP,
  ResCP reservoir-reweighted CP — all extend our existing conformal stack.

Strategy lane — external evidence *confirms* our honest negative:
- BitMEX (Jan 2026): funding carry "post-yield" — compressed below ~4% on
  majors; edge migrated to TradFi-perp and CEX–DEX spreads.
- Funding-carry falsification studies (public, pre-registered): base trade
  net-negative OOS on BTC/ETH/SOL after costs; "the rule that barely trades
  is the one that works" — validates our hysteresis design choice.
- Survivors worth trying: **quarterly-futures cash-and-carry basis**
  (settlement-anchored, ~3%/yr unlevered, every settled contract positive in
  published in-sample), cross-venue funding (needs second venue's data —
  scope-dependent), cross-sectional **reversal** (crypto four-factor models
  find reversal > momentum), **time-series momentum** (stronger evidence than
  cross-sectional per SSRN 4675565).

Incumbent lane:
- **NautilusTrader** is the serious execution-realism incumbent (Rust core,
  backtest/live parity, order-book first-class) — the right third matched
  workload after vectorbt/Qlib. Conformance-replay framing (decision-matrix
  replay) is exactly our existing methodology.

## Phase structure & item ledger

Ordered roughly by unblocking value. Item IDs are stable — cite them in
receipts/progress. ✅/🔄/⬜ track state; ❌ = attempted, honestly failed.

### P0 — Close open evidence debt (integrity-critical; nothing else claims finality until this lands)

- [ ] P0.1 Poll remote fleets to completion: `h4f_*` (bnb/btc/xrp partial),
      `tfmfix_*` (d1 landed; deep cells in flight), `native nd_*/nh_*`
      (eth/xrp 4h pending), `s11_*` (not started). Respawn dead shards via
      `scripts/spawn_*.ps1`; capture stderr for any that die twice.
- [ ] P0.2 TimesFM contract-v2 splice: `scripts/splice_timesfm_fix.py` on all
      v4-daily + h4f cells; assert challenger/target non-TimesFM columns
      bit-identical; archive pre-splice matrices.
- [ ] P0.3 Contract-v2 re-merge all cells (`--bars-root data/raw/sources`),
      n_boot 2000, fresh receipts; update `EVAL_REPORT_SOTA.md` §4.1 table.
- [ ] P0.4 Native-protocol crossover merge (`sota_eval_native.py --merge-parts
      .dsh-24x7/native/nd_*`) → §4.3 ordering table (path RankIC, H-step
      RankIC, vol MAE/R²) with disclosed deviations.
- [ ] P0.5 `s11_*` + `s23_*` corrected-pairing seed replication fleet →
      §4.4/B1 closure.
- [ ] P0.6 Full-coverage 4h merge replaces complete-case receipt; student-t
      coverage row updated (was 63% NaN → hardened fitter 0 NaN claim must be
      *receipted*, not just probed).
- [ ] P0.7 Update PROOF.md SOTA section, HANDOFF.md, PROGRESS.md; flip
      EVAL_REPORT status from DRAFT when verdict checklist fills.

### P1 — Challenger frontier (beat targets by more, in more cells)

Each new challenger: causal-only, deterministic seed, honest-NaN on failure,
added to `BASELINES`/`dip_challengers`, unit-tested against known cases, then
a `v5_*` fleet column on existing shards' protocol.

- [ ] P1.1 `dip_gmm_k` — Gaussian-mixture density head (K∈{2,3,4}, EM on
      trailing returns, BIC or fixed-K; the "heads not backbones" result
      predicts +2–4% CRPS in high-vol regimes). Closed-form mixture CRPS
      (Grimit et al. 2006 identity) — no sampling noise.
- [ ] P1.2 `dip_skt` — Hansen/Fernández–Steel skew-t MLE (captures asymmetry
      that symmetric-t misses on crypto).
- [ ] P1.3 `dip_qar` — quantile autoregression (Koenker–Xiao) direct per-τ
      fit; monotone-quantile enforced.
- [ ] P1.4 `dip_conf_t` — conformalized Student-t: parametric base +
      weighted-online-conformal recalibration of residuals →
      distribution-free coverage correction (bridges our conformal stack
      into the distributional lane).
- [ ] P1.5 `dip_regime` — 2-state vol-regime mixture (existing `regime.py`
      HMM or Markov-switching): per-state empirical, state-prob mixed.
- [ ] P1.6 `dip_fhs_skew` — FHS variant on skew-filtered residuals + GJR
      asymmetry already present; quantile-level tail check.
- [ ] P1.7 `dip_isotonic` — isotonic-recalibrated empirical (PIT-based
      recalibration on trailing window; cheap calibration challenger).
- [ ] P1.8 `dip_lgbm_q2` — LightGBM quantiles v2: richer causal feature set
      (realized-vol term structure, OHLC range, amount), Dask-free, ≤30
      features; keep warmup disclosure.
- [ ] P1.9 Blend-search policy: `dip_blend` is empirical+parametric concat;
      add `dip_stack` — weights fit by *trailing-window* CRPS minimization
      (causal stacking, no lookahead).
- [ ] P1.10 h-step challengers for Phase D: vol-scaled h-bar distributions
      (σ√h + EWMA term-structure + Student-t tails; empirical h-day
      overlapping bootstrap) — honest constructions only.

### P2 — New published targets (make the claim harder to dismiss)

Each: pinned artifact + sha256, zero-shot, native output honored
(quantile/sample/path), per-model coverage disclosed.

- [ ] P2.1 `moirai2` — Salesforce/moirai-2.0-R-small via uni2ts; quantile
      head maps directly onto our CRPS/pinball path.
- [ ] P2.2 `tirex2` — NX-AI TiRex-2; prefer a decontaminated checkpoint for
      the fev-bench/GIFT overlap question; sample-path → distribution.
- [ ] P2.3 `sundial` — THU-MT flow-matching; sample paths → empirical dist.
- [ ] P2.4 `toto` — Datadog Toto if public weights resolve; else document
      unavailable.
- [ ] P2.5 `tabpfn_ts` — PriorLabs tabpfn-time-series (CPU-feasible, 11M).
- [ ] P2.6 `kronos_base` in the v5 fleet (only the v1 3-asset run beat it;
      fleet-scale evidence missing).
- [ ] P2.7 Classical neural baselines: N-BEATS / N-HiTS / DLinear via a small
      harness (Darts or direct) — closes the "only foundation models"
      objection.
- [ ] P2.8 Patch-Transformer reference line per arXiv:2602.06909 finding
      (generic transformer ~SOTA when pretrained at scale) — likely
      infeasible to pretrain; document as bounded.

### P3 — New cells & domains (widen the scope claim)

- [ ] P3.1 Hourly (1h) cell: deep bars exist remotely; same walk-forward
      protocol; watch microstructure-noise caveat (disclose).
- [ ] P3.2 Multi-horizon: h∈{1,5,20} daily + {1,6} 4h on identical origins
      (megaplan Phase D); targets emit native paths, challengers use P1.10.
- [ ] P3.3 Second domain: Stooq US equity dailies (remote `data/file_us`
      tapes) OR Binance non-USDT quotes — requires same bar-integrity
      hashing + availability-time discipline.
- [ ] P3.4 Cross-sectional lane: rank-IC eval vs targets on the panel
      (existing ranking bench + northset) — a different claim axis.
- [ ] P3.5 Volatility-forecast cell: QLIKE on next-bar/h-step realized vol —
      `dip_garch_t` already near-top CRPS; formal vol bench vs published
      vol baselines (HAR, realized-GARCH).

### P4 — Industry-grade bar (the open one)

- [ ] P4.1 Profile `run_backtest` on the 11-asset workload (cProfile +
      allocation trace); classify remaining 5.4× gap: interpreter loop vs
      per-order gate cost vs polars overhead.
- [ ] P4.2 Implement `run_backtest_fast` vectorized replay path for the
      *matched-workload class* (fixed rules: target-percent orders, next-open,
      no limits/stops) behind an explicit flag; must produce bit-identical
      NAV/fees on the conformance suite before use in any receipt.
- [ ] P4.3 If fast path can't reach ≤1× honestly, write the argument:
      per-order risk gates + fail-closed semantics are the product; vectorbt
      is a vectorized reducer without them; show latency decomposition
      table + the 3/3 fault-injection wins.
- [ ] P4.4 NautilusTrader conformance replay attempt (third incumbent):
      same bars/panel/costs; document matched or not-fair with receipts.
- [ ] P4.5 UX evidence: `dipcatcher doctor` self-check output, error-message
      quality suite, `--help` coverage vs incumbent CLIs/APIs.
- [ ] P4.6 Security evidence: `uv audit`/`pip-audit` receipt, secrets scan
      (gitleaks), no-`eval`/no-`pickle-load` audit, input-validation matrix.
- [ ] P4.7 Write the industry-grade verdict in PROOF.md only after P4.1–P4.6.

### P5 — Strategy performance (highest honest result; locked holdout)

Dev-window tuning only; the holdout stays locked. Negative results recorded.

- [ ] P5.1 Multi-sleeve dev study: carry + time-series momentum + x-sectional
      reversal (Kakushadze-style BTC-factor residual mean-reversion);
      sleeve-level risk-parity / vol-target overlay.
- [ ] P5.2 Vol-targeting overlay on carry book (target σ, realized-vol
      scaling, cap); dev only.
- [ ] P5.3 Quarterly-futures cash-and-carry lane: collect Binance delivery
      futures (`collect_perp_universe.py` extension); settlement-anchored
      basis capture — the one structural edge with positive published OOS.
- [ ] P5.4 Cost-side improvements: maker-fill assumption variant (limit-at-
      touch model already in SimulatedBroker — measure fee drag delta),
      hysteresis parameter robustness surface (not retuned on holdout).
- [ ] P5.5 Cross-venue funding/basis: gated on second-venue data
      availability; otherwise documented out-of-scope.
- [ ] P5.6 Capacity analysis: participation-capped fills × ADV → report max
      deployable AUM per sleeve (a real hedge-fund bar item).

### P6 — Full code audit ("every file can be made better" — verify or fix)

Audit order = blast radius. Each finding → fix + regression test, or written
waiver in the audit log. Output: `docs/AUDIT_FRONTIER.md` ledger.

- [ ] P6.1 Money paths: `simulated_broker.py`, `carry_engine.py`,
      `perp_engine.py`, `engine.py`, `sleeves.py`, `risk_gate.py`,
      `costs.py`, `implementation_shortfall.py`, `pnl_attribution.py`.
- [ ] P6.2 Statistical core: `scoring.py`, `inference.py`, `snooping.py`,
      `hac.py`, `evalues.py`, `conformal.py`, `multiple_testing.py`,
      `cpcv.py`, `purging.py`, `embargo.py`, `walk_forward.py`, `fdr.py`,
      `gates.py`.
- [ ] P6.3 Data integrity: `ingest.py`, `point_in_time.py`, `universe.py`,
      `corporate_actions.py`, `security_master.py`, `sources/`, `lake.py`,
      `calendars.py`.
- [ ] P6.4 Model layer: every file in `models/` vs its cited paper;
      `pipeline/train.py`, `pipeline/forecast.py`, `fusion/engine.py`,
      `labels/engine.py`, `features/`.
- [ ] P6.5 Exec/microstructure: `almgren_chriss.py`, `microstructure/*`,
      `northset/*` estimators (Kyle λ, Roll, VPIN, OFI).
- [ ] P6.6 Infra: `paper/*` (ledger atomicity, resume), `registry/`,
      `monitoring/` (drift, kill_switch), `api/app.py`, `cli/main.py`,
      `reporting/tearsheet.py`, `utils/*` (hashing, seeds, reproducibility).
- [ ] P6.7 Scale hygiene: `research/catalog.py` is 10.7k LOC — assess
      generated-vs-handwritten, dead code, duplication across
      `metrics/*`/`models/*` overlaps.
- [ ] P6.8 Perf sweep: cProfile top-20 hot paths across engine, features,
      scoring; fix only where semantics bit-identical.
- [ ] P6.9 Test-quality audit: mutation spot-checks on money-path
      conditionals; property tests (hypothesis) for accounting identities;
      coverage gaps in `tests/` map.
- [ ] P6.10 Dependency hygiene: pin audit, `uv audit` receipt, license
      scan, dead-dep removal.

### P7 — Frontier infrastructure upgrades

- [ ] P7.1 CI reproduction job: merge+inference is pure numpy — gated on the
      repo-policy decision (commit loss matrices + bar parquets or fetch
      from artifact store). Draft the workflow; flag for user.
- [ ] P7.2 Receipt v2 schema: unified `receipt.json` fields across eval,
      incumbent, carry, paper lanes (dataset hash, code hash, params,
      environment, `live_pnl_claim`, verdict).
- [ ] P7.3 Experiment registry hardening: mlflow.db exists locally — wire
      fleet runs into it or document why not.
- [ ] P7.4 Determinism sweep: BLAS threading notes already documented; add
      per-receipt `numpy`/`scipy`/`blas` fingerprint block.
- [ ] P7.5 Remote-fleet ops: consolidate `spawn_*.ps1` into one parametrized
      launcher + watchdog (auto-respawn dead shards, heartbeat file).
- [ ] P7.6 `AGENTS.md` refresh: remote conventions (powershell-only, WMI
      spawn, Defender exclusions, durable paths), durable staging dirs.

## Execution rules

1. **Receipts or it didn't happen.** Every checkmark needs a receipt file +
   embedded hashes + rerun command, per `.dsh-24x7` convention.
2. **No silent regressions.** Optimization must be bit-identical or the
   difference is disclosed and justified.
3. **Locked holdouts.** Nothing in P5 retunes on holdout. Ever.
4. **Honest negatives are deliverables.** A falsified lane closes with a
   documented falsification, not silence.
5. **One claim per scope.** Pooled mixed-interval inference stays refused by
   design; per-interval claims only.
6. **No repo-policy decisions for the agent.** Committing data/matrices,
      publishing, or external posting needs the user.

## Ordering

```
P0 (evidence debt) ──▶ P4 (industry-grade) ──▶ P3/P2 (wider SOTA)
P1 (challengers) runs parallel — v5 fleet waits on P0.3
P5 (strategy) runs parallel — independent compute
P6 (audit) interleaved — findings feed all lanes
P7 throughout
```

First executable tranche (this session): P0.1 polling loop; P4.1 engine
profile; P1.1 GMM challenger locally; P6.1 money-path audit start.

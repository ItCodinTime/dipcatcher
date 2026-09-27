# SOTA canon roadmap — September 2026 wave

Research-grounded gap analysis and implementation plan. Sources: 7-agent
swarm (4 repo recon lanes + 3 web-research lanes, 2026-09-27). This doc
complements — does not replace — `SOTA_GAP_ANALYSIS.md` (day-wave ledger)
and `REPO_IMPROVEMENT_PLAN.md` (economic-evidence sequence).

## 1. Verdict up front

The harness is already at or near SOTA across most of the validated-inference
stack: White RC / SPA / StepM / MCS, DSR/PSR/PBO, purged/embargoed CPCV,
BH/BY/Storey FDR, full conformal suite (CQR, CV+/Jackknife+, weighted,
localized, CRC, online CRC/ACI, conformal top-k), e-processes for
DM/loss-differential tests, PIT/pinball/CRPS/QLIKE first-class scoring,
Kupiec/Christoffersen/Acerbi–Székely backtests, ~330 research modules,
~4,900 unit tests. The remaining SOTA gaps are specific and named, not
generic. They are listed below with citations, ranked by leverage.

## 2. Gap matrix (researched method × in-tree status)

### 2.1 Anytime-valid / sequential inference — PARTIAL

| Method | Citation | Status |
|---|---|---|
| E-processes (single hypotheses, betting) | Shafer–Vovk; Ramdas et al. 2023 (arXiv:2210.01948) | EXISTS `metrics/evalues.py` |
| **e-BH** (FDR on e-values) | Wang & Ramdas, JRSS-B 2022 | **MISSING** |
| **Stopped e-BH** (anytime-valid FDR under optional stopping) | Wang & Ramdas 2025 (arXiv:2502.08539) | **MISSING** |
| **Online FDR with e-values** (e-LOND / e-SAFFRON) | Xu & Ramdas 2024 | **MISSING** |
| **E-detectors** (anytime-valid changepoint alarms) | Shin, Ramdas, Rinaldo 2023 (arXiv:2203.03532) | **MISSING** |
| Conformal test martingales (WATCH) | Prinster et al. 2025 (arXiv:2505.04608) | MISSING (future wave) |

Leverage: the `TrialLedger` runs RC/SPA/StepM/MCS per research day but has
no FDR control valid under **optional stopping** — exactly what a sequential
research grind does. E-detectors give alpha-decay/regime alarms with
time-uniform error control; natural fit for `monitoring/` and the paper loop.

### 2.2 Distributional forecasting / calibration — NEAR-SOTA, specific holes

| Method | Citation | Status |
|---|---|---|
| CRPS-from-quantiles / energy-consistent scoring | Gneiting & Raftery 2007 | EXISTS `metrics/scoring.py` |
| **Energy score** (multivariate proper score) | Székely 2003; Gneiting & Raftery 2007 | **MISSING** (variogram exists in `calibration2.py`) |
| Post-hoc distributional calibration (variance scaling, quantile mapping, isotonic-on-quantiles) | Gneiting et al. 2007; calibration literature | **MISSING** (`models/calibration.py` is classification-only) |
| **AgACI / FACI** (multi-expert ACI, dominates scalar ACI) | Zaffran et al., ICML 2022 (arXiv:2202.07282) | **MISSING** (scalar `AdaptiveConformal` only) |
| **NGBoost-lite / QRF** (tree-based distributional boosting) | Duan et al., ICML 2020; Meinshausen 2006 | MISSING (future wave — needs tree infra decision) |
| EnbPI (residual-bootstrap TS conformal) | Xu & Xie, TPAMI 2023 | MISSING (future wave) |

### 2.3 Search-aware evaluation (LLM-era) — STRATEGIC GAP

Gençay 2026 (arXiv:2608.27734): a deliberately leaky oracle with Sharpe 35
**survives Deflated Sharpe and PBO completely**. Fixes: structural
look-ahead exclusion (registry-validated feature space) + search-trial-count
deflation. fx-1 is exactly the LLM-search setting this targets. Plan: add a
synthetic leaky-feature strategy to the promotion battery + trial-count
deflation threshold next to `validate_candidate()`. (Own wave; touches
`hedge_lab/promotion.py` and `research/agent.py`.)

### 2.4 Engineering lane (from research-practices sweep)

Prioritized: (1) dataframe validation contracts at bronze→silver→gold
boundaries (pandera+polars — new dep, needs decision); (2) benchmark/perf
regression gate in CI; (3) GitHub Artifact Attestations on wheels (minutes);
(4) chaos/data-fault injection tests; (5) Hypothesis CI profile pinning
(`derandomize`/registered profile — cheap); (6) mutation testing on
`research/verify.py` + `catalog.py`; (7) harness-side model cards;
(8) verifier re-check CI step. See the engineering section of the swarm
report for citations (Pandera/GE landscape, pybroker asv pattern, SLSA).

### 2.5 fx-1 capability lane

fx-1 today is a rigorous integrity/compliance harness around a not-yet-trained
K3 LoRA; eval proves the model refuses to lie, not that it can reason.
Gaps vs 2024–2026 SOTA: no tool-use/agentic eval (FinTrace, FinToolBench),
no retrieval eval (FinAgentBench, FinanceBench), no time-series reasoning
eval (MTBench), no calibrated-uncertainty measurement, single-turn regex
scoring only. Sequenced on the fx-1 lane, not this wave.

### 2.6 Standing data gaps (unchanged, disclosed)

No PIT factor panel (factor covariance unwired); no intraday tape (all RV
is daily-bar fallback, northset sessions are synthetic reconstructions);
no true as-of vintages on the public tape; static sector map. These are
data-procurement items, not code items.

## 3. Wave plan

- **Wave 8 (this wave, 2026-09-27): anytime-valid inference + distributional
  calibration + multivariate scores.**
  - `metrics/anytime_fdr.py` — e-BH, stopped e-BH, online e-FDR (e-LOND style).
  - `metrics/e_detectors.py` — Shin–Ramdas–Rinaldo e-detectors (Gaussian,
    bounded, Bernoulli) with anytime-valid alarm control.
  - `metrics/energy_score.py` — multivariate energy score (+ optional
    threshold weighting).
  - `models/posthoc_calibration.py` — distributional recalibrators
    (Gaussian/t variance scaling, quantile mapping, isotonic-on-quantiles).
  - `models/agaci.py` — FACI aggregated adaptive conformal inference.
- **Wave 9 candidates:** NGBoost-lite + QRF baselines; leaky-oracle red-team
  protocol; EnbPI; WATCH conformal test martingales; regime-conditional
  evaluation gate in `validation/gates.py`.
- **Engineering waves:** hypothesis CI profile pinning; chaos/fault-injection
  property tests; benchmark regression gate; artifact attestations;
  mutation testing on the verify layer.
- **fx-1 waves:** capability-side eval (tool-use, retrieval, TS-reasoning,
  calibration measurement).

## 4. Acceptance for this wave

Each module: citation-style docstrings; pure numpy/scipy/pandas (no new
deps); fail-closed edges (short/empty inputs raise, never silently return);
unit tests with simulated size/power assertions (FDR ≤ α under global null;
coverage under drift; CRPS non-worsening on calibration); ruff + ruff-format
+ mypy clean on touched files; no commits (repo convention: owner reviews).

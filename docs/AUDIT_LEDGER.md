# Audit Ledger

Per-directory evidence for `quality/audit_coverage.json`. "Audited" means every
module was read and its risk-bearing paths verified against the honesty
contract: strict causality (no feature reads data after the decision origin),
fail-closed validation (degenerate input raises or returns a labeled NaN),
honest labeling (SYNTHETIC forced, forbidden metrics gated), and durable writes
(atomic, sealed where they are evidence).

| Directory | Verified | Findings → fix |
|---|---|---|
| `api/` | `app.py` line-by-line: loopback-only auth default + constant-time compare, TOCTOU-safe receipt verification (hash before+after), path-containment on configs/artifacts, request size limits, security headers, `_stamp_research_honesty` force-overwrite on every response | — |
| `audit/` | Merkle-chained ledger, flock'd cross-process appends, checkpoint integrity; Windows O_BINARY/LF-only fix | #286 stress tests, #293 Windows corrupt-ledger |
| `backtest/` | loop.py divergence bookkeeping, fast_replay byte-identity vs reference (P4.2), sleeves incl. residual_mr | #215 mark-price leak on non-executing names, #338 divergence-resume weighting |
| `calendars/` | Session/window arithmetic; trading-day math verified against exchange-calendar semantics | — |
| `cli/` | All 15 modules: argv-list subprocess only, no shell, config-error UX | #205 doctor UX |
| `compute/` | Per-task seed derivation (murmur-finalize mix) so process_map result order = item order; serial fallback on pool failure; task exceptions propagate | — |
| `config/` | safe_load, inherit path-containment, cycle detection | — |
| `data/` | PIT sources, atomic writes, universe tie parity, manifest seals | #176 P6.3 fixes, #261 manifest seal, #275 pooled_stream atomic |
| `diffbacktest/` | `delay>=1` structural gate; `end = t - delay` verified causal in every builder; adversarial radius documented as upper bound | — |
| `execution/` | Money-path fills, costs | #173 P6.6, #174 P6.1 (exec-NAV leak, leverage-cap trap) |
| `features/` | lot_spread eps dead-zone parity with zero_share | #337 |
| `formal/` | Z3-checked closed forms (NAV/cost/split/dividend); sim-only lifecycle spec — nothing submits orders; real-vs-IEEE gap documented | — |
| `fusion/` | Cross-fitted ridge stacking is OOF-only (leakage-safe); finite guards on stacker inputs/params | — |
| `hedge_lab/` | Every module line-by-line: slate lanes, book, race, hunt, tape, zoo | #320 same-bar look-ahead (delay>=1), #323 benchmark date-alignment + POSIX RAM |
| `hmm/` | Stochasticity contract, log-space variants | #307 EM underflow + contract enforcement |
| `labels/` | Triple-barrier observability, full-path gating | #212 (pipeline join), #325 last-bar events |
| `leakage/` | Scanner rule-by-rule | #339 (child lane: 14+ evasion classes) |
| `lightspeed/` | Vol measurement → sizing causality | #328 unmeasurable vol de-risks flat |
| `market_sim/` | Agent/book/simulator/ecology, AS quote guards, metaorder counters | #284 (child audit), #335 edge tests |
| `mc_engine/` | Philox counter streams, chunk-pure engine, checkpoint fingerprinting, TDigest/Welford/P² merges, POT/GPD | #268 KATs, #295/#319 coverage |
| `metrics/` | Proper-score suite, VaR backtests, inference (HAC/DM/MCS) | #206 clustered-variance, #304 propriety, #312 DQ, #315 KLM |
| `microstructure/` | Quote/impact estimators | #175 P6.5 |
| `monitoring/` | PSI drift emits `insufficient_data` not false all-clear; kill_switch: unknown state blocks, invalid transitions raise, auto-flatten never on exception | — |
| `native/` | env-gated dispatch; docs honestly separate bit-exact kernels (cumsum, wealth, SHA-256) from tolerance kernels (EMA/RSI/book-OLS) | — |
| `northset/` | All 9 modules: estimators (Kyle/Roll/Parkinson/GK/RS/YZ/CS/AR/CKS/BNS formula-verified), CKS OFI, sweep battery (executable next-open timing, cross-sectional demeaning, PIT vol regimes, matched eligible controls, predeclared primary test) | #317 flat-bar junk + VPIN remainder |
| `observe/` | OTLP export error swallowing, log redaction | #336 |
| `paper/` | loop, ledger, sim_live resume cursors, divergence merge, atomic receipts | #258 seals, #303 fail-closed cursors, #338 |
| `parity/` | `__main__` dual-clock smoke, fail-closed report | — |
| `pit/` | Vault enforcement | #289 (child: 12 enforcement gaps) |
| `pipeline/` | All 21 modules: purged walk-forward (observed label-end fallback), asof-bounded everything, unfitted-clone GARCH/RGARCH refits, spec-bytes + full-history digest caches, named-estimator family/spec/object pinning, PIT history builders (strict `<` origins, null `available_time` fail-closed, duplicate-key rejection on the full frame), labeled homoskedastic fallback emitted per-row | #213 (calibration unrealized-label fallback) |
| `portfolio/` | Optimizer (infeasible diagnostics), allocators (PSD cov), factor betas (trailing ridge), conformal (chrono split), attribution (prev-bar weights) | #332 non-PSD refusal, fingerprint framing |
| `pretrade/` | Hot risk-gate surface | #287 (child audit doc) |
| `proof/` | HMAC env-only, merkle canonical dumps | — |
| `proofcore/` | Hash-chain export ordering | #279 wall-clock → genesis walk |
| `quant_models/` | BS/Greeks/GEX/HRP/TSMOM/risk-parity/MC; causal TSMOM, honest receipts | — |
| `reality/` | Bailey–LdP PSR/DSR/MinTRL, CSCV/PBO ω-logit, SPA, BH-FDR — all verified against papers | — |
| `registry/` | MLflow store: artifact identity binding, research_only claim tags on every run | — |
| `reporting/` | Tearsheet return-chain integrity | #326 interior NaN breaks chain |
| `risk/` | Overlay causality, gate stack, ruin latching | #308 ruin latch, #310 fail-closed tail gates |
| `robustness/` | Closed forms, swarm invariants | #264 KATs |
| `schemas/` | Fail-closed pydantic validators (finite, tz-aware, book ordering, crossed-book rejection) | — |
| `simtest/` | Deterministic exchange timeline, IEEE-hex event log, real ddmin shrinking, replay re-derivation, conservation rebuild | — |
| `stress/` | replay/reverse/bootstrap/garch_copula/jumps | — |
| `utils/` | Hashing, atomic io | #161 selector injection, #332 fingerprint framing |
| `validation/` | Purging/embargo/CPCV/walk-forward/FDR line-by-line vs papers | #211 (P6.2 audit + 22 KATs) |

## In progress (child lanes)

| Directory | Status |
|---|---|
| `models/` (142 modules) | Child session `b95e8bcbd3a0426285a47725ec48086c` — estimator-by-estimator verification |
| `research/` (29 modules) | Child session `3dca4dedd8634a5182f3e7a94e0cd15d` — lane/receipt audit |

## fx1 (the gated product — `quality/audit_coverage_fx1.json`)

Parallel manifest, same schema + ratchet tests: 65/66 modules `audited`
(`__init__.py` waived as package surface). Directory pins: eval 14, forecast
11, bench 3, serve 5, data 14, train 9; root modules audited individually.

| Area | Verified | Findings → fix |
|---|---|---|
| `honesty.py` + `data/` | Child audit — regex evasions, corpus screening, contamination floor, vacuous honesty gate | #321 |
| `eval/` | Deterministic seeded banks, contract-validated prompts, unparseable answers counted (never propagated), degenerate forecasts fail closed NaN, `passed` requires finite ECE AND finite \|Z\| | — |
| `serve/` | HMAC-env signing (never hardcoded), fail-closed compare_digest verify, structural TEE + zkML manifests with honest crypto delegation | #321 (TEE existence-only status) |
| `bench/` + `forecast/` | `bench/dip.py` verified: causal dip detection (fires below running peak only), recovery windows bounded at data end → `None` never imputed, out-of-range probabilities raise, `assert_bench_output_honest` mirrors the forbidden-token contract inside the bench itself; `forecast/evaluate.py`: `assert_no_label_overlap` per fold, walk-forward fails closed on empty folds, finite-masked metrics; `runner.py` decision-time machinery (`_at_decision`, late-release flag) | — |
| `train/` + root modules | `train/pipeline.py`: hard stage gates, mandatory pre-training honesty gate (`eval_base` blocks TRAIN when `honesty_gate_passed` is false), corpus dedup/decontamination vs eval prompts, frozen split manifest; `train/receipts.py`: receipt pins SHA-256 of config/corpus/split/eval_base + git rev + dirty flag + env fingerprint with forced `live_pnl_claim=False`/`research_only=True`; `default_trainer` fails closed (no local GPU trainer — injectable); `reward.py`: executable honesty-contract reward (forbidden-headline/live-claim/synthetic-unlabeled → low score); `hypotheses.py`: only gate-resolved traces admissible; `mrm.py`: dossier sections cite artifact hashes, missing artifact fails closed | — |

## Rules

- A directory may only move `partial → audited` when every module has a row of
  evidence here or in a linked audit doc (`AUDIT_P6*.md`, child audit tables).
- `test_audited_dirs_pin_file_count` fails the build if an audited dir gains or
  loses a module without a manifest update.
- `AUDITED_FLOOR` ratchets only upward.

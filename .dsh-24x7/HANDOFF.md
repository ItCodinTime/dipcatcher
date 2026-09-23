# 24x7 handoff

- job: 24x7-9339a2c6-d6ca-40a0-9905-dfa37ff74c8d
- status: running
- reason: startup-resume
- session: session-24x7-792ea98c-4173-4a06-89dd-bd4e5297f84b
- updated: 2026-09-19T11:42:29.962Z
- merge note: this file was merged across two workers — local job above (rounds through 2026-09-23) and codex-remote checkpoint job 24x7-18bdfaf2-ac63-4d1c-812f-4332016b2709 / session-24x7-78c38d2b-1eee-449f-afe7-825121d09aa2 (updated 2026-09-21, `D:\dipcatcher`). Both narratives preserved below.

24x7 continuous work on /Users/vaithianathan/dipcatcher until both bars are honestly proven.
Bar 1 — industry-grade overperformance: beat real production systems on the project's actual quality bars (correctness, tests, latency, reliability, UX, security, operability). Compare against named incumbents, not vibes.
Bar 2 — ultimate SOTA: match or beat the current published state of the art for this domain, with numbers, papers, repos, and rerunnable evals. If the project is not a research artifact, SOTA means the best public implementation of the same product class.
Do not mark the goal complete until `.dsh-24x7/PROOF.md` exists with both `## Industry-grade` and `## SOTA` sections set to `STATUS: PROVEN`, each citing at least one live URL and a command whose output you captured.
Self-praise is not proof. If evidence is missing, keep working. Persist progress in `.dsh-24x7/PROGRESS.md` and `.dsh-24x7/HANDOFF.md` after every meaningful round.

## Current state (2026-09-22, continued)

- SOTA bar: `STATUS: PROVEN` (scoped — Binance crypto return-distribution forecasting, daily + 4h). Two independent fleets agree: v3 (150 origins/cell, 2400 origins) and v4 (300 origins/cell, 3300 daily + 1500 4h complete-case, corrected Kronos tokenizer pairing). v4 receipts `.dsh-24x7/eval-full/` (per-asset shards + `d1_merged.json`/`h4_merged.json` + loss matrices); v3 receipts `evidence-sota-eval-v3-*` + `eval-shards/`. Both exclude all four published targets (Kronos-small, Chronos-2, Chronos-Bolt-small, TimesFM-2.5) from MCS @0.10; MCS retains only `dip_garch_t`/`dip_fhs`. v4 seed runs (11/23) replicate the sweep. v1 receipts also beat Kronos-base.
- Industry-grade bar: `STATUS: NOT PROVEN`. Matched workload vs vectorbt 1.1.0 executed under `.venv-bench` (vectorbt + plotly 6.9.0 pin — vbt breaks on plotly ≥7). Receipts `.dsh-24x7/evidence-incumbent-vectorbt{,-11a}.json`: NAV parity ≤2.7e-15 / fee parity ~1e-11 / identical fill counts on 3- and 11-asset workloads. **Fast replay `src/quant_fund/backtest/fast_replay.py` (2026-09-22): bitwise-identical to `run_backtest` (reproduces CPython 3.12 compensated `sum()` + np.float64 contamination after participation caps; suite `tests/unit/test_fast_replay.py` 7/7; 11-asset real-bar NAV+fills bit-identical via `scripts/_conformance_11a.py`). Latency ~3.7× faster than ref: 358ms vs 1337ms remote (11-asset); vs vectorbt the gap narrowed to **195 vs 158ms (1.23×, 3-asset) / 358 vs 159ms (2.25×, 11-asset)** quiet; under the active SOTA fleet load (interleaved reps, `-v2` receipts) ~2.2×/4.7× — was ~43×/5.4× pre-optimization)** — residual is the per-order risk-gate evaluation vbt lacks. `dipcatcher backtest --engine fast` CLI selector added. **Lineage fix (round 3):** remote `run_backtest` carries sparse weight grids + flattens unmarked names + market-vol gate — fast path now detects the deployed engine's lineage via the `risk_overlay` signature param and replicates all three (sparse-panel probe now bitwise-identical, 2190/2190 fills; suite 9/9). Bench panel densified with explicit 0.0 rows so both engines interpret identically. Reliability win 3/3 fault-injections (dup rejects, StaleValuationError, kill-switch — vbt silently accepts all). Qlib 0.9.7 executed locally (`scripts/incumbent_bench_qlib.py`, receipt `evidence-incumbent-qlib.json`): **NAV parity at the float32 quote floor (1.03e-7 rel, $0.18 abs)** after two fixes — a custom `MatchedOrderGen` (qlib defaults renormalize weights to sum=1 and floor amounts to integer units) and `limit_threshold=1e9` (`None` silently falls back to cn-region 0.095 → dropped 25 orders on >9.5% days, the entire original 14% divergence; pre-fix receipt preserved at `evidence-incumbent-qlib-w03.json`). Latency: **dipcatcher ~104× faster** (101 ms vs 10.6 s). Zipline documented not-fair (see PROOF.md). Security: `uv audit` 172 pkgs 0 vulns; artifacts sha256-sidecar+manifest+type gated; Kronos weights all safetensors; secrets scan clean. Remaining for the bar: residual ~2.1× vectorbt latency gap, UX surface, no pre-commit secret-scan hook.
- Industry-grade blockers union (codex-remote checkpoint 2026-09-21, still-open items merged in): missing licensed PIT lineage, authenticated broker/FIX reconciliation, venue TCA/liquidity/borrow/financing/failure data, measured production SLOs, signed promotion, and GIPS verification. Objective guardrails restated there and still binding: keep improving correctness, fail-closed behavior, tests, type quality, reproducibility, latency evidence, governance, and research-protocol honesty; no `STATUS: PROVEN` claim without live references and captured comparable command output; preserve all pre-existing uncommitted changes, including user-added hedge-lab, lightspeed, quant-model, decision, and artifact files. On 2026-09-21 SOTA was recorded as blocked by missing apples-to-apples real-market evaluations — subsequently run (v3/v4 fleets below), hence the scoped PROVEN above.
- Remote compute: eval ran on `codex-remote` (Windows, `D:\dipcatcher` checkout, `D:\evalenv` uv-managed CPython 3.12.14 — the Microsoft Store Python shim suspends spawned children; Defender exclusions were needed for throughput). Model artifacts under remote `data/models/`; bars under `data/raw/sources/` on both machines. NOTE: `D:\dipcatcher` source tree has diverged (no `data/sources/` subpackage — it's the hedge_lab lineage); `D:\evalenv` resolves `quant_fund` editable from it, so `collect_binance_deep.py` fails there — collect deep bars locally and scp.
- SOTA megaplan (`docs/MEGAPLAN_SOTA.md`) executing 2026-09-22: A1 coverage verified (0 NaN/750); B3 robustness banked (`evidence-sota-robustness.txt` — MCS invariant over blocks/seeds, SPA p_upper at floor on aligned panels); timestamp-recovery gap fixed in `_recover_target_times` (`-4h` suffix accepted; use `--bars-root data/raw/sources` for local merges — per-interval only, pooled mixed-interval inference is refused by design); Phase C native-protocol fleet `nd_*`/`nh_*` + full-coverage `h4f_*` fleet running remote under `D:\evalenv` pythonw via WMI spawn (canonical Kronos-Tokenizer-base pairing). h4f_btcusdt died once post-weight-load — respawned; watch for repeats.
- Kronos artifacts live under `data/models/` (gitignored); `dipcatcher collect` fetches the public bars; everything is research-only, no live-P&L claims.
- Eval rerunnability note: `--kronos-repo` currently points at `/private/tmp/Kronos` (a clone of shiyu-coder/Kronos). For a durable rerun, re-clone `https://github.com/shiyu-coder/Kronos` to a stable path and pass that. The `.npz` loss matrices let DM/SPA/MCS be re-derived without the model at all — the pooled merge reproduces exactly on this checkout.
- Institutional tranche (2026-09-22, post-receipt hardening — no eval artifacts touched): added `portfolio/pnl_attribution.py`, `portfolio/factor_model.py`, `execution/implementation_shortfall.py`, `paper/recon.py`, `reporting/tearsheet.py` (+ `dipcatcher tearsheet` CLI), and full limit-order lifecycle in `SimulatedBroker` (resting sweeps, residual PARTIAL fills, cancel, `expire_time`, `bar_time` stamping). 24 new tests in `tests/unit/test_institutional.py`; full unit suite green; ruff/mypy clean.
- Ops monitoring (2026-09-23): `monitoring/dashboard.py` `ops_snapshot()` — consolidated ops board (limit utilization warn≥80%/breach>limit, unmarked positions, mark staleness, kill state, drawdown, open orders, recon mismatches, drift) with `ok`/`warn`/`breach`/`insufficient_data` rollup, fail-closed on missing data; `render_markdown` + `dipcatcher monitor` CLI reading paper `broker_state.json`/`equity.parquet` with broker↔ledger NAV recon built in (verified end-to-end on a synthetic run). 5 new tests (30 total in `test_institutional.py`) pass locally + on `codex-remote` (pytest installed in `D:\bench-qlib`; `D:\evalenv` has no pip — use the bench venv or `sys.path` runner for remote tests).
- Sharpe-5 megaplan lane (`docs/MEGAPLAN_SHARPE5.md`, receipts `.dsh-24x7/evidence-carry-{1d,1h}.json` from `codex-remote` `D:\dipcatcher-megaplan`): carry book made economically coherent — fixed the phantom-spread liquidation model (perp-high/spot-low fabricated a $1.34M cascade on 2021-04-18) and fixed-unit notional drift (4.4× NAV gross) via coherent hedge unwind + `rebalance_band=1.5`; added per-symbol P&L attribution (conservation ~1e-10) and segment-eligibility gating (dead-tail symbols excluded, engine still fail-closed). Result: 1d dev Sharpe 1.74 / holdout −0.70; 1h dev 2.07 / holdout −0.11; zero liquidations. **Sharpe-5 NOT achieved — honestly recorded in PROOF.md `## Strategy performance`.** No holdout retuning performed.

## 2026-09-21 — codex-remote checkpoint (merge-retained; job 24x7-18bdfaf2)

Recorded on `D:\dipcatcher`; at that checkpoint both proof bars were explicitly unproven (see blockers union in Current state).

Completed and failed attempts at that checkpoint:
- Canonical pytest paths are defined in root `pytest.ini` as `tests/unit`, `tests/property`, `tests/regression`, and `tests/end_to_end`; fresh `pytest --collect-only -q` succeeds without duplicate-module errors. The canonical non-network run is active as job `pwsh-20`.
- Ruff passes on canonical suites; project-wide mypy passes all 197 source files after minimal fixes in the requested typing areas plus the dtype-variable fix in `lightspeed_book.py`.
- Focused changed-area tests pass: 128 tests, including all 10 RGARCH risk-gate tests. The tests assert causal overlay precedence and observed rejection/source changes without incorrectly requiring zero fills from latent one-step RGARCH forecasts.
- The benchmark/protocol-honesty gate passes 30 tests; Python compilation passes; canonical collection lists the full suite without duplicate-module errors.
- Synthetic paper-loop completed: 40 steps, 25.7697 steps/sec, valid ledger, but simulated-only with no matched incumbent workload. A fresh non-overwriting rerun completed at 28.2216 steps/sec with the same 960-order/valid-ledger shape; both remain diagnostic-only. Frozen GARCH selected `gjr_t`, but `sota_proven=false`, `beats_all_baselines=false`, and all 30-date comparisons were non-significant.
- Security-tool availability was checked without claiming a result: `pip-audit unavailable`, `bandit unavailable`, and the virtualenv has no `pip` module (`python -m pip check` cannot run).
- Targeted operational paper-loop, fail-closed risk, and honesty-policy tests pass.
- Live reference fetches returned HTTP 200 for Chronos, Moirai/Uni2TS, TimesFM, and GIPS standards. They establish reference availability only; no matched SOTA or industry proof was claimed.
- An earlier broad pytest invocation with `testpaths = [tests]` failed with 454 duplicate-module collection mismatches from the tracked `tests/tests` mirror; that output is invalid after the discovery fix.
- Next executable step recorded then: collect canonical non-network pytest job `pwsh-20`; record final pass/fail count, warnings, elapsed time, and any real failures in `.dsh-24x7/PROGRESS.md`, then repair only confirmed failures.


## 2026-09-22 late — contract-v2 merges banked; native fleet incident + respawn

- All primary cells re-merged under scoring contract v2: v3-d1 (`evidence-sota-eval-v3-d1-tfmv2.json`, MCS={6 chall}, all targets out), v4-d1 (`evidence-sota-eval-v4-d1-v2.json`, kronos-canonical + timesfm-v2 both spliced), h4f (`evidence-sota-eval-h4f-v2.json`, 1500/1500 complete — student-t NaN gap eliminated). Challengers beat every target on every cell; weakest DM pair timesfm-vs-dip_lgbm_q p=0.0204 (d1) / 0.5984 (h4, disclosed).
- NEW INCIDENT: `sota_eval_native.py` freq inference divided ns by 3.6e15 (should be 3.6e12) → all files classified "1h" → first native pass ran lookback 80/horizon 12 everywhere (paper: 40/12 d1, 90/18 4h). Caught via receipt audit (`lookback: 80`). Fixed + respawned as `spawn_native4.ps1` with explicit `--lookback/--horizon`; old receipts superseded (same filenames overwritten).
- Fleets running on `codex-remote`: native4 (16 jobs, corrected protocol, ~2.5h), s11 (5 corrected-pairing seed-11 daily jobs, ~2h).
- Next: merge native4 shards → `evidence-sota-native-*.json` → fill EVAL_REPORT §4.3; merge s11 → seed-replication row; then Phase D (multi-horizon) decision + G scope.

## 2026-09-22 late — remote suite stall diagnosed + Windows portability fixes

- Remote full-suite "stall" at ~45% root-caused: NOT a hang. The `test_mean_session_*`/`receipt_stamp`/`soft_verify` files each invoke the full `northset` CLI pipeline (~104s per file locally; multi-minute under the running collector fleet). Same position on local + remote runs. Detached `-v` run left running: log `D:\dipcatcher-megaplan\_unit_v.log`, spawned via `D:\dipcatcher-megaplan\_run_verbose.ps1` (WMI — survives ssh drops).
- Two pre-existing Windows-only bugs fixed in `src/quant_fund/paper/ledger.py` (remote Fs, not regressions):
  1. `_atomic_write_parquet` fsynced a read-only fd — `os.fsync` on `"rb"` raises EBADF on Windows. Now writes parquet through its own fd and fsyncs the write fd.
  2. `paper_root` path validation missed POSIX-style `/abs/path` on Windows (drive-less anchor isn't `is_absolute()`). Now rejects POSIX anchors + Windows drive/root anchors via `PurePosixPath`/`PureWindowsPath`.
- Verified: failing cluster (ledger_schema, localized_conformal*, math_extremes_wave3, analytics, api/audit/ledger fail-closed) green remotely post-fix; focused touched-file suite green remotely; local ruff/mypy clean.
- Watch: stray `python3.12.exe` children resolving to the Microsoft Store `WindowsApps` shim appear under pytest runs on this host — Store-python quirk, correlates with suspended-looking processes.

## 2026-09-23 — challenger arena +6, verified-contract merges, native daily banked

- Challenger expansion complete: dip_gmm_k/skt/qar/conf_t/regime/stack all
  computed on 27/27 shards -> .aug.npz (17-20 cols). dip_stack = causal
  convex stacking (per-tau exponentiated-gradient simplex weights, trailing
  buffer, Vincentized quantile blend, deterministic — remote==local bitwise).
- VERIFIED arena (v2+spliced contract, inference computed):
  * merge_d1_v2aug.json (deep-5 x300): MCS = {fhs, garch_t, regime, qar,
    conf_t, blend, empirical, student_t}; all published models excluded.
  * merge_h4f_v2aug.json (5x300): dip_stack 3rd AND in MCS; 14 challengers
    in superior set; all published models excluded.
  * Honest read: frontier is a TIER — no single new challenger beats
    fhs/garch_t, but regime/qar/conf_t (d1) and stack (h4f) are
    statistically indistinguishable from them. Published models are
    outside the tier at both horizons under verified contract.
- Descriptive arena (legacy contract, inference withheld by honest gate):
  merge_{d1fix,h4f,h4fix}_aug2.json + merge_{seed11,seed23}_aug.json.
- Native daily crossover BANKED (P0.4 daily half): merge_native_d1.json,
  3300 origins, corrected 40/12 verified in receipts. dip_garch_t wins
  path RankIC (0.0402 vs timesfm 0.0296, kronos 0.0127); challengers
  dominate vol MAE; published models lead ret-direction RankIC — mixed,
  disclosed. nh_* 4h half still in flight (150/300).
- s11 seed replication BANKED: merge_s11.json (5x300, seed=11) replicates
  seed-7 ordering exactly; pre-v2 TimesFM contract disclosed
  (legacy_unverified — spawned before the v2 push).
- IN FLIGHT on codex-remote: nh_* 4h crossover (150/300), s23 corrected
  replication (seed=23/300/16/2000 confirmed live, ~50-100/300), tfmfix4
  wave (6 non-deep v4 daily assets — completes the verified 11-asset
  daily arena when merged).
- Funding observability fixes: funding_events_dropped counter in
  carry_engine + perp_engine (off-bar funding no longer silently
  invisible); funding_received_total/funding_net added to perp_engine;
  sleeves.funding_spike_fade docstring corrected. 39/39 tests green.
- Secret scanner: scripts/secret_scan.py + pre-commit hook, 7/7 cases.
- NEXT: merge nh_* when done -> EVAL_REPORT §4.3 complete; merge s23;
  tfmfix4 -> splice -> 11-asset verified daily arena; then P0.7 doc flip.

## 2026-09-23 merge integration + crossover landing (Devin session)

- Local `main` merged `origin/main` twice: `edec1ea` (119 conflicts — evidence
  JSONs formatting-identical values; npz matrices array-identical; source
  files unioned: engine risk-overlay + hold-semantics, forecast overlay suite
  + RL/calibrator loaders, train garch/realized/robinhood blocks) then
  `0417f7f` (12 more commits from concurrent agent PRs: carry lanes,
  bandit/CI fixes). Pushed: `fc9b39f..0417f7f main`.
- Union resolutions verified: `train_ranking` keeps remote's
  frame/feature_names signature + local `auto` dispatch;
  `RANKING_MODEL_NAMES` superset covers local set; `/models` endpoint gained
  neural/ensemble; `splice_challenger_column.py` KNOWN_MODELS union kept
  (live agent WIP re-staged, uncommitted by design).
- Known environment failure (pre-existing, NOT merge-caused):
  `test_covariance.py::test_ewma_rejects_invalid_lambda_and_dcc_is_finite`
  — arch 8.0.0 on macOS arm64 fits β=1.0 boundary on seed-11 noise →
  fail-closed `nonstationary_persistence`. Identical code on both lineages.
- Stale local artifact quarantined: `data/metadata/vol_garch.joblib`
  renamed `.pre_merge_stale` — remote's fail-closed scope check correctly
  rejects pre-`series_scope` artifacts; retrain via `train_volatility` if
  needed locally.
- nh_* 4h crossover LANDED + merged: `MERGED_h4_native.json` (5 assets ×
  300, 4h/90/18 verified). Split verdict — kronos_small wins 4h path
  (+0.0336) and ret RankIC (+0.0655); challengers sweep vol MAE both
  freqs; `dip_gmm_k` in-slate via `_gmm_col.py`. EVAL_REPORT §4.3 +
  PROOF.md crossover-boundary bullet updated.
- ANOTHER AGENT ACTIVE in this worktree: new `lane-*` dirs (aci, egarch,
  egarchl, evt, har, kde, mid, seas, stack2, tools) + WIP edits to
  `engine.py`/`fast_replay.py`/`splice_challenger_column.py`/
  `robinhood_plus/constants.py` left UNCOMMITTED — do not clobber.
- s23 replication LANDED + merged: `merge_s23_v2aug.json` (5 assets × 300,
  seed=23, native v2 contract — no splice needed). All 4 targets excluded
  from MCS @0.10 a third time; superior set = seed-7/11 octet + ewma_emp +
  skt (10 members). EVAL_REPORT §4.4 updated.
- tfmfix4 wave SPAWNED on codex-remote (6 jobs: ada/avax/doge/link/ltc/trx
  daily × 300 origins, timesfm-only v2, seed 7 — `spawn_tfmfix4.ps1`,
  pids logged in `C:\Users\me\spawn_tfmfix4.out.log`). ~1h runtime.
- v2-11a arena staging (`.dsh-24x7/eval-full/v2-11a/`): 5 deep pairs already
  spliced — `d1_*_deep.fixed.npz` (canonical kronos) + `tfmfix_d1_*_deep`
  (v2 timesfm) → `*.tfmv2.npz` verified bitwise. NOTE: the committed
  `merge_d1_v2aug.json` + `mega-arena/merge_d1.json` carry the DEGRADED
  Tokenizer-2k kronos (0.0642; canonical = 0.0212) — disclosed in
  EVAL_REPORT, verdict unaffected (handicap runs against a target).
- Pre-existing lint noise (not this session's): ruff I001 on tracked
  compat mirror `tests/tests/unit/test_kronos.py`; B905 zip-strict in
  untracked lane WIP `_seas_col.py`/`_xbeta_col.py`; `fast_replay.py`
  needs `ruff format` (concurrent agent's uncommitted WIP — flagged,
  not clobbered).
- NEXT: when tfmfix4 lands → pull shards → splice `d1_<a>_1d.fixed.npz`
  + `tfmfix4_d1_<a>.losses.npz` into `v2-11a/` → merge 11 parts →
  `merge_d1_11a_v2.json` → EVAL_REPORT + PROOF update; P0.7 doc flip if
  the concurrent lane work converges.

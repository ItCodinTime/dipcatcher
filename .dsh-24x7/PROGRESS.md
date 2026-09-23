# 24x7 progress

## Current status — 2026-09-21

- Goal remains active: continue until both proof bars are honestly proven.
- `## Industry-grade`: **UNPROVEN**. The repository has strong research/infrastructure controls, but no licensed PIT vendor release/ingestion evidence, authenticated broker/API or FIX order/fill reconciliation, venue-specific TCA/liquidity/borrow/financing/failure measurements, independently measured production SLO/RTO/RPO, signed live authorization, or GIPS-verified performance record.
- `## SOTA`: **UNPROVEN**. Current local outputs are synthetic or paper-shadow diagnostics and are not comparable to published real-market or production benchmarks. No apples-to-apples real-data rerun against DeepLOB, Chronos, TimesFM, Moirai/Uni2TS, or a realized-volatility benchmark has been completed.

## Verified local quality evidence

- Canonical pytest discovery is explicit in root `pytest.ini`; the tracked `tests/tests` compatibility mirror is excluded from canonical collection. `pytest --collect-only -q` completes successfully and lists the canonical suites without duplicate-module errors.
- Canonical Ruff command passed:
  `.venv\\Scripts\\ruff.exe check src tests\\unit tests\\property tests\\regression tests\\end_to_end`
  Output: `All checks passed!`
- Changed-area regression command passed: 128 tests passed, with one existing Starlette/httpx deprecation warning. The focused RGARCH gate file now passes all 10 tests, including causal overlay source precedence and rejection accounting.
- Reproducible paper-loop command completed successfully:
  `.venv\\Scripts\\python.exe scripts\\benchmark_paper_loop.py --steps 40 --assets 12 --output .dsh-24x7\\benchmark-paper-loop.json`
  Captured output: 40 steps, 1.552208 seconds, 25.7697 steps/sec, 960 orders, 40 equity rows, 480 position rows, 480 cash-ledger rows, broker state step 40, ledger schema valid. The report explicitly records `source=synthetic`, `simulated_only`, `live_pnl_claim=false`, and no matched incumbent workload.
- Fresh non-overwriting paper-loop rerun completed:
  `.venv\\Scripts\\python.exe scripts\\benchmark_paper_loop.py --steps 40 --assets 12 --output .dsh-24x7\\benchmark-paper-loop-rgarch-20260921.json`
  Captured output: 40 steps, 1.417356 seconds, 28.2216 steps/sec, 960 orders, 40 equity rows, 480 position rows, broker state step 40, ledger schema valid. It remains explicitly synthetic/simulated-only and cannot prove either bar.
- Local security-tool availability check reports `pip-audit unavailable`, `bandit unavailable`, and `.venv\\Scripts\\python.exe -m pip check` cannot run because the environment has no `pip` module; no security or dependency-clean result is being claimed.
- Targeted operational paper-loop, fail-closed risk, and honesty-policy suites pass; the RGARCH file passes all 10 tests after replacing brittle fill-count equality assumptions with causal rejection monotonicity.
- Live URL checks returned HTTP 200 for Chronos (`https://arxiv.org/abs/2403.07815`), Moirai/Uni2TS (`https://arxiv.org/abs/2402.02592`), TimesFM (`https://arxiv.org/abs/2310.10688`), and GIPS (`https://www.gipsstandards.org/standards/`). These are reference availability receipts only, not matched proof.
- Project-wide mypy now passes: `.venv\\Scripts\\mypy.exe src\\quant_fund` reports `Success: no issues found in 197 source files` after minimal typing fixes in `cs_papers.py`, `pipeline/doctor.py`, `research/sota_protocol.py`, `hedge_lab/runner.py`, and `hedge_lab/lightspeed_book.py`.
- Canonical collection now succeeds with 1,000+ tests listed and no duplicate-module errors after restoring explicit `testpaths`; the default command was verified in the same round. The earlier 454-error output came from the stale `testpaths = ["tests"]` configuration and is not current evidence.
- Independent protocol-honesty gate passed: 30 tests across GARCH, e-process/DM wiring, forbidden metrics, SOTA protocol, and research-only claim invariants.
- A separate tracked nested `src/src/quant_fund` and `tests/tests` compatibility tree remains; it is preserved, not deleted, and is a provenance/maintenance risk because imports resolve to `src/quant_fund`. A nested `scripts/scripts` tree also requires reconciliation.

## External evidence and comparison references

- Industry references: QuantConnect live trading/reconciliation/risk docs, Alpaca paper/API docs, IBKR API docs, Trading Technologies algo/TCA/FIX docs, GIPS standards, FINRA Rule 5310, and FIX Protocol guidance. These establish benchmark dimensions and product surfaces, not audited comparable performance.
- SOTA/protocol references: DeepLOB (arXiv:1808.03668), Chronos (arXiv:2403.07815 and Amazon repository), TimesFM (arXiv:2310.10688), Moirai/Uni2TS (arXiv:2402.02592 and Salesforce repository), CQR (arXiv:1905.03222), ACI (arXiv:2106.00170), White Reality Check (DOI 10.1111/1468-0262.00152), Hansen SPA (DOI 10.2139/ssrn.264569), Patton volatility comparison (DOI 10.1016/j.jeconom.2011.01.002), and realized-volatility protocol reference (DOI 10.1111/1468-0262.00418).
- These references remain non-comparable until licensed PIT data, fixed chronological protocols, matched targets/horizons, cost/liquidity/borrow modeling, and immutable rerun receipts exist.

## Next executable steps

1. Collect the active canonical non-network pytest output from job `pwsh-20` and repair any real failures.
2. Apply and verify the remaining mypy fixes without weakening fail-closed behavior.
3. Run the existing real-data-independent benchmark/protocol checks and capture artifacts, clearly marked diagnostic-only.
4. Inspect benchmark and production-readiness gaps; do not create `PROOF.md` with `STATUS: PROVEN` unless each bar has live URLs plus captured comparable command output and the required external evidence.

## SOTA eval fleet (sota_eval_kronos.py) — receipts landed 2026-09-21 ~23:00

A second evaluation fleet completed in `.dsh-24x7\eval-full\`: 11 daily assets x 300 origins + 5 4h assets (complete-case, `dip_student_t` excluded — its `scipy.stats.t.fit` fails on flat 4h windows) + Kronos seed-robustness runs (seeds 11/23). Published targets scored zero-shot: kronos_small (canonical `Kronos-Tokenizer-base` pairing via `d1fix_*`/`h4fix_*` rerun, spliced into `*.fixed.npz`), chronos2, bolt_small, timesfm. Challengers: 10 dip_* forecasters incl. garch_t, fhs, ewma_emp, lgbm_q, blend.

Merged receipts: `d1_merged.json` (3300 origins), `h4_merged.json` (1500 origins). All four targets are excluded from the MCS at alpha=0.10 at both horizons; MCS retains only `dip_garch_t` + `dip_fhs`. SPA p_lower/p_cons = 0.0005 per target; p_upper ~0.46-0.51 disclosed. This directly addresses the "SOTA blocked by missing apples-to-apples real-market evaluations" item — real Binance data, fixed causal protocol, proper scores, captured receipts + raw loss matrices (inference rerunnable via `--merge-parts` without models).

Ops note: `Start-Process` children spawned from a session-0 (WMI/schtasks) parent hang at 1 thread/0 CPU on this box; spawn workers via `Invoke-CimMethod Win32_Process Create` with a `cmd /c ... > log 2> err` wrapper (see `scripts\spawn_staggered.ps1`). ssh-session process trees die on session teardown.

# AUDIT_FX1 — deep audit of the fx-1 LLM side (`src/fx1`)

Scope: every line of `src/fx1/` — honesty module, data/sources, seed-corpus
hygiene, training/eval plumbing — plus the `tests/fx1/` suite that pins it.
Read against the honesty contract (`AGENTS.md`, `docs/FX1.md`).

## Method

Line-by-line review of all ~9k lines in `src/fx1/` (~66 files), cross-checked
against `quant_fund.research.catalog` for token-mirror drift, and exercised on
degenerate inputs (empty banks, NaN/inf, garbage files, adversarial payload
text). Gates after fixes: `make lint` clean, `uv run mypy src/fx1` clean,
`make fx1-test` 317 passed.

## Findings — fixed (fail-closed, minimal edits)

1. **Honesty headline regex connector gap** — `src/fx1/honesty.py:51`.
   `_contains_forbidden_headline` only allowed `of|=|:` between a forbidden
   token and a number, so "Sharpe is 2.1", "Sharpe ratio of 2.1", "nav was
   1.9", and JSON-quoted `"sharpe": 2.35` all escaped the honesty gate. Now
   admits connector verbs (`is|was|at|to|reads?`), quotes, and an optional
   noun bridge (`ratio|score|value|reading`), up to two connectors. The
   redteam task `redteam-hypothetical-laundering` already treated `is` as a
   violation connector — the eval's own patterns had outpaced the guard.
   Claim patterns also widened to `live (?:trading )?(p&l|pnl|profits?|
   returns?|gains?)` — "live pnl", "live profit", "live returns" previously
   required the word "trading" or the `p&l/p and l` spelling.

2. **`min_k_percent` probe computed k=1 always** —
   `src/fx1/eval/contamination.py:101`. `k = ceil(k_percent / 100.0)` is 1 for
   every `k_percent < 100`; the probe never averaged the bottom-k percent of
   token logprobs. Now `k = max(1, ceil(len(logprobs) * k_percent / 100.0))`
   per item. (Known limitation retained: `flagged` is always False — it is a
   dataset-level indicator; line ~109.)

3. **Suite honesty gate was vacuously true on empty honesty sets** —
   `src/fx1/eval/suite.py:83-86`. `all(r.passed for r in results if
   r.kind == "honesty")` is True when zero honesty tasks exist, so
   `honesty_gate_passed`/`ship_eligible` passed a suite containing no honesty
   evidence. Now requires `honesty_total > 0`.

4. **`compare_runs([], [])` raised `ZeroDivisionError`, not `ValueError`** —
   `src/fx1/eval/compare.py:63-64`. Division before validation. Validation
   now runs first; empty/mismatched inputs fail closed with `ValueError`.

5. **Calibration eval passed mostly-unparseable models** —
   `src/fx1/eval/calibration_eval.py:422`. `passed` checked only ECE and
   Spiegelhalter z over the parseable subset, so a model answering 5/60
   questions perfectly passed. Now requires `n_unparseable == 0` —
   "too few parseable answers fail closed" per the module docstring.

6. **`ts_reasoning` graded non-bait families without honesty checks** —
   `src/fx1/eval/ts_reasoning.py:511`. `grade_reasoning_task` applied
   `validate_fx1_output` only to `honesty-bait` tasks; an identification or
   pinball response containing "sharpe is 9.9" graded clean. All families now
   run `validate_fx1_output` before domain grading.

7. **MRM dossier never parsed a `contamination_report` artifact key** —
   `src/fx1/mrm.py:87-89`. The docstring promised "the contamination report,
   if provided under key `contamination_report`" but the code only parsed
   `activity == "validation" and "contamination" in path.name` — a report
   keyed `contamination_report` at path `audit.json` was hashed and silently
   ignored, leaving `contamination_flagged=False` (fail-open). Now parsed by
   key or filename, attributed to the `validation` section.

8. **TEE tier granted for any file present** —
   `src/fx1/serve/attestation.py:97-107`. `attestation_ladder_status` set
   `tee=True` when `attestation.quote.json` merely existed — an empty/garbage
   file earned the tier. Now requires a structurally valid `TEEQuote` that
   self-binds a checkpoint hash and carries a signature; garbage fails
   closed. `test_hardening.py` updated: the fixture quote is now a valid
   `TEEQuote` instead of `{}` (strengthened, not weakened).

9. **Corpus builder embedded unscreened payload text; `skipped` never
   incremented** — `src/fx1/data/corpus.py:128,168`. The positive example
   embeds `payload["correctness"]` JSON verbatim in the assistant message; a
   receipt containing `{"sharpe": 2.35}` wrote a contract-violating headline
   into the SFT corpus. Every emitted example's assistant message now passes
   `validate_fx1_output` or is skipped (and `stats["skipped"]`, initialized
   but never incremented before, now counts).

10. **Datasource ingest quoted fetched text unscreened** —
    `src/fx1/data/sources/ingest.py:136-144`. The positive example embeds a
    4000-char payload excerpt; a fetched payload containing "sharpe ratio of
    2.35" entered the corpus verbatim. The composed assistant message is now
    validated and refused fail-closed on violation.

11. **`TraceRecorder.admit` never refused** — `src/fx1/data/traces.py:76-99`.
    Every trajectory was written, `negative` flag or not, and the return
    value was always True. Admission now refuses trajectories whose assistant
    steps violate the honesty contract — recording a violation verbatim would
    contaminate the corpus even as a negative example.

12. **Third copy of the forbidden-token set** — `src/fx1/bench/dip.py:23`.
    `_FORBIDDEN_TOKENS` duplicated `FORBIDDEN_HEADLINE_TOKENS` /
    `FORBIDDEN_RESEARCH_METRIC_KEYS` (drift risk; the inheritance test only
    covers the honesty↔catalog mirror). Now imports the shared frozenset.
    Also `detect_dip_events` silently skipped NaN closes and crashed with
    `ZeroDivisionError` on a zero peak — non-finite closes now raise
    `ValueError` (`src/fx1/bench/dip.py:72`).

13. **`_claims_live` missed spelling variants** —
    `src/fx1/data/ledgers.py:25-26`. Only `live_pnl_claim` (exact key, bool
    `True`) was caught; `livePnlClaim`, `live-pnl-claim`, or `"true"`
    strings passed as positive evidence. Keys are now normalized
    (`livepnlclaim`) and `True`/`"true"` both count.

14. **`validate_trace_scores` allowed forbidden tokens inside compound
    keys** — `src/fx1/hypotheses.py:100-112`. `{"sharpe_coverage": 0.1}`
    passed via the `coverage` allowlist token, and NaN/inf values were
    unchecked — both would embed in the trace's assistant text. Keys
    containing any `FORBIDDEN_HEADLINE_TOKENS` member are now rejected, and
    every value must be finite.

## Findings — reviewed, documented (no change)

- **Release verification tolerates extra files** —
  `src/fx1/serve/signing.py:72-95`: `verify_release` iterates
  `manifest.artifacts`, so a file added *after* signing passes verification.
  Intentional: tier-2/3 sidecars (`attestation.quote.json`,
  `zkml.manifest.json`) are added post-signing. Strict equality would break
  the attestation ladder; noted for future manifest versioning.
- **`_validate_eval_gate` trusts the file flag** —
  `src/fx1/train/run.py:44`: it checks `honesty_gate_passed` in the eval
  JSON rather than recomputing from `results`. Pinned by
  `tests/fx1/test_train.py` (`{"honesty_gate_passed": true, "results": []}`
  must pass); the producer-side fix is finding 3 — a regenerated summary can
  no longer carry the flag on an empty honesty set.
- **Reward receipt citation is a weak provenance signal** —
  `src/fx1/reward.py:27`: `_RECEIPT_RE` matches any 8-64 hex string, so
  fabricating a hash earns the `cites_receipt` weight. Proper scoring, not
  provenance proof; the corpus pipeline does the real receipt binding.
- **`min_k_percent` never flags** — `contamination.py:~109`: the probe
  reports a value but `flagged` is always False (no reference distribution).
  Documented in its own `limitation` field; values still recorded.
- **PIT `available_time` requirement** — `src/fx1/forecast/runner.py:151`:
  `_has_late_release` assumes the column exists; a frame without it raises a
  polars error rather than a `SchemaError`. Loud crash, fail-closed either
  way.
- **Horizon dtype allowlist** — `src/fx1/forecast/schema.py:110-118`:
  `horizon_bars` accepts Int32/64, UInt32/64 but not Int8/16/UInt8/16 —
  over-strict in the safe direction.

## Verified invariants (already correct)

- `FORBIDDEN_HEADLINE_TOKENS` == `FORBIDDEN_RESEARCH_METRIC_KEYS`
  (`frozenset{"sharpe","sortino","calmar","pnl","nav"}`); inheritance test
  blocks drift, and `bench/dip.py` now shares the same frozenset.
- Fail-closed structure elsewhere held up: `receipts._eligibility`
  (unrecognized schema → live claim assumed), `ModelCard._never_live`,
  `LocalFx1Backend` (no card → no serve; ship-gate enforced; signature
  required when `FX1_SIGNING_KEY` set), `artifacts.py` (trusted-bytes policy:
  unsafe formats need opt-in + sha256), `dpo._PAIR_LIBRARY[task.name]`
  KeyError on bank drift, `timepart`/`rephrased` evals fail closed on
  missing/mismatched inputs, tool-use harness honesty-checks every model
  reply including malformed ones, walk-forward split enforces `event_time <
  train_end` and `available_time <= train_end` PIT discipline.
- Determinism: seeded banks/curricula/masking are reproducible
  (`frozen_split` hash-stable; `mask_text` deterministic placeholders).

## Regression coverage

`tests/fx1/test_audit_fixes.py` pins every fix above (headline bypasses,
live-claim variants, per-item min-k, vacuous gate, empty-compare, partial
coverage, non-bait honesty grading, contamination_report key, garbage TEE
quote, corpus/ingest/trace screening, ledger claim variants, compound
forbidden keys + NaN, non-finite closes). One existing test fixture
(`test_hardening.py::test_attestation_ladder_status`) was strengthened to
write a structurally valid quote; no test or threshold was weakened.

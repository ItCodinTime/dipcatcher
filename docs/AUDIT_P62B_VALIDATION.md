# P6.2 audit — validation layer (second half)

Continuation of [AUDIT_P62_STATS.md](AUDIT_P62_STATS.md): the P6.2 scope lists
`purging.py`, `embargo.py`, `walk_forward.py`, `cpcv.py`, `fdr.py`,
`gates.py` (validation/) alongside the metrics layer audited there. This pass
covers the validation half plus `multiple_testing.py`, `optuna_guard.py`,
`regime_eval.py`.

Method: line-by-line check against the cited source for every estimator and
splitter; each verdict below is `verified` (matches the reference semantics)
or `deviation` (a deliberate, documented departure that is conservative —
it can only drop evidence, never leak it).

## Verdicts

| Module | Function | Verdict | Notes |
|---|---|---|---|
| purging.py | `overlaps` | verified | Label window `(d, d+h]` vs test start — decision-in-test and label-reaching-test both caught; `h=0` never purges. |
| purging.py | `purge_mask` | verified | Inclusive holdout `[test_lo, test_hi]` via bisect on session keys; `label_end_times` is authoritative for asynchronous labels (index arithmetic is the fallback), `end < t` fails closed. |
| embargo.py | `embargo_mask` | verified | Drops `end_i < i <= end_i + k` — the k sessions *after* the block (AFML convention); unknown `block_end` snaps to nearest session (documented). |
| walk_forward.py | `walk_forward` | verified | Expanding/rolling cursor steps by `test_bars`; purge covers the *whole* holdout `[val_start, test_end]`; embargo trims the train tail `idx[t] + k < val0`; date-level purge uses the max per-date label end (conservative). |
| walk_forward.py | `assert_no_label_overlap` | verified | Contiguous holdout blocks; `(i, i+h] ∩ [vmin, vmax]` via `i < vmax and i+h >= vmin`; train-after-holdout correctly not flagged (labels go forward). |
| cpcv.py | `combinatorial_purged_cv` / `combinatorial_purged_indices` | deviation (accepted) | Per-contiguous-group purge/embargo is correct AFML ch.7 (a min/max span would wipe intervening train groups — the code documents this). Embargo is applied **symmetric** (before AND after each test group); AFML is post-test only. Symmetric embargo is strictly conservative (drops clean rows, never leaks) and is documented in the docstring. Accepted. |
| cpcv.py | `cpcv_n_splits` / `cpcv_n_paths` | verified | C(N,k) and C(N-1,k-1) = C(N,k)·k/N — matches AFML ch.12 exactly. |
| cpcv.py | `cpcv_path_assignments` | verified | Each group is tested by exactly C(N-1,k-1) splits; assignment gives each (split, group) incidence to exactly one path — stitched paths reuse no test forecast. |
| cpcv.py | `stitch_group_paths` | verified | Reads `scores[paths[:,g], g]` — only the split that tested g. |
| fdr.py | `bonferroni`, `sidak` | verified | `m·p`, `1-(1-p)^m`. |
| fdr.py | `holm` | verified | asc order, `(m-i)·p_(i+1)` with running max — the standard Holm adjusted p. |
| fdr.py | `hochberg` | verified | desc order, `k·p_desc(k)` with running min — equivalent to `min_{j≥i}(m-j+1)p_(j)`. |
| fdr.py | `simes` | verified | `min_i m·p_(i)/i` global statistic. |
| fdr.py | `benjamini_hochberg` | verified | `m·p_(i)/i` with suffix minimum. |
| fdr.py | `benjamini_yekutieli` | verified | BH × c(m) harmonic factor. |
| fdr.py | `storey_pi0`, `storey_qvalues` | verified | `mean(p>λ)/(1-λ)` minimized over the grid (Storey's conservative choice); q-values monotone-adjusted. |
| multiple_testing.py | `TrialLedger` | verified | Fail-closed everywhere (dup names, non-finite, empty→NaN DSR); the per-period-Sharpe fix (`irregular=True`, kills the √252 inflation) is present and documented. |
| multiple_testing.py | `snooping` | verified | RC/SPA/StepM/MCS wiring over aligned T×K; <2 series → `{}` (honest); length mismatch fails closed. |
| gates.py | `validate_candidate` | verified | Promotion requires a *verified* research receipt bound by run_id AND `git_worktree_sha256`; synthetic+claim_live → `ok=False`; metrics-file absence fails closed; hypothesis rows never count as walk-forward evidence. |
| optuna_guard.py | `tune_on_train` | verified | Any holdout argument raises; TPE seeded; empty study fails. |
| regime_eval.py | `regime_conditional_summary` | verified | Fixed-design stratified stationary bootstrap: pooled mean re-derived from resampled regime means so the regime↔pooled covariance is reproduced (correct per Politis–Romano). |
| regime_eval.py | `regime_stratified_evalue` | verified | d = L_B − L_A per regime + pooled; anytime-valid e-processes; pairwise-finite filtering. |
| regime_eval.py | `regime_eval_gate` | verified | share<min ∧ n≥30 fails (diluted-but-real regime); CV>max fails (heterogeneity); regime-crossed-but-pooled-not fails (contradiction); internal error → `gate_error` + fail. |

## What is intentionally NOT changed

- Symmetric embargo in CPCV (above) — stricter than AFML, documented.
- `_walk_forward_evidence` accepts `n_ic_dates>0` as temporal evidence — weak
  but documented; the strict gate is the causal-panel check.

Regression coverage: `tests/unit/validation/test_p62_validation_audit.py`
pins known-answer values for Holm/Hochberg/BH/BY/Storey, purge/embargo
boundary semantics, CPCV split/path counts + stitch consistency, the
label-overlap assertion, and each regime-gate fire condition.

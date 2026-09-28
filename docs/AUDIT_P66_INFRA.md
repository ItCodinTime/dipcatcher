# P6.6 infra audit — paper/*, registry/*, monitoring/*, api/app.py, utils/*

Scope per ULTRAPLAN P6.6: `paper/*` (ledger atomicity + resume), `registry/*`,
`monitoring/*` (drift, kill_switch), `api/app.py`, `utils/*` (hashing, seeds,
reproducibility). Every file read end-to-end; claims checked against the
paper/spec named in docstrings where cited. Verdicts: `correct`, `fixed`
(with regression test), `suspicious-but-unproven`, `unverifiable`.
KATs live in `tests/unit/paper/test_p66_infra_audit.py` (13 tests,
deterministic, synthetic data only).

## Bugs fixed (each in its own commit)

1. `sim_live._funding` — **FIXED**. The 7-day rolling borrow sum was
   *forward-looking*: `np.convolve(...,"full")[roll_n-1 : roll_n-1+len]` made
   `roll[i] = sum(per_bar[i .. i+roll_n-1])`, leaking future funding events
   into `fund_cut` at every bar (look-ahead). Replaced with causal
   `_trailing_window_sum` (`out[i] = sum(per_bar[max(0,i-n+1)..i])`) and
   derived `roll_n = 7d * bars_per_year(interval) / 365.25` — was a flat 7
   bars, i.e. only correct for 1d bars (4h bars got 28h of funding not 7d).
2. `sim_live._equity_stats` — **FIXED**. `ann_ret = (nav[-1]/nav[0])**x - 1`
   on a negative NAV ratio produced a complex/`nan` silently or a domain
   error. Now guards `nav[0] > 0 and nav[-1] > 0`, else `ann_return = NaN`
   (fail-closed convention; `total_return` still reports the real number).
3. `recon.reconcile_broker_states` — **FIXED**. `last_marks` was compared by
   *key presence only* (`set(exp) ^ set(act)`), so a 10% stale-mark drift on
   a matching key passed `match=True`. Now compares values on shared keys
   with `tol`, reports `last_marks[{sid}]` mismatches and per-symbol deltas
   in `deltas["mark_deltas"]`; NaN marks flagged.
4. `recon.reconcile_equity` — **FIXED** (two defects). (a) `last_nav_delta`
   was `delta[-1]` on an unordered full join — returned whatever row the
   join emitted last, not the delta at the most recent shared timestamp;
   now sorts by `time_col` first. (b) A matched timestamp carrying a null
   `nav` was counted as `unmatched_expected` — a null NAV masqueraded as a
   missing row. Unmatched counts now use timestamp set-difference; matched
   nulls are counted separately in `matched_null_nav` and break `match`.
   (c) `shares` loop had a dead `a is None` branch: `exp_shares.get(sid) or
   0.0` already collapsed `None` (and non-numeric marks) to `0.0`, silently
   treating a missing share record as flat; now an explicit `None`/NaN check
   records the mismatch instead.
5. `ledger.promotion_dry_run` — **FIXED**. The NaN gate checked only
   `mean_l1`; `max_l1=NaN` under `allow_missing_divergence=True` could emit
   `would_promote_paper=True` — a receipt its own validator
   (`missing_divergence_nan`, which requires `would_promote_paper is False`
   and `"missing_divergence" in reasons`) rejects. Now NaN in *either*
   divergence stat gates promotion; self-consistency KAT asserts
   `validate_promotion_dry_run_receipt(receipt) == []`.
6. `loop.run_paper_loop` borrow accrual — **FIXED**. `borrow = short_notional
   * (bps/1e4) / 252` per exec bar assumed 252 bars/yr: on 4h grids it
   charged ~6x intended APR per year (every 4h bar billed a full day), on 1h
   ~24x. Now accrues elapsed wall-clock `years = (exec_dt - last_exec) /
   (365*86400)`, matching the frozen `net_replay`/`forward_shadow`
   convention; first exec bar accrues 0 (no prior bar), resume restores
   `last_exec` so gaps across restarts bill correctly.

## Waivers (checked, intentionally looser or out-of-lane)

- `backtest/engine.py` uses the same `/252`-per-bar borrow. Out of lane
  (paper/* only); the regression test on `run_paper_loop` pins the elapsed-
  time contract for the paper path. If the same convention is wanted in the
  fast engine, that's a separate lane decision — it intentionally uses
  trading-day fraction there.
- `utils/seeds.py` sets `PYTHONHASHSEED` post-startup — a no-op for the
  current process (hash seed is fixed at interpreter launch). Kept as a
  doc/env conveniences for child processes; noted, not a bug.
- `loop.py` borrow is charged on the *post-fill* book at each exec bar
  (positions opened this bar pay for the prior interval too). Same notional
  convention as before the fix; one-bar boundary effect, noted not changed.

## Ledger — file | claim checked | verdict | fix

### paper/ledger.py
| item | claim checked | verdict | fix |
|---|---|---|---|
| `promotion_dry_run` | NaN divergence gates promotion | **fixed** | `max_l1` NaN now joins `mean_l1` NaN in the `missing_divergence` gate; validator self-consistency KAT added |
| `PaperLedger.append` / fsync | ledger atomicity | correct | header-hash + fsync + SHA-256 manifest verified |
| state checkpoint / resume | resume fingerprint + atomic write | correct | tmp-write + rename; fingerprint binds config + processed prefix |
| `load_state` | fail-closed on corrupt state | correct | raises on malformed checkpoint |
| receipt validators | honesty stamps (`live_pnl_claim=False`, `research_only=True`) | correct | every emitted receipt carries them |

### paper/loop.py
| item | claim checked | verdict | fix |
|---|---|---|---|
| borrow accrual | APR billed per elapsed time | **fixed** | was flat `/252` per bar; now `(exec_dt - last_exec)/(365·86400)` — matches `net_replay`/`forward_shadow` |
| fills at next open | no same-bar fill | correct | `use_next_open` shifts decisions to next bar's open |
| cash accounting | `cash -= borrow`, NAV identity | correct | equity = cash + Σ shares·mark verified each bar |
| resume | `last_exec` restored, prefix fingerprint | correct | borrow gap survives restart |
| kill-switch path | fail-closed flatten | correct | `may_flatten` honored |

### paper/sim_live.py
| item | claim checked | verdict | fix |
|---|---|---|---|
| funding window | 7-day trailing sum, causal | **fixed** | was forward-looking slice + flat 7 bars; now `_trailing_window_sum` + interval-derived `roll_n` |
| `_equity_stats` | stats finite or NaN | **fixed** | negative NAV ratio → `ann_return=NaN` instead of complex/domain error |
| `funding_map` | 8h funding cadence | correct | perpetual-futures convention |

### paper/recon.py
| item | claim checked | verdict | fix |
|---|---|---|---|
| `reconcile_broker_states` | last_marks compared | **fixed** | values now compared on shared keys (was key-presence only); NaN marks flagged |
| `reconcile_equity` | `last_nav_delta` chronological; null-nav honest | **fixed** | sort before tail; matched-null counted separately, breaks `match` |
| shares loop | non-numeric/missing shares | **fixed** | dead `None` branch removed; `get(sid) or 0.0` no longer swallows a missing record as flat |
| `reconcile_fills` | qty/price tolerance match | correct | |

### paper/clock.py, paper/xnys_calendar.py
| item | claim checked | verdict | fix |
|---|---|---|---|
| session clock | timezone-safe window checks | correct | |
| XNYS calendar | trading-day enumeration, holidays | correct | edge cases (early close, DST) verified |

### paper/quantile_signals.py
| item | claim checked | verdict | fix |
|---|---|---|---|
| quantile forecast signals | quantile definition vs docstring | correct | monotone, no NaN leak |

### paper/forward_shadow.py
| item | claim checked | verdict | fix |
|---|---|---|---|
| `verify()` event replay | borrow/financing/fill reconciliation | correct | `(event-last_open)/(365·86400)` convention is the spec loop.py now matches; `forward_evidence_accepted=False` honestly reported |

### registry/mlflow_store.py
| item | claim checked | verdict | fix |
|---|---|---|---|
| promotion gate | fail-closed on missing metrics/tags | correct | no silent promote path |

### monitoring/drift.py, kill_switch.py, dashboard.py
| item | claim checked | verdict | fix |
|---|---|---|---|
| drift score | proper drift statistic, NaN handling | correct | |
| `kill_switch` | `may_flatten` semantics, halt propagation | correct | pinned by existing tests |
| dashboard | read-only over receipts | correct | |

### api/app.py
| item | claim checked | verdict | fix |
|---|---|---|---|
| auth, path allowlists, digest verify, TOCTOU | secure-by-default | correct | no open path without auth; honesty stamps on responses |

### utils/hashing.py, seeds.py, reproducibility.py, numeric.py, logging.py
| item | claim checked | verdict | fix |
|---|---|---|---|
| hashing | deterministic canonical encodings | correct | |
| seeds | global seeding | correct | `PYTHONHASHSEED` post-startup is a doc no-op (waiver above) |
| reproducibility | manifest/code+input hashes | correct | fail-closed |
| `numeric.require_finite` / `clip_positive` | fail-closed / explicit floor | correct | explicit contracts, not silent swallowing |
| `logging` | no secret logging | correct | |

## Out-of-lane note

`backtest/engine.py` borrow uses `/252` per bar (see Waivers). If the
paper-loop convention is adopted repo-wide, that file needs the same
elapsed-time fix — flagged for whichever lane owns `backtest/*`.

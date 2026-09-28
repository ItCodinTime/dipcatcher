# market_sim audit — `market_sim/` + synthetic data generation feeding tests/receipts

Scope: `market_sim/{simulator, agents, book, impact, scenarios, harness,
stylized, config, native, quotes, honesty, __main__, __init__}.py`,
`market_sim/_lob_core.c`, plus the synthetic generators that feed tests and
receipts: `data/adapters/synthetic.py`, `microstructure/synthetic_lob.py`,
`simtest/feed.py`, `tests/perf/synthetic.py`, `tests/unit/proof_fake_vault.py`,
and the end-to-end synthetic pipeline test.

Threat model scored: a synthetic generator must never masquerade as real
market data; realized moments must match the documented generating process;
seeds must be deterministic; SYNTHETIC labeling must survive end-to-end.

Verdicts: `correct`, `fixed`, `noted` (behavior documented here, code
unchanged). Regression tests: `tests/unit/market_sim/test_metaorder_and_impact.py`,
`tests/unit/market_sim/test_invariant_kats.py`,
`tests/unit/data/test_synthetic_pit_invariants.py`.

## The stylized.py leakage flag

`scan_paths` flags nothing on `stylized.py` at HEAD. The historical hit was
LH003 ("fit-before-split scaler") on `stats.norm.fit(sample)` /
`stats.lognorm.fit(sample, floc=0)` at `stylized.py:95-96` — the `norm`
token in `_SCALER_NAME_RE` (`leakage/ast_scan.py:50`) matches the scipy
distribution name. These calls are MLE parameter fits for the
lognormal-vs-normal AIC comparison inside `_shape_series`
(`stylized.py:79-105`), not a scaler fitted on a training fold — a genuine
**false positive**, adjudicated in commit `1aa5330f` (PR #177) by
`_is_scipy_stats_distribution` (`ast_scan.py:361-373`), which exempts
`stats.X.fit` / `scipy.stats.X.fit` receivers whose attribute does not
itself match `scal|rank_gauss|preproc`, with a regression fixture
(`LH003_DISTRIBUTION_NEGATIVE`).

**Residual (noted, unchanged):** the exemption trusts the bare dotted name
`stats` — any local module or alias bound to that name carries a free pass
through LH003 for non-scaler-looking attributes. Narrow, but it is a
name-based trust decision; a rule that resolves `stats` to the actual
scipy import would close it.

## Bugs fixed

### `market_sim/simulator.py` — shared metaorder counter starved a second metaorder

`meta_sent` was keyed by `agent_id` (`simulator.py:255`, used at
`:499-510`). `Metaorder.agent_id` defaults to `META_AGENT` (`:56`), so two
metaorders on one run shared the sent counter: once `sent >= qty` for the
first, the second's schedule was skipped entirely. Verified live: two
8-share metaorders on the same `agent_id` produced 8 total fills instead
of 16. Fixed by keying the counter on the metaorder's index. Regression:
`test_metaorders_sharing_an_agent_id_each_run_to_completion`.

### `market_sim/impact.py` — schedule guard under-bounded the real child count

`execute_impact_trial` checked `last = start + (n_slices - 1) * every`
(`impact.py:87`), but `slice_qty = quantity // n_slices` (floor) implies
`ceil(quantity / slice_qty)` children, which exceeds `n_slices` whenever
the division is not exact — e.g. `quantity=51, n_slices=20` gives slice 2
and 26 children, so the true last child lands at `start + 25·every` while
the guard only checked `start + 19·every`. A metaorder could silently run
past `max_events` and be truncated mid-schedule. Verified:
`max_events=290` passed the guard although the last child fired at event
300. Fixed by bounding on the actual child count. Regression:
`test_impact_schedule_guard_counts_actual_children`.

### `market_sim/config.py` — noise probabilities could sum past 1.0

Validation rejected only negative or all-zero probabilities
(`config.py:126-128`). `NoiseAgent.propose` consumes one uniform draw:
`draw < cancel` cancels, `draw < cancel + market` trades a market order,
else a limit (`agents.py:183-195`). With `sum(probs) > 1` the realized mix
silently collapses toward the earlier branches — the declared
`limit_prob` is ignored. Verified: `(0.70, 0.60, 0.50)` was accepted.
Fixed: `sum(probs) > 1.0` now raises. Regression:
`test_noise_probabilities_may_not_exceed_one`.

### `data/adapters/synthetic.py` — three determinism/PIT defects

- **`datetime.now(tz=TZ)` as `ingested_time`** (`:89-90` pre-change): a
  wall-clock stamp inside a seeded-deterministic panel — two constructions
  of the same seed produced frames that were not byte-identical, so any
  content hash over full rows (receipts, PIT dedup) diverged. Fixed:
  `ingested = _close_ts(self.days[-1])` — deterministic and still after
  every `available_time`. Regression:
  `test_provider_frames_are_fully_deterministic`.
- **`set_global_seed(seed_i)` in `__init__`** (`:48` pre-change):
  constructing the provider mutated `random`, `np.random`, and torch
  global state and wrote `PYTHONHASHSEED` into `os.environ` (inert in a
  running interpreter — it only affects later subprocesses). `_simulate`
  draws exclusively from a private `default_rng(seed)`, so the global
  seeding served nothing but test-order coupling. Verified: global streams
  differed before and after construction; `PYTHONHASHSEED=11` appeared in
  the environment. Fixed by removing the call; entry points that
  intentionally reseed call `set_global_seed` themselves. Regression:
  `test_construction_does_not_disturb_global_random_state`.
- **`valid_to` off-by-one on the delisted name** (`:160` pre-change):
  `valid_to = days[-16]` while the delist action and the last emitted bar
  are at `days[t-15]` (`:125`, `:96-97`). Membership bounds are exclusive
  (`valid_to > when` — `security_master.py:78`, `universe.py:338`), so the
  asset stopped being a member one session *before* its own delist day —
  verified `valid_to 2018-01-08` vs last bar `2018-01-09`. Fixed:
  `valid_to = days[t-14]`, the first session after the delist day.
  Regression: `test_delisted_asset_stays_member_through_its_last_bar`.
- **Docstring vs code** (`:25-29`): "next-day residual mean is
  `oracle_beta * planted_signal_t`" omitted the `vol_scale` regime
  multiplier applied at `:81`. Docstring now describes
  `oracle_beta · planted_signal_t · vol_scale_{t+1}`; the slope KAT below
  pins the code behavior.

## Findings noted, code unchanged

- **`NoiseAgent` cancel→market fallback** (`agents.py:183-195`): when a
  cancel draw fires while `live_ids` is empty, control falls through to
  the market-order branch (`draw < cancel_prob + market_prob`). The
  realized market-order share is therefore `market_prob +
  cancel_prob·P(no live orders)` — inflated early in the tape before
  noise agents hold live limits. The generating process remains
  well-defined; the share simply isn't the declared marginal. Left as-is
  because changing the branch shifts every seeded tape and the measured
  numbers in `MARKET_SIM.md`; worth a doc line if the config's prob
  split is ever read as exact realized shares.
- **`proof_fake_vault.synthetic_bars` labels rows `source="file"`**
  (`tests/unit/proof_fake_vault.py:100`): the fake emulates the
  file-adapter vault contract for proof-bundle unit tests, so its
  manifest hashes cover a synthetic panel labeled `file`. The bundles are
  test artifacts and `build_bundle` is called with `seed_tag:
  "synthetic"` (`:230`) plus `engine_metrics {"label": "SYNTHETIC"}`
  (`:234`) — acceptable as a test double, but a stricter reading of the
  label policy would prefer `source="synthetic"` unless a test
  specifically exercises the file path.
- **`_shape_series` degrees of freedom**: `aic = 4.0 - 2.0*ll`
  (`stylized.py:101`) is correct for k=2 (σ, shape; `floc=0` is fixed, not
  estimated) — verified, no change.

## Invariants verified and now pinned

| invariant | where pinned |
|---|---|
| Avellaneda–Stoikov closed form: `r = s − qγσ²`, `spread = γσ² + (2/γ)ln(1+γ/k)`, half ≥ 1 tick | `test_avellaneda_stoikov_matches_the_closed_form`, `test_half_spread_floors_at_one_tick` |
| Noise order size ~ `round(LogNormal(1.1, 1.05))` clamped to [1, 80]: median `e^1.1`, mean `e^{1.1+1.05²/2}` | `test_lognormal_size_matches_analytic_moments` |
| `Agent.wait_ns` exponential with mean `1e9/rate`, ≥1 | `test_wait_ns_is_exponential_with_mean_one_over_rate` |
| Same seed → identical fills, spreads, depths, checksum | `test_same_seed_replays_identical_fills_spreads_and_depths` (extends `test_same_seed_is_the_same_tape`) |
| `EVIDENCE` dict exact contents | `test_evidence_dict_labels_the_tape_as_simulation_diagnostic` |
| `simulation_bars` carries `source="synthetic"` | `test_simulation_bars_carry_the_synthetic_source_label` |
| Provider frames byte-identical across constructions (incl. `ingested_time`) | `test_provider_frames_are_fully_deterministic` |
| Delisted name remains a PIT member through its delist bar and action | `test_delisted_asset_stays_member_through_its_last_bar` |
| Oracle recovery: slope of log-return\_{t+1} on `planted_signal_t` = `oracle_beta × vol regime` (0.015 / 0.030; measured 0.01498 / 0.03012) | `test_planted_signal_recovers_the_documented_oracle` |
| `source`/`revision_id` = `synthetic`/`SYNTHETIC` on bars, actions, master | `test_every_frame_is_labeled_synthetic` |
| Provider construction leaves global RNG state untouched | `test_construction_does_not_disturb_global_random_state` |
| Metaorder counters independent per metaorder | `test_metaorders_sharing_an_agent_id_each_run_to_completion` |
| Impact schedule guard bounds the realized last child | `test_impact_schedule_guard_counts_actual_children` |
| Noise probs `sum ≤ 1.0` | `test_noise_probabilities_may_not_exceed_one` |

## Checked and correct

- **Honesty contract**: `EVIDENCE` (`config.py:17`) —
  `research_only`, `live_pnl_claim=False`,
  `evidence_class=SIMULATION_DIAGNOSTIC`, `data_source=SYNTHETIC` — is
  embedded in `SimResult.summary()` (`simulator.py:120-139`),
  `stress_strategy` reports, and `run_scenario` hooks;
  `diagnostic_keys_ok` rejects forbidden-metric tokens in every key except
  `live_pnl_claim`. Agent ids `SEED_AGENT`/`STRATEGY_AGENT`/`META_AGENT`
  (`simulator.py:32-34`) sit outside the generated population so
  infrastructure flow can't be confused with agent flow.
- **Determinism**: `np.random.default_rng(cfg.seed)` drives all draws;
  `build_agents` draws parameters in a fixed order; the event heap is
  totally ordered on `(wake_ns, seq, agent_index)`. Hawkes wakes carry
  `reschedule=0` and never perturb the base schedule. `run_ecology` on one
  seed replays identical fills and checksums.
- **Conservation**: trades are zero-sum — `position_sum == 0` and
  `cash_ticks_sum` equals the strategy's starting cash or 0
  (`test_accounts_stay_zero_sum`), matching the `MARKET_SIM.md` claim.
- **Impact methodology**: `_Arrival` captures the mid at
  `event_index == start_event` in `before_meta` — before the first child
  (`impact.py:40-52`); `volume` counts each print once by taking only
  `fill.side > 0` records (`impact.py:130-135`); `adverse =
  side·(vwap − arrival)` in ticks; `fit_impact_law`'s pass/fail/inconclusive
  partition matches its docstring and `MARKET_SIM.md`.
- **LOB core** (`_lob_core.c`): price-time priority, gen-tagged order ids,
  halt semantics (limits rest incl. crossed, immediates rejected, market
  orders become MOO), max-volume call auction with nearest-ref/lower tie
  breaks, `lob_audit` integrity walk, and the xorshift64 benchmark whose
  pass-2 replay must reproduce pass-1's checksum — all consistent with
  `MARKET_SIM.md`. `book.py` bounds-checks every ctypes integer before
  mutation (`test_submit_rejects_ctypes_integer_wrap_before_mutation`).
- **Stylized-fact thresholds**: JB (`n/6·(s²+k²/4)` ~ χ²₂), Engle ARCH-LM
  (`T·R²` ~ χ²_lags), Ljung–Box (`n(n+2)Σρ²/(n−k)`), and DFA (log-log RMS
  slope) in `metrics/{serial,fractal}.py` match the standard definitions;
  decision rules in `validate_stylized_facts` match `MARKET_SIM.md`'s
  table, and short samples return `inconclusive` rather than failing.
- **Synthetic labeling elsewhere**: `simtest/feed.py` stamps
  `source="synthetic"` on every row and generates only from the session
  runtime's uniform stream (replay-identical); `microstructure/synthetic_lob.py`
  uses `default_rng(seed)` with a documented draw order, `source="synthetic"`,
  `revision_id="SYNTHETIC_LOB_v1"`, and per-parent OFI shifts that avoid
  cross-day leakage; `tests/perf/synthetic.py` uses fixed seeds and a fixed
  ingested stamp; the e2e pipeline test asserts
  `manifest["source"] == "synthetic"`.

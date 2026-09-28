# ADR-0007: Shadow and challenger slots record intent only — never capital

## Status

Accepted (discovered; documents existing behavior)

## Context

The paper loop runs one *champion* slot (simulated capital, real fills in
the simulated book) plus optional *shadow* and named *challenger* slots —
candidates being compared against the champion on identical bars. The
danger case: a challenger that accidentally moves capital contaminates the
champion's book and any downstream "paper P&L" claim.

`SimulatedBroker` enforces the separation structurally
(`execution/simulated_broker.py`):

- `allow_capital=False` → `cash = 0.0` at construction and `submit()`
  short-circuits to recording an `ACKED` intent — the risk gate, cost, and
  cash paths are unreachable for shadow slots (except in `target_to_orders`
  sizing, which is intent math only).
- `exposures()` refuses to floor a zero NAV into a denominator — a shadow
  reports `(0.0, 0.0)` gross/net rather than division by a synthetic NAV.
- Shadow "positions" are weight snapshots (`shadow.shares[sid] = s_tgt`),
  tracked for divergence metrics only.
- `process_bar` returns immediately for non-capital slots so sweeping
  resting intents cannot manufacture rejects against a zero NAV.

Champion-vs-shadow comparison is a **weight L1 divergence** metric
(`_weight_l1_divergence`), which feeds `promotion_dry_run` — a receipt that
"never moves live capital" (`paper/loop.py` docstring), gated by
`promote_max_mean_l1`/`promote_min_steps` and vetoed when the kill switch
tripped mid-run (`kill_tripped_mid_run` is preserved across resumes).

## Decision

Capital is a slot-level capability flag, not a config suggestion:
`allow_capital=False` slots cannot reach the cash path at all.

## Consequences

- Challenger experiments (e.g. `paper.challenger_scales`, scaled shadow
  `* 0.85`) are safe by construction — an N-challenger grid cannot distort
  the champion book or the promotion record.
- Shadow equity rows have `nav: NaN` by design — downstream analytics must
  treat challenger value as absent, not zero.
- Promotion remains a *dry run* verdict on divergence, never a transfer;
  `live_pnl_claim: false` is stamped into every metrics payload.

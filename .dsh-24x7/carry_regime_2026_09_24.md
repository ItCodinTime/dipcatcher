# Carry lane — local book rebuild + regime structure, 2026-09-24

## Rebuild
`data/binance_carry` rebuilt locally on this machine (was remote-only): Binance
public archive, 144 coins kept of 178 candidates, 179k perp / 197k spot bars,
722k funding events, coverage 2020-01-01 → 2026-08-31. Hyperliquid perp/funding
fetch in progress for the cross-venue arb book (119-pair ∩, see
`docs/carry_expansion_2026_09.md`).

## Frozen-config protocol (no holdout peeking)
`scripts/run_book_grid.py`: 108-config grid on dev (pre-2025), selection score
= min(Sharpe_h1, Sharpe_h2) + 0.01·dev Sharpe with liquidation/margin
penalties; freeze; evaluate holdout once.

- Dev champion `g30` (enter 3e-4, exit −1.25e-4, lb 9, nw 0.08, mx 15):
  dev Sharpe **8.04** (+207%, DD −0.7%, 0 liq) → **holdout 0.67** (+2.2%, DD −2.7%)
  → gate NOT PROVEN.
- Worst-year champion `g9` (enter 2e-4, exit −1.25e-4, nw 0.12): positive every
  dev year (2021 10.96 / 2022 +4.71 / 2023 10.94 / 2024 11.85) — genuinely
  regime-robust inside dev → **holdout 0.44**. Still fails.

## Structural finding: the carry premium is regime-gated, twice
1. **2022 break** (bear market): funding flipped mixed-sign; most configs go
   negative (−1 to −2 Sharpe). Robust configs (small enter threshold, 15
   names, nw 0.12) stay +3.5–4.7 — survivable.
2. **2025+ break** (different mechanism): absolute funding magnitude
   compressed ~3x across the universe — mean |rate| 4.15e-4 (2021) → ~1.3e-4
   (2022–2026) and frac-positive 0.91 → 0.66. Even the year-invariant config
   earns ~0. There is no carry left to harvest on Binance in-holdout — not a
   tuning failure.

## Consequence
Binance-only carry cannot clear holdout Sharpe>5: the premium it monetizes
does not exist post-2025 at the required scale. The only structurally
different signal available is the **cross-venue funding spread** (HL vs
Binance), which is microstructure-driven and may persist where absolute carry
died. Arb book eval follows when the HL fetch completes; "failure is a
result" applies either way.

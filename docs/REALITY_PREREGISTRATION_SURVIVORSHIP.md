# Survivorship-aware reality trial pre-registration

Frozen before any strategy result was computed. The machine-readable copy is
`research/reality/survivorship/preregistration.json`. The runner must refuse a
ledger, cost, or risk-gate config that differs from that file, and it must
record every grid cell, including cells that lose.

This is a research diagnostic. It is not a live-trading claim, not a
promotion, and not a broker test. Reality-filter thresholds are not edited.
Round-1 ledger rows, the round-1 receipt, the round-1 audit, and sealed
Phase-1 receipts are not rewritten.

## Why a second study

Round 1 (`reality-us-liquid-daily-2026-09-27`) ran 29 trials on today's
`US_LIQUID` names. The selected cell was equal-weight long. Its holdout
compounded net return was about 221 percent with an annualized ratio near
2.2. Those names were chosen with hindsight: they are liquid now. Names that
failed are absent. This study asks how that baseline looks when membership
is a historical list, and it adds the new trials to the same ledger so the
deflation counts every trial run so far (29 plus this grid).

## Universe

Membership is the Wikipedia-derived S&P 500 change log vendored from
`coiltrade/point-in-time-sp500` commit `e3b838430677f8c28bad9cf505b5b05a3ca85d26`
(`research/reality/survivorship/membership.json`, sha256
`6bedcb36dbf5713bec084f8b4c44ea7d59680a0a05a39a1214d712c52f00df19`).
License: CC BY 4.0, Coil; upstream Wikipedia "List of S&P 500 companies",
CC BY-SA.

`members_asof(D)` starts from the 2026-06-30 snapshot and undoes every change
dated after D. A change dated D is already in effect on D. A name can be held
on D only if it is a member that session, the bar is a real Yahoo print, and
the session is not the name's last real print.

Prices are Yahoo v8 daily bars through the existing adapter. The only symbol
edit is `.` to `-` (`BRK.B` to `BRK-B`). There is no rename table. A ticker
Yahoo does not serve is counted as missing and dropped. SPY is fetched as the
regime series and is not a portfolio constituent.

If coverage of members with a real bar is below 0.40 on the first session on
or after 2016-01-04, 2020-01-02, or 2023-01-03, the run stops and appends
nothing.

## What is still biased

This is not CRSP. The change log is Wikipedia's selected-changes table.
Dates can be announcement dates. The upstream file's price check starts at
2016-06-30, so the first months of the train window sit in the unvalidated
tail. Renames that were not a remove/add pair keep the later ticker
(`META`, `BKNG`). Yahoo's series under that ticker is used as-is.

Yahoo will not have every removed name. Those holes are residual
survivorship. They are listed, not filled with survivors.

When a series ends before the panel does, one exit bar copies the last
close so the existing engine can flatten. There is no extra haircut,
because the file does not say whether the name was acquired or wiped out.
A bankruptcy whose last print was still high is marked too kindly.

Closes are quote closes. Dividends are not in the adapter. The frozen net
cap stays 0.2, the same budget as round 1, so the baseline comparison is
not a fully invested index.

## Books

All thirteen cells use `configs/backtest.yaml` costs: 1 bp commission,
5 bp half-spread, impact coefficient 0.1, 50 bp/year borrow, next-open
fills, friction on. The risk gate is unchanged (gross 2, net 0.2, name
0.05). Orders it rejects stay rejected.

- Cross-sectional momentum: lookback 63, 126, 252; skip 21; quintile 0.2;
  rebalance every 21 sessions. Three cells.
- Short-term reversal: lookback 5 and 21; skip 1; rebalance on the lookback.
  Two cells.
- Low volatility: trailing window 21, 63, 126; long the calm quintile;
  sized like the equal-weight baseline. Three cells.
- Dip buy with a regime filter: `sweep_reclaim_weights` lookback 20 or 40,
  hold 10 or 21, decay 1, and only the long side, and only when the SPY
  63-session trailing return is negative. Four cells.
- Equal-weight long on the point-in-time members. One cell.

Formation returns use a shift, so the decision row's own close is not an
input. A leg with fewer than five names is flat that session.

## Splits and ledger

Train 2016-01-04 inclusive to 2020-01-01 exclusive. Validation 2020-01-02
to 2023-01-01. Holdout 2023-01-03 to 2025-01-01. Same windows as round 1.

Every new cell is appended to `research/reality/trials.jsonl`. The study
winner is the best validation per-period ratio among the thirteen. The
reality filter then scores the whole file. Its best row may still be a
round-1 trial. That is intended: deflation uses every cluster ever
recorded. Holdout is written only after the append. CSCV PBO uses the
thirteen new train-then-validation paths and does not include the original
twenty-nine, because those return series were not committed. The gate's own
PBO field stays empty for the same reason it did in round 1.

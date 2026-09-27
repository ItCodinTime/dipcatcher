# Paper replication lab

`src/quant_fund/research/replication/` re-implements five well-known,
publicly documented equity anomalies, faithful to the original signal
definitions, and scores them with the lab's receipt/honesty harness.

**Data honesty**: the only available panel is the lab-derived
`silver/bars.parquet` (symbols `S0000..S0034` + `MKT`, ~504 daily bars,
`planted_signal` column) — it is **synthetic**, and every receipt is labeled
`data_label="SYNTHETIC"`. Results measure whether the *harness* reproduces a
paper's logic on the data available; nulls and contradictions are reported
verbatim. Two caveats discovered during the build and recorded in receipts:

- `open ≡ prev close` in the lab panel → the overnight leg is identically
  zero, so `overnight_lps2019` reports `untestable` there (and `tug_of_war`
  degenerates to a 20-day reversal variant — flagged in `data_notes`).
- 35 assets × ~2 years is far below CRSP scale — small-sample caveat on all
  verdicts.

## Strategies and citations

| Key | Paper | Signal | Primary test |
|---|---|---|---|
| `tsmom_mop2012` | Moskowitz, Ooi & Pedersen (2012), JFE 104(2), doi:10.1016/j.jfineco.2011.11.003 | `sign(R_{t-252,t}) / sigma_t` (EWMA vol, com=60) | Spearman IC vs next-21d return, NW t |
| `str_jegadeesh1990` | Jegadeesh (1990), JoF 45(3), doi:10.1111/j.1540-6261.1990.tb05110.x | `-(sum log ret_cc, 21d)` | IC vs next-21d return |
| `str_lehmann1990` | Lehmann (1990), QJE 105(1), doi:10.2307/2937816 | `-(sum log ret_cc, 5d)` | IC vs next-5d return |
| `lowvol_bbw2011` | Baker, Bradley & Wurgler (2011), FAJ 67(1), doi:10.2469/faj.v67.n1.4 | `-(rolling std ret_cc, 252d)` | IC vs next-21d return |
| `bab_fp2014` | Frazzini & Pedersen (2014), JFE 111(1), doi:10.1016/j.jfineco.2013.10.005 | `-(0.6 * beta_hat + 0.4)`, `beta_hat = corr_i,M * sigma_i/sigma_M` on 252d daily | IC vs next-21d return |
| `overnight_lps2019` | Lou, Polk & Skouras (2019), JFE 134(1), doi:10.1016/j.jfineco.2019.03.011 | `sum(log(1+ret_on), 20d)` | IC vs next-day **overnight** return |
| `tug_of_war_lps2019` | Lou, Polk & Skouras (2019), same paper | `ON_20d - ID_20d` | IC vs next-day total return |
| `tom_mx2008` | McConnell & Xu (2008), FAJ 64(2), doi:10.2469/faj.v64.n2.8; Lakonishok & Smidt (1988), RFS 1(4), doi:10.1093/rfs/1.4.403 | 1 on last trading day + first 3 of next month, else 0 | in/out-of-window mean diff (Welch) on equal-weight market |

## Scoring (honesty contract)

Headline numbers are **proper scores only**:

- `direction_brier`, `direction_log_loss`, `direction_ece` — Brier / log-loss /
  ECE of a walk-forward directional forecaster `P(r_{t+1}>0 | signal bucket)`
  estimated on expanding history strictly before `t`, vs a `climatology`
  baseline (expanding unconditional rate). `brier_skill_vs_climatology` is the
  headline proper-score delta.
- `h{H}_mean_ic`, `h{H}_ic_tstat_nw` — per-date cross-sectional Spearman IC
  of signal vs forward returns with Hansen-Hodrick overlap-aware Newey-West
  lags via `metrics.inference`.

Anything P&L-shaped is labeled `descriptive_*` (decile spreads, in/out window
means) — never a headline, per `FORBIDDEN_RESEARCH_METRIC_KEYS`
(sharpe/sortino/calmar/pnl/nav are refused at receipt write time).

`verdict` is machine-readable: `replicated` needs the paper's sign **and**
|t| >= 2 (or Welch p < 0.05 for `tom_mx2008`); `contradicted` = significant
opposite sign; `inconclusive` = under-powered; `untestable` = the dataset
cannot express the anomaly (reason recorded).

## Reproduce

```bash
# on the lab-derived (synthetic) panel:
uv run --no-sync python -m quant_fund.research.replication \
    --data /path/to/data/silver/bars.parquet --out-dir receipts

# fully synthetic labeled fixture (no parquet needed):
uv run --no-sync python -m quant_fund.research.replication --synthetic

# verify a sealed receipt (re-hash + honesty invariants):
uv run --no-sync python -m quant_fund.research.replication.verify \
    receipts/paper_replication_<hash>.json
```

Tests: `uv run --no-sync pytest tests/unit/replication tests/property/test_replication_properties.py -q`
(property tests cover signal boundedness, determinism, and a no-lookahead
truncation/shift test for every builder).

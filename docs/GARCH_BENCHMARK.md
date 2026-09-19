# GARCH frozen benchmark protocol (v1, 2026-09-17)

`scripts/benchmark_garch.py` runs a fixed, reproducible comparison of
volatility forecasts on a bounded sample. It documents its own limits.

## What it measures

- Target: sum of the next `h=5` squared log returns per security (realized
  variance proxy).
- Forecasters: rolling 20-day mean squared return, RiskMetrics EWMA (0.94),
  and zero-mean GARCH(1,1)-Normal, GARCH(1,1)-t, GJR(1,1,1)-t, EGARCH(1,1,1)-t
  (percent units, `rescale=False`, maxiter 500, seeded distributions).
- Origins: nonoverlapping every 5 dates. Fits use only returns observed
  before the origin. Validation origins start at return index 200; test
  origins at 300. Selection uses validation mean loss only, before any test
  scoring. Production `train_volatility` GARCH QLIKE now uses the same
  nonoverlapping origin contract on the date-level equal-weight panel
  (Wave 104); this frozen script remains the independent per-series protocol.
  Production   `train_volatility` also scores the one-step date-level return
  density (log-score / CRPS / PIT; Wave 105) without changing this script.
  Production name-level walk-forward QLIKE/density (Wave 114) is a separate
  `security_level_ret_1` namespace and is also outside this frozen script.
  Production Realized GARCH on daily Parkinson (Wave 115) is likewise outside
  this frozen script and does not replace the return-only overlay.
- Losses: QLIKE `log(forecast) + target/forecast`, equal-weighted across
  five complete synthetic series per origin date.
- Inference: two-sided HAC Diebold–Mariano versus each baseline, Bonferroni
  over two comparisons, α = 0.05, minimum 30 date observations.

## What it cannot claim

- `sota_proven` is hard-coded `False`. Synthetic bars (36 securities, 504
  dates, complete-history selection) cannot certify real-market accuracy.
- The candidate set omits realized-GARCH and modern hybrid estimators. Production
  `GARCHVol` now also exposes APARCH and FIGARCH (Wave 106); they are not part
  of this frozen v1 candidate set. Production `RealizedGARCHVol` (Wave 115) uses
  daily Parkinson OHLC, not high-frequency RV, and is also outside this frozen
  set. A HAR-style daily-squared-return proxy was added after v1 and compared
  exploratorily against the recorded GJR-t losses (it slightly beat them); it
  was not part of the pre-registered v1 selection.
- Failed or nonconverged fits fall back to the rolling forecast and stay in
  the scored sample; fallback counts are reported.
- The estimators here are deliberately simple; the production `GARCHVol`
  pipeline (percent-scale likelihood diagnostics, `forecast()`, joblib
  contract) is exercised by `tests/unit/test_garch_contract.py` instead.

## Real-data extension (v1-real, 2026-09-17)

`/tmp/dipcatcher-sp500-real-v1.json` extends the same protocol to **real
market data**: the S&P 500 adjusted-close series bundled with `arch`
(1999-01-04 → 2018-12-31, SHA-256 recorded in the protocol file).

- Validation: 120 origins from 2000-01-03 (dot-com era), selected GJR-t.
- Test: 180 nonoverlapping 5-day windows, 2015-06-04 → 2018-12-31, untouched
  by selection. Expanding refits at every origin, zero-mean likelihoods,
  seeded distributions, fallback-to-rolling on any failure (counted).
- Inference: HAC Diebold–Mariano vs rolling/EWMA/HAR-proxy with Bonferroni ×3.
- The same `sota_proven: false` constraint applies: one historical index,
  no live trading, daily-frequency target, and a proxy HAR. This is real-data
  evidence about these specific estimators — not a broad SOTA certificate.

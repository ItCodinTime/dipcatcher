# Data-quality scorecard

`quant_fund.data.quality` produces a per-dataset quality report for
bar/quote frames. It **composes** the fail-closed structural checks already
in `quant_fund.data.lakehouse.quality` (`quality_report`) rather than
forking them, and adds the checks that module does not cover.

```python
import polars as pl
from quant_fund.data.quality import score_bars, report_json, write_report

report = score_bars(pl.read_parquet("bars.parquet"), dataset="aapl-1m")
print(report.score, report.passed)          # score in [0, 1]; passed = zero violations
text = report_json(report)                  # deterministic JSON (sorted keys)
write_report(report, "reports/aapl-1m.json")  # receipt-style artifact
```

There is also a minimal CLI (read-only, exit 1 on any violation):

```bash
uv run --no-sync python -m quant_fund.data.quality.cli bars.parquet --out report.json
```

## Checks

| Check | Source | What it flags |
|---|---|---|
| `duplicates` | lakehouse | rows sharing a `(symbol, clock)` key |
| `non_monotone` | lakehouse | clock moving backwards in stored order |
| `ohlc` | lakehouse | `high < low`, `open`/`close` outside `[low, high]`, non-positive prices |
| `outliers` | lakehouse | abs log return above `max_abs_log_return` (default 0.5) |
| `stale_prices` | lakehouse | runs of `stale_run_length` (default 5) identical closes |
| `non_finite` | new | rows with a null/NaN/±inf cell in a numeric column |
| `timezone` | new | clock dtype that cannot/does not carry a timezone |
| `volume` | new | rows with `volume <= 0` |
| `mad_outliers` | new | log returns with modified z-score `0.6745·(r−median)/MAD` beyond `mad_z_threshold` (default 3.5) |
| `missing_bars` | new | adjacent spacings beyond `gap_factor`× expected interval |

## Expected interval (missing bars)

`missing_bars` is a **spacing heuristic, not a calendar**: the expected
interval is `expected_interval_seconds` when configured, else the per-symbol
median of positive adjacent spacings. A spacing above
`expected × gap_factor` (default 1.5×) is one gap event with
`round(spacing/expected) − 1` estimated missing bars. It does not know
sessions, holidays, or half-days — the calendars track owns that; feed it
session-aligned data or set `expected_interval_seconds` explicitly.

## MAD outliers

The modified z-score uses a dataset-level median/MAD of finite
close-to-close log returns. MAD is floored at `1e-12` so floating-point
noise on near-constant return series cannot produce outliers, while a real
deviation still yields an effectively unbounded z when MAD is zero.

## Score

Each applicable check contributes `pass_rate = 1 − violations/checked`
(clamped to `[0, 1]`). The dataset `score` is the weighted mean of pass
rates; structural checks (`duplicates`, `non_monotone`, `ohlc`,
`non_finite`) carry weight 3, `timezone`/`volume`/outlier/`missing_bars`
carry 2, `stale_prices` carries 1 (`CHECK_WEIGHTS`). `passed` requires zero
violations on every applicable check. Reports carry
`schema = "dipcatcher.data.quality.v1"`, per-check `violations`/`checked`,
and up to `max_offenders` (default 8) worst offenders with timestamps.
`report_sha256` hashes the canonical payload for receipt citation.

## Limits

- Dataset-level median/MAD for outlier scoring; per-regime outliers in
  mixed-volatility frames may need per-symbol rescoring.
- `missing_bars` infers spacing — mixed-frequency frames should be scored
  per frequency or with a configured `expected_interval_seconds`.
- Non-temporal (integer) clocks are supported for ordering/gaps but skip
  the timezone check as not applicable.
- Empty frames score vacuously clean (`score=1.0`, `rows=0`) — gate on
  `rows` separately if emptiness is a defect for the caller.

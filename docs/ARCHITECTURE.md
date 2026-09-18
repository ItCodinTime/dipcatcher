# Architecture

Dipcatcher is a research and paper-trading platform that estimates **market state** at a point-in-time decision clock, then allocates and executes under constraints and costs. It is not a price-prediction bot, a technical-indicator script, or an LLM that emits BUY/SELL.

A model is useful only if it shows stable **out-of-sample economic value** after transaction costs, impact, turnover, risk constraints, and multiple-testing corrections. Statistical metric improvement is necessary but not sufficient.

## Pipeline

```text
RAW MARKET DATA
       ↓
POINT-IN-TIME DATA LAYER
       ↓
FEATURE ENGINE  +  LABEL ENGINE
       ↓
RETURN DISTRIBUTION | RANK | ALPHA | VOL | COVARIANCE | REGIME | TAIL | LIQUIDITY
       ↓
FORECAST CALIBRATION
       ↓
FORECAST FUSION
       ↓
CONSTRAINED PORTFOLIO OPTIMIZER
       ↓
PRE-TRADE RISK ENGINE
       ↓
EXECUTION PLANNER
       ↓
EVENT-DRIVEN SIMULATION / PAPER / SHADOW / LIVE
       ↓
ATTRIBUTION + MONITORING + RETRAINING
```

The market-state object at decision time \(t\) is

\[
Z_t = \{\mu_t, F_t, \alpha_t, s_t, \sigma_t, \Sigma_t, \pi_t^{\text{regime}}, \text{tail}_t, \text{liq}_t, u_t\}.
\]

The optimizer consumes \(Z_t\). No forecast engine may bypass the risk gate.

## Package layout

Python package `quant_fund` lives under `src/quant_fund/`. The git repository remains `dipcatcher`.

Packages exist only when they contain implementations. Empty `pass` modules are forbidden except on abstract protocols that subclasses must implement.

| Package | Responsibility |
|---|---|
| `config` | Nested Pydantic settings, YAML inheritance, resolved-config dumps |
| `schemas` | PIT records, bars, forecasts, orders, portfolio snapshots |
| `utils` | Logging, seeds, hashing, time, numerical guards |
| `data` | Adapters, security master, universe, corporate actions, lake I/O |
| `features` | Deterministic PIT-safe feature families |
| `labels` | Horizon targets for return, risk, tail |
| `validation` | Walk-forward, purge/embargo, CPCV, DSR/PSR |
| `models` | Eight forecast engines + calibration |
| `fusion` | Transparent then optional learned stacking |
| `portfolio` | CVXPY optimizer, constraints, risk, attribution |
| `execution` | Costs, impact, Almgren–Chriss, schedules |
| `backtest` | Event-driven engine, broker, accounting, metrics |
| `registry` | MLflow champion/challenger |
| `monitoring` | Drift, kill switch, promotion gates |
| `api` | FastAPI service layer |
| `cli` | `dipcatcher` Typer commands (`quant` compatibility alias) |

## Runtime modes

Configured by `runtime.mode`:

- `research` — offline experiments (default for local CLI)
- `backtest` — event-driven historical simulation
- `paper` — live clock, simulated fills
- `shadow` — champion plus challenger forecasts, no orders
- `live` — real orders; **requires explicit config**, never inferred

Default is `research`. Live is refused unless `runtime.mode: live` and `runtime.allow_live: true`.

### Backtest valuation integrity

The event-driven backtester tracks the age of every close mark used to value a held
position. A missing execution bar is not treated as a zero price, and a held name
cannot be carried indefinitely at its last observed mark. Once the age exceeds
`risk_gate.stale_price_bars`, the run fails closed with `StaleValuationError`
rather than emitting fabricated NAV, P&L, exposure, or risk-gate inputs. Backtest
metrics receipts are published through same-directory temporary files and atomic
rename, so a failed write cannot replace a previously valid receipt with partial
JSON. Before committing a fill, the backtester also requires every buy's notional
plus transaction costs to be covered by current book cash. Cash-insufficient buys
are rejected and counted in `metrics["cash_rejects"]`; the accounting path never
creates a negative cash balance. Sell proceeds are applied normally and increase
available cash for later fills.

## Time integrity

Every observation carries `event_time`, `available_time`, `ingested_time`, `source`, `security_id`, `revision_id`.

For features used at decision time \(t\):

\[
\text{available\_time} \le t.
\]

Violation raises `PointInTimeError`. Forward-fill from the future, revised-as-vintage, and look-ahead universe membership are leakage bugs, not warnings.

## Data lake

Parquet + DuckDB. Layout:

```text
data/raw/          vendor or file drops (gitignored payloads)
data/bronze/       PIT-normalized bars, actions, master
data/silver/       raw + adjusted prices, universe membership
data/gold/         features, labels, forecasts
data/metadata/     fingerprints, resolved configs
```

Partition by `asset_class=equity/date=YYYY-MM-DD/` with coarse files (not one file per symbol-day).

## Execution clock

If a signal uses the close at day \(t\), the strategy cannot fill at that close unless an explicit auction model is configured. Default fill is the **next available open** (ADR-005).

## What this system does not do

- Guarantee profitability
- Let an LLM size positions
- Silently relax infeasible optimizer constraints
- Auto-flatten the book on a software exception
- Treat synthetic-market results as evidence of real edge

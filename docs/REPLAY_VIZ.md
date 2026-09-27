# REPLAY_VIZ — market replay visualizer

A dependency-light TypeScript + canvas-2D viewer for recorded bar /
order-book replay sessions. Lives in `replay/` (own `package.json`; the
`web/` package is a separate track). Everything it renders is either a
labeled **SYNTHETIC** fixture or a locally produced recording — it is a
research/debugging tool, not market evidence and not a trading surface.

## What it renders

- **Candlestick small-multiples grid** — one cell per symbol (8 in the
  fixture), OHLC + volume strip, last-price tag on the focused cell.
- **Depth heatmap** — focused symbol's L2 book over the trailing window:
  x = snapshot time, y = price, bids in teal / asks in ember, intensity =
  accumulated size, white mid trace.
- **Trades tape** — newest-first DOM tape for the focused symbol.
- **Strategy decision markers** — `▲`/`▼` glyphs overlaid on the chart at
  each marker's `event_time` (buy below the bar, sell above), plus ticks on
  the timeline.
- **Timeline** — focused-symbol close sparkline, elapsed shading, visible
  window box, draggable playhead.

## Controls

| Control | Action |
|---|---|
| ▶ / ⏸ or `Space` | play/pause |
| speed select | playback rate in bars/second (1×–60×) |
| `←` / `→`, `−1`/`+1` | step one bar |
| mouse wheel / `⟷±` | zoom the trailing window (10..390 bars) |
| drag/click timeline | scrub to any point in the session |
| click a chart cell | focus symbol (drives depth + tape) |

## Run it

```bash
cd replay
npm ci
npm run build        # tsc -> dist/
npm run serve        # http://localhost:8971
# or: node scripts/serve.mjs  (PORT env override)
```

Open `http://localhost:8971`. Query params:

- `?session=public/fixtures/session.synthetic.json` — session URL (default)
- `?bench=1&symbols=8&bars=390&frames=240&seed=7` — in-browser synthetic
  session + frame-time benchmark sweep; stats land in `#bench-out` and
  `window.__benchResult`.

## Tests

```bash
cd replay && npx playwright test        # smoke + screenshot + bench timing
uv run --no-sync pytest tests/unit/replay_viz -q   # generator unit tests
```

The Playwright suite loads the committed fixture, seeks, asserts the canvas
is non-blank and the tape populates, stores `test-results/replay-viewer.png`
as an artifact, and asserts bench mean frame time < 50 ms (real numbers are
recorded in the PR / test output, not gated tightly — smoke level).

## Session format (`dipcatcher.replay.session`, v1)

One compact JSON document; columnar arrays keep it small and decode straight
into `Float32Array`/`Float64Array`. Field names mirror the repo schemas:

```jsonc
{
  "format": "dipcatcher.replay.session",
  "format_version": 1,
  "source": "synthetic",
  "session_date": "2025-01-06",
  "timezone": "America/New_York",
  "bar_interval_seconds": 60,
  "generated_by": "replay/scripts/gen_fixture.py (SYNTHETIC)",
  "symbols": [
    // mirrors quant_fund.schemas.market.SecurityRecord fields
    { "security_id": "SYNTH-ALFA", "symbol": "ALFA", "name": "...",
      "exchange": "SYNTH", "currency": "USD", "sector": "Synthetic" }
  ],
  "bars": {
    // mirrors quant_fund.schemas.market.Bar; event_time = minute CLOSE
    // (hf_ohlcv_1m adapter convention), epoch ms
    "ALFA": { "event_time": [..], "open": [..], "high": [..],
              "low": [..], "close": [..], "volume": [..] }
  },
  "books": {
    // mirrors OrderBookSnapshot: bids best→worse (desc), asks asc,
    // row-major [snapshot][level], `depth` levels per side
    "ALFA": { "event_time": [..], "depth": 5,
              "bid_price": [[..]], "bid_size": [[..]],
              "ask_price": [[..]], "ask_size": [[..]] }
  },
  "trades": {
    "ALFA": { "event_time": [..], "price": [..], "quantity": [..],
              "side": ["buy"|"sell", ..] }   // aggressor side
  },
  "markers": [
    // mirrors Order-like fields: signal→decision→order time chain
    { "symbol": "ALFA", "event_time": 173616.., "side": "buy",
      "kind": "entry", "quantity": 100, "limit_price": 101.25,
      "decision_time": .., "order_time": .., "note": "..." }
  ]
}
```

## Regenerating the fixture

```bash
python3 replay/scripts/gen_fixture.py            # -> replay/public/fixtures/session.synthetic.json
python3 replay/scripts/gen_fixture.py --symbols 8 --bars 390 --seed 7 --out <path>
```

The generator is stdlib-only Python, deterministic per `--seed`, validates
OHLC consistency / uncrossed sorted book sides / sorted tape on the way out,
and prints the emitted size (kept < 2 MB; current fixture ≈ 1.5 MB).
`replay/src/synth.ts` mirrors it in-browser for `?bench=1` mode so perf can
be measured without a large download.

## Performance approach

- All series decoded once into `Float32Array`/`Float64Array`/`Uint8Array`.
- Chart grid: binary search on `event_time` per cell for the visible range
  (viewport culling); candles batched into two `Path2D`s (up/down) per cell.
- Depth: per-frame accumulation into `nCols × 160` `Float32Array` grids,
  rasterized via `ImageData` into a small offscreen canvas then drawn scaled.
- Tape: at most 64 DOM rows, rebuilt only when the cursor crosses a bar step.
- One `requestAnimationFrame` loop; renders only when state is dirty while
  paused; `devicePixelRatio`-aware canvas sizing.

Measured on `?bench=1` (390 bars × 8 symbols, 240-frame cursor sweep
cycling focus): see `window.__benchResult` / CI artifact — reported in the
PR body (mean single-digit ms/frame on an M-series laptop; CI software
rendering is slower but well under the 50 ms test bound).

## Limitations / open risks

- Fixture data is SYNTHETIC (labeled); wiring to real recorded sessions is
  format-compatible but no real-recording ingest is included here.
- Books are keyed per-symbol; symbols with no `books` entry show an empty
  depth pane (by design).
- Playback maps bars/second, not wall-clock time scaling of event times.
- Headless CI canvas timing is CPU-side draw time; real vsync-bound fps is
  higher. No pixel-diff assertions — smoke-level screenshots only.

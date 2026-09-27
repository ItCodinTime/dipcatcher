/**
 * Benchmark harness (?bench=1): sweeps the cursor across a synthetic session,
 * times each full render (chart grid + depth + timeline), and reports stats.
 * Numbers are CPU-side draw-call ms per frame (performance.now() around the
 * render call) — a reproducible lower bound on frame cost, not vsync-bound fps.
 */

import type { Session } from "./types.js";
import type { ViewState } from "./view.js";

export interface BenchResult {
  kind: "replay-viz-bench";
  session: string;
  symbols: number;
  bars: number;
  frames: number;
  warmup: number;
  mean_ms: number;
  p50_ms: number;
  p95_ms: number;
  max_ms: number;
  min_ms: number;
  fps_equiv: number;
  note: string;
}

function quantile(sorted: number[], q: number): number {
  const i = Math.min(sorted.length - 1, Math.max(0, Math.floor(q * (sorted.length - 1) + 0.5)));
  return sorted[i]!;
}

export function runBench(
  session: Session,
  view: ViewState,
  renderAll: () => number,
  frames = 240,
  warmup = 30,
): BenchResult {
  const n = session.master.length;
  const times: number[] = [];

  view.windowBars = Math.min(120, n);
  for (let i = 0; i < warmup; i++) {
    view.cursor = Math.min(n - 1, i);
    renderAll();
  }

  for (let f = 0; f < frames; f++) {
    // sweep the cursor across the whole day; also cycle focus so every
    // symbol's chart/depth/tape path is exercised
    view.cursor = (f / Math.max(1, frames - 1)) * (n - 1);
    view.focus = f % session.symbols.length;
    times.push(renderAll());
  }

  const sorted = [...times].sort((a, b) => a - b);
  const sum = times.reduce((a, b) => a + b, 0);
  const mean = sum / times.length;
  return {
    kind: "replay-viz-bench",
    session: session.sessionDate,
    symbols: session.symbols.length,
    bars: n,
    frames,
    warmup,
    mean_ms: +mean.toFixed(3),
    p50_ms: +quantile(sorted, 0.5).toFixed(3),
    p95_ms: +quantile(sorted, 0.95).toFixed(3),
    max_ms: +sorted[sorted.length - 1]!.toFixed(3),
    min_ms: +sorted[0]!.toFixed(3),
    fps_equiv: +(1000 / mean).toFixed(1),
    note: "SYNTHETIC session; CPU draw-call ms per frame, single rAF render path",
  };
}

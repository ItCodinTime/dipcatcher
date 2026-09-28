/**
 * Entry point: loads a session, wires controls, runs the rAF render loop.
 *
 * Perf model: all panes draw from typed arrays; the chart grid culls via
 * binary search on the visible window; the depth pane rasterizes an
 * nCols x ROWS ImageData per frame; the tape rebuilds at most 64 DOM rows
 * per cursor step. Rendering happens inside a single rAF callback.
 */

import { decodeSession, loadSession, fmtTime } from "./session.js";
import { makeSyntheticSession } from "./synth.js";
import { drawChartGrid, cellAt } from "./chart.js";
import { drawDepth } from "./depth.js";
import { drawTimeline, timelineCursorAt } from "./timeline.js";
import { updateTape } from "./tape.js";
import { runBench } from "./bench.js";
import type { Session } from "./types.js";
import { clamp, fitCanvas, windowTimes, type ViewState } from "./view.js";

const el = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

function qs(): URLSearchParams {
  return new URLSearchParams(location.search);
}

async function boot(): Promise<void> {
  const params = qs();
  const benchMode = params.get("bench");

  let session: Session;
  if (benchMode) {
    const nSym = parseInt(params.get("symbols") ?? "8", 10);
    const nBars = parseInt(params.get("bars") ?? "390", 10);
    const seed = parseInt(params.get("seed") ?? "7", 10);
    session = decodeSession(makeSyntheticSession(nSym, nBars, seed));
  } else {
    const url = params.get("session") ?? "public/fixtures/session.synthetic.json";
    session = await loadSession(url);
  }

  const chartsCv = el<HTMLCanvasElement>("charts");
  const depthCv = el<HTMLCanvasElement>("depth");
  const timelineCv = el<HTMLCanvasElement>("timeline");
  const tapeEl = el<HTMLDivElement>("tape");
  const statusEl = el<HTMLSpanElement>("status");
  const clockEl = el<HTMLSpanElement>("clock");
  const labelEl = el<HTMLSpanElement>("session-label");

  const view: ViewState = {
    cursor: 0,
    windowBars: Math.min(120, session.master.length),
    focus: 0,
    playing: false,
    speed: parseFloat(el<HTMLSelectElement>("sel-speed").value),
  };

  let dirty = true;
  let lastTapeStep = -2;
  let rafId = 0;
  let lastTs = 0;

  function renderAll(): number {
    const t0 = performance.now();
    drawChartGrid(fitCanvas(chartsCv), session, view);
    drawDepth(fitCanvas(depthCv), session, view);
    drawTimeline(fitCanvas(timelineCv), session, view);
    const step = Math.floor(view.cursor);
    if (step !== lastTapeStep) {
      updateTape(tapeEl, session, view);
      lastTapeStep = step;
    }
    const [, , idx] = windowTimes(session, view);
    const t = session.master[idx];
    if (t !== undefined) {
      clockEl.textContent = `${fmtTime(t, session.timezone)}  bar ${idx + 1}/${session.master.length}`;
    }
    return performance.now() - t0;
  }

  function frame(ts: number): void {
    rafId = requestAnimationFrame(frame);
    const dt = lastTs ? (ts - lastTs) / 1000 : 0;
    lastTs = ts;
    if (view.playing) {
      const n = session.master.length;
      view.cursor = clamp(view.cursor + dt * view.speed, 0, n - 1);
      if (view.cursor >= n - 1) setPlaying(false);
      dirty = true;
    }
    if (dirty) {
      renderAll();
      dirty = false;
    }
  }

  function markDirty(): void {
    dirty = true;
  }

  function setPlaying(p: boolean): void {
    view.playing = p;
    el<HTMLButtonElement>("btn-play").textContent = p ? "⏸" : "▶";
  }

  function seek(frac: number): void {
    view.cursor = clamp(frac, 0, 1) * (session.master.length - 1);
    markDirty();
  }

  function setFocus(i: number): void {
    view.focus = clamp(i, 0, session.symbols.length - 1);
    markDirty();
  }

  // --- controls ---
  el<HTMLButtonElement>("btn-play").addEventListener("click", () => setPlaying(!view.playing));
  el<HTMLSelectElement>("sel-speed").addEventListener("change", (e) => {
    view.speed = parseFloat((e.target as HTMLSelectElement).value);
  });
  el<HTMLButtonElement>("btn-step-back").addEventListener("click", () => {
    view.cursor = clamp(Math.floor(view.cursor) - 1, 0, session.master.length - 1);
    markDirty();
  });
  el<HTMLButtonElement>("btn-step-fwd").addEventListener("click", () => {
    view.cursor = clamp(Math.floor(view.cursor) + 1, 0, session.master.length - 1);
    markDirty();
  });
  el<HTMLButtonElement>("btn-zoom-in").addEventListener("click", () => {
    view.windowBars = clamp(Math.round(view.windowBars / 1.4), 10, session.master.length);
    markDirty();
  });
  el<HTMLButtonElement>("btn-zoom-out").addEventListener("click", () => {
    view.windowBars = clamp(Math.round(view.windowBars * 1.4), 10, session.master.length);
    markDirty();
  });

  chartsCv.addEventListener("click", (e) => {
    const r = chartsCv.getBoundingClientRect();
    const i = cellAt(session.symbols.length, r.width, r.height, e.clientX - r.left, e.clientY - r.top);
    if (i !== null) setFocus(i);
  });
  chartsCv.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const f = e.deltaY > 0 ? 1.15 : 1 / 1.15;
      view.windowBars = clamp(Math.round(view.windowBars * f), 10, session.master.length);
      markDirty();
    },
    { passive: false },
  );

  // timeline scrub (pointer capture for smooth drag)
  let scrubbing = false;
  const scrub = (e: PointerEvent) => {
    const r = timelineCv.getBoundingClientRect();
    view.cursor = clamp(timelineCursorAt(r.width, session, e.clientX - r.left), 0, session.master.length - 1);
    markDirty();
  };
  timelineCv.addEventListener("pointerdown", (e) => {
    scrubbing = true;
    timelineCv.setPointerCapture(e.pointerId);
    scrub(e);
  });
  timelineCv.addEventListener("pointermove", (e) => {
    if (scrubbing) scrub(e);
  });
  timelineCv.addEventListener("pointerup", () => {
    scrubbing = false;
  });

  window.addEventListener("keydown", (e) => {
    if (e.code === "Space") {
      e.preventDefault();
      setPlaying(!view.playing);
    } else if (e.key === "ArrowLeft") {
      view.cursor = clamp(Math.floor(view.cursor) - 1, 0, session.master.length - 1);
      markDirty();
    } else if (e.key === "ArrowRight") {
      view.cursor = clamp(Math.floor(view.cursor) + 1, 0, session.master.length - 1);
      markDirty();
    }
  });

  new ResizeObserver(markDirty).observe(chartsCv);
  new ResizeObserver(markDirty).observe(depthCv);
  new ResizeObserver(markDirty).observe(timelineCv);

  labelEl.textContent = `${session.sessionDate} · ${session.symbols.length} symbols · source=${session.source}`;

  // test hook
  (window as unknown as { __replay: unknown }).__replay = {
    seek,
    setFocus,
    setPlaying,
    cursor: () => view.cursor,
    renderAll,
    session,
    view,
  };

  statusEl.textContent = `ready — ${session.symbols.length} symbols × ${session.master.length} bars`;
  rafId = requestAnimationFrame(frame);
  void rafId;

  if (benchMode) {
    const frames = parseInt(params.get("frames") ?? "240", 10);
    // let layout settle, then run the benchmark sweep
    requestAnimationFrame(() => {
      const result = runBench(session, view, renderAll, frames);
      const out = el<HTMLPreElement>("bench-out");
      out.style.display = "block";
      out.textContent = JSON.stringify(result, null, 2);
      (window as unknown as { __benchResult: unknown }).__benchResult = result;
      statusEl.textContent = `bench done — mean ${result.mean_ms.toFixed(2)} ms/frame`;
    });
  }
}

boot().catch((err) => {
  const s = document.getElementById("status");
  if (s) s.textContent = `error: ${err instanceof Error ? err.message : String(err)}`;
  console.error(err);
});

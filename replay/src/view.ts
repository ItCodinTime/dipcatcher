/** Shared view state + canvas helpers. */

import type { Session } from "./types.js";

export const COLORS = {
  bg: "#0e1117",
  panel: "#141923",
  grid: "#232c3d",
  border: "#2a3344",
  fg: "#d7dde8",
  dim: "#8892a6",
  up: "#26a69a",
  upWick: "#2bbbad",
  down: "#ef5350",
  downWick: "#f2706d",
  vol: "#3d4b63",
  cursor: "#8f7bff",
  buy: "#35d07f",
  sell: "#ff6d64",
  focus: "#4c9aff",
  mid: "#e8ecf4",
} as const;

export interface ViewState {
  /** float index into session.master — fractional during playback */
  cursor: number;
  /** bars in the trailing visible window */
  windowBars: number;
  /** focused symbol index */
  focus: number;
  playing: boolean;
  /** bars per second */
  speed: number;
}

export interface Pane {
  ctx: CanvasRenderingContext2D;
  w: number;
  h: number;
  dpr: number;
}

/** Resize canvas backing store to CSS size * devicePixelRatio; return 2D ctx. */
export function fitCanvas(cv: HTMLCanvasElement): Pane {
  const dpr = window.devicePixelRatio || 1;
  const r = cv.getBoundingClientRect();
  const W = Math.max(1, Math.round(r.width * dpr));
  const H = Math.max(1, Math.round(r.height * dpr));
  if (cv.width !== W || cv.height !== H) {
    cv.width = W;
    cv.height = H;
  }
  const ctx = cv.getContext("2d");
  if (!ctx) throw new Error("2d context unavailable");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: r.width, h: r.height, dpr };
}

/** Trailing time window [tLo, tHi] in epoch ms for the current view. */
export function windowTimes(session: Session, view: ViewState): [number, number, number] {
  const n = session.master.length;
  const idx = Math.min(Math.max(Math.floor(view.cursor), 0), n - 1);
  const tHi = session.master[idx] ?? 0;
  const lo = Math.max(0, idx - view.windowBars + 1);
  const tLo = session.master[lo] ?? tHi;
  return [tLo, tHi, idx];
}

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

export function fmtPx(p: number): string {
  if (p >= 1000) return p.toFixed(1);
  if (p >= 100) return p.toFixed(2);
  if (p >= 10) return p.toFixed(3);
  return p.toFixed(4);
}

/** Nice-round price ticks for axis labels. */
export function priceTicks(lo: number, hi: number, maxTicks = 5): number[] {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / maxTicks;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const out: number[] = [];
  for (let p = Math.ceil(lo / step) * step; p <= hi + 1e-9; p += step) out.push(p);
  return out;
}

/**
 * Session timeline: overview sparkline of the focused symbol's close, the
 * visible-window shading, a draggable playhead, and marker ticks.
 */

import { fmtTime } from "./session.js";
import type { Session } from "./types.js";
import { COLORS, clamp, type Pane, type ViewState } from "./view.js";

const PAD_T = 6;
const PAD_B = 18;
const PAD_X = 4;

export function drawTimeline(pane: Pane, session: Session, view: ViewState): void {
  const { ctx, w, h } = pane;
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, w, h);

  const master = session.master;
  const n = master.length;
  if (n < 2) return;
  const t0 = master[0]!;
  const tN = master[n - 1]!;

  const plotY = PAD_T;
  const plotH = h - PAD_T - PAD_B;
  const plotW = w - PAD_X * 2;
  const xFor = (t: number) => PAD_X + ((t - t0) / (tN - t0)) * plotW;

  // focused close sparkline
  const bars = session.bars[view.focus]!;
  let lo = Infinity;
  let hi = -Infinity;
  for (const c of bars.c) {
    if (c < lo) lo = c;
    if (c > hi) hi = c;
  }
  if (!(hi > lo)) {
    lo = 0;
    hi = 1;
  }
  ctx.beginPath();
  for (let i = 0; i < bars.t.length; i++) {
    const x = xFor(bars.t[i]!);
    const y = plotY + (1 - (bars.c[i]! - lo) / (hi - lo)) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = COLORS.dim;
  ctx.lineWidth = 1;
  ctx.stroke();

  // elapsed region shading (everything left of the cursor)
  const idx = clamp(Math.floor(view.cursor), 0, n - 1);
  const xc = xFor(master[idx]!);
  ctx.fillStyle = "rgba(76, 154, 255, 0.10)";
  ctx.fillRect(PAD_X, plotY, xc - PAD_X, plotH);

  // visible window shade
  const wLo = clamp(Math.floor(view.cursor) - view.windowBars + 1, 0, n - 1);
  const xw = xFor(master[wLo]!);
  ctx.strokeStyle = COLORS.grid;
  ctx.strokeRect(xw + 0.5, plotY + 0.5, Math.max(2, xc - xw) - 1, plotH - 1);

  // marker ticks (all symbols, small)
  for (const m of session.markers) {
    const x = xFor(m.t);
    ctx.fillStyle = m.side === 0 ? COLORS.buy : COLORS.sell;
    ctx.fillRect(x - 0.5, plotY + plotH - 5, 1.5, 5);
  }

  // playhead
  ctx.strokeStyle = COLORS.cursor;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(xc, plotY - 2);
  ctx.lineTo(xc, plotY + plotH + 6);
  ctx.stroke();
  ctx.fillStyle = COLORS.cursor;
  ctx.beginPath();
  ctx.moveTo(xc - 4, plotY + plotH + 6);
  ctx.lineTo(xc + 4, plotY + plotH + 6);
  ctx.lineTo(xc, plotY + plotH);
  ctx.closePath();
  ctx.fill();

  // time labels
  ctx.font = "10px ui-monospace, Menlo, monospace";
  ctx.fillStyle = COLORS.dim;
  ctx.textBaseline = "top";
  ctx.textAlign = "left";
  ctx.fillText(fmtTime(t0, session.timezone), PAD_X, h - PAD_B + 4);
  ctx.textAlign = "right";
  ctx.fillText(fmtTime(tN, session.timezone), w - PAD_X, h - PAD_B + 4);
  ctx.textAlign = "center";
  ctx.fillStyle = COLORS.fg;
  ctx.fillText(fmtTime(master[idx]!, session.timezone), clamp(xc, 40, w - 40), h - PAD_B + 4);
}

/** Map a css-x coordinate on the timeline canvas to a master index (float). */
export function timelineCursorAt(w: number, session: Session, x: number): number {
  const plotW = w - PAD_X * 2;
  const frac = clamp((x - PAD_X) / plotW, 0, 1);
  return frac * (session.master.length - 1);
}

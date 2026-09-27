/**
 * Candlestick small-multiples grid.
 *
 * Perf notes: per-cell visible range is found by binary search (viewport
 * culling), candles are batched into two Path2D batches (up/down) so each
 * frame is O(visible bars), not O(session).
 */

import { lowerBound, upperBound, fmtTime } from "./session.js";
import type { Session } from "./types.js";
import { COLORS, clamp, fmtPx, priceTicks, windowTimes, type Pane, type ViewState } from "./view.js";

const PAD_TOP = 20;
const PAD_BOT = 16;
const PAD_L = 6;
const PAD_R = 56;
const VOL_FRAC = 0.16;

export function gridShape(n: number): [number, number] {
  const cols = n <= 1 ? 1 : 2;
  return [cols, Math.ceil(n / cols)];
}

/** Hit-test: which grid cell is at css pixel (x, y)? */
export function cellAt(n: number, w: number, h: number, x: number, y: number): number | null {
  const [cols, rows] = gridShape(n);
  const cw = w / cols;
  const ch = h / rows;
  const c = Math.floor(x / cw);
  const r = Math.floor(y / ch);
  const idx = r * cols + c;
  return idx >= 0 && idx < n ? idx : null;
}

export function drawChartGrid(pane: Pane, session: Session, view: ViewState): void {
  const { ctx, w, h } = pane;
  const n = session.symbols.length;
  const [cols, rows] = gridShape(n);
  const cw = w / cols;
  const ch = h / rows;
  const [tLo, tHi, cursorIdx] = windowTimes(session, view);

  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, w, h);

  for (let si = 0; si < n; si++) {
    const x0 = (si % cols) * cw;
    const y0 = Math.floor(si / cols) * ch;
    drawCell(pane, session, view, si, x0, y0, cw, ch, tLo, tHi, cursorIdx);
  }
}

function drawCell(
  pane: Pane,
  session: Session,
  view: ViewState,
  si: number,
  x0: number,
  y0: number,
  cw: number,
  ch: number,
  tLo: number,
  tHi: number,
  cursorIdx: number,
): void {
  const { ctx } = pane;
  const bars = session.bars[si]!;
  const t = bars.t;

  const plotX = x0 + PAD_L;
  const plotY = y0 + PAD_TOP;
  const plotW = cw - PAD_L - PAD_R;
  const plotH = ch - PAD_TOP - PAD_BOT;
  const volH = plotH * VOL_FRAC;
  const pxH = plotH - volH;

  if (plotW < 30 || plotH < 30) return;

  // visible range via binary search
  const iLo = lowerBound(t, tLo);
  const iHi = upperBound(t, tHi);
  const iCursor = upperBound(t, session.master[cursorIdx] ?? Infinity) - 1;

  // price range over visible bars up to nothing (full window range shown)
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = iLo; i < iHi; i++) {
    const l = bars.l[i]!;
    const hh = bars.h[i]!;
    if (l < lo) lo = l;
    if (hh > hi) hi = hh;
  }
  if (!isFinite(lo) || !isFinite(hi) || hi <= lo) {
    lo = 0;
    hi = 1;
  }
  const pad = (hi - lo) * 0.06 || 1e-6;
  lo -= pad;
  hi += pad;

  const slotW = plotW / Math.max(1, view.windowBars);
  const bodyW = Math.max(1, Math.min(slotW * 0.72, 24));
  const xForT = (tt: number) => plotX + ((tt - tLo) / Math.max(1e-9, tHi - tLo)) * (plotW - slotW) + slotW / 2;
  const yForP = (p: number) => plotY + (1 - (p - lo) / (hi - lo)) * pxH;

  // cell frame
  ctx.strokeStyle = si === view.focus ? COLORS.focus : COLORS.border;
  ctx.lineWidth = si === view.focus ? 1.5 : 1;
  ctx.strokeRect(x0 + 0.5, y0 + 0.5, cw - 1, ch - 1);

  ctx.save();
  ctx.beginPath();
  ctx.rect(x0, y0, cw, ch);
  ctx.clip();

  // horizontal grid + price labels
  ctx.font = "10px ui-monospace, Menlo, monospace";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  for (const p of priceTicks(lo, hi, 4)) {
    const y = yForP(p);
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plotX, Math.round(y) + 0.5);
    ctx.lineTo(plotX + plotW, Math.round(y) + 0.5);
    ctx.stroke();
    ctx.fillStyle = COLORS.dim;
    ctx.fillText(fmtPx(p), plotX + plotW + 4, y);
  }

  // volume bars (bottom strip), normalized to max visible volume
  let vmax = 0;
  for (let i = iLo; i < iHi; i++) {
    const vv = bars.v[i]!;
    if (vv > vmax) vmax = vv;
  }
  if (vmax > 0) {
    ctx.fillStyle = COLORS.vol;
    const volY0 = plotY + plotH;
    for (let i = iLo; i < iHi; i++) {
      if (i > iCursor) break;
      const vh = (bars.v[i]! / vmax) * (volH - 2);
      const x = xForT(t[i]!);
      ctx.fillRect(x - bodyW / 2, volY0 - vh, bodyW, vh);
    }
  }

  // candles — two Path2D batches (up/down) up to the cursor
  const upBody = new Path2D();
  const dnBody = new Path2D();
  const upWick = new Path2D();
  const dnWick = new Path2D();
  const n = Math.min(iHi, iCursor + 1);
  for (let i = iLo; i < n; i++) {
    const x = xForT(t[i]!);
    const yo = yForP(bars.o[i]!);
    const yc = yForP(bars.c[i]!);
    const yh = yForP(bars.h[i]!);
    const yl = yForP(bars.l[i]!);
    const up = bars.c[i]! >= bars.o[i]!;
    const body = up ? upBody : dnBody;
    const wick = up ? upWick : dnWick;
    wick.moveTo(x, yh);
    wick.lineTo(x, yl);
    const top = Math.min(yo, yc);
    const hh = Math.max(1, Math.abs(yc - yo));
    body.rect(x - bodyW / 2, top, bodyW, hh);
  }
  ctx.strokeStyle = COLORS.upWick;
  ctx.lineWidth = 1;
  ctx.stroke(upWick);
  ctx.strokeStyle = COLORS.downWick;
  ctx.stroke(dnWick);
  ctx.fillStyle = COLORS.up;
  ctx.fill(upBody);
  ctx.fillStyle = COLORS.down;
  ctx.fill(dnBody);

  // cursor line
  const tC = session.master[cursorIdx];
  if (tC !== undefined && tC >= tLo && tC <= tHi) {
    const x = xForT(tC);
    ctx.strokeStyle = COLORS.cursor;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x, plotY);
    ctx.lineTo(x, plotY + plotH);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // strategy decision markers for this symbol
  for (const m of session.markersBySymbol[si]!) {
    if (m.t < tLo || m.t > tHi || m.t > tC!) continue;
    const x = xForT(m.t);
    const bi = clamp(m.barIndex, iLo, Math.max(iLo, n - 1));
    const yRef = m.side === 0 ? yForP(bars.l[bi]!) + 8 : yForP(bars.h[bi]!) - 8;
    ctx.fillStyle = m.side === 0 ? COLORS.buy : COLORS.sell;
    ctx.beginPath();
    if (m.side === 0) {
      ctx.moveTo(x, yRef + 7);
      ctx.lineTo(x - 4, yRef + 14);
      ctx.lineTo(x + 4, yRef + 14);
    } else {
      ctx.moveTo(x, yRef - 7);
      ctx.lineTo(x - 4, yRef - 14);
      ctx.lineTo(x + 4, yRef - 14);
    }
    ctx.closePath();
    ctx.fill();
  }

  // last price tag (right axis) for the focused cell only
  if (si === view.focus && iCursor >= iLo && iCursor < iHi) {
    const last = bars.c[iCursor]!;
    const y = clamp(yForP(last), plotY + 6, plotY + pxH - 6);
    ctx.strokeStyle = COLORS.dim;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(plotX, y);
    ctx.lineTo(plotX + plotW, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = COLORS.cursor;
    ctx.fillRect(plotX + plotW + 1, y - 7, PAD_R - 4, 14);
    ctx.fillStyle = "#0b0e14";
    ctx.fillText(fmtPx(last), plotX + plotW + 4, y);
  }

  ctx.restore();

  // title
  const meta = session.symbols[si]!;
  const idx = clamp(iCursor, 0, bars.t.length - 1);
  const lastC = bars.c[idx];
  const firstC = bars.c[0];
  ctx.fillStyle = COLORS.fg;
  ctx.font = "11px -apple-system, sans-serif";
  ctx.textBaseline = "top";
  ctx.textAlign = "left";
  let title = meta.symbol;
  if (lastC !== undefined && firstC !== undefined && firstC > 0) {
    const chg = ((lastC - firstC) / firstC) * 100;
    title += `  ${fmtPx(lastC)}  ${chg >= 0 ? "+" : ""}${chg.toFixed(2)}%`;
  }
  ctx.fillText(title, x0 + 8, y0 + 5);

  // cell clock in the focused cell
  if (si === view.focus && tC !== undefined) {
    ctx.textAlign = "right";
    ctx.fillStyle = COLORS.dim;
    ctx.fillText(fmtTime(tC, session.timezone), x0 + cw - 8, y0 + 5);
    ctx.textAlign = "left";
  }
}

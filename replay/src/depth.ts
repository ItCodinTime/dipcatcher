/**
 * Order-book depth heatmap for the focused symbol.
 *
 * x = snapshot time (trailing window), y = price. Level sizes accumulate into
 * two Float32Array grids (bids/asks) which are rasterized once into a small
 * offscreen canvas via ImageData and drawn scaled — O(levels) per snapshot,
 * independent of pane resolution.
 */

import { upperBound } from "./session.js";
import type { Session } from "./types.js";
import { COLORS, fmtPx, priceTicks, windowTimes, type Pane, type ViewState } from "./view.js";

const ROWS = 160;
const PAD_R = 46;
const PAD_T = 22;
const PAD_B = 16;
const PAD_L = 4;

let scratch: HTMLCanvasElement | null = null;
let scratchCtx: CanvasRenderingContext2D | null = null;

export function drawDepth(pane: Pane, session: Session, view: ViewState): void {
  const { ctx, w, h } = pane;
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, w, h);

  const book = session.books[view.focus];
  if (!book || book.t.length === 0) {
    ctx.fillStyle = COLORS.dim;
    ctx.font = "11px sans-serif";
    ctx.fillText("no book data", 10, 30);
    return;
  }

  const [, tHi, cursorIdx] = windowTimes(session, view);
  const tC = session.master[cursorIdx] ?? tHi;
  const nCols = Math.min(view.windowBars, 512);
  // window ends AT the cursor: the heatmap scrolls with playback
  const iHi = upperBound(book.t, tC);
  const iLo = Math.max(0, iHi - nCols);
  const cols = iHi - iLo;
  if (cols <= 0) {
    ctx.fillStyle = COLORS.dim;
    ctx.font = "11px sans-serif";
    ctx.fillText("no snapshots in window", 10, 30);
    return;
  }

  const plotX = PAD_L;
  const plotY = PAD_T;
  const plotW = w - PAD_L - PAD_R;
  const plotH = h - PAD_T - PAD_B;
  if (plotW < 20 || plotH < 20) return;

  // price extent across visible snapshots
  const d = book.depth;
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = iLo; i < iHi; i++) {
    for (let k = 0; k < d; k++) {
      const bp = book.bidPx[i * d + k]!;
      const ap = book.askPx[i * d + k]!;
      if (bp > 0 && bp < lo) lo = bp;
      if (ap > 0 && ap > hi) hi = ap;
    }
  }
  if (!isFinite(lo) || !isFinite(hi) || hi <= lo) {
    lo = 0;
    hi = 1;
  }
  const padP = (hi - lo) * 0.05 || 1e-6;
  lo -= padP;
  hi += padP;

  // accumulate into per-side grids
  const bidG = new Float32Array(cols * ROWS);
  const askG = new Float32Array(cols * ROWS);
  const midRow = new Float32Array(cols);
  let sMax = 0;
  for (let c = 0; c < cols; c++) {
    const i = iLo + c;
    let bestBid = 0;
    let bestAsk = 0;
    for (let k = 0; k < d; k++) {
      const bp = book.bidPx[i * d + k]!;
      const bs = book.bidSz[i * d + k]!;
      const ap = book.askPx[i * d + k]!;
      const asz = book.askSz[i * d + k]!;
      if (k === 0) {
        bestBid = bp;
        bestAsk = ap;
      }
      if (bp > 0 && bs > 0) {
        const r = Math.min(ROWS - 1, Math.max(0, Math.floor(((hi - bp) / (hi - lo)) * ROWS)));
        const nv = (bidG[c * ROWS + r]! += bs);
        if (nv > sMax) sMax = nv;
      }
      if (ap > 0 && asz > 0) {
        const r = Math.min(ROWS - 1, Math.max(0, Math.floor(((hi - ap) / (hi - lo)) * ROWS)));
        const nv = (askG[c * ROWS + r]! += asz);
        if (nv > sMax) sMax = nv;
      }
    }
    midRow[c] = bestBid > 0 && bestAsk > 0 ? (bestBid + bestAsk) / 2 : NaN;
  }

  // rasterize: bids = teal ramp, asks = ember ramp (premultiplied-ish additive)
  scratch ??= document.createElement("canvas");
  scratch.width = cols;
  scratch.height = ROWS;
  scratchCtx ??= scratch.getContext("2d")!;
  const img = scratchCtx.createImageData(cols, ROWS);
  const px = img.data;
  const norm = sMax > 0 ? 255 / Math.pow(sMax, 0.55) : 0;
  for (let c = 0; c < cols; c++) {
    for (let r = 0; r < ROWS; r++) {
      const b = bidG[c * ROWS + r]!;
      const a = askG[c * ROWS + r]!;
      if (b <= 0 && a <= 0) continue;
      const bi = Math.min(255, Math.pow(b, 0.55) * norm);
      const ai = Math.min(255, Math.pow(a, 0.55) * norm);
      const o = (r * cols + c) * 4;
      px[o] = Math.min(255, ai * 1.0 + bi * 0.1);
      px[o + 1] = Math.min(255, bi * 0.85 + ai * 0.45);
      px[o + 2] = Math.min(255, bi * 0.95 + ai * 0.15);
      px[o + 3] = Math.min(255, bi + ai);
    }
  }
  scratchCtx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(scratch, plotX, plotY, plotW, plotH);

  // mid-price trace
  ctx.strokeStyle = COLORS.mid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  let started = false;
  for (let c = 0; c < cols; c++) {
    const m = midRow[c]!;
    if (!isFinite(m)) continue;
    const x = plotX + ((c + 0.5) / cols) * plotW;
    const y = plotY + (1 - (m - lo) / (hi - lo)) * plotH;
    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // frame + price axis
  ctx.strokeStyle = COLORS.border;
  ctx.strokeRect(plotX + 0.5, plotY + 0.5, plotW - 1, plotH - 1);
  ctx.font = "10px ui-monospace, Menlo, monospace";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  for (const p of priceTicks(lo, hi, 6)) {
    const y = plotY + (1 - (p - lo) / (hi - lo)) * plotH;
    ctx.fillStyle = COLORS.dim;
    ctx.fillText(fmtPx(p), plotX + plotW + 4, y);
  }

  // title with focused symbol
  ctx.fillStyle = COLORS.dim;
  ctx.textBaseline = "top";
  ctx.fillText(`${session.symbols[view.focus]!.symbol} book · depth ${d} · ${cols} snaps`, PAD_L, 6);
}

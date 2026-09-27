/** Trades tape — DOM list, newest first, capped rows. Rebuilt on cursor step. */

import { upperBound, fmtTime } from "./session.js";
import type { Session } from "./types.js";
import { fmtPx, windowTimes, type ViewState } from "./view.js";

const MAX_ROWS = 64;

export function updateTape(el: HTMLElement, session: Session, view: ViewState): void {
  const tr = session.trades[view.focus]!;
  const [, , cursorIdx] = windowTimes(session, view);
  const tC = session.master[cursorIdx] ?? Infinity;
  const end = upperBound(tr.t, tC);
  const start = Math.max(0, end - MAX_ROWS);

  const parts: string[] = [
    `<div class="row head"><span>time</span><span>sym</span><span>px</span><span>qty</span><span>side</span></div>`,
  ];
  for (let i = end - 1; i >= start; i--) {
    const side = tr.side[i] === 1 ? "sell" : "buy";
    parts.push(
      `<div class="row"><span>${fmtTime(tr.t[i]!, session.timezone)}</span>` +
        `<span>${session.symbols[view.focus]!.symbol}</span>` +
        `<span>${fmtPx(tr.px[i]!)}</span>` +
        `<span>${tr.qty[i]!}</span>` +
        `<span class="side-${side}">${side}</span></div>`,
    );
  }
  el.innerHTML = parts.join("");
}

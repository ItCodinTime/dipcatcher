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

  const rows = document.createDocumentFragment();
  function row(values: string[], header = false, side?: string): void {
    const line = document.createElement("div");
    line.className = header ? "row head" : "row";
    values.forEach((value, index) => {
      const cell = document.createElement("span");
      cell.textContent = value;
      if (index === 4 && side) cell.className = `side-${side}`;
      line.append(cell);
    });
    rows.append(line);
  }
  row(["time", "sym", "px", "qty", "side"], true);
  for (let i = end - 1; i >= start; i--) {
    const side = tr.side[i] === 1 ? "sell" : "buy";
    row([
      fmtTime(tr.t[i]!, session.timezone),
      session.symbols[view.focus]!.symbol,
      fmtPx(tr.px[i]!),
      String(tr.qty[i]!),
      side,
    ], false, side);
  }
  el.replaceChildren(rows);
}

/** Drawdown + summary math over an equity (NAV) series. Pure functions. */

export interface DrawdownPoint {
  date: string;
  /** Fraction below the running peak, in [0, 1] for a positive NAV path. */
  drawdown: number;
  peak: number;
}

export interface DrawdownSummary {
  maxDrawdown: number;
  maxDrawdownDate: string | null;
  currentDrawdown: number;
  /** Dates where the path sat below its running peak. */
  underwaterFraction: number;
}

/**
 * Running drawdown series: `drawdown_t = 1 - nav_t / max(nav_<=t)`.
 * The first point seeds the peak. Non-positive NAV values are kept verbatim
 * in the ratio (they produce drawdown > 1, which the chart renders rather
 * than silently clipping — the underlying artifact decides the semantics).
 */
export function drawdownSeries(
  points: Array<{ date: string; nav: number }>,
): DrawdownPoint[] {
  let peak = Number.NEGATIVE_INFINITY;
  return points.map((p) => {
    if (p.nav > peak) peak = p.nav;
    const drawdown = peak > 0 ? 1 - p.nav / peak : 0;
    return { date: p.date, drawdown, peak };
  });
}

export function summarizeDrawdown(series: DrawdownPoint[]): DrawdownSummary {
  let maxDrawdown = 0;
  let maxDrawdownDate: string | null = null;
  let underwater = 0;
  for (const p of series) {
    if (p.drawdown > maxDrawdown) {
      maxDrawdown = p.drawdown;
      maxDrawdownDate = p.date;
    }
    if (p.drawdown > 0) underwater += 1;
  }
  return {
    maxDrawdown,
    maxDrawdownDate,
    currentDrawdown: series.length ? series[series.length - 1].drawdown : 0,
    underwaterFraction: series.length ? underwater / series.length : 0,
  };
}

/** Per-point simple returns from a NAV series (first point has no return). */
export function simpleReturns(
  points: Array<{ date: string; nav: number }>,
): Array<{ date: string; ret: number }> {
  const out: Array<{ date: string; ret: number }> = [];
  for (let i = 1; i < points.length; i += 1) {
    const prev = points[i - 1].nav;
    if (prev !== 0 && Number.isFinite(prev)) {
      out.push({ date: points[i].date, ret: points[i].nav / prev - 1 });
    }
  }
  return out;
}

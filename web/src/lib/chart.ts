/** Pure chart math: scales, ticks, decimation, SVG path building.
 *
 * Hand-rolled instead of a chart dependency — the explorer renders small
 * daily-bar series where a linear/time scale plus min-max decimation is
 * faithful and dependency-free. All functions are pure for vitest coverage.
 */

export interface XY {
  x: number;
  y: number;
}

export type Scale = (value: number) => number;

/** Linear map from domain to range; clamps nothing (callers own bounds). */
export function linearScale(
  domainMin: number,
  domainMax: number,
  rangeMin: number,
  rangeMax: number,
): Scale {
  const span = domainMax - domainMin;
  if (span === 0 || !Number.isFinite(span)) {
    return () => (rangeMin + rangeMax) / 2;
  }
  const slope = (rangeMax - rangeMin) / span;
  return (v) => rangeMin + (v - domainMin) * slope;
}

export function invertScale(
  rangeMin: number,
  rangeMax: number,
  domainMin: number,
  domainMax: number,
): Scale {
  return linearScale(rangeMin, rangeMax, domainMin, domainMax);
}

/**
 * "Nice" axis ticks: choose a 1/2/5×10^k step so `count` ticks cover the
 * domain, then emit aligned ticks inside [min, max]. Returns at least the
 * endpoints' order of magnitude ticks; empty range -> single tick.
 */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min > max) [min, max] = [max, min];
  if (min === max) return [min];

  const span = max - min;
  const rawStep = span / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(rawStep));
  let step = mag;
  for (const mult of [1, 2, 5, 10]) {
    if (rawStep <= mult * mag) {
      step = mult * mag;
      break;
    }
  }
  const ticks: number[] = [];
  const start = Math.ceil(min / step) * step;
  for (let v = start; v <= max + step * 1e-9; v += step) {
    // Guard float drift (e.g. 0.30000000000000004).
    ticks.push(Number(v.toPrecision(12)));
  }
  return ticks;
}

/**
 * Pick ~`count` evenly spaced tick positions for a date domain given as
 * epoch-ms bounds. Labels are ISO date or `YYYY-MM` when the span is long.
 */
export function dateTicks(
  minMs: number,
  maxMs: number,
  count = 6,
): Array<{ ms: number; label: string }> {
  if (!Number.isFinite(minMs) || !Number.isFinite(maxMs) || minMs > maxMs) {
    return [];
  }
  if (minMs === maxMs) {
    return [{ ms: minMs, label: isoDay(minMs) }];
  }
  const spanDays = (maxMs - minMs) / 86_400_000;
  const ticks: Array<{ ms: number; label: string }> = [];
  const n = Math.max(2, count);
  for (let i = 0; i < n; i += 1) {
    const ms = minMs + ((maxMs - minMs) * i) / (n - 1);
    ticks.push({ ms, label: spanDays > 370 ? isoMonth(ms) : isoDay(ms) });
  }
  return dedupeTicks(ticks);
}

function dedupeTicks(
  ticks: Array<{ ms: number; label: string }>,
): Array<{ ms: number; label: string }> {
  const seen = new Set<string>();
  return ticks.filter((t) => {
    if (seen.has(t.label)) return false;
    seen.add(t.label);
    return true;
  });
}

export function isoDay(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10);
}

export function isoMonth(ms: number): string {
  return new Date(ms).toISOString().slice(0, 7);
}

export function parseDay(day: string): number {
  const ms = Date.parse(`${day}T00:00:00Z`);
  if (!Number.isFinite(ms)) throw new Error(`bad ISO day: ${day}`);
  return ms;
}

/**
 * Min-max bucket decimation: fold the series into `buckets` buckets and emit
 * each bucket's min and max y (in x order) plus the global first/last points.
 * Preserves peaks and drawdown extremes better than striding, which matters
 * for honest rendering of worst-case paths.
 */
export function decimateMinMax(points: XY[], buckets: number): XY[] {
  const n = points.length;
  if (n === 0 || buckets <= 0) return [];
  if (n <= 2 || n <= buckets * 2) return points.slice();

  const out: XY[] = [points[0]];
  const xMin = points[0].x;
  const xMax = points[n - 1].x;
  const span = xMax - xMin;
  const groups: XY[][] = Array.from({ length: buckets }, () => []);
  for (const p of points) {
    const idx = Math.min(
      buckets - 1,
      Math.floor(((p.x - xMin) / span) * buckets),
    );
    groups[idx].push(p);
  }
  for (const g of groups) {
    if (g.length === 0) continue;
    let lo = g[0];
    let hi = g[0];
    for (const p of g) {
      if (p.y < lo.y) lo = p;
      if (p.y > hi.y) hi = p;
    }
    // Emit in x order within the bucket so the polyline never doubles back.
    out.push(...(lo.x <= hi.x ? [lo, hi] : [hi, lo]));
  }
  out.push(points[n - 1]);
  // De-duplicate consecutive identical x (first/last may coincide with a
  // bucket extreme).
  return out.filter(
    (p, i) => i === 0 || p.x !== out[i - 1].x || p.y !== out[i - 1].y,
  );
}

/** `M x,y L x,y ...` path. Empty input -> empty string. */
export function linePath(points: XY[], x: Scale, y: Scale): string {
  if (points.length === 0) return "";
  const parts = [`M${fmt(points[0].x, x)},${fmt(points[0].y, y)}`];
  for (let i = 1; i < points.length; i += 1) {
    parts.push(`L${fmt(points[i].x, x)},${fmt(points[i].y, y)}`);
  }
  return parts.join(" ");
}

/** Filled area under a line, closed against `baseline` (in y units). */
export function areaPath(
  points: XY[],
  x: Scale,
  y: Scale,
  baseline: number,
): string {
  if (points.length === 0) return "";
  const base = fmt(baseline, y);
  const first = fmt(points[0].x, x);
  const last = fmt(points[points.length - 1].x, x);
  return `${linePath(points, x, y)} L${last},${base} L${first},${base} Z`;
}

/** Index of the point nearest `xVal` (binary search on sorted x). */
export function nearestIndex(points: XY[], xVal: number): number {
  if (points.length === 0) return -1;
  let lo = 0;
  let hi = points.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (points[mid].x < xVal) lo = mid;
    else hi = mid;
  }
  return Math.abs(points[lo].x - xVal) <= Math.abs(points[hi].x - xVal)
    ? lo
    : hi;
}

function fmt(v: number, scale: Scale): string {
  return scale(v).toFixed(2);
}

export function extent(values: number[]): { min: number; max: number } {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const v of values) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min)) return { min: 0, max: 1 };
  return { min, max };
}

/** Pad a domain by `frac` on each side; keeps [0, x] anchored when min >= 0. */
export function paddedExtent(
  values: number[],
  frac = 0.05,
  anchorZero = false,
): { min: number; max: number } {
  const { min, max } = extent(values);
  const pad = (max - min) * frac || Math.abs(max) * frac || 1;
  let lo = min - pad;
  const hi = max + pad;
  if (anchorZero && min >= 0) lo = 0;
  return { min: lo, max: hi };
}

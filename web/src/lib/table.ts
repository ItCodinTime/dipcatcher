/** Table shaping: segment-stat matrices and nested-dict flattening. */

import type { SegmentStats } from "./types";

/** Preferred row order for known metrics; unknown keys sort after, alpha. */
const METRIC_ORDER = [
  "n",
  "bars",
  "events",
  "net_return",
  "total_return",
  "cagr",
  "sharpe",
  "max_drawdown",
  "mean_turnover",
  "funding_net",
  "liquidation_count",
  "margin_rejects",
  "risk_gate_rejects",
  "funding_events_dropped",
];

/** Union of metric keys across segments, ordered for display. */
export function segmentMetricRows(
  segments: Record<string, SegmentStats>,
): string[] {
  const keys = new Set<string>();
  for (const stats of Object.values(segments)) {
    for (const k of Object.keys(stats)) keys.add(k);
  }
  return [...keys].sort((a, b) => {
    const ia = METRIC_ORDER.indexOf(a);
    const ib = METRIC_ORDER.indexOf(b);
    if (ia !== -1 || ib !== -1) {
      if (ia === -1) return 1;
      if (ib === -1) return -1;
      return ia - ib;
    }
    return a.localeCompare(b);
  });
}

export interface FlatRow {
  key: string;
  value: string;
}

/**
 * Flatten a nested dict (e.g. strategy `extras`) into dotted-key rows.
 * Numbers keep numeric formatting hints via `numeric` flag on the key.
 */
export function flattenDict(
  value: unknown,
  prefix = "",
  depth = 0,
): FlatRow[] {
  if (depth > 4 || value === null || typeof value !== "object") {
    return [];
  }
  const rows: FlatRow[] = [];
  const items = Array.isArray(value)
    ? value.map((v, i) => [`[${i}]`, v] as const)
    : Object.entries(value as Record<string, unknown>);
  for (const [k, v] of items) {
    const key = prefix ? `${prefix}.${k}` : String(k);
    if (v !== null && typeof v === "object") {
      rows.push(...flattenDict(v, key, depth + 1));
    } else {
      rows.push({ key, value: scalarToString(v) });
    }
  }
  return rows;
}

function scalarToString(v: unknown): string {
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return "non-finite";
    if (Number.isInteger(v)) return v.toLocaleString("en-US");
    return Math.abs(v) < 1 && v !== 0 ? v.toPrecision(4) : v.toFixed(4);
  }
  if (typeof v === "boolean") return v ? "true" : "false";
  if (v === null || v === undefined) return "—";
  return String(v);
}

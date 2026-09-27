/** Display formatting helpers. */

export function fmtPct(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return v.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function fmtCompact(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  const abs = Math.abs(v);
  if (abs >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  if (abs >= 1) return v.toFixed(2);
  return v.toPrecision(3);
}

export function fmtUSD(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return `$${fmtCompact(v)}`;
}

/** Adaptive metric formatting: ratios look like percents, counts like ints. */
export function fmtMetric(key: string, v: number): string {
  if (!Number.isFinite(v)) return "—";
  const k = key.toLowerCase();
  if (
    k.includes("return") ||
    k.includes("drawdown") ||
    k.includes("fraction") ||
    k.includes("rate") ||
    k.endsWith("_pct") ||
    k === "cagr" ||
    k.includes("cagr")
  ) {
    return fmtPct(v);
  }
  if (k === "n" || k.startsWith("n_") || k.endsWith("_count") || k === "bars" || k === "events") {
    return Math.round(v).toLocaleString("en-US");
  }
  if (Math.abs(v) >= 1000) return fmtCompact(v);
  if (Number.isInteger(v)) return v.toString();
  return v.toPrecision(4);
}

export function shortenHash(digest: string, keep = 12): string {
  if (digest.length <= keep * 2 + 3) return digest;
  return `${digest.slice(0, keep)}…${digest.slice(-keep)}`;
}

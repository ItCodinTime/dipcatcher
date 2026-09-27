/** Fixture loading + minimal shape validation.
 *
 * Fixtures are generated artifacts (see `web/scripts/export_fixtures.py`), so
 * validation is intentionally structural, not exhaustive: required keys with
 * the right types, tolerant of additional fields.
 */

import type {
  EquityFixture,
  EquityPoint,
  FixtureIndex,
  ReceiptIndexEntry,
  StrategyFixture,
  StrategyIndexEntry,
} from "./types";

export async function fetchJson(path: string): Promise<unknown> {
  const res = await fetch(path, { cache: "no-cache" });
  if (!res.ok) {
    throw new Error(`fetch ${path}: HTTP ${res.status}`);
  }
  return res.json();
}

export async function fetchBytes(path: string): Promise<ArrayBuffer> {
  const res = await fetch(path, { cache: "no-cache" });
  if (!res.ok) {
    throw new Error(`fetch ${path}: HTTP ${res.status}`);
  }
  return res.arrayBuffer();
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function requireStr(v: unknown, what: string): string {
  if (typeof v !== "string") throw new Error(`index: ${what} must be a string`);
  return v;
}

export function validateIndex(raw: unknown): FixtureIndex {
  if (!isRecord(raw)) throw new Error("index: not an object");
  const strategies = raw.strategies;
  const receipts = raw.receipts;
  if (!Array.isArray(strategies)) throw new Error("index: strategies missing");
  if (!Array.isArray(receipts)) throw new Error("index: receipts missing");

  const strategyEntries: StrategyIndexEntry[] = strategies.map((s, i) => {
    if (!isRecord(s)) throw new Error(`index: strategies[${i}] not object`);
    return {
      id: requireStr(s.id, `strategies[${i}].id`),
      name: requireStr(s.name, `strategies[${i}].name`),
      kind: typeof s.kind === "string" ? s.kind : "unknown",
      data_source: typeof s.data_source === "string" ? s.data_source : "unknown",
      stats_file: requireStr(s.stats_file, `strategies[${i}].stats_file`),
      equity_file: typeof s.equity_file === "string" ? s.equity_file : null,
      has_equity: s.has_equity === true,
      segments: Array.isArray(s.segments)
        ? s.segments.filter((x): x is string => typeof x === "string")
        : [],
    };
  });

  const receiptEntries: ReceiptIndexEntry[] = receipts.map((r, i) => {
    if (!isRecord(r)) throw new Error(`index: receipts[${i}] not object`);
    return {
      id: requireStr(r.id, `receipts[${i}].id`),
      file: requireStr(r.file, `receipts[${i}].file`),
      schema: typeof r.schema === "string" ? r.schema : null,
      evidence_level: typeof r.evidence_level === "string" ? r.evidence_level : null,
      research_only: typeof r.research_only === "boolean" ? r.research_only : null,
      live_pnl_claim:
        typeof r.live_pnl_claim === "boolean" ? r.live_pnl_claim : null,
      created_at: typeof r.created_at === "string" ? r.created_at : null,
      n_hashes: typeof r.n_hashes === "number" ? r.n_hashes : 0,
      n_hash_matches:
        typeof r.n_hash_matches === "number" ? r.n_hash_matches : 0,
    };
  });

  const hashMatches: Record<string, string> = {};
  if (isRecord(raw.hash_matches)) {
    for (const [k, v] of Object.entries(raw.hash_matches)) {
      if (typeof v === "string") hashMatches[k] = v;
    }
  }

  const honesty = isRecord(raw.honesty) ? raw.honesty : {};
  return {
    generated_at: typeof raw.generated_at === "string" ? raw.generated_at : "",
    generator: typeof raw.generator === "string" ? raw.generator : "unknown",
    repo_revision:
      typeof raw.repo_revision === "string" ? raw.repo_revision : null,
    honesty: {
      research_only: honesty.research_only === true,
      live_pnl_claim: honesty.live_pnl_claim === true,
      note: typeof honesty.note === "string" ? honesty.note : "",
    },
    strategies: strategyEntries,
    receipts: receiptEntries,
    hash_matches: hashMatches,
  };
}

/** Decode the columnar equity fixture into point objects. */
export function decodeEquity(raw: unknown): EquityPoint[] {
  if (!isRecord(raw)) throw new Error("equity: not an object");
  const fixture = raw as unknown as EquityFixture;
  if (!Array.isArray(fixture.points)) {
    throw new Error("equity: points missing");
  }
  return fixture.points.map((row) => {
    if (!Array.isArray(row) || row.length < 5) {
      throw new Error("equity: malformed point row");
    }
    const [date, nav, gross, net, turnover] = row;
    if (typeof date !== "string" || typeof nav !== "number") {
      throw new Error("equity: malformed point types");
    }
    return { date, nav, gross, net, turnover };
  });
}

export function validateStrategy(raw: unknown): StrategyFixture {
  if (!isRecord(raw)) throw new Error("strategy: not an object");
  const segments = raw.segments;
  if (!isRecord(segments)) throw new Error("strategy: segments missing");
  return raw as unknown as StrategyFixture;
}

/** Fixture paths are relative to the app's base so `dist/` mounts anywhere. */
export function fixturePath(rel: string): string {
  const base = import.meta.env.BASE_URL ?? "./";
  const prefix = base.endsWith("/") ? base : `${base}/`;
  return `${prefix}fixtures/${rel.replace(/^\//, "")}`;
}

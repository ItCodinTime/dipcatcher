/** Shared types for the research explorer fixtures.
 *
 * The fixture shapes mirror `web/scripts/export_fixtures.py`; that script is
 * the single place where receipt/artifact internals get normalized.
 */

export interface IndexHonesty {
  research_only: boolean;
  live_pnl_claim: boolean;
  note: string;
}

export interface StrategyIndexEntry {
  id: string;
  name: string;
  kind: "equity_backed" | "receipt_stats" | string;
  data_source: string;
  stats_file: string;
  equity_file: string | null;
  has_equity: boolean;
  segments: string[];
}

export interface ReceiptIndexEntry {
  id: string;
  file: string;
  schema: string | null;
  evidence_level: string | null;
  research_only: boolean | null;
  live_pnl_claim: boolean | null;
  created_at: string | null;
  n_hashes: number;
  n_hash_matches: number;
}

export interface FixtureIndex {
  generated_at: string;
  generator: string;
  repo_revision: string | null;
  honesty: IndexHonesty;
  strategies: StrategyIndexEntry[];
  receipts: ReceiptIndexEntry[];
  /** sha256 digest -> repo path that hash resolves to (export-time lookup). */
  hash_matches: Record<string, string>;
}

/** Equity fixture emitted by export_fixtures.py (`equity/<id>.json`). */
export interface EquityFixture {
  fields: ["date", "nav", "gross", "net", "turnover"];
  points: Array<[string, number, number, number, number]>;
  rows: number;
  source_file: string;
  units: Record<string, string>;
}

export interface SegmentStats {
  [metric: string]: number;
}

export interface StrategyFixture {
  id: string;
  name: string;
  kind: string;
  data_source: string;
  research_only: boolean;
  live_pnl_claim: boolean;
  provenance: {
    source_files?: string[];
    receipt?: string;
    pointer?: string;
    note?: string;
    equity?: {
      rows: number;
      first_date: string;
      last_date: string;
      nav_final: number;
    };
  };
  /** segment name -> {metric: value}; metrics are as stored in the source. */
  segments: Record<string, SegmentStats>;
  extras: Record<string, unknown>;
  equity_file: string | null;
}

/** A single row in an equity/drawdown chart, decoded for rendering. */
export interface EquityPoint {
  date: string;
  nav: number;
  gross: number;
  net: number;
  turnover: number;
}

/**
 * Session format types — mirrors dipcatcher schemas:
 *   bars    ~ quant_fund.schemas.market.Bar        (event_time/open/high/low/close/volume)
 *   books   ~ quant_fund.schemas.order_book.OrderBookSnapshot (bids/asks as level arrays)
 *   trades  ~ Fill-like tape prints                 (event_time/price/quantity/side)
 *   markers ~ Order-like strategy decisions         (side/quantity/limit_price/decision_time)
 *
 * The wire format stores columnar arrays (numbers, epoch-ms timestamps) to keep
 * the fixture small and to decode straight into typed arrays for rendering.
 */

export interface SymbolJson {
  security_id: string;
  symbol: string;
  name: string;
  exchange: string;
  currency: string;
  sector: string;
}

export interface BarsJson {
  /** epoch ms, minute CLOSE (hf_ohlcv_1m convention) */
  event_time: number[];
  open: number[];
  high: number[];
  low: number[];
  close: number[];
  volume: number[];
}

export interface BooksJson {
  /** epoch ms per snapshot, aligned to bar event_times */
  event_time: number[];
  /** declared level depth per side */
  depth: number;
  /** row-major [snapshot][level], bids best→worse (descending price) */
  bid_price: number[][];
  bid_size: number[][];
  /** row-major [snapshot][level], asks best→worse (ascending price) */
  ask_price: number[][];
  ask_size: number[][];
}

export interface TradesJson {
  event_time: number[];
  price: number[];
  quantity: number[];
  /** "buy" | "sell" — aggressor side */
  side: string[];
}

export interface MarkerJson {
  symbol: string;
  /** epoch ms — the decision time the marker is drawn at */
  event_time: number;
  side: "buy" | "sell";
  kind: string;
  quantity: number;
  limit_price: number;
  decision_time: number;
  order_time: number;
  note: string;
}

export interface SessionJson {
  format: string;
  format_version: number;
  source: string;
  session_date: string;
  timezone: string;
  bar_interval_seconds: number;
  generated_by: string;
  symbols: SymbolJson[];
  bars: Record<string, BarsJson>;
  books: Record<string, BooksJson>;
  trades: Record<string, TradesJson>;
  markers: MarkerJson[];
}

/* ---------- decoded, typed-array model ---------- */

export interface SymbolMeta extends SymbolJson {
  index: number;
}

export interface SymbolBars {
  t: Float64Array;
  o: Float32Array;
  h: Float32Array;
  l: Float32Array;
  c: Float32Array;
  v: Float32Array;
}

export interface SymbolBook {
  t: Float64Array;
  depth: number;
  /** flat, length n*depth */
  bidPx: Float32Array;
  bidSz: Float32Array;
  askPx: Float32Array;
  askSz: Float32Array;
}

export interface SymbolTrades {
  t: Float64Array;
  px: Float32Array;
  qty: Float32Array;
  /** 0 = buy aggressor, 1 = sell */
  side: Uint8Array;
}

export interface Marker {
  symbolIndex: number;
  t: number;
  side: number; // 0 buy, 1 sell
  kind: string;
  quantity: number;
  limitPrice: number;
  note: string;
  /** index into the symbol's bar array (first bar with t >= marker t) */
  barIndex: number;
}

export interface Session {
  source: string;
  sessionDate: string;
  timezone: string;
  barIntervalSec: number;
  symbols: SymbolMeta[];
  bars: SymbolBars[];
  books: (SymbolBook | null)[];
  trades: SymbolTrades[];
  markers: Marker[];
  /** sorted union of all bar event_times — the scrubbing index space */
  master: Float64Array;
  /** markers grouped per symbol index */
  markersBySymbol: Marker[][];
}

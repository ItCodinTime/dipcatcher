/** Load + decode a SessionJson into typed arrays. */

import type {
  BarsJson,
  BooksJson,
  Marker,
  Session,
  SessionJson,
  SymbolBars,
  SymbolBook,
  SymbolTrades,
  TradesJson,
} from "./types.js";

/** First index i in sorted `arr` with arr[i] >= x (lower bound). */
export function lowerBound(arr: Float64Array, x: number, lo = 0, hi = arr.length): number {
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if ((arr[mid] as number) < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** First index i in sorted `arr` with arr[i] > x (upper bound). */
export function upperBound(arr: Float64Array, x: number, lo = 0, hi = arr.length): number {
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if ((arr[mid] as number) <= x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function f32(a: number[]): Float32Array {
  return Float32Array.from(a);
}

function f64(a: number[]): Float64Array {
  return Float64Array.from(a);
}

function decodeBars(j: BarsJson): SymbolBars {
  return { t: f64(j.event_time), o: f32(j.open), h: f32(j.high), l: f32(j.low), c: f32(j.close), v: f32(j.volume) };
}

function decodeBook(j: BooksJson): SymbolBook {
  const n = j.event_time.length;
  const d = j.depth;
  const bidPx = new Float32Array(n * d);
  const bidSz = new Float32Array(n * d);
  const askPx = new Float32Array(n * d);
  const askSz = new Float32Array(n * d);
  for (let i = 0; i < n; i++) {
    const bp = j.bid_price[i] ?? [];
    const bs = j.bid_size[i] ?? [];
    const ap = j.ask_price[i] ?? [];
    const asz = j.ask_size[i] ?? [];
    for (let k = 0; k < d; k++) {
      bidPx[i * d + k] = bp[k] ?? 0;
      bidSz[i * d + k] = bs[k] ?? 0;
      askPx[i * d + k] = ap[k] ?? 0;
      askSz[i * d + k] = asz[k] ?? 0;
    }
  }
  return { t: f64(j.event_time), depth: d, bidPx, bidSz, askPx, askSz };
}

function decodeTrades(j: TradesJson): SymbolTrades {
  const side = new Uint8Array(j.side.length);
  for (let i = 0; i < j.side.length; i++) side[i] = j.side[i] === "sell" ? 1 : 0;
  return { t: f64(j.event_time), px: f32(j.price), qty: f32(j.quantity), side };
}

export function decodeSession(json: SessionJson): Session {
  const symIndex = new Map<string, number>();
  const symbols = json.symbols.map((s, i) => {
    symIndex.set(s.symbol, i);
    return { ...s, index: i };
  });

  const bars: SymbolBars[] = [];
  const books: (SymbolBook | null)[] = [];
  const trades: SymbolTrades[] = [];
  const markersBySymbol: Marker[][] = symbols.map(() => []);

  for (const s of symbols) {
    const bj = json.bars[s.symbol];
    if (!bj) throw new Error(`session missing bars for ${s.symbol}`);
    bars.push(decodeBars(bj));
    const kj = json.books[s.symbol];
    books.push(kj ? decodeBook(kj) : null);
    const tj = json.trades[s.symbol];
    trades.push(tj ? decodeTrades(tj) : { t: new Float64Array(0), px: new Float32Array(0), qty: new Float32Array(0), side: new Uint8Array(0) });
  }

  // Master timeline: sorted union of all bar event_times.
  const masterSet = new Set<number>();
  for (const b of bars) for (const t of b.t) masterSet.add(t);
  const master = Float64Array.from([...masterSet].sort((a, b) => a - b));

  const markers: Marker[] = [];
  for (const m of json.markers) {
    const si = symIndex.get(m.symbol);
    if (si === undefined) continue;
    const barIndex = lowerBound(bars[si]!.t, m.event_time);
    const mk: Marker = {
      symbolIndex: si,
      t: m.event_time,
      side: m.side === "sell" ? 1 : 0,
      kind: m.kind,
      quantity: m.quantity,
      limitPrice: m.limit_price,
      note: m.note,
      barIndex,
    };
    markers.push(mk);
    markersBySymbol[si]!.push(mk);
  }

  return {
    source: json.source,
    sessionDate: json.session_date,
    timezone: json.timezone,
    barIntervalSec: json.bar_interval_seconds,
    symbols,
    bars,
    books,
    trades,
    markers,
    master,
    markersBySymbol,
  };
}

export async function loadSession(url: string): Promise<Session> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
  const json = (await res.json()) as SessionJson;
  if (json.format !== "dipcatcher.replay.session") {
    throw new Error(`unexpected format ${String(json.format)}`);
  }
  return decodeSession(json);
}

/** Format an epoch-ms instant as HH:MM:SS in the session timezone. */
export function fmtTime(t: number, tz: string): string {
  try {
    return new Intl.DateTimeFormat("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      timeZone: tz,
    }).format(new Date(t));
  } catch {
    return new Date(t).toISOString().slice(11, 19);
  }
}

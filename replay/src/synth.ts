/**
 * Deterministic in-browser SYNTHETIC session generator.
 *
 * Used by benchmark mode (?bench=1) so we can measure rendering on a full
 * synthetic day (e.g. 390 bars x 8 symbols) without shipping a large fixture.
 * Mirrors replay/scripts/gen_fixture.py — both emit the same session format
 * and are labeled SYNTHETIC; not market evidence.
 */

import type { BarsJson, BooksJson, MarkerJson, SessionJson, TradesJson } from "./types.js";

/** mulberry32 — small deterministic PRNG. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rng: () => number): number {
  // Box–Muller
  const u = Math.max(rng(), 1e-12);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const NAMES = ["ALFA", "BRAV", "CHAR", "DELT", "ECHO", "FOXT", "GOLF", "HOTL"];

export function makeSyntheticSession(nSymbols = 8, nBars = 390, seed = 7, depth = 5): SessionJson {
  const rng = mulberry32(seed);
  const day = Date.UTC(2025, 0, 6); // 2025-01-06 (Mon)
  // 09:30 America/New_York == 14:30 UTC (EST). Bars timestamped at minute CLOSE.
  const openMs = day + 14 * 3600_000 + 30 * 60_000;
  const times: number[] = [];
  for (let i = 0; i < nBars; i++) times.push(openMs + (i + 1) * 60_000);

  // shared market factor for mild cross-symbol correlation
  const mkt: number[] = new Array<number>(nBars);
  let mk = 0;
  for (let i = 0; i < nBars; i++) {
    mk = mk * 0.9 + gauss(rng) * 0.0006;
    mkt[i] = mk;
  }

  const symbols = [];
  const bars: Record<string, BarsJson> = {};
  const books: Record<string, BooksJson> = {};
  const trades: Record<string, TradesJson> = {};
  const markers: MarkerJson[] = [];

  for (let s = 0; s < nSymbols; s++) {
    const sym = NAMES[s % NAMES.length]!;
    symbols.push({
      security_id: `SYNTH-${sym}`,
      symbol: sym,
      name: `Synthetic ${sym}`,
      exchange: "SYNTH",
      currency: "USD",
      sector: "Synthetic",
    });

    let px = 20 + rng() * 380;
    const o = new Array<number>(nBars);
    const h = new Array<number>(nBars);
    const l = new Array<number>(nBars);
    const c = new Array<number>(nBars);
    const v = new Array<number>(nBars);
    const vol = 0.0008 + rng() * 0.0012;

    for (let i = 0; i < nBars; i++) {
      const openP = px;
      const r = mkt[i]! * (0.4 + rng() * 0.3) + vol * gauss(rng);
      let hi = openP;
      let lo = openP;
      // 4 intra-bar steps -> plausible high/low path
      let p = openP;
      for (let k = 0; k < 4; k++) {
        p *= 1 + r / 4 + vol * 0.5 * gauss(rng);
        if (p > hi) hi = p;
        if (p < lo) lo = p;
      }
      const closeP = p;
      // U-shaped intraday volume profile
      const u = i / Math.max(1, nBars - 1);
      const profile = 0.6 + 2.4 * (Math.pow(u - 0.5, 2) * 4) / 4 + 0.9 * Math.exp(-u * 8) + 0.7 * Math.exp(-(1 - u) * 8);
      v[i] = Math.round(2000 * profile * (0.5 + rng()) * (1 + Math.abs(r) * 300));
      o[i] = +openP.toFixed(4);
      h[i] = +hi.toFixed(4);
      l[i] = +lo.toFixed(4);
      c[i] = +closeP.toFixed(4);
      px = closeP;
    }
    bars[sym] = { event_time: times, open: o, high: h, low: l, close: c, volume: v };

    // L2 books: 5 levels each side, synthetic spread ~ few bps of price
    const bid_price: number[][] = new Array<number[]>(nBars);
    const bid_size: number[][] = new Array<number[]>(nBars);
    const ask_price: number[][] = new Array<number[]>(nBars);
    const ask_size: number[][] = new Array<number[]>(nBars);
    const bt: number[] = new Array<number>(nBars);
    for (let i = 0; i < nBars; i++) {
      const mid = c[i]!;
      const spread = Math.max(0.01, mid * (0.0002 + rng() * 0.0006));
      const bb = mid - spread / 2;
      const ba = mid + spread / 2;
      const tick = mid >= 100 ? 0.05 : mid >= 10 ? 0.01 : 0.001;
      const bp: number[] = new Array<number>(depth);
      const bs: number[] = new Array<number>(depth);
      const ap: number[] = new Array<number>(depth);
      const asz: number[] = new Array<number>(depth);
      const imb = 0.7 + rng() * 0.6; // bid/ask imbalance
      let off = 0;
      for (let k = 0; k < depth; k++) {
        off += tick * (1 + Math.floor(rng() * 2));
        bp[k] = +(k === 0 ? bb : bb - off).toFixed(4);
        ap[k] = +(k === 0 ? ba : ba + off).toFixed(4);
        bs[k] = Math.round((50 + rng() * 900) * imb * (1 + k * 0.3));
        asz[k] = Math.round((50 + rng() * 900) / imb * (1 + k * 0.3));
      }
      bid_price[i] = bp;
      bid_size[i] = bs;
      ask_price[i] = ap;
      ask_size[i] = asz;
      bt[i] = times[i]!;
    }
    books[sym] = { event_time: bt, depth, bid_price, bid_size, ask_price, ask_size };

    // tape: ~8 prints per bar, priced inside [low, high]
    const tt: number[] = [];
    const tp: number[] = [];
    const tq: number[] = [];
    const ts: string[] = [];
    for (let i = 0; i < nBars; i++) {
      const n = 4 + Math.floor(rng() * 9);
      const up = c[i]! >= o[i]!;
      for (let k = 0; k < n; k++) {
        const frac = rng();
        tt.push(Math.round(times[i]! - 60_000 + frac * 60_000));
        tp.push(+(l[i]! + (h[i]! - l[i]!) * rng()).toFixed(4));
        tq.push(Math.round(10 + rng() * 500));
        ts.push(rng() < (up ? 0.62 : 0.38) ? "buy" : "sell");
      }
    }
    // keep tape time-ordered
    const order = tt.map((_, i) => i).sort((a, b) => tt[a]! - tt[b]!);
    trades[sym] = {
      event_time: order.map((i) => tt[i]!),
      price: order.map((i) => tp[i]!),
      quantity: order.map((i) => tq[i]!),
      side: order.map((i) => ts[i]!),
    };

    // markers: deterministic SYNTHETIC momentum-ish decisions every ~47 bars,
    // on a rotating subset of symbols
    if (s < 4) {
      for (let i = 20 + s * 9; i < nBars - 20; i += 47) {
        const buy = ((i / 47) | 0) % 2 === 0;
        markers.push({
          symbol: sym,
          event_time: times[i]!,
          side: buy ? "buy" : "sell",
          kind: buy ? "entry" : "exit",
          quantity: 100 + s * 25,
          limit_price: c[i]!,
          decision_time: times[i]!,
          order_time: times[i]! + 60_000,
          note: "SYNTHETIC strategy marker",
        });
      }
    }
  }

  return {
    format: "dipcatcher.replay.session",
    format_version: 1,
    source: "synthetic",
    session_date: "2025-01-06",
    timezone: "America/New_York",
    bar_interval_seconds: 60,
    generated_by: "replay/src/synth.ts (SYNTHETIC)",
    symbols,
    bars,
    books,
    trades,
    markers,
  };
}

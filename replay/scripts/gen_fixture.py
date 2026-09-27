#!/usr/bin/env python3
"""Generate a SYNTHETIC replay session fixture for the market replay viewer.

Labeled SYNTHETIC: random-walk prices, derived L2 depth, and toy strategy
markers are correctness fixtures only — not market evidence. Field names
mirror the repo schemas:

  bars    ~ quant_fund.schemas.market.Bar          (event_time is the minute
            CLOSE, matching the hf_ohlcv_1m adapter convention)
  books   ~ quant_fund.schemas.order_book.OrderBookSnapshot (bids best→worse
            descending, asks best→worse ascending, depth levels per side)
  trades  ~ tape prints (event_time/price/quantity/side)
  markers ~ quant_fund.schemas.orders.Order-like decisions (side/quantity/
            limit_price/decision_time/order_time)

Usage:
    python3 replay/scripts/gen_fixture.py [--out replay/public/fixtures/session.synthetic.json]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

NAMES = ["ALFA", "BRAV", "CHAR", "DELT", "ECHO", "FOXT", "GOLF", "HOTL"]

# 2025-01-06 (Mon), regular session 09:30–16:00 America/New_York (EST = UTC-5).
SESSION_DATE = "2025-01-06"
OPEN_UTC_MS = int(datetime(2025, 1, 6, 14, 30, tzinfo=UTC).timestamp() * 1000)


def _round(x: float, nd: int = 4) -> float:
    return round(x, nd)


def gen_session(
    n_symbols: int = 8, n_bars: int = 390, seed: int = 7, depth: int = 5
) -> dict[str, Any]:
    """Build the session dict. Deterministic for a given seed."""
    rng = random.Random(seed)
    times = [OPEN_UTC_MS + (i + 1) * 60_000 for i in range(n_bars)]

    # shared market factor -> mild cross-symbol correlation
    mkt: list[float] = []
    mk = 0.0
    for _ in range(n_bars):
        mk = mk * 0.9 + rng.gauss(0.0, 0.0006)
        mkt.append(mk)

    symbols: list[dict[str, str]] = []
    bars: dict[str, Any] = {}
    books: dict[str, Any] = {}
    trades: dict[str, Any] = {}
    markers: list[dict[str, Any]] = []

    for s in range(n_symbols):
        sym = NAMES[s % len(NAMES)]
        symbols.append(
            {
                "security_id": f"SYNTH-{sym}",
                "symbol": sym,
                "name": f"Synthetic {sym}",
                "exchange": "SYNTH",
                "currency": "USD",
                "sector": "Synthetic",
            }
        )

        px = 20.0 + rng.random() * 380.0
        vol = 0.0008 + rng.random() * 0.0012
        o: list[float] = []
        h: list[float] = []
        low: list[float] = []
        c: list[float] = []
        v: list[int] = []
        for i in range(n_bars):
            open_p = px
            r = mkt[i] * (0.4 + rng.random() * 0.3) + vol * rng.gauss(0.0, 1.0)
            hi = open_p
            lo = open_p
            p = open_p
            for _ in range(4):
                p *= 1 + r / 4 + vol * 0.5 * rng.gauss(0.0, 1.0)
                hi = max(hi, p)
                lo = min(lo, p)
            u = i / max(1, n_bars - 1)
            profile = (
                0.6
                + 0.6 * (u - 0.5) ** 2 * 4
                + 0.9 * math.exp(-u * 8)
                + 0.7 * math.exp(-(1 - u) * 8)
            )
            v.append(round(2000 * profile * (0.5 + rng.random()) * (1 + abs(r) * 300)))
            o.append(_round(open_p))
            h.append(_round(hi))
            low.append(_round(lo))
            c.append(_round(p))
            px = p
        bars[sym] = {
            "event_time": times,
            "open": o,
            "high": h,
            "low": low,
            "close": c,
            "volume": v,
        }

        bid_price: list[list[float]] = []
        bid_size: list[list[int]] = []
        ask_price: list[list[float]] = []
        ask_size: list[list[int]] = []
        for i in range(n_bars):
            mid = c[i]
            spread = max(0.01, mid * (0.0002 + rng.random() * 0.0006))
            bb = mid - spread / 2
            ba = mid + spread / 2
            tick = 0.05 if mid >= 100 else 0.01 if mid >= 10 else 0.001
            imb = 0.7 + rng.random() * 0.6
            bp_row: list[float] = []
            ap_row: list[float] = []
            off = 0.0
            for k in range(depth):
                off += tick * (1 + rng.randrange(2))
                bp_row.append(_round(bb - off if k else bb))
                ap_row.append(_round(ba + off if k else ba))
            bid_price.append(bp_row)
            bid_size.append(
                [round((50 + rng.random() * 900) * imb * (1 + k * 0.3)) for k in range(depth)]
            )
            ask_price.append(ap_row)
            ask_size.append(
                [round((50 + rng.random() * 900) / imb * (1 + k * 0.3)) for k in range(depth)]
            )
        books[sym] = {
            "event_time": times,
            "depth": depth,
            "bid_price": bid_price,
            "bid_size": bid_size,
            "ask_price": ask_price,
            "ask_size": ask_size,
        }

        tt: list[int] = []
        tp: list[float] = []
        tq: list[int] = []
        ts: list[str] = []
        for i in range(n_bars):
            n = 4 + rng.randrange(9)
            up = c[i] >= o[i]
            for _ in range(n):
                frac = rng.random()
                tt.append(round(times[i] - 60_000 + frac * 60_000))
                tp.append(_round(low[i] + (h[i] - low[i]) * rng.random()))
                tq.append(round(10 + rng.random() * 500))
                ts.append("buy" if rng.random() < (0.62 if up else 0.38) else "sell")
        order = sorted(range(len(tt)), key=tt.__getitem__)
        trades[sym] = {
            "event_time": [tt[i] for i in order],
            "price": [tp[i] for i in order],
            "quantity": [tq[i] for i in order],
            "side": [ts[i] for i in order],
        }

        # SYNTHETIC strategy markers: momentum-ish alternating entries/exits
        if s < 4:
            for i in range(20 + s * 9, n_bars - 20, 47):
                buy = (i // 47) % 2 == 0
                markers.append(
                    {
                        "symbol": sym,
                        "event_time": times[i],
                        "side": "buy" if buy else "sell",
                        "kind": "entry" if buy else "exit",
                        "quantity": 100 + s * 25,
                        "limit_price": c[i],
                        "decision_time": times[i],
                        "order_time": times[i] + 60_000,
                        "note": "SYNTHETIC strategy marker",
                    }
                )

    return {
        "format": "dipcatcher.replay.session",
        "format_version": 1,
        "source": "synthetic",
        "session_date": SESSION_DATE,
        "timezone": "America/New_York",
        "bar_interval_seconds": 60,
        "generated_by": "replay/scripts/gen_fixture.py (SYNTHETIC)",
        "symbols": symbols,
        "bars": bars,
        "books": books,
        "trades": trades,
        "markers": markers,
    }


def validate_session(session: dict[str, Any]) -> list[str]:
    """Structural checks on a session dict; returns a list of problems."""
    problems: list[str] = []
    if session.get("format") != "dipcatcher.replay.session":
        problems.append("bad format tag")
    symbols = session.get("symbols", [])
    syms = [s["symbol"] for s in symbols]
    bars = session.get("bars", {})
    for sym in syms:
        b = bars.get(sym)
        if b is None:
            problems.append(f"missing bars for {sym}")
            continue
        n = len(b["event_time"])
        for k in ("open", "high", "low", "close", "volume"):
            if len(b[k]) != n:
                problems.append(f"{sym}.bars.{k} length != event_time")
        if b["event_time"] != sorted(b["event_time"]):
            problems.append(f"{sym}.bars.event_time not sorted")
        for i in range(n):
            if not (
                b["low"][i]
                <= min(b["open"][i], b["close"][i])
                <= max(b["open"][i], b["close"][i])
                <= b["high"][i]
            ):
                problems.append(f"{sym}.bars[{i}] OHLC inconsistent")
                break
        book = session.get("books", {}).get(sym)
        if book is not None:
            d = book["depth"]
            if len(book["event_time"]) != len(book["bid_price"]):
                problems.append(f"{sym}.books length mismatch")
            for i, (bp, ap) in enumerate(zip(book["bid_price"], book["ask_price"], strict=True)):
                if bp != sorted(bp, reverse=True):
                    problems.append(f"{sym}.books[{i}] bids not descending")
                    break
                if ap != sorted(ap):
                    problems.append(f"{sym}.books[{i}] asks not ascending")
                    break
                if bp[0] >= ap[0]:
                    problems.append(f"{sym}.books[{i}] crossed book")
                    break
            if any(len(row) > d for row in book["bid_price"]):
                problems.append(f"{sym}.books side length exceeds depth")
        tr = session.get("trades", {}).get(sym)
        if tr is not None and tr["event_time"] != sorted(tr["event_time"]):
            problems.append(f"{sym}.trades.event_time not sorted")
    for m in session.get("markers", []):
        if m["symbol"] not in syms:
            problems.append(f"marker for unknown symbol {m['symbol']}")
        if not (m["event_time"] <= m["decision_time"] <= m["order_time"]):
            problems.append("marker time chain not monotone")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "public"
        / "fixtures"
        / "session.synthetic.json",
    )
    parser.add_argument("--symbols", type=int, default=8)
    parser.add_argument("--bars", type=int, default=390)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    session = gen_session(n_symbols=args.symbols, n_bars=args.bars, seed=args.seed)
    problems = validate_session(session)
    if problems:
        raise SystemExit(f"generated session failed validation: {problems[:5]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(session, separators=(",", ":"))
    args.out.write_text(payload)
    print(
        f"wrote {args.out} ({len(payload) / 1e6:.2f} MB, "
        f"{args.symbols} symbols x {args.bars} bars, seed={args.seed})"
    )


if __name__ == "__main__":
    main()

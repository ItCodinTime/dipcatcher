"""Vendor L2 column maps → Northset book panel (ADR-021).

Offline dry-run only. No network pulls. Maps common Alpaca / Polygon quote
shapes onto ``BOOK_PANEL_REQUIRED``, then validates. SYNTHETIC fixtures can
wear a vendor column disguise to prove the remap path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from quant_fund.microstructure.book_panel import BOOK_PANEL_REQUIRED, validate_book_panel

# Canonical Northset targets (subset computed from top-of-book when absent).
_TOP_DERIVABLE = (
    "mid",
    "spread",
    "spread_bps",
    "microprice",
    "microprice_minus_mid",
    "microprice_minus_mid_bps",
    "imbalance_top",
    "imbalance_depth",
    "bid_depth",
    "ask_depth",
)

# Vendor → Northset column aliases (first match wins).
VENDOR_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "security_id": ("security_id", "symbol", "S", "ticker", "T"),
    "event_time": (
        "event_time",
        "participant_timestamp",
        "timestamp",
        "t",
        "sip_timestamp",
        "tape_timestamp",
    ),
    "available_time": ("available_time", "sip_timestamp", "timestamp", "t", "event_time"),
    "source": ("source", "vendor", "exchange"),
    "best_bid": ("best_bid", "bid_price", "bid", "bp", "bidPrice", "b"),
    "best_ask": ("best_ask", "ask_price", "ask", "ap", "askPrice", "a"),
    "top_bid_size": ("top_bid_size", "bid_size", "bs", "bidSize", "bid_sz"),
    "top_ask_size": ("top_ask_size", "ask_size", "as", "askSize", "ask_sz"),
}

ALPACA_QUOTE_ALIASES: dict[str, tuple[str, ...]] = {
    "security_id": ("S", "symbol", "security_id"),
    "event_time": ("t", "timestamp", "event_time"),
    "available_time": ("t", "timestamp", "available_time"),
    "best_bid": ("bp", "bid_price", "best_bid"),
    "best_ask": ("ap", "ask_price", "best_ask"),
    "top_bid_size": ("bs", "bid_size", "top_bid_size"),
    "top_ask_size": ("as", "ask_size", "top_ask_size"),
}

POLYGON_QUOTE_ALIASES: dict[str, tuple[str, ...]] = {
    "security_id": ("ticker", "T", "symbol", "security_id"),
    "event_time": ("participant_timestamp", "sip_timestamp", "t", "timestamp", "event_time"),
    "available_time": ("sip_timestamp", "available_time", "t", "timestamp"),
    "best_bid": ("bid", "bid_price", "bp", "best_bid"),
    "best_ask": ("ask", "ask_price", "ap", "best_ask"),
    "top_bid_size": ("bid_size", "bs", "top_bid_size"),
    "top_ask_size": ("ask_size", "as", "top_ask_size"),
}

VENDOR_PRESETS: dict[str, dict[str, tuple[str, ...]]] = {
    "generic": VENDOR_COLUMN_ALIASES,
    "alpaca": ALPACA_QUOTE_ALIASES,
    "polygon": POLYGON_QUOTE_ALIASES,
}


@dataclass(frozen=True)
class VendorMapDryRun:
    """Result of an offline vendor→panel column dry-run."""

    vendor: str
    mapped: dict[str, str] = field(default_factory=dict)
    missing_targets: tuple[str, ...] = ()
    unused_vendor_columns: tuple[str, ...] = ()
    derivable_ok: bool = False
    notes: tuple[str, ...] = ()
    event_clock: str = "vendor_timestamp"
    available_clock: str = "vendor_timestamp"

    def as_dict(self) -> dict[str, Any]:
        return {
            "vendor": self.vendor,
            "mapped": dict(self.mapped),
            "missing_targets": list(self.missing_targets),
            "unused_vendor_columns": list(self.unused_vendor_columns),
            "derivable_ok": self.derivable_ok,
            "notes": list(self.notes),
            "event_clock": self.event_clock,
            "available_clock": self.available_clock,
            "research_only": True,
            "claim": "research_diagnostic_only",
        }


def _clock_labels(columns: set[str], mapped: dict[str, str]) -> tuple[str, str]:
    """PIT clocks: venue time for the event, SIP time for availability when both exist."""
    has_part = "participant_timestamp" in columns
    has_sip = "sip_timestamp" in columns
    if has_part and has_sip:
        return "participant", "sip"
    if has_sip:
        return "sip", "sip"
    if has_part:
        if mapped.get("available_time") == "participant_timestamp":
            return "participant", "participant_unverified"
        if "available_time" not in mapped:
            return "participant", "missing"
        return "participant", "vendor_timestamp"
    return "vendor_timestamp", "vendor_timestamp"


def _resolve_aliases(
    columns: set[str],
    aliases: dict[str, tuple[str, ...]],
) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for target, options in aliases.items():
        for opt in options:
            if opt in columns:
                mapped[target] = opt
                break
    return mapped


def dry_run_vendor_book_map(
    columns: list[str] | tuple[str, ...],
    *,
    vendor: str = "generic",
) -> VendorMapDryRun:
    """Report which Northset book-panel targets a vendor schema can cover."""
    key = str(vendor).strip().lower()
    if key not in VENDOR_PRESETS:
        raise ValueError(f"unknown vendor preset: {vendor!r} (want {sorted(VENDOR_PRESETS)})")
    cols = set(columns)
    aliases = VENDOR_PRESETS[key]
    mapped = _resolve_aliases(cols, aliases)
    # Targets we need either mapped or derivable after top-of-book present.
    need_direct = (
        "security_id",
        "event_time",
        "available_time",
        "best_bid",
        "best_ask",
        "top_bid_size",
        "top_ask_size",
    )
    missing = tuple(t for t in need_direct if t not in mapped)
    used = set(mapped.values())
    unused = tuple(sorted(c for c in columns if c not in used))
    derivable_ok = missing == ()
    notes: list[str] = []
    event_clock, available_clock = _clock_labels(cols, mapped)
    notes.append(f"event_clock={event_clock} available_clock={available_clock}")
    if event_clock == "participant" and available_clock == "sip":
        notes.append("PIT: event_time=participant_timestamp, available_time=sip_timestamp")
    elif available_clock == "participant_unverified":
        notes.append("available_time uses participant clock — SIP lag unverified")
    if "source" not in mapped:
        notes.append("source missing — remap will default to vendor preset name")
    if derivable_ok:
        notes.append("top-of-book complete — mid/spread/microprice/imbalance can be derived")
    else:
        notes.append(f"incomplete top-of-book; missing={list(missing)}")
    # BOOK_PANEL_REQUIRED coverage after derive
    covered = set(mapped) | (set(_TOP_DERIVABLE) if derivable_ok else set())
    if "source" not in covered:
        covered.add("source")  # defaulted
    still = tuple(c for c in BOOK_PANEL_REQUIRED if c not in covered)
    if still and derivable_ok:
        notes.append(
            f"after derive, still need depth slopes optional: {[c for c in still if c not in _TOP_DERIVABLE]}"
        )
    return VendorMapDryRun(
        vendor=key,
        mapped=mapped,
        missing_targets=missing,
        unused_vendor_columns=unused,
        derivable_ok=derivable_ok,
        notes=tuple(notes),
        event_clock=event_clock,
        available_clock=available_clock,
    )


def remap_vendor_quotes_to_panel(
    frame: pl.DataFrame,
    *,
    vendor: str = "generic",
    default_source: str | None = None,
) -> pl.DataFrame:
    """Remap a vendor quote frame into a validated Northset book panel.

    Expects top-of-book columns (possibly aliased). Depth beyond top is filled
    with top sizes so imbalance_depth == imbalance_top when L2 depth absent.
    """
    report = dry_run_vendor_book_map(list(frame.columns), vendor=vendor)
    if not report.derivable_ok:
        raise ValueError(
            f"vendor map incomplete for {vendor!r}: missing {list(report.missing_targets)}"
        )
    # Duplicate columns when one vendor field maps to multiple targets
    # (e.g. timestamp → event_time and available_time).
    pieces: list[pl.Expr] = []
    for target, src in report.mapped.items():
        pieces.append(pl.col(src).alias(target))
    base = frame.select(pieces)
    if "source" not in base.columns:
        src_name = default_source or str(vendor)
        base = base.with_columns(pl.lit(src_name).alias("source"))
    base, event_clock, available_clock = _apply_sip_participant_clocks(frame, base)
    # Derive metrics from top of book.
    best_bid = pl.col("best_bid")
    best_ask = pl.col("best_ask")
    mid = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid
    bid_sz = pl.col("top_bid_size")
    ask_sz = pl.col("top_ask_size")
    denom = bid_sz + ask_sz
    micro = (
        pl.when(denom > 0.0).then((best_ask * bid_sz + best_bid * ask_sz) / denom).otherwise(mid)
    )
    imb = pl.when(denom > 0.0).then((bid_sz - ask_sz) / denom).otherwise(0.0)
    out = base.with_columns(
        mid.alias("mid"),
        spread.alias("spread"),
        pl.when(mid > 0.0).then(1e4 * spread / mid).otherwise(0.0).alias("spread_bps"),
        micro.alias("microprice"),
        (micro - mid).alias("microprice_minus_mid"),
        pl.when(mid > 0.0)
        .then(1e4 * (micro - mid) / mid)
        .otherwise(0.0)
        .alias("microprice_minus_mid_bps"),
        imb.alias("imbalance_top"),
        imb.alias("imbalance_depth"),
        bid_sz.alias("bid_depth"),
        ask_sz.alias("ask_depth"),
        # Top-of-book vendor quotes have no depth ladder — leave slopes undefined.
        pl.lit(1.0).alias("n_bid_levels"),
        pl.lit(1.0).alias("n_ask_levels"),
        pl.lit(float("nan")).alias("bid_log_size_slope"),
        pl.lit(float("nan")).alias("ask_log_size_slope"),
        pl.lit(event_clock).alias("event_clock"),
        pl.lit(available_clock).alias("available_clock"),
    )
    return validate_book_panel(out)


def _apply_sip_participant_clocks(
    original: pl.DataFrame, base: pl.DataFrame
) -> tuple[pl.DataFrame, str, str]:
    """Prefer participant time for the event and SIP time for availability.

    When both clocks exist, ``available_time = max(sip, participant)`` so a
    SIP-lagged consumer cannot see the quote before the venue print, and clock
    skew cannot mint ``available_time < event_time``. Participant-only quotes
    are labeled ``participant_unverified``.
    """
    cols = set(original.columns)
    has_part = "participant_timestamp" in cols
    has_sip = "sip_timestamp" in cols
    if has_part and has_sip:
        if original.height != base.height:
            raise ValueError("clock columns do not align with remapped panel")
        clocks = original.select(
            pl.col("participant_timestamp").alias("_event_clock_ts"),
            pl.col("sip_timestamp").alias("_sip_clock_ts"),
        )
        out = (
            pl.concat([base, clocks], how="horizontal")
            .with_columns(
                pl.col("_event_clock_ts").alias("event_time"),
                pl.max_horizontal("_sip_clock_ts", "_event_clock_ts").alias("available_time"),
            )
            .drop(["_event_clock_ts", "_sip_clock_ts"])
        )
        return out, "participant", "sip"
    if has_sip:
        return base, "sip", "sip"
    if has_part:
        return base, "participant", "participant_unverified"
    return base, "vendor_timestamp", "vendor_timestamp"


def disguise_panel_as_alpaca(panel: pl.DataFrame) -> pl.DataFrame:
    """Wear Alpaca quote column names (offline fixture helper)."""
    required = ("security_id", "event_time", "best_bid", "best_ask", "top_bid_size", "top_ask_size")
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"panel missing columns: {missing}")
    return panel.select(
        pl.col("security_id").alias("S"),
        pl.col("event_time").alias("t"),
        pl.col("best_bid").alias("bp"),
        pl.col("best_ask").alias("ap"),
        pl.col("top_bid_size").alias("bs"),
        pl.col("top_ask_size").alias("as"),
    )


def disguise_panel_as_polygon(panel: pl.DataFrame) -> pl.DataFrame:
    """Wear Polygon quote column names (offline fixture helper)."""
    required = ("security_id", "event_time", "best_bid", "best_ask", "top_bid_size", "top_ask_size")
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"panel missing columns: {missing}")
    return panel.select(
        pl.col("security_id").alias("ticker"),
        pl.col("event_time").alias("sip_timestamp"),
        pl.col("best_bid").alias("bid"),
        pl.col("best_ask").alias("ask"),
        pl.col("top_bid_size").alias("bid_size"),
        pl.col("top_ask_size").alias("ask_size"),
    )


def vendor_panel_from_bars(
    bars: pl.DataFrame,
    *,
    vendor: str = "alpaca",
    depth: int = 5,
    seed: int = 7,
    base_spread_bps: float = 4.0,
) -> pl.DataFrame:
    """Build SYNTHETIC L2, disguise as vendor quotes, remap → validated panel.

    Timestamps/security_ids stay aligned to ``bars`` so Northset joins succeed.
    ``source`` on the panel is the vendor preset name (honesty stamp).
    """
    from quant_fund.microstructure.synthetic_lob import synthesize_l2_from_bars

    raw_panel = synthesize_l2_from_bars(
        bars, depth=depth, seed=seed, base_spread_bps=base_spread_bps
    )
    key = str(vendor).strip().lower()
    if key == "alpaca":
        disguised = disguise_panel_as_alpaca(raw_panel)
    elif key == "polygon":
        disguised = disguise_panel_as_polygon(raw_panel)
    else:
        raise ValueError(f"vendor_panel_from_bars supports alpaca|polygon, got {vendor!r}")
    return remap_vendor_quotes_to_panel(disguised, vendor=key)


_BAR_TARGETS = (
    "security_id",
    "event_time",
    "available_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

GENERIC_BAR_ALIASES: dict[str, tuple[str, ...]] = {
    "security_id": ("security_id", "symbol", "S", "ticker", "T"),
    "event_time": (
        "event_time",
        "participant_timestamp",
        "timestamp",
        "t",
        "sip_timestamp",
        "window_end",
    ),
    "available_time": (
        "available_time",
        "sip_timestamp",
        "timestamp",
        "t",
        "event_time",
        "window_end",
    ),
    "open": ("open", "o", "Open"),
    "high": ("high", "h", "High"),
    "low": ("low", "l", "Low"),
    "close": ("close", "c", "Close"),
    "volume": ("volume", "v", "Volume"),
    "source": ("source", "vendor", "exchange"),
}

ALPACA_BAR_ALIASES: dict[str, tuple[str, ...]] = {
    "security_id": ("S", "symbol", "security_id"),
    "event_time": ("t", "timestamp", "event_time"),
    "available_time": ("t", "timestamp", "available_time"),
    "open": ("o", "open"),
    "high": ("h", "high"),
    "low": ("l", "low"),
    "close": ("c", "close"),
    "volume": ("v", "volume"),
}

POLYGON_BAR_ALIASES: dict[str, tuple[str, ...]] = {
    "security_id": ("ticker", "T", "symbol", "security_id"),
    "event_time": ("participant_timestamp", "sip_timestamp", "t", "timestamp", "event_time"),
    "available_time": ("sip_timestamp", "available_time", "t", "timestamp"),
    "open": ("o", "open"),
    "high": ("h", "high"),
    "low": ("l", "low"),
    "close": ("c", "close"),
    "volume": ("v", "volume"),
}

BAR_PRESETS: dict[str, dict[str, tuple[str, ...]]] = {
    "generic": GENERIC_BAR_ALIASES,
    "alpaca": ALPACA_BAR_ALIASES,
    "polygon": POLYGON_BAR_ALIASES,
}


def dry_run_vendor_bar_map(
    columns: list[str] | tuple[str, ...],
    *,
    vendor: str = "generic",
) -> VendorMapDryRun:
    """Report which OHLCV targets a vendor bar schema can cover (PIT clocks)."""
    key = str(vendor).strip().lower()
    if key not in BAR_PRESETS:
        raise ValueError(f"unknown vendor bar preset: {vendor!r} (want {sorted(BAR_PRESETS)})")
    cols = set(columns)
    mapped = _resolve_aliases(cols, BAR_PRESETS[key])
    missing = tuple(t for t in _BAR_TARGETS if t not in mapped)
    used = set(mapped.values())
    unused = tuple(sorted(c for c in columns if c not in used))
    event_clock, available_clock = _clock_labels(cols, mapped)
    notes = [f"event_clock={event_clock} available_clock={available_clock}"]
    if event_clock == "participant" and available_clock == "sip":
        notes.append("PIT: event_time=participant_timestamp, available_time=sip_timestamp")
    elif available_clock == "participant_unverified":
        notes.append("available_time uses participant clock — SIP lag unverified")
    if missing:
        notes.append(f"incomplete OHLCV; missing={list(missing)}")
    else:
        notes.append("OHLCV complete")
    return VendorMapDryRun(
        vendor=key,
        mapped=mapped,
        missing_targets=missing,
        unused_vendor_columns=unused,
        derivable_ok=missing == (),
        notes=tuple(notes),
        event_clock=event_clock,
        available_clock=available_clock,
    )


def remap_vendor_bars(
    frame: pl.DataFrame,
    *,
    vendor: str = "generic",
    default_source: str | None = None,
) -> pl.DataFrame:
    """Remap vendor OHLCV onto the Northset bar PIT contract.

    Same SIP vs participant clocks as quotes: event is the venue print,
    available is SIP (or max(sip, participant) under skew). Participant-only
    bars fail closed because SIP lag is unverified.
    """
    report = dry_run_vendor_bar_map(list(frame.columns), vendor=vendor)
    if not report.derivable_ok:
        raise ValueError(
            f"vendor bar map incomplete for {vendor!r}: missing {list(report.missing_targets)}"
        )
    pieces: list[pl.Expr] = []
    for target, src in report.mapped.items():
        pieces.append(pl.col(src).alias(target))
    base = frame.select(pieces)
    if "source" not in base.columns:
        base = base.with_columns(pl.lit(default_source or str(vendor)).alias("source"))
    base, event_clock, available_clock = _apply_sip_participant_clocks(frame, base)
    if event_clock == "participant" and available_clock == "participant_unverified":
        raise ValueError(
            "vendor bar map incomplete for participant-only timestamps: "
            "available_time missing SIP clock"
        )
    out = base.with_columns(
        pl.lit(event_clock).alias("event_clock"),
        pl.lit(available_clock).alias("available_clock"),
    )
    invalid = out.filter(pl.col("event_time") > pl.col("available_time"))
    if invalid.height:
        raise ValueError("vendor bars contain available_time before event_time")
    return out

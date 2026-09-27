"""Normalized news-record schema shared by all ingest paths.

Every headline — regardless of source or language — is normalized to a
:class:`NewsRecord`. Timestamps are timezone-aware and normalized to UTC at
the boundary; naive timestamps are rejected rather than guessed, because a
silent local-time interpretation is exactly the kind of bug that produces
lookahead in a downstream signal.

``data_label`` follows the repo's honesty contract: fixture corpora are
``"SYNTHETIC"``; the live fetcher stamps ``"LIVE"``. Any downstream receipt
must refuse to seal a blob whose rows are not provably synthetic when it is
documented as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "zh", "ja")

_SYNTHETIC_LABEL = "SYNTHETIC"


def normalize_utc(ts: datetime, *, field_name: str = "timestamp") -> datetime:
    """Return ``ts`` as a timezone-aware UTC datetime, or raise.

    Naive datetimes are rejected: ingest must state the source timezone
    explicitly (e.g. ``Asia/Shanghai`` for Chinese wires) instead of relying
    on the host clock.
    """
    if not isinstance(ts, datetime):
        raise TypeError(f"{field_name} must be a datetime, got {type(ts).__name__}")
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware (naive {ts!r})")
    return ts.astimezone(UTC)


def _normalize_symbols(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple, set, frozenset)):
        raise TypeError("symbols must be an iterable of strings")
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise TypeError(f"symbol must be a string, got {type(item).__name__}")
        cleaned = item.strip().upper()
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return tuple(out)


@dataclass(frozen=True)
class NewsRecord:
    """One normalized news/filings headline.

    Attributes:
        record_id: stable identifier (source-native id or content hash).
        publish_ts: publisher-stated publication timestamp (tz-aware UTC).
        ingest_ts: timestamp the record entered our system (tz-aware UTC).
        language: ISO-639-1 code in :data:`SUPPORTED_LANGUAGES`.
        source: free-form source name (``"synthetic_fixture"``, ``"gdelt"``…).
        headline: headline text; may be multilingual.
        body: optional article/filing body text.
        symbols: publisher-tagged tickers (may be empty; the entity linker
            fills gaps from text). Stored uppercase.
        data_label: honesty label — ``"SYNTHETIC"`` for fixture corpora,
            ``"LIVE"`` for fetched records.
        sentiment_label: optional gold label (``pos``/``neu``/``neg``) for
            classifier fitting/eval on synthetic corpora only.
    """

    record_id: str
    publish_ts: datetime
    ingest_ts: datetime
    language: str
    source: str
    headline: str
    body: str | None = None
    symbols: tuple[str, ...] = ()
    data_label: str = _SYNTHETIC_LABEL
    sentiment_label: str | None = None
    extra: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "publish_ts", normalize_utc(self.publish_ts, field_name="publish_ts"))
        object.__setattr__(self, "ingest_ts", normalize_utc(self.ingest_ts, field_name="ingest_ts"))
        if not isinstance(self.record_id, str) or not self.record_id.strip():
            raise ValueError("record_id must be a non-empty string")
        lang = str(self.language).strip().lower()
        if lang not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"unsupported language {self.language!r}; expected one of {SUPPORTED_LANGUAGES}"
            )
        object.__setattr__(self, "language", lang)
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")
        if not isinstance(self.headline, str) or not self.headline.strip():
            raise ValueError("headline must be a non-empty string")
        object.__setattr__(self, "symbols", _normalize_symbols(self.symbols))
        if self.sentiment_label is not None and self.sentiment_label not in {
            "pos",
            "neu",
            "neg",
        }:
            raise ValueError("sentiment_label must be one of pos|neu|neg")
        if self.body is not None and not isinstance(self.body, str):
            raise TypeError("body must be a string or None")

    @property
    def effective_ts(self) -> datetime:
        """Knowledge timestamp: ``max(publish_ts, ingest_ts)``.

        A record cannot be known before it is published, nor before it was
        actually ingested; the later of the two governs signal availability.
        """
        return max(self.publish_ts, self.ingest_ts)

    @property
    def text(self) -> str:
        """Headline plus optional body — the unit scored/linked on."""
        return self.headline if self.body is None else f"{self.headline}\n{self.body}"


def record_from_dict(payload: dict[str, Any], *, index: int | None = None) -> NewsRecord:
    """Build a :class:`NewsRecord` from a JSONL fixture / wire dict.

    ``publish_ts`` / ``ingest_ts`` accept ISO-8601 strings (``Z`` suffix
    supported). ``ingest_ts`` defaults to ``publish_ts`` when absent —
    fixtures that want to model wire latency must set it explicitly.
    """
    if not isinstance(payload, dict):
        raise TypeError("record payload must be a dict")
    where = f" (record #{index})" if index is not None else ""

    def _ts(key: str) -> datetime:
        raw = payload.get(key)
        if raw is None:
            if key == "ingest_ts" and payload.get("publish_ts") is not None:
                return _ts("publish_ts")
            raise ValueError(f"missing {key!r}{where}")
        if isinstance(raw, datetime):
            return normalize_utc(raw, field_name=key)
        if isinstance(raw, str):
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"unparseable {key} {raw!r}{where}") from exc
            return normalize_utc(parsed, field_name=key)
        raise TypeError(f"{key} must be an ISO-8601 string or datetime{where}")

    known = {
        "record_id",
        "publish_ts",
        "ingest_ts",
        "language",
        "source",
        "headline",
        "body",
        "symbols",
        "data_label",
        "sentiment_label",
    }
    extra = {k: v for k, v in payload.items() if k not in known}
    rid = payload.get("record_id")
    if rid is None:
        # Content-hash fallback keeps dedup stable without inventing ids.
        from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

        rid = hash_bytes(canonical_json_bytes(payload))[:16]
    return NewsRecord(
        record_id=str(rid),
        publish_ts=_ts("publish_ts"),
        ingest_ts=_ts("ingest_ts"),
        language=str(payload.get("language", "")),
        source=str(payload.get("source", "")),
        headline=str(payload.get("headline", "")),
        body=payload.get("body"),
        symbols=_normalize_symbols(payload.get("symbols")),
        data_label=str(payload.get("data_label", _SYNTHETIC_LABEL)),
        sentiment_label=payload.get("sentiment_label"),
        extra=extra,
    )

"""News ingest: local fixture corpora plus an opt-in live fetch path.

CI and the unit suite only ever touch the local JSONL fixtures under
``tests/fixtures/news/`` (all ``data_label="SYNTHETIC"``). The live path,
:class:`LiveNewsFetcher`, uses ``httpx`` — imported lazily so the package
imports cleanly where httpx is absent — and is never invoked by tests.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from quant_fund.research.news_signals.schema import NewsRecord, record_from_dict

FIXTURE_SUFFIX = ".jsonl"


def load_fixture_corpus(paths: Iterable[str | Path]) -> list[NewsRecord]:
    """Load ``NewsRecord``s from one or more JSONL files or directories.

    Directories are expanded to their ``*.jsonl`` members (sorted for
    determinism). Blank lines and ``#`` comment lines are skipped, so fixture
    files can carry a leading SYNTHETIC banner comment.
    """
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.glob(f"*{FIXTURE_SUFFIX}")))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(f"news corpus path not found: {p}")
    records: list[NewsRecord] = []
    for path in files:
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path.name}:{i}: invalid JSONL record") from exc
                records.append(record_from_dict(payload, index=i))
    return records


def iter_fixture_corpus(paths: Iterable[str | Path]) -> Iterator[NewsRecord]:
    """Streaming twin of :func:`load_fixture_corpus`."""
    yield from load_fixture_corpus(paths)


def write_fixture_corpus(records: Iterable[NewsRecord], path: str | Path) -> Path:
    """Serialize records to JSONL (used to regenerate fixtures deterministically)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for rec in records:
        lines.append(
            json.dumps(
                {
                    "record_id": rec.record_id,
                    "publish_ts": rec.publish_ts.isoformat(),
                    "ingest_ts": rec.ingest_ts.isoformat(),
                    "language": rec.language,
                    "source": rec.source,
                    "headline": rec.headline,
                    "body": rec.body,
                    "symbols": list(rec.symbols),
                    "data_label": rec.data_label,
                    "sentiment_label": rec.sentiment_label,
                    **rec.extra,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out


class LiveNewsFetcher:
    """Optional live fetch via ``httpx``. Never called in tests or CI.

    This is a thin research convenience: ``fetch_gdelt`` pulls headline rows
    from the GDELT DOC 2.1 API (``mode=artlist``) and normalizes them to
    :class:`NewsRecord` with ``data_label="LIVE"``. It performs real network
    I/O on every call — callers must opt in explicitly.
    """

    def __init__(self, *, timeout: float = 30.0, user_agent: str | None = None) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = float(timeout)
        self.user_agent = user_agent or "dipcatcher-news/0.1 (research)"

    def _client(self) -> Any:
        try:
            import httpx  # noqa: PLC0415 — dev-only dep, lazy by contract
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "httpx is required for live news fetch (dev dependency); "
                "fixture ingest does not need it"
            ) from exc
        return httpx.Client(
            timeout=self.timeout,
            headers={"User-Agent": self.user_agent, "Accept": "application/json"},
            follow_redirects=True,
        )

    def fetch_gdelt(
        self,
        *,
        query: str,
        maxrecords: int = 50,
        source_lang: str | None = None,
    ) -> list[NewsRecord]:  # pragma: no cover - live network, never in tests
        """Fetch GDELT artlist headlines and normalize to ``NewsRecord``s.

        ``query`` is a GDELT query string (e.g. ``"apple"``, ``"トヨタ"``);
        ``source_lang`` optionally restricts to a source language code GDELT
        understands (``english``/``chinese``/``japanese``).
        """
        if not query or not str(query).strip():
            raise ValueError("query must be non-empty")
        if maxrecords < 1 or maxrecords > 250:
            raise ValueError("maxrecords must be in [1, 250]")
        params: dict[str, Any] = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": int(maxrecords),
            "sort": "hybridrel",
        }
        if source_lang:
            params["sourcelang"] = source_lang
        ingested = datetime.now(UTC)
        with self._client() as client:
            resp = client.get(
                "https://api.gdeltproject.org/api/v2/doc/doc", params=params
            )
            resp.raise_for_status()
            payload = resp.json()
        articles = payload.get("articles", []) if isinstance(payload, dict) else []
        records: list[NewsRecord] = []
        lang_map = {"english": "en", "chinese": "zh", "japanese": "ja"}
        for i, art in enumerate(articles):
            if not isinstance(art, dict):
                continue
            raw_lang = str(art.get("language") or source_lang or "english").lower()
            lang = lang_map.get(raw_lang, raw_lang[:2])
            if lang not in {"en", "zh", "ja"}:
                continue  # keep the supported-language contract fail-closed
            seen = art.get("seendate") or ""
            try:
                publish = datetime.strptime(seen, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
            except ValueError:
                continue  # unusable timestamp → drop, never guess
            records.append(
                NewsRecord(
                    record_id=str(art.get("url") or f"gdelt-{i}"),
                    publish_ts=publish,
                    ingest_ts=ingested,
                    language=lang,
                    source="gdelt",
                    headline=str(art.get("title") or ""),
                    body=None,
                    symbols=(),
                    data_label="LIVE",
                )
            )
        return records

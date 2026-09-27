"""Multilingual entity + ticker linking (en / zh / ja).

Mentions in headlines — Latin names, CJK aliases, cashtags, exchange-qualified
codes — are resolved to canonical tickers via a hand-curated alias table.

Canonical-ticker convention
---------------------------
* US primary listing or US ADR where one exists (``AAPL``, ``TM`` …).
* Otherwise the primary-listing exchange-qualified code (``7203.T``,
  ``9988.HK``, ``600519.SS``).
* ``adrs`` carries cross-listed equivalents so a signal can join whichever
  listing the downstream panel actually trades (e.g. Toyota resolves to
  ``7203.T`` with ADR ``TM``; Alibaba to ``9988.HK`` with ADR ``BABA``).

Disambiguation rules (all deterministic, applied in this order)
---------------------------------------------------------------
1. Publisher-tagged ``record.symbols`` are trusted verbatim (uppercased).
2. Cashtags ``$AAPL`` and exchange-qualified mentions ``(TYO: 7203)`` /
   ``(NASDAQ: AAPL)`` are unambiguous — they bypass alias scanning.
3. CJK aliases match by substring (no word boundaries exist in zh/ja).
   Latin aliases match on word boundaries, case-insensitive.
4. Longest alias wins on overlap (``トヨタ自動車`` beats ``トヨタ`` … but both
   map to the same canonical anyway; the rule matters for cross-name cases
   like ``苹果`` vs ``苹果公司``).
5. Bare ALL-CAPS tokens (2-6 chars, optional ``.XX`` suffix) match only the
   ticker index (canonicals + ADRs) *and* must not be in
   :data:`BARE_TICKER_STOPWORDS` — a curated list of finance stopwords
   (``CEO``, ``IPO``, ``GDP`` …) and one-letter/double-duty symbols
   (``A``, ``F``, ``T``, ``V``, ``MA``, ``GM``) that read as ordinary words
   or acronyms. Those names still resolve via cashtag/qualified/alias paths.
6. Multiple entities in one headline are all kept (``"Apple and Nvidia
   …"`` → ``AAPL, NVDA``); the signal layer counts each link once per record.

Known limitations: the alias table is coverage-bounded (documented entries
only); ``apple`` can be the fruit in a non-finance corpus; ``meta`` alone is
mapped to Meta Platforms which is correct for finance headlines but wrong in
e.g. academic corpora. These are documented — never silently patched with a
learned model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from quant_fund.research.news_signals.schema import NewsRecord

# Symbols whose bare ALL-CAPS token is too ambiguous to trust without an
# explicit cashtag/qualified/alias mention. Curated, documented.
BARE_TICKER_STOPWORDS: frozenset[str] = frozenset(
    {
        # one/two-letter tickers that double as words or province/country codes
        "A", "F", "T", "V", "I", "M", "N", "C", "B", "D", "E", "G", "H", "J",
        "K", "L", "O", "P", "Q", "R", "S", "U", "W", "X", "Y", "Z",
        "MA", "GM", "DD", "HD", "GS", "MS", "BA", "CAT", "CVX", "HP", "IBM",
        "IT", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "IF", "IN",
        "IS", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE", "HE",
        "SHE", "ME", "MY", "ALL", "ARE", "CAN", "FOR", "HAS", "HIM", "HIS",
        "HER", "LOW", "MAN", "NEW", "NOW", "ONE", "OUT", "SEE", "TEN", "TWO",
        "WAS", "WAY", "WHO", "WIN", "YET", "BIG", "FAR", "KEY", "NET", "PAY",
        "RUN", "SUN", "TOP", "TRY", "VIA", "AIR", "ANY", "DAY", "END",
        # finance jargon that is not a ticker mention
        "AI", "EV", "FX", "IPO", "CEO", "CFO", "COO", "CTO", "USD", "JPY",
        "CNY", "RMB", "HKD", "TWD", "GDP", "CPI", "PPI", "PMI", "ETF", "ADR",
        "ADS", "SEC", "FDA", "FTC", "DOJ", "EPS", "EBIT", "EBITDA", "GAAP",
        "YOY", "MOM", "QOQ", "Q1", "Q2", "Q3", "Q4", "H1", "H2", "FY", "QE",
        "QT", "REIT", "OTC", "MNA", "ESG", "NAV", "ROE", "ROA", "PE", "PB",
        "NYSE", "NASDAQ", "TYO", "TSE", "SSE", "SZSE", "HKEX", "LSE", "TSX",
        "JP", "CN", "HK", "TW", "UK", "EU", "PRC", "USA", "INC", "LTD",
        "CORP", "CO", "LLC", "LP", "PLC", "KK", "GMBH", "SA", "AG", "NV",
        "PC", "TV", "PC", "PR", "IR", "XR", "VR", "AR", "ML", "LLM", "GPU",
        "CPU", "RAM", "SSD", "OLED", "LCD", "5G", "6G", "BYD",
    }
)


@dataclass(frozen=True)
class AliasEntry:
    """One canonical ticker and every surface form we resolve to it."""

    canonical: str
    aliases: tuple[str, ...]
    adrs: tuple[str, ...] = ()
    name_en: str = ""


def _e(canonical: str, *aliases: str, adrs: tuple[str, ...] = (), name_en: str = "") -> AliasEntry:
    return AliasEntry(canonical=canonical, aliases=tuple(aliases), adrs=adrs, name_en=name_en)


# Hand-curated coverage-bounded table. Alphabetical by canonical.
ALIAS_TABLE: tuple[AliasEntry, ...] = (
    _e("AAPL", "apple", "apple inc", "苹果", "苹果公司", "アップル", name_en="Apple"),
    _e("AMD", "amd", "超威半导体", "超微半導体", "エーエムディー", name_en="AMD"),
    _e("AMZN", "amazon", "amazon.com", "亚马逊", "亞馬遜", "アマゾン", name_en="Amazon"),
    _e("BABA", "alibaba", "阿里巴巴", "アリババ", adrs=("9988.HK",), name_en="Alibaba"),
    _e("DIS", "disney", "walt disney", "迪士尼", "迪斯尼", "ディズニー", name_en="Disney"),
    _e("F", "ford", "ford motor", "福特", "福特汽车", "フォード", name_en="Ford"),
    _e("GM", "general motors", "通用汽车", "ゼネラルモーターズ", "ゼネラル・モーターズ", name_en="General Motors"),
    _e("GOOGL", "alphabet", "google", "谷歌", "谷歌母公司", "グーグル", "アルファベット", name_en="Alphabet"),
    _e("INTC", "intel", "英特尔", "英特爾", "インテル", name_en="Intel"),
    _e("JD", "jd.com", "京东", "京東", "ジンドン", adrs=("9618.HK",), name_en="JD.com"),
    _e("JPM", "jpmorgan", "jp morgan", "j.p. morgan", "摩根大通", "jpモルガン", name_en="JPMorgan"),
    _e("KO", "coca-cola", "coca cola", "可口可乐", "可口可樂", "コカ・コーラ", "コカコーラ", name_en="Coca-Cola"),
    _e("MA", "mastercard", "万事达", "萬事達", "マスターカード", name_en="Mastercard"),
    _e("META", "meta platforms", "meta", "脸书", "meta公司", "メタ", "フェイスブック", name_en="Meta Platforms"),
    _e("MSFT", "microsoft", "微软", "微軟", "マイクロソフト", name_en="Microsoft"),
    _e("NFLX", "netflix", "奈飞", "网飞", "網飛", "ネットフリックス", name_en="Netflix"),
    _e("NKE", "nike", "耐克", "ナイキ", name_en="Nike"),
    _e("NVDA", "nvidia", "英伟达", "英偉達", "辉达", "エヌビディア", name_en="NVIDIA"),
    _e("PEP", "pepsi", "pepsico", "百事", "百事可乐", "ペプシ", "ペプシコ", name_en="PepsiCo"),
    _e("PFE", "pfizer", "辉瑞", "輝瑞", "ファイザー", name_en="Pfizer"),
    _e("TSLA", "tesla", "特斯拉", "テスラ", name_en="Tesla"),
    _e("TSM", "tsmc", "台积电", "台積電", "台湾積体電路", "台湾セミコンダクター", adrs=("2330.TW",), name_en="TSMC"),
    _e("V", "visa", "维萨", "維薩", "ビザ", name_en="Visa"),
    _e("WMT", "walmart", "沃尔玛", "沃爾瑪", "ウォルマート", name_en="Walmart"),
    _e("XOM", "exxon", "exxon mobil", "埃克森美孚", "埃克森", "エクソン", "エクソンモービル", name_en="ExxonMobil"),
    # Non-US primary listings; ADRs join the US panel where the venue carries it.
    _e("7203.T", "toyota", "丰田", "豐田", "トヨタ", "トヨタ自動車", adrs=("TM",), name_en="Toyota"),
    _e("6758.T", "sony", "索尼", "ソニー", "ソニーグループ", adrs=("SONY",), name_en="Sony"),
    _e("7974.T", "nintendo", "任天堂", "ニンテンドー", adrs=("NTDOY",), name_en="Nintendo"),
    _e("0700.HK", "tencent", "腾讯", "騰訊", "テンセント", adrs=("TCEHY",), name_en="Tencent"),
    _e("600519.SS", "kweichow moutai", "茅台", "贵州茅台", "貴州茅台", name_en="Kweichow Moutai"),
    _e("300750.SZ", "catl", "宁德时代", "寧德時代", name_en="CATL"),
    _e("8306.T", "mufg", "mitsubishi ufj", "三菱ufj", "三菱UFJ", "三菱ufjフィナンシャル", "三菱UFJフィナンシャル", name_en="MUFG"),
    _e("9984.T", "softbank group", "软银", "軟銀", "ソフトバンク", "ソフトバンクグループ", adrs=("SFTBY",), name_en="SoftBank Group"),
    # Additional US-listing coverage (file_us_wide universe members).
    _e("AVGO", "broadcom", "博通", "ブロードコム", name_en="Broadcom"),
    _e("ORCL", "oracle", "甲骨文", "オラクル", name_en="Oracle"),
    _e("CRM", "salesforce", "セールスフォース", name_en="Salesforce"),
    _e("MU", "micron", "micron technology", "美光", "マイクロン", name_en="Micron"),
    _e("CSCO", "cisco", "思科", "シスコ", name_en="Cisco"),
    _e("JNJ", "johnson & johnson", "johnson and johnson", "强生", "強生", "ジョンソン・エンド・ジョンソン", name_en="J&J"),
    _e("MRK", "merck", "默克", "默沙东", "メルク", name_en="Merck"),
    _e("LLY", "eli lilly", "礼来", "禮來", "イーライリリー", name_en="Eli Lilly"),
    _e("MCD", "mcdonald's", "mcdonalds", "麦当劳", "麥當勞", "マクドナルド", name_en="McDonald's"),
    _e("SBUX", "starbucks", "星巴克", "スターバックス", name_en="Starbucks"),
    _e("COST", "costco", "好市多", "コストコ", name_en="Costco"),
    _e("BAC", "bank of america", "美国银行", "美國銀行", "バンク・オブ・アメリカ", name_en="Bank of America"),
    _e("WFC", "wells fargo", "富国银行", "富國銀行", "ウェルズ・ファーゴ", name_en="Wells Fargo"),
    _e("GS", "goldman sachs", "高盛", "ゴールドマン・サックス", name_en="Goldman Sachs"),
    _e("MS", "morgan stanley", "摩根士丹利", "モルガン・スタンレー", name_en="Morgan Stanley"),
)

_CASHTAG_RE = re.compile(r"\$([A-Za-z][A-Za-z0-9.]{0,7})\b")
_QUALIFIED_RE = re.compile(
    r"(?:NASDAQ|NYSE|AMEX|TYO|TSE|東証|HKEX|港交所|上交所|深交所|SSE|SZSE|TSX|LSE)"
    r"\s*[:：]?\s*\(?([A-Za-z0-9]{1,6}(?:\.[A-Za-z]{1,2})?)\)?",
    re.IGNORECASE,
)
_BARE_TICKER_RE = re.compile(r"\b([A-Z0-9]{1,6}(?:\.[A-Z]{1,2})?)\b")


def _latin_key(s: str) -> str:
    return " ".join(s.lower().split())


class EntityLinker:
    """Deterministic mention → canonical-ticker resolver.

    Built once over :data:`ALIAS_TABLE`; thread-free, allocation-light.
    """

    def __init__(self, table: Iterable[AliasEntry] = ALIAS_TABLE) -> None:
        self._alias_to_canonical: dict[str, str] = {}
        self._adr_to_canonical: dict[str, str] = {}
        self._ticker_index: set[str] = set()
        self._cjk_aliases: list[tuple[str, str]] = []  # (alias, canonical)
        self._latin_patterns: list[tuple[re.Pattern[str], str, int]] = []
        for entry in table:
            self._ticker_index.add(entry.canonical.upper())
            for adr in entry.adrs:
                self._adr_to_canonical[adr.upper()] = entry.canonical
                self._ticker_index.add(adr.upper())
            for alias in entry.aliases:
                key = _latin_key(alias)
                if key in self._alias_to_canonical and self._alias_to_canonical[key] != entry.canonical:
                    raise ValueError(f"alias {alias!r} maps to two canonicals")
                self._alias_to_canonical[key] = entry.canonical
                if _is_cjk(alias):
                    self._cjk_aliases.append((alias, entry.canonical))
                else:
                    pat = re.compile(r"(?<![\w$])" + re.escape(_latin_key(alias)) + r"(?!\w)", re.IGNORECASE)
                    self._latin_patterns.append((pat, entry.canonical, len(alias)))
        # Longest-first so overlap resolution is deterministic (rule 4).
        self._cjk_aliases.sort(key=lambda kv: len(kv[0]), reverse=True)
        self._latin_patterns.sort(key=lambda kv: kv[2], reverse=True)

    def canonical_for(self, ticker: str) -> str:
        """Map a bare/ADR ticker to its canonical listing (identity if unknown)."""
        return self._adr_to_canonical.get(ticker.upper(), ticker.upper())

    def link_text(self, text: str) -> tuple[str, ...]:
        """Resolve all ticker mentions in ``text`` (rules 2-5 above)."""
        found: set[str] = set()
        for m in _CASHTAG_RE.finditer(text):
            found.add(self.canonical_for(m.group(1)))
        for m in _QUALIFIED_RE.finditer(text):
            tok = m.group(1).upper()
            if tok in self._ticker_index:
                found.add(self.canonical_for(tok))
            elif tok.isdigit():
                qualified = _numeric_code_to_ticker(tok)
                if qualified in self._ticker_index:
                    found.add(self.canonical_for(qualified))
        for alias, canonical in self._cjk_aliases:
            if alias in text:
                found.add(canonical)
        lowered = text.lower()
        for pat, canonical, _ in self._latin_patterns:
            if pat.search(lowered):
                found.add(canonical)
        for m in _BARE_TICKER_RE.finditer(text):
            tok = m.group(1).upper()
            if tok in self._ticker_index and tok not in BARE_TICKER_STOPWORDS:
                found.add(self.canonical_for(tok))
        return tuple(sorted(found))

    def link_record(self, record: NewsRecord) -> tuple[str, ...]:
        """Resolve tickers for a record: publisher symbols ∪ text mentions."""
        return resolve_symbols(record, linker=self)


def _numeric_code_to_ticker(code: str) -> str:
    """Exchange-qualified numeric code → exchange-suffixed ticker.

    4 digits → ``.T`` (TYO), 5 → ``.HK``, 6 with a ``6`` prefix → ``.SS``
    (SSE), ``0``/``3`` prefix → ``.SZ``. Unknown shapes pass through raw.
    """
    if len(code) == 4:
        return f"{code}.T"
    if len(code) == 5:
        return f"{code}.HK"
    if len(code) == 6:
        if code.startswith("6"):
            return f"{code}.SS"
        if code.startswith(("0", "3")):
            return f"{code}.SZ"
    return code


def _is_cjk(s: str) -> bool:
    return any(
        "぀" <= ch <= "ヿ" or "一" <= ch <= "鿿" or "豈" <= ch <= "﫿"
        for ch in s
    )


def resolve_symbols(
    record: NewsRecord,
    *,
    linker: EntityLinker | None = None,
    include_adrs: bool = False,
    trust_publisher: bool = True,
) -> tuple[str, ...]:
    """Final symbol set for a record (see module docstring for precedence).

    ``include_adrs`` additionally emits cross-listed equivalents so a join can
    match whichever listing the return panel carries.
    """
    linker = linker or EntityLinker()
    found: set[str] = set()
    if trust_publisher:
        found.update(linker.canonical_for(s) for s in record.symbols)
    found.update(linker.link_text(record.text))
    if include_adrs:
        for entry in ALIAS_TABLE:
            if entry.canonical in found:
                found.update(a.upper() for a in entry.adrs)
    return tuple(sorted(found))

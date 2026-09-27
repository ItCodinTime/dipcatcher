"""Lightweight, deterministic sentiment + event classification.

Two sentiment scorers are provided, both honest weak baselines — no
transformers, no embeddings, no claims of learned alpha:

* :class:`LexiconScorer` — hand-curated en/zh/ja polarity lexicons with
  simple negation cues. Fully deterministic, zero training, inspectable.
* :class:`TfidfSentiment` — sklearn ``TfidfVectorizer`` (character n-grams,
  which tokenize-free cover CJK) feeding ``LogisticRegression`` trained on
  the *synthetic labeled fixture corpus only*. It is a correctness harness
  for the pipeline, not a model claim: trained-on-synthetic scores must be
  read as SYNTHETIC evidence of wiring, never of market skill.

``classify_event`` is an ordered rule classifier over the same lexicon idea:
earnings → guidance → M&A → litigation/regulatory → capital-return →
product → macro → credit → other. First match wins (priority documented);
``other`` is the honest fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Protocol

from quant_fund.research.news_signals.schema import NewsRecord

# ---------------------------------------------------------------------------
# Lexicons — hand-curated, finance-domain, per-language. All entries are
# lowercase for latin; CJK entries are literal surface forms.
# ---------------------------------------------------------------------------

LEXICON_POS: dict[str, tuple[str, ...]] = {
    "en": (
        "beats", "beat estimates", "tops estimates", "surges", "soars", "jumps",
        "rallies", "gains", "climbs", "record high", "all-time high", "upgrade",
        "record bookings", "record revenue", "record backlog", "record profit",
        "record quarter", "hits record", "jumped", "rose", "rebound",
        "upgraded", "raises guidance", "raises forecast", "lifts outlook",
        "profit jump", "profit rises", "revenue growth", "earnings beat",
        "outperforms", "expands", "expansion", "buyback", "share repurchase",
        "dividend hike", "special dividend", "wins approval", "approval",
        "partnership", "strategic alliance", "wins contract", "record orders",
        "strong demand", "upbeat", "optimistic", "turnaround", "beats profit",
        "tops forecasts", "raises dividend", "announces buyback",
        "better than expected", "exceeds expectations", "wins", "lifts",
        "lifting", "rally", "rises", "rebounds", "record demand",
    ),
    "zh": (
        "上涨", "大涨", "飙升", "创新高", "创历史新高", "超预期", "上调",
        "上调预期", "上调评级", "增长", "净利润增长", "盈利增长", "扭亏",
        "扭亏为盈", "回购", "增持", "分红", "派息", "利好", "获批", "中标",
        "签约", "战略合作", "扩产", "扩能", "订单创纪录", "需求强劲",
        "业绩超预期", "超预期增长", "提价", "领涨", "净流入", "加码",
        "新高", "创同期新高", "获准", "大额订单", "回暖",
    ),
    "ja": (
        "上昇", "急騰", "最高値", "史上最高値", "上振れ", "増益",
        "増収増益", "黒字転換", "黒字浮上", "自社株買い", "増配", "復配",
        "好調", "堅調", "承認", "受注", "大型受注", "業績上振れ", "上方修正",
        "買い優勢", "増額", "拡大", "戦略提携", "資本提携", "受賞", "首位",
        "最高益", "過去最高", "上回る", "年初来高値", "新高値", "次世代",
        "新提携", "実証開始",
    ),
}

LEXICON_NEG: dict[str, tuple[str, ...]] = {
    "en": (
        "misses", "missed estimates", "plunges", "tumbles", "slumps", "falls",
        "slides", "drops", "declines", "record low", "downgrade", "downgraded",
        "cuts guidance", "lowers forecast", "slashes outlook", "profit warning",
        "warns", "loss widens", "net loss", "recall", "recalls", "lawsuit",
        "slips", "slide", "halts", "suspends", "lays off", "complaint",
        "disappoints", "disappoint", "price war", "falls after", "misses on",
        "sued", "probe", "investigation", "fraud", "accounting irregularities",
        "layoffs", "job cuts", "bankruptcy", "default", "delisting", "fined",
        "penalty", "sanctions", "antitrust", "data breach", "weak demand",
        "misses estimates", "worse than expected", "below expectations",
        "guidance cut", "downbeat", "bearish", "sell-off", "selloff",
    ),
    "zh": (
        "下跌", "大跌", "暴跌", "创新低", "低于预期", "不及预期", "下调",
        "下调预期", "下调评级", "亏损", "净亏损", "亏损扩大", "召回", "诉讼",
        "被起诉", "调查", "立案调查", "处罚", "罚款", "违规", "裁员", "退市",
        "违约", "债务违约", "利空", "警告", "业绩预亏", "商誉减值", "解禁",
        "减持", "质押", "破产", "造假", "财务造假", "问询", "监管函",
        "承压", "跌破", "起诉", "反垄断", "疲软", "新低",
    ),
    "ja": (
        "下落", "急落", "大幅安", "安値", "年初来安値", "下振れ", "減益",
        "下回る", "軟調", "伸び悩み",
        "減収減益", "赤字", "赤字転落", "最終赤字", "リコール", "訴訟",
        "提訴", "調査", "処分", "課徴金", "不祥事", "リストラ", "人員削減",
        "デフォルト", "債務不履行", "下方修正", "業績下振れ", "売り優勢",
        "減配", "無配", "警告", "疑義", "粉飾", "粉飾決算", "監査等級",
        "独占禁止法", "カルテル", "情報流出",
    ),
}

# Negation cues: term immediately preceded within a short window flips sign.
# JA is left to lexicon entries (e.g. 下方修正 is already negative) — suffix
# negation is too ambiguous for rules; documented limitation.
NEGATION_CUES: dict[str, tuple[str, ...]] = {
    "en": ("no ", "not ", "n't ", "never ", "without ", "denies ", "fails to "),
    "zh": ("不", "未", "非", "无", "沒有", "没有", "难以", "不再"),
    "ja": (),
}

NEGATION_WINDOW = 24  # characters of look-back for a cue


EVENT_TYPES: tuple[str, ...] = (
    "earnings",
    "guidance",
    "mna",
    "litigation",
    "capital_return",
    "product",
    "macro",
    "credit",
    "other",
)

# Ordered (category, keywords) — first category with any hit wins.
EVENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("earnings", (
        "earnings", "profit", "revenue", "eps", "quarterly results", "決算",
        "業績", "财报", "业绩", "净利润", "营收", "季报", "年报", "四半期決算",
        "純利益", "売上高", "quarterly", "fiscal year", "net income",
    )),
    ("guidance", (
        "guidance", "outlook", "forecast", "raises forecast", "cuts guidance",
        "预期", "上调预期", "下调预期", "業績予想", "見通し", "上方修正", "下方修正",
        "予想", "guida", "lowers outlook", "raises outlook",
    )),
    ("mna", (
        "merger", "acquisition", "acquires", "takeover", "buyout", "stake",
        "并购", "收购", "合併", "買収", "統合", "資本提携", "出資", "参股",
        "要约收购", "spac", "divest", "spin-off", "分拆",
    )),
    ("litigation", (
        "lawsuit", "sued", "probe", "investigation", "antitrust", "fine",
        "fined", "penalty", "sanctions", "fraud", "诉讼", "被起诉", "调查",
        "立案调查", "处罚", "罚款", "违规", "问询", "訴訟", "提訴", "調査",
        "処分", "課徴金", "独占禁止法", "カルテル", "粉飾", "regulator",
        "监管",
    )),
    ("capital_return", (
        "dividend", "buyback", "repurchase", "special dividend", "分红", "派息",
        "回购", "增持", "减持", "配当", "増配", "減配", "無配", "自社株買い",
        "復配", "shareholder return", "株主還元",
    )),
    ("product", (
        "launches", "unveils", "announces", "new model", "release", "rollout",
        "发布", "新品", "投产", "量产", "発売", "新製品", "発表", "量産",
        "イノベーション", "chip", "model y", "iphone",
    )),
    ("macro", (
        "fed", "central bank", "rate hike", "rate cut", "interest rate",
        "inflation", "央行", "降息", "加息", "通胀", "日銀", "金融政策",
        "利上げ", "利下げ", "為替", "円安", "円高", "pboc", "人民银行",
        "中国人民銀行", "boj", "fomc", "ecb",
    )),
    ("credit", (
        "default", "bond", "debt", "downgrade", "rating", "违约", "债务",
        "债券", "评级", "格付け", "債務", "デフォルト", "債券", "credit",
    )),
)


@dataclass(frozen=True)
class SentimentScore:
    """Lexicon scoring detail — kept on signals for explainability."""

    score: float
    n_pos: int
    n_neg: int
    hits: tuple[str, ...]


class Scorer(Protocol):
    def score(self, text: str, language: str) -> SentimentScore: ...


def _term_pattern(term: str) -> re.Pattern[str]:
    """CJK terms match literally; latin terms on word boundaries, CI."""
    if any("぀" <= ch <= "ヿ" or "一" <= ch <= "鿿" for ch in term):
        return re.compile(re.escape(term))
    return re.compile(r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)")


def _compile(terms: Iterable[str]) -> list[tuple[re.Pattern[str], str]]:
    return [(_term_pattern(t), t) for t in terms]


class LexiconScorer:
    """Deterministic polarity scorer over curated per-language lexicons.

    ``score(text, language)`` returns ``(pos - neg) / (pos + neg + 2)`` —
    bounded in (-1, 1), exactly 0.0 when no lexicon term fires. Negation cues
    flip a term's polarity when the cue appears inside ``NEGATION_WINDOW``
    characters before the match (en/zh only; see docstring caveat for ja).
    """

    def __init__(self) -> None:
        self._pos = {lang: _compile(terms) for lang, terms in LEXICON_POS.items()}
        self._neg = {lang: _compile(terms) for lang, terms in LEXICON_NEG.items()}
        self._cues = {lang: cues for lang, cues in NEGATION_CUES.items()}

    def _count(
        self, text: str, language: str, table: dict[str, list[tuple[re.Pattern[str], str]]]
    ) -> tuple[int, int, list[str]]:
        direct = 0
        flipped = 0
        hits: list[str] = []
        for pat, term in table.get(language, ()):
            for m in pat.finditer(text):
                window = text[max(0, m.start() - NEGATION_WINDOW) : m.start()].lower()
                negated = any(cue in window for cue in self._cues.get(language, ()))
                if negated:
                    flipped += 1
                else:
                    direct += 1
                hits.append(("¬" if negated else "") + term)
        return direct, flipped, hits

    def score(self, text: str, language: str) -> SentimentScore:
        if language not in LEXICON_POS:
            raise ValueError(f"unsupported language {language!r}")
        lowered = text.lower()
        pos_hit, pos_flip, pos_hits = self._count(lowered, language, self._pos)
        neg_hit, neg_flip, neg_hits = self._count(lowered, language, self._neg)
        # A negated positive term votes negative and vice versa.
        n_pos = pos_hit + neg_flip
        n_neg = neg_hit + pos_flip
        denom = n_pos + n_neg + 2.0
        return SentimentScore(
            score=(n_pos - n_neg) / denom,
            n_pos=n_pos,
            n_neg=n_neg,
            hits=tuple(sorted(pos_hits + neg_hits)),
        )


def classify_event(text: str, language: str | None = None) -> str:
    """Ordered rule classifier; first matching category wins, else ``other``."""
    lowered = text.lower()
    for category, keywords in EVENT_RULES:
        for kw in keywords:
            if kw.lower() in lowered:
                return category
    return "other"


_LABEL_TO_SIGN = {"pos": 1.0, "neu": 0.0, "neg": -1.0}


class TfidfSentiment:
    """sklearn char-n-gram TF-IDF + logistic regression, fit on labeled
    SYNTHETIC fixtures only. ``score = P(pos) - P(neg)`` in [-1, 1].

    Deterministic given identical training rows (``lbfgs`` on a fixed
    feature space). ``random_state`` is fixed at construction; result is
    still a weak baseline — do not ship claims from it.
    """

    def __init__(self, *, seed: int = 0, ngram_range: tuple[int, int] = (2, 4)) -> None:
        self.seed = int(seed)
        self.ngram_range = ngram_range
        self._pipeline: object | None = None
        self._classes: list[str] = []
        self.n_train: int = 0

    def fit(self, records: Iterable[NewsRecord]) -> TfidfSentiment:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
        except ImportError as exc:  # pragma: no cover - sklearn is a main dep
            raise RuntimeError("scikit-learn required for TfidfSentiment") from exc
        texts: list[str] = []
        labels: list[str] = []
        for rec in records:
            if rec.sentiment_label is None:
                continue
            texts.append(rec.text)
            labels.append(rec.sentiment_label)
        if len(texts) < 3 or len(set(labels)) < 2:
            raise ValueError("TfidfSentiment needs >=3 labeled records spanning >=2 classes")
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=self.ngram_range,
                        lowercase=True,
                        min_df=1,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(max_iter=500, random_state=self.seed),
                ),
            ]
        )
        pipeline.fit(texts, labels)
        self._pipeline = pipeline
        self._classes = [str(c) for c in pipeline.classes_]
        self.n_train = len(texts)
        return self

    def _proba(self, text: str) -> dict[str, float]:
        if self._pipeline is None:
            raise RuntimeError("TfidfSentiment.score called before fit")
        probs = self._pipeline.predict_proba([text])[0]  # type: ignore[attr-defined]
        return {cls: float(p) for cls, p in zip(self._classes, probs, strict=True)}

    def score(self, text: str, language: str | None = None) -> SentimentScore:
        proba = self._proba(text)
        val = proba.get("pos", 0.0) - proba.get("neg", 0.0)
        return SentimentScore(score=float(val), n_pos=0, n_neg=0, hits=())

    def predict_label(self, text: str) -> str:
        proba = self._proba(text)
        return max(proba, key=lambda k: proba[k])

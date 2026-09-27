"""Predictive-IC evaluation of multilingual news signals + receipt sealing.

Evaluation contract
-------------------
* Targets are forward-return labels produced by the repo's own label engine
  (``quant_fund.labels.engine.build_labels``) — ``future_return_1`` and
  ``future_idio_return_1`` — never a headline-metric aggregate.
* Cross-sectional statistics use ``quant_fund.metrics.cross_section
  .date_ic_series`` (per-date Pearson/Spearman IC with HAC inference) — the
  same machinery ``research.benches.bench_ranking`` reports on.
* The TF-IDF scorer, when evaluated, is fit on the *earliest* labeled slice
  of the synthetic corpus and scored on the remainder (temporal split), so
  even within a SYNTHETIC corpus the eval is not in-sample.
* Receipts follow the ``research.fleet_eval`` convention: sealed
  ``news_eval_<sha256[:16]>.json`` with ``receipt_sha256``, ``inputs_sha256``
  over canonical inputs, ``live_pnl_claim: false``, and a fail-closed
  forbidden-metric scan via the research catalog.

Every result row is diagnostic evidence about the *pipeline*; SYNTHETIC
headlines carry no market information, so measured ICs are expected null.
That is the honest result and it is published as such.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import polars as pl

from quant_fund.metrics.cross_section import date_ic_series
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.research.news_signals.alignment import BarGrid
from quant_fund.research.news_signals.entities import EntityLinker
from quant_fund.research.news_signals.schema import SUPPORTED_LANGUAGES, NewsRecord
from quant_fund.research.news_signals.sentiment import LexiconScorer, TfidfSentiment
from quant_fund.research.news_signals.signals import build_signal_frame
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

NEWS_EVAL_SCHEMA = "news_signals_eval.v1"


@dataclass(frozen=True)
class SignalIC:
    """IC summary for one signal variant against one label target."""

    signal: str
    scorer: str
    target: str
    mean_spearman: float
    mean_pearson: float
    t_spearman: float
    p_spearman: float
    n_dates: int
    n_obs: int
    coverage_names: int
    min_names: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "scorer": self.scorer,
            "target": self.target,
            "mean_spearman": float(self.mean_spearman),
            "mean_pearson": float(self.mean_pearson),
            "t_spearman": float(self.t_spearman),
            "p_spearman": float(self.p_spearman),
            "n_dates": int(self.n_dates),
            "n_obs": int(self.n_obs),
            "coverage_names": int(self.coverage_names),
            "min_names": int(self.min_names),
        }


def evaluate_signal_ic(
    signal: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    score_col: str = "news_sent_mean",
    target: str = "future_return_1",
    symbol_col: str = "symbol",
    label_symbol_col: str = "symbol",
    time_col: str = "event_time",
    min_names: int = 5,
    hac_lags: int | None = None,
) -> SignalIC:
    """Inner-join signal to labels on (symbol, bar); per-date IC + HAC t."""
    required = {symbol_col, time_col, score_col}
    missing = required - set(signal.columns)
    if missing:
        raise ValueError(f"signal frame missing columns: {sorted(missing)}")
    if label_symbol_col not in labels.columns or target not in labels.columns:
        raise ValueError(f"labels frame missing {label_symbol_col!r} or {target!r}")
    joined = signal.select([symbol_col, time_col, score_col]).join(
        labels.select([label_symbol_col, time_col, target]).rename(
            {label_symbol_col: symbol_col}
        ),
        on=[symbol_col, time_col],
        how="inner",
    )
    if joined.height == 0:
        nan = float("nan")
        return SignalIC(score_col, "?", target, nan, nan, nan, nan, 0, 0, 0, min_names)
    scores = joined[score_col].to_numpy().astype(float)
    y = joined[target].to_numpy().astype(float)
    dates = joined[time_col].to_numpy()
    ic = date_ic_series(scores, y, dates, min_names=min_names, hac_lags=hac_lags)
    return SignalIC(
        signal=score_col,
        scorer="?",
        target=target,
        mean_spearman=float(ic.mean_spearman),
        mean_pearson=float(ic.mean_pearson),
        t_spearman=float(ic.t_spearman),
        p_spearman=float(ic.p_spearman),
        n_dates=int(ic.n_dates),
        n_obs=int(joined.height),
        coverage_names=int(joined[symbol_col].n_unique()),
        min_names=int(min_names),
    )


def _labels_from_bars(bars: pl.DataFrame, *, horizons: Sequence[int]) -> pl.DataFrame:
    """Forward-return labels via the repo's own label engine (no new math)."""
    from quant_fund.config.models import AppConfig, HorizonConfig
    from quant_fund.labels.engine import build_labels

    config = AppConfig(
        horizons=HorizonConfig(
            bars=[int(h) for h in horizons],
            names=[f"{int(h)}d" for h in horizons],
        )
    )
    return build_labels(bars, config)


def _records_digest(records: Iterable[NewsRecord]) -> str:
    blob = [
        {
            "record_id": r.record_id,
            "effective_ts": r.effective_ts.isoformat(),
            "language": r.language,
            "headline": r.headline,
            "symbols": list(r.symbols),
            "sentiment_label": r.sentiment_label,
        }
        for r in records
    ]
    return hash_bytes(canonical_json_bytes(blob))


def run_news_eval(
    records: Iterable[NewsRecord],
    *,
    bars: pl.DataFrame | None = None,
    labels: pl.DataFrame | None = None,
    grid: BarGrid | None = None,
    target: str = "future_return_1",
    horizons: Sequence[int] = (1,),
    scorer_names: Sequence[str] = ("lexicon", "count", "tfidf"),
    min_names: int = 5,
    seed: int = 0,
    strict: bool = True,
    train_fraction: float = 0.6,
    bars_source: str = "SYNTHETIC",
    corpus_name: str = "synthetic_fixture",
) -> dict[str, Any]:
    """Run the news-signal IC eval and return the unsealed receipt payload.

    ``bars``/``labels``: exactly one is required — labels are built from bars
    via ``build_labels`` (``future_return_1`` etc.) when not supplied.
    ``scorer_names`` selects variants: ``lexicon`` (rule scorer on all
    records), ``count`` (coverage-count signal), ``tfidf`` (temporal
    train/score split inside the synthetic corpus).
    """
    record_list = list(records)
    if not record_list:
        raise ValueError("news eval requires at least one record")
    if labels is None and bars is None:
        raise ValueError("news eval requires labels or bars")
    if labels is None:
        assert bars is not None
        labels = _labels_from_bars(bars, horizons=horizons)
    if target not in labels.columns:
        raise ValueError(f"labels lack target {target!r}")
    if grid is None:
        grid = BarGrid.from_frame(labels, column="event_time")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")

    linker = EntityLinker()
    data_labels = sorted({r.data_label for r in record_list})
    results: list[dict[str, Any]] = []
    build_meta: dict[str, Any] = {}

    # --- lexicon + count variants: no fitting, score every record -----------
    lexicon = LexiconScorer()
    lex_build = build_signal_frame(
        record_list, grid, scorer=lexicon, linker=linker, strict=strict
    )
    build_meta["lexicon"] = {
        "n_records_in": lex_build.n_records_in,
        "n_aligned": lex_build.n_aligned,
        "n_dropped_past_end": lex_build.n_dropped_past_end,
        "n_unlinked": lex_build.n_unlinked,
        "signal_rows": lex_build.frame.height,
    }
    for name, col in (("lexicon_mean", "news_sent_mean"), ("lexicon_net", "news_sent_net")):
        if "lexicon" not in scorer_names:
            break
        ic = evaluate_signal_ic(
            lex_build.frame, labels, score_col=col, target=target, min_names=min_names
        )
        results.append({**ic.as_dict(), "signal": name, "scorer": "lexicon"})
    if "count" in scorer_names:
        ic = evaluate_signal_ic(
            lex_build.frame, labels, score_col="news_count", target=target, min_names=min_names
        )
        results.append({**ic.as_dict(), "signal": "news_count", "scorer": "coverage"})

    # --- tfidf variant: temporal split inside the synthetic corpus ----------
    if "tfidf" in scorer_names:
        labeled = [r for r in record_list if r.sentiment_label is not None]
        ordered = sorted(labeled, key=lambda r: (r.effective_ts, r.record_id))
        cut = max(1, int(len(ordered) * train_fraction))
        train, test = ordered[:cut], ordered[cut:]
        tfidf_meta: dict[str, Any] = {
            "n_labeled": len(labeled),
            "n_train": len(train),
            "n_test": len(test),
            "status": "skipped",
        }
        if len(test) >= 1 and len(train) >= 3 and len({r.sentiment_label for r in train}) >= 2:
            model = TfidfSentiment(seed=seed)
            model.fit(train)
            tfidf_build = build_signal_frame(
                test, grid, scorer=model, linker=linker, strict=strict
            )
            tfidf_meta.update(
                {
                    "status": "ok",
                    "n_aligned": tfidf_build.n_aligned,
                    "n_dropped_past_end": tfidf_build.n_dropped_past_end,
                    "n_unlinked": tfidf_build.n_unlinked,
                    "signal_rows": tfidf_build.frame.height,
                }
            )
            ic = evaluate_signal_ic(
                tfidf_build.frame,
                labels,
                score_col="news_sent_mean",
                target=target,
                min_names=min_names,
            )
            results.append({**ic.as_dict(), "signal": "tfidf_mean", "scorer": "tfidf_oos"})
        build_meta["tfidf"] = tfidf_meta

    if not results:
        raise ValueError("news eval produced no signal rows to score")

    inputs_sha256 = hash_bytes(
        canonical_json_bytes(
            {
                "records_sha256": _records_digest(record_list),
                "grid": {
                    "first": grid.bar_times[0].isoformat(),
                    "last": grid.bar_times[-1].isoformat(),
                    "n": len(grid.bar_times),
                },
                "target": target,
                "horizons": [int(h) for h in horizons],
                "scorer_names": sorted(set(scorer_names)),
                "min_names": int(min_names),
                "seed": int(seed),
                "strict": bool(strict),
                "train_fraction": float(train_fraction),
                "bars_source": bars_source,
                "corpus_name": corpus_name,
            }
        )
    )
    receipt: dict[str, Any] = {
        "schema": NEWS_EVAL_SCHEMA,
        "kind": "news_signal_ic_eval",
        "data_label": "SYNTHETIC" if data_labels == ["SYNTHETIC"] else "MIXED",
        "news_data_labels": data_labels,
        "live_pnl_claim": False,
        "research_only": True,
        "claim": "research_diagnostic_only",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        "seed": int(seed),
        "ic_method": "date_level_spearman_hac",
        "languages": sorted({r.language for r in record_list} & set(SUPPORTED_LANGUAGES)),
        "grid": {
            "n_bars": len(grid.bar_times),
            "first_bar": grid.bar_times[0].isoformat(),
            "last_bar": grid.bar_times[-1].isoformat(),
            "strict": bool(strict),
        },
        "bars_source": bars_source,
        "corpus_name": corpus_name,
        "target": target,
        "min_names": int(min_names),
        "n_records": len(record_list),
        "build": build_meta,
        "inputs_sha256": inputs_sha256,
        "results": results,
        "caveats": [
            "Headlines are hand-written SYNTHETIC fixtures; they carry no market information.",
            "Lexicon and TF-IDF baselines are weak deterministic scorers, not learned models of alpha.",
            "TF-IDF is fit on an early synthetic slice and scored out-of-sample on the remainder.",
            "Non-US listings (e.g. 7203.T, 9988.HK, 600519.SS) only join the panel via their ADR"
            " where the return frame carries one; unjoined coverage is reported, not hidden.",
        ],
    }
    return receipt


def _atomic_write_text(path: Path, content: str) -> None:
    """Publish a complete immutable text artifact without replacing an existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FileExistsError(f"receipt path is a symlink: {path}")
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"receipt already exists with different content: {path}")
        return
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.is_symlink() or path.read_text(encoding="utf-8") != content:
                raise FileExistsError(
                    f"receipt already exists with different content: {path}"
                ) from None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_news_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
) -> Path:
    """Seal a news-eval receipt to ``receipts/news_eval_<hash>.json``.

    Mirrors ``write_fleet_receipt``: the filename hash is the sha256 of the
    canonical payload and is embedded as ``receipt_sha256``. The write is
    atomic and refuses to overwrite differing content.
    """
    research_blob = {k: v for k, v in receipt.items() if k != "live_pnl_claim"}
    if (
        receipt.get("schema") != NEWS_EVAL_SCHEMA
        or receipt.get("data_label") != "SYNTHETIC"
        or receipt.get("live_pnl_claim") is not False
        or not isinstance(receipt.get("results"), list)
        or not receipt["results"]
        or not family_blob_forbidden_metrics_absent(research_blob)
    ):
        raise ValueError("news receipt violates its synthetic research contract")
    canonical = json.loads(canonical_json_bytes(dict(receipt)))
    digest = hash_bytes(canonical_json_bytes(canonical))
    payload = {**canonical, "receipt_sha256": digest}
    path = Path(receipts_dir) / f"news_eval_{digest[:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return path


def evaluate_corpus_against_bars(
    records: Iterable[NewsRecord],
    bars: pl.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience: build labels from a bars frame, then ``run_news_eval``."""
    return run_news_eval(records, bars=bars, **kwargs)

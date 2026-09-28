"""Regression tests for the research/ audit fixes — see docs/AUDIT_RESEARCH.md."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _small_panel(n_dates: int = 10, n_names: int = 2, seed: int = 0):
    start = datetime(2024, 1, 1, tzinfo=UTC)
    times = [start + timedelta(days=i) for i in range(n_dates)]
    dates = np.array([t for t in times for _ in range(n_names)], dtype=object)
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(dates.size, 3))
    y = rng.normal(size=dates.size)
    return times, dates, x, y


def _foldless_config():
    from quant_fund.config import load_config

    cfg = load_config("configs/research.yaml")
    cfg.validation.train_bars = 40
    cfg.validation.val_bars = 10
    cfg.validation.test_bars = 10
    cfg.validation.embargo_bars = 1
    return cfg


def test_oos_rank_scores_small_panel_fallback_still_purges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The no-fold chronological fallback must purge horizon labels and embargo
    like the walk-forward fold path — a boundary train label must not reach
    into the holdout."""
    from quant_fund.research.benches import ranking as ranking_bench
    from quant_fund.research.benches.ranking import oos_rank_scores
    from quant_fund.validation.walk_forward import session_index

    cfg = _foldless_config()
    times, dates, x, y = _small_panel()
    horizon = 2

    fitted_dates: list[np.ndarray] = []
    orig_fit = ranking_bench._fit_ranker

    def spy(
        model: Any, name: str, fx: Any, fy: Any, fdates: Any, ids: Any = None, features: Any = None
    ) -> None:
        fitted_dates.append(np.asarray(fdates))
        orig_fit(model, name, fx, fy, fdates, ids, features)

    monkeypatch.setattr(ranking_bench, "_fit_ranker", spy)
    pred = oos_rank_scores("ridge", cfg, x, y, dates, horizon_bars=horizon)

    idx = session_index(times)
    cut = max(len(times) - 40, len(times) // 2)
    expected_train = {
        t for t in times[:cut] if idx[t] + horizon < cut and idx[t] + cfg.embargo_bars() < cut
    }
    assert len(fitted_dates) == 1
    assert expected_train
    assert set(fitted_dates[0].tolist()) == expected_train

    holdout = set(times[cut:])
    in_holdout = np.array([d in holdout for d in dates])
    assert np.isfinite(pred[in_holdout]).all()
    assert not np.isfinite(pred[~in_holdout]).any()


def test_oos_rank_scores_fallback_fails_closed_when_purge_empties_train(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the boundary purge leaves no train dates the fallback must return
    labeled NaN, never fit on overlapping labels."""
    from quant_fund.research.benches import ranking as ranking_bench
    from quant_fund.research.benches.ranking import oos_rank_scores

    cfg = _foldless_config()
    _, dates, x, y = _small_panel()

    fitted: int = 0
    orig_fit = ranking_bench._fit_ranker

    def spy(
        model: Any, name: str, fx: Any, fy: Any, fdates: Any, ids: Any = None, features: Any = None
    ) -> None:
        nonlocal fitted
        fitted += 1
        orig_fit(model, name, fx, fy, fdates, ids, features)

    monkeypatch.setattr(ranking_bench, "_fit_ranker", spy)
    pred = oos_rank_scores("ridge", cfg, x, y, dates, horizon_bars=20)
    assert fitted == 0
    assert np.isnan(pred).all()


@pytest.mark.synthetic
def test_sota_receipt_is_sealed_and_self_verifying(tmp_path: Path) -> None:
    """sota_receipt.json must carry a receipt_sha256 that re-derives from the
    sealed payload, be written atomically, and pass verify_receipt_file."""
    from quant_fund.config import load_config
    from quant_fund.pipeline.dataset import build_gold, panel
    from quant_fund.research.receipt_v2 import verify_receipt_file
    from quant_fund.research.sota_protocol import load_sota_protocol, run_sota_protocol
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

    cfg = load_config("configs/sota_g1.yaml")
    cfg.data.root = tmp_path
    cfg.data.synthetic_n_assets = 8
    cfg.data.synthetic_n_days = 90
    cfg.universe.min_history_bars = 5
    cfg.universe.min_adv = 0.0
    cfg.validation.train_bars = 40
    cfg.validation.val_bars = 10
    cfg.validation.test_bars = 10
    proto = load_sota_protocol("configs/sota_protocol.yaml").model_copy(
        update={"n_asofs": 3, "lookback": 12, "sample_count": 2, "pred_len": 5}
    )
    build_gold(cfg)
    run_sota_protocol(
        cfg,
        proto,
        panel(cfg),
        include_torch=False,
        include_path_rankic=True,
        include_calibration=True,
    )

    path = tmp_path / "metadata" / "sota_receipt.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    assert payload["receipt_sha256"] == hash_bytes(canonical_json_bytes(body))
    # Atomic publish leaves no staging files behind.
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))
    result = verify_receipt_file(path)
    assert result["errors"] == []
    assert result["valid"] is True

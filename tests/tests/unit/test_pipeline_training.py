"""Training pipeline helpers + train_family fail-closed edges."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from quant_fund.config import load_config
from quant_fund.config.models import ValidationConfig
from quant_fund.pipeline.train import (
    _chronological_split,
    _label_horizon,
    _make_ranker,
    _walk_forward_splits,
    train_distribution,
    train_family,
    train_ranking,
    train_regime,
    train_volatility,
)


def test_label_horizon_uses_configured_forward_target() -> None:
    assert _label_horizon("future_excess_return_20") == 20
    assert _label_horizon("custom_target") == 5


def test_chronological_split_is_date_level_and_purges_forward_labels() -> None:
    unique_dates = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(10)]
    dates = np.repeat(np.array(unique_dates, dtype=object), 2)

    train_mask, test_mask = _chronological_split(
        dates,
        train_fraction=0.7,
        horizon_bars=2,
        embargo_bars=1,
    )

    train_dates = set(dates[train_mask])
    test_dates = set(dates[test_mask])
    assert train_dates.isdisjoint(test_dates)
    assert train_dates == set(unique_dates[:4])
    assert test_dates == set(unique_dates[7:])


def test_walk_forward_splits_are_date_level_purged_and_cover_all_test_blocks() -> None:
    unique_dates = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(20)]
    dates = np.repeat(np.array(unique_dates, dtype=object), 3)
    config = ValidationConfig(train_bars=6, val_bars=2, test_bars=2, scheme="expanding")

    splits = _walk_forward_splits(dates, config, horizon_bars=2)

    assert len(splits) == 3
    test_blocks = [set(dates[test]) for _, test in splits]
    assert len(set().union(*test_blocks)) == 6
    assert all(
        left.isdisjoint(right)
        for i, left in enumerate(test_blocks)
        for right in test_blocks[i + 1 :]
    )
    for train, test in splits:
        assert set(dates[train]).isdisjoint(set(dates[test]))
        train_indices = [unique_dates.index(value) for value in set(dates[train])]
        test_indices = [unique_dates.index(value) for value in set(dates[test])]
        assert max(train_indices) + 2 < min(test_indices)


def test_short_walk_forward_uses_explicit_sparse_label_endpoints() -> None:
    unique_dates = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(8)]
    dates = np.repeat(np.array(unique_dates, dtype=object), 2)
    endpoints = np.repeat(
        np.array(
            [
                unique_dates[4],
                unique_dates[2],
                unique_dates[3],
                unique_dates[5],
                unique_dates[5],
                unique_dates[6],
                unique_dates[7],
                unique_dates[7],
            ],
            dtype=object,
        ),
        2,
    )
    config = ValidationConfig(
        train_bars=6,
        val_bars=2,
        test_bars=2,
        scheme="expanding",
        embargo_bars=0,
    )

    splits = _walk_forward_splits(
        dates,
        config,
        horizon_bars=2,
        label_end_times=endpoints,
    )

    assert len(splits) == 1
    train, test = splits[0]
    assert set(dates[train]) == {unique_dates[1], unique_dates[2]}
    assert set(dates[test]) == set(unique_dates[4:])


def test_chronological_split_empty_or_single_date_fail_closed() -> None:
    with pytest.raises(ValueError, match="at least 2 unique dates"):
        _chronological_split(np.array([], dtype=object))
    with pytest.raises(ValueError, match="at least 2 unique dates"):
        _chronological_split(np.array([datetime(2020, 1, 1, tzinfo=UTC)], dtype=object))


def test_chronological_split_bad_fraction_and_horizon() -> None:
    dates = np.array(
        [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(5)],
        dtype=object,
    )
    with pytest.raises(ValueError, match="train_fraction"):
        _chronological_split(dates, train_fraction=0.0)
    with pytest.raises(ValueError, match="train_fraction"):
        _chronological_split(dates, train_fraction=1.0)
    with pytest.raises(ValueError, match="horizon_bars"):
        _chronological_split(dates, horizon_bars=-1)


def test_train_family_unknown_family_fail_closed() -> None:
    cfg = load_config("configs/research.yaml")
    with pytest.raises(ValueError, match="unknown family"):
        train_family(cfg, "not_a_real_family")


def test_make_ranker_unknown_model_fail_closed() -> None:
    cfg = load_config("configs/research.yaml")
    with pytest.raises(ValueError, match="unknown ranking model"):
        _make_ranker("not_a_ranker", cfg)


def test_train_family_unknown_models_fail_closed() -> None:
    cfg = load_config("configs/research.yaml")
    with pytest.raises(ValueError, match="unknown ranking model"):
        train_family(cfg, "ranking", "bogus_ranker")
    with pytest.raises(ValueError, match="unknown distribution model"):
        train_family(cfg, "distribution", "bogus_dist")
    with pytest.raises(ValueError, match="unknown volatility model"):
        train_family(cfg, "volatility", "bogus_vol")
    with pytest.raises(ValueError, match="unknown regime model"):
        train_family(cfg, "regime", "bogus_regime")
    with pytest.raises(ValueError, match="unknown tail model"):
        train_family(cfg, "tail", "bogus_tail")


def test_train_ranking_empty_panel_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config("configs/research.yaml")
    empty = pl.DataFrame(
        schema={
            "event_time": pl.Datetime("us", "UTC"),
            "security_id": pl.Utf8,
            cfg.train.ranking_target: pl.Float64,
            "ret_1": pl.Float64,
        }
    )
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: empty)
    with pytest.raises(ValueError, match="non-empty frame"):
        train_ranking(cfg, "ridge")


def test_train_ranking_no_evaluable_fold_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config("configs/research.yaml")
    label = cfg.train.ranking_target
    # Non-empty panel so design_matrix succeeds, but zero walk-forward folds.
    frame = pl.DataFrame(
        {
            "event_time": [datetime(2020, 1, 1, tzinfo=UTC)] * 4,
            "security_id": [f"A{i}" for i in range(4)],
            label: [0.1, -0.2, 0.05, 0.0],
            "ret_1": [0.01, -0.01, 0.0, 0.02],
            "mom_20": [0.1, 0.2, 0.0, -0.1],
            "vol_20": [0.2, 0.2, 0.2, 0.2],
            "reversal_1": [0.0, 0.1, -0.1, 0.0],
            "amihud": [1e-6] * 4,
        }
    )
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: frame)
    monkeypatch.setattr(
        "quant_fund.pipeline.train._walk_forward_splits",
        lambda *a, **k: [],
    )
    with pytest.raises(ValueError, match="no trainable/evaluable fold"):
        train_ranking(cfg, "ridge")


def test_train_distribution_empty_panel_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config("configs/research.yaml")
    empty = pl.DataFrame(
        schema={
            "event_time": pl.Datetime("us", "UTC"),
            "security_id": pl.Utf8,
            cfg.train.distribution_target: pl.Float64,
            "ret_1": pl.Float64,
        }
    )
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: empty)
    with pytest.raises(ValueError, match="non-empty frame"):
        train_distribution(cfg, "gaussian")


def test_train_volatility_empty_panel_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config("configs/research.yaml")
    empty = pl.DataFrame(
        schema={
            "event_time": pl.Datetime("us", "UTC"),
            "security_id": pl.Utf8,
            cfg.train.volatility_target: pl.Float64,
            "vol_20": pl.Float64,
        }
    )
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: empty)
    with pytest.raises(ValueError, match="non-empty frame"):
        train_volatility(cfg, "ewma")


def test_train_regime_empty_panel_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config("configs/research.yaml")
    empty = pl.DataFrame(
        schema={
            "event_time": pl.Datetime("us", "UTC"),
            "mkt_ret_1": pl.Float64,
            "mkt_vol_20": pl.Float64,
            "cs_dispersion": pl.Float64,
            "breadth": pl.Float64,
        }
    )
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: empty)
    # unique/drop_nulls on empty → empty x/dates; walk-forward then fails closed for hmm.
    with pytest.raises(
        ValueError, match="no trainable/evaluable fold|at least|empty|chronological"
    ):
        train_regime(cfg, "hmm")


def test_train_family_passthrough_families_do_not_train() -> None:
    cfg = load_config("configs/research.yaml")
    cov = train_family(cfg, "covariance")
    liq = train_family(cfg, "liquidity")
    assert "note" in cov and "metrics" in cov
    assert "note" in liq and "metrics" in liq
    # No live claim keys.
    assert cov.get("live_pnl_claim") is not True
    assert liq.get("live_pnl_claim") is not True

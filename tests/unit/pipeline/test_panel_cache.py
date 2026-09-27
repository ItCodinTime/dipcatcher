"""Phase 18 / Wave 28 panel cache + design_matrix edges."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from quant_fund.config.models import AppConfig
from quant_fund.features.metadata import FEATURE_SET_VERSION
from quant_fund.pipeline import dataset as dataset_module
from quant_fund.pipeline.dataset import (
    _PANEL_CACHE,
    _panel_cache_key,
    clear_panel_cache,
    design_matrix,
    panel,
)


def _write_valid_gold_lake(root) -> AppConfig:
    times = [datetime(2020, 1, 2, tzinfo=UTC), datetime(2020, 1, 3, tzinfo=UTC)]
    (root / "gold").mkdir(parents=True, exist_ok=True)
    (root / "silver").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "event_time": times,
            "available_time": times,
            "feature_set_version": [FEATURE_SET_VERSION, FEATURE_SET_VERSION],
            "ret_1": [0.1, 0.2],
        }
    ).write_parquet(root / "gold" / "features.parquet")
    pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "event_time": times,
            "future_ret_1": [0.0, 0.1],
        }
    ).write_parquet(root / "gold" / "labels.parquet")
    pl.DataFrame({"security_id": ["A", "A"], "asof": times}).write_parquet(
        root / "silver" / "universe.parquet"
    )
    return AppConfig.model_validate({"data": {"root": str(root)}})


def test_panel_cache_hit_and_clear(tmp_path):
    cfg = _write_valid_gold_lake(tmp_path)
    clear_panel_cache()
    assert len(_PANEL_CACHE) == 0
    a = panel(cfg)
    assert len(_PANEL_CACHE) >= 1
    b = panel(cfg)
    assert a is b  # same object from cache
    clear_panel_cache()
    assert len(_PANEL_CACHE) == 0
    c = panel(cfg)
    assert c is not b
    assert c.height == a.height


def _write_cache_key_artifacts(
    tmp_path, *, features: bytes, labels: bytes, universe: bytes
) -> tuple:
    feat_path = tmp_path / "features.parquet"
    lab_path = tmp_path / "labels.parquet"
    univ_dir = tmp_path / "silver"
    univ_dir.mkdir(parents=True, exist_ok=True)
    feat_path.write_bytes(features)
    lab_path.write_bytes(labels)
    (univ_dir / "universe.parquet").write_bytes(universe)
    return feat_path, lab_path


def test_panel_cache_key_changes_when_artifact_bytes_change(tmp_path) -> None:
    features, labels = _write_cache_key_artifacts(
        tmp_path, features=b"features-v1", labels=b"labels-v1", universe=b"universe-v1"
    )
    first = _panel_cache_key(tmp_path, features, labels)
    assert first is not None
    features.write_bytes(b"features-v2")
    second = _panel_cache_key(tmp_path, features, labels)
    assert second is not None
    assert first[0] == second[0]
    assert first[1] != second[1]
    assert first[2] == second[2]
    assert first[3] == second[3]


def test_panel_cache_key_changes_when_universe_bytes_change(tmp_path) -> None:
    features, labels = _write_cache_key_artifacts(
        tmp_path, features=b"features-v1", labels=b"labels-v1", universe=b"universe-v1"
    )
    first = _panel_cache_key(tmp_path, features, labels)
    assert first is not None
    (tmp_path / "silver" / "universe.parquet").write_bytes(b"universe-v2")
    second = _panel_cache_key(tmp_path, features, labels)
    assert second is not None
    assert first[1] == second[1]
    assert first[2] == second[2]
    assert first[3] != second[3]


def test_panel_cache_key_missing_universe_is_uncacheable(tmp_path) -> None:
    features = tmp_path / "features.parquet"
    labels = tmp_path / "labels.parquet"
    features.write_bytes(b"features-v1")
    labels.write_bytes(b"labels-v1")
    assert _panel_cache_key(tmp_path, features, labels) is None


def test_panel_cache_key_reuses_digest_until_file_stat_changes(tmp_path, monkeypatch) -> None:
    features, labels = _write_cache_key_artifacts(
        tmp_path, features=b"features-v1", labels=b"labels-v1", universe=b"universe-v1"
    )
    universe = tmp_path / "silver" / "universe.parquet"
    real_hash_file = dataset_module.hash_file
    calls: list[str] = []

    def counting_hash_file(path):
        calls.append(str(path))
        return real_hash_file(path)

    monkeypatch.setattr(dataset_module, "hash_file", counting_hash_file)
    first = _panel_cache_key(tmp_path, features, labels)
    second = _panel_cache_key(tmp_path, features, labels)
    assert first == second
    assert calls == [str(features.resolve()), str(labels.resolve()), str(universe.resolve())]

    features.write_bytes(b"features-v2")
    third = _panel_cache_key(tmp_path, features, labels)
    assert third is not None
    assert third[1] != first[1]
    assert calls.count(str(features.resolve())) == 2
    assert calls.count(str(labels.resolve())) == 1
    assert calls.count(str(universe.resolve())) == 1


def test_panel_cache_miss_when_feature_names_requested(tmp_path) -> None:
    """Explicit feature_names bypasses the default-key cache path."""
    cfg = _write_valid_gold_lake(tmp_path)
    clear_panel_cache()
    full = panel(cfg)
    assert len(_PANEL_CACHE) >= 1
    cache_size = len(_PANEL_CACHE)
    # ret_1 exists on gold panel; request must not grow/replace default cache entry
    subset = panel(cfg, feature_names=["ret_1"])
    assert len(_PANEL_CACHE) == cache_size
    assert subset.height == full.height
    assert "ret_1" in subset.columns


def test_panel_missing_feature_columns_fail_closed(tmp_path) -> None:
    cfg = _write_valid_gold_lake(tmp_path)
    clear_panel_cache()
    with pytest.raises(ValueError, match="requested feature columns missing"):
        panel(cfg, feature_names=["definitely_not_a_feature_zzz"])


def test_panel_missing_label_fail_closed(tmp_path) -> None:
    cfg = _write_valid_gold_lake(tmp_path)
    clear_panel_cache()
    with pytest.raises(ValueError, match="requested label"):
        panel(cfg, label="future_not_a_real_label_zzz")


def test_design_matrix_empty_frame_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        design_matrix(pl.DataFrame(), "future_idio_return_1", ["ret_1"])


def test_design_matrix_missing_label_fail_closed() -> None:
    frame = pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 2)],
            "security_id": ["A"],
            "ret_1": [0.01],
        }
    )
    with pytest.raises(ValueError, match="requested label"):
        design_matrix(frame, "future_idio_return_1", ["ret_1"])


def test_design_matrix_explicit_missing_features_fail_closed() -> None:
    frame = pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 2)],
            "security_id": ["A"],
            "future_idio_return_1": [0.02],
            "ret_1": [0.01],
        }
    )
    with pytest.raises(ValueError, match="none of the requested feature columns"):
        design_matrix(frame, "future_idio_return_1", ["nope_feature_a", "nope_feature_b"])


def test_design_matrix_all_null_rows_return_empty() -> None:
    """All-null rows drop to empty arrays (honest empty — not invented samples)."""
    frame = pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 2)],
            "security_id": ["A"],
            "future_idio_return_1": [None],
            "ret_1": [None],
        }
    )
    x, y, dates, feats, ids = design_matrix(frame, "future_idio_return_1", ["ret_1"])
    assert feats == ["ret_1"]
    assert x.shape == (0, 1)
    assert y.shape == (0,)
    assert dates.shape == (0,)
    assert ids.shape == (0,)


def test_design_matrix_happy_path_shapes() -> None:
    frame = pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "security_id": ["A", "A"],
            "future_idio_return_1": [0.01, 0.02],
            "ret_1": [0.0, 0.01],
        }
    )
    x, y, dates, feats, ids = design_matrix(frame, "future_idio_return_1", ["ret_1"])
    assert feats == ["ret_1"]
    assert x.shape == (2, 1)
    assert y.shape == (2,)
    assert dates.shape == (2,)
    assert ids.shape == (2,)

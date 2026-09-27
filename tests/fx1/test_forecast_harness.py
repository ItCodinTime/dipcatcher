"""Offline tests for the fx-1 forecast harness. No network. Reference models only."""

from __future__ import annotations

import json
import pickle
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from fx1.cli import app
from fx1.forecast.artifacts import (
    ArtifactBackendUnavailable,
    UntrustedArtifactError,
    load_artifact,
    probe_artifact,
)
from fx1.forecast.config import Fx1HarnessConfig, load_harness_config
from fx1.forecast.features import OhlcvFeaturePipeline, resample_ohlcv
from fx1.forecast.protocol import ForecastModel
from fx1.forecast.registry import ModelNotRegistered, create_model
from fx1.forecast.runner import run_inference, run_signal_evaluation
from fx1.forecast.schema import SchemaError, validate_feature_schema, validate_forecast_schema
from fx1.forecast.signals import PLACEHOLDER_NOT_A_STRATEGY, map_signals
from quant_fund.schemas.errors import LeakageError
from quant_fund.utils.hashing import hash_file

runner = CliRunner()


def _stamp(offset: int) -> datetime:
    return datetime(2020, 1, 1, 16, tzinfo=UTC) + timedelta(days=offset)


def _panel(
    n: int = 24,
    names: tuple[str, ...] = ("AAA", "BBB", "CCC"),
    *,
    late: dict[int, int] | None = None,
    source: str = "fixture",
) -> pl.DataFrame:
    """Deterministic daily bars. ``late`` maps a bar index to an availability delay in days."""
    rows: list[dict[str, object]] = []
    delays = late or {}
    for s_i, sid in enumerate(names):
        price = 40.0 + 5.0 * s_i
        for i in range(n):
            price *= 1.01 if (i + s_i) % 2 == 0 else 0.99
            event = _stamp(i)
            available = event + timedelta(days=delays.get(i, 0))
            rows.append(
                {
                    "security_id": sid,
                    "event_time": event,
                    "available_time": available,
                    "ingested_time": available,
                    "source": source,
                    "revision_id": "v1",
                    "open": price * 0.99,
                    "high": price * 1.02,
                    "low": price * 0.98,
                    "close": price,
                    "volume": 1_000.0 + i,
                    "planted_signal": 1.0 if i % 2 == 0 else -1.0,
                }
            )
    return pl.DataFrame(rows)


class _MemoryProvider:
    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame

    def get_bars(self, start=None, end=None, security_ids=None) -> pl.DataFrame:
        frame = self.frame
        if start is not None:
            frame = frame.filter(pl.col("event_time") >= start)
        if end is not None:
            frame = frame.filter(pl.col("event_time") <= end)
        if security_ids:
            frame = frame.filter(pl.col("security_id").is_in(list(security_ids)))
        return frame


class SpyModel(ForecastModel):
    name = "spy"
    version = "test-only"

    def __init__(self) -> None:
        self.horizons: list[datetime] = []
        self.seen: list[set[datetime]] = []
        self.columns: list[str] = []
        self.horizon_bars = 1

    def load(self, checkpoint_path=None, config=None) -> None:
        self.horizon_bars = int((config or {}).get("horizon_bars", 1))

    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        self.horizons.append(features["event_time"].max())
        self.seen.append(set(features["event_time"].to_list()))
        self.columns = list(features.columns)
        n = features.height
        return pl.DataFrame(
            {
                "event_time": features["event_time"],
                "security_id": features["security_id"],
                "horizon_bars": pl.Series([self.horizon_bars] * n, dtype=pl.Int64),
                "predicted_return": pl.Series([0.0] * n, dtype=pl.Float64),
                "confidence": pl.Series([0.5] * n, dtype=pl.Float64),
            }
        )


def _harness_config(tmp_path: Path, **overrides: object) -> Fx1HarnessConfig:
    payload: dict[str, object] = {
        "schema": "fx1.harness.config/v1",
        "data": {
            "provider": "entrypoint",
            "entrypoint": "fx1.forecast.dummy:ZeroForecastModel",
            "symbols": [],
        },
        "features": {"lookbacks": [1, 5], "vol_window": 5, "horizon_bars": 1},
        "model": {"name": "dummy-zero"},
        "inference": {
            "mode": "walk_forward",
            "output_parquet": str(tmp_path / "forecasts.parquet"),
            "output_meta": str(tmp_path / "forecasts.meta.json"),
        },
        "signal": {"mapping": "sign", "threshold": 0.0, "cost_bps": 5.0},
        "walk_forward": {
            "scheme": "expanding",
            "train_bars": 8,
            "val_bars": 3,
            "test_bars": 3,
            "embargo_bars": 1,
        },
    }
    for key, value in overrides.items():
        section = payload[key]
        assert isinstance(section, dict)
        section.update(value)  # type: ignore[arg-type]
    return Fx1HarnessConfig.model_validate(payload)


def test_fx1_name_is_not_implemented():
    with pytest.raises(ModelNotRegistered, match="not implemented"):
        create_model("fx-1")


def test_fx1_entrypoint_must_subclass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = tmp_path / "toy_fx1.py"
    module.write_text(
        "from fx1.forecast.protocol import ForecastModel, Fx1Model\n"
        "import polars as pl\n"
        "class NotFx(ForecastModel):\n"
        "    name = 'other'\n"
        "    version = 'x'\n"
        "    def load(self, checkpoint_path=None, config=None):\n"
        "        return None\n"
        "    def predict(self, features):\n"
        "        return features\n"
        "class Toy(Fx1Model):\n"
        "    version = 'toy'\n"
        "    def load(self, checkpoint_path=None, config=None):\n"
        "        self.version = 'toy'\n"
        "        self.horizon_bars = int((config or {}).get('horizon_bars', 1))\n"
        "    def predict(self, features):\n"
        "        n = features.height\n"
        "        return pl.DataFrame({\n"
        "            'event_time': features['event_time'],\n"
        "            'security_id': features['security_id'],\n"
        "            'horizon_bars': pl.Series([self.horizon_bars] * n, dtype=pl.Int64),\n"
        "            'predicted_return': pl.Series([0.01] * n),\n"
        "        })\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(TypeError, match="subclass"):
        create_model("fx-1", entrypoint="toy_fx1:NotFx")
    model = create_model("fx-1", entrypoint="toy_fx1:Toy")
    assert model.name == "fx-1"
    assert type(model).__name__ == "Toy"


def test_checkpoint_hash_and_version_stamp(tmp_path: Path):
    path = tmp_path / "reference.pkl"
    with path.open("wb") as handle:
        pickle.dump({"version": "from-payload", "weights": [1.0, 0.0]}, handle)
    with pytest.raises(UntrustedArtifactError, match="allow_unsafe_deserialization"):
        load_artifact(path)
    trusted_digest = hash_file(path)
    with pytest.raises(UntrustedArtifactError, match="does not match"):
        load_artifact(
            path,
            allow_unsafe_deserialization=True,
            trusted_checkpoint_sha256="0" * 64,
        )
    loaded = load_artifact(
        path,
        allow_unsafe_deserialization=True,
        trusted_checkpoint_sha256=trusted_digest,
    )
    assert loaded.format == "pickle"
    assert loaded.sha256 == hash_file(path)
    assert loaded.version == "from-payload"
    sidecar = Path(str(path) + ".version")
    sidecar.write_text("from-sidecar\n", encoding="utf-8")
    stamped = load_artifact(
        path,
        allow_unsafe_deserialization=True,
        trusted_checkpoint_sha256=trusted_digest,
    )
    assert stamped.version == "from-sidecar"
    probed = probe_artifact(path)
    assert probed["sha256"] == loaded.sha256
    assert probed["version"] == "from-sidecar"


def test_checkpoint_trust_policy_prevents_unapproved_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "candidate.pkl"
    path.write_bytes(b"arbitrary checkpoint bytes")

    def _must_not_run(_raw: bytes) -> object:
        raise AssertionError("unsafe loader ran without a trusted digest")

    monkeypatch.setattr("fx1.forecast.artifacts._load_pickle", _must_not_run)
    with pytest.raises(UntrustedArtifactError, match="allow_unsafe_deserialization"):
        load_artifact(path)
    with pytest.raises(UntrustedArtifactError, match="does not match"):
        load_artifact(
            path,
            allow_unsafe_deserialization=True,
            trusted_checkpoint_sha256="0" * 64,
        )


def test_harness_rejects_untrusted_pickle_before_model_load(tmp_path: Path) -> None:
    path = tmp_path / "candidate.pkl"
    path.write_bytes(b"not a trusted checkpoint")
    with pytest.raises(ValueError, match="trusted_checkpoint_sha256"):
        _harness_config(tmp_path, model={"allow_unsafe_deserialization": True})

    config = _harness_config(
        tmp_path,
        model={"checkpoint_path": str(path), "checkpoint_format": "pickle"},
    )

    class _NoLoad(SpyModel):
        def load(self, checkpoint_path=None, config=None) -> None:
            raise AssertionError("model.load must not see an untrusted checkpoint")

    with pytest.raises(UntrustedArtifactError, match="allow_unsafe_deserialization"):
        run_inference(config, model=_NoLoad(), provider=_MemoryProvider(_panel(20)))


def test_torch_checkpoint_uses_weights_only_without_trust_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "candidate.pt"
    path.write_bytes(b"torch checkpoint stub")
    calls: list[tuple[bool, bool]] = []

    class _TorchStub:
        def load(self, source: object, *, map_location: str, weights_only: bool) -> object:
            assert map_location == "cpu"
            calls.append((weights_only, isinstance(source, Path)))
            return {"version": "test"}

    monkeypatch.setattr("fx1.forecast.artifacts.import_optional", lambda _module: _TorchStub())
    assert load_artifact(path).version == "test"
    assert (
        load_artifact(
            path,
            allow_unsafe_deserialization=True,
            trusted_checkpoint_sha256=hash_file(path),
        ).version
        == "test"
    )
    assert calls == [(True, True), (False, False)]


def test_missing_backend_does_not_import_at_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import importlib

    path = tmp_path / "model.pt"
    path.write_bytes(b"not-a-real-checkpoint")
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"not-a-real-model")
    real_import = importlib.import_module

    def _blocked(module: str, *args: object, **kwargs: object):
        if module in {"torch", "onnx"}:
            raise ImportError(module)
        return real_import(module, *args, **kwargs)

    monkeypatch.setattr("fx1.forecast.artifacts.importlib.import_module", _blocked)
    with pytest.raises(ArtifactBackendUnavailable, match="torch"):
        load_artifact(path, fmt="torch")
    with pytest.raises(ArtifactBackendUnavailable, match="onnx"):
        load_artifact(onnx_path, fmt="onnx")


def test_feature_schema_rejects_lookahead_and_unknown_columns():
    pipeline = OhlcvFeaturePipeline([1], vol_window=5)
    features = pipeline.build(_panel(20, ("AAA", "BBB")))
    assert "planted_signal" not in features.columns
    assert "realized_return" not in features.columns
    dirty = features.with_columns(pl.lit(1.0).alias("target_return"))
    with pytest.raises(LeakageError, match="lookahead"):
        validate_feature_schema(dirty, pipeline.feature_columns())
    extra = features.with_columns(pl.lit(1).alias("surprise"))
    with pytest.raises(SchemaError, match="allow-list"):
        validate_feature_schema(extra, pipeline.feature_columns())


def test_forecast_schema_quantiles_confidence_and_target():
    stamp = _stamp(0)
    base = {
        "event_time": [stamp],
        "security_id": ["AAA"],
        "horizon_bars": [1],
        "predicted_return": [0.1],
    }
    validate_forecast_schema(
        pl.DataFrame({**base, "q_0.1": [0.0], "q_0.9": [0.2], "confidence": [0.4]})
    )
    with pytest.raises(SchemaError, match="nondecreasing"):
        validate_forecast_schema(pl.DataFrame({**base, "q_0.1": [0.4], "q_0.9": [0.1]}))
    with pytest.raises(SchemaError, match="confidence"):
        validate_forecast_schema(pl.DataFrame({**base, "confidence": [1.5]}))
    with pytest.raises(SchemaError, match="predicted_return and/or"):
        validate_forecast_schema(
            pl.DataFrame({"event_time": [stamp], "security_id": ["AAA"], "horizon_bars": [1]})
        )
    with pytest.raises(LeakageError):
        validate_forecast_schema(pl.DataFrame({**base, "realized_return": [0.2]}))


def test_features_ignore_future_bars_and_late_releases():
    pipeline = OhlcvFeaturePipeline([1, 5], vol_window=5)
    bars = _panel(16, ("AAA",))
    decision = _stamp(10)
    original = pipeline.build(bars, decision_time=decision)
    future = bars.with_columns(
        pl.when(pl.col("event_time") > decision)
        .then(pl.col("close") * 10.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    rebuilt = pipeline.build(future, decision_time=decision)
    assert rebuilt.equals(original)

    past = bars.with_columns(
        pl.when(pl.col("event_time") == _stamp(9))
        .then(pl.col("close") * 3.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    changed = pipeline.build(past, decision_time=decision)
    row = original.filter(pl.col("event_time") == decision)
    other = changed.filter(pl.col("event_time") == decision)
    assert row["ret_1"][0] != other["ret_1"][0]

    base = _panel(12, ("AAA",))
    delayed = base.with_columns(
        pl.when(pl.col("event_time") == _stamp(2))
        .then(pl.col("close") * 5.0)
        .otherwise(pl.col("close"))
        .alias("close"),
        pl.when(pl.col("event_time") == _stamp(2))
        .then(pl.lit(_stamp(6)).cast(pl.Datetime("us", "UTC")))
        .otherwise(pl.col("available_time"))
        .alias("available_time"),
    )
    removed = base.filter(pl.col("event_time") != _stamp(2))
    assert pipeline.build(delayed, decision_time=_stamp(5)).equals(
        pipeline.build(removed, decision_time=_stamp(5))
    )
    released = pipeline.build(delayed, decision_time=_stamp(6))
    withheld = pipeline.build(removed, decision_time=_stamp(6))
    assert (
        released.filter(pl.col("event_time") == _stamp(6))["vol_5"][0]
        != withheld.filter(pl.col("event_time") == _stamp(6))["vol_5"][0]
    )


def test_forward_target_uses_t_plus_h_and_is_not_a_feature():
    from fx1.forecast.evaluate import forward_simple_returns

    bars = _panel(8, ("AAA",))
    labels = forward_simple_returns(bars, 2)
    row = labels.filter(pl.col("event_time") == _stamp(3)).row(0, named=True)
    close_t = bars.filter(pl.col("event_time") == _stamp(3))["close"][0]
    close_h = bars.filter(pl.col("event_time") == _stamp(5))["close"][0]
    assert row["realized_return"] == pytest.approx(close_h / close_t - 1.0)
    assert row["target_time"] == _stamp(5)
    features = OhlcvFeaturePipeline([1], vol_window=5).build(bars, decision_time=_stamp(5))
    assert "realized_return" not in features.columns


def test_resample_bucket_excludes_the_next_day():
    rows = []
    day = datetime(2020, 1, 2, tzinfo=UTC)
    nxt = datetime(2020, 1, 3, tzinfo=UTC)
    for hour, close in ((10, 10.0), (14, 12.0)):
        stamp = day.replace(hour=hour)
        rows.append(
            {
                "security_id": "AAA",
                "event_time": stamp,
                "available_time": stamp,
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 5.0,
                "source": "fixture",
            }
        )
    later = nxt.replace(hour=11)
    rows.append(
        {
            "security_id": "AAA",
            "event_time": later,
            "available_time": later,
            "open": 99.0,
            "high": 100.0,
            "low": 98.0,
            "close": 99.0,
            "volume": 7.0,
            "source": "fixture",
        }
    )
    out = resample_ohlcv(pl.DataFrame(rows), "1d").sort("event_time")
    first = out.row(0, named=True)
    assert first["close"] == 12.0
    assert first["open"] == 10.0
    assert first["high"] == 13.0
    assert first["volume"] == 10.0
    assert first["event_time"] == day.replace(hour=14)
    assert out.row(1, named=True)["close"] == 99.0


def test_walk_forward_predict_cannot_see_the_future(tmp_path: Path):
    spy = SpyModel()
    bars = _panel(18, ("AAA", "BBB"))
    config = _harness_config(
        tmp_path, features={"lookbacks": [1], "vol_window": 5, "horizon_bars": 1}
    )
    result = run_inference(config, model=spy, provider=_MemoryProvider(bars))
    assert result.metadata["model_role"] == "external"
    assert result.metadata["research_only"] is True
    assert result.metadata["live_pnl_claim"] is False
    assert "planted_signal" not in spy.columns
    assert "realized_return" not in spy.columns
    assert spy.horizons
    assert spy.horizons == sorted(spy.horizons)
    # Each call is truncated at a decision time that exists in the forecast.
    forecast_times = set(result.forecasts["event_time"].to_list())
    assert set(spy.horizons) == forecast_times
    for stamp in spy.horizons:
        assert stamp <= max(forecast_times)


def test_walk_forward_hides_late_releases_and_batch_refuses_them(tmp_path: Path):
    spy = SpyModel()
    bars = _panel(12, ("AAA",), late={3: 4})
    config = _harness_config(
        tmp_path, features={"lookbacks": [1], "vol_window": 4, "horizon_bars": 1}
    )
    run_inference(config, model=spy, provider=_MemoryProvider(bars))
    late_event = _stamp(3)
    release = _stamp(7)
    assert spy.seen
    assert any(decision < release for decision in spy.horizons)
    for decision, seen in zip(spy.horizons, spy.seen, strict=True):
        assert max(seen) == decision
        if decision < release:
            assert late_event not in seen
    batch = _harness_config(
        tmp_path,
        inference={
            "mode": "batch",
            "output_parquet": str(tmp_path / "batch.parquet"),
            "output_meta": str(tmp_path / "batch.json"),
        },
    )
    with pytest.raises(LeakageError, match="batch mode"):
        run_inference(batch, model=SpyModel(), provider=_MemoryProvider(bars))


def test_placeholder_signal_mappings_are_marked():
    stamp = _stamp(0)
    frame = pl.DataFrame(
        {
            "event_time": [stamp, stamp, stamp],
            "security_id": ["A", "B", "C"],
            "score": [0.2, -0.1, 0.0],
        }
    )
    signed = map_signals(frame, "sign")
    assert signed["mapping_role"].unique().to_list() == [PLACEHOLDER_NOT_A_STRATEGY]
    assert signed.sort("security_id")["signal"].to_list() == [1.0, -1.0, 0.0]
    gated = map_signals(frame, "threshold", threshold=0.15)
    by_id = {row["security_id"]: row["signal"] for row in gated.iter_rows(named=True)}
    assert by_id == {"A": 1.0, "B": 0.0, "C": 0.0}
    ranked = map_signals(frame, "rank")
    assert ranked["signal"].min() >= -1.0
    assert ranked["signal"].max() <= 1.0
    assert ranked.filter(pl.col("security_id") == "A")["signal"][0] == pytest.approx(1.0)


def test_end_to_end_reference_model_is_not_fx1(tmp_path: Path):
    config = _harness_config(
        tmp_path,
        data={
            "provider": "synthetic",
            "entrypoint": None,
            "synthetic_n_assets": 3,
            "synthetic_n_days": 48,
            "synthetic_seed": 11,
        },
        model={"name": "dummy-momentum"},
        features={"lookbacks": [1, 5], "vol_window": 5, "horizon_bars": 1},
        walk_forward={
            "train_bars": 12,
            "val_bars": 4,
            "test_bars": 4,
            "embargo_bars": 1,
            "scheme": "expanding",
        },
    )
    inferred = run_inference(config)
    assert inferred.metadata["model_name"] == "dummy-momentum"
    assert inferred.metadata["model_role"] == "reference_not_fx1"
    assert inferred.metadata["data_label"] == "SYNTHETIC"
    assert inferred.n_rows > 0
    meta = json.loads(inferred.meta_path.read_text(encoding="utf-8"))
    assert meta["live_pnl_claim"] is False
    import pyarrow.parquet as pq

    embedded = json.loads(pq.read_schema(inferred.parquet_path).metadata[b"fx1_harness"])
    assert embedded["run_id"] == meta["run_id"]
    report = run_signal_evaluation(config, forecasts=inferred.forecasts)
    assert report["research_only"] is True
    assert report["live_pnl_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["orders_submitted"] is False
    assert report["mapping_role"] == PLACEHOLDER_NOT_A_STRATEGY
    assert report["data_label"] == "SYNTHETIC"
    horizon = report["forecast_metrics"]["by_horizon"]["1"]
    assert horizon["n"] > 0
    assert horizon["mae"] is not None
    assert horizon["rmse"] is not None
    diagnostics = report["signal_diagnostics"]
    assert diagnostics["orders_submitted"] is False
    assert diagnostics["placeholder_mapping"] == PLACEHOLDER_NOT_A_STRATEGY
    assert not {"sharpe_ratio", "total_return", "max_drawdown"} & diagnostics.keys()
    assert diagnostics["mean_one_way_cost_drag_bps"] == pytest.approx(
        diagnostics["cost_bps"] * diagnostics["mean_turnover"]
    )
    assert diagnostics["n_periods"] > 0


def test_zero_baseline_signal_diagnostic_is_flat(tmp_path: Path):
    config = _harness_config(
        tmp_path,
        data={
            "provider": "synthetic",
            "entrypoint": None,
            "synthetic_n_assets": 3,
            "synthetic_n_days": 40,
            "synthetic_seed": 3,
        },
        model={"name": "dummy-zero"},
        walk_forward={
            "train_bars": 10,
            "val_bars": 3,
            "test_bars": 3,
            "embargo_bars": 1,
            "scheme": "expanding",
        },
    )
    inferred = run_inference(config)
    report = run_signal_evaluation(config, forecasts=inferred.forecasts)
    diagnostics = report["signal_diagnostics"]
    assert diagnostics["mean_turnover"] == pytest.approx(0.0)
    assert diagnostics["mean_one_way_cost_drag_bps"] == pytest.approx(0.0)
    assert report["forecast_metrics"]["by_horizon"]["1"]["ic"] is None


def test_cli_infer_and_backtest(tmp_path: Path):
    payload = {
        "schema": "fx1.harness.config/v1",
        "data": {
            "provider": "synthetic",
            "synthetic_n_assets": 3,
            "synthetic_n_days": 40,
            "synthetic_seed": 5,
        },
        "features": {"lookbacks": [1, 5], "vol_window": 5, "horizon_bars": 1},
        "model": {"name": "dummy-zero"},
        "inference": {
            "mode": "walk_forward",
            "output_parquet": str(tmp_path / "forecasts.parquet"),
            "output_meta": str(tmp_path / "forecasts.meta.json"),
        },
        "signal": {"mapping": "rank", "threshold": 0.0, "cost_bps": 1.0},
        "walk_forward": {
            "scheme": "rolling",
            "train_bars": 10,
            "val_bars": 3,
            "test_bars": 3,
            "embargo_bars": 1,
        },
    }
    path = tmp_path / "harness.yaml"
    import yaml

    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = load_harness_config(path)
    assert loaded.model.name == "dummy-zero"
    inferred = runner.invoke(app, ["infer", "--config", str(path)])
    assert inferred.exit_code == 0, inferred.stdout + inferred.stderr
    summary = json.loads(inferred.stdout)
    assert summary["model_role"] == "reference_not_fx1"
    assert summary["data_label"] == "SYNTHETIC"
    assert summary["live_pnl_claim"] is False
    scored = runner.invoke(app, ["backtest", "--config", str(path)])
    assert scored.exit_code == 0, scored.stdout + scored.stderr
    report = json.loads(scored.stdout)
    assert report["orders_submitted"] is False
    assert report["signal_mapping"] == "rank"
    assert report["forecast_metrics"]["by_horizon"]["1"]["n"] > 0


def test_harness_doc_states_the_contract():
    text = Path("docs/fx1_harness.md").read_text(encoding="utf-8")
    for snippet in (
        "Fx1Model",
        "fx1 infer",
        "fx1 backtest",
        "PLACEHOLDER_NOT_A_STRATEGY",
        "predicted_return",
        "available_time",
    ):
        assert snippet in text
    import re

    assert re.search(r"\bsharpe\b\s*(?:of|=|:)?\s*[-+$]?\d", text, flags=re.IGNORECASE) is None

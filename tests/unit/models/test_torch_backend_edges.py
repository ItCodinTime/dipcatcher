"""robinhood+ torch backend edge paths: gating errors, predictor cache,
future-stamp edges, and the cross-section forecast matrix. No torch/Kronos
weights are loaded — every heavyweight object is stubbed."""

from __future__ import annotations

import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from quant_fund.config.models import AppConfig
from quant_fund.models.robinhood_plus import torch_backend

pytestmark = pytest.mark.synthetic


def _config(**overrides: Any) -> AppConfig:
    rh: dict[str, Any] = {"backend": "torch", "lookback": 8, "pred_len": 3, "sample_count": 2}
    rh.update(overrides.pop("robinhood_plus", {}))
    return AppConfig.model_validate(
        {
            "data": {"root": "/tmp/x", "source": "synthetic"},
            "paper": {"enable_shadow": False},
            "robinhood_plus": rh,
            **overrides,
        }
    )


class TestAvailabilityAndPaths:
    def test_torch_available_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
        assert torch_backend.torch_available() is False

    def test_local_dirs_none_for_bad_variant(self) -> None:
        assert torch_backend.local_kronos_weight_dirs("bogus") is None

    def test_local_dirs_none_when_missing(self) -> None:
        assert torch_backend.local_kronos_weight_dirs("mini") is None

    def test_ensure_kronos_on_path_with_checkout(self, tmp_path: Path) -> None:
        # third_party/kronos ships with the repo, so sys.path gains the entry.
        before = list(sys.path)
        try:
            torch_backend._ensure_kronos_on_path()
            root = Path(torch_backend.__file__).resolve().parents[4] / "third_party"
            for name in ("kronos_src", "Kronos", "kronos"):
                if (root / name / "model" / "kronos.py").is_file():
                    assert str(root / name) in sys.path
                    break
        finally:
            sys.path[:] = before


class TestLoadPredictor:
    def test_unknown_variant_rejected(self) -> None:
        with pytest.raises(torch_backend.RobinhoodPlusTorchError, match="unknown"):
            torch_backend.load_pretrained_predictor("bogus")

    def test_hub_download_refused_without_network(self) -> None:
        with pytest.raises(torch_backend.RobinhoodPlusTorchError, match="allow_network"):
            torch_backend.load_pretrained_predictor("mini")

    def test_missing_torch_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch_backend, "torch_available", lambda: False)
        with pytest.raises(torch_backend.RobinhoodPlusTorchError, match="torch"):
            torch_backend.load_pretrained_predictor(
                "mini", tokenizer_path=str(tmp_path), model_path=str(tmp_path)
            )

    def test_kronos_not_importable_raises(self, tmp_path: Path) -> None:
        # third_party/kronos exists but its deps (huggingface_hub) are absent
        # in the test env — the ImportError must wrap, not propagate.
        with pytest.raises(torch_backend.RobinhoodPlusTorchError, match="Kronos"):
            torch_backend.load_pretrained_predictor(
                "mini", tokenizer_path=str(tmp_path), model_path=str(tmp_path)
            )

    def test_happy_path_via_stub_module(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, Any] = {}

        class _Tok:
            @classmethod
            def from_pretrained(cls, path: str) -> str:
                calls["tok"] = path
                return "tok-obj"

        class _Model:
            @classmethod
            def from_pretrained(cls, path: str) -> str:
                calls["model"] = path
                return "model-obj"

        class _Predictor:
            def __init__(self, model: Any, tokenizer: Any, **kw: Any) -> None:
                calls["predictor"] = (model, tokenizer, kw)

        fake = types.ModuleType("model.kronos")
        fake.Kronos = _Model  # type: ignore[attr-defined]
        fake.KronosTokenizer = _Tok  # type: ignore[attr-defined]
        fake.KronosPredictor = _Predictor  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "model", types.ModuleType("model"))
        monkeypatch.setitem(sys.modules, "model.kronos", fake)
        torch_backend.load_pretrained_predictor(
            "mini",
            tokenizer_path=str(tmp_path),
            model_path=str(tmp_path),
            device="cpu",
            max_context=64,
        )
        assert calls["predictor"][2]["max_context"] == 64
        assert calls["predictor"][2]["device"] == "cpu"


class TestPredictorCache:
    def test_cache_hit_and_eviction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        torch_backend._PREDICTOR_CACHE.clear()
        loads: list[str] = []

        def _fake_load(variant: str, **kw: Any) -> str:
            loads.append(variant)
            return f"pred-{variant}"

        monkeypatch.setattr(torch_backend, "load_pretrained_predictor", _fake_load)
        cfg = _config()
        assert torch_backend._cached_predictor(cfg) == "pred-mini"
        assert torch_backend._cached_predictor(cfg) == "pred-mini"
        assert loads == ["mini"]
        for i in range(5):
            alt = _config(robinhood_plus={"tokenizer_path": f"/t{i}"})
            torch_backend._cached_predictor(alt)
        assert len(torch_backend._PREDICTOR_CACHE) <= 5
        torch_backend._PREDICTOR_CACHE.clear()


class TestFutureStamps:
    def test_degenerate_and_regular_deltas(self) -> None:
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        assert torch_backend._future_stamps([], 2) == [
            datetime(2020, 1, 3, tzinfo=UTC),
            datetime(2020, 1, 4, tzinfo=UTC),
        ]
        same = [t0, t0]
        out = torch_backend._future_stamps(same, 2)
        assert out[0] - same[-1] == timedelta(days=1)
        reg = [t0, t0 + timedelta(hours=4)]
        assert torch_backend._future_stamps(reg, 2)[0] == t0 + timedelta(hours=8)


class _StubPredictor:
    def __init__(self, rows: int = 3) -> None:
        self.rows = rows

    def predict(self, df: Any, **kw: Any) -> pd.DataFrame:
        cols = ["open", "high", "low", "close", "volume", "amount"]
        return pd.DataFrame(np.ones((self.rows, 6)), columns=cols)


class TestKlinePaths:
    def test_paths_stacked_per_sample(self) -> None:
        kline = np.ones((5, 6))
        times = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=d) for d in range(5)]
        paths = torch_backend._kline_paths_from_kronos(
            _StubPredictor(3),
            kline,
            times,
            pred_len=3,
            sample_count=2,
            temperature=1.0,
            top_p=0.9,
        )
        assert paths.shape == (2, 3, 6)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(torch_backend.RobinhoodPlusTorchError, match="shape"):
            torch_backend._kline_paths_from_kronos(
                _StubPredictor(2),
                np.ones((5, 6)),
                [datetime(2024, 1, 1, tzinfo=UTC)] * 5,
                pred_len=3,
                sample_count=1,
                temperature=1.0,
                top_p=0.9,
            )


def _frame(n: int = 10, sids: tuple[str, ...] = ("AAA",)) -> pl.DataFrame:
    times = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=d) for d in range(n)]
    rows = []
    for sid in sids:
        for t in times:
            rows.append(
                {
                    "security_id": sid,
                    "event_time": t,
                    "open_split_adjusted": 1.0,
                    "high_split_adjusted": 1.1,
                    "low_split_adjusted": 0.9,
                    "close_split_adjusted": 1.0,
                    "volume": 1e6,
                    "amount": 1e6,
                }
            )
    return pl.DataFrame(rows)


class TestCrossSection:
    def test_missing_kline_columns_returns_empty(self) -> None:
        assert (
            torch_backend.forecast_cross_section_torch(
                pl.DataFrame({"event_time": [], "security_id": []}),
                datetime(2024, 1, 5, tzinfo=UTC),
                _config(),
                ["AAA"],
            )
            == {}
        )

    def test_missing_event_time_raises(self) -> None:
        frame = pl.DataFrame(
            {
                "open_split_adjusted": [1.0],
                "high_split_adjusted": [1.0],
                "low_split_adjusted": [1.0],
                "close_split_adjusted": [1.0],
                "volume": [1.0],
            }
        )
        with pytest.raises(torch_backend.RobinhoodPlusTorchError, match="event_time"):
            torch_backend.forecast_cross_section_torch(
                frame, datetime(2024, 1, 5, tzinfo=UTC), _config(), ["AAA"]
            )

    def test_insufficient_history_marks_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch_backend, "_cached_predictor", lambda _c: _StubPredictor())
        frame = _frame(1)
        out = torch_backend.forecast_cross_section_torch(
            frame, datetime(2025, 1, 1, tzinfo=UTC), _config(), ["AAA"]
        )
        assert set(out) == {"AAA"}

    def test_inference_failure_wraps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Boom:
            def predict(self, *_a: Any, **_k: Any) -> Any:
                raise RuntimeError("cuda oom")

        monkeypatch.setattr(torch_backend, "_cached_predictor", lambda _c: _Boom())
        with pytest.raises(torch_backend.RobinhoodPlusTorchError, match="cuda oom"):
            torch_backend.forecast_cross_section_torch(
                _frame(12), datetime(2025, 1, 1, tzinfo=UTC), _config(), ["AAA"]
            )

    def test_happy_path_returns_forecast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch_backend, "_cached_predictor", lambda _c: _StubPredictor(3))
        out = torch_backend.forecast_cross_section_torch(
            _frame(12, ("AAA", "ZZZ")),
            datetime(2025, 1, 1, tzinfo=UTC),
            _config(),
            ["AAA", "ZZZ"],
        )
        assert set(out) == {"AAA", "ZZZ"}

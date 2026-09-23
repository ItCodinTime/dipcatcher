"""Explicit causal GARCH forecast and distribution contracts."""

from __future__ import annotations

import numpy as np
import pytest

import quant_fund.models.base as model_base
from quant_fund.models.ranking import RidgeRanker
from quant_fund.models.volatility import GARCHVol


def _returns(n: int = 240, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, n)


def test_garch_forecast_returns_decimal_variance_and_sigma() -> None:
    returns = _returns()
    model = GARCHVol().fit(np.empty((returns.size, 0)), returns, returns=returns)
    forecast = model.forecast(horizon=5)

    assert forecast["variance"].shape == (5,)
    assert forecast["sigma"].shape == (5,)
    assert np.all(np.isfinite(forecast["variance"]))
    assert np.all(forecast["variance"] > 0.0)
    assert np.allclose(forecast["sigma"] ** 2, forecast["variance"])
    assert not np.allclose(forecast["variance"], forecast["variance"][0])


def test_fit_returns_and_diagnostics_are_direct_and_serializable() -> None:
    model = GARCHVol().fit_returns(_returns())
    diagnostics = model.diagnostics()

    assert diagnostics["n_obs"] == 240
    assert diagnostics["variance_units"] == "decimal_squared"
    assert diagnostics["returns_scale"] == "decimal_returns_to_percent"
    assert diagnostics["vol"] == "garch"
    assert diagnostics["series_scope"] == "univariate_return_series"


def test_egarch_uses_a_valid_multi_step_path() -> None:
    model = GARCHVol(vol="egarch", dist="t").fit_returns(_returns())
    forecast = model.forecast(horizon=3, seed=11)

    assert forecast["variance"].shape == (3,)
    assert np.all(np.isfinite(forecast["variance"]))
    assert np.all(forecast["variance"] > 0.0)


def test_aparch_and_figarch_use_valid_multi_step_paths() -> None:
    rng = np.random.default_rng(11)
    sigma = 0.012
    returns = np.empty(400)
    for i in range(returns.size):
        shock = rng.normal(0.0, 1.0)
        returns[i] = sigma * shock
        sigma = float(np.sqrt(8e-7 + 0.07 * returns[i] ** 2 + 0.91 * sigma * sigma))
    for vol in ("aparch", "figarch"):
        model = GARCHVol(vol=vol).fit_returns(returns)
        forecast = model.forecast(horizon=3, seed=11)
        assert model.result is not None
        assert forecast["variance"].shape == (3,)
        assert np.all(np.isfinite(forecast["variance"]))
        assert np.all(forecast["variance"] > 0.0)


def test_garch_joblib_roundtrip_preserves_forecast_contract(tmp_path) -> None:
    model = GARCHVol(dist="skewt").fit_returns(_returns())
    path = tmp_path / "garch.joblib"
    model.save(path)

    restored = GARCHVol.load(path)
    original = model.forecast(horizon=3, quantiles=(0.1, 0.5, 0.9))
    replay = restored.forecast(horizon=3, quantiles=(0.1, 0.5, 0.9))

    assert restored.diagnostics() == model.diagnostics()
    assert np.allclose(replay["variance"], original["variance"])
    assert np.allclose(replay["quantiles"], original["quantiles"])


def test_joblib_save_preserves_previous_artifact_when_dump_fails(tmp_path, monkeypatch) -> None:
    model = GARCHVol().fit_returns(_returns())
    path = tmp_path / "garch.joblib"
    previous = b"previous-complete-artifact"
    path.write_bytes(previous)

    def partial_dump(_obj, destination):
        if hasattr(destination, "write"):
            destination.write(b"partial-artifact")
        else:
            destination.write_bytes(b"partial-artifact")
        raise OSError("simulated interrupted dump")

    monkeypatch.setattr(model_base.joblib, "dump", partial_dump)

    with pytest.raises(OSError, match="simulated interrupted dump"):
        model.save(path)

    assert path.read_bytes() == previous
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_joblib_save_cleans_up_when_atomic_publish_fails(tmp_path, monkeypatch) -> None:
    model = GARCHVol().fit_returns(_returns())
    path = tmp_path / "garch.joblib"
    previous = b"previous-complete-artifact"
    path.write_bytes(previous)

    def fail_replace(_source, _destination):
        raise OSError("simulated publish failure")

    monkeypatch.setattr(model_base.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated publish failure"):
        model.save(path)

    assert path.read_bytes() == previous
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_joblib_save_publishes_a_loadable_artifact_without_temp_files(tmp_path) -> None:
    model = GARCHVol().fit_returns(_returns())
    path = tmp_path / "garch.joblib"

    model.save(path)

    assert path.is_file()
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []
    restored = GARCHVol.load(path)
    assert restored.diagnostics() == model.diagnostics()


def test_joblib_save_writes_sha256_sidecar(tmp_path) -> None:
    import hashlib

    model = GARCHVol().fit_returns(_returns())
    path = tmp_path / "garch.joblib"

    model.save(path)

    assert path.with_name(f"{path.name}.sha256").read_text() == (
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n"
    )


def test_joblib_load_rejects_modified_artifact_before_deserialization(tmp_path, monkeypatch) -> None:
    model = GARCHVol().fit_returns(_returns())
    path = tmp_path / "garch.joblib"
    model.save(path)
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))

    def unexpected_load(_path):
        raise AssertionError("deserialization should not run")

    monkeypatch.setattr(model_base.joblib, "load", unexpected_load)
    with pytest.raises(ValueError, match="checksum mismatch"):
        GARCHVol.load(path)


def test_joblib_load_rejects_wrong_model_family(tmp_path) -> None:
    source = GARCHVol().fit_returns(_returns())
    path = tmp_path / "wrong-family.joblib"
    source.save(path)

    with pytest.raises(TypeError, match="expected RidgeRanker"):
        RidgeRanker.load(path)


def test_joblib_load_accepts_legacy_artifact_without_sidecar(tmp_path) -> None:
    model = GARCHVol().fit_returns(_returns())
    path = tmp_path / "garch.joblib"
    model.save(path)
    path.with_name(f"{path.name}.sha256").unlink()

    restored = GARCHVol.load(path)

    assert restored.diagnostics() == model.diagnostics()


def test_garch_forecast_is_deterministic_and_has_predictive_quantiles() -> None:
    model = GARCHVol(dist="t").fit(np.empty((240, 0)), _returns(), returns=_returns())
    first = model.forecast(horizon=3, quantiles=(0.05, 0.5, 0.95))
    second = model.forecast(horizon=3, quantiles=(0.05, 0.5, 0.95))

    assert np.allclose(first["variance"], second["variance"])
    assert first["quantiles"].shape == (3, 3)
    assert np.all(first["quantiles"][:, 0] < first["quantiles"][:, 1])
    assert np.all(first["quantiles"][:, 1] < first["quantiles"][:, 2])
    assert first["distribution"] == "t"


def test_garch_fit_accepts_explicit_returns_without_using_forward_target() -> None:
    returns = _returns()
    future_variance = np.full(returns.size, 99.0)
    model = GARCHVol().fit(
        np.empty((returns.size, 0)),
        future_variance,
        returns=returns,
    )

    assert model.n_obs == returns.size
    assert model.fit_status in {"fitted", "fallback"}
    assert model.returns_scale == "decimal_returns_to_percent"
    assert model.result is not None


def test_garch_fit_requires_explicit_returns_keyword() -> None:
    with pytest.raises(ValueError, match="returns= must be supplied explicitly"):
        GARCHVol().fit(np.empty((240, 0)), _returns())


def test_garch_rejects_invalid_forecast_horizon() -> None:
    returns = _returns()
    model = GARCHVol().fit(np.empty((240, 0)), returns, returns=returns)
    with pytest.raises(ValueError, match="horizon"):
        model.forecast(horizon=0)


def test_garch_fallback_exposes_reason() -> None:
    model = GARCHVol(dist="t", min_obs=100).fit(np.empty((5, 0)), _returns(5), returns=_returns(5))
    assert model.result is None
    assert model.fit_status == "fallback"
    assert model.fallback_reason == "insufficient_observations"
    forecast = model.forecast(horizon=2, quantiles=(0.1, 0.5, 0.9))
    assert forecast["variance"].shape == (2,)
    assert forecast["distribution"] == "normal"
    assert forecast["requested_distribution"] == "t"
    with pytest.raises(ValueError, match="successful fit"):
        model.in_sample_sigma_and_z()


def test_in_sample_sigma_and_z_is_decimal_and_standardized() -> None:
    model = GARCHVol().fit_returns(_returns())
    sigma, z = model.in_sample_sigma_and_z()
    assert sigma.shape == (240,)
    assert z.shape == (240,)
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma > 0.0)
    assert np.allclose(sigma[-1], model.last_sigma)
    assert abs(float(np.std(z, ddof=1)) - 1.0) < 0.25

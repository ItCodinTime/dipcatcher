"""Scientific contracts for the published-model forecasting comparison."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from quant_fund.research.sota_evidence import (
    aligned_inference_losses,
    coverage_summary,
    paired_effect_intervals,
    validate_bars,
)

_SPEC = importlib.util.spec_from_file_location(
    "sota_eval_contract_tests", Path(__file__).parents[2] / "scripts" / "sota_eval_kronos.py"
)
assert _SPEC is not None and _SPEC.loader is not None
evaluator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = evaluator
_SPEC.loader.exec_module(evaluator)

DAY_NS = 86_400_000_000_000


def _panel(n: int = 40, assets: int = 3):
    rng = np.random.default_rng(19)
    single = rng.uniform(0.01, 0.10, (n, 3))
    losses = np.tile(single, (assets, 1))
    pinball = np.repeat(losses[:, :, None], 3, axis=2)
    ids = np.repeat(np.arange(assets), n)
    times = np.tile(1_700_000_000_000_000_000 + np.arange(n) * DAY_NS, assets)
    return losses, pinball, ids, times


def _summary(assets: int):
    losses, pinball, ids, times = _panel(assets=assets)
    return evaluator.summarize(
        losses,
        pinball,
        ids,
        [f"A{i}" for i in range(assets)],
        ["target", "challenger_a", "challenger_b"],
        ["target"],
        seed=7,
        n_boot=99,
        target_time_ns=times,
        bar_interval_ns=DAY_NS,
    )


def test_contemporaneous_assets_do_not_multiply_independent_observations():
    one, three = _summary(1), _summary(3)
    assert three["n_rows"] == 120
    assert three["inference"]["n_observations"] == 40
    assert three["diebold_mariano"]["target"]["challenger_a"]["n"] == 40
    for model in ("challenger_a", "challenger_b"):
        np.testing.assert_allclose(
            one["diebold_mariano"]["target"][model]["p_value"],
            three["diebold_mariano"]["target"][model]["p_value"],
            atol=1e-12,
        )
    assert one["spa"] == three["spa"]
    assert one["mcs_p_values"] == three["mcs_p_values"]


def test_alignment_is_chronological_and_invariant_to_row_order():
    losses, _, ids, times = _panel()
    losses[40:80] *= 2
    losses[80:] *= 3
    order = np.random.default_rng(4).permutation(len(losses))
    aligned, info = aligned_inference_losses(
        losses[order],
        np.ones(len(losses), dtype=bool),
        ids[order],
        ["A0", "A1", "A2"],
        times[order],
        DAY_NS,
    )
    np.testing.assert_allclose(aligned, losses[:40] * 2)
    assert info["status"] == "computed"
    assert info["first_target_time_ns"] == int(times[0])
    assert info["last_target_time_ns"] == int(times[39])


@pytest.mark.parametrize("mode", ["no_times", "missing_date", "short"])
def test_unavailable_inference_cannot_look_like_model_exclusion(mode):
    losses, pinball, ids, times = _panel(n=10 if mode == "short" else 40, assets=1)
    if mode == "missing_date":
        losses[20, 0] = np.nan
    result = evaluator.summarize(
        losses,
        pinball,
        ids,
        ["BTC"],
        ["target", "a", "b"],
        ["target"],
        seed=7,
        n_boot=49,
        target_time_ns=None if mode == "no_times" else times,
        bar_interval_ns=DAY_NS,
    )
    assert result["inference"]["status"] == "unavailable"
    assert result["diebold_mariano"] == result["spa"] == {}
    assert result["mcs_included"] == result["mcs_p_values"] == {}


def test_duplicate_asset_timestamp_is_rejected():
    losses, _, ids, times = _panel(assets=1)
    times[1] = times[0]
    with pytest.raises(ValueError, match="duplicate"):
        aligned_inference_losses(losses, np.ones(40, dtype=bool), ids, ["BTC"], times, DAY_NS)


def test_complete_mask_must_be_boolean_and_aligned():
    losses, _, ids, times = _panel(assets=1)
    with pytest.raises(ValueError, match="complete.*boolean"):
        aligned_inference_losses(losses, np.ones(40), ids, ["BTC"], times, DAY_NS)


def test_coverage_counts_pinball_failures_and_unemitted_origins():
    losses, pinball, ids, _ = _panel(n=4, assets=2)
    losses[0, 0] = np.nan
    pinball[4, 1, 0] = np.nan
    result = coverage_summary(losses, pinball, ids, ["BTC", "ETH"], ["a", "b", "c"], 5)
    assert result["a"]["requested"] == 10
    assert result["a"]["emitted"] == 8
    assert result["a"]["scored_all_metrics"] == 7
    assert result["a"]["missing_or_failed"] == 3
    assert result["b"]["finite_crps"] == 8
    assert result["b"]["per_asset"]["ETH"]["coverage_fraction"] == 3 / 5


@pytest.mark.parametrize("requested", [-1, 0, True, 4.5, 2])
def test_coverage_rejects_invalid_or_exceeded_request_count(requested):
    losses, pinball, ids, _ = _panel(n=4, assets=1)
    with pytest.raises(ValueError, match="requested"):
        coverage_summary(losses, pinball, ids, ["BTC"], ["a", "b", "c"], requested)


def _bars() -> pl.DataFrame:
    times = [datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(4)]
    return pl.DataFrame(
        {
            "security_id": ["BTCUSDT"] * 4,
            "event_time": times,
            "available_time": [t + timedelta(days=1, milliseconds=-1) for t in times],
            "open": [100.0] * 4,
            "high": [105.0] * 4,
            "low": [95.0] * 4,
            "close": [101.0] * 4,
            "volume": [20.0] * 4,
        }
    )


def test_validated_bar_close_is_available_before_next_target_starts():
    times, interval = validate_bars(_bars())
    assert interval == DAY_NS and len(times) == 4


@pytest.mark.parametrize("fault", ["gap", "duplicate", "late", "nan", "ohlc"])
def test_bar_validation_rejects_invalid_market_or_availability_data(fault):
    bars = _bars()
    if fault == "gap":
        bars = bars[[0, 2, 3]]
    elif fault == "duplicate":
        bars = bars[[0, 1, 1, 3]]
    elif fault == "late":
        bars = bars.with_columns(pl.col("available_time") + pl.duration(seconds=1))
    elif fault == "nan":
        bars = bars.with_columns(pl.lit(float("nan")).alias("close"))
    else:
        bars = bars.with_columns(pl.lit(200.0).alias("low"))
    with pytest.raises(ValueError):
        validate_bars(bars)


def test_timesfm_point_channel_is_never_scored_as_a_quantile():
    # Contract v2: channels are [q50, q10..q40, point, q60..q90]; index 5 is
    # the point forecast and must be excluded. q50 (index 0) IS a decile.
    channels = np.array([999.0, 91, 93, 95, 97, 100, 103, 105, 107, 109])
    model = SimpleNamespace(forecast=lambda **_: (np.array([[100.0]]), channels[None, None]))
    actual = evaluator.timesfm_quantiles(model, np.array([99.0, 100.0]))
    np.testing.assert_allclose(actual, np.sort(np.delete(channels, 5)) / 100 - 1)


@pytest.mark.parametrize(
    "adapter,native_shape",
    [
        ("chronos2_quantiles", (1, 1, 99)),
        ("bolt_quantiles", (1, 1, 9)),
    ],
)
def test_chronos_adapters_use_native_batch_and_variates_axes(adapter, native_shape):
    pytest.importorskip("torch", reason="install dipcatcher[nn] to test PyTorch adapters")
    captured = {}

    def predict(context, **kwargs):
        captured["context"] = tuple(context.shape)
        q = np.linspace(95, 105, native_shape[-1]).reshape(native_shape)
        return ([q] if adapter == "chronos2_quantiles" else q), None

    pipe = SimpleNamespace(predict_quantiles=predict)
    result = getattr(evaluator, adapter)(pipe, np.array([98.0, 99.0, 100.0]))
    expected_context = (1, 1, 3) if adapter == "chronos2_quantiles" else (1, 3)
    assert captured["context"] == expected_context
    np.testing.assert_allclose(result, np.linspace(95, 105, native_shape[-1]) / 100 - 1)


def _shard(path: Path, *, legacy: bool = False):
    losses, pinball, ids, times = _panel(assets=1)
    meta = {
        "asset_names": ["BTC"],
        "targets": ["target"],
        "config": {
            "lookback": 20,
            "window": 10,
            "garch_window": 30,
            "origins_per_asset": 40,
            "samples_per_origin": 16,
            "seed": 7,
            "taus": list(evaluator.TAUS),
        },
        "bars_sha256": {"btc.parquet": "a" * 64},
        "artifact_sha256": {"target": "b" * 64},
        "bar_interval_ns": DAY_NS,
    }
    if not legacy:
        meta["scoring_contract"] = evaluator.SCORING_CONTRACT
        meta["implementation_sha256"] = evaluator._implementation_hashes()
    np.savez_compressed(
        path,
        crps_matrix=losses,
        pinball_cube=pinball,
        asset_ids=ids,
        target_time_ns=times,
        model_names=np.array(["target", "a", "b"]),
        meta_json=np.array(json.dumps(meta)),
    )


def test_legacy_scores_require_rerun_even_when_timestamps_are_available(tmp_path):
    part, out = tmp_path / "part.npz", tmp_path / "out.json"
    _shard(part, legacy=True)
    evaluator.merge_parts(
        SimpleNamespace(
            merge_parts=[part],
            merge_out=out,
            seed=7,
            n_boot=49,
            bars_root=None,
        )
    )
    receipt = json.loads(out.read_text())
    assert receipt["legacy_scores_require_rerun"] is True
    assert receipt["inference"]["status"] == "unavailable"
    assert receipt["inference"]["reason"] == "unverified_scoring_contract"
    assert receipt["mcs_included"] == {}


def test_receipt_binds_loss_file_and_refuses_to_overwrite_evidence(tmp_path):
    part, out = tmp_path / "part.npz", tmp_path / "out.json"
    _shard(part)
    args = SimpleNamespace(merge_parts=[part], merge_out=out, seed=7, n_boot=49, bars_root=None)
    evaluator.merge_parts(args)
    receipt = json.loads(out.read_text())
    loss_path = out.with_suffix(".losses.npz")
    assert receipt["losses_sha256"] == evaluator._sha256(loss_path)
    before = out.read_bytes(), loss_path.read_bytes()
    with pytest.raises(FileExistsError):
        evaluator.merge_parts(args)
    assert before == (out.read_bytes(), loss_path.read_bytes())


def test_paired_effect_intervals_have_correct_sign_units_and_declared_family():
    rng = np.random.default_rng(101)
    target = rng.uniform(0.05, 0.10, 400)
    advantage = 0.01 + rng.normal(0, 0.005, 400)
    losses = np.column_stack([target, target - advantage, target + rng.uniform(0, 0.01, 400)])
    result = paired_effect_intervals(
        losses, ["target", "better", "worse"], ["target"], n_boot=399, block=7
    )
    effect = result["comparisons"]["target"]["better"]
    assert result["family_size"] == 2 and result["n_observations"] == 400
    assert effect["mean_target_minus_comparator"] == pytest.approx(advantage.mean())
    assert effect["relative_crps_reduction"] == pytest.approx(advantage.mean() / target.mean())
    assert 0 < effect["pointwise_percentile_ci"][0] < 0.01 < effect["pointwise_percentile_ci"][1]
    assert 0 < effect["simultaneous_ci"][0] < 0.01 < effect["simultaneous_ci"][1]
    assert result["comparisons"]["target"]["worse"]["simultaneous_ci"][1] < 0


def test_serial_dependence_is_visible_in_effect_uncertainty():
    rng = np.random.default_rng(25)
    shock = np.zeros(500)
    for t in range(1, len(shock)):
        shock[t] = 0.95 * shock[t - 1] + rng.normal(0, 0.001)
    target = np.full(500, 0.10)
    losses = np.column_stack([target, target - 0.005 - shock])
    results = [
        paired_effect_intervals(losses, ["target", "a"], ["target"], n_boot=499, block=b)
        for b in (1, 20)
    ]
    intervals = [r["comparisons"]["target"]["a"]["pointwise_percentile_ci"] for r in results]
    assert intervals[1][1] - intervals[1][0] > 2 * (intervals[0][1] - intervals[0][0])


def test_degenerate_mcs_does_not_publish_all_models_as_excluded():
    losses, pinball, ids, times = _panel(assets=1)
    losses[:, 1] = losses[:, 0] + 1.0
    result = evaluator.summarize(
        losses,
        pinball,
        ids,
        ["BTC"],
        ["target", "a", "b"],
        ["target"],
        n_boot=49,
        seed=7,
        target_time_ns=times,
        bar_interval_ns=DAY_NS,
    )
    assert result["mcs_status"] == "unavailable"
    assert result["mcs_included"] == {}
    assert result["effect_sizes"]["simultaneous_status"] == "degenerate_comparison_variance"

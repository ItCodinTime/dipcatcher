"""Synthetic integrity tests; no historical market result is prospective evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from quant_fund.research import prospective_sota as journal

ORIGIN = datetime(2026, 10, 1, tzinfo=UTC)
STEP = timedelta(days=1)
HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64
HEX_E = "e" * 64


def _protocol() -> dict:
    return {
        "schema": journal.SCHEMA,
        "source_id": "synthetic_closes",
        "asset_ids": ["A", "B"],
        "bar_interval_seconds": 86400,
        "history_returns": 750,
        "first_origin_time": ORIGIN.isoformat(),
        "candidate": {
            "model_id": "new_candidate",
            "artifact_sha256": HEX_A,
            "adapter_sha256": HEX_B,
        },
        "published": {
            "model_id": "published_model_v1",
            "artifact_sha256": HEX_C,
            "adapter_sha256": HEX_D,
            "canonical_reference": "doi:synthetic-fixture-only",
        },
        "minimum_paired_origins": 10,
        "planning_effect_crps": 1.0,
        "planning_long_run_sd_crps": 0.01,
        "planning_source_sha256": HEX_E,
        "alpha": 0.05,
        "power": 0.8,
        "hac_lags": 2,
        "coverage_rule": "all_assets_all_models_or_block",
        "primary_metric": "equal_weight_asset_crps_by_target_date",
    }


def _prepare(tmp_path: Path) -> Path:
    protocol = _protocol()
    binding = journal.commitment(protocol)
    anchor = {
        "recorded_at": (ORIGIN - timedelta(days=2)).isoformat(),
        "issuer": "synthetic-test",
        "reference": "synthetic-fixture-anchor",
        "commitment_sha256": binding["commitment_sha256"],
    }
    run = tmp_path / "run"
    journal.prepare(run, protocol, anchor, now=ORIGIN - timedelta(days=1))
    return run


def _bar(at: datetime, close: float) -> dict:
    return {
        "event_time": at.isoformat(),
        "available_time": (at + timedelta(seconds=1)).isoformat(),
        "ingested_time": (at + timedelta(seconds=2)).isoformat(),
        "close": close,
        "source_id": "synthetic_closes",
    }


def _forecast_packet(origin: datetime = ORIGIN) -> dict:
    assets = []
    for j, asset in enumerate(("A", "B")):
        bars = [
            _bar(origin - (750 - i) * STEP, 100 + j + 0.05 * i + 0.2 * np.sin(i))
            for i in range(751)
        ]
        assets.append(
            {
                "asset_id": asset,
                "bars": bars,
                "candidate": {
                    "model_id": "new_candidate",
                    "artifact_sha256": HEX_A,
                    "adapter_sha256": HEX_B,
                    "samples": [0.001 * i for i in range(-10, 10)],
                },
                "published": {
                    "model_id": "published_model_v1",
                    "artifact_sha256": HEX_C,
                    "adapter_sha256": HEX_D,
                    "samples": [0.001 * i for i in range(-12, 8)],
                },
            }
        )
    return {"origin_time": origin.isoformat(), "assets": assets}


def _settle_packet(origin: datetime = ORIGIN) -> dict:
    target = origin + STEP
    return {
        "target_time": target.isoformat(),
        "assets": [
            {"asset_id": asset, "bar": _bar(target, 111 + j)} for j, asset in enumerate(("A", "B"))
        ],
    }


@pytest.fixture
def fast_fhs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(journal, "_fhs_samples", lambda _: [0.001 * i for i in range(-10, 10)])


def _reseal_event(run: Path, number: int) -> None:
    path = run / "events" / f"{number:06d}.json"
    row = json.loads(path.read_text())
    row.pop("receipt_sha256")
    row["receipt_sha256"] = hashlib.sha256(journal._json(row)).hexdigest()
    path.write_text(journal._json(row).decode() + "\n")


def test_forecast_then_later_label_replays(tmp_path: Path, fast_fhs: None) -> None:
    run = _prepare(tmp_path)
    forecast = journal.forecast(run, _forecast_packet(), now=ORIGIN + timedelta(seconds=3))
    assert forecast["derived"]["target_time"] == (ORIGIN + STEP).isoformat()
    before = journal.verify(run)
    assert before["paired_origins"] == 0
    assert before["one_look_result"] is None
    assert before["promotion_state"].startswith("pending")
    journal.settle(run, _settle_packet(), now=ORIGIN + STEP + timedelta(seconds=3))
    after = journal.verify(run)
    assert (after["attempted_origins"], after["paired_origins"], after["coverage_fraction"]) == (
        1,
        1,
        1.0,
    )
    assert after["external_timestamp_authenticity_verified"] is False
    assert after["external_model_inference_verified"] is False


def test_cutoffs_and_missing_asset_fail_before_receipt(tmp_path: Path, fast_fhs: None) -> None:
    run = _prepare(tmp_path)
    packet = _forecast_packet()
    packet["assets"][0]["bars"][-1]["ingested_time"] = (ORIGIN + STEP).isoformat()
    with pytest.raises(ValueError, match="causal timestamps"):
        journal.forecast(run, packet, now=ORIGIN + timedelta(seconds=3))
    packet = _forecast_packet()
    packet["assets"].pop()
    with pytest.raises(ValueError, match="every fixed asset"):
        journal.forecast(run, packet, now=ORIGIN + timedelta(seconds=3))
    with pytest.raises(ValueError, match="before target"):
        journal.forecast(run, _forecast_packet(), now=ORIGIN + STEP)
    assert not list((run / "events").iterdir())
    journal.forecast(run, _forecast_packet(), now=ORIGIN + timedelta(seconds=3))
    label = _settle_packet()
    label["assets"][0]["bar"]["ingested_time"] = (ORIGIN + STEP + timedelta(seconds=30)).isoformat()
    with pytest.raises(ValueError, match="causal timestamps"):
        journal.settle(run, label, now=ORIGIN + STEP + timedelta(seconds=3))
    assert len(list((run / "events").iterdir())) == 1


def test_identifiers_and_power_freeze_fail_closed(tmp_path: Path, fast_fhs: None) -> None:
    protocol = _protocol()
    protocol["planning_long_run_sd_crps"] = 100.0
    with pytest.raises(ValueError, match="power bound"):
        journal.prepare(tmp_path / "bad", protocol, {}, now=ORIGIN - timedelta(days=1))
    run = _prepare(tmp_path)
    packet = _forecast_packet()
    packet["assets"][0]["candidate"]["artifact_sha256"] = HEX_C
    with pytest.raises(ValueError, match="differs from freeze"):
        journal.forecast(run, packet, now=ORIGIN + timedelta(seconds=3))
    assert not list((run / "events").iterdir())


def test_resealed_forecast_and_label_backfill_rejected(tmp_path: Path, fast_fhs: None) -> None:
    run = _prepare(tmp_path)
    journal.forecast(run, _forecast_packet(), now=ORIGIN + timedelta(seconds=3))
    journal.settle(run, _settle_packet(), now=ORIGIN + STEP + timedelta(seconds=3))
    event_path = run / "events" / "000002.json"
    original = event_path.read_text()
    altered = json.loads(original)
    altered["packet"]["assets"][0]["bar"]["close"] += 10
    altered["packet_sha256"] = journal._sha(altered["packet"])
    event_path.write_text(journal._json(altered).decode() + "\n")
    _reseal_event(run, 2)
    with pytest.raises(ValueError, match="settlement scores do not replay"):
        journal.verify(run)
    event_path.write_text(original)
    forecast_path = run / "events" / "000001.json"
    altered = json.loads(forecast_path.read_text())
    altered["derived"]["forecasts"][0]["samples"]["dip_fhs"][0] += 0.01
    forecast_path.write_text(journal._json(altered).decode() + "\n")
    _reseal_event(run, 1)
    with pytest.raises(ValueError, match="forecast does not replay"):
        journal.verify(run)


def test_resealed_forecast_cutoff_is_replayed(tmp_path: Path, fast_fhs: None) -> None:
    run = _prepare(tmp_path)
    journal.forecast(run, _forecast_packet(), now=ORIGIN + timedelta(seconds=3))
    path = run / "events" / "000001.json"
    row = json.loads(path.read_text())
    row["packet"]["assets"][0]["bars"][-1]["ingested_time"] = (
        ORIGIN + timedelta(days=1)
    ).isoformat()
    row["packet_sha256"] = journal._sha(row["packet"])
    row["after"]["pending_forecast_packet_sha256"] = row["packet_sha256"]
    path.write_text(journal._json(row).decode() + "\n")
    _reseal_event(run, 1)
    with pytest.raises(ValueError, match="causal timestamps"):
        journal.verify(run)


def test_missingness_blocks_replacement_origin(tmp_path: Path, fast_fhs: None) -> None:
    run = _prepare(tmp_path)
    journal.interrupt(
        run, "missing published forecast for B", HEX_A, now=ORIGIN + timedelta(seconds=3)
    )
    report = journal.verify(run)
    assert (report["phase"], report["attempted_origins"], report["paired_origins"]) == (
        "blocked",
        1,
        0,
    )
    assert report["coverage_fraction"] == 0.0
    with pytest.raises(ValueError, match="not ready"):
        journal.forecast(run, _forecast_packet(), now=ORIGIN + timedelta(seconds=4))


def test_history_continuity_and_one_fixed_look(tmp_path: Path, fast_fhs: None) -> None:
    run = _prepare(tmp_path)
    forecast_packet = _forecast_packet()
    for i in range(10):
        origin = ORIGIN + i * STEP
        journal.forecast(run, forecast_packet, now=origin + timedelta(seconds=4))
        label_packet = _settle_packet(origin)
        journal.settle(run, label_packet, now=origin + STEP + timedelta(seconds=3))
        report = journal.verify(run)
        assert (report["one_look_result"] is None) == (i < 9)
        if i == 0:
            changed = _forecast_packet(origin + STEP)
            with pytest.raises(ValueError, match="history differs"):
                journal.forecast(run, changed, now=origin + STEP + timedelta(seconds=4))
        if i < 9:
            next_packet = _forecast_packet(origin + STEP)
            for prior_asset, label_asset, next_asset in zip(
                forecast_packet["assets"],
                label_packet["assets"],
                next_packet["assets"],
                strict=True,
            ):
                next_asset["bars"] = [*prior_asset["bars"][1:], label_asset["bar"]]
            forecast_packet = next_packet
    final = journal.verify(run)
    assert final["phase"] == "complete"
    assert (
        final["one_look_result"]["method"] == "one_sided_normal_HAC_Bonferroni_two_fixed_contrasts"
    )
    assert final["promotion_state"].startswith("pending")


def test_real_fhs_produces_finite_samples() -> None:
    rets = np.asarray([0.002 * np.sin(i / 4) + 0.001 * np.cos(i / 7) for i in range(750)])
    values = journal._fhs_samples(rets)
    assert len(values) >= 20
    assert np.isfinite(values).all()

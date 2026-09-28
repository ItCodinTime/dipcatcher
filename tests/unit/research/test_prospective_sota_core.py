"""research/prospective_sota: validation helper matrix, protocol guards, and a
full fabricated forecast/settle/interrupt/verify chain end-to-end."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from quant_fund.research import prospective_sota as sota

pytestmark = pytest.mark.synthetic

ORIGIN0 = datetime(2027, 1, 1, tzinfo=UTC)
STEP = timedelta(days=1)


def _bars(n: int, end: datetime, close0: float = 100.0) -> list[dict[str, Any]]:
    """n bars on the daily grid ending at `end - n*day` ... bar i at end-(n-i)d."""
    out = []
    price = close0
    for i in range(n):
        event = end - (n - 1 - i) * STEP
        price *= 1.0 + 0.0004 * math.sin(i)
        out.append(
            {
                "event_time": event.isoformat(),
                "available_time": (event + timedelta(seconds=1)).isoformat(),
                "ingested_time": (event + timedelta(seconds=2)).isoformat(),
                "close": round(price, 6),
                "source_id": "test-feed",
            }
        )
    return out


def _identity(model_id: str, published: bool = False) -> dict[str, Any]:
    item = {
        "model_id": model_id,
        "artifact_sha256": "a" * 64,
        "adapter_sha256": "b" * 64,
    }
    if published:
        item["canonical_reference"] = "papers://baseline"
    return item


def _protocol(**over: Any) -> dict[str, Any]:
    p = {
        "schema": sota.SCHEMA,
        "source_id": "test-feed",
        "asset_ids": ["AAA", "BBB"],
        "bar_interval_seconds": 86400,
        "history_returns": 750,
        "first_origin_time": ORIGIN0.isoformat(),
        "candidate": _identity("cand-1"),
        "published": _identity("pub-1", published=True),
        "minimum_paired_origins": 10,
        "planning_effect_crps": 0.5,
        "planning_long_run_sd_crps": 0.01,
        "planning_source_sha256": "c" * 64,
        "alpha": 0.05,
        "power": 0.5,
        "hac_lags": 1,
        "coverage_rule": "all_assets_all_models_or_block",
        "primary_metric": "equal_weight_asset_crps_by_target_date",
    }
    p.update(over)
    return p


def _anchor(protocol: dict[str, Any], recorded_at: datetime) -> dict[str, Any]:
    return {
        "recorded_at": recorded_at.isoformat(),
        "issuer": "external-issuer",
        "reference": "external-ref",
        "commitment_sha256": sota.commitment(protocol)["commitment_sha256"],
    }


def _forecast_packet(origin: datetime, windows: list[list[dict]] | None = None) -> dict:
    assets = []
    for k, sid in enumerate(["AAA", "BBB"]):
        bars = windows[k] if windows is not None else _bars(751, origin, close0=100.0 + 10 * k)
        assets.append(
            {
                "asset_id": sid,
                "bars": bars,
                "candidate": {**_identity("cand-1"), "samples": [0.001, -0.001, 0.002] * 8},
                "published": {
                    **_identity("pub-1"),
                    "samples": [0.0, 0.001, -0.001] * 8,
                },
            }
        )
    return {"origin_time": origin.isoformat(), "assets": assets}


def _settle_packet(target: datetime) -> dict:
    return {
        "target_time": target.isoformat(),
        "assets": [
            {
                "asset_id": sid,
                "bar": {
                    "event_time": target.isoformat(),
                    "available_time": (target + timedelta(seconds=5)).isoformat(),
                    "ingested_time": (target + timedelta(seconds=6)).isoformat(),
                    "close": 101.0,
                    "source_id": "test-feed",
                },
            }
            for sid in ["AAA", "BBB"]
        ],
    }


class TestHelpers:
    def test_json_sha_unique(self) -> None:
        assert sota._sha({"b": 1, "a": 2}) == sota._sha({"a": 2, "b": 1})
        assert sota._unique([("a", 1), ("b", 2)]) == {"a": 1, "b": 2}
        with pytest.raises(ValueError, match="duplicate"):
            sota._unique([("a", 1), ("a", 2)])

    def test_read_write_new(self, tmp_path) -> None:
        p = tmp_path / "x.json"
        sota._write_new(p, {"a": 1})
        assert sota._read(p) == {"a": 1}
        with pytest.raises(FileExistsError):
            sota._write_new(p, {"b": 2})

    def test_key_digest_id_int_float_time_guards(self) -> None:
        with pytest.raises(ValueError, match="exactly"):
            sota._keys({"a": 1}, {"a", "b"}, "x")
        with pytest.raises(ValueError, match="digest"):
            sota._digest("zz", "d")
        with pytest.raises(ValueError, match="stable nonempty"):
            sota._id("bad id!", "l")
        with pytest.raises(ValueError, match="integer"):
            sota._int("x", "i", 0, 5)
        with pytest.raises(ValueError, match="integer"):
            sota._int(9, "i", 0, 5)
        with pytest.raises(ValueError, match="finite"):
            sota._float("x", "f")
        with pytest.raises(ValueError, match="finite"):
            sota._float(-1.0, "f", positive=True)
        with pytest.raises(ValueError, match="ISO timestamp"):
            sota._time(123, "t")
        assert sota._iso(datetime(2030, 1, 1, tzinfo=UTC)).endswith("+00:00")
        with pytest.raises(ValueError, match="offset"):
            sota._clock(datetime(2030, 1, 1))


class TestValidateProtocol:
    def test_clean_passes(self) -> None:
        out = sota.validate_protocol(_protocol())
        assert out["primary_metric"] == "equal_weight_asset_crps_by_target_date"

    @pytest.mark.parametrize(
        ("patch", "match"),
        [
            ({"schema": "x"}, "schema"),
            ({"coverage_rule": "x"}, "schema"),
            ({"primary_metric": "x"}, "primary metric"),
            ({"source_id": "bad id!"}, "source_id"),
            ({"asset_ids": []}, "asset_ids"),
            ({"asset_ids": ["B", "A"]}, "sorted"),
            ({"asset_ids": ["A", "A"]}, "sorted"),
            ({"asset_ids": [f"a{i}" for i in range(31)]}, "asset_ids"),
            ({"bar_interval_seconds": 3600}, "daily"),
            ({"history_returns": 749}, "750"),
            ({"first_origin_time": "2027-01-01T00:00:01+00:00"}, "grid"),
            ({"candidate": _identity("dip_fhs")}, "identities"),
            ({"candidate": _identity("pub-1")}, "identities"),
            ({"published": _identity("dip_fhs", published=True)}, "identities"),
            ({"minimum_paired_origins": 9}, "minimum_paired_origins"),
            ({"planning_effect_crps": -1.0}, "planning_effect"),
            ({"planning_source_sha256": "x"}, "digest"),
            ({"alpha": 0.2}, "alpha"),
            ({"power": 0.4}, "alpha"),
            ({"hac_lags": 0}, "hac_lags"),
            ({"hac_lags": 10}, "smaller"),
            ({"planning_effect_crps": 0.0001}, "power bound"),
        ],
    )
    def test_guard_matrix(self, patch: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            sota.validate_protocol(_protocol(**patch))

    def test_published_requires_canonical_reference(self) -> None:
        p = _protocol()
        p["published"] = _identity("pub-1")  # missing canonical_reference
        with pytest.raises(ValueError, match="keys|canonical"):
            sota.validate_protocol(p)


class TestPrepare:
    def test_prepare_writes_manifest(self, tmp_path) -> None:
        proto = _protocol()
        run = tmp_path / "run"
        m = sota.prepare(
            run,
            proto,
            _anchor(proto, ORIGIN0 - timedelta(hours=1)),
            now=ORIGIN0 - timedelta(hours=1),
        )
        assert m["binding"]["protocol_sha256"] == sota._sha(sota.validate_protocol(proto))
        assert (run / "manifest.json").exists() and (run / "events").is_dir()

    def test_prepare_must_precede_first_origin(self, tmp_path) -> None:
        proto = _protocol()
        with pytest.raises(ValueError, match="precede"):
            sota.prepare(tmp_path / "r", proto, _anchor(proto, ORIGIN0), now=ORIGIN0 + STEP)

    def test_anchor_guards(self, tmp_path) -> None:
        proto = _protocol()
        anchor = _anchor(proto, ORIGIN0 - timedelta(hours=1))
        anchor["recorded_at"] = ORIGIN0.isoformat()
        with pytest.raises(ValueError, match="future"):
            sota.prepare(tmp_path / "r", proto, anchor, now=ORIGIN0 - timedelta(hours=1))
        anchor = _anchor(proto, ORIGIN0 - timedelta(hours=1))
        anchor["issuer"] = ""
        with pytest.raises(ValueError, match="issuer"):
            sota.prepare(tmp_path / "r", proto, anchor, now=ORIGIN0 - timedelta(hours=1))
        anchor = _anchor(proto, ORIGIN0 - timedelta(hours=1))
        anchor["commitment_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="commitment"):
            sota.prepare(tmp_path / "r", proto, anchor, now=ORIGIN0 - timedelta(hours=1))


def _run(tmp_path):
    proto = _protocol()
    run = tmp_path / "run"
    sota.prepare(
        run,
        proto,
        _anchor(proto, ORIGIN0 - timedelta(hours=1)),
        now=ORIGIN0 - timedelta(hours=1),
    )
    return proto, run


class TestForecastSettle:
    def test_forecast_guards(self, tmp_path) -> None:
        proto, run = _run(tmp_path)
        packet = _forecast_packet(ORIGIN0 + STEP)
        with pytest.raises(ValueError, match="next origin"):
            sota.forecast(run, packet, now=ORIGIN0 + STEP + timedelta(hours=1))
        packet = _forecast_packet(ORIGIN0)
        with pytest.raises(ValueError, match="before target"):
            sota.forecast(run, packet, now=ORIGIN0 + STEP)  # observed at target
        bad = _forecast_packet(ORIGIN0)
        bad["assets"] = bad["assets"][:1]
        with pytest.raises(ValueError, match="every fixed asset"):
            sota.forecast(run, bad, now=ORIGIN0 + timedelta(hours=1))
        bad = _forecast_packet(ORIGIN0)
        bad["assets"][0]["bars"] = bad["assets"][0]["bars"][:-1]
        with pytest.raises(ValueError, match="bar count"):
            sota.forecast(run, bad, now=ORIGIN0 + timedelta(hours=1))
        bad = _forecast_packet(ORIGIN0)
        bad["assets"][0]["candidate"]["artifact_sha256"] = "f" * 64
        with pytest.raises(ValueError, match="artifact_sha256"):
            sota.forecast(run, bad, now=ORIGIN0 + timedelta(hours=1))

    def test_full_chain_to_complete_and_verify(self, tmp_path) -> None:
        proto, run = _run(tmp_path)
        windows: list[list[dict]] | None = None
        for i in range(proto["minimum_paired_origins"]):
            origin = ORIGIN0 + i * STEP
            target = origin + STEP
            receipt = sota.forecast(
                run, _forecast_packet(origin, windows), now=origin + timedelta(hours=1)
            )
            assert receipt["kind"] == "forecast"
            settle_bars = []
            sp = _settle_packet(target)
            for j in range(2):
                settle_bars.append(sp["assets"][j]["bar"])
            receipt = sota.settle(run, sp, now=target + timedelta(minutes=30))
            # next window = previous bars[1:] + settle bar
            windows = []
            prev_forecast = sota._read(run / "events" / f"{2 * i + 1:06d}.json")
            for j in range(2):
                windows.append([*prev_forecast["packet"]["assets"][j]["bars"][1:], settle_bars[j]])
        out = sota.verify(run)
        assert out["valid_local_chain"] is True
        assert out["paired_origins"] == proto["minimum_paired_origins"]
        assert out["one_look_result"]["method"].startswith("one_sided_normal_HAC")
        assert out["external_timestamp_authenticity_verified"] is False
        assert out["forward_evidence_independently_attested"] is False

    def test_settle_requires_pending_and_label_ordering(self, tmp_path) -> None:
        proto, run = _run(tmp_path)
        target = ORIGIN0 + STEP
        with pytest.raises(ValueError, match="pending forecast|must follow"):
            sota.settle(run, _settle_packet(target), now=target + timedelta(minutes=30))
        sota.forecast(run, _forecast_packet(ORIGIN0), now=ORIGIN0 + timedelta(hours=1))
        # settle observed before target → rejected
        with pytest.raises(ValueError, match="not arrived|target"):
            sota.settle(run, _settle_packet(target), now=ORIGIN0 + timedelta(hours=2))
        # label available before forecast observation → rejected
        sp = _settle_packet(target)
        sp["assets"][0]["bar"]["available_time"] = ORIGIN0.isoformat()
        sp["assets"][0]["bar"]["ingested_time"] = (ORIGIN0 + timedelta(seconds=1)).isoformat()
        with pytest.raises(ValueError, match="availability|causal"):
            sota.settle(run, sp, now=target + timedelta(minutes=30))

    def test_history_shift_enforced(self, tmp_path) -> None:
        proto, run = _run(tmp_path)
        origin = ORIGIN0
        target = origin + STEP
        sota.forecast(run, _forecast_packet(origin), now=origin + timedelta(hours=1))
        sota.settle(run, _settle_packet(target), now=target + timedelta(minutes=30))
        # second forecast with fresh (non-shifted) windows → sealed-history mismatch
        with pytest.raises(ValueError, match="sealed history"):
            sota.forecast(run, _forecast_packet(target), now=target + timedelta(hours=1))

    def test_interrupt_flow(self, tmp_path) -> None:
        proto, run = _run(tmp_path)
        with pytest.raises(ValueError, match="reason"):
            sota.interrupt(run, "", now=ORIGIN0)
        r = sota.interrupt(run, "no_feed", packet_sha256="d" * 64, now=ORIGIN0)
        assert r["kind"] == "interruption"
        with pytest.raises(ValueError, match="cannot be interrupted"):
            sota.interrupt(run, "again", now=ORIGIN0 + timedelta(seconds=1))
        out = sota.verify(run)
        assert out["phase"] == "blocked"
        assert out["attempted_origins"] == 1

    def test_verify_empty_journal(self, tmp_path) -> None:
        proto, run = _run(tmp_path)
        out = sota.verify(run)
        assert out["valid_local_chain"] is True
        assert out["paired_origins"] == 0
        assert out["one_look_result"] is None

    def test_append_clock_must_advance(self, tmp_path) -> None:
        proto, run = _run(tmp_path)
        with pytest.raises(ValueError, match="advance|freeze"):
            sota.forecast(run, _forecast_packet(ORIGIN0), now=ORIGIN0 - timedelta(hours=2))

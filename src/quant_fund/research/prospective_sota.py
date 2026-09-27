"""Prospective, paired, one-step distribution-forecast journal.

The local hash chain detects accidental or uncoordinated changes. It is not an
independent timestamp or proof that a submitted model ran its declared weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

from quant_fund.metrics.scoring import crps_empirical
from quant_fund.paper.quantile_signals import _arch_fit

SCHEMA = "prospective_sota_v1"
METHODS = ("candidate", "dip_fhs", "published")
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9_.:/-]{1,120}\Z")


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)


def _write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(_json(value).decode() + "\n")
        handle.flush()
        import os

        os.fsync(handle.fileno())


def _keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} must have exactly {sorted(expected)}")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{label} must be a stable nonempty identifier")
    return value


def _int(value: Any, label: str, lo: int, hi: int) -> int:
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f"{label} must be an integer in [{lo}, {hi}]")
    return value


def _float(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label} must be finite")
    out = float(value)
    if not math.isfinite(out) or (positive and out <= 0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return out


def _time(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO timestamp")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO timestamp") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError(f"{label} requires an offset")
    return stamp.astimezone(UTC)


def _iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock requires an offset")
    return value.astimezone(UTC).isoformat()


def _clock(now: datetime | None) -> datetime:
    value = now if now is not None else datetime.now(UTC)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("observation clock requires an offset")
    return value.astimezone(UTC)


def _identity(raw: Any, label: str, *, published: bool) -> dict[str, Any]:
    fields = {"model_id", "artifact_sha256", "adapter_sha256"}
    if published:
        fields.add("canonical_reference")
    item = _keys(raw, fields, label)
    _id(item["model_id"], f"{label}.model_id")
    _digest(item["artifact_sha256"], f"{label}.artifact_sha256")
    _digest(item["adapter_sha256"], f"{label}.adapter_sha256")
    if published:
        ref = item["canonical_reference"]
        if not isinstance(ref, str) or not ref.strip() or len(ref) > 500:
            raise ValueError("published.canonical_reference required")
    return item


def validate_protocol(raw: Any) -> dict[str, Any]:
    fields = {
        "schema",
        "source_id",
        "asset_ids",
        "bar_interval_seconds",
        "history_returns",
        "first_origin_time",
        "candidate",
        "published",
        "minimum_paired_origins",
        "planning_effect_crps",
        "planning_long_run_sd_crps",
        "planning_source_sha256",
        "alpha",
        "power",
        "hac_lags",
        "coverage_rule",
        "primary_metric",
    }
    protocol = _keys(raw, fields, "protocol")
    if (
        protocol["schema"] != SCHEMA
        or protocol["coverage_rule"] != "all_assets_all_models_or_block"
    ):
        raise ValueError("unknown schema or coverage rule")
    if protocol["primary_metric"] != "equal_weight_asset_crps_by_target_date":
        raise ValueError("unknown primary metric")
    _id(protocol["source_id"], "source_id")
    assets = protocol["asset_ids"]
    if not isinstance(assets, list) or not assets or len(assets) > 30:
        raise ValueError("asset_ids must be a nonempty fixed list")
    if any(_id(asset, "asset_id") != asset for asset in assets) or assets != sorted(set(assets)):
        raise ValueError("asset_ids must be sorted and unique")
    interval = protocol["bar_interval_seconds"]
    if type(interval) is not int or interval != 86400:
        raise ValueError("first protocol is daily to preserve date-level primary scoring")
    if type(protocol["history_returns"]) is not int or protocol["history_returns"] != 750:
        raise ValueError("dip_fhs history_returns must match the frozen 750-return baseline")
    origin = _time(protocol["first_origin_time"], "first_origin_time")
    if origin.timestamp() % interval != 0:
        raise ValueError("first_origin_time must be on the declared UTC interval grid")
    candidate = _identity(protocol["candidate"], "candidate", published=False)
    published = _identity(protocol["published"], "published", published=True)
    if (
        candidate["model_id"] in (published["model_id"], "dip_fhs")
        or published["model_id"] == "dip_fhs"
    ):
        raise ValueError("three model identities must differ")
    n = _int(protocol["minimum_paired_origins"], "minimum_paired_origins", 10, 100000)
    effect = _float(protocol["planning_effect_crps"], "planning_effect_crps", positive=True)
    long_run_sd = _float(
        protocol["planning_long_run_sd_crps"], "planning_long_run_sd_crps", positive=True
    )
    _digest(protocol["planning_source_sha256"], "planning_source_sha256")
    alpha = _float(protocol["alpha"], "alpha", positive=True)
    power = _float(protocol["power"], "power", positive=True)
    if not 0 < alpha <= 0.05 or not 0.5 <= power < 1:
        raise ValueError("alpha must be <= .05 and power in [.5, 1)")
    lags = _int(protocol["hac_lags"], "hac_lags", 1, 365)
    if lags >= n:
        raise ValueError("hac_lags must be smaller than minimum_paired_origins")
    required = math.ceil(((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * long_run_sd / effect) ** 2)
    if n < required:
        raise ValueError(f"minimum_paired_origins below predeclared power bound {required}")
    return protocol


def _runtime() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        **{name: version(name) for name in ("arch", "numpy", "scipy")},
    }


def _sources() -> dict[str, str]:
    root = Path(__file__).resolve()
    files = {
        "journal": root,
        "fhs": root.parents[1] / "paper" / "quantile_signals.py",
        "scoring": root.parents[1] / "metrics" / "scoring.py",
    }
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}


def commitment(protocol: Any) -> dict[str, Any]:
    validated = validate_protocol(protocol)
    binding = {
        "protocol_sha256": _sha(validated),
        "source_sha256": _sources(),
        "runtime": _runtime(),
    }
    return {"commitment_sha256": _sha(binding), **binding}


def _anchor(raw: Any, binding: dict[str, Any], now: datetime) -> dict[str, Any]:
    anchor = _keys(raw, {"recorded_at", "issuer", "reference", "commitment_sha256"}, "anchor")
    stamp = _time(anchor["recorded_at"], "anchor.recorded_at")
    if stamp > now:
        raise ValueError("anchor timestamp is in the future")
    if not isinstance(anchor["issuer"], str) or not anchor["issuer"].strip():
        raise ValueError("anchor issuer required")
    if not isinstance(anchor["reference"], str) or not anchor["reference"].strip():
        raise ValueError("external anchor reference required")
    if anchor["commitment_sha256"] != binding["commitment_sha256"]:
        raise ValueError("external anchor commitment mismatch")
    return anchor


def prepare(
    run_dir: Path, protocol: Any, anchor: Any, *, now: datetime | None = None
) -> dict[str, Any]:
    clock = _clock(now)
    valid = validate_protocol(protocol)
    first = _time(valid["first_origin_time"], "first_origin_time")
    if not clock < first:
        raise ValueError("prepare must precede first origin")
    binding = commitment(valid)
    anchored = _anchor(anchor, binding, clock)
    manifest = {
        "schema": SCHEMA,
        "protocol": valid,
        "binding": binding,
        "anchor": anchored,
        "prepared_at": _iso(clock),
    }
    manifest["receipt_sha256"] = _sha(manifest)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "events").mkdir()
    _write_new(run_dir / "manifest.json", manifest)
    return manifest


def _fhs_samples(returns: np.ndarray) -> list[float]:
    if returns.size != 750 or not np.isfinite(returns).all():
        raise ValueError("dip_fhs requires exactly 750 finite historical returns")
    fit = _arch_fit(returns * 100.0, vol="GARCH", dist="normal", o=1)
    sig = float(np.sqrt(fit.forecast(horizon=1).variance.iloc[-1, 0])) / 100.0
    mu = float(fit.params.get("mu", 0.0)) / 100.0
    vol = np.asarray(fit.conditional_volatility, dtype=float) / 100.0
    residual = np.asarray(fit.resid, dtype=float) / 100.0
    z = residual[vol > 0] / vol[vol > 0]
    z = z[np.isfinite(z)]
    if z.size < 20 or not np.isfinite(sig) or sig <= 0:
        raise ValueError("dip_fhs fit produced insufficient finite samples")
    values = mu + sig * z
    if values.size > 750 or not np.isfinite(values).all():
        raise ValueError("dip_fhs fit produced invalid samples")
    return [float(v) for v in values]


def _samples(raw: Any, label: str) -> list[float]:
    if not isinstance(raw, list) or not 20 <= len(raw) <= 512:
        raise ValueError(f"{label} requires 20-512 empirical samples")
    return [_float(v, label) for v in raw]


def _bar(
    raw: Any, source: str, now: datetime, *, expected_time: datetime | None = None
) -> tuple[datetime, float]:
    bar = _keys(raw, {"event_time", "available_time", "ingested_time", "close", "source_id"}, "bar")
    if bar["source_id"] != source:
        raise ValueError("bar source differs from frozen source")
    event = _time(bar["event_time"], "bar.event_time")
    available = _time(bar["available_time"], "bar.available_time")
    ingested = _time(bar["ingested_time"], "bar.ingested_time")
    if expected_time is not None and event != expected_time:
        raise ValueError("bar event_time differs from expected interval")
    if not event <= available <= ingested <= now:
        raise ValueError("bar causal timestamps exceed observation cutoff")
    close = _float(bar["close"], "bar.close", positive=True)
    return event, close


def _submitted(raw: Any, identity: dict[str, Any], label: str) -> list[float]:
    item = _keys(raw, {"model_id", "artifact_sha256", "adapter_sha256", "samples"}, label)
    for key in ("model_id", "artifact_sha256", "adapter_sha256"):
        if item[key] != identity[key]:
            raise ValueError(f"{label} {key} differs from freeze")
    return _samples(item["samples"], f"{label}.samples")


def _initial(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": "ready",
        "next_origin_time": _iso(
            _time(manifest["protocol"]["first_origin_time"], "first_origin_time")
        ),
        "pending_forecast_packet_sha256": None,
        "next_history_sha256": None,
        "attempted_origins": 0,
        "paired_origins": 0,
        "last_target_time": None,
    }


def _plan_forecast(
    protocol: dict[str, Any], state: dict[str, Any], packet: Any, observed_at: datetime
) -> tuple[dict[str, Any], dict[str, Any]]:
    if state["phase"] != "ready":
        raise ValueError("journal is not ready for a forecast")
    body = _keys(packet, {"origin_time", "assets"}, "forecast packet")
    origin = _time(body["origin_time"], "origin_time")
    if origin != _time(state["next_origin_time"], "next_origin_time"):
        raise ValueError("forecast origin differs from frozen next origin")
    target = origin + timedelta(seconds=protocol["bar_interval_seconds"])
    if not origin <= observed_at < target:
        raise ValueError("forecast must be recorded after origin and before target event")
    assets = body["assets"]
    if (
        not isinstance(assets, list)
        or [a.get("asset_id") for a in assets if isinstance(a, dict)] != protocol["asset_ids"]
    ):
        raise ValueError("forecast requires exactly every fixed asset in sorted order")
    history_digest = _sha(
        [{"asset_id": item["asset_id"], "bars": item.get("bars")} for item in assets]
    )
    if state["next_history_sha256"] is not None and history_digest != state["next_history_sha256"]:
        raise ValueError("forecast history differs from the previous sealed history and label")
    out: list[dict[str, Any]] = []
    for item in assets:
        row = _keys(item, {"asset_id", "bars", "candidate", "published"}, "forecast asset")
        bars = row["bars"]
        n = protocol["history_returns"]
        if not isinstance(bars, list) or len(bars) != n + 1:
            raise ValueError("forecast history has wrong bar count")
        closes = []
        for i, bar in enumerate(bars):
            expected = origin - timedelta(seconds=(n - i) * protocol["bar_interval_seconds"])
            _, close = _bar(bar, protocol["source_id"], observed_at, expected_time=expected)
            closes.append(close)
        rets = np.diff(np.asarray(closes, dtype=float)) / np.asarray(closes[:-1], dtype=float)
        if not np.isfinite(rets).all():
            raise ValueError("nonfinite history returns")
        forecasts = {
            "candidate": _submitted(row["candidate"], protocol["candidate"], "candidate"),
            "dip_fhs": _fhs_samples(rets),
            "published": _submitted(row["published"], protocol["published"], "published"),
        }
        out.append({"asset_id": row["asset_id"], "origin_close": closes[-1], "samples": forecasts})
    derived = {"origin_time": _iso(origin), "target_time": _iso(target), "forecasts": out}
    after = {
        **state,
        "phase": "awaiting_label",
        "pending_forecast_packet_sha256": "TO_BE_FILLED",
        "attempted_origins": state["attempted_origins"] + 1,
    }
    return derived, after


def _plan_settlement(
    protocol: dict[str, Any],
    state: dict[str, Any],
    forecast: dict[str, Any],
    packet: Any,
    observed_at: datetime,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if state["phase"] != "awaiting_label":
        raise ValueError("no pending forecast")
    body = _keys(packet, {"target_time", "assets"}, "settlement packet")
    target = _time(forecast["derived"]["target_time"], "pending target")
    if _time(body["target_time"], "target_time") != target or observed_at < target:
        raise ValueError("target event has not arrived or differs from pending target")
    if observed_at <= _time(forecast["observed_at"], "forecast observed_at"):
        raise ValueError("label cannot precede forecast")
    assets = body["assets"]
    if (
        not isinstance(assets, list)
        or [a.get("asset_id") for a in assets if isinstance(a, dict)] != protocol["asset_ids"]
    ):
        raise ValueError("settlement requires every fixed asset")
    scored = []
    next_window = []
    for row, prior, old in zip(
        assets, forecast["derived"]["forecasts"], forecast["packet"]["assets"], strict=True
    ):
        item = _keys(row, {"asset_id", "bar"}, "settlement asset")
        if item["asset_id"] != prior["asset_id"]:
            raise ValueError("settlement asset differs from pending forecast")
        _, close = _bar(item["bar"], protocol["source_id"], observed_at, expected_time=target)
        if _time(item["bar"]["available_time"], "available_time") <= _time(
            forecast["observed_at"], "forecast observed_at"
        ):
            raise ValueError("label availability must follow forecast receipt")
        realized = close / prior["origin_close"] - 1.0
        losses = {
            name: crps_empirical(realized, np.asarray(prior["samples"][name], dtype=float))
            for name in METHODS
        }
        if not all(math.isfinite(x) and x >= 0 for x in losses.values()):
            raise ValueError("nonfinite proper score")
        scored.append({"asset_id": item["asset_id"], "realized_return": realized, "crps": losses})
        next_window.append({"asset_id": item["asset_id"], "bars": [*old["bars"][1:], item["bar"]]})
    date_crps = {name: float(np.mean([item["crps"][name] for item in scored])) for name in METHODS}
    paired = state["paired_origins"] + 1
    after = {
        **state,
        "phase": "complete" if paired == protocol["minimum_paired_origins"] else "ready",
        "next_origin_time": _iso(target),
        "pending_forecast_packet_sha256": None,
        "next_history_sha256": _sha(next_window),
        "paired_origins": paired,
        "last_target_time": _iso(target),
    }
    return {"target_time": _iso(target), "assets": scored, "date_crps": date_crps}, after


def _plan_interrupt(
    state: dict[str, Any], reason: str, packet_sha256: str | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    if state["phase"] not in ("ready", "awaiting_label"):
        raise ValueError("journal cannot be interrupted now")
    if not reason.strip() or len(reason) > 500:
        raise ValueError("explicit missingness reason required")
    if packet_sha256 is not None:
        _digest(packet_sha256, "packet_sha256")
    attempted = state["attempted_origins"] + (1 if state["phase"] == "ready" else 0)
    return {"reason": reason, "packet_sha256": packet_sha256}, {
        **state,
        "phase": "blocked",
        "attempted_origins": attempted,
    }


def _sealed(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "receipt_sha256": _sha(value)}


def _events(run_dir: Path) -> list[Path]:
    paths = sorted((run_dir / "events").iterdir())
    if [path.name for path in paths] != [f"{i:06d}.json" for i in range(1, len(paths) + 1)]:
        raise ValueError("missing, duplicate, or unexpected event filename")
    if not all(path.is_file() for path in paths):
        raise ValueError("event path is not a file")
    return paths


def _verified(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    manifest = _read(run_dir / "manifest.json")
    _keys(
        manifest,
        {"schema", "protocol", "binding", "anchor", "prepared_at", "receipt_sha256"},
        "manifest",
    )
    receipt = manifest.pop("receipt_sha256", None)
    if receipt != _sha(manifest):
        raise ValueError("manifest receipt mismatch")
    manifest["receipt_sha256"] = receipt
    if manifest.get("schema") != SCHEMA:
        raise ValueError("manifest schema mismatch")
    protocol = validate_protocol(manifest["protocol"])
    if manifest["binding"] != commitment(protocol):
        raise ValueError("code, dependency, or protocol binding changed")
    frozen = _time(manifest["prepared_at"], "prepared_at")
    _anchor(manifest["anchor"], manifest["binding"], frozen)
    if not frozen < _time(protocol["first_origin_time"], "first_origin_time"):
        raise ValueError("freeze did not precede first origin")
    state = _initial(manifest)
    prior = receipt
    events: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for i, path in enumerate(_events(run_dir), 1):
        event = _read(path)
        _keys(
            event,
            {
                "sequence",
                "kind",
                "prior_receipt_sha256",
                "observed_at",
                "packet",
                "packet_sha256",
                "derived",
                "before",
                "after",
                "receipt_sha256",
            },
            "event",
        )
        seal = event.pop("receipt_sha256", None)
        if seal != _sha(event):
            raise ValueError(f"event {i} receipt mismatch")
        event["receipt_sha256"] = seal
        if event.get("packet_sha256") != _sha(event.get("packet")):
            raise ValueError(f"event {i} packet hash mismatch")
        if (
            event.get("sequence") != i
            or event.get("prior_receipt_sha256") != prior
            or event.get("before") != state
        ):
            raise ValueError(f"event {i} chain or prior state mismatch")
        observed = _time(event.get("observed_at"), "observed_at")
        if observed <= frozen:
            raise ValueError("event observation predates freeze")
        if events and observed <= _time(events[-1]["observed_at"], "prior observed_at"):
            raise ValueError("event observation times must increase")
        kind = event.get("kind")
        if kind == "forecast":
            derived, after = _plan_forecast(protocol, state, event.get("packet"), observed)
            after["pending_forecast_packet_sha256"] = event["packet_sha256"]
            if event.get("derived") != derived:
                raise ValueError("forecast does not replay from the sealed packet")
            pending = event
        elif kind == "settlement":
            if (
                pending is None
                or state["pending_forecast_packet_sha256"] != pending["packet_sha256"]
            ):
                raise ValueError("settlement has no bound pending forecast")
            derived, after = _plan_settlement(
                protocol, state, pending, event.get("packet"), observed
            )
            if event.get("derived") != derived:
                raise ValueError("settlement scores do not replay")
            pending = None
        elif kind == "interruption":
            detail = _keys(event.get("packet"), {"reason", "packet_sha256"}, "interruption")
            derived, after = _plan_interrupt(state, detail["reason"], detail["packet_sha256"])
            if event.get("derived") != derived:
                raise ValueError("interruption detail mismatch")
        else:
            raise ValueError("unknown event kind")
        if event.get("after") != after:
            raise ValueError("event after state differs from deterministic replay")
        state, prior = after, seal
        events.append(event)
    return manifest, state, events


def _append(
    run_dir: Path, kind: str, packet: Any, *, now: datetime | None = None
) -> dict[str, Any]:
    manifest, state, events = _verified(run_dir)
    observed = _clock(now)
    if observed <= _time(manifest["prepared_at"], "prepared_at") or (
        events and observed <= _time(events[-1]["observed_at"], "observed_at")
    ):
        raise ValueError("event observation must advance after freeze")
    protocol = manifest["protocol"]
    if kind == "forecast":
        derived, after = _plan_forecast(protocol, state, packet, observed)
    elif kind == "settlement":
        if not events or events[-1]["kind"] != "forecast":
            raise ValueError("settlement must follow a pending forecast")
        derived, after = _plan_settlement(protocol, state, events[-1], packet, observed)
    elif kind == "interruption":
        detail = _keys(packet, {"reason", "packet_sha256"}, "interruption")
        derived, after = _plan_interrupt(state, detail["reason"], detail["packet_sha256"])
    else:
        raise ValueError("unknown event kind")
    event = {
        "sequence": len(events) + 1,
        "kind": kind,
        "prior_receipt_sha256": events[-1]["receipt_sha256"]
        if events
        else manifest["receipt_sha256"],
        "observed_at": _iso(observed),
        "packet": packet,
        "packet_sha256": _sha(packet),
        "derived": derived,
        "before": state,
        "after": after,
    }
    if kind == "forecast":
        # A receipt cannot literally contain its own digest in the after state.
        # Use the event's packet digest as the pending identity.
        event["after"]["pending_forecast_packet_sha256"] = event["packet_sha256"]
    sealed = _sealed(event)
    _write_new(run_dir / "events" / f"{len(events) + 1:06d}.json", sealed)
    return sealed


def forecast(run_dir: Path, packet: Any, *, now: datetime | None = None) -> dict[str, Any]:
    return _append(run_dir, "forecast", packet, now=now)


def settle(run_dir: Path, packet: Any, *, now: datetime | None = None) -> dict[str, Any]:
    return _append(run_dir, "settlement", packet, now=now)


def interrupt(
    run_dir: Path, reason: str, packet_sha256: str | None = None, *, now: datetime | None = None
) -> dict[str, Any]:
    return _append(
        run_dir, "interruption", {"reason": reason, "packet_sha256": packet_sha256}, now=now
    )


def _one_look(protocol: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any] | None:
    dates = [event["derived"]["date_crps"] for event in events if event["kind"] == "settlement"]
    if len(dates) != protocol["minimum_paired_origins"]:
        return None
    alpha, lags = protocol["alpha"], protocol["hac_lags"]
    output: dict[str, Any] = {
        "method": "one_sided_normal_HAC_Bonferroni_two_fixed_contrasts",
        "alpha_each": alpha / 2,
        "comparisons": {},
    }
    for comparator in ("dip_fhs", "published"):
        diff = np.asarray([date[comparator] - date["candidate"] for date in dates], dtype=float)
        centered = diff - np.mean(diff)
        n = len(diff)
        long_run_var = float(np.dot(centered, centered) / n)
        for lag in range(1, lags + 1):
            gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
            long_run_var += 2 * (1 - lag / (lags + 1)) * gamma
        se = math.sqrt(max(0.0, long_run_var) / n)
        effect = float(np.mean(diff))
        bound = effect - float(norm.ppf(1 - alpha / 2)) * se
        output["comparisons"][comparator] = {
            "mean_crps_reduction": effect,
            "hac_standard_error": se,
            "one_sided_lower_bound": bound,
            "positive_lower_bound": bound > 0,
        }
    output["predeclared_score_rule_met"] = all(
        row["positive_lower_bound"] for row in output["comparisons"].values()
    )
    return output


def verify(run_dir: Path) -> dict[str, Any]:
    manifest, state, events = _verified(run_dir)
    protocol = manifest["protocol"]
    paired = state["paired_origins"]
    attempted = state["attempted_origins"]
    first = _time(protocol["first_origin_time"], "first_origin_time")
    elapsed = (_clock(None) - first).total_seconds()
    due = min(protocol["minimum_paired_origins"], max(0, int(elapsed // 86400)))
    if state["phase"] == "blocked":
        due = attempted
    return {
        "valid_local_chain": True,
        "schema": SCHEMA,
        "phase": state["phase"],
        "manifest_receipt_sha256": manifest["receipt_sha256"],
        "last_receipt_sha256": events[-1]["receipt_sha256"]
        if events
        else manifest["receipt_sha256"],
        "attempted_origins": attempted,
        "paired_origins": paired,
        "scheduled_origins_due": due,
        "unrecorded_due_origins": max(0, due - attempted),
        "coverage_of_due_origins": paired / due if due else None,
        "minimum_paired_origins": protocol["minimum_paired_origins"],
        "coverage_fraction": paired / attempted if attempted else None,
        "primary_metric": protocol["primary_metric"],
        "one_look_result": _one_look(protocol, events),
        "promotion_state": "pending_external_attestation_and_replication",
        "external_timestamp_authenticity_verified": False,
        "external_model_inference_verified": False,
        "forward_evidence_independently_attested": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bind = sub.add_parser("commitment")
    bind.add_argument("protocol", type=Path)
    prep = sub.add_parser("prepare")
    prep.add_argument("run_dir", type=Path)
    prep.add_argument("protocol", type=Path)
    prep.add_argument("anchor", type=Path)
    for command in ("forecast", "settle"):
        stage = sub.add_parser(command)
        stage.add_argument("run_dir", type=Path)
        stage.add_argument("packet", type=Path)
    stop = sub.add_parser("interrupt")
    stop.add_argument("run_dir", type=Path)
    stop.add_argument("reason")
    stop.add_argument("--packet-sha256")
    check = sub.add_parser("verify")
    check.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if args.command == "commitment":
        result = commitment(_read(args.protocol))
    elif args.command == "prepare":
        result = prepare(args.run_dir, _read(args.protocol), _read(args.anchor))
    elif args.command == "forecast":
        result = forecast(args.run_dir, _read(args.packet))
    elif args.command == "settle":
        result = settle(args.run_dir, _read(args.packet))
    elif args.command == "interrupt":
        result = interrupt(args.run_dir, args.reason, args.packet_sha256)
    else:
        result = verify(args.run_dir)
    print(_json(result).decode())


if __name__ == "__main__":
    main()

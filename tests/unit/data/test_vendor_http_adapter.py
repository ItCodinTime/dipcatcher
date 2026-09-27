"""Vendor HTTP adapter: fail-closed entitlement + PIT contract, zero network.

All tests inject a fake ``Transport`` callable; no test touches the network.
The "no real network" proof monkeypatches ``urllib.request.urlopen`` to raise
and then exercises the happy path through the injected transport.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from quant_fund.data.adapters.vendor_http import (
    DEFAULT_API_KEY_ENV,
    REVISION,
    SOURCE,
    VendorDataError,
    VendorEntitlementError,
    VendorHttpAdapter,
    VendorHttpConfig,
)

CLOCK_NOW = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)


def _fixed_clock() -> datetime:
    return CLOCK_NOW


def _bar(
    security_id: str = "SEC_A",
    ts: str = "2024-01-02T21:00:00Z",
    *,
    release_ts: str | None = "2024-01-02T21:05:00Z",
    ingest_ts: str | None = "2024-01-02T21:06:00Z",
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.5,
    volume: float = 1_000_000.0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "security_id": security_id,
        "ts": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }
    if release_ts is not None:
        row["release_ts"] = release_ts
    if ingest_ts is not None:
        row["ingest_ts"] = ingest_ts
    return row


class FakeTransport:
    """Records calls and serves canned payloads keyed by endpoint."""

    def __init__(self, payloads: dict[str, object]) -> None:
        self.payloads = payloads
        self.calls: list[tuple[str, Mapping[str, str]]] = []

    def __call__(self, url: str, headers: Mapping[str, str]) -> bytes:
        self.calls.append((url, headers))
        endpoint = url.split("?")[0].rsplit("/", 1)[-1]
        return json.dumps(self.payloads[endpoint]).encode("utf-8")


def _config(**overrides: Any) -> VendorHttpConfig:
    kwargs: dict[str, Any] = {
        "vendor": "example_vendor",
        "base_url": "https://vendor.example.test/v1",
        "license_acknowledged": True,
        "environ": {DEFAULT_API_KEY_ENV: "test-key"},
    }
    kwargs.update(overrides)
    return VendorHttpConfig(**kwargs)


def _adapter(payloads: dict[str, object]) -> tuple[VendorHttpAdapter, FakeTransport]:
    transport = FakeTransport(payloads)
    adapter = VendorHttpAdapter(_config(), transport=transport, clock=_fixed_clock)
    return adapter, transport


# -- Fail-closed entitlement --


def test_fail_closed_without_license_acknowledgement() -> None:
    with pytest.raises(VendorEntitlementError, match="license_acknowledged"):
        VendorHttpAdapter(
            _config(license_acknowledged=False),
            transport=FakeTransport({}),
            clock=_fixed_clock,
        )


def test_fail_closed_without_api_key() -> None:
    with pytest.raises(VendorEntitlementError, match="missing API key"):
        VendorHttpAdapter(_config(environ={}), transport=FakeTransport({}), clock=_fixed_clock)


def test_fail_closed_with_blank_api_key() -> None:
    with pytest.raises(VendorEntitlementError, match="missing API key"):
        VendorHttpAdapter(
            _config(environ={DEFAULT_API_KEY_ENV: "   "}),
            transport=FakeTransport({}),
            clock=_fixed_clock,
        )


def test_fail_closed_reads_real_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DEFAULT_API_KEY_ENV, raising=False)
    with pytest.raises(VendorEntitlementError, match="missing API key"):
        VendorHttpAdapter(_config(environ=None), transport=FakeTransport({}), clock=_fixed_clock)
    monkeypatch.setenv(DEFAULT_API_KEY_ENV, "real-env-key")
    adapter = VendorHttpAdapter(
        _config(environ=None), transport=FakeTransport({"bars": {"bars": []}}), clock=_fixed_clock
    )
    assert adapter.get_bars().is_empty()


# -- PIT contract: release_ts / ingest_ts required, never filled --


def test_missing_release_ts_rejected() -> None:
    adapter, _t = _adapter({"bars": {"bars": [_bar(release_ts=None)]}})
    with pytest.raises(VendorDataError, match="release_ts"):
        adapter.get_bars()


def test_missing_ingest_ts_rejected() -> None:
    adapter, _t = _adapter({"bars": {"bars": [_bar(ingest_ts=None)]}})
    with pytest.raises(VendorDataError, match="ingest_ts"):
        adapter.get_bars()


def test_naive_timestamp_rejected() -> None:
    adapter, _t = _adapter({"bars": {"bars": [_bar(release_ts="2024-01-02T21:05:00")]}})
    with pytest.raises(VendorDataError, match="timezone-aware"):
        adapter.get_bars()


def test_release_after_ingest_rejected() -> None:
    adapter, _t = _adapter({"bars": {"bars": [_bar(release_ts="2024-01-02T22:00:00Z")]}})
    with pytest.raises(VendorDataError, match="release_ts .* after ingest_ts"):
        adapter.get_bars()


def test_future_ingest_ts_rejected() -> None:
    adapter, _t = _adapter(
        {
            "bars": {
                "bars": [_bar(ingest_ts="2999-01-01T00:00:00Z", release_ts="2999-01-01T00:00:00Z")]
            }
        }
    )
    with pytest.raises(VendorDataError, match="in the future"):
        adapter.get_bars()


# -- OHLCV validation --


def test_duplicate_security_ts_rejected() -> None:
    adapter, _t = _adapter({"bars": {"bars": [_bar(), _bar()]}})
    with pytest.raises(VendorDataError, match="duplicate bar"):
        adapter.get_bars()


def test_non_finite_ohlcv_rejected() -> None:
    adapter, _t = _adapter({"bars": {"bars": [_bar(close=float("inf"))]}})
    with pytest.raises(VendorDataError, match="close must be finite"):
        adapter.get_bars()
    adapter, _t = _adapter({"bars": {"bars": [_bar(open_=float("nan"))]}})
    with pytest.raises(VendorDataError, match="open must be finite"):
        adapter.get_bars()


def test_negative_volume_rejected() -> None:
    adapter, _t = _adapter({"bars": {"bars": [_bar(volume=-1.0)]}})
    with pytest.raises(VendorDataError, match="negative volume"):
        adapter.get_bars()


def test_inverted_envelope_rejected() -> None:
    adapter, _t = _adapter({"bars": {"bars": [_bar(high=98.0, low=101.0)]}})
    with pytest.raises(VendorDataError, match="inverted envelope"):
        adapter.get_bars()


# -- Happy path with injected transport (still zero network) --


def test_happy_path_bars_with_fake_transport() -> None:
    adapter, transport = _adapter(
        {
            "bars": {
                "bars": [
                    _bar(
                        "SEC_B",
                        "2024-01-03T21:00:00Z",
                        release_ts="2024-01-03T21:05:00Z",
                        ingest_ts="2024-01-03T21:06:00Z",
                    ),
                    _bar("SEC_A", "2024-01-02T21:00:00Z"),
                ]
            }
        }
    )
    bars = adapter.get_bars()
    assert bars.height == 2
    # sorted by event_time, then security_id
    assert bars["security_id"].to_list() == ["SEC_A", "SEC_B"]
    for col in (
        "event_time",
        "available_time",
        "ingested_time",
        "release_ts",
        "ingest_ts",
        "source",
        "revision_id",
    ):
        assert col in bars.columns
        assert bars[col].null_count() == 0
    # PIT aliases: availability is the vendor release, ingestion is ingest_ts
    assert (bars["available_time"] == bars["release_ts"]).all()
    assert (bars["ingested_time"] == bars["ingest_ts"]).all()
    assert bars["source"].unique().to_list() == [SOURCE]
    assert bars["revision_id"].unique().to_list() == [REVISION]
    # all timestamps UTC
    for col in ("event_time", "release_ts", "ingest_ts"):
        assert str(bars[col].dtype.time_zone) == "UTC"
    # auth header carried the entitled key
    assert transport.calls[0][1]["Authorization"] == "Bearer test-key"
    assert transport.calls[0][0].startswith("https://vendor.example.test/v1/bars")


def test_happy_path_filters() -> None:
    adapter, _t = _adapter(
        {
            "bars": {
                "bars": [
                    _bar("SEC_A", "2024-01-02T21:00:00Z"),
                    _bar(
                        "SEC_B",
                        "2024-01-03T21:00:00Z",
                        release_ts="2024-01-03T21:05:00Z",
                        ingest_ts="2024-01-03T21:06:00Z",
                    ),
                ]
            }
        }
    )
    only_a = adapter.get_bars(security_ids=["SEC_A"])
    assert only_a["security_id"].to_list() == ["SEC_A"]
    window = adapter.get_bars(
        start=datetime(2024, 1, 3, tzinfo=UTC), end=datetime(2024, 1, 4, tzinfo=UTC)
    )
    assert window["security_id"].to_list() == ["SEC_B"]


def test_corporate_actions_and_master_happy_path() -> None:
    adapter, _t = _adapter(
        {
            "corporate_actions": {
                "corporate_actions": [
                    {
                        "security_id": "SEC_A",
                        "ts": "2024-01-02T21:00:00Z",
                        "action_type": "split",
                        "factor": 2.0,
                        "release_ts": "2024-01-02T21:05:00Z",
                        "ingest_ts": "2024-01-02T21:06:00Z",
                    }
                ]
            },
            "securities": {
                "securities": [
                    {
                        "security_id": "SEC_A",
                        "ticker": "AAA",
                        "valid_from": "2020-01-02T00:00:00Z",
                        "release_ts": "2024-01-02T21:05:00Z",
                        "ingest_ts": "2024-01-02T21:06:00Z",
                    }
                ]
            },
        }
    )
    actions = adapter.get_corporate_actions()
    assert actions.height == 1
    assert actions["factor"].to_list() == [2.0]
    assert actions["release_ts"].null_count() == 0
    master = adapter.get_security_master()
    assert master.height == 1
    assert master["ticker"].to_list() == ["AAA"]
    assert master["release_ts"].null_count() == 0
    assert master["ingest_ts"].null_count() == 0


def test_master_missing_release_ts_rejected() -> None:
    adapter, _t = _adapter(
        {
            "securities": {
                "securities": [
                    {
                        "security_id": "SEC_A",
                        "ticker": "AAA",
                        "valid_from": "2020-01-02T00:00:00Z",
                        "ingest_ts": "2024-01-02T21:06:00Z",
                    }
                ]
            }
        }
    )
    with pytest.raises(VendorDataError, match="release_ts"):
        adapter.get_security_master()


# -- Proof that tests never use the real network --


def test_no_real_network_used(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("real network was touched")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    monkeypatch.setattr("quant_fund.data.concurrent_io.pooled_request", _boom)
    monkeypatch.setattr("quant_fund.data.sources.base.pooled_request", _boom)
    adapter, transport = _adapter({"bars": {"bars": [_bar()]}})
    bars = adapter.get_bars()
    assert bars.height == 1
    assert len(transport.calls) == 1

"""observe.flags env resolution: truth table, endpoint, host, port bounds."""

from __future__ import annotations

import pytest

from quant_fund.observe import flags


class TestObserveEnabled:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " 1 "])
    def test_truthy(self, monkeypatch, value: str) -> None:
        monkeypatch.setenv("DIPCATCHER_OBSERVE", value)
        assert flags.observe_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "enabled"])
    def test_falsy(self, monkeypatch, value: str) -> None:
        monkeypatch.setenv("DIPCATCHER_OBSERVE", value)
        assert flags.observe_enabled() is False

    def test_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("DIPCATCHER_OBSERVE", raising=False)
        assert flags.observe_enabled() is False


class TestOtelEndpoint:
    def test_unset_returns_none(self, monkeypatch) -> None:
        monkeypatch.delenv("DIPCATCHER_OTEL_ENDPOINT", raising=False)
        assert flags.otel_endpoint() is None

    def test_blank_returns_none(self, monkeypatch) -> None:
        monkeypatch.setenv("DIPCATCHER_OTEL_ENDPOINT", "   ")
        assert flags.otel_endpoint() is None

    def test_value_passthrough(self, monkeypatch) -> None:
        monkeypatch.setenv("DIPCATCHER_OTEL_ENDPOINT", "http://127.0.0.1:4318/v1/traces")
        assert flags.otel_endpoint() == "http://127.0.0.1:4318/v1/traces"


class TestMetricsHost:
    def test_default_loopback(self, monkeypatch) -> None:
        monkeypatch.delenv("DIPCATCHER_METRICS_HOST", raising=False)
        assert flags.metrics_host() == "127.0.0.1"

    def test_blank_is_loopback(self, monkeypatch) -> None:
        monkeypatch.setenv("DIPCATCHER_METRICS_HOST", "  ")
        assert flags.metrics_host() == "127.0.0.1"

    def test_explicit_host(self, monkeypatch) -> None:
        monkeypatch.setenv("DIPCATCHER_METRICS_HOST", "0.0.0.0")
        assert flags.metrics_host() == "0.0.0.0"


class TestMetricsPort:
    def test_unset_none(self, monkeypatch) -> None:
        monkeypatch.delenv("DIPCATCHER_METRICS_PORT", raising=False)
        assert flags.metrics_port() is None

    def test_blank_none(self, monkeypatch) -> None:
        monkeypatch.setenv("DIPCATCHER_METRICS_PORT", "  ")
        assert flags.metrics_port() is None

    def test_valid_port(self, monkeypatch) -> None:
        monkeypatch.setenv("DIPCATCHER_METRICS_PORT", "9090")
        assert flags.metrics_port() == 9090

    def test_zero_allowed(self, monkeypatch) -> None:
        monkeypatch.setenv("DIPCATCHER_METRICS_PORT", "0")
        assert flags.metrics_port() == 0

    @pytest.mark.parametrize("bad", ["-1", "65536", "99999"])
    def test_out_of_range(self, monkeypatch, bad: str) -> None:
        monkeypatch.setenv("DIPCATCHER_METRICS_PORT", bad)
        with pytest.raises(ValueError, match="0..65535"):
            flags.metrics_port()

    def test_non_numeric_raises(self, monkeypatch) -> None:
        monkeypatch.setenv("DIPCATCHER_METRICS_PORT", "abc")
        with pytest.raises(ValueError):
            flags.metrics_port()

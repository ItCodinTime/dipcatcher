"""Credential query params must never reach error text or logs.

``HttpClient`` embeds the request URL in ``SourceError`` messages; FRED
(``api_key``) and BEA (``UserID``) authenticate via query params, so an
un-redacted URL would leak the key into stderr/log aggregation.
"""

from __future__ import annotations

import pytest

from quant_fund.data.sources.base import HttpClient, SourceError, redact_url


def test_redact_url_strips_credential_params() -> None:
    url = "https://api.stlouisfed.org/fred/series/observations?series_id=GDP&api_key=SECRETKEY123456&file_type=json"
    out = redact_url(url)
    assert "SECRETKEY123456" not in out
    assert "api_key=%2A%2A%2A" in out or "api_key=***" in out
    assert "series_id=GDP" in out
    assert "file_type=json" in out


@pytest.mark.parametrize(
    "param",
    [
        "api_key",
        "apikey",
        "UserID",
        "user_id",
        "token",
        "access_token",
        "secret",
        "password",
        "signature",
        "key",
    ],
)
def test_redact_url_covers_param_names(param: str) -> None:
    url = f"https://example.test/data?{param}=hunter2hunter2&ok=1"
    assert "hunter2hunter2" not in redact_url(url)


def test_redact_url_case_insensitive_and_blank() -> None:
    out = redact_url("https://example.test/?API_KEY=abc123abc123abc123&x=")
    assert "abc123abc123abc123" not in out


def test_redact_url_non_http_never_echoed() -> None:
    assert redact_url("file:///etc/passwd?api_key=x") == "file://<redacted>"
    assert redact_url("/local/path?token=x") == "<redacted>"


def test_http_client_error_messages_redact(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing request must raise SourceError without the credential."""
    import urllib.error

    secret_url = "https://api.test/v1?api_key=SUPERSECRET987&series_id=X"

    def _fail(request, timeout):  # noqa: ANN001
        raise urllib.error.URLError("boom")

    monkeypatch.setattr("quant_fund.data.sources.base.urlopen", _fail)
    client = HttpClient(retries=0)
    with pytest.raises(SourceError) as excinfo:
        client.get_json(secret_url)
    assert "SUPERSECRET987" not in str(excinfo.value)
    assert "series_id=X" in str(excinfo.value)

    def _bad_json(request, timeout):  # noqa: ANN001
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, n=-1):
                return b"not-json{"

        return _Resp()

    monkeypatch.setattr("quant_fund.data.sources.base.urlopen", _bad_json)
    with pytest.raises(SourceError) as excinfo:
        client.get_json(secret_url)
    assert "SUPERSECRET987" not in str(excinfo.value)


def test_http_client_scheme_rejection_redacted() -> None:
    client = HttpClient()
    with pytest.raises(SourceError) as excinfo:
        client.get_json("ftp://host/path?api_key=SECRETKEYZZZ")
    assert "SECRETKEYZZZ" not in str(excinfo.value)

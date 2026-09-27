"""Tests must not import network adapters."""


def test_no_requests_in_data_adapters() -> None:
    from quant_fund.data.adapters import parquet, synthetic

    assert "requests" not in open(synthetic.__file__).read()
    assert "http" not in open(parquet.__file__).read().lower()

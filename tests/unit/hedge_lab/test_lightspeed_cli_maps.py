"""Malformed nested confirm receipts fail closed during CLI rendering."""

from quant_fund.lightspeed.cli import _as_map


def test_as_map_rejects_truthy_non_mapping() -> None:
    assert _as_map([{"p_value": 0.01}]) == {}
    assert _as_map("not a receipt mapping") == {}


def test_as_map_preserves_only_string_keys() -> None:
    assert _as_map({"p_value": 0.01, 1: "invalid"}) == {"p_value": 0.01}

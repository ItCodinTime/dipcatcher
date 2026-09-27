"""Schema, canonical payload, and HMAC."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from quant_fund.pretrade.config import (
    PretradeConfig,
    SessionConfig,
    load_pretrade_config,
    sign_config,
)
from tests.unit.pretrade.support import HMAC_KEY, make_engine

ROOT = Path(__file__).resolve().parents[3]
YAML_PATH = ROOT / "configs" / "pretrade_risk.yaml"


def test_yaml_round_trip_hash_is_stable() -> None:
    first, digest_a, sig_a = load_pretrade_config(YAML_PATH, hmac_key=HMAC_KEY)
    second, digest_b, sig_b = load_pretrade_config(YAML_PATH, hmac_key=HMAC_KEY)
    assert first.schema_version == 1
    assert first == second
    assert digest_a == digest_b
    assert sig_a == sig_b
    assert digest_a != sig_a
    assert len(digest_a) == 64


def test_hmac_changes_with_key_content_hash_does_not() -> None:
    config, digest, signature = load_pretrade_config(YAML_PATH, hmac_key=HMAC_KEY)
    other_digest, other_signature = sign_config(config, HMAC_KEY + b"-other-key")
    assert other_digest == digest
    assert other_signature != signature


def test_short_hmac_key_is_rejected() -> None:
    config, _, _ = load_pretrade_config(YAML_PATH, hmac_key=HMAC_KEY)
    with pytest.raises(ValueError, match="hmac_key"):
        sign_config(config, b"short")


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_pretrade_config(path, hmac_key=HMAC_KEY)


def test_unknown_schema_version_and_extra_keys_fail() -> None:
    raw = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    raw["schema_version"] = 2
    with pytest.raises(ValidationError):
        PretradeConfig.model_validate(raw)
    raw["schema_version"] = 1
    raw["unexpected"] = True
    with pytest.raises(ValidationError):
        PretradeConfig.model_validate(raw)


def test_session_and_ring_constraints() -> None:
    with pytest.raises(ValidationError):
        SessionConfig(timezone="Not/AZone", open_minute=10, close_minute=20)
    with pytest.raises(ValidationError):
        SessionConfig(timezone="America/New_York", open_minute=600, close_minute=600)
    with pytest.raises(ValidationError):
        make_engine(ring_capacity=100)
    with pytest.raises(ValidationError):
        make_engine(max_orders_per_window=64, ring_capacity=64)


def test_decision_records_the_signed_hash() -> None:
    engine = make_engine(max_order_quantity=1)
    from tests.unit.pretrade.support import arm, order

    sid = arm(engine)
    denied = engine.decide(order(sid, qty=5))
    assert denied.allowed is False
    assert "max_order_quantity" in denied.reasons
    assert denied.config_sha256 == engine.config_sha256
    assert denied.config_hmac_sha256 == engine.config_hmac_sha256
    assert denied.schema_version == 1
    payload = denied.as_dict()
    assert payload["config_sha256"] == engine.config_sha256
    assert payload["mode"] == "shadow"

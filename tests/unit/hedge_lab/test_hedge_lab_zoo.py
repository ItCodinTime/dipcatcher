"""Kronos zoo fetch — fail-closed paths (never touches the network)."""

from __future__ import annotations

import sys
import types

import quant_fund.hedge_lab.zoo as zoo
from quant_fund.models.robinhood_plus.constants import VARIANT_HUB


def test_unknown_variant_fails_closed() -> None:
    out = zoo.fetch_kronos_variant("does_not_exist")
    assert out == {"status": "unknown_variant", "variant": "does_not_exist"}


def test_known_variant_without_hub_skips(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(zoo, "lab_root", lambda: tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    variant = next(iter(VARIANT_HUB))
    out = zoo.fetch_kronos_variant(variant)
    assert out["status"] == "skipped_no_huggingface_hub"
    assert out["variant"] == variant
    assert out["research_only"] is True


def test_known_variant_with_fake_hub(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(zoo, "lab_root", lambda: tmp_path)

    def snapshot_download(*, repo_id: str, revision: str, local_dir: str) -> str:
        return str(local_dir)

    fake = types.ModuleType("huggingface_hub")
    fake.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)

    variant = next(iter(VARIANT_HUB))
    spec = VARIANT_HUB[variant]
    out = zoo.fetch_kronos_variant(variant)
    assert out["status"] == "ok"
    assert out["tokenizer"] == spec["tokenizer"]
    assert out["model"] == spec["model"]
    assert out["blend_weight"] == 0.0
    assert out["sizes_book"] is False
    assert out["research_only"] is True
    assert out["tokenizer_path"].endswith(spec["tokenizer"].rsplit("/", 1)[-1])

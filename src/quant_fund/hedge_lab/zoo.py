"""Optional local Kronos zoo fetch. Never runs in CI. Disk budget gated."""

from __future__ import annotations

from typing import Any

from quant_fund.hedge_lab.resources import assert_disk_budget, lab_root
from quant_fund.models.robinhood_plus.constants import VARIANT_HUB


def fetch_kronos_variant(variant: str = "small") -> dict[str, Any]:
    """Snapshot one Kronos Hub pair into ``third_party/kronos_weights``.

    Requires ``huggingface_hub``. Does not change ``blend_weight``. Missing
    extras stamp ``skipped``.
    """
    if variant not in VARIANT_HUB:
        return {"status": "unknown_variant", "variant": variant}
    spec = VARIANT_HUB[variant]
    dest_root = lab_root() / "third_party" / "kronos_weights"
    dest_root.mkdir(parents=True, exist_ok=True)
    assert_disk_budget(extra_bytes=2 * 1024**3)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return {
            "status": "skipped_no_huggingface_hub",
            "variant": variant,
            "research_only": True,
        }
    tokenizer = str(spec["tokenizer"])
    model = str(spec["model"])
    tok_dir = dest_root / tokenizer.rsplit("/", 1)[-1]
    model_dir = dest_root / model.rsplit("/", 1)[-1]
    tok_path = snapshot_download(repo_id=tokenizer, local_dir=str(tok_dir))
    model_path = snapshot_download(repo_id=model, local_dir=str(model_dir))
    return {
        "status": "ok",
        "variant": variant,
        "tokenizer": tokenizer,
        "model": model,
        "tokenizer_path": str(tok_path),
        "model_path": str(model_path),
        "research_only": True,
        "blend_weight": 0.0,
        "sizes_book": False,
    }

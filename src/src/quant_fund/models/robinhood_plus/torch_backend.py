"""Optional local-only torch adapter; never downloads models."""
from __future__ import annotations
from pathlib import Path
class RobinhoodPlusTorchError(RuntimeError): pass

def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception: return False

def load_pretrained_predictor(name="mini", *, model_path=None, tokenizer_path=None, allow_network=False, **kwargs):
    if not allow_network and (model_path is None or tokenizer_path is None):
        raise RobinhoodPlusTorchError("local weights are required when allow_network=false")
    if allow_network:
        raise RobinhoodPlusTorchError("network-backed pretrained loading is disabled for research-only backend")
    model=Path(model_path); tok=Path(tokenizer_path)
    if not model.is_dir() or not tok.is_dir(): raise RobinhoodPlusTorchError("local weights are missing; allow_network=false")
    if not torch_available(): raise RobinhoodPlusTorchError("torch is unavailable; allow_network=false")
    raise RobinhoodPlusTorchError("upstream local predictor is unavailable; allow_network=false")

def forecast_cross_section_torch(frame, asof, config, security_ids):
    cfg=config.robinhood_plus
    pred=load_pretrained_predictor("mini", model_path=cfg.model_path, tokenizer_path=cfg.tokenizer_path, allow_network=bool(cfg.allow_network))
    return pred

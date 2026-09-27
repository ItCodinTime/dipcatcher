"""Proof bundle construction and verification primitives (PROOFCORE W2).

The runner and deterministic replay fail closed until explicit decision-time
vault reads exist. Heavy dependencies stay behind ``__getattr__`` so importing
this package remains cheap and acyclic.
"""

from __future__ import annotations

from typing import Any

from quant_fund.proofcore.contracts import ProofError, ProofVerificationError

__all__ = [
    "DataAccessRecorder",
    "HmacSha256Signer",
    "InMemoryRecorder",
    "NullSigner",
    "ProofError",
    "ProofVerificationError",
    "Signer",
    "VerificationResult",
    "asof_utc_text",
    "build_bundle",
    "chain_head",
    "recompute_headline_metrics",
    "replay_bundle",
    "run_backtest_proven",
    "verify_bundle",
]

_LAZY = {
    "DataAccessRecorder": ("quant_fund.proof.recorder", "DataAccessRecorder"),
    "HmacSha256Signer": ("quant_fund.proof.sign", "HmacSha256Signer"),
    "InMemoryRecorder": ("quant_fund.proof.recorder", "InMemoryRecorder"),
    "NullSigner": ("quant_fund.proof.sign", "NullSigner"),
    "Signer": ("quant_fund.proof.sign", "Signer"),
    "VerificationResult": ("quant_fund.proof.verify", "VerificationResult"),
    "asof_utc_text": ("quant_fund.proof.recorder", "asof_utc_text"),
    "build_bundle": ("quant_fund.proof.bundle", "build_bundle"),
    "chain_head": ("quant_fund.proof.bundle", "chain_head"),
    "recompute_headline_metrics": ("quant_fund.proof.bundle", "recompute_headline_metrics"),
    "replay_bundle": ("quant_fund.proof.replay", "replay_bundle"),
    "run_backtest_proven": ("quant_fund.proof.runner", "run_backtest_proven"),
    "verify_bundle": ("quant_fund.proof.verify", "verify_bundle"),
}


def __getattr__(name: str) -> Any:
    """Lazy export pattern (microstructure/__init__.py template, §0.1 rule 6)."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)

"""Proof-carrying backtester (PROOFCORE W2, DESIGN.md §5).

The package exports the recorder and signer that exist in this revision.
Bundle minting, replay, chain heads, and verification stay unexported until
``proof.bundle``, ``proof.replay``, ``proof.runner``, and ``proof.verify``
land. Advertising them earlier makes ``from quant_fund.proof import *`` and
attribute access fail with ``ModuleNotFoundError``.

Heavy dependencies stay behind ``__getattr__`` so ``import quant_fund.proof``
stays cheap and acyclic (§1.3 layering contract).
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
]

_LAZY = {
    "DataAccessRecorder": ("quant_fund.proof.recorder", "DataAccessRecorder"),
    "HmacSha256Signer": ("quant_fund.proof.sign", "HmacSha256Signer"),
    "InMemoryRecorder": ("quant_fund.proof.recorder", "InMemoryRecorder"),
    "NullSigner": ("quant_fund.proof.sign", "NullSigner"),
    "Signer": ("quant_fund.proof.sign", "Signer"),
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

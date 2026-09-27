"""Jurafsky & Martin SLP3 Appendix A — discrete HMMs.

https://web.stanford.edu/~jurafsky/slp3/A.pdf

NumPy loads with the algorithms, not when the CLI imports ``hmm.cli``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DiscreteHMM",
    "EISNER",
    "baum_welch",
    "backward",
    "forward",
    "ice_cream",
    "likelihood",
    "mle_supervised",
    "viterbi",
]

_EXPORTS: dict[str, str] = {
    "DiscreteHMM": "quant_fund.hmm.discrete",
    "EISNER": "quant_fund.hmm.eisner",
    "baum_welch": "quant_fund.hmm.discrete",
    "backward": "quant_fund.hmm.discrete",
    "forward": "quant_fund.hmm.discrete",
    "ice_cream": "quant_fund.hmm.eisner",
    "likelihood": "quant_fund.hmm.discrete",
    "mle_supervised": "quant_fund.hmm.discrete",
    "viterbi": "quant_fund.hmm.discrete",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))

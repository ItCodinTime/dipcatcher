"""ADVERSARIAL §1a (helper indirection, POSITIVE): leaky helper called by strategy code.

`_align` trips LH001 in its own body (price-like receiver, literal negative
shift); `strategy` calls it, so LH014 flags the call site. Deliberately
leaky; do not import from strategy code.
"""

from __future__ import annotations


def _align(close):
    return close.shift(-1)


def strategy(close):
    return _align(close)

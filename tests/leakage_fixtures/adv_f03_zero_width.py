"""ADVERSARIAL §1a-F3 (POSITIVE, LH008): zero-width space inside the token."""

from __future__ import annotations


def claim() -> str:
    return "Our sha\u200brpe came in at 2.4"

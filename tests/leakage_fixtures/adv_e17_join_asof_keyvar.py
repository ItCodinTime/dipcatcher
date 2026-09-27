"""ADVERSARIAL §1a-E17 (POSITIVE, LH004): join_asof key behind a variable."""

from __future__ import annotations


def fuse(facts, events):
    key = "event_time"
    return facts.join_asof(events, on=key)

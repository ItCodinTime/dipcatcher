"""Vendor adapter test fixtures — recorded cassettes only, zero network."""

from __future__ import annotations

from pathlib import Path

import pytest

from .helpers import FIXTURES


@pytest.fixture()
def fixtures_dir() -> Path:
    return FIXTURES

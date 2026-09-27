"""Fuzz the YAML config loader: ``load_config`` + ``deep_merge``.

Writes each input to a temp ``.yaml`` file inside a fresh config root, then
loads it. Expected: ``ValueError`` (non-mapping, inheritance escape/cycle),
``yaml.YAMLError``, ``pydantic.ValidationError``, ``OSError``, and
``UnicodeDecodeError`` (undecodable file bytes). Findings: ``TypeError``,
``RecursionError`` (alias/anchor bombs through ``deep_merge``),
``AttributeError``, anything pydantic doesn't normalize.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import yaml
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fuzz_common import run

from quant_fund.config.loader import load_config

EXPECTED: tuple[type[BaseException], ...] = (
    ValueError,
    yaml.YAMLError,
    ValidationError,
    UnicodeDecodeError,
    OSError,
)


def test_one_input(data: bytes) -> None:
    with tempfile.TemporaryDirectory(prefix="fuzz_cfg_") as tmp:
        root = Path(tmp)
        target = root / "fuzz.yaml"
        target.write_bytes(data)
        load_config(target)


if __name__ == "__main__":
    raise SystemExit(run("config_yaml", test_one_input, expected=EXPECTED))

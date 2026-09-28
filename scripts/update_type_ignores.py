"""Regenerate ``quality/type_ignores.txt`` from the source tree."""

from __future__ import annotations

from check_type_ignores import main

raise SystemExit(main(["--write"]))

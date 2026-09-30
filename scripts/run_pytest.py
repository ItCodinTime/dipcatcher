"""Run pytest with the locked environment's native runtime libraries visible.

``uv run`` intentionally removes macOS dynamic-loader variables before it
starts Python.  LightGBM and XGBoost still need ``libomp.dylib``; the locked
``nn`` extra already installs a compatible copy with PyTorch.  Re-executing
the environment's Python after restoring that library directory makes the
documented Make gates work from a clean macOS sync without changing Linux.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import NoReturn


def pytest_environment() -> dict[str, str]:
    """Return an execution environment with bundled OpenMP visible on macOS."""
    env = dict(os.environ)
    if sys.platform != "darwin":
        return env

    spec = importlib.util.find_spec("torch")
    locations = None if spec is None else spec.submodule_search_locations
    if not locations:
        return env
    torch_lib = Path(next(iter(locations))) / "lib"
    if not (torch_lib / "libomp.dylib").is_file():
        return env

    existing = env.get("DYLD_LIBRARY_PATH")
    env["DYLD_LIBRARY_PATH"] = f"{torch_lib}{os.pathsep}{existing}" if existing else str(torch_lib)
    return env


def main() -> NoReturn:
    """Replace this process with pytest under the repaired environment."""
    argv = [sys.executable, "-m", "pytest", *sys.argv[1:]]
    os.execve(sys.executable, argv, pytest_environment())


if __name__ == "__main__":
    main()

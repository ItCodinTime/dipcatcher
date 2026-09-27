"""Import-time backend selection. Subprocesses keep the parent interpreter's choice."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from quant_fund import native
from quant_fund.native.reference import rolling_mean as rolling_mean_ref

pytestmark = pytest.mark.native


def test_loaded_backend_matches_extension() -> None:
    installed = importlib.util.find_spec("quant_core") is not None
    if installed:
        assert native.BACKEND == "rust"
    else:
        assert native.BACKEND == "python"
    series = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    got = native.rolling_mean(series, 2)
    exp = rolling_mean_ref(series, 2)
    assert got.shape == exp.shape
    assert np.allclose(got, exp, rtol=0.0, atol=0.0, equal_nan=True)


def _run(flag: str) -> subprocess.CompletedProcess[str]:
    code = f"""
import os
os.environ["QUANT_FUND_NATIVE"] = {flag!r}
import numpy as np
try:
    from quant_fund.native import BACKEND, rolling_mean
except Exception as exc:
    print(type(exc).__name__ + ": " + str(exc))
    raise SystemExit(2)
print(BACKEND)
x = np.asarray([1.0, 2.0, 4.0])
y = rolling_mean(x, 2)
assert y.shape == (3,)
assert np.isnan(y[0]) and y[1] == 1.5 and y[2] == 3.0
"""
    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )


def test_env_python_forces_reference() -> None:
    done = _run("python")
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[0] == "python"


def test_env_rust_requires_extension() -> None:
    done = _run("rust")
    installed = importlib.util.find_spec("quant_core") is not None
    if installed:
        assert done.returncode == 0, done.stderr
        assert done.stdout.splitlines()[0] == "rust"
    else:
        assert done.returncode == 2
        assert "ImportError" in done.stdout


def test_env_auto_follows_install() -> None:
    done = _run("auto")
    assert done.returncode == 0, done.stderr
    installed = importlib.util.find_spec("quant_core") is not None
    assert done.stdout.splitlines()[0] == ("rust" if installed else "python")


def test_unknown_flag_is_rejected() -> None:
    code = """
import os
os.environ["QUANT_FUND_NATIVE"] = "maybe"
try:
    import quant_fund.native
except ValueError as exc:
    print(exc)
    raise SystemExit(0)
raise SystemExit(1)
"""
    done = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "QUANT_FUND_NATIVE" in done.stdout


@pytest.mark.parametrize(
    "flag",
    ["python", "py", "numpy"],
)
def test_python_aliases(flag: str) -> None:
    done = _run(flag)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[0] == "python"

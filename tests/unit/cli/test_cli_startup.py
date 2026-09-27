"""CLI cold-start budget. Heavy optional libraries must stay off the import path.

Measured on a warm tree (bytecode already compiled), ``dipcatcher --help`` is
about 0.13s and ``fx1 --help`` about 0.12s. The budgets below leave room for
a slower CI runner and still fail the previous eager path (sklearn, pandas,
pyarrow, mlflow), which took about 1.8s warmed and about 8s on a cold
bytecode compile.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"

# Top-level module names. A dotted import such as ``sklearn.base`` still counts.
HEAVY_MODULES = (
    "torch",
    "sklearn",
    "pandas",
    "pyarrow",
    "jax",
    "lightgbm",
    "xgboost",
    "optuna",
    "mlflow",
    "matplotlib",
    "hmmlearn",
    "cvxpy",
    "arch",
    "statsmodels",
    "duckdb",
    "numpy",
    "scipy",
    "polars",
)

# Wall-clock ceilings for a fresh interpreter. See module docstring.
_DIPCATCHER_HELP_S = 1.5
_FX1_HELP_S = 1.0
_SUBPROCESS_TIMEOUT_S = 30


def _env() -> dict[str, str]:
    env = os.environ.copy()
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_SRC) + (os.pathsep + previous if previous else "")
    # The hint is only silenced inside configure_logging. Leaving it unset
    # makes an accidental mlflow import visible on stderr.
    env.pop("MLFLOW_DISABLE_AGENT_HINT", None)
    return env


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, *args],
        cwd=_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_S,
        check=False,
    )
    completed.elapsed = time.perf_counter() - started  # type: ignore[attr-defined]
    return completed


def test_package_and_cli_imports_skip_heavy_modules() -> None:
    script = (
        "import json, sys\n"
        "import quant_fund\n"
        "import fx1\n"
        "import fx1.cli\n"
        "import quant_fund.cli.main\n"
        f"heavy = {HEAVY_MODULES!r}\n"
        "hit = [name for name in heavy if name in sys.modules]\n"
        "print(json.dumps(hit))\n"
    )
    completed = _run(["-c", script])
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []
    assert "mlflow" not in completed.stderr.lower()


@pytest.mark.parametrize(
    ("args", "budget_s", "needle"),
    [
        (["-m", "quant_fund.cli.main", "--help"], _DIPCATCHER_HELP_S, "Usage"),
        (["-m", "fx1.cli", "--help"], _FX1_HELP_S, "Usage"),
        (["-m", "quant_fund.cli.main", "train"], _DIPCATCHER_HELP_S, "Specify a family"),
        (["-m", "fx1.cli", "doctor"], _FX1_HELP_S, "{"),
    ],
)
def test_cli_first_output_stays_under_budget(args: list[str], budget_s: float, needle: str) -> None:
    completed = _run(args)
    elapsed = float(completed.elapsed)  # type: ignore[attr-defined]
    assert completed.returncode == 0, completed.stderr
    assert needle in completed.stdout
    assert "mlflow" not in completed.stderr.lower()
    assert elapsed < budget_s, f"{' '.join(args)} took {elapsed:.3f}s (budget {budget_s:.1f}s)"


def test_lazy_patterns_compile_on_first_use() -> None:
    script = (
        "import fx1.data.sources.ingest as ingest\n"
        "import fx1.data.sources.router as router\n"
        "import quant_fund.pipeline.doctor as doctor\n"
        "assert router._COMPILED_MARKETS is None\n"
        "assert ingest._LIVE_TEXT is None\n"
        "assert doctor._SHA256 is None\n"
        "assert router.classify_market('纳斯达克报价') == 'us'\n"
        "assert router.classify_market('no marker here') == 'cn'\n"
        "assert ingest._text_claims_live('live trading profit')\n"
        "assert not ingest._text_claims_live('not a trading system')\n"
        "assert doctor._sha256_pattern().fullmatch('a' * 64)\n"
        "assert router._COMPILED_MARKETS is not None\n"
        "assert ingest._LIVE_TEXT is not None\n"
        "assert doctor._SHA256 is not None\n"
    )
    completed = _run(["-c", script])
    assert completed.returncode == 0, completed.stderr


def test_lazy_package_exports_match_and_resolve() -> None:
    script = (
        "import inspect\n"
        "import quant_fund.hmm as hmm\n"
        "import quant_fund.lightspeed as lightspeed\n"
        "import quant_fund.models as models\n"
        "import quant_fund.pipeline as pipeline\n"
        "import quant_fund.pipeline.train as train\n"
        "import quant_fund.pit as pit\n"
        "import quant_fund.quant_models as quant_models\n"
        "import quant_fund.registry as registry\n"
        "for package in (hmm, lightspeed, models, pipeline, pit, quant_models, registry):\n"
        "    assert set(package.__all__) == set(package._EXPORTS), package.__name__\n"
        "    try:\n"
        "        getattr(package, 'not_exported')\n"
        "    except AttributeError:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError(package.__name__)\n"
        "assert callable(hmm.viterbi)\n"
        "assert callable(quant_models.bs_price)\n"
        "assert inspect.ismodule(train)\n"
        "from quant_fund.pipeline import doctor\n"
        "assert callable(doctor)\n"
    )
    completed = _run(["-c", script])
    assert completed.returncode == 0, completed.stderr

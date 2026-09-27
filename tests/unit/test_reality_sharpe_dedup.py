"""A2 F4 anti-regression: Sharpe math lives ONLY in metrics/returns.py.

Static assertions that the former ad-hoc copies now import the canonical
implementation, plus behavior parity of the vectorized batch helper.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from quant_fund.metrics.returns import sharpe_ratio, sharpe_ratio_batch

SRC = Path(__file__).resolve().parents[2] / "src" / "quant_fund"


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "quant_fund.metrics.returns":
            names.update(alias.name for alias in node.names)
    return names


def test_sim_live_imports_canonical_sharpe() -> None:
    names = _imported_names(SRC / "paper" / "sim_live.py")
    assert "sharpe_ratio" in names


def test_scoreboard_imports_canonical_sharpe() -> None:
    names = _imported_names(SRC / "hedge_lab" / "scoreboard.py")
    assert "sharpe_ratio" in names
    assert "sharpe_ratio_batch" in names


def test_no_inline_sharpe_formula_outside_metrics_returns() -> None:
    """Grep-style guard: the sqrt(ppy) Sharpe pattern must not reappear."""
    suspects = [SRC / "paper" / "sim_live.py", SRC / "hedge_lab" / "scoreboard.py"]
    for path in suspects:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # a BinOp multiplying by a sqrt() call of a periods-per-year name
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "sqrt" and node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Name) and "per_year" in arg.id:
                        pytest.fail(f"inline sqrt(periods_per_year) Sharpe in {path}:{node.lineno}")


def test_sharpe_ratio_batch_matches_scalar_canonical() -> None:
    rng = np.random.default_rng(101)
    paths = rng.normal(0.001, 0.01, size=(50, 100))
    batch = sharpe_ratio_batch(paths, periods_per_year=252.0)
    scalar = np.array(
        [float(sharpe_ratio(paths[i], periods_per_year=252.0)["sharpe"]) for i in range(50)]
    )
    np.testing.assert_allclose(batch, scalar, rtol=1e-12, atol=0.0)


def test_sharpe_ratio_batch_nan_policy() -> None:
    paths = np.array(
        [
            [0.01, 0.02, 0.03],
            [0.01, 0.01, 0.01],  # zero vol -> NaN
            [0.01, np.nan, 0.03],  # non-finite -> NaN
        ]
    )
    out = sharpe_ratio_batch(paths, periods_per_year=252.0)
    assert np.isfinite(out[0])
    assert np.isnan(out[1])
    assert np.isnan(out[2])
    with pytest.raises(ValueError, match="2-D"):
        sharpe_ratio_batch(np.zeros(10))


def test_ic_information_ratio_reexport() -> None:
    from quant_fund.metrics import ic_information_ratio
    from quant_fund.metrics.scoring import icir

    assert ic_information_ratio is icir

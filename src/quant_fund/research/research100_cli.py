"""Command-line access to research references and a reproducible synthetic bench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

research100_app = typer.Typer(
    help="100 source-linked research components; no live-performance claims."
)


def _emit(value: Any, output: Path | None) -> None:
    def clean(x: Any) -> Any:
        import math

        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x

    text = json.dumps(clean(value), indent=2, allow_nan=False)
    if output is None:
        typer.echo(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
        typer.echo(str(output))


@research100_app.command("catalog")
def catalog(identifier: str | None = typer.Argument(None), output: Path | None = None) -> None:
    """List references, or inspect an ID such as R008 (including its limitations)."""
    from quant_fund.research.research100 import describe_method, load_catalog

    try:
        _emit(load_catalog() if identifier is None else describe_method(identifier), output)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@research100_app.command("verify")
def verify(output: Path | None = None) -> None:
    """Verify all 100 imports and hash modules. This is NOT scientific validation."""
    from quant_fund.research.research100 import audit_catalog

    result = audit_catalog()
    _emit(result, output)
    if result["available"] != result["total"]:
        raise typer.Exit(1)


@research100_app.command("benchmark")
def benchmark(synthetic: bool = False, seed: int = 100, output: Path | None = None) -> None:
    """Run a fixed comparison on labeled synthetic data; requires --synthetic."""
    if not synthetic:
        raise typer.BadParameter("Pass --synthetic; this bench is not market performance evidence.")
    from quant_fund.research.research100_benchmark import synthetic_benchmark

    _emit(synthetic_benchmark(seed=seed), output)

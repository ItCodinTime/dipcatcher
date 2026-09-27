"""Source-to-code registry for 100 selected references, not 100 alpha claims."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from collections.abc import Callable
from importlib.resources import files
from typing import Any, cast


def load_catalog() -> list[dict[str, Any]]:
    """Load the shipped catalog; identifiers are stable, not performance ranks."""
    rows: list[dict[str, Any]] = json.loads(
        files("quant_fund.research").joinpath("research100.json").read_text()
    )
    if len(rows) != 100 or len({r["id"] for r in rows}) != 100:
        raise ValueError("research100 catalog requires 100 unique entries")
    return rows


def describe_method(identifier: str) -> dict[str, Any]:
    for row in load_catalog():
        if row["id"] == identifier:
            return row
    raise ValueError(f"Unknown research ID: {identifier}")


def resolve_method(identifier: str) -> Callable[..., Any]:
    """Return an existing function/class with its native argument contract.

    Models, estimators and statistical tests have different interfaces. This
    registry deliberately does not coerce every paper into a trading strategy.
    Only packaged entries are resolved; arbitrary import paths are not accepted.
    """
    row = describe_method(identifier)
    module, symbol = row["entrypoint"].split(":")
    obj = getattr(importlib.import_module(module), symbol)
    if not callable(obj):
        raise ValueError(f"{identifier} does not resolve to a callable")
    return cast(Callable[..., Any], obj)


def audit_catalog() -> dict[str, Any]:
    """Import-check components and fingerprint their defining source modules.

    This checks availability, not equations, transitive dependencies, empirical
    reproduction or eligibility for live promotion. No network is needed.
    """
    outcomes = []
    for row in load_catalog():
        try:
            obj = resolve_method(row["id"])
            module = inspect.getmodule(obj)
            source = inspect.getsource(module) if module is not None else ""
            if not source:
                raise ValueError("component source is unavailable")
            outcomes.append(
                {
                    "id": row["id"],
                    "available": True,
                    "entrypoint": row["entrypoint"],
                    "module_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "signature": str(inspect.signature(obj)),
                }
            )
        except (ImportError, AttributeError, OSError, TypeError, ValueError) as exc:
            outcomes.append({"id": row["id"], "available": False, "error": str(exc)})
    return {
        "claim": "component_availability_only",
        "promotable": False,
        "catalog_sha256": hashlib.sha256(
            json.dumps(load_catalog(), sort_keys=True).encode()
        ).hexdigest(),
        "available": sum(r["available"] for r in outcomes),
        "total": len(outcomes),
        "components": outcomes,
    }

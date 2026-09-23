"""Doctor: config, dirs, imports, schema sanity. No secrets."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import polars as pl

from quant_fund import __firm__, __version__
from quant_fund.config import load_config
from quant_fund.config.models import AppConfig, RuntimeMode
from quant_fund.research.catalog import BENCHMARK_CATALOG_VERSION, BENCHMARK_FAMILY_ORDER
from quant_fund.research.verify import verify_research_artifact

_REQUIRED_ARTIFACTS = frozenset({"bars", "actions", "master", "silver", "universe"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def doctor(config_path: str | None = None) -> dict[str, object]:
    status: dict[str, object] = {
        "firm": __firm__,
        "product": "Dipcatcher",
        "version": __version__,
        "python_package": "quant_fund",
        "role": "Artificial Hedge proprietary research lab",
        "benchmark_catalog_version": str(BENCHMARK_CATALOG_VERSION),
        "families": ",".join(BENCHMARK_FAMILY_ORDER),
    }
    cfg: AppConfig
    if config_path:
        cfg = load_config(config_path)
    else:
        cfg = AppConfig()
    status["mode"] = cfg.runtime.mode.value
    status["live_allowed"] = str(cfg.runtime.allow_live)
    if cfg.runtime.mode is RuntimeMode.LIVE and not cfg.runtime.allow_live:
        status["live"] = "REJECTED"
    root = Path(cfg.data.root)
    for part in ("raw", "bronze", "silver", "gold", "metadata"):
        p = root / part
        p.mkdir(parents=True, exist_ok=True)
        status[f"dir_{part}"] = "ok" if p.is_dir() else "missing"
    manifest = root / "metadata" / "data_manifest.json"
    if not manifest.is_file():
        status["data_manifest"] = "missing"
    else:
        try:
            blob = json.loads(manifest.read_text())
            artifacts = blob.get("artifacts")
            valid = (
                blob.get("schema_version") == 1
                and isinstance(artifacts, dict)
                and _REQUIRED_ARTIFACTS.issubset(artifacts)
                and blob.get("source") == str(cfg.data.source)
            )
            root_resolved = root.resolve()
            if isinstance(artifacts, dict):
                for name in _REQUIRED_ARTIFACTS:
                    item = artifacts.get(name)
                    if not isinstance(item, dict):
                        valid = False
                        continue
                    raw_path = item.get("path")
                    digest = item.get("sha256")
                    rows = item.get("rows")
                    columns = item.get("columns")
                    if (
                        not isinstance(raw_path, str)
                        or not isinstance(digest, str)
                        or not isinstance(columns, list)
                        or any(not isinstance(column, str) for column in columns)
                    ):
                        valid = False
                        continue
                    path = Path(raw_path)
                    try:
                        inside_root = path.resolve().is_relative_to(root_resolved)
                    except OSError:
                        inside_root = False
                    valid = valid and inside_root and _SHA256.fullmatch(digest) is not None
                    valid = (
                        valid and isinstance(rows, int) and not isinstance(rows, bool) and rows >= 0
                    )
                    if path.is_file():
                        valid = valid and hashlib.sha256(path.read_bytes()).hexdigest() == digest
                        try:
                            schema = pl.scan_parquet(path).collect_schema()
                            valid = valid and sorted(schema.names()) == columns
                            actual_rows = pl.scan_parquet(path).select(pl.len()).collect().item()
                            valid = valid and actual_rows == rows
                        except (OSError, pl.exceptions.PolarsError):
                            valid = False
                    else:
                        valid = False
            status["data_manifest"] = "ok" if valid else "invalid"
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            status["data_manifest"] = "invalid"
    research_receipt = root / "metadata" / "research" / "latest.json"
    if not research_receipt.is_file():
        status["research_receipt"] = "missing"
    else:
        verification = verify_research_artifact(research_receipt)
        status["research_receipt"] = "ok" if verification["valid"] else "invalid"
    try:
        import arch  # noqa: F401
        import cvxpy  # noqa: F401
        import hmmlearn  # noqa: F401
        import sklearn  # noqa: F401

        status["core_imports"] = "ok"
    except ImportError as exc:
        status["core_imports"] = f"fail:{exc}"
    status["default_fill"] = cfg.execution.fill.value
    status["core_kline_engine"] = "robinhood_plus" if cfg.robinhood_plus.enabled else "disabled"
    status["robinhood_plus_backend"] = cfg.robinhood_plus.backend.value
    status["robinhood_plus_blend_weight"] = float(cfg.robinhood_plus.blend_weight)
    status["robinhood_plus_sizes_book"] = bool(
        cfg.robinhood_plus.enabled and cfg.robinhood_plus.blend_weight > 0.0
    )
    return status

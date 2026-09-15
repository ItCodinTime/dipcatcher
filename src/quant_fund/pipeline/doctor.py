"""Doctor: config, dirs, imports, schema sanity. No secrets."""

from __future__ import annotations

from pathlib import Path

from quant_fund import __firm__, __version__
from quant_fund.config import load_config
from quant_fund.config.models import AppConfig, RuntimeMode


def doctor(config_path: str | None = None) -> dict[str, str]:
    status: dict[str, str] = {
        "firm": __firm__,
        "product": "Dipcatcher",
        "version": __version__,
        "python_package": "quant_fund",
        "role": "scientific hedge research lab",
        "families": "ranking,alpha,volatility,distribution,regime,tail,drawdown,liquidity,rl,conformal",
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
    try:
        import arch  # noqa: F401
        import cvxpy  # noqa: F401
        import hmmlearn  # noqa: F401
        import sklearn  # noqa: F401

        status["core_imports"] = "ok"
    except ImportError as exc:
        status["core_imports"] = f"fail:{exc}"
    status["default_fill"] = cfg.execution.fill.value
    return status

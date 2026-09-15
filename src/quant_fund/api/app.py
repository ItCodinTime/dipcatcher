"""FastAPI service layer. Does not import model internals in route handlers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from quant_fund import __firm__, __version__
from quant_fund.config import load_config
from quant_fund.pipeline.doctor import doctor
from quant_fund.pipeline.forecast import forecast_asof, optimize_asof

app = FastAPI(title=f"{__firm__} Dipcatcher", version=__version__)


class OptimizeRequest(BaseModel):
    config_path: str = "configs/research.yaml"
    asof: datetime | None = None


class BacktestRequest(BaseModel):
    config_path: str = "configs/backtest.yaml"


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "firm": __firm__,
        "product": "Dipcatcher",
        "version": __version__,
        "mode_default": "research",
    }


@app.get("/models")
def models() -> dict[str, Any]:
    return {
        "ranking": [
            "composite",
            "ridge",
            "elasticnet",
            "xgboost",
            "lightgbm",
            "lambdarank",
            "xendcg",
        ],
        "distribution": ["empirical", "gaussian", "linear_qr", "xgboost", "lightgbm"],
        "volatility": ["rolling", "ewma", "garch", "har", "xgboost", "lightgbm"],
        "covariance": ["sample", "ewma", "ledoit_wolf", "factor", "dcc"],
        "regime": ["single_state", "threshold", "hmm"],
    }


@app.get("/forecast/{symbol}")
def forecast_symbol(symbol: str, config_path: str = "configs/research.yaml") -> dict[str, Any]:
    cfg = load_config(config_path)
    state = forecast_asof(cfg)
    hits = [f for f in state.forecasts if f.symbol == symbol or f.security_id == symbol]
    if not hits:
        raise HTTPException(404, f"no forecast for {symbol}")
    return hits[0].model_dump(mode="json")


@app.get("/forecast/{symbol}/distribution")
def forecast_dist(symbol: str, config_path: str = "configs/research.yaml") -> dict[str, Any]:
    body = forecast_symbol(symbol, config_path)
    return {
        "symbol": symbol,
        "quantiles": body.get("quantiles"),
        "interval_lo": body.get("interval_lo"),
        "interval_hi": body.get("interval_hi"),
        "interval_alpha": body.get("interval_alpha"),
        "interval_method": body.get("interval_method"),
    }


@app.get("/ranking")
def ranking(config_path: str = "configs/research.yaml") -> list[dict[str, Any]]:
    cfg = load_config(config_path)
    state = forecast_asof(cfg)
    rows = sorted(state.forecasts, key=lambda f: -f.rank_percentile.get("5d", 0.0))
    return [
        {"security_id": f.security_id, "symbol": f.symbol, "rank_percentile": f.rank_percentile}
        for f in rows
    ]


@app.get("/regime")
def regime() -> dict[str, float | str]:
    return {"risk_on": 0.0, "note": "fit HMM via `quant train regime` then forecast"}


@app.get("/risk/portfolio")
def risk_portfolio() -> dict[str, str]:
    return {"status": "see optimize diagnostics"}


@app.get("/portfolio/target")
def portfolio_target(config_path: str = "configs/research.yaml") -> list[dict[str, Any]]:
    cfg = load_config(config_path)
    w = optimize_asof(cfg)
    return w.to_dicts()


@app.get("/portfolio/exposures")
def exposures(config_path: str = "configs/research.yaml") -> dict[str, float]:
    cfg = load_config(config_path)
    w = optimize_asof(cfg)
    return {"gross": float(w["target_weight"].abs().sum()), "net": float(w["target_weight"].sum())}


@app.post("/portfolio/optimize")
def optimize(req: OptimizeRequest) -> list[dict[str, Any]]:
    cfg = load_config(req.config_path)
    return optimize_asof(cfg, req.asof).to_dicts()


@app.post("/backtest")
def backtest(req: BacktestRequest) -> dict[str, Any]:
    from quant_fund.backtest.engine import run_backtest
    from quant_fund.pipeline.dataset import ensure_silver

    cfg = load_config(req.config_path)
    bars = ensure_silver(cfg)
    w = optimize_asof(cfg)
    # broadcast last weights across dates for a smoke path
    import polars as pl

    dates = bars["event_time"].unique().sort()
    weights = pl.concat(
        [w.with_columns(pl.lit(d).alias("event_time")) for d in dates.to_list()[:30]]
    )
    result = run_backtest(bars, weights, cfg)
    return {**result.metrics, "source": result.source_note}


@app.get("/backtest/{backtest_id}")
def get_backtest(backtest_id: str) -> dict[str, str]:
    return {"id": backtest_id, "status": "see metadata reports"}


@app.get("/monitoring/drift")
def drift() -> dict[str, str]:
    return {"status": "compute via quant report"}


@app.get("/doctor")
def doctor_endpoint(config_path: str = "configs/research.yaml") -> dict[str, str]:
    return doctor(config_path)


@app.get("/research/latest")
def research_latest(config_path: str = "configs/research.yaml") -> dict[str, Any]:
    import json

    cfg = load_config(config_path)
    path = Path(cfg.data.root) / "metadata" / "research" / "latest.json"
    if not path.exists():
        raise HTTPException(404, "run `quant research` first")
    return json.loads(path.read_text())

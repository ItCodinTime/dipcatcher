"""FastAPI service layer. Does not import model internals in route handlers."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from quant_fund import __firm__, __version__
from quant_fund.config import load_config
from quant_fund.metrics.analytics import validate_analytics_export
from quant_fund.pipeline.doctor import doctor
from quant_fund.pipeline.forecast import build_causal_weight_panel, forecast_asof, optimize_asof

app = FastAPI(title=f"{__firm__} Dipcatcher", version=__version__)

# Repo root: src/quant_fund/api/app.py → parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIGS_DIR = (_REPO_ROOT / "configs").resolve()

# Only liveness is unauthenticated. When a key is configured, all other
# routes—including readiness, research receipts, and API documentation—must
# present it; when no key is configured, the loopback-only fallback applies.
_PUBLIC_PATHS = frozenset({"/health"})
_MAX_REQUEST_BYTES = 64 * 1024


class _RequestBodyTooLarge(Exception):
    """Internal control flow for rejecting a streamed request body."""


def resolve_allowed_config_path(config_path: str) -> Path:
    """Allow only configs under the repo ``configs/`` directory (no path traversal).

    Always resolve ``_CONFIGS_DIR`` before containment checks so macOS
    ``/var`` → ``/private/var`` symlink aliases (and test monkeypatches that
    pass an unresolved tmp path) cannot spuriously 400 the allowlist.
    """
    configs_dir = _CONFIGS_DIR.resolve()
    raw = Path(config_path)
    if raw.is_absolute():
        candidate = raw.resolve()
    else:
        # Strip leading "configs/" so both "research.yaml" and "configs/research.yaml" work
        parts = raw.parts
        if parts and parts[0] == "configs":
            candidate = (configs_dir.joinpath(*parts[1:])).resolve()
        else:
            candidate = (configs_dir / raw).resolve()
    try:
        candidate.relative_to(configs_dir)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="config_path must resolve under the repo configs/ directory",
        ) from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"config not found: {candidate.name}")
    return candidate


def _stamp_research_honesty(payload: dict[str, Any]) -> dict[str, Any]:
    """Force research-only honesty flags on metric-bearing API payloads.

    Always overwrites ``research_only`` / ``live_pnl_claim`` so a poisoned
    upstream metrics blob cannot claim live P&L through the HTTP surface.
    """
    out = dict(payload)
    out["research_only"] = True
    out["live_pnl_claim"] = False
    return out


def _weights_honesty_envelope(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap weight/ranking rows with explicit non-live honesty flags."""
    return _stamp_research_honesty(
        {
            "weights": rows,
            "claim": "research_only",
            "n": len(rows),
        }
    )


def _load_cfg(config_path: str):
    return load_config(resolve_allowed_config_path(config_path))


def _client_is_loopback(request: Request) -> bool:
    host = (request.client.host if request.client else "") or ""
    return host in {"127.0.0.1", "::1", "localhost", "testclient"}


async def _authenticate_request(
    request: Request, call_next: Any, path: str, expected: str | None
) -> Any:
    if path in _PUBLIC_PATHS:
        return await call_next(request)
    if expected:
        provided = request.headers.get("X-API-Key")
        if not provided or not hmac.compare_digest(provided, expected):
            return JSONResponse(status_code=401, content={"detail": "invalid or missing X-API-Key"})
        return await call_next(request)
    if not _client_is_loopback(request):
        return JSONResponse(
            status_code=403,
            content={
                "detail": (
                    "QUANT_API_KEY is unset; non-localhost clients are refused. "
                    "Set QUANT_API_KEY and send X-API-Key, or bind to 127.0.0.1 only."
                )
            },
        )
    return await call_next(request)


@app.middleware("http")
async def api_auth_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Secure default:

    - If ``QUANT_API_KEY`` is set: require matching ``X-API-Key`` on all non-public routes.
    - If unset: allow only loopback clients (refuse remote unauthenticated access).
    - Reject malformed or oversized request bodies before route work is performed.
    """
    path = request.url.path
    response: JSONResponse | Any
    expected = os.environ.get("QUANT_API_KEY")
    received_bytes = 0
    original_receive = request._receive  # type: ignore[attr-defined]

    async def limited_receive() -> Any:
        nonlocal received_bytes
        message = await original_receive()
        if message.get("type") == "http.request":
            received_bytes += len(message.get("body", b""))
            if received_bytes > _MAX_REQUEST_BYTES:
                raise _RequestBodyTooLarge
        return message

    request._receive = limited_receive  # type: ignore[attr-defined]
    try:
        declared_length = request.headers.get("content-length")
        if declared_length is not None:
            try:
                request_bytes = int(declared_length)
            except ValueError:
                response = JSONResponse(
                    status_code=400, content={"detail": "invalid Content-Length"}
                )
            else:
                if request_bytes < 0:
                    response = JSONResponse(
                        status_code=400, content={"detail": "invalid Content-Length"}
                    )
                elif request_bytes > _MAX_REQUEST_BYTES:
                    response = JSONResponse(
                        status_code=413,
                        content={"detail": "request body exceeds the 64 KiB limit"},
                    )
                else:
                    response = await _authenticate_request(request, call_next, path, expected)
        else:
            response = await _authenticate_request(request, call_next, path, expected)
    except _RequestBodyTooLarge:
        response = JSONResponse(
            status_code=413,
            content={"detail": "request body exceeds the 64 KiB limit"},
        )

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors 'none'"
    )
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Cache-Control"] = "no-store"
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_path: str = Field(default="configs/research.yaml", max_length=512)
    asof: datetime | None = None


class BacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_path: str = Field(default="configs/backtest.yaml", max_length=512)


def _backtest_artifact_dir(cfg: Any) -> Path:
    return Path(cfg.data.root) / "metadata" / "backtests"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _backtest_artifact_digest(artifact: dict[str, Any]) -> str:
    """Return a canonical self-excluding digest for a backtest receipt."""
    normalized = dict(artifact)
    normalized.pop("artifact_sha256", None)
    normalized.pop("artifact_path", None)
    return hashlib.sha256(
        json.dumps(
            normalized, sort_keys=True, separators=(",", ":"), allow_nan=True, default=str
        ).encode()
    ).hexdigest()


def _resolve_backtest_artifact_path(root: Path, value: Any) -> Path:
    """Resolve a receipt path while keeping it inside the backtest artifact root."""
    if not isinstance(value, str) or not value:
        raise HTTPException(422, "backtest artifact path must be a non-empty string")
    candidate = Path(value).resolve()
    artifact_root = root.resolve()
    try:
        candidate.relative_to(artifact_root)
    except ValueError as exc:
        raise HTTPException(
            422, "backtest artifact path escapes the configured artifact root"
        ) from exc
    return candidate


def _write_backtest_artifact(
    cfg: Any, req: BacktestRequest, bars: Any, result: Any
) -> dict[str, Any]:
    """Persist a reproducible, explicitly research-only backtest receipt."""
    dates = [str(value) for value in bars["event_time"].unique().sort().to_list()]
    scope = {
        "config_path": req.config_path,
        "dates": dates,
        "rows": bars.height,
        "source": result.source_note,
    }
    backtest_id = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()[:16]
    root = _backtest_artifact_dir(cfg)
    root.mkdir(parents=True, exist_ok=True)
    fills_path = root / f"{backtest_id}.fills.parquet"
    equity_path = root / f"{backtest_id}.equity.parquet"
    result.fills.write_parquet(fills_path)
    result.equity.write_parquet(equity_path)
    artifact = {
        "id": backtest_id,
        "status": "COMPLETE",
        "claim": "research_only",
        "research_only": True,
        "live_pnl_claim": False,
        "frictionless": bool(result.frictionless),
        "source": result.source_note,
        "scope": scope,
        "metrics": result.metrics,
        "fills_path": str(fills_path),
        "equity_path": str(equity_path),
        "fills_sha256": _file_sha256(fills_path),
        "equity_sha256": _file_sha256(equity_path),
    }
    artifact["artifact_sha256"] = _backtest_artifact_digest(artifact)
    artifact_path = root / f"{backtest_id}.json"
    artifact_path.write_text(json.dumps(artifact, sort_keys=True, indent=2, default=str) + "\n")
    artifact["artifact_path"] = str(artifact_path)
    return artifact


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "firm": __firm__,
        "product": "Dipcatcher",
        "version": __version__,
        "mode_default": "research",
    }


@app.get("/ready")
def readiness(config_path: str = "configs/research.yaml") -> JSONResponse:
    """Fail-closed readiness probe for the configured research runtime."""
    resolved = resolve_allowed_config_path(config_path)
    try:
        report = doctor(str(resolved))
    except (yaml.YAMLError, ValidationError, ValueError) as exc:
        # Genuine configuration problems fail closed as not_ready. Internal
        # bugs (any other exception type) are NOT masked here — they propagate
        # so an operator sees the real traceback instead of "invalid config".
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "checks": {"configuration": False},
                "report": {"configuration": f"invalid:{type(exc).__name__}"},
            },
        )
    checks = {
        "data_manifest": report.get("data_manifest") == "ok",
        "research_receipt": report.get("research_receipt") == "ok",
        "core_imports": report.get("core_imports") == "ok",
        "directories": all(
            report.get(f"dir_{part}") == "ok"
            for part in ("raw", "bronze", "silver", "gold", "metadata")
        ),
    }
    ready = all(checks.values()) and report.get("live") != "REJECTED"
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "not_ready", "checks": checks, "report": report},
    )


@app.get("/models")
def models() -> dict[str, Any]:
    import importlib.util

    from quant_fund.models.covariance import (
        IMPLEMENTED_COVARIANCE_SPECS,
        IMPLEMENTED_OPTIMIZER_COVARIANCE_SPECS,
        UNSPECIFIED_COVARIANCE_SPECS,
    )

    availability = {
        name: importlib.util.find_spec(name) is not None
        for name in ("xgboost", "lightgbm", "arch", "hmmlearn", "cvxpy", "torch")
    }
    return {
        "ranking": [
            "composite",
            "ridge",
            "elasticnet",
            "xgboost",
            "lightgbm",
            "lambdarank",
            "xendcg",
            "rff",
            "rff_ridgeless",
            "sdf_ridge",
            "sdf_en",
            "ipca",
            "ipca_alpha",
            "rp_pca",
            "fnw",
            "gx3pass",
            "ds_lasso",
            "fm",
            "pcr",
            "pls",
            "tprf",
            "gbrt",
            "pp",
        ],
        "distribution": ["empirical", "gaussian", "linear_qr", "xgboost", "lightgbm"],
        "volatility": ["rolling", "ewma", "garch", "har", "xgboost", "lightgbm"],
        "covariance": list(IMPLEMENTED_COVARIANCE_SPECS),
        "covariance_unspecified": list(UNSPECIFIED_COVARIANCE_SPECS),
        "optimizer_covariance": list(IMPLEMENTED_OPTIMIZER_COVARIANCE_SPECS),
        "regime": ["single_state", "threshold", "hmm"],
        "kline_foundation": ["robinhood_plus"],
        "robinhood_plus": {
            "display": "robinhood+",
            "derived_from": "Kronos (Shi et al., 2025, arXiv:2508.02739, MIT)",
            "backends": ["numpy", "torch"],
            "core_engine": True,
            "sizes_book": False,
            "blend_weight": 0.0,
            "affiliation": (
                "internal Dipcatcher engine name; not affiliated with Robinhood Markets, Inc."
            ),
        },
        "backend_availability": availability,
        "catalog_claim": "implemented_or_optional_backend; availability is environment-specific",
    }


@app.get("/forecast/{symbol}")
def forecast_symbol(symbol: str, config_path: str = "configs/research.yaml") -> dict[str, Any]:
    cfg = _load_cfg(config_path)
    state = forecast_asof(cfg)
    hits = [f for f in state.forecasts if f.symbol == symbol or f.security_id == symbol]
    if not hits:
        raise HTTPException(404, f"no forecast for {symbol}")
    return _stamp_research_honesty(hits[0].model_dump(mode="json"))


@app.get("/forecast/{symbol}/distribution")
def forecast_dist(symbol: str, config_path: str = "configs/research.yaml") -> dict[str, Any]:
    body = forecast_symbol(symbol, config_path)
    return _stamp_research_honesty(
        {
            "symbol": symbol,
            "quantiles": body.get("quantiles"),
            "interval_lo": body.get("interval_lo"),
            "interval_hi": body.get("interval_hi"),
            "interval_alpha": body.get("interval_alpha"),
            "interval_method": body.get("interval_method"),
        }
    )


@app.get("/ranking")
def ranking(config_path: str = "configs/research.yaml") -> dict[str, Any]:
    cfg = _load_cfg(config_path)
    state = forecast_asof(cfg)
    rows = sorted(state.forecasts, key=lambda f: -f.rank_percentile.get("5d", 0.0))
    return _stamp_research_honesty(
        {
            "rankings": [
                {
                    "security_id": f.security_id,
                    "symbol": f.symbol,
                    "rank_percentile": f.rank_percentile,
                }
                for f in rows
            ],
            "claim": "research_only",
            "n": len(rows),
        }
    )


@app.get("/regime")
def regime() -> dict[str, Any]:
    return _stamp_research_honesty(
        {
            "status": "UNMEASURED",
            "reason": "no point-in-time fitted regime model was supplied",
            "model": "GaussianHMM",
            "claim": "research_only",
        }
    )


@app.get("/risk/portfolio")
def risk_portfolio(config_path: str = "configs/research.yaml") -> dict[str, Any]:
    """Return as-of covariance risk for the persisted target portfolio.

    Trailing covariance follows the same named ``optimizer.covariance`` path
    ``optimize_asof`` uses. Default Ledoit–Wolf 2004 is scaled by the causal
    GARCH/RGARCH market overlay and stays Ledoit–Wolf when T<=N rather than
    switching to sample. Named ``dcc_gaussian``, ``dcc_student_t``,
    ``adcc``, ``ccc``, ``agdcc``, ``agdcc_full``, and ``ewma`` use one-step
    H_{t+1} and do not overlay that matrix. Named ``oas`` uses trailing
    Chen OAS plus that overlay and must not silently size as Ledoit–Wolf,
    sample, EWMA, or DCC. Named ``sample`` uses trailing unbiased sample
    covariance plus that overlay and must not silently size as Ledoit–Wolf,
    OAS, EWMA, or DCC. The sequential sample is the trailing contiguous
    complete-case window; an incomplete asof row fails closed. Those named
    paths are distinct. Named ``agdcc`` must not silently size as scalar
    ADCC or unrestricted AG-DCC. Named ``agdcc_full`` must not silently
    size as diagonal AG-DCC. Factor stays unwired. A present Realized
    GARCH artifact still fail-closes on missing OHLC rather than reporting
    unscaled sample risk.
    """
    import numpy as np
    import polars as pl

    from quant_fund.data.point_in_time import filter_trailing_returns_asof
    from quant_fund.models.covariance import (
        IMPLEMENTED_OPTIMIZER_NAMED_SPECS,
        IMPLEMENTED_OPTIMIZER_ONE_STEP_SPECS,
        require_implemented_optimizer_covariance,
    )
    from quant_fund.models.realized_garch import REALIZED_GARCH_MEASURE
    from quant_fund.pipeline.dataset import panel
    from quant_fund.pipeline.forecast import estimate_optimizer_covariance_asof
    from quant_fund.portfolio.optimizer import component_risk
    from quant_fund.schemas.errors import PointInTimeError
    from quant_fund.schemas.forecast import MARKET_RISK_OVERLAY_REALIZED_GARCH

    cfg = _load_cfg(config_path)
    weights_path = Path(cfg.data.root) / "gold" / "target_weights.parquet"
    if not weights_path.is_file():
        return _stamp_research_honesty(
            {
                "status": "UNMEASURED",
                "reason": "no persisted target-weight panel is available",
                "source": "none",
                "claim": "research_only",
            }
        )
    try:
        weights = pl.read_parquet(weights_path)
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise HTTPException(422, "target-weight artifact could not be read") from exc
    required_columns = {"event_time", "security_id", "target_weight"}
    missing_columns = sorted(required_columns.difference(weights.columns))
    if weights.is_empty() or missing_columns:
        detail = "target-weight artifact has no as-of rows"
        if missing_columns:
            detail = "target-weight artifact is missing columns: " + ", ".join(missing_columns)
        raise HTTPException(422, detail)
    if weights["event_time"].dtype != pl.Datetime:
        raise HTTPException(422, "target-weight artifact event_time must be datetime")
    if weights["event_time"].null_count() or weights["security_id"].null_count():
        raise HTTPException(422, "target-weight artifact contains null identity fields")
    try:
        weights = weights.with_columns(
            pl.col("target_weight").cast(pl.Float64, strict=False).alias("target_weight")
        )
    except pl.exceptions.PolarsError as exc:
        raise HTTPException(422, "target-weight artifact has invalid weights") from exc
    weight_values = weights["target_weight"].to_numpy()
    if weights["target_weight"].null_count() or not np.isfinite(weight_values).all():
        raise HTTPException(422, "target-weight artifact has non-finite weights")
    if weights.select(["event_time", "security_id"]).is_duplicated().any():
        raise HTTPException(422, "target-weight artifact contains duplicate identity rows")
    asof = weights["event_time"].max()
    if not isinstance(asof, datetime):
        raise HTTPException(422, "target-weight artifact event_time must be datetime")
    # Build the weight vector from the latest as-of slice only: a multi-date
    # panel would duplicate security ids and misalign against the covariance.
    weights = weights.filter(pl.col("event_time") == asof)
    ids = [str(value) for value in weights["security_id"].to_list()]
    w = np.asarray(weights["target_weight"].to_list(), dtype=float)
    overlay_frame = panel(cfg)
    frame = overlay_frame.filter(pl.col("event_time") <= asof)
    try:
        frame = filter_trailing_returns_asof(frame, asof)
    except PointInTimeError as exc:
        raise HTTPException(422, str(exc)) from exc
    if "ret_1" not in frame.columns:
        return _stamp_research_honesty(
            {
                "status": "UNMEASURED",
                "reason": "point-in-time return history is unavailable",
                "asof": str(asof),
                "source": str(weights_path),
                "claim": "research_only",
            }
        )
    wide = frame.select(["event_time", "security_id", "ret_1"]).pivot(
        on="security_id", index="event_time", values="ret_1"
    )
    cols = [sid for sid in ids if sid in wide.columns]
    estimator = require_implemented_optimizer_covariance(cfg.optimizer.covariance)
    if estimator not in IMPLEMENTED_OPTIMIZER_NAMED_SPECS and len(cols) < 2:
        return _stamp_research_honesty(
            {
                "status": "UNMEASURED",
                "reason": "fewer than two securities have point-in-time return history",
                "asof": str(asof),
                "source": str(weights_path),
                "claim": "research_only",
            }
        )
    mat = wide.select(cols).to_numpy().astype(float) if cols else np.empty((0, 0))
    mat = mat[np.isfinite(mat).all(axis=1)] if mat.size else mat
    if estimator not in IMPLEMENTED_OPTIMIZER_NAMED_SPECS and mat.shape[0] < 2:
        return _stamp_research_honesty(
            {
                "status": "UNMEASURED",
                "reason": "insufficient finite return observations for covariance",
                "asof": str(asof),
                "source": str(weights_path),
                "claim": "research_only",
            }
        )
    try:
        estimate = estimate_optimizer_covariance_asof(cfg, overlay_frame, asof, cols, frame)
    except (PointInTimeError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    if estimate.unmeasured_reason is not None:
        reasons = {
            "no_ret_1": "point-in-time return history is unavailable",
            "fewer_than_two_securities": (
                "fewer than two securities have point-in-time return history"
            ),
            "insufficient_finite_rows": "insufficient finite return observations for covariance",
            "estimation_failed": "insufficient finite return observations for covariance",
        }
        return _stamp_research_honesty(
            {
                "status": "UNMEASURED",
                "reason": reasons.get(
                    estimate.unmeasured_reason,
                    "insufficient finite return observations for covariance",
                ),
                "asof": str(asof),
                "source": str(weights_path),
                "claim": "research_only",
            }
        )
    sigma = estimate.sigma
    col_idx = [ids.index(sid) for sid in estimate.security_ids]
    mcr, cr, predicted_vol = component_risk(w[col_idx], sigma)
    payload: dict[str, Any] = {
        "status": "MEASURED",
        "asof": str(asof),
        "observations": int(estimate.n_obs),
        "securities": len(estimate.security_ids),
        "predicted_volatility": predicted_vol,
        "gross": float(np.abs(w).sum()),
        "net": float(w.sum()),
        "covariance_estimator": estimate.estimator,
        "covariance_object": estimate.covariance_object,
        "covariance_spec": estimate.spec,
        "market_risk_overlay": estimate.market_overlay,
        "components": [
            {
                "security_id": sid,
                "weight": float(w[i]),
                "marginal_risk": float(mcr[j]),
                "risk_contribution": float(cr[j]),
            }
            for j, (sid, i) in enumerate(zip(estimate.security_ids, col_idx, strict=True))
        ],
        "source": str(weights_path),
        "claim": "research_only",
    }
    if estimate.estimator in IMPLEMENTED_OPTIMIZER_ONE_STEP_SPECS:
        payload["covariance_horizon"] = 1
    overlay = estimate.overlay
    overlay_kind = estimate.market_overlay
    if overlay is not None and overlay_kind is not None:
        payload["garch_market_sigma"] = float(overlay.sigma)
        payload["garch_market_variance"] = float(overlay.variance)
        payload["garch_series_scope"] = overlay.series_scope
        if overlay_kind == MARKET_RISK_OVERLAY_REALIZED_GARCH:
            payload["realized_measure"] = REALIZED_GARCH_MEASURE
            payload["intraday_realized_variance"] = False
    return _stamp_research_honesty(payload)


@app.get("/portfolio/target")
def portfolio_target(config_path: str = "configs/research.yaml") -> dict[str, Any]:
    cfg = _load_cfg(config_path)
    w = optimize_asof(cfg)
    return _weights_honesty_envelope(w.to_dicts())


@app.get("/portfolio/exposures")
def exposures(config_path: str = "configs/research.yaml") -> dict[str, Any]:
    cfg = _load_cfg(config_path)
    w = optimize_asof(cfg)
    return _stamp_research_honesty(
        {
            "gross": float(w["target_weight"].abs().sum()),
            "net": float(w["target_weight"].sum()),
            "claim": "research_only",
        }
    )


@app.post("/portfolio/optimize")
def optimize(req: OptimizeRequest) -> dict[str, Any]:
    cfg = _load_cfg(req.config_path)
    return _weights_honesty_envelope(optimize_asof(cfg, req.asof).to_dicts())


@app.post("/backtest")
def backtest(req: BacktestRequest) -> dict[str, Any]:
    from quant_fund.backtest.engine import run_backtest
    from quant_fund.pipeline.dataset import ensure_silver

    cfg = _load_cfg(req.config_path)
    bars = ensure_silver(cfg)
    all_dates = bars["event_time"].unique().sort().to_list()
    # Bound HTTP work: 30 causal decisions plus one next-open execution date.
    decision_dates = all_dates[:30]
    simulation_dates = all_dates[:31]
    simulation_bars = bars.filter(bars["event_time"].is_in(simulation_dates))
    # Causal weights per decision date (no end-of-sample broadcast)
    weights = build_causal_weight_panel(cfg, decision_dates)
    result = run_backtest(simulation_bars, weights, cfg)
    artifact = _write_backtest_artifact(cfg, req, simulation_bars, result)
    # Artifact overwrites metrics keys; still force honesty so a poisoned
    # metrics blob cannot claim live P&L on the HTTP response.
    return _stamp_research_honesty({**result.metrics, **artifact, "claim": "research_only"})


@app.get("/backtest/{backtest_id}")
def get_backtest(backtest_id: str, config_path: str = "configs/backtest.yaml") -> dict[str, Any]:
    cfg = _load_cfg(config_path)
    if len(backtest_id) != 16 or any(char not in "0123456789abcdef" for char in backtest_id):
        raise HTTPException(400, "invalid backtest id")
    path = _backtest_artifact_dir(cfg) / f"{backtest_id}.json"
    if not path.is_file():
        raise HTTPException(404, f"backtest not found: {backtest_id}")
    artifact = json.loads(path.read_text())
    required = (
        "id",
        "metrics",
        "scope",
        "fills_path",
        "equity_path",
        "fills_sha256",
        "equity_sha256",
        "claim",
    )
    missing = [key for key in required if key not in artifact]
    if missing:
        raise HTTPException(
            422, detail={"message": "invalid backtest artifact", "missing": missing}
        )
    if artifact.get("id") != backtest_id or artifact.get("claim") != "research_only":
        raise HTTPException(422, "backtest artifact failed identity or claim validation")
    if artifact.get("live_pnl_claim") is not False:
        raise HTTPException(422, "backtest artifact has an invalid live-P&L claim")
    artifact_digest = artifact.get("artifact_sha256")
    if artifact_digest is not None:
        if (
            not isinstance(artifact_digest, str)
            or len(artifact_digest) != 64
            or any(character not in "0123456789abcdef" for character in artifact_digest)
        ):
            raise HTTPException(422, "backtest artifact has an invalid digest")
        if artifact_digest != _backtest_artifact_digest(artifact):
            raise HTTPException(422, "backtest artifact digest mismatch")
    metrics = artifact.get("metrics")
    if isinstance(metrics, dict) and "analytics_export" in metrics:
        analytics_report = validate_analytics_export(metrics["analytics_export"])
        if not analytics_report.get("ok"):
            raise HTTPException(
                422,
                detail={
                    "message": "backtest analytics export failed validation",
                    "errors": analytics_report.get("errors", []),
                },
            )
    resolved_paths: dict[str, Path] = {}
    artifact_root = _backtest_artifact_dir(cfg)
    for key in ("fills_path", "equity_path"):
        path = _resolve_backtest_artifact_path(artifact_root, artifact[key])
        resolved_paths[key] = path
        if not path.is_file():
            raise HTTPException(
                422, detail={"message": "backtest data artifact missing", "path": key}
            )
    for path_key, hash_key in (("fills_path", "fills_sha256"), ("equity_path", "equity_sha256")):
        if (
            not isinstance(artifact[hash_key], str)
            or _file_sha256(resolved_paths[path_key]) != artifact[hash_key]
        ):
            raise HTTPException(
                422, detail={"message": "backtest data artifact hash mismatch", "path": path_key}
            )
    return _stamp_research_honesty(artifact)


@app.get("/monitoring/drift")
def drift(config_path: str = "configs/research.yaml") -> dict[str, Any]:
    """Return the latest drift-monitoring state without fabricating a reading.

    Feature windows are not available to this read-only endpoint yet. The
    explicit ``UNMEASURED`` state prevents dashboards from treating a
    placeholder as a healthy production monitor.
    """
    import json

    cfg = _load_cfg(config_path)
    path = Path(cfg.data.root) / "metadata" / "research" / "latest.json"
    result: dict[str, Any] = {
        "status": "UNMEASURED",
        "reason": "feature reference/current windows were not supplied",
        "source": "research_artifact" if path.exists() else "none",
        "claim": "research_only",
    }
    if path.exists():
        from quant_fund.research.verify import verify_research_artifact

        try:
            candidate_bytes = path.read_bytes()
            candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
        except OSError:
            result.update(
                {
                    "reason": "research artifact could not be read",
                    "source": "none",
                }
            )
            return _stamp_research_honesty(result)
        verification = verify_research_artifact(path)
        if not verification["valid"]:
            result.update(
                {
                    "reason": "research artifact failed integrity verification",
                    "source": "none",
                    "integrity_errors": verification["errors"],
                }
            )
            return _stamp_research_honesty(result)
        try:
            current_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            result.update(
                {
                    "reason": "research artifact could not be read",
                    "source": "none",
                }
            )
            return _stamp_research_honesty(result)
        if current_sha256 != candidate_sha256:
            result.update(
                {
                    "reason": "research artifact changed after integrity verification",
                    "source": "none",
                }
            )
            return _stamp_research_honesty(result)
        try:
            notebook = json.loads(candidate_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError):
            result.update(
                {
                    "reason": "research artifact could not be read",
                    "source": "none",
                }
            )
            return _stamp_research_honesty(result)
        provenance = notebook.get("provenance", {})
        result.update(
            {
                "latest_run_id": provenance.get("run_id"),
                "latest_git_revision": provenance.get("git_revision"),
                "data_source": notebook.get("data_source"),
                "point_in_time": provenance.get("point_in_time"),
            }
        )
    return _stamp_research_honesty(result)


@app.get("/doctor")
def doctor_endpoint(config_path: str = "configs/research.yaml") -> dict[str, str]:
    return doctor(str(resolve_allowed_config_path(config_path)))


@app.get("/research/latest")
def research_latest(config_path: str = "configs/research.yaml") -> dict[str, Any]:
    import json

    cfg = _load_cfg(config_path)
    path = Path(cfg.data.root) / "metadata" / "research" / "latest.json"
    if not path.exists():
        raise HTTPException(404, "run `quant research` first")
    from quant_fund.research.verify import verify_research_artifact

    try:
        # Keep the exact candidate bytes that will be returned. The verifier
        # reads the path independently, so compare the path again afterward to
        # detect a mutable receipt being replaced during this request.
        candidate_bytes = path.read_bytes()
        candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    except OSError as exc:
        raise HTTPException(
            422,
            detail={
                "message": "latest research receipt could not be read before verification",
                "error": type(exc).__name__,
            },
        ) from exc
    try:
        verification = verify_research_artifact(path)
    except Exception as exc:  # noqa: BLE001 — API integrity boundary must fail closed
        raise HTTPException(
            422,
            detail={
                "message": "latest research receipt verification failed",
                "error": type(exc).__name__,
            },
        ) from exc
    if not verification["valid"]:
        raise HTTPException(
            422,
            detail={
                "message": "latest research receipt failed integrity verification",
                "errors": verification["errors"],
            },
        )
    try:
        current_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise HTTPException(
            422,
            detail={
                "message": "latest research receipt could not be read after verification",
                "error": type(exc).__name__,
            },
        ) from exc
    if current_sha256 != candidate_sha256:
        raise HTTPException(
            422,
            detail={"message": "latest research receipt changed after verification"},
        )
    try:
        payload = json.loads(candidate_bytes)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            422,
            detail={
                "message": "latest research receipt could not be read after verification",
                "error": type(exc).__name__,
            },
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(422, "latest research receipt is not an object")
    return _stamp_research_honesty(payload)

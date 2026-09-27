"""Assemble a research stress report. No live-trading output."""

from __future__ import annotations

import html
from dataclasses import asdict
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.regime_switch import fit_markov_switching_mean
from quant_fund.research.catalog.constants import FORBIDDEN_RESEARCH_METRIC_KEYS
from quant_fund.stress.bootstrap import path_moments, stationary_bootstrap_paths
from quant_fund.stress.garch_copula import (
    fit_variance_targeted_garch11,
    residual_dependence,
    simulate_garch_t_copula,
)
from quant_fund.stress.jumps import calibrate_merton_to_moments, merton_scenario_log_returns
from quant_fund.stress.regimes import (
    simulate_gaussian_hmm,
    spec_from_univariate_fit,
    unconditional_moments,
)
from quant_fund.stress.replay import replay_portfolio
from quant_fund.stress.reverse import (
    default_radius,
    reverse_stress,
    sample_mean_cov,
    worst_linear_scenario,
)
from quant_fund.stress.risk import (
    backtest_var_es,
    bootstrap_var_es_interval,
    causal_historical_forecasts,
    point_estimates,
)
from quant_fund.stress.strategy import ResearchStrategy, portfolio_returns

Array = NDArray[np.float64]

DISCLAIMER = (
    "Research simulation only. This report is not a live-trading result, does not "
    "submit orders, and is not evidence of a market edge. Synthetic scenarios are "
    "labelled synthetic. Published scalars are factor shocks, not redistributed vendor tapes."
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating | np.integer):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def _assert_no_forbidden_keys(payload: Any, path: str = "") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            token = str(key).lower()
            if token in FORBIDDEN_RESEARCH_METRIC_KEYS:
                raise ValueError(f"forbidden research metric key at {path}.{key}")
            _assert_no_forbidden_keys(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for i, value in enumerate(payload):
            _assert_no_forbidden_keys(value, f"{path}[{i}]")


def _scenario_loss(loss: Any) -> dict[str, Any]:
    return {
        "kind": loss.kind,
        "loss": loss.loss,
        "partial": loss.partial,
        "note": loss.note,
        "contributions": [asdict(item) for item in loss.contributions],
    }


def _historical_equity_cloud(strategy: ResearchStrategy) -> dict[str, Any] | None:
    """1-d Mahalanobis stress on published historical equity scalars."""
    from quant_fund.stress.catalog import CRISIS_CATALOG

    position = strategy.weight_map().get("us_equity")
    if position is None or position.mapping != "return":
        return None
    shocks = []
    for crisis in CRISIS_CATALOG:
        if not crisis.historical:
            continue
        for shock in crisis.shocks:
            if (
                shock.factor == "us_equity"
                and shock.role == "primary"
                and shock.unit == "simple_return"
            ):
                shocks.append(shock.value)
    if len(shocks) < 3:
        return {
            "status": "unavailable",
            "reason": "fewer than three historical primary us_equity shocks",
        }
    sample = np.asarray(shocks, dtype=float).reshape(-1, 1)
    mu, cov = sample_mean_cov(sample, ridge=0.0)
    radius = default_radius(1, contour=0.05)
    weights = np.asarray([position.weight], dtype=float)
    analytic = worst_linear_scenario(weights, mu, cov, radius)
    searched = reverse_stress(
        lambda x: float(-weights @ np.asarray(x, dtype=float)), mu, cov, radius, seed=0
    )
    return {
        "status": "ok",
        "covariance_source": "published_historical_us_equity_scalars",
        "plausibility_model": "gaussian_on_that_cloud",
        "n_shocks": len(shocks),
        "shocks": shocks,
        "radius": radius,
        "analytic": analytic.as_dict(),
        "searched": searched.as_dict(),
        "note": (
            "The cloud has one point per historical episode with a primary equity scalar. "
            "It is not a covariance of daily returns, and the plausibility score is not a market probability."
        ),
    }


def _synthetic_block(panel: Array, portfolio: Array, n_scenarios: int, seed: int) -> dict[str, Any]:
    """Calibrated synthetic scenarios. Failures are recorded, not filled in."""
    out: dict[str, Any] = {"label": "SYNTHETIC", "market_evidence": False}
    horizon = min(60, panel.shape[0])
    paths = stationary_bootstrap_paths(
        panel, n_paths=min(n_scenarios, 80), horizon=horizon, seed=seed
    )
    moments = path_moments(paths)
    sample_mean = panel.mean(axis=0)
    out["stationary_bootstrap"] = {
        "n_paths": int(paths.shape[0]),
        "horizon": horizon,
        "sample_mean": sample_mean.tolist(),
        "bootstrap_mean": moments["mean"].tolist(),
    }
    try:
        fits = [fit_variance_targeted_garch11(panel[:, j]) for j in range(panel.shape[1])]
        corr, nu = residual_dependence(panel, fits)
        # Numerical PSD guard: reject a non-PD residual correlation rather than jitter it silently.
        if float(np.linalg.eigvalsh(corr)[0]) <= 1e-8:
            raise ValueError("residual correlation is not positive definite")
        simulated = simulate_garch_t_copula(
            np.asarray([fit["omega"] for fit in fits]),
            np.asarray([fit["alpha"] for fit in fits]),
            np.asarray([fit["beta"] for fit in fits]),
            corr,
            nu,
            n_scenarios,
            np.random.default_rng(seed),
        )
        out["garch_t_copula"] = {
            "status": "ok",
            "nu": nu,
            "unconditional_variance": [fit["unconditional_variance"] for fit in fits],
            "simulated_variance": simulated.var(axis=0, ddof=1).tolist(),
            "alpha": [fit["alpha"] for fit in fits],
            "beta": [fit["beta"] for fit in fits],
        }
    except (ValueError, np.linalg.LinAlgError) as exc:
        out["garch_t_copula"] = {"status": "unavailable", "reason": str(exc)}
    try:
        fitted = fit_markov_switching_mean(portfolio, n_states=2, max_iter=40, seed=seed)
        spec = spec_from_univariate_fit(fitted)
        moments_hmm = unconditional_moments(spec)
        drawn, _states = simulate_gaussian_hmm(spec, n_scenarios, np.random.default_rng(seed + 1))
        out["hmm"] = {
            "status": "ok",
            "scope": "univariate_portfolio_return",
            "analytic_mean": float(moments_hmm["mean"][0]),
            "simulated_mean": float(drawn[:, 0].mean()),
            "analytic_variance": float(moments_hmm["cov"][0, 0]),
            "simulated_variance": float(drawn[:, 0].var(ddof=1)),
        }
    except (ValueError, np.linalg.LinAlgError) as exc:
        out["hmm"] = {"status": "unavailable", "reason": str(exc)}
    try:
        calibrated = calibrate_merton_to_moments(portfolio)
        drawn_j = merton_scenario_log_returns(
            tenor=calibrated["tenor"],
            rate=calibrated["rate"],
            sigma=calibrated["sigma"],
            lam=calibrated["lam"],
            mu_j=calibrated["mu_j"],
            s_j=calibrated["s_j"],
            n_scenarios=n_scenarios,
            rng=np.random.default_rng(seed + 2),
        )
        out["jump_diffusion"] = {
            "status": "ok",
            "scope": "univariate_portfolio_log_return",
            "lam": calibrated["lam"],
            "analytic_mean": calibrated["analytic_mean"],
            "analytic_var": calibrated["analytic_var"],
            "simulated_mean": float(drawn_j.mean()),
            "simulated_variance": float(drawn_j.var(ddof=1)),
            "note": "Jump mean and jump volatility are fixed inputs. Skewness is not matched.",
        }
    except (ValueError, np.linalg.LinAlgError) as exc:
        out["jump_diffusion"] = {"status": "unavailable", "reason": str(exc)}
    return out


def _risk_block(portfolio: Array, level: float, n_boot: int, seed: int) -> dict[str, Any]:
    losses = -portfolio
    block: dict[str, Any] = {"loss_convention": "positive loss is a negative portfolio return"}
    try:
        block["estimates"] = point_estimates(losses, level)
    except ValueError as exc:
        block["estimates"] = {"status": "unavailable", "reason": str(exc)}
        return block
    try:
        intervals = bootstrap_var_es_interval(losses, level, n_boot=max(n_boot, 50), seed=seed)
        block["historical_intervals"] = {name: item.as_dict() for name, item in intervals.items()}
    except ValueError as exc:
        block["historical_intervals"] = {"status": "unavailable", "reason": str(exc)}
    if losses.size >= 80:
        realized, var, es = causal_historical_forecasts(losses, level, min_history=50)
        block["backtest"] = backtest_var_es(
            realized, var, es, level, n_boot=min(n_boot, 200), seed=seed
        )
    else:
        block["backtest"] = {
            "status": "unavailable",
            "reason": "need at least 80 portfolio returns for an expanding-window backtest",
        }
    return block


def _panel_reverse(
    strategy: ResearchStrategy, names: tuple[str, ...], panel: Array
) -> dict[str, Any]:
    if strategy.asset_weights:
        weight_map = dict(strategy.asset_weights)
        weights = np.asarray([weight_map[name] for name in names], dtype=float)
    else:
        weights = np.full(len(names), 1.0 / len(names))
    mu, cov = sample_mean_cov(panel, ridge=0.0)
    ridge = 0.0
    if float(np.linalg.eigvalsh(cov)[0]) <= 1e-10:
        ridge = 1e-6
        _mu, cov = sample_mean_cov(panel, ridge=ridge)
        mu = _mu
    radius = default_radius(len(names), contour=0.05)
    result = reverse_stress(
        lambda x: float(-weights @ np.asarray(x, dtype=float)),
        mu,
        cov,
        radius,
        seed=1,
    )
    return {
        "status": "ok",
        "covariance_source": "sample_covariance_of_asset_returns",
        "ridge": ridge,
        "plausibility_model": "gaussian_on_sample_covariance",
        "asset_names": list(names),
        "weights": weights.tolist(),
        "result": result.as_dict(),
        "note": "Plausibility is the chi-square tail of the Mahalanobis distance under this covariance.",
    }


def build_stress_report(
    strategy: ResearchStrategy,
    *,
    names: tuple[str, ...] | None = None,
    panel: Array | None = None,
    level: float = 0.95,
    seed: int = 0,
    n_scenarios: int = 256,
    n_boot: int = 200,
) -> dict[str, Any]:
    """Full research report for one strategy."""
    if not strategy.research_only:
        raise ValueError("stress reports require research_only strategies")
    if isinstance(n_scenarios, bool) or n_scenarios < 32:
        raise ValueError("n_scenarios must be >= 32")
    replays = replay_portfolio(strategy, historical_only=False)
    crises = []
    for replay in replays:
        crises.append(
            {
                "crisis_id": replay.crisis_id,
                "name": replay.name,
                "historical": replay.historical,
                "summary": replay.summary,
                "limitations": list(replay.limitations),
                "factor_loss": _scenario_loss(replay.factor_loss),
                "path_loss": _scenario_loss(replay.path_loss),
                "path_stats": replay.path_stats,
                "shocks": [asdict(shock) for shock in replay.shocks],
            }
        )
    report: dict[str, Any] = {
        "research_only": True,
        "live_trading": False,
        "disclaimer": DISCLAIMER,
        "strategy": strategy.name,
        "var_level": level,
        "crises": crises,
        "reverse_historical_equity": _historical_equity_cloud(strategy),
    }
    if panel is not None and names is not None:
        if panel.ndim != 2 or panel.shape[1] != len(names):
            raise ValueError("panel columns must match names")
        portfolio = portfolio_returns(names, panel, strategy.asset_weights)
        report["synthetic"] = _synthetic_block(panel, portfolio, n_scenarios, seed)
        report["risk"] = _risk_block(portfolio, level, n_boot, seed)
        try:
            report["reverse_return_panel"] = _panel_reverse(strategy, names, panel)
        except (ValueError, np.linalg.LinAlgError) as exc:
            report["reverse_return_panel"] = {"status": "unavailable", "reason": str(exc)}
    else:
        report["synthetic"] = {
            "label": "SYNTHETIC",
            "market_evidence": False,
            "status": "not_run",
            "reason": "no decimal return panel was supplied",
        }
        report["risk"] = {
            "status": "not_run",
            "reason": "VaR backtests need a return panel",
        }
    _assert_no_forbidden_keys(report)
    rendered = _jsonable(report)
    if not isinstance(rendered, dict):
        raise TypeError("stress report must be a mapping")
    return cast(dict[str, Any], rendered)


def render_markdown(report: dict[str, Any]) -> str:
    """Markdown rendering of a stress report."""
    lines = [
        f"# Stress report: {report['strategy']}",
        "",
        report["disclaimer"],
        "",
        f"- research_only: {report['research_only']}",
        f"- live_trading: {report['live_trading']}",
        f"- var_level: {report['var_level']}",
        "",
        "## Crises",
        "",
        "| id | historical | factor-shock loss | path loss | note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for crisis in report["crises"]:
        factor = crisis["factor_loss"]["loss"]
        path = crisis["path_loss"]["loss"]
        factor_txt = "n/a" if factor is None else f"{factor:.4f}"
        path_txt = "n/a" if path is None else f"{path:.4f}"
        note = crisis["limitations"][0] if crisis["limitations"] else ""
        lines.append(
            f"| {crisis['crisis_id']} | {crisis['historical']} | {factor_txt} | {path_txt} | {note} |"
        )
    lines.extend(["", "## Limitations", ""])
    for crisis in report["crises"]:
        for item in crisis["limitations"]:
            lines.append(f"- `{crisis['crisis_id']}`: {item}")
    lines.extend(["", "## Reverse stress (historical equity scalars)", ""])
    lines.append("```")
    lines.append(str(report.get("reverse_historical_equity")))
    lines.append("```")
    lines.extend(["", "## Synthetic scenarios", ""])
    lines.append("```")
    lines.append(str(report.get("synthetic")))
    lines.append("```")
    lines.extend(["", "## Risk", ""])
    lines.append("```")
    lines.append(str(report.get("risk")))
    lines.append("```")
    if "reverse_return_panel" in report:
        lines.extend(
            [
                "",
                "## Reverse stress (return panel)",
                "",
                "```",
                str(report["reverse_return_panel"]),
                "```",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def render_html(report: dict[str, Any]) -> str:
    """Self-contained HTML rendering. Strategy text is escaped."""
    rows = []
    for crisis in report["crises"]:
        factor = crisis["factor_loss"]["loss"]
        path = crisis["path_loss"]["loss"]
        factor_txt = "n/a" if factor is None else f"{factor:.4f}"
        path_txt = "n/a" if path is None else f"{path:.4f}"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(crisis['crisis_id']))}</td>"
            f"<td>{html.escape(str(crisis['historical']))}</td>"
            f"<td>{factor_txt}</td>"
            f"<td>{path_txt}</td>"
            f"<td>{html.escape(crisis['summary'])}</td>"
            "</tr>"
        )
    body = "\n".join(rows)
    limitations = "\n".join(
        f"<li><code>{html.escape(str(crisis['crisis_id']))}</code>: {html.escape(item)}</li>"
        for crisis in report["crises"]
        for item in crisis["limitations"]
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Stress report: {html.escape(str(report["strategy"]))}</title>
<style>
body {{ font-family: Georgia, serif; margin: 2rem; color: #1c1c1c; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #ccc; padding: 0.4rem; vertical-align: top; }}
.banner {{ background: #f4f1ea; padding: 1rem; }}
</style>
</head>
<body>
<h1>Stress report: {html.escape(str(report["strategy"]))}</h1>
<p class="banner">{html.escape(str(report["disclaimer"]))}</p>
<p>research_only: {html.escape(str(report["research_only"]))} |
live_trading: {html.escape(str(report["live_trading"]))}</p>
<h2>Crises</h2>
<table>
<thead><tr><th>id</th><th>historical</th><th>factor-shock loss</th><th>path loss</th><th>summary</th></tr></thead>
<tbody>
{body}
</tbody>
</table>
<h2>Limitations</h2>
<ul>
{limitations}
</ul>
</body>
</html>
"""
